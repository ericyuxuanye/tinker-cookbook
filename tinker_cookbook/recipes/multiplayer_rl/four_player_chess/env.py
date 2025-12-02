"""Four-player chess environment for tinker RL."""

import asyncio
import re
from dataclasses import dataclass
from typing import ClassVar, Sequence

import chz
import jax
import jax.numpy as jnp
import four_player_chess as fpc
import tinker
from tinker import types
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

STOP_CONDITION = ["]\n"]
ILLEGAL_MOVE_REWARD = -5.0


class FourPlayerCoordinator:
    """Coordinates a single four player chess game. See text_arena for similar pattern."""

    def __init__(self, jax_env: fpc.FourPlayerChessEnv, initial_state: fpc.EnvState):
        self.jax_env = jax_env
        self.state = initial_state  # Current JAX state
        self.condition = asyncio.Condition()
        self.illegal_player_id: int | None = None
        self.game_done = False

    @property
    def current_player_id(self) -> int:
        """Get the current player ID from the environment state."""
        return int(self.state.current_player)

    async def wait_across_env(self, player_id: int) -> None:
        """Wait until this player's turn comes up"""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player_id == player_id or self.game_done
            )

    async def make_move(self, player_id: int, move_action: int) -> tuple[float, dict]:
        """Make a move and notify waiting players. Returns (reward, info)."""
        async with self.condition:
            if self.game_done:
                return 0.0, {}

            if self.current_player_id != player_id:
                raise ValueError(
                    f"Not player {player_id}'s turn (current: {self.current_player_id})"
                )

            # Execute move in JAX environment
            key = jax.random.PRNGKey(0)  # Deterministic for now
            next_state, obs, reward, done, info = self.jax_env.step(
                key, self.state, move_action
            )

            # Check if move was valid
            move_valid = info.get("move_valid", False)
            if not move_valid:
                # Mark this player as having made an illegal move
                self.illegal_player_id = player_id
                self.game_done = True
            else:
                self.state = next_state
                self.game_done = bool(done)

            # Notify all waiting players about the state change
            self.condition.notify_all()

            return float(reward), dict(info)


def board_to_text(state: fpc.EnvState, player_id: int) -> str:
    """Convert JAX board state to text description for the LLM."""
    lines = []

    # Header with current player info
    player_names = ["Red", "Blue", "Yellow", "Green"]
    player_colors = ["🔴", "🔵", "🟡", "🟢"]

    lines.append(f"You are {player_colors[player_id]} {player_names[player_id]}.")
    lines.append(f"Move {int(state.move_count)}")
    lines.append("")

    # Show scores and active status
    lines.append("Scores:")
    for i in range(4):
        status = "Active" if state.player_active[i] else "Eliminated"
        score = int(state.player_scores[i])
        lines.append(f"  {player_colors[i]} {player_names[i]}: {score} points ({status})")
    lines.append("")

    # Render the board
    board = state.board
    lines.append("Board (14x14 cross-shaped):")
    lines.append("   " + "".join([f"{i:3d}" for i in range(14)]))

    piece_chars = {
        (0, 1): "rP", (0, 2): "rN", (0, 3): "rB", (0, 4): "rR", (0, 5): "rQ", (0, 6): "rK",
        (1, 1): "bP", (1, 2): "bN", (1, 3): "bB", (1, 4): "bR", (1, 5): "bQ", (1, 6): "bK",
        (2, 1): "yP", (2, 2): "yN", (2, 3): "yB", (2, 4): "yR", (2, 5): "yQ", (2, 6): "yK",
        (3, 1): "gP", (3, 2): "gN", (3, 3): "gB", (3, 4): "gR", (3, 5): "gQ", (3, 6): "gK",
    }

    for row in range(14):
        row_str = f"{row:2d}:"
        for col in range(14):
            is_valid = board[row, col, 3] > 0  # CHANNEL_VALID_SQUARE
            if not is_valid:
                row_str += "   "
            else:
                piece_type = int(board[row, col, 0])  # CHANNEL_PIECE_TYPE
                owner = int(board[row, col, 1])  # CHANNEL_OWNER
                if piece_type == 0:  # EMPTY
                    row_str += "  ."
                else:
                    piece = piece_chars.get((owner, piece_type), "??")
                    row_str += f" {piece}"
        lines.append(row_str)

    lines.append("")
    lines.append("Your turn! Make a move in the format: [(source_row, source_col) -> (dest_row, dest_col)]")
    lines.append("For pawn promotion, add =Q (Queen), =R (Rook), =B (Bishop), or =N (Knight)")
    lines.append("Example: [(12, 6) -> (10, 6)]")

    return "\n".join(lines)


def parse_move_action(action_text: str) -> tuple[int, int, int, int, int]:
    """
    Parse move from LLM output.
    Returns: (source_row, source_col, dest_row, dest_col, promotion_type)
    """
    # Try to match pattern like [(12, 6) -> (10, 6)] or [(7, 3) -> (3, 3)=Q]
    pattern = r"\[\((\d+),\s*(\d+)\)\s*->\s*\((\d+),\s*(\d+)\)\s*(?:=([QRBN]))?\]"
    match = re.search(pattern, action_text)

    if not match:
        # Return invalid move (out of bounds)
        return -1, -1, -1, -1, 0

    source_row = int(match.group(1))
    source_col = int(match.group(2))
    dest_row = int(match.group(3))
    dest_col = int(match.group(4))

    # Parse promotion type
    promotion_map = {"Q": 0, "R": 1, "B": 2, "N": 3}
    promotion = match.group(5)
    promotion_type = promotion_map.get(promotion, 0) if promotion else 0

    return source_row, source_col, dest_row, dest_col, promotion_type


def encode_action(source_row: int, source_col: int, dest_row: int, dest_col: int,
                  promotion_type: int, valid_mask: jnp.ndarray) -> int:
    """
    Encode move coordinates into action index.
    This reverses the decode_action function from the JAX environment.
    """
    # Create flattened valid square indices
    valid_indices = jnp.argwhere(valid_mask, size=160, fill_value=-1)

    # Find source square index
    source_idx = -1
    for i, (r, c) in enumerate(valid_indices):
        if int(r) == source_row and int(c) == source_col:
            source_idx = i
            break

    # Find dest square index
    dest_idx = -1
    for i, (r, c) in enumerate(valid_indices):
        if int(r) == dest_row and int(c) == dest_col:
            dest_idx = i
            break

    if source_idx == -1 or dest_idx == -1:
        return 0  # Invalid action

    # Encode: source * (160 * 4) + dest * 4 + promotion
    action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
    return int(action)


@dataclass
class FourPlayerChessEnv(Env):
    """Four player chess environment for a single player."""

    player_id: int  # 0, 1, 2, or 3
    coordinator: FourPlayerCoordinator
    self_play: bool
    renderer: Renderer
    opponent_policies: list[TinkerMessageCompleter | None]  # 3 opponents (or None for self-play)
    valid_mask: jnp.ndarray

    def __post_init__(self):
        if self.self_play:
            assert all(p is None for p in self.opponent_policies), (
                "For self_play, all opponent_policies must be None"
            )
        else:
            assert sum(p is not None for p in self.opponent_policies) == 3, (
                "For non-self-play, exactly 3 opponent policies must be provided"
            )

    @property
    def stop_condition(self) -> StopCondition:
        return STOP_CONDITION

    async def wait_for_turn(self) -> None:
        """Wait until it's this player's turn"""
        if not self.coordinator.game_done:
            if self.self_play:
                await self.coordinator.wait_across_env(self.player_id)
            else:
                # Opponents take their turns
                while (not self.coordinator.game_done and
                       self.coordinator.current_player_id != self.player_id):
                    await self.opponent_step()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.player_id != 0:
            await self.wait_for_turn()
        return self.get_observation(), self.stop_condition

    async def opponent_step(self) -> None:
        """When not self_play, an opponent policy takes a step"""
        current_player = self.coordinator.current_player_id

        # Get the opponent policy for this player
        opponent_idx = (current_player - self.player_id - 1) % 4
        opponent_policy = self.opponent_policies[opponent_idx]

        if opponent_policy is None:
            raise ValueError(f"No opponent policy for player {current_player}")

        # Get observation for opponent
        observation_text = board_to_text(self.coordinator.state, current_player)
        opponent_convo: list[Message] = [{"role": "user", "content": observation_text}]

        # Get opponent's action
        opponent_response = await opponent_policy(opponent_convo)
        action_text: str = opponent_response["content"]

        # Parse and execute move
        source_row, source_col, dest_row, dest_col, promotion = parse_move_action(action_text)
        action_idx = encode_action(source_row, source_col, dest_row, dest_col,
                                   promotion, self.valid_mask)

        await self.coordinator.make_move(current_player, action_idx)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self.coordinator.game_done:
            return self.get_done_step()

        assert self.coordinator.current_player_id == self.player_id, "Not the current player's turn"

        # Parse action from LLM output
        action_message: Message = self.renderer.parse_response(action)[0]
        action_text = action_message["content"]

        source_row, source_col, dest_row, dest_col, promotion = parse_move_action(action_text)
        action_idx = encode_action(source_row, source_col, dest_row, dest_col,
                                   promotion, self.valid_mask)

        # Execute move
        reward, info = await self.coordinator.make_move(self.player_id, action_idx)

        # Wait for next turn
        await self.wait_for_turn()

        return StepResult(
            reward=self.compute_reward(reward, info),
            episode_done=self.coordinator.game_done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def get_done_step(self) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )

    def compute_reward(self, base_reward: float, info: dict) -> float:
        """Compute reward for this player."""
        # Check if this player made an illegal move
        if self.coordinator.illegal_player_id == self.player_id:
            return ILLEGAL_MOVE_REWARD

        # Use the reward from the JAX environment (capture points, etc.)
        return base_reward

    def get_observation(self) -> types.ModelInput:
        """Get text observation for current player."""
        if self.coordinator.game_done:
            return types.ModelInput.empty()

        # Only provide observation if it's this player's turn
        if self.coordinator.current_player_id != self.player_id:
            return types.ModelInput.empty()

        observation_text = board_to_text(self.coordinator.state, self.player_id)
        return self.renderer.build_generation_prompt([{"role": "user", "content": observation_text}])


@dataclass
class FourPlayerChessEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of four player chess environments sharing the same game."""

    renderer: Renderer
    num_envs: int
    self_play: bool
    num_players: ClassVar[int] = 4
    opponent_policies: list[TinkerMessageCompleter | None] | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments sharing the same chess game."""
        if self.num_envs % 4 != 0:
            raise ValueError("num_envs must be divisible by 4 (one env per player)")

        def _construct_coordinator() -> FourPlayerCoordinator:
            """Create a new game coordinator with initial state."""
            jax_env = fpc.FourPlayerChessEnv()
            key = jax.random.PRNGKey(0)  # TODO: randomize
            initial_state, _ = jax_env.reset(key)
            return FourPlayerCoordinator(jax_env=jax_env, initial_state=initial_state)

        envs = []
        for _ in range(self.num_envs // 4):
            if self.self_play:
                # All players share the same coordinator for self-play
                coordinator = _construct_coordinator()
                coordinators = [coordinator for _ in range(self.num_players)]
                opponent_lists = [[None, None, None] for _ in range(self.num_players)]
            else:
                # Each env gets its own coordinator and opponent policies
                coordinators = [_construct_coordinator() for _ in range(self.num_players)]
                opponent_lists = [self.opponent_policies for _ in range(self.num_players)]

            # Create valid mask once (same for all envs)
            valid_mask = fpc.board.create_valid_square_mask()

            envs += [
                FourPlayerChessEnv(
                    player_id=i,
                    coordinator=coordinators[i],
                    renderer=self.renderer,
                    self_play=self.self_play,
                    opponent_policies=opponent_lists[i],
                    valid_mask=valid_mask,
                )
                for i in range(4)
            ]

        return envs


class FourPlayerChessDataset(RLDataset):
    """Dataset for four-player chess environments."""

    def __init__(self, batch_size: int, builder: FourPlayerChessEnvGroupBuilder, num_datapoints: int):
        self.batch_size = batch_size
        self.builder = builder
        self.num_datapoints = num_datapoints
        assert self.num_datapoints % self.builder.num_players == 0, (
            "num_datapoints must be divisible by num_players (4)"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self.builder
            for i in range(self.batch_size // self.builder.num_players)
            if (index * self.batch_size + self.builder.num_players * i) < self.num_datapoints
        ]

    def __len__(self) -> int:
        return self.num_datapoints // self.batch_size


@chz.chz
class FourPlayerChessDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    num_train_datapoints: int
    num_test_datapoints: int
    base_url: str | None = None
    model_name: str
    renderer_name: str

    def _construct_opponent_policies(self, renderer: Renderer) -> list[TinkerMessageCompleter]:
        """Create fixed opponent policies for testing (3 opponents)."""
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.model_name)

        # Create 3 opponent policies (one for each of the other players)
        return [
            TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=64,
                stop_condition=STOP_CONDITION,
            )
            for _ in range(3)
        ]

    async def __call__(self) -> tuple[FourPlayerChessDataset, FourPlayerChessDataset | None]:
        """Build the dataset for training and testing."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        # Training dataset performs self-play (all 4 players learning)
        train_builder = FourPlayerChessEnvGroupBuilder(
            renderer=renderer,
            num_envs=4,
            self_play=True,
        )
        train_dataset = FourPlayerChessDataset(
            batch_size=self.batch_size,
            builder=train_builder,
            num_datapoints=self.num_train_datapoints,
        )

        # Testing dataset plays against fixed policies
        test_builder = FourPlayerChessEnvGroupBuilder(
            renderer=renderer,
            num_envs=4,
            self_play=False,
            opponent_policies=self._construct_opponent_policies(renderer),
        )
        test_dataset = FourPlayerChessDataset(
            batch_size=self.num_test_datapoints,
            builder=test_builder,
            num_datapoints=self.num_test_datapoints,
        )

        return train_dataset, test_dataset
