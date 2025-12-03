"""Refactored single-step four-player chess environment for tinker RL.

This version enumerates all legal moves and has the model pick one number.
The model sees a numbered list of all possible moves like:
  1. P (Pawn) from (12, 3) to (11, 3)
  2. P (Pawn) from (12, 3) to (10, 3)
  3. P (Pawn) from (12, 4) to (11, 4)
  ...
And outputs just a number to select that move.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
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
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

STOP_CONDITION = ["\n\n"]  # Two newlines to avoid stopping too early
ILLEGAL_MOVE_REWARD = -5.0
MAX_INVALID_RETRIES = 10


class FourPlayerCoordinator:
    """Coordinates a single four player chess game."""

    def __init__(self, jax_env: fpc.FourPlayerChessEnv, initial_state: fpc.EnvState):
        self.jax_env = jax_env
        self.state = initial_state
        self.condition = asyncio.Condition()
        self.game_done = False
        self.move_history: list[dict] = []

    @property
    def current_player_id(self) -> int:
        return int(self.state.current_player)

    async def wait_across_env(self, player_id: int) -> None:
        """Wait until this player's turn comes up"""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player_id == player_id or self.game_done
            )

    async def make_move(self, player_id: int, move_action: int, move_text: str = "") -> tuple[float, dict]:
        """Make a move and notify waiting players. Returns (reward, info)."""
        async with self.condition:
            if self.game_done:
                return 0.0, {}

            if self.current_player_id != player_id:
                raise ValueError(
                    f"Not player {player_id}'s turn (current: {self.current_player_id})"
                )

            # Execute move in JAX environment
            key = jax.random.PRNGKey(0)
            next_state, obs, reward, done, info = self.jax_env.step(
                key, self.state, move_action
            )

            move_valid = info.get("move_valid", False)

            if move_valid:
                self.move_history.append({
                    "player_id": player_id,
                    "move_text": move_text,
                    "move_number": int(self.state.move_count),
                    "reward": float(reward),
                    "scores": [int(s) for s in next_state.player_scores],
                    "active_players": [bool(a) for a in next_state.player_active],
                })

                self.state = next_state
                self.game_done = bool(done)
                self.condition.notify_all()
            else:
                logger.debug(f"Invalid move by player {player_id}: {move_text}")

            return float(reward), dict(info)


def get_player_pieces(state: fpc.EnvState, player_id: int, valid_mask: jnp.ndarray) -> list[tuple[int, int, str]]:
    """Get all pieces owned by a player.

    Returns: List of (row, col, piece_name) tuples.
    Note: We don't check for legal moves here for performance. If a piece has no legal moves,
    the user will find out when they select it.
    """
    # print(f"[DEBUG] get_player_pieces scanning board for player {player_id}", flush=True)
    board = state.board
    pieces = []

    piece_names = {
        1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"
    }
    piece_abbrev = {
        1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"
    }

    # Scan board for player's pieces
    for row in range(14):
        for col in range(14):
            if not valid_mask[row, col]:
                continue

            piece_type = int(board[row, col, 0])  # CHANNEL_PIECE_TYPE
            owner = int(board[row, col, 1])  # CHANNEL_OWNER

            if piece_type != 0 and owner == player_id:  # Not empty and owned by player
                name = piece_names.get(piece_type, "Unknown")
                abbrev = piece_abbrev.get(piece_type, "?")
                pieces.append((row, col, f"{abbrev} ({name})"))

    # print(f"[DEBUG] get_player_pieces done, found {len(pieces)} pieces", flush=True)
    return pieces


def has_legal_moves_for_piece(
    state: fpc.EnvState,
    row: int,
    col: int,
    player_id: int,
    valid_mask: jnp.ndarray
) -> bool:
    """Check if a piece has any legal moves."""
    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.rules import is_move_legal

    # Get pseudo-legal moves
    pseudo_moves = get_pseudo_legal_moves(
        state.board, row, col, player_id, valid_mask, state.en_passant_square
    )

    # Check if any pseudo-legal move is actually legal
    for dest_row in range(14):
        for dest_col in range(14):
            if pseudo_moves[dest_row, dest_col]:
                # Check full legality (doesn't leave king in check)
                is_legal = is_move_legal(
                    state.board,
                    row, col,
                    dest_row, dest_col,
                    player_id,
                    state.king_positions[player_id],
                    state.player_active,
                    valid_mask,
                    state.en_passant_square
                )
                if is_legal:
                    return True

    return False


def get_legal_moves_for_piece(
    state: fpc.EnvState,
    row: int,
    col: int,
    player_id: int,
    valid_mask: jnp.ndarray
) -> list[tuple[int, int]]:
    """Get all legal destination squares for a piece.

    Returns: List of (row, col) tuples for valid destinations.
    """
    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.rules import is_move_legal

    # Get pseudo-legal moves
    pseudo_moves = get_pseudo_legal_moves(
        state.board, row, col, player_id, valid_mask, state.en_passant_square
    )

    # Filter to only fully legal moves
    legal_moves = []
    for dest_row in range(14):
        for dest_col in range(14):
            if pseudo_moves[dest_row, dest_col]:
                is_legal = is_move_legal(
                    state.board,
                    row, col,
                    dest_row, dest_col,
                    player_id,
                    state.king_positions[player_id],
                    state.player_active,
                    valid_mask,
                    state.en_passant_square
                )
                if is_legal:
                    legal_moves.append((dest_row, dest_col))

    return legal_moves


def get_all_legal_moves(
    state: fpc.EnvState,
    player_id: int,
    valid_mask: jnp.ndarray
) -> list[tuple[int, int, int, int, str]]:
    """Get all legal moves for a player.

    Returns: List of (source_row, source_col, dest_row, dest_col, piece_name) tuples.
    """
    pieces = get_player_pieces(state, player_id, valid_mask)
    all_moves = []

    for row, col, piece_name in pieces:
        destinations = get_legal_moves_for_piece(state, row, col, player_id, valid_mask)
        for dest_row, dest_col in destinations:
            all_moves.append((row, col, dest_row, dest_col, piece_name))

    return all_moves


def encode_action(
    source_row: int, source_col: int, dest_row: int, dest_col: int,
    promotion_type: int, valid_mask: jnp.ndarray
) -> int:
    """Encode move coordinates into action index."""
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


def format_move_selection_prompt(state: fpc.EnvState, player_id: int, valid_mask: jnp.ndarray) -> str:
    """Create a prompt showing all legal moves."""
    player_names = ["Red", "Blue", "Yellow", "Green"]
    player_colors = ["🔴", "🔵", "🟡", "🟢"]

    lines = []
    lines.append(f"You are {player_colors[player_id]} {player_names[player_id]}.")
    lines.append(f"Move {int(state.move_count)}")
    lines.append("")

    # Show scores
    lines.append("Scores:")
    for i in range(4):
        status = "Active" if state.player_active[i] else "Eliminated"
        score = int(state.player_scores[i])
        lines.append(f"  {player_colors[i]} {player_names[i]}: {score} points ({status})")
    lines.append("")

    # Get all legal moves
    all_moves = get_all_legal_moves(state, player_id, valid_mask)

    if not all_moves:
        lines.append("You have no legal moves!")
        return "\n".join(lines)

    lines.append("YOUR LEGAL MOVES:")
    for i, (src_row, src_col, dest_row, dest_col, piece_name) in enumerate(all_moves, 1):
        lines.append(f"  {i}. {piece_name} from ({src_row}, {src_col}) to ({dest_row}, {dest_col})")
    lines.append("")

    lines.append("Choose your move.")
    lines.append("Output ONLY a number from the list above.")
    lines.append("Example: 1")

    return "\n".join(lines)


def parse_move_selection(response_text: str, num_moves: int) -> int | None:
    """Parse move selection from model output.

    Returns: move index (0-based) or None if invalid.
    """
    # Try to extract a number
    match = re.search(r'\b(\d+)\b', response_text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_moves:
            return num - 1  # Convert to 0-based
    return None


@dataclass
class FourPlayerChessEnv(Env):
    """Four player chess environment with single-step move selection."""

    player_id: int
    coordinator: FourPlayerCoordinator
    self_play: bool
    renderer: Renderer
    opponent_policies: list[TinkerMessageCompleter | None]
    valid_mask: jnp.ndarray

    # State for retries
    retry_count: int = 0
    has_had_initial_turn: bool = False  # Track if we've had our first turn

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
                # print(f"[DEBUG] Player {self.player_id}: Calling coordinator.wait_across_env...", flush=True)
                await self.coordinator.wait_across_env(self.player_id)
                # print(f"[DEBUG] Player {self.player_id}: Returned from coordinator.wait_across_env", flush=True)
            else:
                while (not self.coordinator.game_done and
                       self.coordinator.current_player_id != self.player_id):
                    await self.opponent_step()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # print(f"[DEBUG] initial_observation called for player {self.player_id}", flush=True)

        # All players return immediately - we'll wait in step() instead
        # This prevents deadlock in asyncio.gather
        player_names = ["Red", "Blue", "Yellow", "Green"]
        player_colors = ["🔴", "🔵", "🟡", "🟢"]

        # Only show header if it's our turn
        if self.coordinator.current_player_id == self.player_id:
            print(f"\n{'=' * 60}")
            print(f"{player_colors[self.player_id]} Player {self.player_id} ({player_names[self.player_id]}) - Move {int(self.coordinator.state.move_count)}")
            print(f"{'=' * 60}")

        obs = await self._get_observation_async()
        # print(f"[DEBUG] Returning from initial_observation for player {self.player_id}", flush=True)

        return obs, self.stop_condition

    async def opponent_step(self) -> None:
        """When not self_play, an opponent policy takes a step"""
        # Simplified opponent behavior - would need full 2-step implementation
        # For now, opponents can use the old single-step approach
        pass

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment (single-step move selection)."""
        if self.coordinator.game_done:
            return await self.get_done_step()

        player_names = ["Red", "Blue", "Yellow", "Green"]

        # Wait for our turn (this is crucial for multiplayer self-play)
        await self.wait_for_turn()

        if self.coordinator.game_done:
            return await self.get_done_step()

        # Parse action from LLM output
        try:
            action_message: Message = self.renderer.parse_response(action)[0]
            action_text = action_message["content"]
        except Exception as e:
            print(f"[DEBUG] Player {self.player_id}: ERROR parsing action: {e}", flush=True)
            raise

        print(f"Player {self.player_id} ({player_names[self.player_id]}) response: {repr(action_text[:100] if action_text else action_text)}", flush=True)

        # Get all legal moves
        all_moves = get_all_legal_moves(self.coordinator.state, self.player_id, self.valid_mask)

        if not all_moves:
            # No legal moves - game should end
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=await self._get_observation_async(),
                next_stop_condition=self.stop_condition,
                metrics={"no_legal_moves": 1},
            )

        # Parse move selection
        move_idx = parse_move_selection(action_text, len(all_moves))

        if move_idx is None:
            # Invalid selection
            self.retry_count += 1
            if self.retry_count > MAX_INVALID_RETRIES:
                print(f"Player {self.player_id} exceeded max retries ({MAX_INVALID_RETRIES}). Ending game for all.", flush=True)
                self.coordinator.game_done = True
                async with self.coordinator.condition:
                    self.coordinator.condition.notify_all()

                return StepResult(
                    reward=ILLEGAL_MOVE_REWARD,
                    episode_done=True,
                    next_observation=await self._get_observation_async(),
                    next_stop_condition=self.stop_condition,
                    metrics={"invalid_selection_exceeded_retries": 1},
                )

            # Retry
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=await self._get_observation_with_error_async("Invalid move selection. Choose a number from the list."),
                next_stop_condition=self.stop_condition,
                metrics={"invalid_move_selection": 1},
            )

        # Valid move selected
        src_row, src_col, dest_row, dest_col, piece_name = all_moves[move_idx]
        move_text = f"[{piece_name} ({src_row}, {src_col}) -> ({dest_row}, {dest_col})]"

        print(f"Player {self.player_id} ({player_names[self.player_id]}) moves: {piece_name} ({src_row}, {src_col}) -> ({dest_row}, {dest_col})")

        # Encode and execute move
        action_idx = encode_action(src_row, src_col, dest_row, dest_col, 0, self.valid_mask)
        reward, info = await self.coordinator.make_move(self.player_id, action_idx, move_text)

        move_valid = info.get("move_valid", False)
        if move_valid:
            print(f"  ✓ Move successful! Reward: {reward:.2f}")
        else:
            print(f"  ✗ Move was invalid (this shouldn't happen!)")

        # Reset retry count
        self.retry_count = 0

        # Create metrics
        metrics = {
            "move": move_text,
            "player": player_names[self.player_id],
            "move_number": int(self.coordinator.state.move_count),
            "reward": float(reward),
            "score": int(self.coordinator.state.player_scores[self.player_id]),
        }

        # Wait for next turn
        await self.wait_for_turn()

        # Print header for next turn if game continues
        if not self.coordinator.game_done:
            player_colors = ["🔴", "🔵", "🟡", "🟢"]
            print(f"\n{'=' * 60}")
            print(f"{player_colors[self.player_id]} Player {self.player_id} ({player_names[self.player_id]}) - Move {int(self.coordinator.state.move_count)}")
            print(f"{'=' * 60}")

        return StepResult(
            reward=reward,
            episode_done=self.coordinator.game_done,
            next_observation=await self._get_observation_async(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def get_done_step(self) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=await self._get_observation_async(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )

    async def _get_observation_async(self) -> types.ModelInput:
        """Get observation showing all legal moves (async to avoid blocking)."""
        loop = asyncio.get_running_loop()
        observation_text = await loop.run_in_executor(
            None,
            format_move_selection_prompt,
            self.coordinator.state,
            self.player_id,
            self.valid_mask
        )
        return self.renderer.build_generation_prompt([{"role": "user", "content": observation_text}])

    async def _get_observation_with_error_async(self, error_msg: str) -> types.ModelInput:
        """Get observation with an error message (async)."""
        loop = asyncio.get_running_loop()
        observation_text = await loop.run_in_executor(
            None,
            format_move_selection_prompt,
            self.coordinator.state,
            self.player_id,
            self.valid_mask
        )
        full_text = f"ERROR: {error_msg}\n\n{observation_text}"
        return self.renderer.build_generation_prompt([{"role": "user", "content": full_text}])

    def get_observation(self) -> types.ModelInput:
        """Get observation showing all legal moves."""
        observation_text = format_move_selection_prompt(
            self.coordinator.state, self.player_id, self.valid_mask
        )
        return self.renderer.build_generation_prompt([{"role": "user", "content": observation_text}])

    def get_observation_with_error(self, error_msg: str) -> types.ModelInput:
        """Get observation with an error message."""
        observation_text = format_move_selection_prompt(
            self.coordinator.state, self.player_id, self.valid_mask
        )
        full_text = f"ERROR: {error_msg}\n\n{observation_text}"
        return self.renderer.build_generation_prompt([{"role": "user", "content": full_text}])


@dataclass
class FourPlayerChessEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of four player chess environments sharing the same game."""

    renderer: Renderer
    num_envs: int
    self_play: bool
    num_players: ClassVar[int] = 4
    opponent_policies: list[TinkerMessageCompleter | None] | None = None

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, dict]]:
        """Compute final rewards and trajectory-level metrics."""
        coordinator = env_group[0].coordinator  # type: ignore

        # Get final game state
        final_state = coordinator.state
        player_names = ["Red", "Blue", "Yellow", "Green"]
        player_colors = ["🔴", "🔵", "🟡", "🟢"]

        scores = [int(s) for s in final_state.player_scores]
        active_players = [bool(a) for a in final_state.player_active]

        winner_score = -1
        winner_id = -1
        for i, (score, active) in enumerate(zip(scores, active_players)):
            if active and score > winner_score:
                winner_score = score
                winner_id = i

        # Log game summary
        if coordinator.move_history:
            logger.info("\n" + "=" * 60)
            logger.info("GAME SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total moves: {len(coordinator.move_history)}")
            logger.info(f"Winner: {player_colors[winner_id]} {player_names[winner_id]} (score: {winner_score})")
            logger.info("\nFinal Scores:")
            for i, (score, active) in enumerate(zip(scores, active_players)):
                status = "Active" if active else "Eliminated"
                logger.info(f"  {player_colors[i]} {player_names[i]}: {score} points ({status})")

        # Create trajectory-level metrics
        results = []
        for i, env in enumerate(env_group):
            player_id = env.player_id  # type: ignore
            metrics = {
                "game_length": len(coordinator.move_history),
                "final_score": scores[player_id],
                "won_game": 1 if player_id == winner_id else 0,
                "survived": 1 if active_players[player_id] else 0,
                "winner_player": player_names[winner_id] if winner_id >= 0 else "None",
            }
            results.append((0.0, metrics))

        return results

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments sharing the same chess game."""
        if self.num_envs % 4 != 0:
            raise ValueError("num_envs must be divisible by 4 (one env per player)")

        def _construct_coordinator() -> FourPlayerCoordinator:
            jax_env = fpc.FourPlayerChessEnv()
            key = jax.random.PRNGKey(0)
            initial_state, _ = jax_env.reset(key)
            return FourPlayerCoordinator(jax_env=jax_env, initial_state=initial_state)

        envs = []
        for _ in range(self.num_envs // 4):
            if self.self_play:
                coordinator = _construct_coordinator()
                coordinators = [coordinator for _ in range(self.num_players)]
                opponent_lists = [[None, None, None] for _ in range(self.num_players)]
            else:
                coordinators = [_construct_coordinator() for _ in range(self.num_players)]
                opponent_lists = [self.opponent_policies for _ in range(self.num_players)]

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
        assert self.num_datapoints % self.builder.num_players == 0

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
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.model_name)

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
