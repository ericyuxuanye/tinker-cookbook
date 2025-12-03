"Four-player chess environment for tinker RL."

import asyncio
import logging
import re
import random
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

STOP_CONDITION = ["\\]\n"]
ILLEGAL_MOVE_REWARD = -1.0  # Small penalty for illegal move (since we fallback)
MAX_INVALID_RETRIES = 0     # Immediate fallback


class FourPlayerCoordinator:
    """Coordinates a single four player chess game. See text_arena for similar pattern."""

    def __init__(self, jax_env: fpc.FourPlayerChessEnv, initial_state: fpc.EnvState):
        self.jax_env = jax_env
        self.state = initial_state  # Current JAX state
        self.condition = asyncio.Condition()
        self.game_done = False
        self.last_move_valid = True  # Track if the last attempted move was valid
        self.move_history: list[dict] = []  # Track all moves for logging

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

    async def make_move(self, player_id: int, move_action: int, move_text: str = "") -> tuple[float, dict]:
        """Make a move and notify waiting players. Returns (reward, info)."""
        async with self.condition:
            # print(f"Making move, {player_id=}, {move_action=}, {move_text=}")
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
            self.last_move_valid = move_valid

            if move_valid:
                # Log the move
                self.move_history.append({
                    "player_id": player_id,
                    "move_text": move_text,
                    "move_number": int(self.state.move_count),
                    "reward": float(reward),
                    "scores": [int(s) for s in next_state.player_scores],
                    "active_players": [bool(a) for a in next_state.player_active],
                })

                # Update state only if move was valid
                self.state = next_state
                self.game_done = bool(done)
                # Notify all waiting players about the state change
                self.condition.notify_all()
            else:
                # Invalid move - will be handled by the env's retry logic
                print(f"Invalid move by player {player_id}: {move_text}")

            return float(reward), dict(info)


def get_player_pieces(state: fpc.EnvState, player_id: int, valid_mask: jnp.ndarray) -> list[tuple[int, int, str]]:
    """Get all pieces owned by a player."""
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
    return pieces


def get_pseudo_legal_moves_for_piece(
    state: fpc.EnvState,
    row: int,
    col: int,
    player_id: int,
    valid_mask: jnp.ndarray
) -> list[tuple[int, int]]:
    """Get all pseudo-legal destination squares for a piece."""
    from four_player_chess.pieces import get_pseudo_legal_moves

    # Get pseudo-legal moves mask
    pseudo_moves_mask = get_pseudo_legal_moves(
        state.board, row, col, player_id, valid_mask, state.en_passant_square
    )

    # Convert boolean mask to list of coordinates
    destinations = []
    rows, cols = jnp.where(pseudo_moves_mask)
    for r, c in zip(rows, cols):
        destinations.append((int(r), int(c)))

    return destinations


def get_all_pseudo_legal_moves(
    state: fpc.EnvState,
    player_id: int,
    valid_mask: jnp.ndarray
) -> list[tuple[int, int, int, int, str]]:
    """Get all pseudo-legal moves for a player.

    Returns: List of (source_row, source_col, dest_row, dest_col, piece_name) tuples.
    """
    pieces = get_player_pieces(state, player_id, valid_mask)
    all_moves = []

    for row, col, piece_name in pieces:
        destinations = get_pseudo_legal_moves_for_piece(state, row, col, player_id, valid_mask)
        for dest_row, dest_col in destinations:
            all_moves.append((row, col, dest_row, dest_col, piece_name))

    return all_moves


def get_random_legal_move(
    state: fpc.EnvState,
    player_id: int,
    valid_mask: jnp.ndarray
) -> tuple[int, int, int, int, str] | None:
    """
    Find a random legal move by sampling pseudo-legal moves.
    Returns: (source_row, source_col, dest_row, dest_col, piece_name) or None
    """
    from four_player_chess.rules import is_move_legal

    # 1. Get all pseudo-legal moves (fast)
    pseudo_moves = get_all_pseudo_legal_moves(state, player_id, valid_mask)
    
    # 2. Shuffle to ensure randomness
    random.shuffle(pseudo_moves)

    # 3. Check legality one by one until we find a valid one
    for move in pseudo_moves:
        src_row, src_col, dest_row, dest_col, piece_name = move
        
        is_legal = is_move_legal(
            state.board,
            src_row, src_col,
            dest_row, dest_col,
            player_id,
            state.king_positions[player_id],
            state.player_active,
            valid_mask,
            state.en_passant_square
        )
        
        if is_legal:
            return move

    return None


def board_to_text(state: fpc.EnvState, player_id: int) -> str:
    """Convert JAX board state to text description for the LLM."""
    lines = []

    # Header with current player info
    player_names = ["Red", "Blue", "Yellow", "Green"]
    player_colors = ["🔴", "🔵", "🟡", "🟢"]
    player_positions = [
        "BOTTOM (rows 12-13)",
        "RIGHT (columns 12-13)",
        "TOP (rows 0-1)",
        "LEFT (columns 0-1)"
    ]
    player_piece_prefixes = ["r", "b", "y", "g"]

    lines.append(f"You are {player_colors[player_id]} {player_names[player_id]}.")
    lines.append(f"YOUR PIECES: Look for pieces starting with '{player_piece_prefixes[player_id]}' in {player_positions[player_id]}")
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

    # List current player's pieces for clarity
    if int(state.move_count) == 0:  # Only show starting pieces at game start
        lines.append("YOUR STARTING PIECES:")
        if player_id == 0:  # Red
            lines.append("  Row 12: rP at columns 3,4,5,6,7,8,9,10 (8 pawns)")
            lines.append("  Row 13: rR(3), rN(4), rB(5), rQ(6), rK(7), rB(8), rN(9), rR(10)")
        elif player_id == 1:  # Blue
            lines.append("  Col 12: bP at rows 3,4,5,6,7,8,9,10 (8 pawns)")
            lines.append("  Col 13: bR(3), bN(4), bB(5), bQ(6), bK(7), bB(8), bN(9), bR(10)")
        elif player_id == 2:  # Yellow
            lines.append("  Row 1: yP at columns 3,4,5,6,7,8,9,10 (8 pawns)")
            lines.append("  Row 0: yR(3), yN(4), yB(5), yK(6), yQ(7), yB(8), yN(9), yR(10)")
        elif player_id == 3:  # Green
            lines.append("  Col 1: gP at rows 3,4,5,6,7,8,9,10 (8 pawns)")
            lines.append("  Col 0: gR(3), gN(4), gB(5), gK(6), gQ(7), gB(8), gN(9), gR(10)")
        lines.append("")

    lines.append("MOVEMENT RULES:")
    lines.append("  Pawn (P): Moves forward 1 square (or 2 from starting position), captures diagonally")
    lines.append("  Knight (N): Moves in L-shape (2 squares in one direction, 1 perpendicular)")
    lines.append("  Bishop (B): Moves diagonally any number of squares")
    lines.append("  Rook (R): Moves horizontally or vertically any number of squares")
    lines.append("  Queen (Q): Moves in any direction any number of squares")
    lines.append("  King (K): Moves 1 square in any direction")
    lines.append("")
    lines.append("OUTPUT FORMAT:")
    lines.append("First analyze the position, then output your move.")
    lines.append("")
    lines.append("EXAMPLES:")
    lines.append("Analysis: I'll advance my center pawn for better control.")
    lines.append("Move: [(12, 6) -> (11, 6)]")
    lines.append("")
    lines.append("Analysis: Knight development to control key squares.")
    lines.append("Move: [(13, 4) -> (11, 5)]")
    lines.append("")
    lines.append("Now analyze and output your move in format [(row, col) -> (row, col)]:")

    return "\n".join(lines)


def parse_move_action(action_text: str) -> tuple[int, int, int, int, int]:
    """
    Parse move from LLM output.
    Returns: (source_row, source_col, dest_row, dest_col, promotion_type)

    Extracts the LAST valid move pattern found in the text.
    For GPT-OSS, extracts from the final channel if present.
    """
    # For GPT-OSS format, try to extract from final channel first
    final_channel_match = re.search(r'<\|channel\|>final<\|message\|>(.+?)(?:<\|end\|>|$)', action_text, re.DOTALL)
    if final_channel_match:
        # Use only the final channel content
        action_text = final_channel_match.group(1)

    # Try to match pattern like [(12, 6) -> (10, 6)] or [(7, 3) -> (3, 3)=Q]
    pattern = r"\[\((\d+),\s*(\d+)\)\s*->\s*\((\d+),\s*(\d+)\)\]\s*(?:=([QRBN]))?"

    # Find ALL matches and use the last one (most likely to be the actual move)
    matches = list(re.finditer(pattern, action_text))

    if not matches:
        return -1, -1, -1, -1, 0

    # Use the last match found
    match = matches[-1]

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
    retry_count: int = 0  # Track number of invalid move retries

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
        print(f"[DEBUG] Player {self.player_id} wait_for_turn: game_done={self.coordinator.game_done}, current_player={self.coordinator.current_player_id}, self_play={self.self_play}", flush=True)
        if not self.coordinator.game_done:
            if self.self_play:
                print(f"[DEBUG] Player {self.player_id} waiting in self_play mode", flush=True)
                await self.coordinator.wait_across_env(self.player_id)
                print(f"[DEBUG] Player {self.player_id} done waiting in self_play mode", flush=True)
            else:
                # Opponents take their turns
                print(f"[DEBUG] Player {self.player_id} entering opponent turn loop", flush=True)
                while (
                    not self.coordinator.game_done
                    and self.coordinator.current_player_id != self.player_id
                ):
                    print(f"[DEBUG] Player {self.player_id} calling opponent_step (current_player={self.coordinator.current_player_id})", flush=True)
                    await self.opponent_step()
                print(f"[DEBUG] Player {self.player_id} exited opponent turn loop", flush=True)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        print(f"[DEBUG] Player {self.player_id} initial_observation called", flush=True)
        if self.player_id != 0:
            print(f"[DEBUG] Player {self.player_id} waiting for turn before initial observation", flush=True)
            await self.wait_for_turn()
            print(f"[DEBUG] Player {self.player_id} done waiting for turn before initial observation", flush=True)

        # Print header for first move
        player_names = ["Red", "Blue", "Yellow", "Green"]
        player_colors = ["🔴", "🔵", "🟡", "🟢"]
        if self.coordinator.current_player_id == self.player_id:
            print(f"\n{'=' * 60}")
            print(f"{player_colors[self.player_id]} Player {self.player_id} ({player_names[self.player_id]}) - Move {int(self.coordinator.state.move_count)}")
            print(f"{'=' * 60}")

        print(f"[DEBUG] Player {self.player_id} getting observation", flush=True)
        obs = self.get_observation()
        print(f"[DEBUG] Player {self.player_id} returning observation", flush=True)
        return obs, self.stop_condition

    async def opponent_step(self) -> None:
        """When not self_play, an opponent policy takes a step"""
        print(f"[DEBUG] Player {self.player_id} opponent_step called (current_player={self.coordinator.current_player_id})", flush=True)
        # (Logic omitted for brevity, same as before but wrapped in proper checks if needed)
        # Assuming standard behavior is fine for opponents
        # TODO: Implement opponent logic - for now just waiting
        print(f"[DEBUG] Player {self.player_id} opponent_step is a STUB (not implemented)", flush=True)
        await asyncio.sleep(0.1)  # Small delay to avoid tight loop
        pass

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        print(f"[DEBUG] Player {self.player_id} step called, game_done={self.coordinator.game_done}", flush=True)
        if self.coordinator.game_done:
            print(f"[DEBUG] Player {self.player_id} game is done, returning done step", flush=True)
            return self.get_done_step()

        print(f"[DEBUG] Player {self.player_id} checking turn (current={self.coordinator.current_player_id})", flush=True)
        assert self.coordinator.current_player_id == self.player_id, "Not the current player's turn"

        # Parse action from LLM output
        print(f"[DEBUG] Player {self.player_id} parsing action", flush=True)
        try:
            action_message: Message = self.renderer.parse_response(action)[0]
            action_text = action_message["content"]
            print(f"[DEBUG] Player {self.player_id} raw action text (first 200 chars): {repr(action_text[:200])}", flush=True)

            # Check if final channel exists
            if "<|channel|>final<|message|>" in action_text:
                print(f"[DEBUG] Player {self.player_id} found final channel in output", flush=True)
            else:
                print(f"[DEBUG] Player {self.player_id} WARNING: No final channel found!", flush=True)

        except Exception as e:
             print(f"[DEBUG] Player {self.player_id} error parsing response: {e}", flush=True)
             action_text = ""

        source_row, source_col, dest_row, dest_col, promotion = parse_move_action(action_text)
        print(f"[DEBUG] Player {self.player_id} parsed move: ({source_row}, {source_col}) -> ({dest_row}, {dest_col})", flush=True)
        action_idx = encode_action(source_row, source_col, dest_row, dest_col,
                                   promotion, self.valid_mask)
        
        # If parse failed (invalid format), action_idx will be 0 (invalid) or -1/-1/-1/-1 will result in 0

        # Execute move
        reward, info = await self.coordinator.make_move(self.player_id, action_idx, action_text)

        # Check if move was valid
        if not self.coordinator.last_move_valid:
            # Invalid move - immediately fallback to random legal move
            print(f"Invalid move by Player {self.player_id}: {action_text}. Fallback to random.", flush=True)
            
            # Get a random legal move (run in executor to avoid blocking)
            loop = asyncio.get_running_loop()
            rand_move = await loop.run_in_executor(
                None,
                get_random_legal_move,
                self.coordinator.state,
                self.player_id,
                self.valid_mask
            )

            if rand_move:
                # Pick random move
                src_r, src_c, dst_r, dst_c, piece = rand_move
                rand_move_text = f"[Random Fallback: {piece} ({src_r}, {src_c}) -> ({dst_r}, {dst_c})]"
                print(f"  -> Executing: {rand_move_text}")

                rand_action_idx = encode_action(src_r, src_c, dst_r, dst_c, 0, self.valid_mask)
    
                # Execute random move
                reward, info = await self.coordinator.make_move(self.player_id, rand_action_idx, rand_move_text)
                
                # Apply penalty for invalid attempt
                reward = ILLEGAL_MOVE_REWARD
                
                metrics = {
                    "invalid_move_fallback": 1,
                    "move": rand_move_text,
                    "original_move": action_text,
                    "reward": float(reward),
                }
            else:
                # No legal moves available (stalemate/checkmate but game not detected yet?)
                # Or simple failure.
                print(f"  -> No legal moves available for random fallback. Ending episode.")
                return StepResult(
                    reward=ILLEGAL_MOVE_REWARD,
                    episode_done=True,
                    next_observation=self.get_observation(),
                    next_stop_condition=self.stop_condition,
                    metrics={"no_legal_moves_fallback": 1},
                )
        else:
            # Valid move
            metrics = {
                "move": action_text,
                "reward": float(reward),
            }

        # Wait for next turn
        await self.wait_for_turn()
        
        # Print header for next turn if game continues
        if not self.coordinator.game_done:
            player_names = ["Red", "Blue", "Yellow", "Green"]
            player_colors = ["🔴", "🔵", "🟡", "🟢"]
            print(f"\n{'=' * 60}")
            print(f"{player_colors[self.player_id]} Player {self.player_id} ({player_names[self.player_id]}) - Move {int(self.coordinator.state.move_count)}")
            print(f"{'=' * 60}")

        return StepResult(
            reward=reward,
            episode_done=self.coordinator.game_done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def get_done_step(self) -> StepResult:
        # Even when done, provide a valid observation showing final state
        observation_text = board_to_text(self.coordinator.state, self.player_id)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt([{"role": "user", "content": observation_text}]),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )

    def get_observation(self) -> types.ModelInput:
        """Get text observation for current player. Always returns a valid observation."""
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

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, dict]]:
        """
        Compute final rewards and trajectory-level metrics for each player.
        """
        coordinator = env_group[0].coordinator  # type: ignore
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
        print(f"[DEBUG] make_envs: Creating environments, num_envs={self.num_envs}, self_play={self.self_play}", flush=True)
        if self.num_envs % 4 != 0:
            raise ValueError("num_envs must be divisible by 4 (one env per player)")

        def _construct_coordinator() -> FourPlayerCoordinator:
            jax_env = fpc.FourPlayerChessEnv()
            key = jax.random.PRNGKey(0)
            initial_state, _ = jax_env.reset(key)
            return FourPlayerCoordinator(jax_env=jax_env, initial_state=initial_state)

        envs = []
        for game_idx in range(self.num_envs // 4):
            print(f"[DEBUG] make_envs: Creating game {game_idx}", flush=True)
            if self.self_play:
                print(f"[DEBUG] make_envs: Game {game_idx} - self_play mode, creating shared coordinator", flush=True)
                coordinator = _construct_coordinator()
                coordinators = [coordinator for _ in range(self.num_players)]
                opponent_lists = [[None, None, None] for _ in range(self.num_players)]
            else:
                print(f"[DEBUG] make_envs: Game {game_idx} - non-self_play mode, creating separate coordinators", flush=True)
                coordinators = [_construct_coordinator() for _ in range(self.num_players)]
                opponent_lists = [self.opponent_policies for _ in range(self.num_players)]
                print(f"[DEBUG] make_envs: Game {game_idx} - opponent_policies={self.opponent_policies is not None}", flush=True)

            valid_mask = fpc.board.create_valid_square_mask()

            print(f"[DEBUG] make_envs: Game {game_idx} - creating 4 player environments", flush=True)
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

        print(f"[DEBUG] make_envs: Created {len(envs)} total environments", flush=True)
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
