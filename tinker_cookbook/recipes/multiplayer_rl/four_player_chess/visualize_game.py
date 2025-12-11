"""Visualize a 4-player chess game from JSON log as a GIF."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import four_player_chess as fpc
import jax


# Player colors matching web UI
PLAYER_COLORS = {
    0: "#e74c3c",  # Red
    1: "#3498db",  # Blue
    2: "#f39c12",  # Yellow
    3: "#27ae60",  # Green
}

PLAYER_NAMES = ["Red", "Blue", "Yellow", "Green"]
PLAYER_SYMBOLS = ["🔴", "🔵", "🟡", "🟢"]

# Piece symbols (Unicode chess pieces - matching web UI)
PIECE_SYMBOLS = {
    1: "♟",   # Pawn
    2: "♞",   # Knight
    3: "♝",   # Bishop
    4: "♜",   # Rook
    5: "♛",   # Queen
    6: "♚",   # King
}

# Board colors matching web UI
LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
INVALID_SQUARE = "#888888"
HIGHLIGHT_SOURCE = "#fdd835"  # Yellow highlight for source
HIGHLIGHT_DEST = "#7fc97f"    # Green highlight for destination


def parse_move_text(move_text: str) -> tuple[int, int, int, int] | None:
    """Parse move text like '[(12, 6) -> (11, 6)]' to get source and dest."""
    import re

    # Try multiple patterns
    patterns = [
        r'\((\d+),\s*(\d+)\)\s*->\s*\((\d+),\s*(\d+)\)',
        r'\[.*?\((\d+),\s*(\d+)\).*?->\s*\((\d+),\s*(\d+)\)',
    ]

    for pattern in patterns:
        match = re.search(pattern, move_text)
        if match:
            src_r, src_c, dst_r, dst_c = map(int, match.groups())
            return src_r, src_c, dst_r, dst_c
    return None


def render_board_state(state: fpc.EnvState,
                       last_move: tuple[int, int, int, int] | None = None,
                       move_number: int = 0,
                       current_player: int = 0,
                       move_text: str = "") -> Image.Image:
    """Render a board state as a PIL Image matching the web UI style."""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    ax.set_xlim(-0.5, 18.5)  # Extended for info panel
    ax.set_ylim(-1.5, 13.5)  # Extended for title
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Row 0 at top

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Draw border around valid board area
    border = mpatches.Rectangle((-0.5, -0.5), 14, 14,
                                facecolor='none', edgecolor='#333', linewidth=3, zorder=1)
    ax.add_patch(border)

    # Draw board squares
    board = state.board
    # Channel 3 is CHANNEL_VALID_SQUARE
    CHANNEL_VALID_SQUARE = 3
    CHANNEL_PIECE_TYPE = 0
    CHANNEL_OWNER = 1

    for row in range(14):
        for col in range(14):
            # Check if square is valid (part of cross shape)
            is_valid = board[row, col, CHANNEL_VALID_SQUARE] > 0

            if not is_valid:
                # Invalid square (corners) - gray background
                rect = mpatches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor=INVALID_SQUARE, edgecolor='none', zorder=0)
                ax.add_patch(rect)
                continue

            # Checkerboard pattern for valid squares
            is_light = (row + col) % 2 == 0
            color = LIGHT_SQUARE if is_light else DARK_SQUARE

            # Highlight last move
            if last_move is not None:
                src_r, src_c, dst_r, dst_c = last_move
                if (row == src_r and col == src_c):
                    color = HIGHLIGHT_SOURCE  # Yellow for source
                elif (row == dst_r and col == dst_c):
                    color = HIGHLIGHT_DEST    # Green for destination

            rect = mpatches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                     facecolor=color, edgecolor='#666', linewidth=0.5, zorder=0)
            ax.add_patch(rect)

            # Draw piece
            piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
            if piece_type != 0:  # 0 = EMPTY
                owner = int(board[row, col, CHANNEL_OWNER])

                # Get piece symbol (piece types: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King)
                symbol = PIECE_SYMBOLS.get(piece_type, "?")

                piece_color = PLAYER_COLORS[owner]

                # Draw piece directly (no background circle - cleaner like web UI)
                ax.text(col, row, symbol, fontsize=36, ha='center', va='center',
                       color=piece_color, weight='bold', zorder=2,
                       family='DejaVu Sans')

    # Add move info box at the top (styled like web UI)
    if move_number > 0:
        title_text = f"4-Player Chess - Move {move_number}"
        ax.text(7, -1.3, title_text, fontsize=18, weight='bold', ha='center',
               color='#333')

        # Current player indicator (use text instead of emoji to avoid font issues)
        player_text = f"{PLAYER_NAMES[current_player]}'s Turn"
        ax.text(7, -0.5, player_text, fontsize=14, weight='bold', ha='center',
               color=PLAYER_COLORS[current_player])

        # Show move text if available
        if move_text:
            display_text = move_text if len(move_text) < 50 else move_text[:50] + "..."
            ax.text(7, 0.2, display_text, fontsize=9, ha='center',
                   color='#666', family='monospace', style='italic')
    else:
        title_text = "4-Player Chess - Initial Position"
        ax.text(7, -0.8, title_text, fontsize=18, weight='bold', ha='center',
               color='#333')

    # Add player status panel on the right (styled like web UI info panel)
    panel_x = 14.5
    panel_y_start = 1

    for i in range(4):
        active = bool(state.player_active[i])
        score = int(state.player_scores[i])

        y_offset = panel_y_start + i * 2.5

        # Player name with color (no emoji to avoid font issues)
        name_text = PLAYER_NAMES[i]
        if i == current_player and move_number > 0 and active:
            name_text = f"> {name_text}"  # Arrow for current player

        ax.text(panel_x, y_offset, name_text, fontsize=13, weight='bold',
               color=PLAYER_COLORS[i], va='top')

        # Score
        score_text = f"Score: {score}"
        ax.text(panel_x, y_offset + 0.5, score_text, fontsize=11,
               color='#666', va='top')

        # Status (use text instead of emoji)
        if active:
            status_text = "[Active]"
            status_color = '#27ae60'
        else:
            status_text = "[Eliminated]"
            status_color = '#e74c3c'

        ax.text(panel_x, y_offset + 1.0, status_text, fontsize=11,
               color=status_color, weight='bold', va='top')

        # Background box for current player
        if i == current_player and move_number > 0 and active:
            highlight_box = mpatches.FancyBboxPatch(
                (panel_x - 0.2, y_offset - 0.3), 3.5, 2.0,
                boxstyle="round,pad=0.1", edgecolor=PLAYER_COLORS[i],
                facecolor='#e8f5e9', linewidth=2, alpha=0.3, zorder=-1
            )
            ax.add_patch(highlight_box)

    # Add move count at bottom
    move_count_text = f"Total Moves: {int(state.move_count)}"
    ax.text(7, 14.2, move_count_text, fontsize=10, ha='center',
           color='#999')

    # Convert figure to PIL Image
    fig.tight_layout(pad=0.5)

    # Use BytesIO buffer - more robust across platforms
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()  # Copy so we can close the buffer
    buf.close()
    plt.close(fig)

    return img


def apply_move_to_state(state: fpc.EnvState, move: tuple[int, int, int, int]) -> fpc.EnvState:
    """Apply a move to the board state (simple version - doesn't handle all chess rules)."""
    src_r, src_c, dst_r, dst_c = move

    # Copy the board (JAX arrays are immutable)
    import jax.numpy as jnp

    # Move piece to destination (copy all 4 channels)
    new_board = state.board.at[dst_r, dst_c].set(state.board[src_r, src_c])

    # Mark destination piece as having moved (channel 2)
    new_board = new_board.at[dst_r, dst_c, 2].set(1)

    # Clear source square
    new_board = new_board.at[src_r, src_c, 0].set(0)  # Clear piece type
    new_board = new_board.at[src_r, src_c, 1].set(0)  # Clear owner
    new_board = new_board.at[src_r, src_c, 2].set(0)  # Clear has_moved

    # Note: We don't update channel 3 (valid_square) as that's fixed

    # Update move count
    new_move_count = state.move_count + 1

    # Advance current player (simple rotation, assuming no eliminated players)
    new_current_player = (state.current_player + 1) % 4

    # Create new state with updated board
    new_state = fpc.EnvState(
        board=new_board,
        current_player=new_current_player,
        player_scores=state.player_scores,
        player_active=state.player_active,
        move_count=new_move_count,
        en_passant_square=state.en_passant_square,
        king_positions=state.king_positions,
        castling_rights=state.castling_rights,
        last_capture_move=state.last_capture_move,
        promoted_pieces=state.promoted_pieces,
    )

    return new_state


def create_game_gif(json_path: str, output_path: str, frame_duration: int = 1000):
    """Create a GIF from a game JSON log.

    Args:
        json_path: Path to JSON log file
        output_path: Path to save GIF
        frame_duration: Duration per frame in milliseconds (default: 1000ms = 1s)
    """
    print(f"Loading game log from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Model: {data['config']['model_name']}")
    print(f"Total moves: {data['total_moves']}")

    if data.get('interrupted', False):
        print("⚠️  Note: This was an interrupted game (partial data)")

    # Initialize JAX environment for initial state
    jax_env = fpc.FourPlayerChessEnv()
    current_state, initial_obs = jax_env.reset(jax.random.key(0))

    print("\nGenerating frames...")
    frames = []

    # Add initial state frame (hold for 2x duration)
    print("  Frame 0: Initial position")
    frame = render_board_state(current_state, last_move=None, move_number=0,
                               current_player=0, move_text="")
    frames.append((frame, frame_duration * 2))  # Hold initial position longer

    # Use move_history for chronological order (preferred) or fall back to reconstruction
    move_history = data.get('move_history', [])

    if move_history:
        # Use the actual chronological order from the coordinator
        print("Using move_history for accurate chronological order")

        # Cap at 200 frames for reasonable GIF size
        max_frames = min(len(move_history), 200)

        for idx, move_info in enumerate(move_history[:max_frames]):
            player_id = move_info.get('player_id', 0)
            move_text = move_info.get('move_text', 'Unknown move')

            # Parse move
            parsed = parse_move_text(move_text)
            move_number = idx + 1

            if parsed:
                print(f"  Frame {move_number}: Player {player_id} - {move_text[:50]}")

                # Apply move to update board state
                current_state = apply_move_to_state(current_state, parsed)

                # Render frame with updated board state
                frame = render_board_state(current_state, last_move=parsed,
                                          move_number=move_number,
                                          current_player=player_id,
                                          move_text=move_text)
            else:
                print(f"  Frame {move_number}: Player {player_id} - (unparseable move)")
                # Can't apply unparseable move, just show current state
                frame = render_board_state(current_state, last_move=None,
                                          move_number=move_number,
                                          current_player=player_id,
                                          move_text=move_text)

            frames.append((frame, frame_duration))

    else:
        # Fallback: reconstruct order by round-robin (less accurate with eliminations)
        print("⚠️  No move_history found - using round-robin reconstruction (may be inaccurate)")

        player_move_queues = {i: [] for i in range(4)}
        for player_data in data['players']:
            player_id = player_data['player_id']
            player_move_queues[player_id] = list(player_data['moves'])

        current_player = int(current_state.current_player)
        move_number = 0
        max_total_moves = min(data['total_moves'], 200)

        while move_number < max_total_moves:
            if len(player_move_queues[current_player]) > 0:
                move_data = player_move_queues[current_player].pop(0)
                move_text = move_data['metrics'].get('move', 'Unknown move')
                parsed = parse_move_text(move_text)
                move_number += 1

                if parsed:
                    print(f"  Frame {move_number}: Player {current_player} - {move_text[:50]}")
                    current_state = apply_move_to_state(current_state, parsed)
                    frame = render_board_state(current_state, last_move=parsed,
                                              move_number=move_number,
                                              current_player=current_player,
                                              move_text=move_text)
                else:
                    print(f"  Frame {move_number}: Player {current_player} - (unparseable move)")
                    frame = render_board_state(current_state, last_move=None,
                                              move_number=move_number,
                                              current_player=current_player,
                                              move_text=move_text)

                frames.append((frame, frame_duration))

            current_player = (current_player + 1) % 4

            if all(len(q) == 0 for q in player_move_queues.values()):
                break

    if len(frames) == 0:
        print("Error: No frames generated!")
        return

    print(f"\nGenerated {len(frames)} frames")
    print(f"Creating GIF...")

    # Save as GIF using PIL
    images = [frame for frame, _ in frames]
    durations = [duration for _, duration in frames]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,  # Loop forever
        optimize=False,
    )

    total_duration = sum(durations) / 1000  # Convert to seconds
    print(f"✅ Saved GIF to: {output_path}")
    print(f"   {len(frames)} frames, {frame_duration}ms per frame")
    print(f"   Total duration: {total_duration:.1f}s")

    # File size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 4-player chess game as GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_game.py /tmp/chess_rollout.json
  python visualize_game.py /tmp/chess_rollout.json -o output.gif -d 800

Note: The visualization shows move highlights on the initial board state.
      For a full game replay with pieces moving, more complex reconstruction
      would be needed.
        """
    )
    parser.add_argument("json_file", type=str, help="Path to JSON log file")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output GIF path (default: same name as JSON with .gif extension)")
    parser.add_argument("-d", "--duration", type=int, default=1000,
                       help="Duration per frame in milliseconds (default: 1000ms)")

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    if args.output:
        output_path = args.output
    else:
        output_path = str(json_path.with_suffix('.gif'))

    create_game_gif(str(json_path), output_path, args.duration)
    return 0


if __name__ == "__main__":
    exit(main())
