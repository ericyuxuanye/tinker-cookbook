"""Supervised warm-start for chess to teach move format before RL."""

import random
from tinker_cookbook.recipes.multiplayer_rl.four_player_chess.env import (
    get_all_pseudo_legal_moves,
    board_to_text,
)
import four_player_chess as fpc
import jax


def generate_valid_move_examples(num_examples: int = 100):
    """Generate examples of valid chess moves for supervised learning."""
    examples = []

    for _ in range(num_examples):
        # Create a new game
        env = fpc.FourPlayerChessEnv()
        key = jax.random.PRNGKey(random.randint(0, 10000))
        state, _ = env.reset(key)
        valid_mask = fpc.board.create_valid_square_mask()

        # Make a few random moves to get variety
        for _ in range(random.randint(0, 5)):
            # Get all pseudo-legal moves for current player
            player_id = int(state.current_player)
            moves = get_all_pseudo_legal_moves(state, player_id, valid_mask)

            if not moves:
                break

            # Pick random move
            src_row, src_col, dest_row, dest_col, piece = random.choice(moves)
            move_text = f"<think>\nI will move {piece} from ({src_row}, {src_col}) to ({dest_row}, {dest_col})\n</think>\n[({src_row}, {src_col}) -> ({dest_row}, {dest_col})]"

            # Create conversation example
            prompt = board_to_text(state, player_id)
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": move_text}
                ]
            })

            # Execute the move (simplified - just for variety)
            break

    return examples


if __name__ == "__main__":
    examples = generate_valid_move_examples(100)

    import json
    with open("/tmp/chess_warmstart.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} examples at /tmp/chess_warmstart.jsonl")
    print("Now run supervised learning on this file before RL!")
