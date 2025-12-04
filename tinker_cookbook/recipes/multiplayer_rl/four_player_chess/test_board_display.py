"""Simple script to test board display and manual move input."""

import asyncio
import jax
import four_player_chess as fpc
from tinker_cookbook.recipes.multiplayer_rl.four_player_chess.env import (
    FourPlayerCoordinator,
    board_to_text,
    parse_move_action,
    encode_action,
    get_random_legal_move,
)


async def main():
    # Create a new game
    jax_env = fpc.FourPlayerChessEnv()
    key = jax.random.PRNGKey(0)
    initial_state, _ = jax_env.reset(key)
    coordinator = FourPlayerCoordinator(jax_env=jax_env, initial_state=initial_state)
    valid_mask = fpc.board.create_valid_square_mask()

    print("=" * 80)
    print("FOUR PLAYER CHESS - MANUAL TEST")
    print("=" * 80)
    print()

    # Show initial board for all players
    while True:
        print("\n" + "=" * 80)
        print(f"MOVE {int(coordinator.state.move_count)}")
        print("=" * 80)

        # Get current player
        current_player = int(coordinator.state.current_player)
        player_names = ["Red", "Blue", "Yellow", "Green"]
        player_colors = ["🔴", "🔵", "🟡", "🟢"]

        print(f"\nCurrent Player: {player_colors[current_player]} {player_names[current_player]}")
        print()

        # Display board for current player
        board_text = board_to_text(coordinator.state, current_player, coordinator)
        print(board_text)
        print()

        # Get move input
        print("-" * 80)
        print("OPTIONS:")
        print("  1. Enter move manually: [(row, col) -> (row, col)]")
        print("  2. Type 'random' for random legal move")
        print("  3. Type 'all' to see all 4 player perspectives")
        print("  4. Type 'quit' to exit")
        print("-" * 80)

        user_input = input("Your choice: ").strip()

        if user_input.lower() == 'quit':
            print("\nExiting...")
            break

        if user_input.lower() == 'all':
            print("\n" + "=" * 80)
            print("ALL PLAYER PERSPECTIVES")
            print("=" * 80)
            for pid in range(4):
                print(f"\n{'=' * 80}")
                print(f"{player_colors[pid]} {player_names[pid]} Perspective")
                print(f"{'=' * 80}")
                print(board_to_text(coordinator.state, pid, coordinator))
            print()
            continue

        if user_input.lower() == 'random':
            # Get random legal move
            rand_move = get_random_legal_move(coordinator.state, current_player, valid_mask)
            if rand_move:
                src_r, src_c, dst_r, dst_c, piece = rand_move
                move_text = f"[(({src_r}, {src_c}) -> ({dst_r}, {dst_c})] - {piece}"
                print(f"\n🎲 Random move: {move_text}")
                action_idx = encode_action(src_r, src_c, dst_r, dst_c, 0, valid_mask)
            else:
                print("\n❌ No legal moves available!")
                continue
        else:
            # Parse manual input
            move_text = user_input
            src_r, src_c, dst_r, dst_c, promo = parse_move_action(move_text)

            if src_r == -1:
                print(f"\n❌ Invalid format! Could not parse: {move_text}")
                print("   Expected format: [(12, 6) -> (11, 6)]")
                continue

            print(f"\n✓ Parsed: ({src_r}, {src_c}) -> ({dst_r}, {dst_c})")
            action_idx = encode_action(src_r, src_c, dst_r, dst_c, promo, valid_mask)

        # Execute move
        key = jax.random.PRNGKey(0)
        next_state, obs, reward, done, info = jax_env.step(key, coordinator.state, action_idx)

        move_valid = info.get("move_valid", False)

        if move_valid:
            print(f"✅ Move executed successfully!")
            print(f"   Reward: {float(reward):.2f}")
            coordinator.move_history.append({
                "player_id": current_player,
                "move_text": move_text,
                "move_number": int(coordinator.state.move_count),
                "reward": float(reward),
                "scores": [int(s) for s in next_state.player_scores],
                "active_players": [bool(a) for a in next_state.player_active],
            })
            coordinator.state = next_state
            coordinator.game_done = bool(done)

            if done:
                print("\n" + "=" * 80)
                print("GAME OVER!")
                print("=" * 80)
                print(f"Final board state:")
                print(board_to_text(coordinator.state, current_player, coordinator))
                print("\nFinal scores:")
                for pid in range(4):
                    score = int(coordinator.state.player_scores[pid])
                    active = bool(coordinator.state.player_active[pid])
                    status = "Active" if active else "Eliminated"
                    print(f"  {player_colors[pid]} {player_names[pid]}: {score} points ({status})")
                break
        else:
            print(f"❌ Invalid move! The move was illegal.")
            print(f"   Try again or type 'random' for a legal move.")
            continue


if __name__ == "__main__":
    asyncio.run(main())
