"""Run a single 4-player chess rollout without training to test the environment."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.multiplayer_rl.four_player_chess.env import FourPlayerChessDatasetBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import Transition, Trajectory


@chz.chz
class Config:
    model_name: str = "openai/gpt-oss-20b"
    renderer_name: str | None = None
    max_tokens: int = 512
    temperature: float = 1.0
    output_file: str = "/tmp/chess_rollout.json"  # Where to save structured log

    # LoRA configuration for players
    # Format: "player_id:lora_path,player_id:lora_path,..."
    # Example: "0:tinker://my-lora-run,2:tinker://another-lora"
    # Players without LoRA specified will use the base model
    player_loras: str = ""  # Comma-separated list of player_id:lora_path pairs


async def run_single_player_with_logging(
    player_id: int,
    env,
    policy,
    tokenizer,
    raw_outputs: list,  # Shared list to collect raw outputs
):
    """Run a single player's rollout and collect raw outputs."""
    transitions = []
    moves = []

    # Get initial observation
    ob, stop_condition = await env.initial_observation()

    step_num = 0
    while True:
        # Sample action
        ac_with_logprobs = await policy(ob, stop_condition)

        # Decode and store raw output
        raw_output = tokenizer.decode(ac_with_logprobs.tokens)

        # Step environment
        step_result = await env.step(ac_with_logprobs.tokens)

        # Record move data
        move_data = {
            "player_id": player_id,
            "step": step_num,
            "raw_output": raw_output,
            "tokens": ac_with_logprobs.tokens,
            "reward": float(step_result.reward),
            "episode_done": step_result.episode_done,
            "metrics": step_result.metrics,
        }
        moves.append(move_data)

        # Store in shared list with player info
        raw_outputs.append({
            "player_id": player_id,
            "step": step_num,
            "raw_output": raw_output,
            "move_data": move_data,
        })

        # Build transition
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)

        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        step_num += 1

        if step_result.episode_done:
            break

    trajectory = Trajectory(transitions=transitions, final_ob=ob)
    return {
        "player_id": player_id,
        "moves": moves,
        "trajectory": trajectory,
    }


async def main():
    cli_config = chz.entrypoint(Config)

    print("=" * 80)
    print("FOUR PLAYER CHESS - SINGLE ROLLOUT TEST")
    print("=" * 80)
    print(f"Model: {cli_config.model_name}")
    print(f"Max tokens: {cli_config.max_tokens}")
    print(f"Temperature: {cli_config.temperature}")
    print(f"Output file: {cli_config.output_file}")
    print("=" * 80)
    print()

    # Parse player LoRA configuration
    player_lora_map: dict[int, str] = {}
    if cli_config.player_loras:
        print("\nPlayer LoRA Configuration:")
        print("-" * 80)
        for pair in cli_config.player_loras.split(","):
            pair = pair.strip()
            if not pair:
                continue
            try:
                player_id_str, lora_path = pair.split(":", 1)
                player_id = int(player_id_str.strip())
                lora_path = lora_path.strip()
                player_lora_map[player_id] = lora_path
                player_names = ["Red", "Blue", "Yellow", "Green"]
                player_colors = ["🔴", "🔵", "🟡", "🟢"]
                print(f"{player_colors[player_id]} Player {player_id} ({player_names[player_id]}): {lora_path}")
            except ValueError as e:
                raise ValueError(f"Invalid player_loras format: {pair}. Expected 'player_id:lora_path'") from e

        # Show which players use base model
        for player_id in range(4):
            if player_id not in player_lora_map:
                player_names = ["Red", "Blue", "Yellow", "Green"]
                player_colors = ["🔴", "🔵", "🟡", "🟢"]
                print(f"{player_colors[player_id]} Player {player_id} ({player_names[player_id]}): Base Model ({cli_config.model_name})")
        print("-" * 80)
        print()

    # Create service client
    service_client = tinker.ServiceClient()

    # Get renderer
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    print(f"Using renderer: {renderer_name}")

    tokenizer = get_tokenizer(cli_config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Create env group builder
    from tinker_cookbook.recipes.multiplayer_rl.four_player_chess.env import FourPlayerChessEnvGroupBuilder

    env_group_builder = FourPlayerChessEnvGroupBuilder(
        renderer=renderer,
        num_envs=4,
        self_play=True,
    )

    print("\nStarting rollout...")
    print("=" * 80)

    # Create envs
    envs_G = await env_group_builder.make_envs()

    # Create sampling clients and policies for each player
    policies = []
    for player_id in range(4):
        if player_id in player_lora_map:
            # Create sampling client with LoRA checkpoint
            model_path = player_lora_map[player_id]
            print(f"Creating LoRA sampling client for player {player_id}: {model_path}")
            sampling_client = service_client.create_sampling_client(
                model_path=model_path,
            )
        else:
            # Create base model sampling client
            print(f"Creating base model sampling client for player {player_id}")
            sampling_client = service_client.create_sampling_client(
                base_model=cli_config.model_name
            )

        # Create policy for this player
        policy = TinkerTokenCompleter(
            sampling_client=sampling_client,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
        )
        policies.append(policy)

    # Shared list to collect raw outputs from all players
    raw_outputs = []

    # Run all 4 players in parallel (like the training loop does)
    print("\nRunning 4-player game in parallel...")
    print("(Press Ctrl-C to save partial results and exit)")

    interrupted = False
    player_data_list = []
    tasks = []

    try:
        # Create tasks so we can cancel them on interrupt
        # Each player uses their own policy
        tasks = [
            asyncio.create_task(run_single_player_with_logging(player_id, env, policies[player_id], tokenizer, raw_outputs))
            for player_id, env in enumerate(envs_G)
        ]

        # Wait for all tasks to complete
        player_data_list = await asyncio.gather(*tasks)
        print(f"\n🏁 Game complete! Collected {len(raw_outputs)} total moves")

    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        print("\n\n⚠️  Interrupted by user! Saving partial results...")
        interrupted = True

        # Cancel all running tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait briefly for tasks to finish cancelling
        await asyncio.sleep(0.1)

        # Build partial player data from what we have so far
        player_data_list = []
        for player_id, env in enumerate(envs_G):
            # Get moves for this player from raw_outputs
            player_moves = [ro["move_data"] for ro in raw_outputs if ro["player_id"] == player_id]

            player_data_list.append({
                "player_id": player_id,
                "moves": player_moves,
                "trajectory": None,  # Not available for interrupted runs
            })

        print(f"Collected {len(raw_outputs)} moves before interruption")

    # Compute group rewards (skip if interrupted since we don't have complete trajectories)
    if not interrupted:
        trajectories = [pd["trajectory"] for pd in player_data_list]
        rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories, envs_G)
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

        # Add final rewards to player data
        for pd, final_reward, metadata in zip(player_data_list, rewards_G, metrics_G):
            pd["final_reward"] = float(final_reward)
            pd["metadata"] = metadata
    else:
        # For interrupted runs, set dummy values
        for pd in player_data_list:
            pd["final_reward"] = None
            pd["metadata"] = {"interrupted": True}

    # Build structured output
    player_names = ["Red", "Blue", "Yellow", "Green"]
    player_colors = ["🔴", "🔵", "🟡", "🟢"]

    # Get chronological move history from coordinator (if available)
    move_history = []
    if len(envs_G) > 0:
        coordinator = envs_G[0].coordinator
        move_history = coordinator.move_history

    structured_output = {
        "config": {
            "model_name": cli_config.model_name,
            "renderer_name": renderer_name,
            "max_tokens": cli_config.max_tokens,
            "temperature": cli_config.temperature,
            "timestamp": datetime.now().isoformat(),
            "player_loras": player_lora_map,
        },
        "interrupted": interrupted,
        "total_moves": len(raw_outputs),
        "move_history": move_history,  # Chronological order of all moves
        "players": []
    }

    # Add player data
    for pd in player_data_list:
        player_id = pd["player_id"]
        structured_output["players"].append({
            "player_id": player_id,
            "player_name": player_names[player_id],
            "lora_path": player_lora_map.get(player_id, None),
            "num_moves": len(pd["moves"]),
            "final_reward": pd["final_reward"],
            "metadata": pd["metadata"],
            "moves": pd["moves"],
        })

    # Save to JSON file
    output_path = Path(cli_config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured_output, f, indent=2)

    if interrupted:
        print(f"\n✅ Saved partial results to: {cli_config.output_file}")
    else:
        print(f"\n✅ Saved structured log to: {cli_config.output_file}")

    # Print human-readable summary
    print("\n" + "=" * 80)
    if interrupted:
        print("ROLLOUT INTERRUPTED (PARTIAL RESULTS)")
    else:
        print("ROLLOUT COMPLETE!")
    print("=" * 80)

    print("\nTRAJECTORY SUMMARY:")
    print("-" * 80)

    for pd in player_data_list:
        player_id = pd["player_id"]
        player_header = f"{player_colors[player_id]} {player_names[player_id]}"
        if player_id in player_lora_map:
            player_header += f" (LoRA: {player_lora_map[player_id]})"
        else:
            player_header += " (Base Model)"

        print(f"\n{player_header}:")
        print(f"  Moves: {len(pd['moves'])}")

        if pd['final_reward'] is not None:
            print(f"  Final reward: {pd['final_reward']:.2f}")
            print(f"  Metadata: {pd['metadata']}")

            # Sum up step rewards
            step_reward_sum = sum(m["reward"] for m in pd["moves"])
            total_reward = step_reward_sum + pd["final_reward"]
            print(f"  Step rewards sum: {step_reward_sum:.2f}")
            print(f"  Total reward: {total_reward:.2f}")
        else:
            print(f"  Final reward: N/A (interrupted)")

        # Count invalid move fallbacks
        invalid_moves = sum(1 for m in pd["moves"] if m["metrics"].get("invalid_move_fallback", 0) == 1)
        if invalid_moves > 0:
            print(f"  ⚠️  Invalid moves (random fallback): {invalid_moves}/{len(pd['moves'])}")

    # Detailed move log - reconstruct chronological order from coordinator
    # Since we ran in parallel, we need to look at the move history in the coordinator
    print("\n" + "=" * 80)
    print("DETAILED MOVE LOG (first 40 moves):")
    print("=" * 80)

    # Get move history from coordinator
    coordinator = envs_G[0].coordinator
    move_history = coordinator.move_history

    # Match move history with raw outputs
    # For each move in history, find the corresponding raw output
    for idx, move_info in enumerate(move_history[:40]):
        player_id = move_info["player_id"]
        player_marker = f"{player_colors[player_id]} {player_names[player_id]}"

        # Find the raw output for this move
        # The move_number in move_info corresponds to the global move counter
        # We need to find which step this was for the player
        player_moves = player_data_list[player_id]["moves"]

        # Find the move by matching the move_text or metrics
        matching_move = None
        for move_data in player_moves:
            if move_data["metrics"].get("move") == move_info.get("move_text"):
                matching_move = move_data
                break

        if matching_move:
            if matching_move['metrics'].get('invalid_move_fallback'):
                original = matching_move['metrics'].get('original_move', 'unknown')
                fallback = matching_move['metrics'].get('move', 'unknown')
                print(f"\n{idx:3d}. {player_marker:20s} ❌ INVALID → 🎲 RANDOM")
                print(f"     Original: {original[:80]}")
                print(f"     Fallback: {fallback}")
                print(f"     Reward: {matching_move['reward']:.2f}")
                print(f"     Raw output (first 200 chars):")
                print(f"     {matching_move['raw_output'][:200]}")
            elif matching_move['metrics'].get('no_legal_moves_fallback'):
                print(f"\n{idx:3d}. {player_marker:20s} ⛔ NO LEGAL MOVES")
                print(f"     Reward: {matching_move['reward']:.2f}")
            else:
                move_text = move_info.get('move_text', 'unknown')
                print(f"\n{idx:3d}. {player_marker:20s} ✓ {move_text}")
                print(f"     Reward: {matching_move['reward']:.2f}")
        else:
            # Fallback if we can't match
            print(f"\n{idx:3d}. {player_marker:20s} {move_info.get('move_text', 'unknown')}")

    if len(move_history) > 40:
        print(f"\n... ({len(move_history) - 40} more moves not shown)")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS:")
    print("=" * 80)

    # Collect all moves from all players
    all_moves = []
    for pd in player_data_list:
        all_moves.extend(pd["moves"])

    total_moves_count = len(all_moves)
    invalid_count = sum(1 for m in all_moves if m['metrics'].get('invalid_move_fallback'))
    valid_count = total_moves_count - invalid_count

    print(f"Total moves: {total_moves_count}")
    if total_moves_count > 0:
        print(f"Valid moves: {valid_count} ({100 * valid_count / total_moves_count:.1f}%)")
        print(f"Invalid moves (random fallback): {invalid_count} ({100 * invalid_count / total_moves_count:.1f}%)")
    else:
        print("No moves collected (interrupted immediately?)")

    # Winner (only for complete games)
    if not interrupted:
        print("\n" + "=" * 80)
        print("GAME OUTCOME:")
        print("=" * 80)

        rewards = [pd["final_reward"] for pd in player_data_list]
        winner_idx = rewards.index(max(rewards))

        winner_header = f"{player_colors[winner_idx]} {player_names[winner_idx]}"
        if winner_idx in player_lora_map:
            winner_header += f" (LoRA: {player_lora_map[winner_idx]})"
        else:
            winner_header += " (Base Model)"

        print(f"\n🏆 Winner: {winner_header}")
        print("\nFinal rewards (centered):")
        for i, reward in enumerate(rewards):
            indicator = " 👑" if i == winner_idx else ""
            model_type = f" [LoRA]" if i in player_lora_map else " [Base]"
            print(f"  {player_colors[i]} {player_names[i]:10s}: {reward:+.2f}{model_type}{indicator}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    if interrupted:
        print(f"\n📄 Partial results saved to: {cli_config.output_file}")
    else:
        print(f"\n📄 Full structured log saved to: {cli_config.output_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This catches interrupts that happen outside the async context
        # (e.g., during initialization before the game starts)
        print("\n\n⚠️  Interrupted during startup - no data to save.")
