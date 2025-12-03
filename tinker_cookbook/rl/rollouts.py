import asyncio
from typing import Sequence

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree


@logtree.scope_header_decorator
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    print(f"[DEBUG] do_single_rollout: Starting rollout for env", flush=True)
    transitions = []
    print(f"[DEBUG] do_single_rollout: Getting initial observation", flush=True)
    ob, stop_condition = await env.initial_observation()
    print(f"[DEBUG] do_single_rollout: Got initial observation, starting loop", flush=True)
    step_count = 0
    while True:
        print(f"[DEBUG] do_single_rollout: Step {step_count} - calling policy", flush=True)
        ac_with_logprobs = await policy(ob, stop_condition)
        print(f"[DEBUG] do_single_rollout: Step {step_count} - got action, calling env.step", flush=True)
        step_result = await env.step(ac_with_logprobs.tokens)
        print(f"[DEBUG] do_single_rollout: Step {step_count} - got step result, episode_done={step_result.episode_done}", flush=True)
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
        step_count += 1
        if step_result.episode_done:
            print(f"[DEBUG] do_single_rollout: Episode done after {step_count} steps", flush=True)
            break
    print(f"[DEBUG] do_single_rollout: Returning trajectory with {len(transitions)} transitions", flush=True)
    return Trajectory(transitions=transitions, final_ob=ob)


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    print(f"[DEBUG] do_group_rollout: Creating envs from builder", flush=True)
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    print(f"[DEBUG] do_group_rollout: Created {len(envs_G)} envs, starting rollouts in parallel", flush=True)
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    print(f"[DEBUG] do_group_rollout: All rollouts completed, computing group rewards", flush=True)
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    print(f"[DEBUG] do_group_rollout: Group rewards computed", flush=True)

    # Log trajectory tables with final rewards
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            rows = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                rows.append(
                    {
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                )
            # Add final row with final observation and computed reward
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                }
            )
            # Add total reward row
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
