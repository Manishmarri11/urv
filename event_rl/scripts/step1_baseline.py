# -*- coding: utf-8 -*-
"""Phase 0, step 1: plain PPO on CartPole, no custom encoder at all.

Sanity check that PPO itself trains correctly in this environment before any
custom code (event frames, STNet modules, EventEncoder) enters the picture.
If this doesn't show improvement, the problem is the base setup, not
anything built later in the pipeline.

Expected: episode reward climbs from ~20-30 (near-random policy) toward
~500 (CartPole-v1's max) over training.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)

    mean_before, std_before = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Before training: mean_reward={mean_before:.1f} +/- {std_before:.1f}")

    model.learn(total_timesteps=50_000)

    mean_after, std_after = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"After training:  mean_reward={mean_after:.1f} +/- {std_after:.1f}")

    assert mean_after > mean_before, (
        "PPO did not improve over the untrained baseline -- something is wrong "
        "with the base PPO/gymnasium setup, not with anything built later in "
        "the pipeline. Fix this before moving on to step2."
    )
    print("PASSED: PPO successfully learned CartPole in this environment.")


if __name__ == "__main__":
    main()
