# -*- coding: utf-8 -*-
"""Loads a saved PPO checkpoint and renders real RGB video of the trained
agent acting in the real environment -- for visually confirming whether
later checkpoints keep the pole up longer than earlier ones, independent of
the numeric reward curve.

Renders the REAL scene (env.render(), true RGB pixels) for the video, not
the event-derived observation tensors the policy actually sees -- the video
is for human eyes only; the policy still acts purely on events-only
observations underneath, exactly as during training.

IMPORTANT -- TemporalBranch state at inference time: model.predict() alone
does NOT reset TemporalBranch's hidden state between episodes or steps (that
only happens via the training-time callbacks in callbacks.py, which aren't
running here). This script replicates that reset policy manually so the
evaluated behavior matches how the checkpoint was actually trained:
  - default: reset() once per episode (matches TemporalResetCallback)
  - --stateless_temporal: reset() every single step (matches
    StatelessTemporalCallback) -- pass this when evaluating a checkpoint
    that was itself trained with --stateless_temporal (Test D).
"""
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "encoder"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "env"))
sys.path.insert(0, _SCRIPT_DIR)

import imageio.v2 as imageio
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from make_env import make_env  # noqa: E402
from train import resolve_checkpoint_path  # noqa: E402  (same --resume_model path contract)


def main():
    parser = argparse.ArgumentParser(description="Render video of a trained checkpoint acting in the real env")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Full path, or a name under saved_models/, of the checkpoint to load "
                              "(same contract as train.py's --resume_model).")
    parser.add_argument("--env_id", type=str, default="CartpoleWorldDualCamera-v0",
                         help="Must match whatever env_id the checkpoint was actually trained with "
                              "(train.py's --env_id) -- a mismatch risks an action-space mismatch, not "
                              "just a visually wrong video.")
    parser.add_argument("--camera_name", type=str, default="world",
                         help="Must match --env_id and whatever camera_name the checkpoint was trained "
                              "with (train.py's --camera_name). NOTE: this default ('world', for an "
                              "easier-to-watch 3rd-person demo video) intentionally differs from train.py's "
                              "default ('pole') -- pass --camera_name pole explicitly when evaluating a "
                              "checkpoint trained with train.py's own default camera, or the video will "
                              "silently show/evaluate the wrong observation distribution.")
    parser.add_argument("--event_source", type=str, default="fake", choices=["fake", "v2e"],
                         help="Must match whatever event_source the checkpoint was actually trained with "
                              "(train.py's --event_source) -- a mismatch means the policy sees a "
                              "meaningfully different observation distribution than it was trained on.")
    parser.add_argument("--max_count", type=float, default=5.0,
                         help="Must match train.py's --max_count used for this checkpoint. See train.py's help "
                              "for the measured caveat on what this value actually does to the observation.")
    parser.add_argument("--event_threshold", type=float, default=0.015,
                         help="--event_source fake only. Must match train.py's --event_threshold.")
    parser.add_argument("--v2e_pos_thres", type=float, default=0.2,
                         help="--event_source v2e only. Must match train.py's --v2e_pos_thres -- v2e's own "
                              "default (0.2) is ~17x sparser than what --reward_shaping event_stillness's "
                              "default coefficient was calibrated against (0.05); a mismatch here silently "
                              "evaluates the policy on a different observation distribution than it trained on.")
    parser.add_argument("--v2e_neg_thres", type=float, default=0.2,
                         help="--event_source v2e only. Must match train.py's --v2e_neg_thres. See --v2e_pos_thres.")
    parser.add_argument("--v2e_device", type=str, default="cpu",
                         help="--event_source v2e only. Must match train.py's --v2e_device.")
    parser.add_argument("--reward_shaping", type=str, default="none", choices=["none", "event_stillness"],
                         help="Must match whatever reward_shaping the checkpoint was actually trained with "
                              "(train.py's --reward_shaping) -- this only affects the printed reward summary "
                              "below, not the policy's behavior, but a mismatch makes that summary meaningless "
                              "as a comparison against the training curves.")
    parser.add_argument("--event_shaping_coef", type=float, default=2.0,
                         help="--reward_shaping event_stillness only. Must match train.py's --event_shaping_coef.")
    parser.add_argument("--n_episodes", type=int, default=10, help="")
    parser.add_argument("--max_episode_steps", type=int, default=300,
                         help="must match the value used during training")
    parser.add_argument("--obs_height", type=int, default=128, help="must match the value used during training")
    parser.add_argument("--obs_width", type=int, default=128, help="must match the value used during training")
    parser.add_argument("--n_bins", type=int, default=5, help="must match the value used during training")
    parser.add_argument("--channels", type=int, default=3, help="must match the value used during training")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--fps", type=int, default=25, help="video playback fps (independent of sim step rate)")
    parser.add_argument("--out_dir", type=str, default="videos", help="")
    parser.add_argument("--stateless_temporal", action="store_true",
                         help="Reset TemporalBranch's state every step instead of once per episode -- pass this "
                              "when evaluating a checkpoint trained with train.py's --stateless_temporal (Test D). "
                              "Leave unset for a normally-trained (Test A) checkpoint.")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(args.out_dir, checkpoint_name)
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(env_id=args.env_id, camera_name=args.camera_name,
                    obs_height=args.obs_height, obs_width=args.obs_width,
                    n_bins=args.n_bins, channels=args.channels,
                    event_source=args.event_source,
                    max_count=args.max_count, event_threshold=args.event_threshold,
                    v2e_pos_thres=args.v2e_pos_thres, v2e_neg_thres=args.v2e_neg_thres,
                    v2e_device=args.v2e_device,
                    reward_shaping=args.reward_shaping, event_shaping_coef=args.event_shaping_coef)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    # No env= passed to load() -- inference-only (.predict()), never .learn()
    # here, so PPO doesn't need an attached env to reconstruct training state.
    model = PPO.load(checkpoint_path, device=args.device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"stateless_temporal={args.stateless_temporal}")
    print(f"Rendering {args.n_episodes} episodes to: {out_dir}")

    episode_summaries = []
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        model.policy.features_extractor.reset()  # episode-boundary reset, matches TemporalResetCallback
        frames = [env.render()]
        total_reward = 0.0
        total_shaping = 0.0  # only meaningful when --reward_shaping event_stillness
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            # deterministic=True: mean action, not a stochastic sample -- we
            # want to see the trained behavior itself, not exploration noise.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            total_reward += float(reward)
            total_shaping += info.get("reward_shaping", 0.0)
            steps += 1
            if args.stateless_temporal:
                model.policy.features_extractor.reset()  # matches StatelessTemporalCallback's every-step reset

        video_path = os.path.join(out_dir, f"episode_{ep:02d}.mp4")
        imageio.mimsave(video_path, frames, fps=args.fps)
        episode_summaries.append((steps, total_reward, total_shaping))
        shaping_note = f", shaping_sum={total_shaping:7.3f}" if args.reward_shaping != "none" else ""
        print(f"  episode {ep:2d}: {steps:4d} steps, reward={total_reward:7.2f}{shaping_note} -> {video_path}")

    env.close()

    ep_lens = [s for s, _, _ in episode_summaries]
    ep_rews = [r for _, r, _ in episode_summaries]
    print()
    print(f"Summary over {args.n_episodes} episodes:")
    print(f"  ep_len_mean = {sum(ep_lens) / len(ep_lens):.2f}")
    print(f"  ep_rew_mean = {sum(ep_rews) / len(ep_rews):.2f}")
    if args.reward_shaping != "none":
        ep_shaping = [sh for _, _, sh in episode_summaries]
        print(f"  shaping_sum_mean = {sum(ep_shaping) / len(ep_shaping):.3f}  "
              f"(negative = motion penalized; compare against ep_rew_mean's +1/step ceiling)")


if __name__ == "__main__":
    main()
