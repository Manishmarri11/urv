# -*- coding: utf-8 -*-
"""The real training entry point -- wires EventEncoder into PPO and trains.

Structure follows URV2/event/main_event.py's pattern (argparse config,
wandb + tensorboard logging, checkpoint saving) but with PPO instead of
that script's SAC, and EventEncoder instead of its disabled EncoderCNN.

STATUS: both former `# TODO(env):` markers are resolved -- build_env() now
constructs a real, runnable environment via env/make_env.py: CartpoleWorld-v0
(a real MuJoCo cartpole) wrapped by EventsOnlyCartpoleWrapper, which strips
the ground-truth 4-vector observation entirely and replaces it with
event-derived frames (currently via fake_events.rgb_to_events(), a
placeholder frame-differencing converter -- swap that one function for the
real RGB-to-event converter when it's ready; nothing here needs to change).
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "encoder"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
sys.path.insert(0, os.path.dirname(__file__))

import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from encoder import EventEncoder
from make_env import make_env
from callbacks import TemporalResetCallback


def resolve_checkpoint_path(resume_model: str) -> str:
    """Matches URV2/event/main_tracking_path.py's --resume_model contract:
    accepts either a full path, or a bare name looked up under saved_models/
    (with or without the .zip SB3 appends). Raises a clear error rather than
    letting PPO.load() fail on a not-obviously-related FileNotFoundError."""
    candidates = [
        resume_model,
        resume_model + ".zip",
        os.path.join("saved_models", resume_model),
        os.path.join("saved_models", resume_model + ".zip"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"--resume_model {resume_model!r} not found. Tried: {candidates}. "
        f"Pass either a full path, or a name under saved_models/ (with or without .zip)."
    )


def build_env(config):
    # CartpoleWorld-v0 itself is already registered with max_episode_steps=1000
    # (a hard ceiling); this outer TimeLimit applies config["max_episode_steps"]
    # as a shorter, user-controllable episode length on top of that -- useful
    # for fast iteration (shorter episodes = faster wall-clock feedback), not
    # a bug or an accidental double-wrap.
    env = make_env(
        obs_height=config["obs_height"],
        obs_width=config["obs_width"],
        n_bins=config["n_bins"],
        channels=config["channels"],
    )
    env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])
    return env


def main():
    parser = argparse.ArgumentParser(description="Train PPO with EventEncoder")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="")
    parser.add_argument("--n_steps", type=int, default=2048, help="rollout length per PPO update")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO epochs per rollout")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--features_dim", type=int, default=256, help="EventEncoder output vector size")
    parser.add_argument("--max_episode_steps", type=int, default=300, help="")
    parser.add_argument("--wandb_project", type=str, default="event_rl", help="")
    parser.add_argument("--obs_height", type=int, default=128, help="downscaled render height fed to the event pipeline")
    parser.add_argument("--obs_width", type=int, default=128, help="downscaled render width fed to the event pipeline")
    parser.add_argument("--n_bins", type=int, default=5, help="must match EventEncoder's n_bins (default 5)")
    parser.add_argument("--channels", type=int, default=3, help="must match EventEncoder's channels (default 3)")
    parser.add_argument("--device", type=str, default="cuda",
                         help="SB3/torch device, e.g. 'cuda', 'cpu', 'cuda:0'. Default 'cuda' for real runs on a "
                              "GPU cluster; pass --device cpu for local smoke testing on a CPU-only machine.")
    parser.add_argument("--resume_model", type=str, default=None,
                         help="Full path, or a name under saved_models/, of a checkpoint to resume training from. "
                              "Matches URV2/event/main_tracking_path.py's --resume_model contract. "
                              "total_timesteps continues counting from where the checkpoint left off, not from 0.")
    parser.add_argument("--checkpoint_freq", type=int, default=10_000,
                         help="Save a resumable checkpoint every N timesteps (in addition to the final save). "
                              "Matters for real runs that might exceed the cluster's time limit or get pre-empted "
                              "-- without this, --resume_model has nothing to resume from if a run gets cut off.")
    args = parser.parse_args()

    config = {
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "features_dim": args.features_dim,
        "max_episode_steps": args.max_episode_steps,
        "obs_height": args.obs_height,
        "obs_width": args.obs_width,
        "n_bins": args.n_bins,
        "channels": args.channels,
        "device": args.device,
        "checkpoint_freq": args.checkpoint_freq,
    }

    experiment_name = f"ppo_event_{int(time.time() * 1000)}"
    resuming = args.resume_model is not None
    if resuming:
        config["resumed_from"] = args.resume_model  # visible in the new wandb run's config

    run = wandb.init(
        project=args.wandb_project,
        config=config,
        name=experiment_name,
        sync_tensorboard=True,
    )

    env = build_env(config)

    if resuming:
        checkpoint_path = resolve_checkpoint_path(args.resume_model)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device=config["device"])
        print(f"  resumed at num_timesteps={model.num_timesteps}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            learning_rate=config["learning_rate"],
            policy_kwargs=dict(
                features_extractor_class=EventEncoder,
                features_extractor_kwargs=dict(
                    features_dim=config["features_dim"],
                    n_bins=config["n_bins"],
                    channels=config["channels"],
                ),
            ),
            device=config["device"],
            verbose=1,
            tensorboard_log=f"./event_rl_tensorboard/{experiment_name}",
        )

    os.makedirs("saved_models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path="saved_models",
        name_prefix=experiment_name,
    )

    print(f"Learning (device={config['device']}, resuming={resuming})")
    model.learn(
        total_timesteps=config["total_timesteps"],
        # reset_num_timesteps=False when resuming so num_timesteps/tensorboard's
        # x-axis continue from the checkpoint instead of restarting at 0.
        reset_num_timesteps=not resuming,
        # TemporalResetCallback: resets TemporalBranch's SNN state whenever an
        # episode ends during rollout collection (see callbacks.py for why this
        # has to be a callback, not something inside the env). detach_state()
        # needs no callback -- EventEncoder.forward() already calls it
        # automatically every forward pass.
        # CheckpointCallback: periodic saves so a run cut off by the cluster's
        # time limit (or pre-emption) has something to --resume_model from,
        # instead of only ever saving once at the very end.
        callback=[WandbCallback(), TemporalResetCallback(), checkpoint_callback],
    )

    model.save(os.path.join("saved_models", experiment_name))
    run.finish()


if __name__ == "__main__":
    main()
