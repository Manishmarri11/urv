# -*- coding: utf-8 -*-
"""The real training entry point -- wires EventEncoder into PPO and trains.

Structure follows URV2/event/main_event.py's pattern (argparse config,
wandb + tensorboard logging, checkpoint saving) but with PPO instead of
that script's SAC, and EventEncoder instead of its disabled EncoderCNN.

STATUS: both former `# TODO(env):` markers are resolved -- build_env() now
constructs a real, runnable environment via env/make_env.py: CartpoleWorld-v0
(a real MuJoCo cartpole) wrapped by EventsOnlyCartpoleWrapper, which strips
the ground-truth 4-vector observation entirely and replaces it with
event-derived frames, via either RGB-to-event converter --event_source
selects (default "fake": fake_events.rgb_to_events(), a placeholder
frame-differencing converter; "v2e": v2e_events.V2EEventGenerator, a real
DVS camera simulation -- both under env/rgb_to_event/).
"""
import os
import re
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from encoder import EventEncoder
from make_env import make_env
from callbacks import TemporalResetCallback, StatelessTemporalCallback, SafeVecNormalizeSaveCallback


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


def resolve_vecnormalize_path(checkpoint_path: str):
    """Finds the VecNormalize running-stats file saved alongside a model
    checkpoint (CheckpointCallback(save_vecnormalize=True) saves periodic
    checkpoints as '..._vecnormalize_<N>_steps.pkl'; the final save at the
    end of training writes '..._vecnormalize.pkl' with no step suffix).
    Returns None if no match exists -- resuming without it still works, it
    just restarts reward-normalization statistics from scratch instead of
    continuing them, which is a soft degradation, not a crash."""
    m = re.search(r"_(\d+)_steps\.zip$", checkpoint_path)
    if m:
        candidate = checkpoint_path[: m.start()] + f"_vecnormalize_{m.group(1)}_steps.pkl"
        if os.path.isfile(candidate):
            return candidate
    if checkpoint_path.endswith(".zip"):
        candidate = checkpoint_path[: -len(".zip")] + "_vecnormalize.pkl"
        if os.path.isfile(candidate):
            return candidate
    return None


def build_env(config):
    # CartpoleWorld-v0 itself is already registered with max_episode_steps=1000
    # (a hard ceiling); this outer TimeLimit applies config["max_episode_steps"]
    # as a shorter, user-controllable episode length on top of that -- useful
    # for fast iteration (shorter episodes = faster wall-clock feedback), not
    # a bug or an accidental double-wrap.
    env = make_env(
        env_id=config["env_id"],
        camera_name=config["camera_name"],
        obs_height=config["obs_height"],
        obs_width=config["obs_width"],
        n_bins=config["n_bins"],
        channels=config["channels"],
        event_source=config["event_source"],
    )
    env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])
    # Monitor must be wrapped explicitly here: SB3 only auto-adds it when you
    # hand PPO a raw (non-Vec) env. Since main() builds its own VecEnv/
    # VecNormalize stack below (needed for reward normalization), that
    # auto-wrap never happens -- without this, ep_len_mean/ep_rew_mean
    # logging would silently stop working, since those come from Monitor's
    # per-episode info dict, not from VecNormalize or the raw env.
    env = Monitor(env)
    return DummyVecEnv([lambda: env])


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
    parser.add_argument("--env_id", type=str, default="CartpoleWorld-v0",
                         help="Real MuJoCo env to wrap. 'CartpoleWorld-v0' (default, single-axis, camera "
                              "'side') or 'CartpoleWorldDualCamera-v0' (environment teammate's 2D/cliff-path "
                              "env, cameras 'world'/'pole'). --camera_name must match whichever env_id you "
                              "choose.")
    parser.add_argument("--camera_name", type=str, default="side",
                         help="Must be valid for --env_id: 'side' for CartpoleWorld-v0; 'pole' (true "
                              "egocentric feed, intended for the event pipeline) or 'world' (3rd-person "
                              "convenience view) for CartpoleWorldDualCamera-v0.")
    parser.add_argument("--event_source", type=str, default="fake", choices=["fake", "v2e"],
                         help="RGB-to-event converter. 'fake' (default): stateless two-frame diff placeholder "
                              "(rgb_to_event/fake_events.py), unchanged behavior from all earlier runs. 'v2e': a real, "
                              "stateful DVS camera simulation (env/rgb_to_event/v2e_events.py) -- real per-pixel threshold "
                              "mismatch, leak/shot noise, and sub-frame-resolved event timestamps instead of "
                              "randomly-assigned ones.")
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
    parser.add_argument("--stateless_temporal", action="store_true",
                         help="Ablation: reset TemporalBranch's state after EVERY rollout step instead of only "
                              "at episode boundaries, effectively disabling cross-step recurrence (each "
                              "observation still spans its own n_bins time-binned window internally). Compare "
                              "the reward curve against a normal run to see whether cross-step recurrence "
                              "actually matters for this task.")
    args = parser.parse_args()

    config = {
        "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "features_dim": args.features_dim,
        "max_episode_steps": args.max_episode_steps,
        "env_id": args.env_id,
        "camera_name": args.camera_name,
        "event_source": args.event_source,
        "obs_height": args.obs_height,
        "obs_width": args.obs_width,
        "n_bins": args.n_bins,
        "channels": args.channels,
        "device": args.device,
        "checkpoint_freq": args.checkpoint_freq,
        "stateless_temporal": args.stateless_temporal,
    }

    experiment_name = f"ppo_event_{int(time.time() * 1000)}"
    if args.stateless_temporal:
        experiment_name += "_stateless"
    # Printed on its own clearly-greppable line so a wrapping shell script
    # (e.g. a SLURM submit script that wants to auto-plot this run's curves
    # afterward) can extract it from the job log without needing it passed
    # in or predicted in advance -- experiment_name is timestamp-generated,
    # not knowable before this line runs.
    print(f"Experiment name: {experiment_name}")
    resuming = args.resume_model is not None
    if resuming:
        config["resumed_from"] = args.resume_model  # visible in the new wandb run's config

    run = wandb.init(
        project=args.wandb_project,
        config=config,
        name=experiment_name,
        sync_tensorboard=True,
    )

    env = build_env(config)  # raw DummyVecEnv, not yet reward-normalized

    # VecNormalize(norm_reward=True): rescales rewards to roughly unit
    # variance before the value function sees them. Diagnosed need: real
    # training runs (Test A/D) showed explained_variance stuck at ~0 the
    # entire run while value_loss climbed steadily (~7 -> ~70+) -- the
    # classic signature of a value target whose scale keeps drifting (return
    # scale grows with episode length, which itself grows as the policy
    # improves). norm_obs=False because observations are already normalized
    # to [0,1] by event_to_frame.py -- normalizing them again would distort
    # the actual event-count semantics the network was designed around.
    if resuming:
        checkpoint_path = resolve_checkpoint_path(args.resume_model)
        vecnorm_path = resolve_vecnormalize_path(checkpoint_path)
        if vecnorm_path:
            print(f"Resuming VecNormalize stats from: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
        else:
            print("No matching VecNormalize stats found for this checkpoint -- "
                  "starting reward normalization fresh instead of continuing it.")
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device=config["device"])
        print(f"  resumed at num_timesteps={model.num_timesteps}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
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
        # save_vecnormalize=True is intentionally NOT used here -- it has no
        # error handling around VecNormalize.save(), which can fail on the
        # cluster's live EGL render context and would crash the whole run.
        # SafeVecNormalizeSaveCallback below does the same save, defensively.
    )
    vecnormalize_callback = SafeVecNormalizeSaveCallback(
        save_freq=config["checkpoint_freq"],
        save_path="saved_models",
        name_prefix=experiment_name,
    )

    # StatelessTemporalCallback (--stateless_temporal): resets TemporalBranch's
    # state after EVERY rollout step, not just at episode ends -- the Test D
    # recurrence-ablation variant. TemporalResetCallback (default): resets only
    # at episode boundaries, matching the normal training behavior. Either way
    # detach_state() needs no callback -- EventEncoder.forward() already calls
    # it automatically every forward pass.
    reset_callback = StatelessTemporalCallback() if config["stateless_temporal"] else TemporalResetCallback()

    print(f"Learning (device={config['device']}, resuming={resuming}, stateless_temporal={config['stateless_temporal']})")
    model.learn(
        total_timesteps=config["total_timesteps"],
        # reset_num_timesteps=False when resuming so num_timesteps/tensorboard's
        # x-axis continue from the checkpoint instead of restarting at 0.
        reset_num_timesteps=not resuming,
        # CheckpointCallback: periodic saves so a run cut off by the cluster's
        # time limit (or pre-emption) has something to --resume_model from,
        # instead of only ever saving once at the very end.
        callback=[WandbCallback(), reset_callback, checkpoint_callback, vecnormalize_callback],
    )

    model.save(os.path.join("saved_models", experiment_name))
    try:
        env.save(os.path.join("saved_models", experiment_name + "_vecnormalize.pkl"))
    except Exception as e:
        # Same live-EGL-context pickling risk as SafeVecNormalizeSaveCallback --
        # training already finished and model.save() above already succeeded,
        # so this failing shouldn't take that down with it.
        print(f"WARNING: failed to save final VecNormalize stats: {type(e).__name__}: {e}")
    run.finish()


if __name__ == "__main__":
    main()
