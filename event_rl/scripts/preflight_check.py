# -*- coding: utf-8 -*-
"""Run this FIRST on the cluster, before submitting a real training job.
Checks everything that would otherwise fail deep into a run and waste
GPU-hours: package availability, actual CUDA visibility (not just installed,
actually usable), and a real end-to-end render through the real
CartpoleWorld-v0 env (catches headless-rendering / MUJOCO_GL problems
immediately instead of after the job queues and starts).

Exits 0 and prints "ALL CHECKS PASSED" only if everything needed for a real
training run actually works. Exits 1 with a specific, actionable message
otherwise -- never silently continues past a real problem.
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "env"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "encoder"))

_parser = argparse.ArgumentParser(description="Pre-flight checks before a real training job")
_parser.add_argument("--env_id", type=str, default="CartpoleWorld-v0",
                      help="Must match whatever env_id the real training job will use -- checking the "
                           "wrong one gives false confidence instead of actually validating what's about "
                           "to run (e.g. CartpoleWorldDualCamera-v0 needs env/assets/ present, which "
                           "checking CartpoleWorld-v0 would never catch).")
_parser.add_argument("--camera_name", type=str, default="side",
                      help="Must match --env_id (e.g. 'pole' or 'world' for CartpoleWorldDualCamera-v0).")
_parser.add_argument("--event_source", type=str, default="fake", choices=["fake", "v2e"],
                      help="Must match whatever event_source the real training job will use -- 'v2e' "
                           "exercises the real DVS simulation (env/rgb_to_event/v2e_events.py) instead of the fake "
                           "placeholder, so a training-job-shaped preflight run should pass this.")
_args = _parser.parse_args()

failures = []


def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}" + (f" -- {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {name} -- {type(e).__name__}: {e}")
        failures.append(name)
        return False


print("=" * 70)
print("1. PACKAGE IMPORTS")
print("=" * 70)


def _torch():
    import torch
    return f"version {torch.__version__}"


def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False -- either this node has no GPU "
            "allocated (check #SBATCH -G / -p partition), or torch was installed "
            "without CUDA support in the 'event' conda env."
        )
    name = torch.cuda.get_device_name(0)
    return f"{name}, {torch.cuda.device_count()} device(s), CUDA {torch.version.cuda}"


def _mujoco():
    import mujoco
    return f"version {mujoco.__version__}"


def _gymnasium():
    import gymnasium
    return f"version {gymnasium.__version__}"


def _sb3():
    import stable_baselines3
    return f"version {stable_baselines3.__version__}"


def _wandb():
    import wandb
    return f"version {wandb.__version__}"


def _pillow():
    import PIL
    return f"version {PIL.__version__}"


check("torch importable", _torch)
check("torch.cuda actually available (not just torch installed)", _cuda)
check("mujoco importable", _mujoco)
check("gymnasium importable", _gymnasium)
check("stable_baselines3 importable", _sb3)
check("wandb importable", _wandb)
check("PIL/Pillow importable", _pillow)

print()
print("=" * 70)
print("2. HEADLESS RENDERING (the classic cluster failure point)")
print("=" * 70)
print(f"  MUJOCO_GL env var: {os.environ.get('MUJOCO_GL', '(not set)')}")


def _render():
    import cartpole_world_env  # noqa: F401  (registers CartpoleWorld-v0)
    import cartpole_world_env_dual_camera  # noqa: F401  (registers CartpoleWorldDualCamera-v0)
    import gymnasium as gym
    env = gym.make(_args.env_id, render_mode="rgb_array", width=480, height=320, camera_name=_args.camera_name)
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame is not None and frame.shape == (320, 480, 3), f"unexpected frame shape: {getattr(frame, 'shape', None)}"
    return f"got a real {frame.shape} frame"


check(f"real render through {_args.env_id} (needs MUJOCO_GL set correctly on a headless node)", _render)

print()
print("=" * 70)
print("3. FULL PIPELINE, ONE STEP (events -> EventEncoder -> PPO policy, on GPU if available)")
print("=" * 70)


def _full_pipeline():
    import torch
    from make_env import make_env
    from encoder import EventEncoder
    from stable_baselines3 import PPO

    env = make_env(env_id=_args.env_id, camera_name=_args.camera_name, obs_height=128, obs_width=128,
                    event_source=_args.event_source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "CnnPolicy", env,
        policy_kwargs=dict(features_extractor_class=EventEncoder, features_extractor_kwargs=dict(features_dim=256)),
        n_steps=8, batch_size=8, n_epochs=1, device=device, verbose=0,
    )
    model.learn(total_timesteps=8)
    env.close()
    return f"one PPO update completed on device={device}"


check("one real PPO update end-to-end", _full_pipeline)

print()
print("=" * 70)
if failures:
    print(f"PREFLIGHT FAILED: {len(failures)} check(s) failed: {failures}")
    print("Fix these before submitting a real training job -- do not proceed.")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED -- safe to submit the real training job.")
    sys.exit(0)
