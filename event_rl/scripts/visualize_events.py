# -*- coding: utf-8 -*-
"""Runs the wrapped events-only CartpoleWorld env for ~30 steps and saves the
resulting event frames as images, so you can eyeball whether the fake
converter produces something that looks like a moving pole before trusting
a training run on top of it.

For each step, the 5 time-bins are collapsed (max across bins) into one
positive map and one negative map, then rendered false-color: blue=positive,
red=negative -- the same convention URV2/event/eventCamera.py's own
visualize() uses. Also saves the true RGB render alongside each one, and one
combined grid image for a quick overview without opening 30 files.
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "env"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "encoder"))

import numpy as np
from PIL import Image

from make_env import make_env  # noqa: E402

N_STEPS = 30
OUT_DIR = os.path.join(_SCRIPT_DIR, "..", "event_viz_output")


def obs_to_false_color(obs: np.ndarray, n_bins: int, channels: int) -> np.ndarray:
    """(2*n_bins*channels, H, W) -> (H, W, 3) uint8, blue=positive, red=negative."""
    pos = obs[: n_bins * channels].reshape(n_bins, channels, obs.shape[1], obs.shape[2])
    neg = obs[n_bins * channels:].reshape(n_bins, channels, obs.shape[1], obs.shape[2])
    pos_map = pos[:, 0, :, :].max(axis=0)  # (H, W), channel 0 (all 3 channels identical by construction)
    neg_map = neg[:, 0, :, :].max(axis=0)
    img = np.zeros((obs.shape[1], obs.shape[2], 3), dtype=np.uint8)
    img[:, :, 2] = (pos_map * 255).astype(np.uint8)  # blue = positive
    img[:, :, 0] = (neg_map * 255).astype(np.uint8)  # red = negative
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = make_env(obs_height=128, obs_width=128)

    obs, info = env.reset(seed=0)
    event_imgs = []
    rgb_imgs = []

    nonzero_counts = []
    for i in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        event_img = obs_to_false_color(obs, env.n_bins, env.channels)
        rgb_img = env.render()  # native-resolution true RGB, for comparison

        Image.fromarray(event_img).save(os.path.join(OUT_DIR, f"step{i:02d}_events.png"))
        Image.fromarray(rgb_img).save(os.path.join(OUT_DIR, f"step{i:02d}_rgb.png"))

        event_imgs.append(event_img)
        rgb_imgs.append(rgb_img)
        nonzero_counts.append(int((obs > 0).sum()))

        if terminated or truncated:
            print(f"  step {i}: episode ended (terminated={terminated}, truncated={truncated}), resetting")
            obs, info = env.reset()

    env.close()

    # combined grid: 6 columns x 5 rows of the event-frame visualizations
    cols, rows = 6, 5
    h, w = event_imgs[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(event_imgs):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    grid_path = os.path.join(OUT_DIR, "grid_all_steps.png")
    Image.fromarray(grid).save(grid_path)

    print(f"\nSaved {N_STEPS} per-step event/rgb image pairs and one grid overview to:\n  {OUT_DIR}")
    print(f"Grid overview: {grid_path}")
    print(f"\nNonzero pixel-count per step's observation (30,H,W) -- 0 means NO events fired that step:")
    print(f"  {nonzero_counts}")
    print(f"  steps with zero events: {sum(1 for c in nonzero_counts if c == 0)} / {N_STEPS}")


if __name__ == "__main__":
    main()
