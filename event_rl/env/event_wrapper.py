# -*- coding: utf-8 -*-
"""Events-only Gym wrapper around CartpoleWorld-v0.

Wraps (never modifies) the real cartpole_world_env.CartpoleWorldEnv. Renders
each step, converts consecutive frames into a synthetic event stream via
fake_events.rgb_to_events() (the single swap point for the real RGB-to-event
converter), and feeds those events through the REAL event_to_frame.py into
the exact (2*n_bins*channels, H, W) shape EventEncoder expects.

RESEARCH CONSTRAINT ENFORCED HERE: CartpoleWorldEnv's real observation (the
4-vector [cart_pos, pole_angle, cart_vel, pole_angvel]) is read internally
(MujocoEnv.step()/reset() require it) but never returned to the agent -- this
wrapper's own observation_space and the values it returns are built entirely
from event-derived frames. Reward, terminated, and truncated are passed
through from the real env unchanged (those aren't part of the events-only
constraint -- only the observation is).
"""
import os
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENV_DIR not in sys.path:
    sys.path.insert(0, _ENV_DIR)
_ENCODER_DIR = os.path.join(_ENV_DIR, "..", "encoder")
if _ENCODER_DIR not in sys.path:
    sys.path.insert(0, _ENCODER_DIR)

import cartpole_world_env  # noqa: E402,F401  (registers "CartpoleWorld-v0", read-only import)
from fake_events import rgb_to_events  # noqa: E402
from event_to_frame import events_to_frames  # noqa: E402  (real module, read-only import)


class EventsOnlyCartpoleWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_width: int = 480,
        render_height: int = 320,
        camera_name: str = "side",
        obs_height: int = 128,
        obs_width: int = 128,
        n_bins: int = 5,
        channels: int = 3,
        event_threshold: float = 0.015,
        max_count: float = 5.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._cartpole = gym.make(
            "CartpoleWorld-v0",
            render_mode="rgb_array",
            width=render_width,
            height=render_height,
            camera_name=camera_name,
        )
        # frame_skip * XML timestep, read from the real env rather than
        # assumed/hardcoded -- confirmed empirically to be 0.04s (40ms) with
        # CartpoleWorldEnv's defaults (frame_skip=2, XML timestep=0.02).
        self._step_dt = float(self._cartpole.unwrapped.dt)

        self.obs_h, self.obs_w = obs_height, obs_width
        self.n_bins, self.channels = n_bins, channels
        self.event_threshold = event_threshold
        self.max_count = max_count
        self._rng = np.random.default_rng(seed)

        obs_channels = 2 * n_bins * channels
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_channels, obs_height, obs_width), dtype=np.float32
        )
        # Real, unmodified action space from CartpoleWorldEnv (Box(-3,3,(1,)) --
        # confirmed by reading the XML's slide_motor ctrlrange) -- action is NOT
        # part of the events-only constraint, only the observation is.
        self.action_space = self._cartpole.action_space

        self._prev_frame_small = None  # (H, W, 3) uint8, downscaled

    def _render_downscaled(self) -> np.ndarray:
        frame = self._cartpole.render()  # (render_height, render_width, 3) uint8
        img = Image.fromarray(frame).resize((self.obs_w, self.obs_h), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)  # (obs_h, obs_w, 3)

    def _events_to_obs(self, prev_small: np.ndarray, curr_small: np.ndarray) -> np.ndarray:
        x, y, polarity, t = rgb_to_events(
            prev_small, curr_small, t_start=0.0, t_end=self._step_dt,
            threshold=self.event_threshold, rng=self._rng,
        )
        pos_frames, neg_frames = events_to_frames(
            x, y, polarity, t, height=self.obs_h, width=self.obs_w,
            t_start=0.0, t_end=self._step_dt, n_bins=self.n_bins,
            channels=self.channels, max_count=self.max_count,
        )
        # pos_frames then neg_frames, each n_bins tensors (channels,H,W) --
        # matches EventEncoder._split()'s expected channel ordering exactly
        # (confirmed by reading encoder.py, not assumed).
        frames = [f.numpy() for f in pos_frames] + [f.numpy() for f in neg_frames]
        return np.concatenate(frames, axis=0).astype(np.float32)  # (2*n_bins*channels, H, W)

    def reset(self, *, seed=None, options=None):
        _real_obs, info = self._cartpole.reset(seed=seed)  # real_obs (4-vec) read but never exposed
        curr_small = self._render_downscaled()
        self._prev_frame_small = curr_small
        # First-frame case: no real previous frame exists yet. Rather than a
        # separate special-cased "return zeros" path, call the same event
        # pipeline with prev==curr -- the diff is naturally exactly zero
        # everywhere, so events_to_frames() naturally returns all-zero
        # frames through the ordinary code path (one fewer code path to get
        # wrong, per the same event math either way).
        obs = self._events_to_obs(curr_small, curr_small)
        return obs, info

    def step(self, action):
        _real_obs, reward, terminated, truncated, info = self._cartpole.step(action)
        curr_small = self._render_downscaled()
        obs = self._events_to_obs(self._prev_frame_small, curr_small)
        self._prev_frame_small = curr_small
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._cartpole.render()

    def close(self):
        self._cartpole.close()
