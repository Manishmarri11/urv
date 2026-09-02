# -*- coding: utf-8 -*-
"""Events-only Gym wrapper around either real cartpole environment.

Wraps (never modifies) the real cartpole_world_env.CartpoleWorldEnv or
cartpole_world_env_dual_camera.CartpoleWorldDualCameraEnv (your environment
teammate's second, in-progress env -- 2D tilt/cart-correction, cliff-path
terrain, "world"/"pole" cameras instead of the original single "side" one).
Renders each step and converts consecutive frames into an event stream via
one of two interchangeable converters, both under rgb_to_event/
(event_source=):
  - "fake" (default): rgb_to_event/fake_events.py's rgb_to_events(), a
    stateless two-frame diff-and-threshold placeholder with
    randomly-assigned timestamps.
  - "v2e": rgb_to_event/v2e_events.py's V2EEventGenerator, backed by a real,
    stateful DVS camera simulation (a vendored extraction of SensorsINI/v2e's
    EventEmulator) -- real per-pixel threshold mismatch, leak/shot noise,
    and sub-frame-resolved event timestamps instead of random ones. See
    that module's docstring for why this needs a running simulation clock
    instead of a per-step two-frame diff.
Either way, those events get fed through the REAL event_to_frame.py into the
exact (2*n_bins*channels, H, W) shape EventEncoder expects.

RESEARCH CONSTRAINT ENFORCED HERE: the real environment's ground-truth
observation (4 elements for CartpoleWorld-v0, 8 for CartpoleWorldDualCamera-v0)
is read internally (MujocoEnv.step()/reset() require it) but never returned
to the agent -- this wrapper's own observation_space and the values it
returns are built entirely from event-derived frames, regardless of which
real env or how many ground-truth dimensions it has. Reward, terminated, and
truncated are passed through from the real env unchanged (those aren't part
of the events-only constraint -- only the observation is).
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
_RGB_TO_EVENT_DIR = os.path.join(_ENV_DIR, "rgb_to_event")
if _RGB_TO_EVENT_DIR not in sys.path:
    sys.path.insert(0, _RGB_TO_EVENT_DIR)

import cartpole_world_env  # noqa: E402,F401  (registers "CartpoleWorld-v0", read-only import)
import cartpole_world_env_dual_camera  # noqa: E402,F401  (registers "CartpoleWorldDualCamera-v0", read-only import)
from fake_events import rgb_to_events  # noqa: E402
from v2e_events import V2EEventGenerator  # noqa: E402
from event_to_frame import events_to_frames  # noqa: E402  (real module, read-only import)


class EventsOnlyCartpoleWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str = "CartpoleWorld-v0",
        render_width: int = 480,
        render_height: int = 320,
        camera_name: str = "side",
        obs_height: int = 128,
        obs_width: int = 128,
        n_bins: int = 5,
        channels: int = 3,
        event_threshold: float = 0.015,
        max_count: float = 5.0,
        event_source: str = "fake",
        seed: Optional[int] = None,
    ):
        super().__init__()
        if event_source not in ("fake", "v2e"):
            raise ValueError(f"event_source must be 'fake' or 'v2e', got {event_source!r}")
        self.event_source = event_source
        # env_id/camera_name must match: "CartpoleWorld-v0" only has camera
        # "side" (the old default); "CartpoleWorldDualCamera-v0" only has
        # "world" (3rd-person convenience view) and "pole" (the true
        # egocentric feed your environment teammate intends for the event
        # pipeline -- see their own comment in render_cartpole_world.py).
        # Passing "side" with the dual-camera env (or vice versa) fails at
        # gym.make() with an unknown-camera error, not a silent misconfig.
        self._cartpole = gym.make(
            env_id,
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
        # Real, unmodified action space -- Box(-3,3,(1,)) for CartpoleWorld-v0,
        # Box(-3,3,(2,)) for CartpoleWorldDualCamera-v0 (confirmed by reading
        # each XML's motor ctrlrange). Read dynamically from whichever env_id
        # was requested rather than hardcoded, so this line needs no edit when
        # switching between them -- action is NOT part of the events-only
        # constraint, only the observation is.
        self.action_space = self._cartpole.action_space

        self._prev_frame_small = None  # (H, W, 3) uint8, downscaled -- event_source="fake" only
        self._v2e_gen = None  # event_source="v2e" only
        self._sim_time = 0.0  # running clock EventEmulator needs -- event_source="v2e" only
        if self.event_source == "v2e":
            self._v2e_gen = V2EEventGenerator(seed=seed if seed is not None else 0)

    def _render_downscaled(self) -> np.ndarray:
        frame = self._cartpole.render()  # (render_height, render_width, 3) uint8
        img = Image.fromarray(frame).resize((self.obs_w, self.obs_h), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)  # (obs_h, obs_w, 3)

    def _frames_from_events(self, x, y, polarity, t, t_start: float, t_end: float) -> np.ndarray:
        pos_frames, neg_frames = events_to_frames(
            x, y, polarity, t, height=self.obs_h, width=self.obs_w,
            t_start=t_start, t_end=t_end, n_bins=self.n_bins,
            channels=self.channels, max_count=self.max_count,
        )
        # pos_frames then neg_frames, each n_bins tensors (channels,H,W) --
        # matches EventEncoder._split()'s expected channel ordering exactly
        # (confirmed by reading encoder.py, not assumed).
        frames = [f.numpy() for f in pos_frames] + [f.numpy() for f in neg_frames]
        return np.concatenate(frames, axis=0).astype(np.float32)  # (2*n_bins*channels, H, W)

    def _events_to_obs_fake(self, prev_small: np.ndarray, curr_small: np.ndarray) -> np.ndarray:
        x, y, polarity, t = rgb_to_events(
            prev_small, curr_small, t_start=0.0, t_end=self._step_dt,
            threshold=self.event_threshold, rng=self._rng,
        )
        return self._frames_from_events(x, y, polarity, t, t_start=0.0, t_end=self._step_dt)

    def _events_to_obs_v2e(self, curr_small: np.ndarray) -> np.ndarray:
        # V2EEventGenerator is stateful/streaming (see its module docstring)
        # -- it needs one call per frame at a strictly increasing timestamp,
        # not a two-frame diff, so this wrapper owns a running sim clock
        # instead of a stored previous frame.
        t_start = self._sim_time
        self._sim_time += self._step_dt
        x, y, polarity, t = self._v2e_gen.generate(curr_small, self._sim_time)
        return self._frames_from_events(x, y, polarity, t, t_start=t_start, t_end=self._sim_time)

    def reset(self, *, seed=None, options=None):
        _real_obs, info = self._cartpole.reset(seed=seed)  # real_obs (4-vec) read but never exposed
        curr_small = self._render_downscaled()
        if self.event_source == "v2e":
            self._v2e_gen.reset()
            self._sim_time = 0.0
            # First call to a freshly (re)constructed EventEmulator always
            # returns no events by its own design (it just initializes
            # internal photoreceptor/threshold state) -- gives the same
            # all-zero first observation as the "fake" path's prev==curr
            # trick, for free, through the ordinary code path.
            obs = self._events_to_obs_v2e(curr_small)
        else:
            self._prev_frame_small = curr_small
            # First-frame case: no real previous frame exists yet. Rather
            # than a separate special-cased "return zeros" path, call the
            # same event pipeline with prev==curr -- the diff is naturally
            # exactly zero everywhere, so events_to_frames() naturally
            # returns all-zero frames through the ordinary code path.
            obs = self._events_to_obs_fake(curr_small, curr_small)
        return obs, info

    def step(self, action):
        _real_obs, reward, terminated, truncated, info = self._cartpole.step(action)
        curr_small = self._render_downscaled()
        if self.event_source == "v2e":
            obs = self._events_to_obs_v2e(curr_small)
        else:
            obs = self._events_to_obs_fake(self._prev_frame_small, curr_small)
            self._prev_frame_small = curr_small
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._cartpole.render()

    def close(self):
        self._cartpole.close()
