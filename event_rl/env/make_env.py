# -*- coding: utf-8 -*-
"""Environment factory for train.py. Swap this function's body for the real
event environment / real RGB-to-event converter later -- callers (train.py,
the step scripts) shouldn't need to change, only this function's internals.
"""
import os
import sys

_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENV_DIR not in sys.path:
    sys.path.insert(0, _ENV_DIR)

from event_wrapper import EventsOnlyCartpoleWrapper  # noqa: E402


def make_env(
    env_id: str = "CartpoleWorldDualCamera-v0",
    obs_height: int = 128,
    obs_width: int = 128,
    render_width: int = 480,
    render_height: int = 320,
    camera_name: str = "pole",
    n_bins: int = 5,
    channels: int = 3,
    event_threshold: float = 0.015,
    max_count: float = 5.0,
    event_source: str = "fake",
    v2e_pos_thres: float = 0.2,
    v2e_neg_thres: float = 0.2,
    v2e_device: str = "cpu",
    reward_shaping: str = "none",
    event_shaping_coef: float = 2.0,
    seed=None,
):
    """Returns a Gym env whose observation is (2*n_bins*channels, obs_height,
    obs_width) event-derived frames only -- the exact shape EventEncoder
    expects.

    env_id/camera_name must be a matching pair:
      - "CartpoleWorldDualCamera-v0" (default) -- 2D/cliff-path environment,
        the environment teammate's intended final version. "pole" (default)
        is the true egocentric feed intended for the event pipeline; "world"
        is a 3rd-person convenience view for demo renders only.
      - "CartpoleWorld-v0" only has camera "side" -- the original
        single-camera, single-axis environment, superseded by the above but
        kept for anyone still referencing it explicitly.

    event_source picks the RGB-to-event converter: "fake" (default,
    stateless two-frame diff placeholder, see rgb_to_event/fake_events.py)
    or "v2e" (a real, stateful DVS camera simulation, see
    rgb_to_event/v2e_events.py).

    reward_shaping="event_stillness" (default "none") adds an event-derived
    motion penalty on top of the real env's ground-truth survival reward --
    see EventsOnlyCartpoleWrapper.step()/event_shaping_coef's docstring
    comments in event_wrapper.py for the exact formula and calibration.
    event_shaping_coef is calibrated against --v2e_pos_thres 0.05
    --v2e_neg_thres 0.05, NOT v2e's own sparser defaults -- pass both
    together, or re-derive the coefficient for whatever thresholds you use.
    """
    return EventsOnlyCartpoleWrapper(
        env_id=env_id,
        render_width=render_width,
        render_height=render_height,
        camera_name=camera_name,
        obs_height=obs_height,
        obs_width=obs_width,
        n_bins=n_bins,
        channels=channels,
        event_threshold=event_threshold,
        max_count=max_count,
        event_source=event_source,
        v2e_pos_thres=v2e_pos_thres,
        v2e_neg_thres=v2e_neg_thres,
        v2e_device=v2e_device,
        reward_shaping=reward_shaping,
        event_shaping_coef=event_shaping_coef,
        seed=seed,
    )
