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
    obs_height: int = 128,
    obs_width: int = 128,
    render_width: int = 480,
    render_height: int = 320,
    camera_name: str = "side",
    n_bins: int = 5,
    channels: int = 3,
    event_threshold: float = 0.015,
    max_count: float = 5.0,
    seed=None,
):
    """Returns a Gym env whose observation is (2*n_bins*channels, obs_height,
    obs_width) event-derived frames only -- the exact shape EventEncoder
    expects. Currently backed by CartpoleWorld-v0 + a fake RGB-to-event
    converter (see fake_events.py); swap the body of this function for the
    real environment/converter when ready."""
    return EventsOnlyCartpoleWrapper(
        render_width=render_width,
        render_height=render_height,
        camera_name=camera_name,
        obs_height=obs_height,
        obs_width=obs_width,
        n_bins=n_bins,
        channels=channels,
        event_threshold=event_threshold,
        max_count=max_count,
        seed=seed,
    )
