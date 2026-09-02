# -*- coding: utf-8 -*-
"""Real RGB -> event-stream conversion, backed by v2e_emulator.EventEmulator
(a vendored, framework-free extraction of v2e/SensorsINI's DVS simulator --
see v2e_emulator.py's module docstring for exactly what was stripped and
why). This is the real implementation of the RGB-to-event swap point
fake_events.rgb_to_events() stood in for -- see ../../README.md for the
exact downstream contract (event_to_frame.events_to_frames()'s expected
input).

fake_events.rgb_to_events() is STATELESS: called with two frames per step,
it diffs them and assigns each resulting event a uniformly random timestamp
within the step window -- an explicitly flagged simplification (see its own
docstring), since a single frame diff carries no real information about
*when* within the window a pixel actually changed.

v2e's EventEmulator is STATEFUL and STREAMING instead: it's fed one new
frame at a time, at a strictly increasing timestamp, and carries per-pixel
photoreceptor/threshold/leak state across calls. That state is what buys
v2e's more realistic behavior -- events get real sub-frame-resolved
timestamps (derived from how many threshold-crossings a pixel needed, not
assigned randomly) and an actual sensor noise model (leak events, shot
noise, per-pixel threshold mismatch) instead of none at all. The cost is
that it can't be called as a stateless per-step diff function like
rgb_to_events() -- V2EEventGenerator below wraps that statefulness behind a
single per-frame .generate(frame, t) call, so the caller only needs to track
a running simulation clock. See event_wrapper.py's
EventsOnlyCartpoleWrapper(event_source="v2e") for the integration; the
default event_source is still "fake" (unchanged behavior) -- pass
event_source="v2e" to opt into this real converter.
"""
import os
import sys
from typing import Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from v2e_emulator import EventEmulator  # noqa: E402

# Rec.709 luma weights -- the same formula this project's own inspo codebase
# already uses (URV-Summer-2026/cartpole_video/event/eventCamera.py's
# EventCamera.update()), kept consistent rather than switching to a
# different grayscale convention (e.g. cv2's default BT.601 coefficients).
_LUMA_R, _LUMA_G, _LUMA_B = 0.2126, 0.7152, 0.0722


def _to_grayscale_0_255(frame_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (H,W) float64 grayscale in [0,255] -- the input
    range EventEmulator.generate_events() expects (see lin_log() in
    v2e_emulator_utils.py: 'the input linear value in range 0-255')."""
    frame = frame_rgb.astype(np.float64)
    return _LUMA_R * frame[:, :, 0] + _LUMA_G * frame[:, :, 1] + _LUMA_B * frame[:, :, 2]


class V2EEventGenerator:
    """Streaming RGB -> (x,y,polarity,t) event generator, backed by v2e's
    EventEmulator. Owns one EventEmulator instance per environment; call
    .generate() once per newly-rendered frame with a strictly-increasing
    timestamp (e.g. a running simulation clock the caller advances by
    step_dt each step), .reset() at episode boundaries.
    """

    def __init__(
        self,
        device: str = "cpu",
        pos_thres: float = 0.2,
        neg_thres: float = 0.2,
        sigma_thres: float = 0.03,
        cutoff_hz: float = 0.0,
        leak_rate_hz: float = 0.1,
        shot_noise_rate_hz: float = 0.0,
        seed: int = 0,
    ):
        # Defaults mirror v2e's own EventEmulator defaults, except
        # shot_noise_rate_hz=0.0 (v2e's default too) and device="cpu" (v2e
        # defaults to "cuda" since it assumes a GPU workstation; this
        # project needs to run on a CPU-only laptop too -- pass
        # device="cuda" explicitly on the GPU cluster).
        self._kwargs = dict(
            device=device, pos_thres=pos_thres, neg_thres=neg_thres,
            sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
            leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
            seed=seed,
        )
        self._emulator = EventEmulator(**self._kwargs)

    def generate(self, frame_rgb: np.ndarray, t: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """frame_rgb: (H,W,3) uint8. t: seconds, strictly greater than the
        timestamp passed to the previous call (see reset() at episode
        boundaries). Returns (x, y, polarity, t) -- four parallel 1D arrays,
        event_to_frame.py's events_to_frames() input contract. Empty arrays
        if no events fired -- which is always true on the very first call
        after construction/reset(): EventEmulator's first frame only
        initializes its internal photoreceptor/threshold state, matching
        event_wrapper.py's existing "first observation is all-zero" contract
        for free, rather than needing a separate prev==curr special case."""
        gray = _to_grayscale_0_255(frame_rgb)
        events = self._emulator.generate_events(gray, t)  # None, or (N,4) [t,x,y,p]
        if events is None or len(events) == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty, empty
        # v2e's column order is [t,x,y,p]; events_to_frames() wants (x,y,polarity,t)
        return events[:, 1], events[:, 2], events[:, 3], events[:, 0]

    def reset(self) -> None:
        """A fresh EventEmulator instance, rather than calling its own
        .reset() -- EventEmulator.reset() doesn't clear t_previous, which
        would raise ValueError the moment a new episode's first timestamp
        (naturally lower, if the caller resets its running clock to 0 too)
        is compared against the last episode's final t_previous. A new
        instance sidesteps that instead of relying on a partial reset."""
        self._emulator = EventEmulator(**self._kwargs)
