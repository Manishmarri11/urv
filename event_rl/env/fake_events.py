# -*- coding: utf-8 -*-
"""Fake RGB-to-event converter -- placeholder for the real one, not yet
ready from the teammate.

rgb_to_events() is the ONLY function in this file, and the single spot to
swap out once the real converter exists: it must keep the same signature
(two consecutive frames + a time window in, four parallel event arrays out)
for event_wrapper.py to keep working unmodified.
"""
from typing import Optional, Tuple

import numpy as np


def rgb_to_events(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    t_start: float,
    t_end: float,
    threshold: float = 0.015,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FAKE event generator via grayscale frame-differencing + thresholding.
    NOT a physically accurate event-camera simulation -- a placeholder that
    produces the right shape/format of data (x, y, polarity, t) so the rest
    of the pipeline (event_to_frame.py onward) can be built and tested
    before the real converter exists.

    prev_frame, current_frame: (H, W, 3) uint8 RGB arrays, same shape.
    t_start, t_end: the time window these two frames span (e.g. one
        env.step()'s worth of sim time) -- events get timestamps within
        this range.
    threshold: minimum grayscale change (in [0,1] units) to count as an
        event. This is a real tunable parameter, same caution as
        event_to_frame.py's max_count -- too low and near-static regions
        produce noise events every step; too high and slow motion produces
        no events at all.

    Returns x, y, polarity, t -- four parallel 1D arrays, the exact format
    event_to_frame.py's events_to_frames() expects.

    HONEST SIMPLIFICATION, not a hidden approximation: a single frame diff
    carries no real information about *when* within the window each pixel
    actually changed, since it's just two snapshots. Timestamps are
    assigned uniformly at random within [t_start, t_end) per event, which
    is enough for event_to_frame.py's n_bins time-slicing to do something
    reasonable, but is not a substitute for genuine sub-frame timing a real
    event camera would provide.
    """
    if rng is None:
        rng = np.random.default_rng()

    prev_gray = prev_frame.astype(np.float32).mean(axis=2) / 255.0
    curr_gray = current_frame.astype(np.float32).mean(axis=2) / 255.0
    diff = curr_gray - prev_gray

    pos_y, pos_x = np.nonzero(diff > threshold)
    neg_y, neg_x = np.nonzero(diff < -threshold)

    x = np.concatenate([pos_x, neg_x]).astype(np.int64)
    y = np.concatenate([pos_y, neg_y]).astype(np.int64)
    polarity = np.concatenate([
        np.ones(len(pos_x), dtype=np.int8),
        -np.ones(len(neg_x), dtype=np.int8),
    ])

    n = len(x)
    t = rng.uniform(t_start, t_end, size=n).astype(np.float64) if n > 0 else np.zeros(0, dtype=np.float64)

    return x, y, polarity, t
