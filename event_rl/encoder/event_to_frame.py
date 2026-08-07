# -*- coding: utf-8 -*-
"""Event stream -> frame tensors. Contract, usage, and hyperparameter notes: ../README.md"""

from typing import List, Optional, Tuple

import numpy as np
import torch


def _normalize_polarity(polarity: np.ndarray) -> np.ndarray:
    pol = np.asarray(polarity)
    if pol.dtype == bool:
        return np.where(pol, 1, -1).astype(np.int8)
    return np.where(pol.astype(np.int64) > 0, 1, -1).astype(np.int8)


def _bucket_counts(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((height, width), dtype=np.float32)
    # drop out-of-bounds rather than clip, so they don't pile onto the border row/column
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[in_bounds], y[in_bounds]
    flat = y.astype(np.int64) * width + x.astype(np.int64)
    counts = np.bincount(flat, minlength=height * width)[: height * width]
    return counts.reshape(height, width).astype(np.float32)


def _counts_to_frame(counts: np.ndarray, channels: int, max_count: float) -> torch.Tensor:
    normalized = np.clip(counts / max_count, 0.0, 1.0)
    return torch.from_numpy(normalized).float().unsqueeze(0).repeat(channels, 1, 1)


def events_to_frames(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    t: np.ndarray,
    height: int,
    width: int,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    n_bins: int = 5,
    channels: int = 3,
    max_count: float = 5.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t, dtype=np.float64)
    pol = _normalize_polarity(polarity)

    if t_start is None:
        t_start = float(t.min()) if len(t) else 0.0
    if t_end is None:
        t_end = float(t.max()) + 1e-9 if len(t) else t_start + 1e-9
    assert t_end > t_start, "t_end must be strictly greater than t_start"

    in_window = (t >= t_start) & (t < t_end)
    x, y, pol, t = x[in_window], y[in_window], pol[in_window], t[in_window]

    bin_width = (t_end - t_start) / n_bins
    bin_idx = np.clip(((t - t_start) / bin_width).astype(np.int64), 0, n_bins - 1)

    pos_frames, neg_frames = [], []
    for b in range(n_bins):
        in_bin = bin_idx == b
        pos_mask = in_bin & (pol > 0)
        neg_mask = in_bin & (pol < 0)

        pos_counts = _bucket_counts(x[pos_mask], y[pos_mask], height, width)
        neg_counts = _bucket_counts(x[neg_mask], y[neg_mask], height, width)

        pos_frames.append(_counts_to_frame(pos_counts, channels, max_count))
        neg_frames.append(_counts_to_frame(neg_counts, channels, max_count))

    return pos_frames, neg_frames


class EventFrameBuilder:
    """Streaming wrapper around events_to_frames(). See ../README.md."""

    def __init__(self, height: int, width: int, n_bins: int = 5, channels: int = 3,
                 max_count: float = 5.0):
        self.height = height
        self.width = width
        self.n_bins = n_bins
        self.channels = channels
        self.max_count = max_count
        self._buf_x: List[np.ndarray] = []
        self._buf_y: List[np.ndarray] = []
        self._buf_p: List[np.ndarray] = []
        self._buf_t: List[np.ndarray] = []
        self._last_t_end: Optional[float] = None

    def add_events(self, x, y, polarity, t) -> None:
        self._buf_x.append(np.asarray(x))
        self._buf_y.append(np.asarray(y))
        self._buf_p.append(np.asarray(polarity))
        self._buf_t.append(np.asarray(t))

    def build_frames(self, t_end: float, t_start: Optional[float] = None
                      ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = np.concatenate(self._buf_x) if self._buf_x else np.array([])
        y = np.concatenate(self._buf_y) if self._buf_y else np.array([])
        p = np.concatenate(self._buf_p) if self._buf_p else np.array([])
        t = np.concatenate(self._buf_t) if self._buf_t else np.array([])

        if t_start is None:
            t_start = self._last_t_end if self._last_t_end is not None else (
                float(t.min()) if len(t) else t_end - 1e-9)

        pos_frames, neg_frames = events_to_frames(
            x, y, p, t, self.height, self.width,
            t_start=t_start, t_end=t_end, n_bins=self.n_bins,
            channels=self.channels, max_count=self.max_count,
        )

        self._last_t_end = t_end
        self._buf_x, self._buf_y, self._buf_p, self._buf_t = [], [], [], []
        return pos_frames, neg_frames

    def reset(self) -> None:
        self._buf_x, self._buf_y, self._buf_p, self._buf_t = [], [], [], []
        self._last_t_end = None
