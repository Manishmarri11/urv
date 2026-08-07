# -*- coding: utf-8 -*-
"""EventEncoder: chains SpatialBranch -> TemporalBranch -> Fusion -> pooled
vector, wrapped as a Stable-Baselines3 BaseFeaturesExtractor. See ../README.md.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stnet_modules import SpatialBranch, TemporalBranch, Fusion


class EventEncoder(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_bins: int = 5,
        channels: int = 3,
        auto_detach_state: bool = True,
    ):
        super().__init__(observation_space, features_dim)
        self.auto_detach_state = auto_detach_state

        expected_input_channels = 2 * n_bins * channels
        assert observation_space.shape[0] == expected_input_channels, (
            f"observation_space channel dim ({observation_space.shape[0]}) must equal "
            f"2 * n_bins * channels ({expected_input_channels}) -- pos frames then neg "
            f"frames, each n_bins x channels, stacked along the channel axis"
        )

        self.n_bins = n_bins
        self.channels = channels

        fusion_out_planes = 256

        self.spatial = SpatialBranch()
        self.temporal = TemporalBranch()
        self.fusion = Fusion(out_planes=fusion_out_planes)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fusion's output channel count is fixed regardless of H/W (pool collapses
        # spatial dims to 1x1), so -- unlike EncoderCNN/FlowCNN's dry-run trick --
        # the flattened size is known without running a forward pass.
        self.linear = nn.Sequential(nn.Linear(fusion_out_planes, features_dim), nn.ReLU())

    def _split(self, observations: torch.Tensor):
        """(B, 2*n_bins*channels, H, W) -> (list of n_bins (B,channels,H,W), list of n_bins (B,channels,H,W))"""
        chunks = torch.split(observations, self.channels, dim=1)
        pos_frames = list(chunks[: self.n_bins])
        neg_frames = list(chunks[self.n_bins: 2 * self.n_bins])
        return pos_frames, neg_frames

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        pos_frames, neg_frames = self._split(observations)

        _, output_lowres = self.spatial(pos_frames, neg_frames)
        tem_fea, spa_fea = self.temporal(pos_frames, neg_frames, output_lowres, first_seq=False)
        fused = self.fusion(tem_fea, spa_fea)

        if self.auto_detach_state:
            # SB3's PPO calls forward() again for every epoch (default 10) on the
            # SAME rollout batch during evaluate_actions(), and TemporalBranch's
            # hidden state persists across those calls. Without detaching here,
            # epoch N+1's forward() reuses state that was part of epoch N's graph
            # -- a graph epoch N's .backward() already consumed -- and PyTorch
            # raises "Trying to backward through the graph a second time".
            # Detaching after every forward() call truncates BPTT to a single
            # call's own internal 5-frame loop: state VALUES still carry forward
            # (the SNN still "remembers"), but gradients never flow across
            # separate forward() calls. This makes training crash-safe under
            # vanilla PPO; it does NOT make the recurrence fully correct across
            # PPO's shuffled/repeated minibatches -- see ../README.md.
            self.temporal.detach_state()

        pooled = self.pool(fused).flatten(1)
        return self.linear(pooled)

    def reset(self) -> None:
        """Call at episode boundaries. Delegates to TemporalBranch.reset()."""
        self.temporal.reset()

    def detach_state(self) -> None:
        """Call between training steps. Delegates to TemporalBranch.detach_state()."""
        self.temporal.detach_state()
