# -*- coding: utf-8 -*-
"""Phase 0, step 2: minimal custom feature extractor wiring pattern.

Confirms `policy_kwargs=dict(features_extractor_class=...)` works correctly
with an image-shaped observation space, BEFORE the real (heavy)
SpatialBranch+TemporalBranch+Fusion architecture enters the picture in
step3. Deliberately uses a tiny placeholder CNN -- same
BaseFeaturesExtractor pattern as URV2/event/encoderCNN.py -- so that if
something breaks here, it's isolated to the wiring itself, not to anything
about the real architecture's complexity.
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinimalCNN(BaseFeaturesExtractor):
    """Placeholder only -- not meant to produce useful features, just to
    exercise the same policy_kwargs wiring the real EventEncoder uses."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class DummyImageEnv(gym.Env):
    """Stand-in with an image-shaped observation (channels, H, W), matching
    the convention the real event-frame pipeline uses -- filled with random
    data, since only the wiring is under test here, not learning quality."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30, 128, 128), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._t += 1
        obs = self.observation_space.sample()
        return obs, 1.0, self._t >= 20, False, {}


def main():
    env = DummyImageEnv()
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=dict(features_extractor_class=MinimalCNN, features_extractor_kwargs=dict(features_dim=64)),
        n_steps=32,
        batch_size=16,
        verbose=1,
    )
    model.learn(total_timesteps=128)
    print("PASSED: custom features_extractor_class wiring works correctly with PPO.")


if __name__ == "__main__":
    main()
