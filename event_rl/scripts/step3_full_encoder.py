# -*- coding: utf-8 -*-
"""Phase 0, step 3: the real EventEncoder (SpatialBranch + TemporalBranch +
Fusion), trained against a placeholder environment.

Confirms the full architecture survives real PPO training end to end
before the real environment or real event data enter the picture. This is
the permanent, checked-in version of the ad hoc test used while building
encoder.py -- same observation contract, same dummy env pattern.

Once the real environment exists (see env/), swap DummyEventEnv below for
the real one and this becomes close to scripts/train.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "encoder"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from encoder import EventEncoder

H, W = 128, 128
N_BINS, CHANNELS = 5, 3
OBS_CHANNELS = 2 * N_BINS * CHANNELS  # 30: 5 pos + 5 neg frames, 3 channels each


class DummyEventEnv(gym.Env):
    """Placeholder standing in for the real event environment -- same
    observation contract EventEncoder expects (pos+neg frames stacked
    along the channel axis), filled with random data. Not meant to produce
    a learnable task; this only exercises the pipeline's plumbing."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_CHANNELS, H, W), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._t += 1
        obs = self.observation_space.sample()
        return obs, 0.0, self._t >= 5, False, {}


def main():
    env = DummyEventEnv()
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=dict(features_extractor_class=EventEncoder, features_extractor_kwargs=dict(features_dim=256)),
        n_steps=8,
        batch_size=8,
        verbose=1,
    )
    print("PPO constructed with EventEncoder (SpatialBranch+TemporalBranch+Fusion) as features_extractor_class.")
    model.learn(total_timesteps=32)
    print("PASSED: full encoder survives real PPO training across multiple rollout/train cycles.")


if __name__ == "__main__":
    main()
