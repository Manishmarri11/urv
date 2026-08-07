# -*- coding: utf-8 -*-
"""SB3 callback(s) for train.py.

WHY THIS IS A CALLBACK, NOT SOMETHING INSIDE THE ENVIRONMENT: EventEncoder
lives inside model.policy.features_extractor -- SB3 deliberately keeps the
environment and the policy/feature-extractor as separate concerns, and a
Gym env has no reference to the model training on it. There is no clean way
for EventsOnlyCartpoleWrapper.reset() to reach "the policy's encoder" without
either a global/singleton (bad practice) or this callback, which is the
standard SB3-idiomatic way to hook into episode boundaries from outside the
env. detach_state() does NOT need a callback -- EventEncoder.forward()
already calls it automatically on every forward pass (auto_detach_state=True
by default, confirmed by reading encoder.py).
"""
from stable_baselines3.common.callbacks import BaseCallback


class TemporalResetCallback(BaseCallback):
    """Calls model.policy.features_extractor.reset() whenever an episode
    ends during rollout collection, so TemporalBranch's SNN membrane
    potential doesn't leak across episode boundaries.

    KNOWN LIMITATION, flagged rather than silently handled: TemporalBranch's
    hidden state is a single batch of tensors shared across every parallel
    environment (batch dim = n_envs). This callback resets that WHOLE batch
    whenever ANY environment's episode ends. With n_envs=1 (train.py's
    current default) that's exactly correct -- there's only one env, so
    "an episode ended" and "the whole batch should reset" are the same
    thing. With n_envs>1 this would incorrectly zero out still-running
    environments' memory too, since TemporalBranch has no per-environment
    partial-reset mechanism (its state is a plain tensor list, not
    indexable per batch element for a masked reset). Do not enable
    vectorized environments with this callback as-is without addressing
    that first.
    """

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and any(dones):
            self.model.policy.features_extractor.reset()
        return True
