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
import os

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


class StatelessTemporalCallback(BaseCallback):
    """Ablation variant of TemporalResetCallback: calls
    model.policy.features_extractor.reset() after EVERY rollout step, not
    just at episode boundaries -- zeroes TemporalBranch's membrane
    potential/spike state before each subsequent forward() call, so no
    memory carries across steps at all (only the within-call 5-frame
    time-binned window each observation already encodes remains).

    Used to test whether TemporalBranch's cross-step recurrence contributes
    anything for this task, by comparing a training run using this callback
    against an otherwise-identical run using TemporalResetCallback.

    Same n_envs=1 caveat as TemporalResetCallback: reset() zeroes the whole
    batch of state, which is only correct when there's a single environment.
    """

    def _on_step(self) -> bool:
        self.model.policy.features_extractor.reset()
        return True


class SafeVecNormalizeSaveCallback(BaseCallback):
    """Periodically saves VecNormalize's running reward-normalization stats,
    wrapped in try/except.

    WHY THE TRY/EXCEPT: VecNormalize.save() pickles itself -- which includes
    a reference to the ENTIRE wrapped environment stack, not just its own
    mean/variance arrays (confirmed empirically: the saved files are several
    MB, not the few KB pure stats would be). On the cluster that stack holds
    a live MUJOCO_GL=egl rendering context, which isn't guaranteed to be
    picklable. SB3's own CheckpointCallback(save_vecnormalize=True) has no
    error handling around this call -- a pickling failure there would
    propagate up and kill the entire training run mid-checkpoint. Losing one
    checkpoint's worth of resumable reward-normalization stats is a minor,
    recoverable problem (VecNormalize just restarts its running statistics
    fresh on resume); losing hours of training to a crash over a side
    feature is not worth that risk, so this callback is used instead of the
    built-in flag.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vecnormalize_env = self.model.get_vec_normalize_env()
            if vecnormalize_env is not None:
                path = os.path.join(
                    self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl"
                )
                try:
                    vecnormalize_env.save(path)
                except Exception as e:
                    print(
                        f"WARNING: failed to save VecNormalize stats to {path}: "
                        f"{type(e).__name__}: {e} -- continuing training without this checkpoint's stats."
                    )
        return True
