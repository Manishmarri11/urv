# event_rl

Event-native representation learning for RL. See the project roadmap for the
full picture (pipeline, phases, team dependencies). This README documents the
representational-learning-to-PPO side of the project — `encoder/event_to_frame.py`,
`encoder/stnet_modules.py`, `encoder/encoder.py`, `env/event_wrapper.py`,
`scripts/callbacks.py`, `scripts/train.py`, `scripts/render_checkpoint_video.py`
— what each file does, its exact contract, and how they fit together, so the
source files can stay short and comment-light.

## How it all fits together

These seven files form one continuous pipeline, from raw events to a trained,
visually-evaluable policy. One full step, traced start to finish:

```
env/event_wrapper.py (EventsOnlyCartpoleWrapper.step)
  │  steps the real MuJoCo cartpole, renders the new frame
  │  diffs it against the previous frame via fake_events.rgb_to_events()
  │  (not part of this README -- the RGB-to-event teammate's swap point)
  ▼
encoder/event_to_frame.py (events_to_frames)
  │  buckets the raw (x,y,polarity,t) events into 5 positive + 5 negative
  │  time-binned tensors, each (channels, H, W)
  ▼
  (event_wrapper.py concatenates pos-then-neg into one (2*n_bins*channels,H,W)
   observation array -- this is what SB3 hands to the policy)
  ▼
encoder/encoder.py (EventEncoder.forward)
  │  splits the observation back into the 10 frames
  │  runs them through stnet_modules.py's three branches
  │  pools + projects to a fixed-size feature vector
  ▼
encoder/stnet_modules.py (SpatialBranch -> TemporalBranch -> Fusion)
  │  the actual representation-learning network -- transformer + spiking NN
  │  fused via cross-attention into one feature map
  ▼
  (feature vector feeds PPO's actor/critic heads, which pick an action --
   loop back to the top for the next step)

scripts/callbacks.py
  -- runs alongside every step during training, controlling when
     EventEncoder's TemporalBranch state resets (episode end, every step for
     the ablation variant, or neither -- it just periodically saves
     VecNormalize's stats safely)

scripts/train.py
  -- the entry point that builds the environment, wraps it in reward
     normalization, constructs PPO with EventEncoder as its feature
     extractor, and runs model.learn() with the callbacks above attached

scripts/render_checkpoint_video.py
  -- after training, loads a saved checkpoint and runs it standalone
     (no training callbacks active) to produce real video, manually
     replicating whichever state-reset behavior that checkpoint was
     actually trained under
```

The one contract that has to hold across every file in this chain:
`2 * n_bins * channels` (the observation's channel count) must agree between
`event_wrapper.py`'s `observation_space`, `EventEncoder`'s constructor
assertion, `SpatialBranch.concat`'s hardcoded `3*5*2` input channels, and
`train.py`'s `--n_bins`/`--channels` CLI args. Those CLI args exist
specifically so this is one number you change in one place, not four
independent constants that can silently drift out of sync.

## `encoder/event_to_frame.py` — Event stream → frame tensors

This is the "Event → frame" pipeline stage: it turns a raw event stream —
`(x, y, polarity, timestamp)` tuples — into the tensor shape
`SpatialBranch`/`TemporalBranch` (`encoder/stnet_modules.py`) actually
consume: **5 positive frames + 5 negative frames, each `(channels, H, W)`**.

It's deliberately independent of any specific environment or event source.
The RGB-to-event teammate's code can hand events to this module in either of
two shapes, covered by two different entry points:

- **One batch covering a whole Δt window** → `events_to_frames()`.
- **Events trickling in over time** (e.g. arriving piecemeal across several
  calls before you need a frame) → `EventFrameBuilder`, which buffers events
  and builds frames on demand. This is the natural fit if event conversion
  ends up living inside `env.step()` — see the "team dependencies" section of
  the roadmap for that still-open decision.

### Event format contract

Everywhere in this file, an event stream is four parallel 1D arrays:
`x`, `y`, `polarity`, `t`.

| field | meaning |
|---|---|
| `x`, `y` | pixel coordinates, `0 <= x < width`, `0 <= y < height`. Events landing outside that range are **silently dropped**, not clipped — clipping would pile spurious counts onto the border row/column. |
| `polarity` | any of `{-1, +1}`, `{0, 1}`, or `{False, True}`. Normalized internally, so the caller and the event-source teammate don't need to agree on a convention up front. |
| `t` | timestamp, any consistent unit (seconds, microseconds, simulation steps...) — consistent just means it has to agree with whatever `t_start`/`t_end` you pass in. |

### Channel convention

Each output frame has `channels` **identical copies** of a single event-count
image (default `channels=3`, matching the 3-channel input the STNet conv
stack expects). This mirrors the repeat-to-3-channel pattern already used
elsewhere in the project (`eventCamera.py`'s `TimeSurface` visualization does
`np.repeat(timeSurface[0,:,:,None], 3, axis=2)` for the same reason).

If a richer per-channel encoding is wanted later — e.g. one channel per finer
time sub-bin, or `(count, mean-timestamp, std)` as three distinct channels —
that only requires changing `_counts_to_frame`; everything around it is
agnostic to what "3 channels" actually contain.

### `events_to_frames(x, y, polarity, t, height, width, t_start=None, t_end=None, n_bins=5, channels=3, max_count=5.0)`

Buckets one Δt window of events into `n_bins` positive + `n_bins` negative
frames. The window `[t_start, t_end)` is split into `n_bins` equal-duration
sub-bins; each event is assigned to exactly one sub-bin by timestamp, then
counted into that sub-bin's positive or negative image by polarity.

Returns `(pos_frames, neg_frames)`, each a list of `n_bins` tensors shaped
`(channels, height, width)` — feed these straight into `SpatialBranch`/
`TemporalBranch` after adding a batch dimension (`[f.unsqueeze(0) for f in pos_frames]`).

```python
pos_frames, neg_frames = events_to_frames(
    x, y, polarity, t, height=128, width=128, t_start=0.0, t_end=1.0,
)
```

### `EventFrameBuilder` — streaming wrapper

Mirrors the class/`.update()`/`.reset()` style already used by
`EventCamera`/`TimeSurface` in `eventCamera.py`, so it drops into the
project's existing conventions rather than introducing a new pattern.

```python
builder = EventFrameBuilder(height=128, width=128)

# call as events arrive, however often that happens
builder.add_events(x, y, polarity, t)

# call once per RL step to get the 5+5 frames for that step
pos_frames, neg_frames = builder.build_frames(t_end=now)

# call at episode boundaries
builder.reset()
```

`build_frames(t_end, t_start=None)` buckets everything added since the last
call (or since construction/`reset()`), then clears the buffer. If
`t_start` is omitted it defaults to the *previous* call's `t_end` (or, on the
first call, the earliest buffered timestamp), so consecutive calls tile the
timeline with no gaps or overlap.

`reset()` clears buffered events and forgets the last window's end time, so
the next `build_frames()` call after an episode reset doesn't try to bucket a
window that spans the reset.

### Hyperparameters you need to actually tune, not leave at the default

- **`max_count`** — the raw-count value that maps to `1.0` after
  normalization. Same category of hyperparameter as the Δt window itself
  (per the roadmap's gotchas): too low and dense regions saturate to `1.0`
  and lose relative magnitude; too high and typical frames are near-zero
  almost everywhere. Tune against your actual event rate.
- **`t_end - t_start` (the Δt window)** — too short and you get incomplete/
  sparse frames; too long and fast motion blurs together across bins.

### If the event source gives you dense frames instead of sparse tuples

If the RGB-to-event teammate's code ends up handing you dense per-pixel
frames directly (the way `EventCamera.update()` in `URV2/event` already
does, rather than literal `(x,y,p,t)` arrays), you don't need this bucketing
step at all — go straight from their dense frame to pos/neg + channel-repeat,
skipping the histogram step entirely. Confirm which format you're actually
getting before building further on top of this file.

## `encoder/stnet_modules.py` — Spatial / Temporal / Fusion branches

Standalone, framework-free extraction of STNet's three architecture pieces
(`SpatialBranch`, `TemporalBranch`, `Fusion`). Full breakdown of each class,
its shapes, and known issues lives in the extraction session notes; the
short version:

- **`SpatialBranch(img_pos, img_neg)`** → `(output_sig, output_lowres)`.
  Takes the `5, (B,3,H,W)` pos/neg frames from `event_to_frame.py`, returns a
  `(B, 256, H', W')` feature map.
- **`TemporalBranch(input_pos, input_neg, transformer_fea, first_seq)`** →
  `(tem_fea, spa_fea)`, both `(B, 256, H', W')`. Carries spiking-neuron
  hidden state across calls — call `.reset()` at episode boundaries and
  `.detach_state()` between PPO training steps, or you'll hit "Trying to
  backward through the graph a second time".
- **`Fusion(tem_fea, spa_fea)`** → `(B, 256, H', W')` fused feature map —
  still spatial, not a vector. Add your own pooling (`AdaptiveAvgPool2d` +
  `Linear`) on top for the PPO encoder's `z_t`.

## `encoder/encoder.py` — `EventEncoder`, the SB3 feature extractor

Chains the three `stnet_modules.py` pieces into a single `BaseFeaturesExtractor`
PPO can use directly via `policy_kwargs=dict(features_extractor_class=EventEncoder)`.

- **`_split(observations)`** — undoes `event_wrapper.py`'s channel
  concatenation: slices `(B, 2*n_bins*channels, H, W)` back into the `n_bins`
  positive + `n_bins` negative frame lists `SpatialBranch`/`TemporalBranch`
  expect.
- **`forward(observations)`** — `_split` → `SpatialBranch` → `TemporalBranch`
  → `Fusion` → `AdaptiveAvgPool2d(1)` → `Linear` → the fixed-size feature
  vector PPO's actor/critic heads consume. By default (`auto_detach_state=True`)
  this also calls `self.temporal.detach_state()` at the end of every call —
  **required**, not optional, under PPO: SB3 calls `forward()` repeatedly
  over the same rollout batch across multiple training epochs, and without
  detaching, the second epoch's backward pass hits "Trying to backward
  through the graph a second time" (PyTorch refusing to reuse a graph a prior
  `.backward()` already consumed).
- **`reset()` / `detach_state()`** — thin delegations to
  `TemporalBranch.reset()`/`.detach_state()`. Exposed here so `callbacks.py`
  can reach into the encoder via `model.policy.features_extractor.reset()`
  without needing to know the encoder's internal structure.

**Known ceiling, not a bug:** `TemporalBranch`'s hidden state only carries
real cross-step memory during rollout *collection* (batch size = `n_envs`).
During PPO's training-phase forward passes (batch size = `batch_size`,
shuffled minibatches), the state gets reinitialized to zero, because its own
batch-size-safety check in `stnet_modules.py` fires. The gradients that
actually shape the network's weights therefore never see real cross-step
memory — see `scripts/callbacks.py`'s `StatelessTemporalCallback` for the
experiment built to test how much this actually costs in practice.

## `env/event_wrapper.py` — `EventsOnlyCartpoleWrapper`

The Gym `Env` that turns a real MuJoCo cartpole into an events-only one.
Wraps, never modifies, either real environment:

- `"CartpoleWorld-v0"` (`env/cartpole_world_env.py`) — single camera
  (`"side"`), 1D action space.
- `"CartpoleWorldDualCamera-v0"` (`env/cartpole_world_env_dual_camera.py`,
  environment teammate's in-progress work) — cameras `"world"`
  (3rd-person convenience view) or `"pole"` (the true egocentric feed
  intended for the event pipeline), 2D action space, cliff-path terrain.
  **Not runnable yet** — its XML references texture/heightmap assets under
  `env/assets/` that aren't in this repo; get them from the environment
  teammate before passing `env_id="CartpoleWorldDualCamera-v0"`.

`env_id`/`camera_name` must be a matching pair — passing `"side"` with the
dual-camera env (or vice versa) fails loudly at `gym.make()`, not silently.

Each `step()`/`reset()`: steps or resets the real env (its true numeric
observation is read, since `MujocoEnv` requires it internally, but is never
returned — this is where the events-only research constraint is actually
enforced), renders the new frame, diffs it against the previous stored frame
via `fake_events.rgb_to_events()`, converts that into frames via
`event_to_frame.events_to_frames()`, and returns the concatenated
`(2*n_bins*channels, H, W)` result as the observation. Reward/terminated/
truncated pass straight through from the real env, untouched — the events-only
constraint applies only to the observation.

`action_space` is copied verbatim from whichever real env was requested, so
it adapts automatically to 1D or 2D actions with no code change needed.

## `scripts/callbacks.py` — controlling `EventEncoder`'s hidden state from outside

A `gym.Env` has no reference to the model training on it, so there's no clean
way for the environment itself to tell `EventEncoder` when to reset its
`TemporalBranch` state — these SB3 `BaseCallback`s are the standard way to
bridge that gap from outside.

- **`TemporalResetCallback`** — resets state whenever an episode ends during
  rollout collection (the default training mode). Only correct with
  `n_envs=1`: it resets the *whole* batch of state on any episode ending,
  which would incorrectly zero out other still-running environments' memory
  too if you ever vectorize past a single environment.
- **`StatelessTemporalCallback`** — resets state after *every* rollout step
  instead, disabling cross-step memory entirely. This is the recurrence
  ablation (`train.py --stateless_temporal`): compare its reward curve
  against a normal run to see whether the cross-step memory `TemporalBranch`
  carries is actually contributing anything for this task.
- **`SafeVecNormalizeSaveCallback`** — unrelated to `TemporalBranch`.
  Periodically saves `VecNormalize`'s running reward-normalization stats,
  wrapped in try/except: that save pickles the *entire* wrapped environment
  stack (confirmed empirically — the saved files are several MB, not the few
  KB pure stats would be), which on the cluster includes a live
  `MUJOCO_GL=egl` render context that isn't guaranteed to pickle cleanly. A
  failure here warns and continues instead of taking the whole training run
  down with it.

## `scripts/train.py` — the real training entry point

Wires everything above into one runnable script.

- **`build_env(config)`** — `make_env()` → `TimeLimit` → `Monitor` (wrapped
  explicitly here, since building a custom `VecEnv`/`VecNormalize` stack
  bypasses SB3's automatic Monitor-wrapping, which would otherwise silently
  break `ep_len_mean`/`ep_rew_mean` logging) → `DummyVecEnv`.
- **`main()`** — parses CLI args, wraps the env in `VecNormalize(norm_reward=True)`
  (rescales rewards to roughly unit variance before the value function trains
  on them — added after real training runs showed `explained_variance` stuck
  at ~0 the entire run while `value_loss` grew unboundedly with episode
  length; this fix bounded `value_loss` completely and roughly doubled the
  policy's actual learning speed, even though it didn't fully resolve
  `explained_variance` itself — see the git history/session notes for the
  full diagnosis), builds or resumes `PPO` with `EventEncoder` as its feature
  extractor, then calls `model.learn()` with `WandbCallback`, the chosen
  reset callback (`TemporalResetCallback` or `StatelessTemporalCallback`, via
  `--stateless_temporal`), `CheckpointCallback`, and
  `SafeVecNormalizeSaveCallback` attached.
- **`resolve_checkpoint_path`/`resolve_vecnormalize_path`** — `--resume_model`
  accepts a bare name or full path; the matching `VecNormalize` stats file is
  located automatically by pattern-matching the model checkpoint's own
  filename convention, falling back to a fresh (not resumed) normalization
  state if no match exists.

Key CLI knobs: `--env_id`/`--camera_name` (which real environment/camera to
use), `--stateless_temporal` (the recurrence ablation), `--checkpoint_freq`
(how often to save a resumable checkpoint — matters since cluster jobs can
hit their time limit or get preempted mid-run).

## `scripts/render_checkpoint_video.py` — visual evaluation

Loads a saved checkpoint (same path-resolution contract as `--resume_model`)
and runs it standalone, outside of any training loop, to produce real video.

Two things this has to do manually that training's callbacks handled
automatically: reset `EventEncoder`'s `TemporalBranch` state once per episode
(via `model.policy.features_extractor.reset()`), since plain `model.predict()`
doesn't trigger any callback; and, if evaluating a checkpoint trained with
`--stateless_temporal`, pass `--stateless_temporal` here too so the state
resets every step instead — otherwise you'd be evaluating the model under a
different reset policy than it was actually trained under, which would make
the results misleading. Records `env.render()` (the true visual scene) for
the video, not the event tensors the policy actually sees underneath.
