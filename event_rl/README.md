# event_rl

Event-native representation learning for RL. See the project roadmap for the
full picture (pipeline, phases, team dependencies). This README documents the
representational-learning-to-PPO side of the project — `encoder/event_to_frame.py`,
`encoder/stnet_modules.py`, `encoder/encoder.py`, `env/event_wrapper.py`,
`env/rgb_to_event/v2e_events.py`, `scripts/callbacks.py`, `scripts/train.py`,
`scripts/render_checkpoint_video.py`
— what each file does, its exact contract, and how they fit together, so the
source files can stay short and comment-light.

## How it all fits together

These seven files form one continuous pipeline, from raw events to a trained,
visually-evaluable policy. One full step, traced start to finish:

```
env/event_wrapper.py (EventsOnlyCartpoleWrapper.step)
  │  steps the real MuJoCo cartpole, renders the new frame
  │  converts it to events via one of two interchangeable sources
  │  (event_source=): "fake" (fake_events.rgb_to_events(), a placeholder
  │  two-frame diff) or "v2e" (v2e_events.V2EEventGenerator, a real DVS
  │  camera simulation -- see the env/rgb_to_event/v2e_events.py section below)
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

### Measured event density on this project's own scene

Numbers below are from the dual-camera env, `pole` camera, 128×128
observations, `step_dt` 0.04 s, averaged over 15 steps. All four knobs are
reachable from `train.py`'s CLI — `--max_count`, `--event_threshold`
(fake only), `--v2e_pos_thres` / `--v2e_neg_thres` (v2e only).

| setting | nonzero obs pixels | obs max |
|---|---|---|
| `fake`, defaults (`event_threshold=0.015`, `max_count=5.0`) | 2.7% | 0.200 |
| `fake`, `event_threshold=0.005` | 4.6% | 0.200 |
| `fake`, `max_count=1.0` | 2.3% | 1.000 |
| `v2e`, defaults (`pos/neg_thres=0.2`) | **0.13%** | 0.200 |
| `v2e`, `pos/neg_thres=0.05` | **2.2%** | 1.000 |

Two things worth knowing before choosing settings:

1. **`v2e` at its own defaults is ~17x sparser than `fake`** — and `fake` is
   the distribution every earlier successful run (ep_len 26–36) trained on.
   That sparsity is realistic DVS behavior, not a bug, but it is a materially
   different input distribution. `--v2e_pos_thres 0.05 --v2e_neg_thres 0.05`
   brings it to ~2.2%, essentially matching `fake`, if a v2e run fails to
   learn at the defaults. `v2e` also produces fully-blank observations for
   roughly the first ~5 steps of each episode (the emulator initializing its
   per-pixel photoreceptor state) plus occasional isolated blanks later.

2. **`max_count=5.0` leaves 80% of the value range unused.** At that default,
   observation values are only ever `0.0` or exactly `0.2` (= 1/5) for *both*
   event sources — no pixel accumulates more than one event per bin, so the
   encoder effectively sees a binary mask rather than graded intensity.
   `--max_count 1.0` uses the full `[0,1]` range. The default is left at 5.0
   so results stay comparable with every earlier run; treat changing it as an
   experiment, not a free fix.

`v2e` is **not** slower despite the extra simulation: measured ~130 ms/step
end to end vs ~150 ms/step for `fake`, because `fake` generates far more
events for `events_to_frames()` to bucket. `--v2e_device` stays `cpu`
accordingly; `cuda` buys nothing at 128×128.

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

- `"CartpoleWorldDualCamera-v0"` (default, `env/cartpole_world_env_dual_camera.py`)
  — the environment teammate's intended final version: cliff-path terrain,
  2D action space, cameras `"pole"` (default — the true egocentric feed the
  event pipeline is actually meant to consume for training) or `"world"`
  (3rd-person convenience view — `render_checkpoint_video.py` alone
  defaults to this one, since a demo video is easier to visually
  sanity-check from 3rd-person; training/preflight still default to
  `"pole"`). Matches `igbmn` (the source-of-truth dev folder for this
  environment) byte-for-byte as of this writing — XML, env class, and
  terrain-generation script.
- `"CartpoleWorld-v0"` (`env/cartpole_world_env.py`) — the original,
  single-camera (`"side"`), 1D-action environment this project started
  with. Superseded by the dual-camera env above; kept only for anyone still
  referencing it explicitly.

`env_id`/`camera_name` must be a matching pair — passing `"side"` with the
dual-camera env (or vice versa) fails loudly at `gym.make()`, not silently.

Each `step()`/`reset()`: steps or resets the real env (its true numeric
observation is read, since `MujocoEnv` requires it internally, but is never
returned — this is where the events-only research constraint is actually
enforced), renders the new frame, converts it to events via whichever
`event_source` was requested (`"fake"` or `"v2e"` — see the two branch
methods below and the `v2e_events.py` section), converts that into frames via
`event_to_frame.events_to_frames()`, and returns the concatenated
`(2*n_bins*channels, H, W)` result as the observation. Reward/terminated/
truncated pass straight through from the real env, untouched — the events-only
constraint applies only to the observation.

`action_space` is copied verbatim from whichever real env was requested, so
it adapts automatically to 1D or 2D actions with no code change needed.

The two `event_source` branches need different bookkeeping between steps,
because the two converters have fundamentally different calling
conventions:

- **`_events_to_obs_fake`** — `fake_events.rgb_to_events()` is stateless: it
  diffs `self._prev_frame_small` (stored after every step) against the new
  frame, always over the fixed window `[0, step_dt)`.
- **`_events_to_obs_v2e`** — `V2EEventGenerator` is stateful/streaming (see
  its own module docstring): it needs one call per frame at a *strictly
  increasing* timestamp, not a two-frame diff. The wrapper tracks a running
  `self._sim_time` clock instead of a stored previous frame, advancing it by
  `step_dt` each call.

Both branches share `_frames_from_events()` for the last step (bucket
`(x,y,polarity,t)` into `event_to_frame.events_to_frames()`), so the two
converters are genuinely interchangeable from that point on.

## `env/rgb_to_event/v2e_events.py` — `V2EEventGenerator`, the real RGB-to-event converter

The real implementation of the RGB-to-event swap point `fake_events.py`
stood in for — pass `event_source="v2e"` to `make_env()`/`train.py`
(`--event_source v2e`)/`render_checkpoint_video.py`/`preflight_check.py` to
use it instead of the fake placeholder. Default everywhere is still
`"fake"`, so nothing about existing runs changes unless you opt in.

Backed by `env/rgb_to_event/v2e_emulator.py`'s `EventEmulator`, a standalone,
framework-free extraction of **[v2e](https://github.com/SensorsINI/v2e)**'s
(SensorsINI) real DVS camera simulation — per-pixel threshold mismatch
(`sigma_thres`), leak events (`leak_rate_hz`), shot noise
(`shot_noise_rate_hz`), and photoreceptor lowpass filtering (`cutoff_hz`),
plus sub-frame-resolved event timestamps derived from how many
threshold-crossings each pixel actually needed — not the fake path's
uniformly-random ones. `env/rgb_to_event/v2e_emulator_utils.py` holds the underlying
math (lin-log intensity mapping, IIR lowpass, event quantization) verbatim
from the same source. See `v2e_emulator.py`'s module docstring for exactly
what was stripped from the original (GUI display windows, the AEDat/HDF5
file-output writers — one of which pulls in `dv_processing`, a real-hardware
binding with no reason to install here — and single-pixel debug recording)
and why: none of it survives contact with "runs headless on a cluster,
consumes events in-memory."

`V2EEventGenerator.generate(frame_rgb, t)` converts the RGB frame to
grayscale via Rec.709 luma weights (matching this project's own inspo
codebase's `eventCamera.py`, not cv2's default BT.601 coefficients), feeds
it to the emulator, and reorders v2e's native `[t,x,y,p]` event columns into
the `(x,y,polarity,t)` order `event_to_frame.events_to_frames()` expects.
`.reset()` replaces the whole `EventEmulator` instance rather than calling
its own (incomplete) `.reset()` — see the method's docstring for why a
partial reset would raise `ValueError` on the next episode's first frame.

**Tuning note, same category as `event_to_frame.py`'s `max_count`**: the
constructor's defaults (`pos_thres=neg_thres=0.2`, `sigma_thres=0.03`,
`leak_rate_hz=0.1`, `cutoff_hz=shot_noise_rate_hz=0`) are v2e's own
generic defaults, not tuned against this project's actual rendered scene
(checkered ground, textured terrain, specific camera distance/motion
speed). Watch the "no signal events generated for frame" warnings and the
actual nonzero-pixel counts in early training — too-high thresholds starve
the policy of any signal at all, too-low thresholds saturate every frame
with noise.

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

## `env/rgb_to_event/` — verification status of the real (`v2e`) converter

What's actually been run and passed, not just written and assumed to work:

1. **Direct generator test** (`V2EEventGenerator` in isolation, via
   `make_env(event_source="v2e")`): `reset()` produces an all-zero
   observation (correct — `EventEmulator`'s first frame only initializes
   internal state, fires no events by design), then 10 `step()` calls with
   random actions produce real, nonzero event counts that track the
   cart/pole actually moving — confirming events are genuinely derived from
   scene motion, not synthetic/random noise.
2. **Regression check**: the default `event_source="fake"` path re-run
   after wiring `"v2e"` in, producing the same shapes/behavior as before —
   confirms adding the real converter didn't disturb the existing placeholder
   path.
3. **`scripts/preflight_check.py --event_source v2e`** — the project's own
   pre-cluster-job gate: package imports, a real headless MuJoCo render, and
   one full real PPO update through `make_env → EventEncoder → PPO`, all
   with the real converter live. Passed every check except
   `torch.cuda available` (this machine has no GPU — the same single
   expected failure every local run on this machine has had, `fake` or
   `v2e`).
4. **Multi-episode training stress test** — 1,000 timesteps,
   `--event_source v2e`, `--max_episode_steps 100` (so multiple episodes,
   multiple `EventsOnlyCartpoleWrapper.reset()` calls, each one
   reconstructing a fresh `EventEmulator`): completed cleanly, exit code 0,
   `ep_len_mean`/`ep_rew_mean` logging intact, no crashes across repeated
   resets.
5. **Re-verified after the `rgb_to_event/` reorg** — re-ran checks 2 and 3
   above after moving `fake_events.py`/`v2e_events.py`/`v2e_emulator.py`/
   `v2e_emulator_utils.py` into their own subdirectory and updating every
   import path; both still pass identically.

None of this used a GPU (this machine has none) or the real v2e CLI tool —
everything above ran through this project's own scripts, on CPU, end to end.

**Input → output, concretely**: `V2EEventGenerator.generate()` takes exactly
what `EventsOnlyCartpoleWrapper._render_downscaled()` already produces from
the real MuJoCo camera — an `(H, W, 3)` `uint8` RGB frame (`env.render()`,
downscaled via `PIL`) — and a timestamp. It returns `(x, y, polarity, t)`:
four parallel 1D arrays (the same four-array contract `fake_events.py`
already used), covering whatever events fired since the previous frame.

**How that connects into Manish's representation-learning side** — this is
*not* `EventFrameBuilder` (the streaming buffer meant for events trickling
in piecemeal across several calls before a frame is needed): each
`V2EEventGenerator.generate()` call already returns one RL step's complete,
time-bounded batch of events in one shot, which is exactly what
`events_to_frames()` (the batch entry point) wants directly —
`EventFrameBuilder` would just be re-buffering something that's already
batched. Full chain, one step:

```
CartpoleWorldDualCamera-v0's "pole" camera (default; cliff-path terrain,
  egocentric feed) -- or "world", or the legacy CartpoleWorld-v0's "side"
  │  env.render() -- true RGB pixels of the actual simulated scene
  ▼
EventsOnlyCartpoleWrapper._render_downscaled()
  │  (render_h, render_w, 3) uint8 RGB -> downscaled (obs_h, obs_w, 3) uint8 RGB
  ▼
V2EEventGenerator.generate(frame_rgb, t)                    [rgb_to_event/v2e_events.py]
  │  Rec.709 luma -> (obs_h, obs_w) float64 grayscale, [0,255]
  ▼
EventEmulator.generate_events(gray, t)                      [rgb_to_event/v2e_emulator.py]
  │  real DVS simulation: per-pixel log-intensity thresholds,
  │  leak/shot noise, sub-frame timestamp interpolation
  │  -> None (no events) or (N,4) [t, x, y, polarity]
  ▼
V2EEventGenerator.generate() reorders columns
  │  -> (x, y, polarity, t): four parallel 1D arrays
  ▼
events_to_frames(x, y, polarity, t, t_start, t_end, n_bins=5, ...)  [encoder/event_to_frame.py]
  │  buckets the step's events into 5 positive + 5 negative
  │  time-binned (channels, H, W) tensors
  ▼
EventsOnlyCartpoleWrapper concatenates pos-then-neg
  │  -> (2*n_bins*channels, H, W) observation, handed to SB3
  ▼
EventEncoder.forward()                                       [encoder/encoder.py]
  │  SpatialBranch -> TemporalBranch -> Fusion -> pooled feature vector
  ▼
PPO actor/critic heads -> action
```

Everything from `events_to_frames()` onward is exactly the same code path
`fake_events.py` already fed — Manish's representation-learning side needed
zero changes to consume the real converter's output; `event_source="v2e"`
vs `"fake"` only changes what happens upstream of that boundary.
