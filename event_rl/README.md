# event_rl

Event-native representation learning for RL. See the project roadmap for the
full picture (pipeline, phases, team dependencies). This README documents the
`encoder/` code itself — what each file does, its exact contract, and how to
use it — so the source files can stay short and comment-light.

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
