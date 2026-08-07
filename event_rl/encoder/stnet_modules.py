# -*- coding: utf-8 -*-
"""
Standalone, framework-free extraction of the three STNet architecture pieces
(CVPR 2022, "Spiking Transformers for Event-based Single Object Tracking").

Source (original videoanalyst-framework locations):
  - SpatialBranch  <- Transformer   in videoanalyst/model/transfor/transfor_impl/transfor.py
  - TemporalBranch <- SNN3          in videoanalyst/model/backbone/backbone_impl/snn3.py
  - Fusion         <- Interatten    in videoanalyst/model/attention/attention_impl/interatten.py
                       (+ BasicConv pulled in from attention_impl/cbam.py, its only
                       dependency from that file)

What was stripped (plumbing only, no architecture change):
  - `@TRACK_*.register` / `@VOS_*.register` class decorators (videoanalyst.utils.Registry
    -- a plain dict-based class registry, no side effects on the class itself).
  - `ModuleBase` base class -> replaced with plain `nn.Module`. ModuleBase only added a
    `_hyper_params` dict and a `load_model_param()` checkpoint-loading helper; nothing
    that affects forward().
  - Dead imports: all three original files imported
    `videoanalyst.model.common_opr.common_block.conv_bn_relu` but never called it.

What was changed beyond plumbing (each one called out again inline where it occurs):
  1. `torch.arange(...).cuda()` hardcoded device calls in the Transformer's positional
     embedding -> now derived from the module's own parameter device, so this works on
     CPU or GPU instead of unconditionally requiring CUDA.
  2. The SNN spike function used a module-level `global thresh` read inside
     `torch.autograd.Function.forward/backward`. That's shared, mutable global state --
     unsafe the moment you have more than one TemporalBranch instance (e.g. multiple
     parallel envs) or run forward calls concurrently. It's now threaded through
     `ctx` instead. The numeric behavior is identical.
  3. TemporalBranch no longer takes an externally-constructed hidden-state list sized
     via hardcoded constants (`cfg_kernel`/`cfg_kernel_first`, which in the original repo
     are only valid for exactly z_size=127 / x_size=303 crops). State is now lazily
     shape-inferred from the actual input on first call, plus explicit `reset()` /
     `detach_state()` methods. See the TemporalBranch docstring for why this matters for
     PPO/BPTT training.

Everything else -- every conv, every attention computation, every fusion arithmetic
expression -- is copied verbatim from the original repo.
"""

import copy
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList, MultiheadAttention
from torch.nn.init import xavier_uniform_


# ============================================================================
# SpatialBranch  (Transformer / attention-based branch)
# source: videoanalyst/model/transfor/transfor_impl/transfor.py
# ============================================================================


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class _Cattention(nn.Module):
    """Cross-attention gate used inside each encoder layer to mix in the
    un-positionally-embedded stream (`srcc`) with the main stream. Channel
    count is fixed at 192 by the caller (see `_TransformerEncoderLayer`),
    independent of `d_model` -- a hardcoded coupling from the original repo,
    preserved as-is."""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x, y):
        ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        weight = self.conv1(torch.cat((x, y), 1)) * ww
        return x + self.gamma * weight * x


class _TransformerEncoderLayer(Module):
    """Standard pre-LN-ish encoder layer (self-attn + feedforward) with an
    extra `_Cattention` cross-mix step spliced in between self-attn and the
    feedforward block. `channel=192` is hardcoded here (see `_Cattention`
    note above) -- keep `d_model == 192` unless you also edit this."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        channel = 192
        self.cross_attn = _Cattention(channel)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, srcc: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        b, c, s = src.permute(1, 2, 0).size()
        input_feature = self.norm0(src + srcc)
        src2 = self.self_attn(input_feature, input_feature, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        side = int(s ** 0.5)
        src = self.cross_attn(
            src.view(b, c, side, side),
            srcc.contiguous().view(b, c, side, side),
        ).view(b, c, -1).permute(2, 0, 1)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class _TransformerDecoderLayer(Module):
    """Standard decoder layer (self-attn + cross-attn to encoder memory +
    feedforward). No `_Cattention` here -- only the encoder layer has it."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class _TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, srcc: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, srcc, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class _TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SpatialBranch(nn.Module):
    """Transformer-based spatial branch (originally `Transformer` in transfor.py).

    Consumes 5 positive + 5 negative event frames (a fixed time_window of 5 is
    baked into `self.concat`'s in_channels = 3*5*2 -- don't change the list
    lengths below without also editing that conv), downsamples them with 3
    strided convs, adds a learned 2D positional embedding, and runs them
    through a full encoder/decoder transformer stack.

    Input:
        img_pos: list of 5 tensors, each (B, 3, H, W)
        img_neg: list of 5 tensors, each (B, 3, H, W)
        (H, W must be small enough that the post-downsample spatial size is
        <= 35 in each dim -- `row_embed`/`col_embed` are `nn.Embedding(35, ...)`
        tables indexed by position. The repo's own crops -- 127 and 303 --
        downsample to 11x11 and 33x33 respectively, both within range.)

    Output:
        output_sig:    shape (1, 1) -- NOTE this is pooled over the *batch*
                        dimension (see forward(), `torch.mean(..., dim=0)`),
                        not per-sample. It is a single scalar gate shared by
                        the whole batch, not one value per batch element.
                        If you run PPO with a vectorized batch of independent
                        envs, this couples them together -- see summary notes.
        output_lowres: shape (B, 256, H', W') where H', W' are the
                        post-downsample spatial size (33x33 for a 303-px
                        input crop, 11x11 for a 127-px crop, per the repo's
                        own conv arithmetic).
    """

    def __init__(self, d_model: int = 192, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder=None, custom_decoder=None) -> None:
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = _TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = _TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = _TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.concat = nn.Conv2d(3 * 5 * 2, 64, kernel_size=1, stride=1, padding=0)

        self.downsampler1 = nn.Conv2d(64, 128, stride=2, padding=0, kernel_size=11)
        self.downsampler2 = nn.Conv2d(128, 164, stride=2, padding=0, kernel_size=9)
        self.downsampler3 = nn.Conv2d(164, 192, stride=2, padding=0, kernel_size=5)
        self.conver1 = nn.Conv2d(192, 256, stride=1, padding=0, kernel_size=1)
        self.row_embed = nn.Embedding(35, 192 // 2)
        self.col_embed = nn.Embedding(35, 192 // 2)
        self.avgpol = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, img_pos: Sequence[Tensor], img_neg: Sequence[Tensor],
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        all_input = self.downsampler1(self.concat(torch.cat(
            [img_pos[0], img_pos[1], img_pos[2], img_pos[3], img_pos[4],
             img_neg[0], img_neg[1], img_neg[2], img_neg[3], img_neg[4]], dim=1)))
        all_input = self.downsampler3(self.downsampler2(all_input))
        src = all_input
        srcc = all_input
        tgt = all_input

        h, w = src.shape[-2:]
        # ADAPTATION: original used `.cuda()` unconditionally here; derive the
        # device from the module's own weights instead so this runs on CPU too.
        device = self.row_embed.weight.device
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(src.shape[0], 1, 1, 1)

        b, c, w, h = src.size()
        src = (pos + src).view(b, c, -1).permute(2, 0, 1)
        srcc = (pos + srcc).view(b, c, -1).permute(2, 0, 1)
        tgt = tgt.view(b, c, -1).permute(2, 0, 1)

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, srcc, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        output = output.permute(1, 2, 0).view(b, c, w, h)
        output_sig = self.sig(torch.mean(torch.sum(self.avgpol(output), dim=1), dim=0))
        output_lowres = self.conver1(output)
        return output_sig, output_lowres

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# ============================================================================
# TemporalBranch  (spiking neural network branch with adaptive threshold)
# source: videoanalyst/model/backbone/backbone_impl/snn3.py
# ============================================================================

_THRESH_BIAS = 0.3   # multiplier applied to the dynamic threshold
_LENS = 0.5           # surrogate-gradient window half-width
_DECAY = 0.2          # membrane potential decay constant


class _SpatialGroupEnhance(nn.Module):
    """Dynamic spiking threshold estimated from the spatial (transformer)
    branch's features -- combines a sigmoid-gated pooled signal with a
    normalized spatial-entropy term. Output is a scalar tensor."""

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        xn = x * self.avg_pool(x)
        xn = xn.mean(dim=1, keepdim=True)
        entro = torch.mean(xn, dim=0).squeeze()
        h, w = entro.size()
        entro = entro.view(-1)
        max_ = torch.max(entro)
        min_ = torch.min(entro)
        entro = (entro - min_) / (max_ - min_) * 255
        his = torch.histc(entro, bins=256, min=0, max=255) / (h * w)
        entro_final = torch.sum(his * -torch.log(his + 1e-8))
        entro_final = entro_final / torch.count_nonzero(his)
        x = self.sig(xn)
        x = torch.mean(x)
        return x + entro_final * 10


class _SpikeFunction(torch.autograd.Function):
    """Surrogate-gradient spiking nonlinearity (rectangular window around
    threshold, per Wu et al. STBP). ADAPTATION: the original read/wrote a
    module-level `global thresh` in both forward and backward; that's
    replaced here by threading `thresh` through `ctx`, which is thread-safe
    and instance-safe. The math (hard threshold forward, rectangular-window
    surrogate backward) is unchanged."""

    @staticmethod
    def forward(ctx, input: Tensor, thresh: Tensor):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (input - ctx.thresh).abs() < _LENS
        return grad_input * temp.float(), None


def _mem_update(ops, x, mem, spike, thresh):
    mem = mem * _DECAY * (1. - spike) + ops(x)
    spike = _SpikeFunction.apply(mem, thresh)
    return mem, spike


class TemporalBranch(nn.Module):
    """Spiking-neural-network branch with an input-dependent (dynamic) firing
    threshold (originally `SNN3` in snn3.py).

    Input per forward() call:
        input_pos: list of T tensors, each (B, 3, H, W)  (T=5 in the source repo)
        input_neg: list of T tensors, each (B, 3, H, W)
        transformer_fea: (B, 256, H', W') feature map from SpatialBranch's
                          `output_lowres`, or None to use a fixed threshold of 0.3
        first_seq: bool -- True for the "template"/first call in a sequence,
                   False for subsequent calls. Only affects how `spa_fea` is
                   derived (see KNOWN ISSUE below); the temporal recurrence
                   (mem/spike update) runs the same way either way.

    Output:
        tem_fea: (B, 256, H', W') -- time-averaged 3rd-layer membrane potential,
                 same spatial size as `input_pos[i]` downsampled by the branch's
                 own 3 stride-2 convs (kernel 11/9/5), i.e. matches
                 SpatialBranch's output_lowres spatial size for the same H, W.
        spa_fea: if first_seq=True,  conv(kernel=13, stride=2) applied to
                 transformer_fea (see KNOWN ISSUE)
                 if first_seq=False, transformer_fea passed through unchanged

    Hidden state (membrane potential + spike, 3 conv layers -> 6 tensors):
        Persists across forward() calls by design (this is what makes it a
        recurrent/spiking model -- the original repo's own tracking loop
        carries this state across every tracked video frame). For RL/PPO
        training this means: call `.detach_state()` after each optimization
        step (or whenever you stop needing gradients through past timesteps),
        otherwise autograd will try to backprop through the entire episode
        history on the next `.backward()` call and PyTorch will raise
        "Trying to backward through the graph a second time". Call `.reset()`
        at the start of a new episode/rollout to zero the state again.

    KNOWN ISSUE (found by re-deriving the conv arithmetic, not by running the
    code -- flagged rather than silently fixed, since the task was to preserve
    architecture exactly): when first_seq=True, `spa_fea` is produced by
    `conv33_11` (kernel_size=13, stride=2, padding=0) applied to
    `transformer_fea`. With the source repo's own default z_size=127 template
    crop, SpatialBranch's output_lowres for that crop is 11x11 -- smaller than
    the kernel -- which would raise a RuntimeError if this branch were ever
    exercised end-to-end with the repo's own default template size. Whether
    this path was actually run by the authors (vs. always short-circuited via
    the `else` branch in their pipeline) is unverified. If you drive this
    module with first_seq=True, make sure `transformer_fea` is at least
    13x13 spatially, or you will hit the same error.
    """

    def __init__(self):
        super().__init__()

        in_planes, out_planes, stride, padding, kernel_size = (3, 64, 2, 0, 11)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = (64, 128, 2, 0, 9)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = (128, 256, 2, 0, 5)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_tem = nn.BatchNorm2d(256)
        self.relu_tem = nn.ReLU()

        self.fuse_snn_transfor = nn.Conv2d(out_planes * 2, out_planes, kernel_size=1, stride=1, padding=0)
        self.thre_w = _SpatialGroupEnhance()
        self.conv33_11 = nn.Conv2d(256, 256, kernel_size=13, stride=2, padding=0)
        self.bn_spa = nn.BatchNorm2d(256)
        self.relu_spa = nn.ReLU()

        self._channel_plan = ((64, 11, 2), (128, 9, 2), (256, 5, 2))
        self.state: Optional[List[Tensor]] = None

    def reset(self) -> None:
        """Clear hidden state. Call at the start of every new episode/rollout
        so the SNN doesn't carry membrane potential from a previous episode
        into a new one."""
        self.state = None

    def detach_state(self) -> None:
        """Detach the hidden state from the autograd graph in place. Call
        this during PPO/BPTT training after each step you've backpropagated
        through (or, for simple truncated-BPTT, after every step), otherwise
        the next forward() call extends a graph that already had .backward()
        called on part of it, and PyTorch raises 'Trying to backward through
        the graph a second time'."""
        if self.state is not None:
            self.state = [t.detach() for t in self.state]

    def _init_state(self, batch_size: int, height: int, width: int, device, dtype) -> List[Tensor]:
        state = []
        h, w = height, width
        for channels, kernel, stride in self._channel_plan:
            h = (h - kernel) // stride + 1
            w = (w - kernel) // stride + 1
            mem = torch.zeros(batch_size, channels, h, w, device=device, dtype=dtype)
            spike = torch.zeros(batch_size, channels, h, w, device=device, dtype=dtype)
            state.extend([mem, spike])
        return state

    def forward(self, input_pos: Sequence[Tensor], input_neg: Sequence[Tensor],
                transformer_fea: Optional[Tensor], first_seq: bool):
        if transformer_fea is None:
            thresh = torch.tensor(0.3, device=input_pos[0].device)
        else:
            thresh = self.thre_w(transformer_fea) * _THRESH_BIAS

        b, _, h, w = input_pos[0].shape
        if self.state is None or self.state[0].shape[0] != b:
            # Reinitialize rather than reuse mismatched-batch state. This matters
            # for SB3 PPO specifically: rollout collection calls forward() with
            # batch=n_envs, but the training phase calls it with batch=n_steps
            # (all buffered timesteps stacked as one "batch" for gradient
            # computation, NOT n_steps parallel environments). Reusing state
            # sized for one batch shape against the other would silently
            # broadcast-mismatch and crash downstream in Fusion. Reinitializing
            # here is safe but means state does NOT carry real cross-timestep
            # memory into PPO's training-phase forward calls -- only during
            # actual sequential rollout collection (where batch stays constant
            # at n_envs) does the recurrence reflect true past history.
            self.state = self._init_state(b, h, w, input_pos[0].device, input_pos[0].dtype)
        trans_snn = self.state

        time_window = len(input_pos)
        tem_c3m = 0
        for step in range(time_window):
            x_pos = input_pos[step]
            x_neg = input_neg[step]
            x = torch.where(x_pos > x_neg, x_pos, x_neg)
            c1_mem, c1_spike = _mem_update(self.conv1, x.float(), trans_snn[0], trans_snn[1], thresh)
            c2_mem, c2_spike = _mem_update(self.conv2, c1_spike, trans_snn[2], trans_snn[3], thresh)
            c3_mem, c3_spike = _mem_update(self.conv3, c2_spike, trans_snn[4], trans_snn[5], thresh)
            trans_snn = [c1_mem, c1_spike, c2_mem, c2_spike, c3_mem, c3_spike]
            tem_c3m = tem_c3m + c3_mem
        tem_fea = tem_c3m / time_window
        tem_fea = self.relu_tem(self.bn_tem(tem_fea))

        self.state = trans_snn

        if first_seq:
            spa_fea = self.relu_spa(self.bn_spa(self.conv33_11(transformer_fea)))
        else:
            spa_fea = transformer_fea

        return tem_fea, spa_fea


# ============================================================================
# Fusion  (cross-domain attention fusion of spatial + temporal features)
# source: videoanalyst/model/attention/attention_impl/interatten.py
#         (+ BasicConv from videoanalyst/model/attention/attention_impl/cbam.py)
# ============================================================================


class _BasicConv(nn.Module):
    """Pulled in from cbam.py -- only dependency Fusion needs from that file.
    Note: instantiated as `self.conv_cat` below but, per the original repo,
    never actually called in forward(). Kept for exact parameter-for-parameter
    fidelity with the original checkpoint's state_dict (dropping it would
    silently break loading any pretrained weights that include it)."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class _FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class _FSP(nn.Module):
    """Cross-Domain Integrator (CI): gates one feature stream (`guidePath`)
    by a channel-attention weight computed from the concatenation of both
    streams, then adds it into the other stream (`mainPath`)."""

    def __init__(self, in_planes, out_planes, reduction=16):
        super().__init__()
        self.filter = _FilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        return mainPath + channel_weight * guidePath


class _SpaModule(nn.Module):
    """Spatial self-attention (position-wise, from DANet-style attention)."""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return self.gamma * out + x


class _TemModule(nn.Module):
    """Channel self-attention (position-wise, from DANet-style attention)."""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return self.gamma * out + x


class Fusion(nn.Module):
    """Cross-domain attention fusion of temporal (SNN) and spatial
    (transformer) features (originally `Interatten` in interatten.py).

    Input:
        tem_fea: (B, 256, H, W) -- TemporalBranch's `tem_fea` output
        spa_fea: (B, 256, H, W) -- TemporalBranch's `spa_fea` output
                 (same spatial size as tem_fea -- both derive from the same
                 input crop via parallel stride-2 conv stacks)

    Output:
        (B, 256, H, W) fused feature map -- same shape as the inputs. This
        is still a spatial feature map, not a vector; add your own pooling
        (e.g. AdaptiveAvgPool2d + Linear) on top to get a fixed-size vector
        for a PPO policy/value head.
    """

    def __init__(self, in_planes: int = 256, out_planes: int = 256):
        super().__init__()
        self.inter_attention = _FSP(in_planes, out_planes)
        self.tem_attention = _TemModule(in_planes)
        self.spa_attention = _SpaModule(in_planes)
        self.conv_cat = _BasicConv(4 * in_planes, out_planes, kernel_size=1, stride=1,
                                    padding=0, relu=True, bn=True, bias=False)

    def forward(self, tem_fea: Tensor, spa_fea: Tensor) -> Tensor:
        tem_fea = self.tem_attention(tem_fea)
        spa_fea = self.inter_attention(tem_fea, spa_fea)
        spa_fea = self.spa_attention(spa_fea)
        return spa_fea
