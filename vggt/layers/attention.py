# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import math
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

# Env override: allow FlexAttention at large sequence lengths (may OOM!)
_ALLOW_FLEX_LARGE_N = str(os.getenv("VGGT_ALLOW_FLEX_LARGE_N", "0")).lower() in ("1", "true", "yes")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            # attn_mask is an additive bias broadcastable to (B, num_heads, N, N)
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=q.dtype)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                # attn_mask is additive bias; more negative => lower attention
                attn = attn + attn_mask.to(dtype=attn.dtype)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, attn_mask: Tensor | None = None) -> Tensor:
        # Fallback path when xFormers is not available: delegate to base Attention
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                # When nested tensors are requested, xFormers is required
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, attn_mask=attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSparseAttention(Attention):
    """
    Global attention that supports true sparsification via PyTorch FlexAttention.

    Usage:
      - If attn_mask is a dict context with keys:
          { 'adj': (S,S) bool or None,
            'frame_ids': (L,) int64,
            'is_hub': (L,) bool,
            'mask_hub_tokens': bool,
            'soft_mask': bool,
            'frame_bias': (S,S) float32 or None }
        it will run flex_attention with a score_mod that skips disallowed pairs
        and adds frame-level bias without creating an NxN dense mask.

      - Otherwise, falls back to base Attention (SDPA path).
    """

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:  # type: ignore[override]
        # Detect whether we received a sparse context dict
        use_sparse = isinstance(attn_mask, dict)

        if not use_sparse:
            return super().forward(x, pos=pos, attn_mask=attn_mask)

        # Build q, k, v first (rope aware, same as base)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Align dtypes for attention kernels: prefer v's dtype (usually AMP dtype)
        common_dtype = v.dtype
        if q.dtype != common_dtype:
            q = q.to(common_dtype)
        if k.dtype != common_dtype:
            k = k.to(common_dtype)

        # Extract sparse context
        ctx = attn_mask
        adj = ctx.get('adj', None)              # (S,S) bool or None
        frame_ids = ctx.get('frame_ids', None)  # (L,) int64
        is_hub = ctx.get('is_hub', None)        # (L,) bool
        mask_hub_tokens = bool(ctx.get('mask_hub_tokens', False))
        soft_mask = bool(ctx.get('soft_mask', False))
        frame_bias = ctx.get('frame_bias', None)  # (S,S) float or None

        # Telemetry holder (optional)
        telemetry = ctx.get('telemetry', None)

        # Try to import FlexAttention, handling both function and submodule forms
        flex_fn = None
        try:
            from torch.nn.attention import flex_attention as _flex_obj  # type: ignore[attr-defined]
            # _flex_obj may be a function or a module
            flex_fn = _flex_obj if callable(_flex_obj) else getattr(_flex_obj, 'flex_attention', None)
        except Exception:
            try:
                import importlib
                _mod = importlib.import_module('torch.nn.attention.flex_attention')
                flex_fn = getattr(_mod, 'flex_attention', None)
            except Exception:
                flex_fn = None

        # If no actual masking is requested, route to dense SDPA/Flash
        if (adj is None) and (not mask_hub_tokens) and (not soft_mask) and (frame_bias is None):
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = False
                telemetry['used_flex_attention'] = False
            return super().forward(x, pos=pos, attn_mask=None)

        # If soft mask without adjacency, Flex may choose dense math path; use chunked fallback to avoid OOM
        if (adj is None) and soft_mask:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            return self._fallback_chunked_sparse(x, pos, attn_mask)

        # Decide Flex vs chunked fallback conservatively to avoid dense math path
        N_total = N  # sequence length per batch element
        use_flex = (
            (flex_fn is not None)
            and (adj is not None)          # must have adjacency (top-k) to be worthwhile
            and (not soft_mask)            # soft-only tends to trigger dense path
            and ((N_total <= 4096) or _ALLOW_FLEX_LARGE_N)  # allow override via env
        )
        if not use_flex:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            return self._fallback_chunked_sparse(x, pos, attn_mask)
        else:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = True

        if frame_ids is None or is_hub is None:
            # Not enough info, fallback
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            return super().forward(x, pos=pos, attn_mask=None)

        # Ensure dtypes/devices
        device = q.device
        frame_ids = frame_ids.to(device)
        is_hub = is_hub.to(device)
        if adj is not None:
            adj = adj.to(device)
        if frame_bias is not None:
            frame_bias = frame_bias.to(device=device, dtype=q.dtype)

        # Define score modifier to skip non-neighbor pairs and add bias without NxN allocation
        # Must be torch.compile-friendly: no Python bool casts or data-dependent branches.
        def score_mod(score, b: int, h: int, qi: int, kj: int):
            fi = frame_ids[qi]
            fj = frame_ids[kj]

            # adj permission (0-d bool): if no adj, treat as allowed
            if adj is not None:
                adj_val = adj[fi, fj]
                adj_allowed = adj_val if adj_val.dtype == torch.bool else (adj_val != 0)
            else:
                adj_allowed = torch.tensor(True, dtype=torch.bool, device=score.device)

            # hub masking: allow when same frame OR neither token is hub
            hub_q = is_hub[qi]
            hub_k = is_hub[kj]
            same_frame = (fi == fj)
            neither_hub = ~(hub_q | hub_k)

            # Combine hub rule with a tensor flag for mask_hub_tokens
            hub_rule = same_frame | neither_hub  # allowed if true
            hub_flag = torch.tensor(mask_hub_tokens, dtype=torch.bool, device=score.device)
            # If hub_flag == False, hub_rule is ignored (treated as True)
            hub_allowed = (~hub_flag) | hub_rule

            allowed = adj_allowed & hub_allowed

            # Optional soft bias
            if soft_mask and (frame_bias is not None):
                bias_val = frame_bias[fi, fj]
            else:
                bias_val = torch.zeros((), dtype=score.dtype, device=score.device)

            score_plus = score + bias_val
            neg_inf = torch.tensor(float('-inf'), dtype=score.dtype, device=score.device)
            return torch.where(allowed, score_plus, neg_inf)

        # Run FlexAttention with a proper BlockMask to enforce block-sparse compute
        flex_kwargs = {"score_mod": score_mod}
        if adj is not None:
            try:
                from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask

                hub_flag = torch.tensor(mask_hub_tokens, dtype=torch.bool, device=q.device)

                def mask_mod(b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, kj: torch.Tensor):
                    fi = frame_ids[qi]
                    fj = frame_ids[kj]
                    # adjacency allow
                    adj_val = adj[fi, fj]
                    adj_allowed = adj_val if adj_val.dtype == torch.bool else (adj_val != 0)
                    # hub rule: allow when same frame OR neither token is hub; disable only if flag is on
                    hub_q = is_hub[qi]
                    hub_k = is_hub[kj]
                    same_frame = (fi == fj)
                    neither_hub = ~(hub_q | hub_k)
                    hub_allowed = (~hub_flag) | (same_frame | neither_hub)
                    return adj_allowed & hub_allowed

                # BLOCK_SIZE in tokens; since tokens are grouped per-frame with size P, use (P,P)
                block_mask = _create_block_mask(
                    mask_mod,
                    B=q.shape[0],
                    H=q.shape[1],
                    Q_LEN=N,
                    KV_LEN=N,
                    device=q.device,
                    BLOCK_SIZE=(int(P), int(P)),
                )
                flex_kwargs["block_mask"] = block_mask
            except Exception:
                pass

        # Execute FlexAttention. It expects [B, H, N, Dh]
        out = flex_fn(q, k, v, **flex_kwargs)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _fallback_chunked_sparse(self, x: Tensor, pos, ctx: dict) -> Tensor:
        """Fallback path when FlexAttention is unavailable.
        Implements per-frame chunked SDPA with small per-chunk masks, avoiding NxN.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        device = q.device
        # Align dtypes for SDPA
        common_dtype = v.dtype
        if q.dtype != common_dtype:
            q = q.to(common_dtype)
        if k.dtype != common_dtype:
            k = k.to(common_dtype)
        dtype = common_dtype

        adj = ctx.get('adj', None)
        frame_ids = ctx.get('frame_ids', None)
        is_hub = ctx.get('is_hub', None)
        mask_hub_tokens = bool(ctx.get('mask_hub_tokens', False))
        soft_mask = bool(ctx.get('soft_mask', False))
        frame_bias = ctx.get('frame_bias', None)

        if frame_ids is None:
            # Nothing to do; dense fallback
            return super().forward(x, pos=pos, attn_mask=None)

        frame_ids = frame_ids.to(device)
        if is_hub is None:
            is_hub = torch.zeros(N, dtype=torch.bool, device=device)
        else:
            is_hub = is_hub.to(device)
        if adj is not None:
            adj = adj.to(device).to(torch.bool)
            S = int(adj.shape[0])
        else:
            S = int(frame_ids.max().item()) + 1
        if frame_bias is not None:
            frame_bias = frame_bias.to(device).to(torch.float32)

        # Infer P from N and S
        assert N % S == 0, f"Sequence length N={N} must be divisible by number of frames S={S}"
        P = N // S

        # If there is effectively no masking or bias, go dense SDPA directly
        if (adj is None) and (not mask_hub_tokens) and (not soft_mask) and (frame_bias is None):
            return super().forward(x, pos=pos, attn_mask=None)

        # Pre-allocate output buffer
        out_buf = torch.empty_like(q)

        # Build per-frame neighbor lists
        all_frames = torch.arange(S, device=device)

        for s in range(S):
            # q slice for frame s
            q_start = s * P
            q_end = (s + 1) * P
            q_s = q[:, :, q_start:q_end, :]  # [B,H,P,Dh]

            # Select neighbor frames for keys/values
            if adj is None:
                neigh = all_frames
            else:
                neigh = torch.nonzero(adj[s], as_tuple=True)[0]
                if neigh.numel() == 0:
                    # at least allow self
                    neigh = torch.tensor([s], device=device)

            # Gather keys/values for neighbor frames
            k_blocks = []
            v_blocks = []
            for j in neigh.tolist():
                ks = k[:, :, j * P : (j + 1) * P, :]
                vs = v[:, :, j * P : (j + 1) * P, :]
                k_blocks.append(ks)
                v_blocks.append(vs)
            k_s = torch.cat(k_blocks, dim=2)  # [B,H,P*|N_s|,Dh]
            v_s = torch.cat(v_blocks, dim=2)

            Klen = k_s.shape[2]

            # Build small additive bias [P, Klen]
            bias_2d = torch.zeros(P, Klen, dtype=torch.float32, device=device)
            any_bias = False

            # Add soft frame-level bias
            if soft_mask and (frame_bias is not None):
                for t_idx, j in enumerate(neigh.tolist()):
                    fb = float(frame_bias[s, j].item())
                    if fb != 0.0:
                        bias_2d[:, t_idx * P : (t_idx + 1) * P] += fb
                        any_bias = True

            # Apply hub masking across frames if requested
            if mask_hub_tokens:
                # Row mask: hub queries from s cannot attend to cross-frame keys
                row_is_hub = is_hub[q_start:q_end]  # [P]
                if row_is_hub.any():
                    for t_idx, j in enumerate(neigh.tolist()):
                        if j == s:
                            continue
                        bias_2d[row_is_hub, t_idx * P : (t_idx + 1) * P] = float('-inf')
                        any_bias = True

                # Column mask: cross-frame hub keys cannot be attended by any query
                for t_idx, j in enumerate(neigh.tolist()):
                    if j == s:
                        continue
                    col_hub = is_hub[j * P : (j + 1) * P]  # [P]
                    if col_hub.any():
                        # Broadcast to all rows
                        bias_block = bias_2d[:, t_idx * P : (t_idx + 1) * P]
                        bias_block[:, col_hub] = float('-inf')
                        any_bias = True

            # SDPA expects bias dtype to match q dtype
            bias = None
            if any_bias:
                bias = bias_2d.to(dtype).view(1, 1, P, Klen)

            # Perform SDPA on the chunk
            out_s = F.scaled_dot_product_attention(
                q_s, k_s, v_s, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
            )  # [B,H,P,Dh]

            # Scatter to full sequence positions
            out_buf[:, :, q_start:q_end, :] = out_s

        # Project back to model dim
        out = out_buf.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
