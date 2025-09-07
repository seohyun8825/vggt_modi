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

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


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

        # Try to import flex_attention lazily
        telemetry = None
        try:
            from torch.nn.attention import flex_attention as _flex_attention
            telemetry = attn_mask.get('telemetry', None)
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = True
        except Exception:
            # Flex not available -> fallback to dense SDPA without mask
            telemetry = attn_mask.get('telemetry', None)
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            return super().forward(x, pos=pos, attn_mask=None)

        # Extract sparse context
        ctx = attn_mask
        adj = ctx.get('adj', None)              # (S,S) bool or None
        frame_ids = ctx.get('frame_ids', None)  # (L,) int64
        is_hub = ctx.get('is_hub', None)        # (L,) bool
        mask_hub_tokens = bool(ctx.get('mask_hub_tokens', False))
        soft_mask = bool(ctx.get('soft_mask', False))
        frame_bias = ctx.get('frame_bias', None)  # (S,S) float or None

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
            frame_bias = frame_bias.to(device)

        # Define score modifier to skip non-neighbor pairs and add bias without NxN allocation
        def score_mod(score, b: int, h: int, qi: int, kj: int):
            fi = frame_ids[qi]
            fj = frame_ids[kj]
            allow = True
            if adj is not None:
                allow = bool(adj[fi, fj])
            if allow and mask_hub_tokens and (fi != fj) and (bool(is_hub[qi]) or bool(is_hub[kj])):
                allow = False
            if not allow:
                return -math.inf
            if soft_mask and (frame_bias is not None):
                # frame_bias holds additive penalties (usually <= 0)
                return score + frame_bias[fi, fj]
            return score

        # Run flex attention. It expects [B, H, N, Dh]
        out = _flex_attention(q, k, v, score_mod=score_mod)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
