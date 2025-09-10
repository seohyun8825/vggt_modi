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
import hashlib
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

# Env override: allow FlexAttention at large sequence lengths (may OOM!)
_ALLOW_FLEX_LARGE_N = str(os.getenv("VGGT_ALLOW_FLEX_LARGE_N", "0")).lower() in ("1", "true", "yes")

# Cache for BlockMask objects to avoid expensive re-creation
_BLOCK_MASK_CACHE: dict[tuple, object] = {}

# Optional override for FlexAttention block size in tokens
try:
    _FLEX_BLOCK_OVERRIDE = int(os.getenv("VGGT_FLEX_BLOCK", "0"))
except Exception:
    _FLEX_BLOCK_OVERRIDE = 0


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

        # Telemetry (debug) holder
        telemetry = ctx.get('telemetry', None)
        if telemetry is None:
            telemetry = {}
            ctx['telemetry'] = telemetry
        telemetry.update({
            'entered_sparse': True,
            'adj_is_none': adj is None,
            'soft_mask': bool(soft_mask),
            'mask_hub_tokens': bool(mask_hub_tokens),
            'q_len_init': int(N),
        })

        def _dbg(msg: str, **kw):
            telemetry.update(kw)
            if os.getenv("VGGT_DEBUG", "0").lower() in ("1", "true", "yes"):
                try:
                    print(f"[VGGT][GlobalSparseAttention] {msg} :: {kw}")
                except Exception:
                    pass

        def _fail_fallback(reason: str):
            telemetry['fallback_reason'] = reason
            if os.getenv("VGGT_FAIL_ON_FALLBACK", "0").lower() in ("1", "true", "yes"):
                raise RuntimeError(f"Forced fail on fallback: {reason}")

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
        telemetry['flex_import_ok'] = bool(flex_fn)
        if not telemetry['flex_import_ok']:
            _dbg("Flex import failed")

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

        # Decide Flex vs chunked fallback. Try Flex when available and adjacency is provided.
        # Optional global disable via env
        _disable_flex_env = str(os.getenv("VGGT_DISABLE_FLEX", "0")).lower() in ("1", "true", "yes")
        # Avoid repeated OOMs within the same module instance
        if not hasattr(self, "_flex_failed_once"):
            self._flex_failed_once = False

        use_flex = (
            (flex_fn is not None)
            and (adj is not None)          # must have adjacency (top-k) to be worthwhile
            and (not soft_mask)            # soft-only tends to trigger dense path
        )
        if _disable_flex_env or self._flex_failed_once:
            use_flex = False
        if not use_flex:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            _fail_fallback("gate_use_flex_false")
            return self._fallback_chunked_sparse(x, pos, attn_mask)
        else:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False  # will set True after successful flex call

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
        # Only pass score_mod when features beyond adjacency are required
        use_score_mod = bool(soft_mask) or bool(mask_hub_tokens)
        flex_kwargs = {"score_mod": score_mod} if use_score_mod else {}
        created_block_mask = False
        if adj is not None:
            # Derive S and P from frame_ids (tokens per frame is uniform)
            S0 = int(frame_ids.max().item()) + 1  # real frames (before padding)
            assert N % S0 == 0, f"Sequence length N={N} must be divisible by frames S0={S0}"
            P = int(N // S0)

            # Choose 128-aligned block sizes for fast Flex kernels (override via env)
            QBS = KBS = int(_FLEX_BLOCK_OVERRIDE) if _FLEX_BLOCK_OVERRIDE > 0 else 128
            BLOCK_SIZE = (QBS, KBS)
            telemetry.update({'QBS': int(QBS), 'KBS': int(KBS)})

            # 2.6 compatibility shim: pad to N_pad multiple of 128 so BlockMask shape matches q/k/v
            orig_N = int(N)
            N_pad = int(((N + QBS - 1) // QBS) * QBS)
            do_pad = (N_pad != N)
            if do_pad:
                pad_len = N_pad - N
                Bq, Hq, Dh = q.shape[0], q.shape[1], q.shape[-1]
                # pad q/k/v with zeros at the end along sequence dim
                q_pad = q.new_zeros((Bq, Hq, N_pad, Dh)); q_pad[:, :, :N, :] = q; q = q_pad
                k_pad = k.new_zeros((Bq, Hq, N_pad, Dh)); k_pad[:, :, :N, :] = k; k = k_pad
                v_pad = v.new_zeros((Bq, Hq, N_pad, Dh)); v_pad[:, :, :N, :] = v; v = v_pad
                # pad frame_ids with a dummy frame id S0
                frame_ids_pad = torch.empty(N_pad, dtype=frame_ids.dtype, device=frame_ids.device)
                frame_ids_pad[:N] = frame_ids
                frame_ids_pad[N:] = int(S0)
                frame_ids = frame_ids_pad
                # ensure is_hub has padded zeros for dummy tokens
                if is_hub is None:
                    is_hub = torch.zeros(N_pad, dtype=torch.bool, device=device)
                else:
                    if is_hub.numel() != N_pad:
                        is_hub_pad = torch.zeros(N_pad, dtype=is_hub.dtype, device=is_hub.device)
                        is_hub_pad[:N] = is_hub
                        is_hub = is_hub_pad
                # expand adjacency with dummy row/col all False
                adj_bool0 = adj.to(torch.bool)
                adj_pad = torch.zeros((S0 + 1, S0 + 1), dtype=torch.bool, device=adj_bool0.device)
                adj_pad[:S0, :S0] = adj_bool0
                adj = adj_pad
                # logical frame count increases by 1 (dummy frame)
                S = S0 + 1
                N = N_pad
            else:
                S = S0
            telemetry.update({'do_pad': bool(do_pad), 'N_pad': int(N), 'S0': int(S0), 'P': int(P)})
            try:
                telemetry['frame_ids_max'] = int(frame_ids.max().item())
                telemetry['frame_ids_len'] = int(frame_ids.numel())
            except Exception:
                pass

            # Try robust import for BlockMask
            _BlockMask = None
            try:
                from torch.nn.attention.flex_attention import BlockMask as _BM  # type: ignore[attr-defined]
                _BlockMask = _BM
            except Exception:
                try:
                    import importlib
                    _mod = importlib.import_module('torch.nn.attention.flex_attention')
                    _BlockMask = getattr(_mod, 'BlockMask', None)
                except Exception:
                    _BlockMask = None
            telemetry['blockmask_import_ok'] = bool(_BlockMask)
            if not telemetry['blockmask_import_ok']:
                _dbg("BlockMask import failed")
                _fail_fallback("no_blockmask_class")

            if _BlockMask is not None:
                # Build BlockMask on a 128-token block grid by mapping frame adjacency onto block indices
                adj_bool = adj.to(torch.bool)
                # ensure square and diagonal allowed
                if adj_bool.shape != (S, S):
                    if isinstance(telemetry, dict):
                        telemetry['used_sparse_context'] = True
                        telemetry['used_flex_attention'] = False
                    _fail_fallback("adj_shape_mismatch")
                    return self._fallback_chunked_sparse(x, pos, attn_mask)
                if (~adj_bool.diagonal()).any():
                    adj_bool = adj_bool.clone()
                    adj_bool.fill_diagonal_(True)
                # record diagonal check
                try:
                    telemetry['adj_diag_all_true'] = bool(torch.all(torch.diag(adj_bool)))
                except Exception:
                    pass

                # Compute number of query/kv blocks for 128-sized tiles
                Q_blocks = (N + QBS - 1) // QBS
                KV_blocks = (N + KBS - 1) // KBS
                telemetry.update({'Q_blocks': int(Q_blocks), 'KV_blocks': int(KV_blocks)})

                # For each query block, compute which frames it overlaps (robust via frame_ids)
                blk_start = torch.arange(Q_blocks, device=device, dtype=torch.int32) * QBS
                blk_end = torch.minimum(blk_start + QBS, torch.tensor(N, device=device, dtype=torch.int32))
                qf_start = frame_ids[blk_start.to(torch.long)]
                qf_endm1 = frame_ids[(blk_end - 1).to(torch.long)]

                row0 = adj_bool[qf_start]  # [Q_blocks, S]
                same = (qf_endm1 == qf_start).unsqueeze(1)
                row1 = torch.where(same, torch.zeros_like(row0), adj_bool[qf_endm1])
                kv_frames_allowed = (row0 | row1)  # [Q_blocks, S]

                # Map allowed frames to KV block index ranges
                frames_real = torch.arange(S0, device=device)
                frame_st_blk_real = ((frames_real * P) // KBS).to(torch.int32)  # [S0]
                frame_ed_blk_real = ((((frames_real + 1) * P + KBS - 1) // KBS) - 1).to(torch.int32)  # [S0]
                if do_pad:
                    st_blk_pad = torch.tensor(int(N // KBS), device=device, dtype=torch.int32)
                    ed_blk_pad = torch.tensor(int((N - 1) // KBS), device=device, dtype=torch.int32)
                    # padded region spans [N_pad - pad_len, N_pad), but since N was updated to N_pad, start index of pad is orig_N
                    st_blk_pad = torch.tensor(int(orig_N // KBS), device=device, dtype=torch.int32)
                    ed_blk_pad = torch.tensor(int((N_pad - 1) // KBS), device=device, dtype=torch.int32)
                    frame_st_blk = torch.cat([frame_st_blk_real, st_blk_pad.view(1)])
                    frame_ed_blk = torch.cat([frame_ed_blk_real, ed_blk_pad.view(1)])
                else:
                    frame_st_blk = frame_st_blk_real
                    frame_ed_blk = frame_ed_blk_real

                kv_indices = torch.full((Q_blocks, KV_blocks), -1, dtype=torch.int32, device=device)
                kv_counts = torch.zeros(Q_blocks, dtype=torch.int32, device=device)

                # Note: Q_blocks is typically modest (e.g., ~N/128). Simple Python loop is acceptable.
                for qb in range(int(Q_blocks)):
                    allow_frames = torch.nonzero(kv_frames_allowed[qb], as_tuple=True)[0]
                    if allow_frames.numel() == 0:
                        allow_frames = qf_start[qb:qb+1]
                    # Gather block ranges and take union
                    starts = frame_st_blk[allow_frames]
                    ends = frame_ed_blk[allow_frames]
                    blocks_list = []
                    for s_blk, e_blk in zip(starts.tolist(), ends.tolist()):
                        if e_blk >= s_blk:
                            blocks_list.append(torch.arange(s_blk, e_blk + 1, device=device, dtype=torch.int32))
                    if len(blocks_list) > 0:
                        block_ids = torch.unique(torch.cat(blocks_list), sorted=True)
                        n = int(block_ids.numel())
                        kv_indices[qb, :n] = block_ids
                        kv_counts[qb] = n

                # kv_counts stats for debugging
                try:
                    telemetry['kv_counts_min'] = int(kv_counts.min().item())
                    telemetry['kv_counts_max'] = int(kv_counts.max().item())
                    telemetry['kv_counts_mean'] = float(kv_counts.float().mean().item())
                    telemetry['kv_counts_p95'] = int(torch.quantile(kv_counts.float(), 0.95).item())
                except Exception:
                    pass

                # Estimate density and available memory; optionally early-fallback if too dense
                try:
                    avg_k_blocks = float(kv_counts.float().mean().item()) if kv_counts.numel() > 0 else 0.0
                    est_k_per_q = avg_k_blocks * float(KBS)
                    est_density = float(est_k_per_q / float(N)) if N > 0 else 1.0
                    telemetry['est_density'] = est_density
                    if torch.cuda.is_available():
                        free_b, total_b = torch.cuda.mem_get_info()
                        telemetry['cuda_free_GB'] = float(free_b / (1024**3))
                        telemetry['cuda_total_GB'] = float(total_b / (1024**3))
                    # Experience-based guard: if too dense, prefer fallback
                    if est_density > 0.30:
                        _fail_fallback("est_density_high")
                        return self._fallback_chunked_sparse(x, pos, attn_mask)
                except Exception:
                    pass

                # Build BlockMask (prefer seq_lengths if available)
                kv_num_blocks = kv_counts.view(1, 1, int(Q_blocks))
                kv_indices_4d = kv_indices.view(1, 1, int(Q_blocks), int(KV_blocks))

                def _adj_fingerprint(t: torch.Tensor) -> str:
                    try:
                        tb = t.to(torch.bool).to(torch.uint8).contiguous().cpu().numpy().tobytes()
                        return hashlib.sha1(tb).hexdigest()
                    except Exception:
                        return f"S{S}_nnz{int(t.to(torch.bool).sum().item())}"

                kv_cols = int(kv_indices_4d.shape[-1])
                cache_key = (int(N), int(S0), int(P), BLOCK_SIZE, kv_cols, _adj_fingerprint(adj_bool))

                block_mask = _BLOCK_MASK_CACHE.get(cache_key)

                if block_mask is None:
                    try:
                        import inspect
                        sig = inspect.signature(_BlockMask.from_kv_blocks)
                        if 'seq_lengths' in sig.parameters:
                            block_mask = _BlockMask.from_kv_blocks(
                                kv_num_blocks,
                                kv_indices_4d,
                                BLOCK_SIZE=BLOCK_SIZE,
                                seq_lengths=(int(N), int(N)),
                            )
                        else:
                            block_mask = _BlockMask.from_kv_blocks(
                                kv_num_blocks,
                                kv_indices_4d,
                                BLOCK_SIZE=BLOCK_SIZE,
                            )
                    except Exception:
                        block_mask = _BlockMask.from_kv_blocks(
                            kv_num_blocks,
                            kv_indices_4d,
                            BLOCK_SIZE=BLOCK_SIZE,
                        )
                    _BLOCK_MASK_CACHE[cache_key] = block_mask

                # Record BlockMask shape for debugging (do not pre-fallback)
                try:
                    bm_q, bm_k = getattr(block_mask, 'shape', (None, None))
                    telemetry['block_mask_shape'] = (int(bm_q), int(bm_k))
                except Exception:
                    telemetry['block_mask_shape'] = None

                flex_kwargs["block_mask"] = block_mask
                created_block_mask = True

        # If we couldn't create a block_mask, fall back to chunked sparse to avoid dense Flex OOM
        if not created_block_mask:
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
            _fail_fallback("no_blockmask_built")
            return self._fallback_chunked_sparse(x, pos, attn_mask)

        # Execute FlexAttention. It expects [B, H, N, Dh]
        try:
            out = flex_fn(q, k, v, **flex_kwargs)
            # If padded, slice back to original N before projection
            if 'do_pad' in locals() and do_pad:
                out = out[:, :, :orig_N, :]
                N = orig_N
            if isinstance(telemetry, dict):
                telemetry['used_flex_attention'] = True
                telemetry['fallback_reason'] = None
        except Exception as e:
            # Mark this module as failed once to avoid repeated OOM retries
            self._flex_failed_once = True
            if isinstance(telemetry, dict):
                telemetry['used_sparse_context'] = True
                telemetry['used_flex_attention'] = False
                try:
                    telemetry['flex_error'] = str(e)[:400]
                except Exception:
                    pass
            _dbg("FlexAttention raised, fallback", error=str(e)[:160])
            _fail_fallback("flex_call_exception")
            return self._fallback_chunked_sparse(x, pos, ctx)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _fallback_chunked_sparse(self, x: Tensor, pos, ctx: dict) -> Tensor:
        """Fallback path when FlexAttention is unavailable.
        Implements per-frame chunked SDPA with small per-chunk masks, avoiding NxN.
        """
        telemetry = None
        try:
            telemetry = ctx.get('telemetry', None) if isinstance(ctx, dict) else None
        except Exception:
            telemetry = None
        if isinstance(telemetry, dict):
            telemetry['took_fallback_chunked'] = True
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
