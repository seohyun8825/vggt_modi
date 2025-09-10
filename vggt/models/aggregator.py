# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.attention import GlobalSparseAttention
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        # Masked global attention config
        mask_type: str = "none",  # ["none", "topk", "soft"]
        topk_neighbors: int = 0,
        mutual: bool = True,
        soft_mask: bool = False,
        mask_hub_tokens: bool = False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    attn_class=GlobalSparseAttention,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        # Masking configuration
        self.mask_type = mask_type
        self.topk_neighbors = int(topk_neighbors)
        self.mutual = bool(mutual)
        # allow both flags; if mask_type=="soft" and soft_mask is False, enable soft
        self.soft_mask = bool(soft_mask) or (mask_type == "soft")
        self.mask_hub_tokens = bool(mask_hub_tokens)
        # placeholders for externally-provided adjacency/bias (per-call override)
        self._next_adjacency: Optional[torch.Tensor] = None  # (S, S) bool or float weights
        self._next_bias: Optional[torch.Tensor] = None       # (S, S) float additive bias

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # Build sparse attention context (no NxN allocation)
        # Frame-level adjacency and bias (S,S)
        adj, frame_bias = self._compute_frame_adjacency(S, device=tokens.device)

        # Token -> frame mapping and hub tokens
        N = S * P
        frame_ids = torch.arange(S, device=tokens.device).repeat_interleave(P)  # (N,)
        t_in_frame = torch.arange(P, device=tokens.device).repeat(S)            # (N,)
        is_register = (t_in_frame >= 1) & (t_in_frame < self.patch_start_idx)
        # Treat camera(token 0) and register tokens as hubs for optional masking
        is_hub = is_register | (t_in_frame == 0)

        # Telemetry to detect flex usage inside attention kernel
        telemetry: Dict[str, Any] = {"used_flex_attention": False, "used_sparse_context": False}

        sparse_ctx = {
            'adj': adj,                      # (S,S) bool or None
            'frame_ids': frame_ids,          # (N,)
            'is_hub': is_hub,                # (N,)
            'mask_hub_tokens': self.mask_hub_tokens,
            'soft_mask': (self.mask_type == "soft") or self.soft_mask,
            'frame_bias': frame_bias,        # (S,S) float or None (for soft)
            'telemetry': telemetry,
        }

        # Precompute sparsity metrics for logging
        total_pairs = int(N) * int(N)
        hubs_per_frame = int(self.patch_start_idx)  # 1 camera + num_register_tokens
        nonhub_per_frame = max(int(P) - hubs_per_frame, 0)

        if adj is None:
            # All frames connected (soft-only or no topk)
            if self.mask_hub_tokens:
                # intra-frame fully allowed + cross-frame nonhub only
                allowed_intra = S * (P * P)
                allowed_cross = S * (S - 1) * (nonhub_per_frame * nonhub_per_frame)
                allowed_pairs = allowed_intra + allowed_cross
            else:
                allowed_pairs = total_pairs
            adj_density = 1.0
        else:
            # Count directed frame pairs (i,j)
            adj_bool = adj.to(torch.bool)
            adj_density = (adj_bool.float().mean()).item()
            # include self-edges explicitly (should already be True)
            # Sum over all (i,j)
            num_allowed = 0
            for i in range(S):
                for j in range(S):
                    if i == j:
                        num_allowed += (P * P)
                    elif adj_bool[i, j]:
                        if self.mask_hub_tokens:
                            num_allowed += (nonhub_per_frame * nonhub_per_frame)
                        else:
                            num_allowed += (P * P)
            allowed_pairs = int(num_allowed)

        sparsity = float(allowed_pairs) / float(total_pairs) if total_pairs > 0 else 1.0
        self.last_sparsity_info = {
            "S": int(S),
            "P": int(P),
            "N": int(N),
            "hubs_per_frame": hubs_per_frame,
            "nonhub_per_frame": nonhub_per_frame,
            "total_pairs": total_pairs,
            "allowed_pairs": int(allowed_pairs),
            "sparsity": float(sparsity),
            "adj_density": float(adj_density),
            "mask_type": self.mask_type,
            "topk_neighbors": int(self.topk_neighbors),
            "mutual": bool(self.mutual),
            "soft_mask": bool((self.mask_type == "soft") or self.soft_mask),
            "mask_hub_tokens": bool(self.mask_hub_tokens),
            "used_flex_attention": False,  # to be updated by telemetry after forward
        }

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            # Only pass sparse_ctx when masking is actually active; otherwise None keeps baseline path
            masking_active = (
                (adj is not None)
                or self.mask_hub_tokens
                or ((self.mask_type == "soft") or self.soft_mask)
            )

            ctx = sparse_ctx if masking_active else None

            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, ctx, use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=ctx)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        # Record telemetry (whether flex attention was actually used)
        if hasattr(self, "last_sparsity_info") and isinstance(self.last_sparsity_info, dict):
            self.last_sparsity_info["used_flex_attention"] = bool(telemetry.get("used_flex_attention", False))
            self.last_sparsity_info["used_sparse_context"] = bool(telemetry.get("used_sparse_context", False))

        return tokens, global_idx, intermediates

    # --- Masking utilities ---
    def set_next_adjacency(self, adjacency: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        """
        Optionally set an external adjacency or bias to be used for the next forward pass only.
        - adjacency: (S, S) bool or float weights; True/positive means allowed/strong.
        - bias: (S, S) float additive attention bias (e.g., from similarities).
        After one use, these will be cleared.
        """
        self._next_adjacency = adjacency
        self._next_bias = bias

    def _compute_frame_adjacency(self, S: int, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (adjacency_bool_or_weights, bias_float) both of shape (S, S) or (None, None) if no masking.
        """
        if self.mask_type == "none" or S == 1:
            # No mask
            return None, None

        # If provided externally, use and clear
        if self._next_adjacency is not None or self._next_bias is not None:
            adj = self._next_adjacency
            bias = self._next_bias
            self._next_adjacency, self._next_bias = None, None
            # Ensure diagonal allowed
            if adj is not None:
                adj = adj.to(device)
                if adj.dtype == torch.bool:
                    adj = adj.clone()
                    adj.fill_diagonal_(True)
                else:
                    # weights: enforce self weight = max
                    diag_val = adj.max() if adj.numel() > 0 else torch.tensor(0.0, device=device)
                    adj = adj.clone()
                    adj.fill_diagonal_(diag_val)
            if bias is not None:
                bias = bias.to(device)
            return adj, bias

        # Default adjacency from index-based neighbors (fallback when no external graph is given)
        # Build directed top-k by temporal index distance
        if self.mask_type in ("topk", "soft") or self.soft_mask:
            K = max(int(self.topk_neighbors), 0)
            if K > 0:
                idx = torch.arange(S, device=device)
                adj = torch.zeros(S, S, dtype=torch.bool, device=device)
                for i in range(S):
                    # choose neighbors by sorted absolute index distance excluding itself
                    d = torch.abs(idx - i)
                    order = torch.argsort(d)
                    neigh = order[1 : 1 + K]
                    adj[i, i] = True
                    adj[i, neigh] = True
                if self.mutual:
                    adj = adj & adj.t()
            else:
                # No hard neighbor limit -> allow all pairs (handled via soft bias only)
                adj = None
        else:
            adj = None

        bias = None
        if self.mask_type == "soft" or self.soft_mask:
            # derive a simple similarity-based bias: higher similarity -> 0, lower -> negative
            # Here we approximate similarity by inverse of index distance
            i = torch.arange(S, device=device).view(-1, 1)
            j = torch.arange(S, device=device).view(1, -1)
            dist = (i - j).abs().clamp(min=0)
            sim = 1.0 / (1.0 + dist.float())  # in (0,1]
            sim = sim / sim.max()  # normalize to (0,1]
            # Convert to additive bias in logit space: map sim in (0,1] to bias in [-alpha, 0]
            alpha = 2.0  # strength of down-weighting
            bias = -alpha * (1.0 - sim)
            # Ensure self is zero bias
            bias.fill_diagonal_(0.0)
            if adj is not None and self.topk_neighbors > 0:
                # if topk also requested, zero out bias for non-adj pairs by setting strong negative
                not_adj = ~adj
                bias = bias.masked_fill(not_adj, float('-inf'))

        return adj, bias

    def _build_global_attn_bias(
        self, B: int, S: int, P: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Build an additive attention bias of shape [1, 1, N, N] where N=S*P, or None if no masking.
        - Hard mask: entries set to -inf for disallowed pairs
        - Soft mask: entries store negative penalties based on pairwise similarity
        Also supports masking hub/register tokens from cross-frame interactions.
        """
        adj, frame_bias = self._compute_frame_adjacency(S, device)
        if adj is None and frame_bias is None:
            return None

        N = S * P
        # Map token -> frame index and identify register tokens per frame
        frame_ids = torch.arange(S, device=device).repeat_interleave(P)
        # token indices within a frame [0..P-1]
        t_in_frame = torch.arange(P, device=device).repeat(S)
        num_register_tokens = self.patch_start_idx - 1  # indices [1 .. patch_start_idx-1]
        is_register = (t_in_frame >= 1) & (t_in_frame < self.patch_start_idx)

        # Start with allowed pairs from adjacency
        if adj is None:
            allowed = torch.ones(S, S, dtype=torch.bool, device=device)
        else:
            allowed = adj.to(torch.bool)
        allowed_pairs = allowed[frame_ids[:, None], frame_ids[None, :]]  # (N, N)

        # Apply hub/register masking across frames if requested
        if self.mask_hub_tokens:
            cross = frame_ids[:, None] != frame_ids[None, :]
            hub_cross = cross & (is_register[:, None] | is_register[None, :])
            allowed_pairs = allowed_pairs & (~hub_cross)

        # Build additive bias
        attn_bias = torch.zeros(N, N, dtype=torch.float32, device=device)
        # Hard mask for disallowed pairs
        if not (self.mask_type == "soft" or self.soft_mask):
            attn_bias = attn_bias.masked_fill(~allowed_pairs, float('-inf'))
        else:
            # Soft bias from frame-level similarity (broadcast to tokens)
            if frame_bias is None:
                # default small penalty for non-adjacent frames
                fb = torch.zeros(S, S, dtype=torch.float32, device=device)
            else:
                fb = frame_bias.to(torch.float32)
            token_bias = fb[frame_ids[:, None], frame_ids[None, :]]  # (N, N)
            attn_bias = attn_bias + token_bias
            # if also not allowed by adjacency (when combining topk + soft), enforce hard mask
            if adj is not None and self.topk_neighbors > 0:
                attn_bias = attn_bias.masked_fill(~allowed_pairs, float('-inf'))

            if self.mask_hub_tokens:
                # ensure hub cross-frame is strictly masked
                cross = frame_ids[:, None] != frame_ids[None, :]
                hub_cross = cross & (is_register[:, None] | is_register[None, :])
                attn_bias = attn_bias.masked_fill(hub_cross, float('-inf'))

        # Reshape to [1, 1, N, N] for broadcast across batch and heads
        attn_bias = attn_bias.view(1, 1, N, N)
        return attn_bias

    # --- External adjacency helpers (for MegaLoc integration) ---
    def set_next_adjacency_from_json(self, json_path: str, image_paths: list[str]):
        """
        Load an external co-occurrence graph from JSON for the next forward pass.
        Accepts either:
          - {"adjacency": [[0/1,...], ...]} shape SxS
          - {"bias": [[float,...], ...]} shape SxS (soft mask bias)
          - {"neighbors": {"basename": ["neighbor_basename", ...], ...}} (optionally weighted)
            • Also supports {"neighbors": {"basename": {"nbr": weight, ...}}}
        image_paths are used to map basenames to indices 0..S-1 in order.
        """
        try:
            import json, os
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return

        S = len(image_paths)
        # Map basenames
        bases = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
        index_of = {b: i for i, b in enumerate(bases)}

        adj_tensor = None
        bias_tensor = None

        if isinstance(data, dict) and 'adjacency' in data:
            import torch as _torch
            A = data['adjacency']
            try:
                adj_tensor = _torch.tensor(A, dtype=_torch.bool, device=self.camera_token.device)
            except Exception:
                pass
        if isinstance(data, dict) and 'bias' in data:
            import torch as _torch
            B = data['bias']
            try:
                bias_tensor = _torch.tensor(B, dtype=_torch.float32, device=self.camera_token.device)
            except Exception:
                pass
        if isinstance(data, dict) and 'neighbors' in data and (adj_tensor is None or bias_tensor is None):
            import torch as _torch
            nbrs = data['neighbors']
            # Initialize only if missing
            if adj_tensor is None:
                adj_tensor = _torch.zeros(S, S, dtype=_torch.bool, device=self.camera_token.device)
            if bias_tensor is None and self.soft_mask:
                bias_tensor = _torch.zeros(S, S, dtype=_torch.float32, device=self.camera_token.device)
            for bname, neigh in nbrs.items():
                if bname not in index_of:
                    continue
                i = index_of[bname]
                # neigh can be list[str] or dict[str->float]
                if isinstance(neigh, dict):
                    for nb, w in neigh.items():
                        j = index_of.get(nb, None)
                        if j is None:
                            continue
                        adj_tensor[i, j] = True
                        if bias_tensor is not None:
                            try:
                                bias_tensor[i, j] = float(w)
                            except Exception:
                                pass
                elif isinstance(neigh, list):
                    for nb in neigh:
                        j = index_of.get(nb, None)
                        if j is None:
                            continue
                        adj_tensor[i, j] = True

        # Enforce self-connections
        if adj_tensor is not None:
            adj_tensor = adj_tensor.clone()
            adj_tensor.fill_diagonal_(True)

        # Set for next call
        self.set_next_adjacency(adjacency=adj_tensor, bias=bias_tensor)


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
