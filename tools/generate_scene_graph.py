#!/usr/bin/env python
"""
Generate a simple per-scene co-occurrence graph ("scene graph") for global attention masking.

This tool scans an image folder, computes lightweight global descriptors per image,
builds pairwise similarities, then exports:
  - adjacency: SxS 0/1 matrix (top-K neighbors per image; self-edges included)
  - bias: SxS float matrix (optional soft mask; negative penalties based on dissimilarity)
  - neighbors: mapping basename -> top-K neighbor basenames (optionally with weights)

Example:
  python vggt/tools/generate_scene_graph.py \
    --image_folder examples/kitchen/images \
    --output_json examples/scene_graph.json \
    --topk 4 --mutual true --with_bias true --metric downsample

The resulting JSON can be provided to run_ablation.py/test_co3d.py via --adjacency_json.

Notes:
  - This is a lightweight approximation (downsampled grayscale features). For best results,
    replace with stronger retrieval descriptors (e.g., MegaLoc) and keep the same JSON shape.
"""

import argparse
import json
import os
from glob import glob
from typing import List, Tuple

import numpy as np

try:
    from PIL import Image
except Exception as _e:
    Image = None


def list_images(folder: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> List[str]:
    paths: List[str] = []
    for e in exts:
        paths.extend(sorted(glob(os.path.join(folder, e))))
    return paths


def load_feature(path: str, size: int = 32) -> np.ndarray:
    """Very lightweight descriptor: grayscale downsample + per-vector standardization.
    Returns a 1D float32 vector of length size*size.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required: pip install pillow")
    img = Image.open(path).convert("L").resize((size, size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    v = x.reshape(-1)
    m, s = float(v.mean()), float(v.std())
    if s > 1e-6:
        v = (v - m) / s
    else:
        v = v - m
    return v


def pairwise_cosine(feats: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (SxD) -> (SxS) in [-1,1]."""
    X = feats.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T


def build_topk_adjacency(sim: np.ndarray, k: int, mutual: bool) -> np.ndarray:
    S = sim.shape[0]
    adj = np.zeros((S, S), dtype=bool)
    for i in range(S):
        # exclude self idx
        order = np.argsort(-sim[i])  # descending by similarity
        neigh = [j for j in order if j != i][:k]
        adj[i, i] = True
        adj[i, neigh] = True
    if mutual:
        adj = np.logical_and(adj, adj.T)
        # ensure diagonal stays True
        np.fill_diagonal(adj, True)
    return adj


def sim_to_bias(sim: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Map similarity [-1,1] to additive bias in [-alpha, 0], diag=0."""
    # Normalize to [0,1]
    sim01 = (sim + 1.0) * 0.5
    bias = -alpha * (1.0 - sim01)
    np.fill_diagonal(bias, 0.0)
    return bias.astype(np.float32)


def main():
    ap = argparse.ArgumentParser("Generate per-scene co-occurrence graph JSON")
    ap.add_argument("--image_folder", required=True, type=str)
    ap.add_argument("--output_json", required=True, type=str)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--mutual", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    ap.add_argument("--with_bias", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    ap.add_argument("--feature_size", type=int, default=32)
    args = ap.parse_args()

    paths = list_images(args.image_folder)
    if len(paths) == 0:
        raise ValueError(f"No images found in {args.image_folder}")

    bases = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    # Compute features
    feats = np.stack([load_feature(p, args.feature_size) for p in paths], axis=0)
    sim = pairwise_cosine(feats)

    # Build adjacency and (optional) bias
    adj = build_topk_adjacency(sim, k=max(args.topk, 0), mutual=args.mutual)
    bias = sim_to_bias(sim, alpha=2.0) if args.with_bias else None

    # Build neighbors mapping
    neighbors = {}
    for i, b in enumerate(bases):
        order = np.argsort(-sim[i])
        neigh = [bases[j] for j in order if j != i][: max(args.topk, 0)]
        neighbors[b] = neigh

    out = {
        "neighbors": neighbors,
        "adjacency": adj.astype(int).tolist(),
    }
    if bias is not None:
        out["bias"] = bias.tolist()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Wrote scene graph JSON to {args.output_json} (S={len(paths)}, topk={args.topk}, mutual={args.mutual})")


if __name__ == "__main__":
    main()

