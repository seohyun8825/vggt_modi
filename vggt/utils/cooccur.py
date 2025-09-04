"""
Utilities to integrate an external image co-occurrence graph (e.g., from MegaLoc).

Typical workflow:
1) Precompute a neighbor list for each image (e.g., via MegaLoc retrieval) and save
   a JSON like: {"image_id": {"neighbors": ["id_b", "id_c", ...], "scores": [0.98, 0.92, ...]}, ...}
2) At runtime, map the current scene's images to their IDs and build an adjacency matrix.
3) Provide that adjacency (and optional bias) to the Aggregator via set_next_adjacency().

This module provides simple helpers to load such a JSON and build (S,S) tensors.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional

import torch


def load_neighbor_graph(json_path: str) -> Dict[str, Dict[str, List]]:
    """
    Load a neighbor graph JSON file.

    Expected schema per image_id:
      {
        "neighbors": ["id_b", "id_c", ...],
        "scores": [0.98, 0.92, ...]  # optional
      }
    """
    with open(json_path, "r") as f:
        return json.load(f)


def build_adjacency(
    ids: List[str],
    graph: Dict[str, Dict[str, List]],
    topk: int = 0,
    mutual: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a boolean adjacency (S,S) where True means allowed attention across frames.
    If topk>0, restrict to top-k neighbors; otherwise connect all pairs.
    If mutual is True, keep edges only if both sides select each other.
    """
    S = len(ids)
    device = device or torch.device("cpu")
    id_to_idx = {i: t for t, i in enumerate(ids)}
    adj = torch.zeros(S, S, dtype=torch.bool, device=device)
    for i, iid in enumerate(ids):
        adj[i, i] = True
        if iid not in graph:
            continue
        nbrs = graph[iid].get("neighbors", [])
        if topk > 0:
            nbrs = nbrs[:topk]
        for n in nbrs:
            if n in id_to_idx:
                adj[i, id_to_idx[n]] = True
    if mutual and topk > 0:
        adj = adj & adj.t()
    return adj


def build_bias_from_scores(
    ids: List[str],
    graph: Dict[str, Dict[str, List]],
    epsilon: float = 1e-6,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build an additive attention bias (S,S) from retrieval scores, mapping high scores to 0
    and low scores to negative penalties.
    """
    S = len(ids)
    device = device or torch.device("cpu")
    id_to_idx = {i: t for t, i in enumerate(ids)}
    bias = torch.zeros(S, S, dtype=torch.float32, device=device)
    # collect per-row max for normalization
    for i, iid in enumerate(ids):
        scores = graph.get(iid, {}).get("scores", [])
        nbrs = graph.get(iid, {}).get("neighbors", [])
        if not scores or not nbrs:
            continue
        smax = max(scores) if scores else 1.0
        for n, sc in zip(nbrs, scores):
            if n in id_to_idx:
                j = id_to_idx[n]
                # normalize and convert similarity to penalty
                sim = sc / (smax + epsilon)
                bias[i, j] = -2.0 * (1.0 - sim)  # in [-2, 0]
    # self bias = 0
    bias.fill_diagonal_(0.0)
    return bias

