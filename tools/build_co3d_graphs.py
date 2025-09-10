#!/usr/bin/env python
"""
Build per-sequence adjacency JSONs for CO3D evaluation using MegaLoc (or fallback).

Layout:
  {out_dir}/{category}/{seq_name}.json

Each JSON contains:
{
  "images": [list of selected image paths in order],
  "topk": int,
  "mutual": bool,
  "score_type": "cosine_*",
  "adjacency": SxS 0/1 list,
  "bias": (optional) SxS float list
}
"""

import argparse
import gzip
import json
import os
import random
from typing import List, Tuple

import numpy as np


def _load_megaloc(device: str):
    try:
        import torch
        try:
            print("[INFO] Loading MegaLoc via torch.hub gmberton/MegaLoc@get_trained_model")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            model.eval().to(device)
            return model
        except Exception as e_main:
            print(f"[WARN] MegaLoc@get_trained_model failed: {e_main}")
        try:
            print("[INFO] Loading MegaLoc via torch.hub gmberton/MegaLoc@megaloc")
            model = torch.hub.load("gmberton/MegaLoc", "megaloc", trust_repo=True)
            model.eval().to(device)
            return model
        except Exception as e_alt:
            print(f"[WARN] MegaLoc@megaloc failed: {e_alt}")
        return None
    except Exception:
        return None


def _compute_desc_megaloc(model, paths: List[str], device: str) -> np.ndarray:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    tfm = T.Compose([
        T.Resize(512), T.CenterCrop(512), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        bs = 16
        for i in range(0, len(paths), bs):
            ims = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
            x = torch.stack(ims, 0).to(device)
            d = model(x)
            if isinstance(d, (list, tuple)):
                d = d[0]
            d = torch.nn.functional.normalize(d, p=2, dim=-1)
            out.append(d.detach().cpu())
    return torch.cat(out, 0).numpy()


def _compute_desc_resnet50(paths: List[str], device: str) -> np.ndarray:
    import torch
    import torchvision.transforms as T
    import torchvision.models as M
    from torch import nn
    from PIL import Image
    weights = M.ResNet50_Weights.DEFAULT
    model = M.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    tfm = weights.transforms()
    out = []
    with torch.no_grad():
        bs = 16
        for i in range(0, len(paths), bs):
            ims = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
            x = torch.stack(ims, 0).to(device)
            f = model(x)
            f = torch.nn.functional.normalize(f, p=2, dim=-1)
            out.append(f.detach().cpu())
    return torch.cat(out, 0).numpy()


def _compute_desc_fallback(paths: List[str]) -> np.ndarray:
    from PIL import Image
    D = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((32, 32))
        a = np.asarray(im, dtype=np.float32) / 255.0
        v = a.reshape(-1)
        m, s = float(v.mean()), float(v.std())
        v = (v - m) / (s + 1e-6)
        v = v / (np.linalg.norm(v) + 1e-8)
        D.append(v)
    return np.stack(D, 0)


def build_topk(desc: np.ndarray, k: int, mutual: bool) -> Tuple[np.ndarray, np.ndarray]:
    sim = desc @ desc.T
    np.fill_diagonal(sim, -np.inf)
    S = sim.shape[0]
    adj = np.zeros((S, S), dtype=np.uint8)
    if k > 0:
        kk = min(k, S - 1)
        idx = np.argpartition(-sim, kth=kk-1, axis=1)[:, :kk]
        for i in range(S):
            adj[i, idx[i]] = 1
        if mutual:
            adj = (adj & adj.T).astype(np.uint8)
    else:
        adj[:] = 1
        np.fill_diagonal(adj, 1)
    return sim, adj


def build_hybrid(desc: np.ndarray, k: int, mutual: bool, window_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid neighbors: union of temporal window ±W and retrieval to fill up to K.
    Keeps out-degree <= K before mutual filtering.
    """
    sim = desc @ desc.T
    np.fill_diagonal(sim, -np.inf)
    S = sim.shape[0]
    adj = np.zeros((S, S), dtype=np.uint8)
    k = max(0, int(k))
    W = max(0, int(window_w))
    # 1) Add temporal window neighbors
    if W > 0:
        for i in range(S):
            j0 = max(0, i - W)
            j1 = min(S - 1, i + W)
            if j0 <= j1:
                adj[i, j0 : j1 + 1] = 1
            adj[i, i] = 0  # exclude self for now
    # 2) Fill with retrieval until per-row count reaches K
    per_row = adj.sum(axis=1)
    target = np.minimum(k, S - 1)
    # mask out already-selected and self
    sim_masked = sim.copy()
    for i in range(S):
        # Exclude already selected and self
        sim_masked[i, i] = -np.inf
        if per_row[i] > 0:
            sim_masked[i, adj[i] > 0] = -np.inf
        need = int(max(0, target - per_row[i]))
        if need > 0:
            # pick top-need
            idx = np.argpartition(-sim_masked[i], kth=min(need - 1, S - 2), axis=0)[:need]
            adj[i, idx] = 1
    # Restore self if desired downstream
    if mutual:
        adj = (adj & adj.T).astype(np.uint8)
    else:
        # leave as directed; we'll set diagonal later when consumed by aggregator
        pass
    return sim, adj


def sim_to_bias(sim: np.ndarray, temperature: float = 0.25, alpha: float = 1.0) -> np.ndarray:
    s = (sim + 1.0) * 0.5
    s = np.clip(s, 0.0, 1.0)
    bias = alpha * (s - 1.0) / max(1e-6, temperature)
    np.fill_diagonal(bias, 0.0)
    return bias.astype(np.float32)


def main():
    ap = argparse.ArgumentParser("Build per-sequence CO3D adjacency JSONs")
    ap.add_argument("--co3d_dir", required=True)
    ap.add_argument("--co3d_anno_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast_eval", action="store_true", help="Match eval: only 10 sequences per category")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", type=str, default="auto", choices=["auto", "resnet50", "fallback"],
                    help="Descriptor backbone for graph build")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--soft_bias", action="store_true")
    ap.add_argument("--bias_temperature", type=float, default=0.25)
    ap.add_argument("--bias_alpha", type=float, default=1.0)
    ap.add_argument("--window_w", type=int, default=0, help="Temporal window size W for hybrid union (±W)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Categories: mirror test_co3d.py for now (apple only unless debug)
    categories = ["apple"]

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare descriptor backend
    score_type = "cosine"
    model = None
    if args.backbone == "auto":
        model = _load_megaloc(args.device)
        if model is not None:
            score_type = "cosine_megaloc"
        else:
            score_type = "cosine_resnet50"
    elif args.backbone == "resnet50":
        score_type = "cosine_resnet50"
    else:
        score_type = "cosine_fallback"

    for category in categories:
        anno_path = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")
        try:
            with gzip.open(anno_path, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"[WARN] Missing annotation for {category}; skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names) >= 10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            # Gather all image paths in this sequence
            all_image_paths = [os.path.join(args.co3d_dir, item["filepath"]) for item in seq_data]
            if len(all_image_paths) < 2:
                continue
            # Sample num_frames deterministically
            ids = np.random.choice(len(all_image_paths), args.num_frames, replace=False)
            ids = np.sort(ids)
            image_names = [all_image_paths[i] for i in ids]

            # Compute descriptors
            try:
                if args.backbone == "fallback":
                    desc = _compute_desc_fallback(image_names)
                elif args.backbone == "resnet50" or (args.backbone == "auto" and model is None):
                    desc = _compute_desc_resnet50(image_names, args.device)
                else:
                    # auto + megaloc available
                    desc = _compute_desc_megaloc(model, image_names, args.device)
            except Exception as e:
                print(f"[WARN] Descriptor build failed for {category}/{seq_name}: {e}; falling back to simple")
                desc = _compute_desc_fallback(image_names)
                score_type = "cosine_fallback"

            if args.window_w and args.window_w > 0:
                sim, adj = build_hybrid(desc, k=max(0, int(args.topk)), mutual=bool(args.mutual), window_w=args.window_w)
            else:
                sim, adj = build_topk(desc, k=max(0, int(args.topk)), mutual=bool(args.mutual))
            out = {
                "images": image_names,
                "topk": int(args.topk),
                "mutual": bool(args.mutual),
                "score_type": score_type,
                "adjacency": adj.astype(int).tolist(),
            }
            if args.soft_bias:
                out["bias"] = sim_to_bias(sim, temperature=float(args.bias_temperature), alpha=float(args.bias_alpha)).tolist()
            if args.window_w and args.window_w > 0:
                out["window_w"] = int(args.window_w)

            out_cat = os.path.join(args.out_dir, category)
            os.makedirs(out_cat, exist_ok=True)
            out_path = os.path.join(out_cat, f"{seq_name}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[OK] Wrote {out_path} (S={len(image_names)}, topk={args.topk}, mutual={bool(args.mutual)}, score={score_type})")


if __name__ == "__main__":
    main()
