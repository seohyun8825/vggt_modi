#!/usr/bin/env python
"""
Build a co-occurrence graph via MegaLoc (or a lightweight fallback) and save as adjacency JSON.

Outputs JSON schema:
{
  "images": [list of image paths],
  "topk": int,
  "mutual": bool,
  "score_type": "cosine",
  "adjacency": SxS 0/1 list,
  "bias": (optional) SxS float list (additive bias)
}

If MegaLoc is unavailable, falls back to simple normalized RGB downsample descriptors.
"""

import argparse
import json
import os
import glob
import sys
from typing import List, Tuple

import numpy as np


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(folder, e))))
    if not files:
        raise FileNotFoundError(f"No images found under {folder}")
    return files


def _load_megaloc(device: str):
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[WARN] torch not available: {e}")
        return None
    try:
        import torch
        # Official: gmberton/MegaLoc get_trained_model
        try:
            print("[INFO] Trying torch.hub load: gmberton/MegaLoc@get_trained_model")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            model.eval().to(device)
            return model
        except Exception as e_main:
            print(f"[WARN] MegaLoc@get_trained_model load failed: {e_main}")
        # Fallback: alternative entry name
        try:
            print("[INFO] Trying torch.hub load: gmberton/MegaLoc@megaloc")
            model = torch.hub.load("gmberton/MegaLoc", "megaloc", trust_repo=True)
            model.eval().to(device)
            return model
        except Exception as e_alt:
            print(f"[WARN] MegaLoc@megaloc load failed: {e_alt}")
        return None
    except Exception as e_all:
        print(f"[WARN] MegaLoc load attempts failed: {e_all}")
        return None


def _compute_desc_megaloc(model, paths: List[str], device: str) -> np.ndarray:
    import torch
    from PIL import Image
    import torchvision.transforms as T

    tfm = T.Compose([
        T.Resize(512), T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    descs = []
    with torch.no_grad():
        bs = 16
        for i in range(0, len(paths), bs):
            ims = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
            x = torch.stack(ims, 0).to(device)
            # Adapt to model API; try common names
            try:
                d = model(x)
            except Exception:
                # try attribute
                if hasattr(model, "extract_global"):
                    d = model.extract_global(x)
                else:
                    raise
            if isinstance(d, (list, tuple)):
                d = d[0]
            d = torch.nn.functional.normalize(d, p=2, dim=-1)
            descs.append(d.detach().cpu())
    return torch.cat(descs, 0).numpy()


def _compute_desc_fallback(paths: List[str]) -> np.ndarray:
    from PIL import Image
    D = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((32, 32))
        a = np.asarray(im, dtype=np.float32) / 255.0
        v = a.reshape(-1)
        m, s = float(v.mean()), float(v.std())
        v = (v - m) / (s + 1e-6)
        # L2 norm
        v = v / (np.linalg.norm(v) + 1e-8)
        D.append(v)
    return np.stack(D, 0)


def _compute_desc_resnet50(paths: List[str], device: str) -> np.ndarray:
    import torch
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as M
    from torch import nn

    weights = M.ResNet50_Weights.DEFAULT
    model = M.resnet50(weights=weights)
    # Replace final FC with identity to get 2048-d pooled features
    model.fc = nn.Identity()
    model.eval().to(device)

    tfm = T.Compose([
        weights.transforms(),  # includes resize/crop/normalize
    ])

    descs = []
    with torch.no_grad():
        bs = 16
        for i in range(0, len(paths), bs):
            ims = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
            x = torch.stack(ims, 0).to(device)
            f = model(x)  # [B, 2048]
            f = torch.nn.functional.normalize(f, p=2, dim=-1)
            descs.append(f.detach().cpu())
    return torch.cat(descs, 0).numpy()


def build_topk(desc: np.ndarray, k: int, mutual: bool) -> Tuple[np.ndarray, np.ndarray]:
    sim = desc @ desc.T
    np.fill_diagonal(sim, -np.inf)
    S = sim.shape[0]
    adj = np.zeros((S, S), dtype=np.uint8)
    if k > 0:
        idx = np.argpartition(-sim, kth=min(k-1, S-2), axis=1)[:, :k]
        for i in range(S):
            adj[i, idx[i]] = 1
        if mutual:
            adj = (adj & adj.T).astype(np.uint8)
    else:
        adj[:] = 1
        np.fill_diagonal(adj, 1)
    return sim, adj


def sim_to_bias(sim: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    # Normalize to [0,1] assuming cosine in [-1,1]
    s = (sim + 1.0) * 0.5
    s = np.clip(s, 0.0, 1.0)
    bias = (s - 1.0) / max(1e-6, temperature)  # 0 for diag, negative elsewhere
    np.fill_diagonal(bias, 0.0)
    return bias.astype(np.float32)


def main():
    ap = argparse.ArgumentParser("Build MegaLoc co-occurrence graph JSON")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--soft_bias", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", type=str, default="auto", choices=["auto", "resnet50", "fallback"], help="Descriptor backbone: auto(MegaLoc->ResNet50), resnet50, or simple fallback")
    args = ap.parse_args()

    paths = list_images(args.image_dir)
    score_type = "cosine"
    if args.backbone == "resnet50":
        print("[INFO] Using torchvision ResNet50 global descriptors")
        desc = _compute_desc_resnet50(paths, device=args.device)
        score_type = "cosine_resnet50"
    elif args.backbone == "fallback":
        print("[INFO] Using lightweight fallback descriptors")
        desc = _compute_desc_fallback(paths)
        score_type = "cosine_fallback"
    else:
        # auto: try MegaLoc, fall back to ResNet50, then simple fallback
        model = _load_megaloc(args.device)
        if model is not None:
            try:
                desc = _compute_desc_megaloc(model, paths, device=args.device)
                score_type = "cosine_megaloc"
            except Exception as e:
                print(f"[WARN] MegaLoc inference failed ({e}); trying ResNet50.")
                try:
                    desc = _compute_desc_resnet50(paths, device=args.device)
                    score_type = "cosine_resnet50"
                except Exception as e2:
                    print(f"[WARN] ResNet50 fallback failed ({e2}); using simple fallback.")
                    desc = _compute_desc_fallback(paths)
                    score_type = "cosine_fallback"
        else:
            try:
                desc = _compute_desc_resnet50(paths, device=args.device)
                score_type = "cosine_resnet50"
            except Exception as e2:
                print(f"[WARN] ResNet50 fallback failed ({e2}); using simple fallback.")
                desc = _compute_desc_fallback(paths)
                score_type = "cosine_fallback"

    sim, adj = build_topk(desc, k=max(0, int(args.topk)), mutual=bool(args.mutual))

    out = {
        "images": paths,
        "topk": int(args.topk),
        "mutual": bool(args.mutual),
        "score_type": score_type,
        "adjacency": adj.astype(int).tolist(),
    }
    if args.soft_bias:
        out["bias"] = sim_to_bias(sim).tolist()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Wrote graph JSON: {args.out} (S={len(paths)}, topk={args.topk}, mutual={bool(args.mutual)}, score_type={score_type})")


if __name__ == "__main__":
    main()
