#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)
    # Also stream stdout for visibility
    print(proc.stdout)


def read_metrics(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser("Quick runtime/VRAM bench: baseline vs K=2")
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--output_root", default="results/bench")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_ablation = os.path.join(repo_root, "vggt", "tools", "run_ablation.py")

    # 1) Baseline (no mask)
    out_base = os.path.join(args.output_root, "baseline_noeval")
    os.makedirs(out_base, exist_ok=True)
    cmd_base = [
        args.python, run_ablation,
        "--image_folder", args.image_folder,
        "--output_dir", out_base,
        "--mask_type", "none",
        "--eval_co3d", "0",
    ]
    run(cmd_base)
    m_base = read_metrics(os.path.join(out_base, "metrics.json"))

    # 2) K=2 (mutual)
    out_k2 = os.path.join(args.output_root, "k2_noeval")
    os.makedirs(out_k2, exist_ok=True)
    cmd_k2 = [
        args.python, run_ablation,
        "--image_folder", args.image_folder,
        "--output_dir", out_k2,
        "--mask_type", "topk",
        "--topk_neighbors", "2",
        "--mutual", "true",
        "--eval_co3d", "0",
    ]
    run(cmd_k2)
    m_k2 = read_metrics(os.path.join(out_k2, "metrics.json"))

    def pick(d, keys, default=None):
        out = {}
        for k in keys:
            out[k] = d.get(k, default)
        return out

    print("\n=== Summary ===")
    print("Baseline:")
    print(json.dumps(pick(m_base, [
        "runtime_s", "peak_VRAM_GB", "torch_version", "flex_attention_available", "sdp_backends", "sparsity"
    ]), indent=2))

    print("\nK=2 (mutual):")
    print(json.dumps(pick(m_k2, [
        "runtime_s", "peak_VRAM_GB", "torch_version", "flex_attention_available", "sdp_backends", "sparsity"
    ]), indent=2))

    print("\nTip: For larger speed deltas, try more frames (bigger N) or provide --adjacency_json with a strong scene graph.")


if __name__ == "__main__":
    main()

