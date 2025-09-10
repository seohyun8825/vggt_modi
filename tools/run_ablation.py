import argparse
import json
import os
import time
import sys
import re
import subprocess
from glob import glob

import torch
from importlib import import_module

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    p = argparse.ArgumentParser("VGGT ablation runner")
    p.add_argument("--image_folder", type=str, required=True, help="Path to folder containing only images")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save logs and metrics")
    p.add_argument("--mask_type", type=str, default="none", choices=["none", "topk", "soft"], help="Global attention masking mode")
    p.add_argument("--topk_neighbors", type=int, default=0, help="Top-K neighbors for masking")
    p.add_argument("--mutual", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True, help="Use mutual neighbors")
    p.add_argument("--soft_mask", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=False, help="Enable soft masking (additive bias)")
    p.add_argument("--mask_hub_tokens", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=False, help="Disable cross-frame hub/register attention")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="AMP dtype")
    p.add_argument("--pretrained", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True, help="Load pretrained weights from Hugging Face (ignored if --model_path is set)")
    p.add_argument("--hf_repo", type=str, default="facebook/VGGT-1B", help="Hugging Face repo id for pretrained weights")
    # Local checkpoint for both ablation and eval
    p.add_argument("--model_path", type=str, default="/workspace/toddler/vggt/model_tracker_fixed_e20.pt", help="Path to local VGGT checkpoint (.pt). If set, overrides --pretrained/--hf_repo.")
    # CO3D evaluation toggle & args
    p.add_argument("--eval_co3d", type=lambda x: str(x).lower() in ["1","true","yes"], default=False, help="Also run CO3D evaluation and record AUCs")
    p.add_argument("--co3d_dir", type=str, default=None, help="Path to CO3D dataset (images)")
    p.add_argument("--co3d_anno_dir", type=str, default=None, help="Path to CO3D annotations")
    p.add_argument("--seed", type=int, default=0, help="Seed for eval")
    p.add_argument("--fast_eval", type=lambda x: str(x).lower() in ["1","true","yes"], default=True, help="CO3D fast eval (10 seqs)")
    p.add_argument("--use_ba", type=lambda x: str(x).lower() in ["1","true","yes"], default=False, help="Enable BA in CO3D eval")
    p.add_argument("--adjacency_json", type=str, default=None, help="Path to per-scene adjacency JSON (MegaLoc output)")
    p.add_argument("--max_frames", type=int, default=0, help="Limit number of frames (use only first N images)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Collect images
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths = []
    for e in exts:
        image_paths.extend(sorted(glob(os.path.join(args.image_folder, e))))
    if args.max_frames and args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if len(image_paths) == 0:
        raise ValueError(f"No images found under {args.image_folder}")

    # FlexAttention availability probe (robust): check submodule import
    flex_available = False
    try:
        import importlib
        _fa_mod = importlib.import_module('torch.nn.attention.flex_attention')
        flex_available = hasattr(_fa_mod, 'flex_attention')
    except Exception:
        flex_available = False

    # Build model (prefer local checkpoint if provided)
    if args.model_path and os.path.isfile(args.model_path):
        model = VGGT().to(device)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
    elif args.pretrained:
        model = VGGT.from_pretrained(args.hf_repo).to(device)
    else:
        model = VGGT().to(device)

    # Inference/ablation runs: ensure eval mode to avoid checkpointing overhead
    model.eval()

    # Apply masking flags after construction/load so eval uses same "방법론"
    agg = model.aggregator
    agg.mask_type = args.mask_type
    agg.topk_neighbors = args.topk_neighbors
    agg.mutual = args.mutual
    agg.soft_mask = args.soft_mask or (args.mask_type == "soft")
    agg.mask_hub_tokens = args.mask_hub_tokens

    images = load_and_preprocess_images(image_paths).to(device)

    # If external adjacency is provided, set it for the next forward pass
    if args.adjacency_json and os.path.isfile(args.adjacency_json):
        try:
            model.aggregator.set_next_adjacency_from_json(args.adjacency_json, image_paths)
        except Exception:
            pass

    # Measure runtime and memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Optional warmup to exclude compile time (only when masking enabled)
    try:
        if agg.mask_type and agg.mask_type != "none":
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype) if device == "cuda" else torch.autocast("cpu", enabled=False):
                    _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp_dtype) if device == "cuda" else torch.autocast("cpu", enabled=False):
            _ = model(images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    peak_vram = (
        (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    )

    results = {
        "experiment": {
            "mask_type": args.mask_type,
            "topk_neighbors": args.topk_neighbors,
            "mutual": args.mutual,
            "soft_mask": args.soft_mask,
            "mask_hub_tokens": args.mask_hub_tokens,
            "weights": args.model_path if (args.model_path and os.path.isfile(args.model_path)) else args.hf_repo,
        },
        "runtime_s": total_time,
        "peak_VRAM_GB": peak_vram,
        "flex_attention_available": flex_available,
        "torch_version": torch.__version__,
        # Placeholders for quantitative metrics (computed elsewhere in eval pipeline)
        "pose_AUC": None,
        "depth_RMSE": None,
    }

    # Attach SDPA backend status for debugging Flash usage
    try:
        # Try legacy API name
        from torch.backends.cuda import sdp_kernel as _sdp
    except Exception:
        try:
            # Newer API may be named sdpa_kernel
            from torch.backends.cuda import sdpa_kernel as _sdp  # type: ignore
        except Exception:
            _sdp = None
    if _sdp is not None:
        try:
            results["sdp_backends"] = {
                "flash": bool(_sdp.is_flash_sdp_enabled()),
                "mem_efficient": bool(_sdp.is_mem_efficient_sdp_enabled()),
                "math": bool(_sdp.is_math_sdp_enabled()),
            }
        except Exception:
            pass

    # Attach sparsity info & telemetry if model computed it
    try:
        agg = model.aggregator
        sparse_info = getattr(agg, 'last_sparsity_info', None)
        if isinstance(sparse_info, dict):
            # Convert any tensors to plain types
            clean = {}
            for k, v in sparse_info.items():
                if hasattr(v, 'item'):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                clean[k] = v
            results["sparsity"] = clean
        telemetry = getattr(agg, 'last_telemetry', None)
        if isinstance(telemetry, dict):
            tel_clean = {}
            for k, v in telemetry.items():
                if hasattr(v, 'item'):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                elif isinstance(v, (tuple, list)):
                    try:
                        v = [int(x) if hasattr(x, 'item') else (int(x) if isinstance(x, int) else x) for x in v]
                    except Exception:
                        pass
                tel_clean[k] = v
            results["telemetry"] = tel_clean
    except Exception:
        pass

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Optionally run CO3D evaluation and append AUCs
    if args.eval_co3d:
        if not args.co3d_dir or not args.co3d_anno_dir:
            print("[WARN] --eval_co3d is set but --co3d_dir / --co3d_anno_dir are missing. Skipping CO3D eval.")
        else:
            # Resolve path to evaluation script
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "test_co3d.py"))
            json_out = os.path.join(args.output_dir, "co3d_eval.json")
            cmd = [
                sys.executable, script_path,
                "--model_path", args.model_path,
                "--co3d_dir", args.co3d_dir,
                "--co3d_anno_dir", args.co3d_anno_dir,
                "--seed", str(args.seed),
                "--mask_type", args.mask_type,
                "--topk_neighbors", str(args.topk_neighbors),
                "--mutual", str(int(args.mutual)),
                "--soft_mask", str(int(args.soft_mask)),
                "--mask_hub_tokens", str(int(args.mask_hub_tokens)),
                "--json_out", json_out,
            ]
            if args.adjacency_json:
                cmd.extend(["--adjacency_json", args.adjacency_json])
            if args.fast_eval:
                cmd.append("--fast_eval")
            if args.use_ba:
                cmd.append("--use_ba")

            print(f"[INFO] Running CO3D eval: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(proc.stdout)
            if proc.returncode != 0:
                print("[ERROR] CO3D eval failed:", proc.stderr)
            # Try to read JSON results first
            auc = None
            if os.path.isfile(json_out):
                try:
                    with open(json_out, "r") as jf:
                        j = json.load(jf)
                    auc = j.get("mean", {})
                except Exception as e:
                    print(f"[WARN] Failed to parse {json_out}: {e}")
            # Fallback: parse stdout
            if not auc:
                m = re.search(r"Mean AUC:\s*([0-9.]+)\s*\(AUC@30\),\s*([0-9.]+)\s*\(AUC@15\),\s*([0-9.]+)\s*\(AUC@5\),\s*([0-9.]+)\s*\(AUC@3\)", proc.stdout)
                if m:
                    auc = {
                        "Auc_30": float(m.group(1)),
                        "Auc_15": float(m.group(2)),
                        "Auc_5": float(m.group(3)),
                        "Auc_3": float(m.group(4)),
                    }
            if auc:
                results["pose_AUC"] = auc.get("Auc_30")
                results["co3d_eval"] = auc
                with open(metrics_path, "w") as f:
                    json.dump(results, f, indent=2)
                print("[INFO] CO3D AUC appended to metrics.json:", json.dumps(auc, indent=2))
            else:
                print("[WARN] Could not extract AUC from CO3D eval.")


if __name__ == "__main__":
    main()
