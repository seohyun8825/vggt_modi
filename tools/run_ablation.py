import argparse
import json
import os
import time
from glob import glob

import torch

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
    p.add_argument("--pretrained", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True, help="Load pretrained weights from Hugging Face")
    p.add_argument("--hf_repo", type=str, default="facebook/VGGT-1B", help="Hugging Face repo id for pretrained weights")
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
    if len(image_paths) == 0:
        raise ValueError(f"No images found under {args.image_folder}")

    # Build model
    if args.pretrained:
        model = VGGT.from_pretrained(args.hf_repo).to(device)
        # Apply masking flags after construction
        agg = model.aggregator
        agg.mask_type = args.mask_type
        agg.topk_neighbors = args.topk_neighbors
        agg.mutual = args.mutual
        agg.soft_mask = args.soft_mask or (args.mask_type == "soft")
        agg.mask_hub_tokens = args.mask_hub_tokens
    else:
        model = VGGT(
            mask_type=args.mask_type,
            topk_neighbors=args.topk_neighbors,
            mutual=args.mutual,
            soft_mask=args.soft_mask,
            mask_hub_tokens=args.mask_hub_tokens,
        ).to(device)

    images = load_and_preprocess_images(image_paths).to(device)

    # Measure runtime and memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype) if device == "cuda" else torch.autocast("cpu", enabled=False):
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
        },
        "runtime_s": total_time,
        "peak_VRAM_GB": peak_vram,
        # Placeholders for quantitative metrics (computed elsewhere in eval pipeline)
        "pose_AUC": None,
        "depth_RMSE": None,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
