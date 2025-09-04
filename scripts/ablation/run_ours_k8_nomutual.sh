#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/ours_k8_nomutual"}

mkdir -p "$OUTPUT_DIR"

python tools/run_ablation.py \
  --image_folder "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mask_type topk \
  --topk_neighbors 8 \
  --mutual false \
  2>&1 | tee "$OUTPUT_DIR/log.txt"

