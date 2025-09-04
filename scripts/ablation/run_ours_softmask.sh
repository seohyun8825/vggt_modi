#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/ours_softmask"}

mkdir -p "$OUTPUT_DIR"

python tools/run_ablation.py \
  --image_folder "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mask_type soft \
  --soft_mask true \
  --mutual true \
  2>&1 | tee "$OUTPUT_DIR/log.txt"

