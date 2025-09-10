#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/ours_window2"}
CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}

mkdir -p "$OUTPUT_DIR"

# Windowed = nearest-by-index neighbors; same as topk by index distance
python tools/run_ablation.py \
  --image_folder "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mask_type topk \
  --topk_neighbors 2 \
  --mutual true \
  --eval_co3d 1 \
  --co3d_dir "$CO3D_DIR" \
  --co3d_anno_dir "$CO3D_ANNO_DIR" \
  --fast_eval 1 \
  2>&1 | tee "$OUTPUT_DIR/log.txt"

