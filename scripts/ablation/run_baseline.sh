#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/baseline"}
CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}
# Which stages to run: forward | eval | both
RUN_MODE=${RUN_MODE:-both}

mkdir -p "$OUTPUT_DIR"

if [[ "$RUN_MODE" == "forward" || "$RUN_MODE" == "both" ]]; then
  OUT_FWD="$OUTPUT_DIR/forward"
  mkdir -p "$OUT_FWD"
  echo "[RUN] Baseline forward -> $OUT_FWD"
  python tools/run_ablation.py \
    --image_folder "$IMAGE_DIR" \
    --output_dir "$OUT_FWD" \
    --mask_type none \
    --runtime_source forward \
    --eval_co3d 0 \
    2>&1 | tee "$OUT_FWD/log.txt"
fi

if [[ "$RUN_MODE" == "eval" || "$RUN_MODE" == "both" ]]; then
  OUT_EVAL="$OUTPUT_DIR/eval"
  mkdir -p "$OUT_EVAL"
  echo "[RUN] Baseline CO3D eval -> $OUT_EVAL"
  python tools/run_ablation.py \
    --image_folder "$IMAGE_DIR" \
    --output_dir "$OUT_EVAL" \
    --mask_type none \
    --runtime_source co3d_eval \
    --eval_co3d 1 \
    --co3d_dir "$CO3D_DIR" \
    --co3d_anno_dir "$CO3D_ANNO_DIR" \
    --fast_eval 1 \
    2>&1 | tee "$OUT_EVAL/log.txt"
fi
