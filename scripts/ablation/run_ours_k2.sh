#!/usr/bin/env bash
set -euo pipefail

# Allow FlexAttention on long sequences
export VGGT_ALLOW_FLEX_LARGE_N=${VGGT_ALLOW_FLEX_LARGE_N:-1}

CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_CAT=${CO3D_CAT:-apple}
FWD_IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/ours_k2"}
RUN_MODE=${RUN_MODE:-both}

mkdir -p "$OUTPUT_DIR"

if [[ "$RUN_MODE" == "forward" || "$RUN_MODE" == "both" ]]; then
  OUT_FWD="$OUTPUT_DIR/forward"
  mkdir -p "$OUT_FWD"
  echo "[RUN] Ours K=2 forward -> $OUT_FWD"
  python tools/run_ablation.py \
    --image_folder "$FWD_IMAGE_DIR" \
    --output_dir "$OUT_FWD" \
    --mask_type topk \
    --topk_neighbors 2 \
    --mutual true \
    --runtime_source forward \
    --eval_co3d 0 \
    2>&1 | tee "$OUT_FWD/log.txt"
fi

if [[ "$RUN_MODE" == "eval" || "$RUN_MODE" == "both" ]]; then
  OUT_EVAL="$OUTPUT_DIR/eval"
  mkdir -p "$OUT_EVAL"
  echo "[RUN] Ours K=2 CO3D eval -> $OUT_EVAL"
  python tools/run_ablation.py \
    --image_folder "$FWD_IMAGE_DIR" \
    --output_dir "$OUT_EVAL" \
    --mask_type topk \
    --topk_neighbors 2 \
    --mutual true \
    --runtime_source co3d_eval \
    --eval_co3d 1 \
    --co3d_dir "$CO3D_DIR" \
    --co3d_anno_dir "$CO3D_ANNO_DIR" \
    --fast_eval 1 \
    2>&1 | tee "$OUT_EVAL/log.txt"
fi
