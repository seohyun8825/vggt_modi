#!/usr/bin/env bash
set -euo pipefail

# Sweep K over a comma-separated list and mutual in {true,false}

export VGGT_ALLOW_FLEX_LARGE_N=${VGGT_ALLOW_FLEX_LARGE_N:-1}
export VGGT_DEBUG=${VGGT_DEBUG:-0}

CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_CAT=${CO3D_CAT:-apple}
SEQ_DIR=$(find "$CO3D_ANNO_DIR/$CO3D_CAT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
IMAGE_DIR=${1:-"${SEQ_DIR}/images"}
OUT_ROOT=${2:-"results/sweep_k"}
KS=${5:-"8,10,12,14,16"}

mkdir -p "$OUT_ROOT"

IFS=',' read -ra KLIST <<< "$KS"
for K in "${KLIST[@]}"; do
  for M in true false; do
    TAG="K${K}_mutual_${M}"
    OUTDIR="$OUT_ROOT/$TAG"
    mkdir -p "$OUTDIR"
    echo "[RUN] K=$K mutual=$M -> $OUTDIR"
    python tools/run_ablation.py \
      --image_folder "$IMAGE_DIR" \
      --output_dir "$OUTDIR" \
      --mask_type topk \
      --topk_neighbors "$K" \
      --mutual "$M" \
      --eval_co3d 1 \
      --co3d_dir "$CO3D_DIR" \
      --co3d_anno_dir "$CO3D_ANNO_DIR" \
      --fast_eval 1 \
      2>&1 | tee "$OUTDIR/log.txt"
  done
done

echo "[OK] Sweep finished under $OUT_ROOT"
