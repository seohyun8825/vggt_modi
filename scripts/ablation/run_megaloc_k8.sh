#!/usr/bin/env bash
set -euo pipefail

# MegaLoc-based graph + Top-K masking (optionally soft bias). Builds graph then runs ablation.

export VGGT_ALLOW_FLEX_LARGE_N=${VGGT_ALLOW_FLEX_LARGE_N:-1}
export VGGT_DEBUG=${VGGT_DEBUG:-0}

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/megaloc_k8"}
CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}
ADJ_PROVIDER=${ADJ_PROVIDER:-megaloc}
CO3D_NUM_FRAMES=${CO3D_NUM_FRAMES:-10}

# Controls
TOPK=${TOPK:-8}
MUTUAL=${MUTUAL:-true}
SOFT_BIAS=${SOFT_BIAS:-false}
RUN_MODE=${RUN_MODE:-both}   # forward | eval | both

mkdir -p "$OUTPUT_DIR"

GRAPH_JSON="$OUTPUT_DIR/adjacency.json"
BACKBONE=${BACKBONE:-auto}   # auto | resnet50 | fallback

echo "[BUILD] MegaLoc graph -> $GRAPH_JSON (K=$TOPK, mutual=$MUTUAL, soft_bias=$SOFT_BIAS)"
python tools/build_megaloc_graph.py \
  --image_dir "$IMAGE_DIR" \
  --out "$GRAPH_JSON" \
  --topk "$TOPK" \
  --backbone "$BACKBONE" \
  $( [[ "$MUTUAL" == "true" ]] && echo "--mutual" ) \
  $( [[ "$SOFT_BIAS" == "true" ]] && echo "--soft_bias" )

if [[ "$RUN_MODE" == "forward" || "$RUN_MODE" == "both" ]]; then
  OUT_FWD="$OUTPUT_DIR/forward"
  mkdir -p "$OUT_FWD"
  echo "[RUN] MegaLoc K=$TOPK forward -> $OUT_FWD"
  python tools/run_ablation.py \
    --image_folder "$IMAGE_DIR" \
    --output_dir "$OUT_FWD" \
    --mask_type topk \
    --topk_neighbors "$TOPK" \
    --mutual "$MUTUAL" \
    --adjacency_json "$GRAPH_JSON" \
    --runtime_source forward \
    --eval_co3d 0 \
    2>&1 | tee "$OUT_FWD/log.txt"
fi

if [[ "$RUN_MODE" == "eval" || "$RUN_MODE" == "both" ]]; then
  OUT_EVAL="$OUTPUT_DIR/eval"
  mkdir -p "$OUT_EVAL"
  # Build per-sequence adjacency JSONs for CO3D to avoid mismatch and control K saturation
  CO3D_ADJ_DIR="$OUTPUT_DIR/co3d_adj"
  echo "[BUILD] CO3D per-sequence graphs -> $CO3D_ADJ_DIR (S=$CO3D_NUM_FRAMES, K=$TOPK, mutual=$MUTUAL)"
  python tools/build_co3d_graphs.py \
    --co3d_dir "$CO3D_DIR" \
    --co3d_anno_dir "$CO3D_ANNO_DIR" \
    --out_dir "$CO3D_ADJ_DIR" \
    --num_frames "$CO3D_NUM_FRAMES" \
    --seed 0 \
    --topk "$TOPK" \
    $( [[ "$MUTUAL" == "true" ]] && echo "--mutual" ) \
    --device cuda \
    --backbone auto \
    $( [[ "$SOFT_BIAS" == "true" ]] && echo "--soft_bias" ) \
    --fast_eval
  echo "[RUN] MegaLoc K=$TOPK CO3D eval -> $OUT_EVAL"
  python tools/run_ablation.py \
    --image_folder "$IMAGE_DIR" \
    --output_dir "$OUT_EVAL" \
    --mask_type topk \
    --topk_neighbors "$TOPK" \
    --mutual "$MUTUAL" \
    --runtime_source co3d_eval \
    --eval_co3d 1 \
    --co3d_dir "$CO3D_DIR" \
    --co3d_anno_dir "$CO3D_ANNO_DIR" \
    --fast_eval 1 \
    --co3d_num_frames "$CO3D_NUM_FRAMES" \
    --adjacency_json_dir "$CO3D_ADJ_DIR" \
    --soft_mask 0 \
    2>&1 | tee "$OUT_EVAL/log.txt"
fi
