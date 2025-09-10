#!/usr/bin/env bash
set -euo pipefail

# CO3D eval using Top-K=8, mutual=true, with soft bias on allowed pairs (bias from MegaLoc similarity).

export VGGT_ALLOW_FLEX_LARGE_N=${VGGT_ALLOW_FLEX_LARGE_N:-1}
export VGGT_DEBUG=${VGGT_DEBUG:-0}

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUTPUT_DIR=${2:-"results/all_runs/run_co3d_k8_softbias"}
CO3D_DIR=${3:-"/workspace/toddler/vggt/co3d_annotations_full"}
CO3D_ANNO_DIR=${4:-"/workspace/toddler/vggt/co3d_annotations_full"}

TOPK=${TOPK:-8}
MUTUAL=${MUTUAL:-true}
CO3D_NUM_FRAMES=${CO3D_NUM_FRAMES:-10}
BIAS_T=${BIAS_T:-0.25}
BIAS_ALPHA=${BIAS_ALPHA:-1.0}

mkdir -p "$OUTPUT_DIR"

# Forward graph (plain)
GRAPH_JSON="$OUTPUT_DIR/forward_adjacency.json"
python tools/build_megaloc_graph.py \
  --image_dir "$IMAGE_DIR" \
  --out "$GRAPH_JSON" \
  --topk "$TOPK" \
  $( [[ "$MUTUAL" == "true" ]] && echo "--mutual" )

OUT_FWD="$OUTPUT_DIR/forward"; mkdir -p "$OUT_FWD"
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

# Per-seq graphs with soft bias included in JSON
CO3D_ADJ_DIR="$OUTPUT_DIR/co3d_adj"
python tools/build_co3d_graphs.py \
  --co3d_dir "$CO3D_DIR" \
  --co3d_anno_dir "$CO3D_ANNO_DIR" \
  --out_dir "$CO3D_ADJ_DIR" \
  --num_frames "$CO3D_NUM_FRAMES" \
  --topk "$TOPK" \
  $( [[ "$MUTUAL" == "true" ]] && echo "--mutual" ) \
  --device cuda \
  --backbone auto \
  --soft_bias \
  --bias_temperature "$BIAS_T" \
  --bias_alpha "$BIAS_ALPHA" \
  --fast_eval

OUT_EVAL="$OUTPUT_DIR/eval"; mkdir -p "$OUT_EVAL"
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
  --soft_mask 1 \
  2>&1 | tee "$OUT_EVAL/log.txt"

