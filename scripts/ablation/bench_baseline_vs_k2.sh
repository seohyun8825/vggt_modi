#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUT_ROOT=${2:-"results/bench"}

python /workspace/toddler/vggt/tools/bench_runtime.py \
  --image_folder "$IMAGE_DIR" \
  --output_root "$OUT_ROOT"

