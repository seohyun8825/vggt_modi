#!/usr/bin/env bash
set -euo pipefail

# Allow FlexAttention on long sequences (N > 4096)
export VGGT_ALLOW_FLEX_LARGE_N=${VGGT_ALLOW_FLEX_LARGE_N:-1}
# Enable verbose Flex/Fallback telemetry logs
export VGGT_DEBUG=${VGGT_DEBUG:-1}

IMAGE_DIR=${1:-"examples/kitchen/images"}
OUT_ROOT=${2:-"results/bench"}

python /workspace/toddler/vggt/tools/bench_runtime.py \
  --image_folder "$IMAGE_DIR" \
  --output_root "$OUT_ROOT"
