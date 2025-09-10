#!/usr/bin/env bash
set -euo pipefail

# Run all ablation scripts in this folder sequentially.
# Controls:
#   RUN_MODE=both|forward|eval   # default: both (each script runs forward timing and CO3D eval)
#   STOP_ON_ERROR=0|1            # default: 0 (continue on failure)
#   CO3D_DIR, CO3D_ANNO_DIR      # default paths below
#   OUT_ROOT                     # default: results/all_runs
#   VGGT_DEBUG                   # optional verbose telemetry

RUN_MODE=${RUN_MODE:-both}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}
OUT_ROOT=${OUT_ROOT:-results/all_runs}
CO3D_DIR=${CO3D_DIR:-/workspace/toddler/vggt/co3d_annotations_full}
CO3D_ANNO_DIR=${CO3D_ANNO_DIR:-/workspace/toddler/vggt/co3d_annotations_full}

export RUN_MODE

scripts=(
  # run_baseline.sh
  # run_ours_k2.sh
  # run_ours_k4.sh
  # run_ours_k8.sh
  # run_ours_k8_nomutual.sh
  # run_ours_k10.sh
  # run_ours_k10_nomutual.sh
  # run_ours_k12.sh
  # run_ours_k12_nomutual.sh
  # run_ours_k14.sh
  # run_ours_k14_nomutual.sh
  # run_ours_k16.sh
  # run_ours_masked_hub.sh
  # run_ours_softmask.sh
  # run_ours_window2.sh
  # run_ours_window4.sh
  # run_softmix_k8.sh
  # run_softmix_k10.sh
  # run_softmix_k12.sh
  run_k8_block64.sh
  run_k8_fallback_only.sh
  run_co3d_k8_megaloc.sh
  run_co3d_k8_resnet50.sh
  run_co3d_k8_nomutual.sh
  run_co3d_k8_hybrid_w2.sh
  run_co3d_k8_softbias.sh
  run_co3d_k8_hybrid_w2_softbias.sh
)

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Repo root is two levels up from this ablation folder
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

mkdir -p "$OUT_ROOT"

echo "[INFO] Running ablations to $OUT_ROOT (RUN_MODE=$RUN_MODE, STOP_ON_ERROR=$STOP_ON_ERROR)"

fails=()
for s in "${scripts[@]}"; do
  base="${s%.sh}"
  out="$OUT_ROOT/$base"
  echo
  echo "[RUN] $s -> $out"
  mkdir -p "$out"
  # Always execute each script from the repository root so relative paths like tools/run_ablation.py resolve
  if ( cd "$REPO_ROOT" && bash "scripts/ablation/$s" "" "$out" "$CO3D_DIR" "$CO3D_ANNO_DIR" ); then
    echo "[OK] $s"
  else
    echo "[FAIL] $s"
    fails+=("$s")
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      echo "[STOP] STOP_ON_ERROR=1; aborting. Failed: ${fails[*]}"
      exit 1
    fi
  fi
done

echo "\n[SUMMARY] Completed ablations. Output root: $OUT_ROOT"
if ((${#fails[@]})); then
  echo "Failed scripts (${#fails[@]}): ${fails[*]}"
else
  echo "All scripts succeeded."
fi
