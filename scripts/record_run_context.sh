#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
SC2PATH="${SC2PATH:-$HOME/MARL/pymarl/3rdparty/StarCraftII_srv}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="$RESULTS_DIR/run_context_$STAMP.txt"

mkdir -p "$RESULTS_DIR"

{
  echo "timestamp=$STAMP"
  echo "project_dir=$(pwd)"
  echo "results_dir=$RESULTS_DIR"
  echo "sc2path=$SC2PATH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "sc2_versions=$(find "$SC2PATH/Versions" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null || true)"
  echo "algorithm=QMIX"
  echo "baseline_map=2s3z"
  echo "default_multiseeds=1 2 3"
  echo "note=Paper-faithful SMAC results used SC2.4.6.2.69232; this setup currently runs SC2.4.10/Base75689."
  echo
  echo "[nvidia-smi]"
  nvidia-smi || true
  echo
  echo "[python-import-check]"
  python -c "import torch, pysc2, smac; print('torch_ok pysc2_ok smac_ok')" || true
} > "$OUT_FILE"

echo "Saved run context to $OUT_FILE"
