#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
SC2PATH_DEFAULT="$PROJECT_DIR/3rdparty/StarCraftII"
SC2PATH_SRV="$PROJECT_DIR/3rdparty/StarCraftII_srv"
SC2PATH_CLEAN="$PROJECT_DIR/3rdparty/StarCraftII_clean"
SEED="${SEED:-1}"
MAP_NAME="${MAP_NAME:-2s3z}"

if [ "${CONDA_DEFAULT_ENV:-}" != "pymarl-sc2" ]; then
  echo "Please run: conda activate pymarl-sc2"
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${CUDA_DEVICE:-}" ]; then
  echo "Please set CUDA_VISIBLE_DEVICES or CUDA_DEVICE to an idle GPU."
  exit 1
fi

if [ -d "$SC2PATH_SRV" ] && { [ -z "${SC2PATH:-}" ] || [ "${SC2PATH:-}" = "$SC2PATH_DEFAULT" ]; }; then
  export SC2PATH="$SC2PATH_SRV"
elif [ -z "${SC2PATH:-}" ]; then
  if [ -d "$SC2PATH_SRV" ]; then
    export SC2PATH="$SC2PATH_SRV"
  elif [ -d "$SC2PATH_CLEAN" ]; then
    export SC2PATH="$SC2PATH_CLEAN"
  else
    export SC2PATH="$SC2PATH_DEFAULT"
  fi
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICE}"
mkdir -p "$RESULTS_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
if [[ "$SC2PATH" != *"StarCraftII_srv" ]]; then
  echo "WARNING: formal training is expected to use StarCraftII_srv; current SC2PATH is $SC2PATH"
fi
if [ -d "$SC2PATH/Versions" ]; then
  echo "SC2_VERSIONS=$(find "$SC2PATH/Versions" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null || true)"
fi
echo "RESULTS_DIR=$RESULTS_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SEED=$SEED"
echo "MAP_NAME=$MAP_NAME"

cmd=(
  python src/main.py
  --config=qmix_attnres
  --env-config=sc2
  with
  env_args.map_name="$MAP_NAME"
  seed="$SEED"
  local_results_path="$RESULTS_DIR"
  use_cuda=True
  t_max=2050000
  test_nepisode=32
  save_model=True
  save_model_interval=500000
)

echo "COMMAND=${cmd[*]}"
"${cmd[@]}"
