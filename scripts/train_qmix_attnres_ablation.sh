#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
SC2PATH_DEFAULT="$PROJECT_DIR/3rdparty/StarCraftII"
SC2PATH_SRV="$PROJECT_DIR/3rdparty/StarCraftII_srv"
SC2PATH_CLEAN="$PROJECT_DIR/3rdparty/StarCraftII_clean"
CONFIG="${CONFIG:-qmix_attnres_l2}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-2050000}"

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
echo "RESULTS_DIR=$RESULTS_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONFIG=$CONFIG"
echo "MAP_NAME=$MAP_NAME"
echo "SEED=$SEED"
echo "T_MAX=$T_MAX"

cmd=(
  python src/main.py
  --config="$CONFIG"
  --env-config=sc2
  with
  env_args.map_name="$MAP_NAME"
  seed="$SEED"
  local_results_path="$RESULTS_DIR"
  use_cuda=True
  t_max="$T_MAX"
  test_nepisode=32
  save_model=True
  save_model_interval=500000
)

echo "COMMAND=${cmd[*]}"
"${cmd[@]}"
