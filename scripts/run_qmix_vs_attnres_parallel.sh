#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ "${CONDA_DEFAULT_ENV:-}" != "pymarl-sc2" ]; then
  echo "Please run: conda activate pymarl-sc2"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for parallel launching."
  exit 1
fi

if [ -z "${GPUS:-}" ]; then
  echo "Please provide four idle GPUs, for example:"
  echo '  GPUS="0 4 5 6" bash scripts/run_qmix_vs_attnres_parallel.sh'
  exit 1
fi

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
  echo "GPUS must contain at least four GPU ids. Got: $GPUS"
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
MAP_NAME="${MAP_NAME:-2s3z}"
SC2PATH_DEFAULT="$PROJECT_DIR/3rdparty/StarCraftII"
SC2PATH_SRV="$PROJECT_DIR/3rdparty/StarCraftII_srv"
SC2PATH_CLEAN="$PROJECT_DIR/3rdparty/StarCraftII_clean"

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

mkdir -p "$RESULTS_DIR/launcher_logs"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "GPUS=$GPUS"
echo "MAP_NAME=$MAP_NAME"

echo "Running smoke test on GPU ${GPU_LIST[0]}..."
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" CUDA_DEVICE="${GPU_LIST[0]}" bash scripts/server_smoke_test.sh

conda_base="$(conda info --base)"
conda_activate="source \"$conda_base/etc/profile.d/conda.sh\" && conda activate pymarl-sc2"

launch_job() {
  local session="$1"
  local gpu="$2"
  local config="$3"
  local seed="$4"
  local log_file="$RESULTS_DIR/launcher_logs/${session}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; refusing to overwrite it."
    exit 1
  fi

  local cmd
  cmd="set -euo pipefail; cd \"$PROJECT_DIR\"; $conda_activate; export SC2PATH=\"$SC2PATH\"; export RESULTS_DIR=\"$RESULTS_DIR\"; export CUDA_VISIBLE_DEVICES=\"$gpu\"; export CUDA_DEVICE=\"$gpu\"; python src/main.py --config=$config --env-config=sc2 with env_args.map_name=$MAP_NAME seed=$seed local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=2050000 test_nepisode=32 save_model=True save_model_interval=500000 2>&1 | tee \"$log_file\""

  echo "Launching $session on GPU $gpu: $config seed=$seed"
  tmux new-session -d -s "$session" "bash -lc '$cmd'"
}

launch_job "${MAP_NAME}_qmix_s1_gpu${GPU_LIST[0]}" "${GPU_LIST[0]}" "qmix" "1"
launch_job "${MAP_NAME}_qmix_s2_gpu${GPU_LIST[1]}" "${GPU_LIST[1]}" "qmix" "2"
launch_job "${MAP_NAME}_qmix_attnres_s1_gpu${GPU_LIST[2]}" "${GPU_LIST[2]}" "qmix_attnres" "1"
launch_job "${MAP_NAME}_qmix_attnres_s2_gpu${GPU_LIST[3]}" "${GPU_LIST[3]}" "qmix_attnres" "2"

echo
echo "Launched first batch. Monitor with:"
echo "  tmux ls"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $RESULTS_DIR/launcher_logs/<session>.log"
echo
echo "After one GPU becomes free, run seed 3 jobs manually, replacing <GPU> with the idle GPU:"
echo "  CUDA_VISIBLE_DEVICES=<GPU> CUDA_DEVICE=<GPU> python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=$MAP_NAME seed=3 local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=2050000 test_nepisode=32 save_model=True save_model_interval=500000"
echo "  CUDA_VISIBLE_DEVICES=<GPU> CUDA_DEVICE=<GPU> python src/main.py --config=qmix_attnres --env-config=sc2 with env_args.map_name=$MAP_NAME seed=3 local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=2050000 test_nepisode=32 save_model=True save_model_interval=500000"
