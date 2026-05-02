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
  echo "tmux is required for queued parallel launching."
  exit 1
fi

if [ -z "${GPUS:-}" ]; then
  echo "Please provide idle GPUs, for example:"
  echo '  GPUS="0 1 4 5" bash scripts/run_attnres_sc2_diagnostic_matrix.sh'
  exit 1
fi

read -r -a GPU_LIST <<< "$GPUS"
RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
MAPS="${MAPS:-5m_vs_6m 3s5z}"
CONFIGS="${CONFIGS:-qmix qmix_attnres qmix_attnres_l2 qmix_attnres_block qmix_depth_mlp}"
SEEDS="${SEEDS:-1 2 3}"
T_MAX="${T_MAX:-2050000}"
TEST_NEPISODE="${TEST_NEPISODE:-32}"
SAVE_MODEL="${SAVE_MODEL:-True}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-500000}"
SESSION_PREFIX="${SESSION_PREFIX:-attnres_diag}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

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

mkdir -p "$RESULTS_DIR/launcher_logs" "$RESULTS_DIR/run_manifests"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "GPUS=$GPUS"
echo "MAPS=$MAPS"
echo "CONFIGS=$CONFIGS"
echo "SEEDS=$SEEDS"
echo "T_MAX=$T_MAX"
echo "TEST_NEPISODE=$TEST_NEPISODE"
echo "SAVE_MODEL=$SAVE_MODEL"
echo "SESSION_PREFIX=$SESSION_PREFIX"
echo "DRY_RUN=$DRY_RUN"
echo

manifest="$RESULTS_DIR/run_manifests/${SESSION_PREFIX}_$(date +%Y%m%d_%H%M%S).tsv"
printf "map\tconfig\tseed\tgpu\tlog_file\n" > "$manifest"

jobs=()
for map_name in $MAPS; do
  for config in $CONFIGS; do
    for seed in $SEEDS; do
      jobs+=("$map_name|$config|$seed")
    done
  done
done

for idx in "${!jobs[@]}"; do
  IFS="|" read -r map_name config seed <<< "${jobs[$idx]}"
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  log_file="$RESULTS_DIR/launcher_logs/${map_name}_${config}_s${seed}_gpu${gpu}.log"
  printf "%s\t%s\t%s\t%s\t%s\n" "$map_name" "$config" "$seed" "$gpu" "$log_file" >> "$manifest"
done

echo "Wrote manifest: $manifest"

if [ "$DRY_RUN" = "1" ]; then
  column -t -s $'\t' "$manifest" || cat "$manifest"
  exit 0
fi

if [ "$RUN_SMOKE_TEST" = "1" ]; then
  echo "Running smoke test on GPU ${GPU_LIST[0]}..."
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" CUDA_DEVICE="${GPU_LIST[0]}" bash scripts/server_smoke_test.sh
fi

conda_base="$(conda info --base)"
conda_activate="source \"$conda_base/etc/profile.d/conda.sh\" && conda activate pymarl-sc2"

launch_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local session="${SESSION_PREFIX}_gpu${gpu}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; refusing to overwrite it."
    exit 1
  fi

  local worker_cmd
  worker_cmd="set -euo pipefail; cd \"$PROJECT_DIR\"; $conda_activate; export SC2PATH=\"$SC2PATH\"; export RESULTS_DIR=\"$RESULTS_DIR\"; export CUDA_VISIBLE_DEVICES=\"$gpu\"; export CUDA_DEVICE=\"$gpu\";"

  for idx in "${!jobs[@]}"; do
    if [ "$((idx % ${#GPU_LIST[@]}))" -ne "$worker_idx" ]; then
      continue
    fi
    IFS="|" read -r map_name config seed <<< "${jobs[$idx]}"
    log_file="$RESULTS_DIR/launcher_logs/${map_name}_${config}_s${seed}_gpu${gpu}.log"
    worker_cmd+=" echo \"[$(date +%F_%T)] Running map=$map_name config=$config seed=$seed gpu=$gpu\";"
    worker_cmd+=" python src/main.py --config=$config --env-config=sc2 with env_args.map_name=$map_name seed=$seed local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=$T_MAX test_nepisode=$TEST_NEPISODE save_model=$SAVE_MODEL save_model_interval=$SAVE_MODEL_INTERVAL 2>&1 | tee \"$log_file\";"
  done
  worker_cmd+=" echo \"All queued jobs finished for GPU $gpu\""

  echo "Launching worker $session"
  tmux new-session -d -s "$session" "bash -lc '$worker_cmd'"
}

for worker_idx in "${!GPU_LIST[@]}"; do
  launch_worker "$worker_idx" "${GPU_LIST[$worker_idx]}"
done

echo
echo "Launched ${#GPU_LIST[@]} GPU workers for ${#jobs[@]} jobs."
echo "Monitor with:"
echo "  tmux ls"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $RESULTS_DIR/launcher_logs/<map>_<config>_s<seed>_gpu<gpu>.log"
echo
echo "After jobs finish, summarize with:"
echo "  python scripts/summarize_attnres_sc2_diagnostics.py --sacred-dir results/sacred --output-dir \"$RESULTS_DIR/diagnostics\""
