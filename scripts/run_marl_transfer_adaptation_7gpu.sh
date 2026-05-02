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

RESULTS_DIR="${RESULTS_DIR:-/home/xhl009/MARL/pymarl-results}"
SC2PATH="${SC2PATH:-/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv}"
T_MAX="${T_MAX:-2050000}"
TEST_NEPISODE="${TEST_NEPISODE:-32}"
SAVE_MODEL="${SAVE_MODEL:-True}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-500000}"
SESSION_PREFIX="${SESSION_PREFIX:-adapt}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
SMOKE_GPU="${SMOKE_GPU:-0}"

mkdir -p "$RESULTS_DIR/launcher_logs" "$RESULTS_DIR/run_manifests"

# Fields: gpu|map|config|seed|name_override|extra sacred overrides
JOBS=(
  "0|5m_vs_6m|qmix_depth_mlp|2||"
  "0|5m_vs_6m|iql|1||"
  "0|5m_vs_6m|vdn_attnres|1|vdn_attnres_l2|attn_res_layers=2"

  "1|5m_vs_6m|qmix_depth_mlp|3||"
  "1|5m_vs_6m|iql|2||"
  "1|5m_vs_6m|vdn_attnres|2|vdn_attnres_l2|attn_res_layers=2"

  "3|3s5z|qmix|3||"
  "3|5m_vs_6m|iql|3||"
  "3|5m_vs_6m|vdn_attnres|3|vdn_attnres_l2|attn_res_layers=2"

  "4|3s5z|qmix_attnres_block|3||"
  "4|5m_vs_6m|iql_attnres|1|iql_attnres_l2|attn_res_layers=2"

  "5|3s5z|qmix_depth_mlp|2||"
  "5|5m_vs_6m|iql_attnres|2|iql_attnres_l2|attn_res_layers=2"

  "6|3s5z|qmix_depth_mlp|3||"
  "6|5m_vs_6m|iql_attnres|3|iql_attnres_l2|attn_res_layers=2"

  "7|5m_vs_6m|vdn|1||"
  "7|5m_vs_6m|vdn|2||"
  "7|5m_vs_6m|vdn|3||"
)

manifest="$RESULTS_DIR/run_manifests/${SESSION_PREFIX}_transfer_$(date +%Y%m%d_%H%M%S).tsv"
printf "gpu\tmap\tconfig\tseed\tname\textra\tsession\tlog_file\n" > "$manifest"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "T_MAX=$T_MAX"
echo "TEST_NEPISODE=$TEST_NEPISODE"
echo "SAVE_MODEL=$SAVE_MODEL"
echo "SESSION_PREFIX=$SESSION_PREFIX"
echo "RUN_SMOKE_TEST=$RUN_SMOKE_TEST"
echo "SMOKE_GPU=$SMOKE_GPU"
echo "DRY_RUN=$DRY_RUN"
echo

for job in "${JOBS[@]}"; do
  IFS="|" read -r gpu map_name config seed name_override extra <<< "$job"
  log_name="${name_override:-$config}"
  session="${SESSION_PREFIX}_gpu${gpu}"
  log_file="$RESULTS_DIR/launcher_logs/${map_name}_${log_name}_s${seed}_gpu${gpu}.log"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$gpu" "$map_name" "$config" "$seed" "$log_name" "$extra" "$session" "$log_file" >> "$manifest"
done

echo "Wrote manifest: $manifest"
if [ "$DRY_RUN" = "1" ]; then
  column -t -s $'\t' "$manifest" || cat "$manifest"
  exit 0
fi

if [ "$RUN_SMOKE_TEST" = "1" ]; then
  echo "Running smoke test on GPU $SMOKE_GPU..."
  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" CUDA_DEVICE="$SMOKE_GPU" bash scripts/server_smoke_test.sh
fi

conda_base="$(conda info --base)"
conda_activate="source \"$conda_base/etc/profile.d/conda.sh\" && conda activate pymarl-sc2"

launch_gpu_queue() {
  local gpu="$1"
  local session="${SESSION_PREFIX}_gpu${gpu}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; refusing to overwrite it."
    exit 1
  fi

  local cmd
  cmd="set -euo pipefail; cd \"$PROJECT_DIR\"; $conda_activate; export SC2PATH=\"$SC2PATH\"; export RESULTS_DIR=\"$RESULTS_DIR\"; mkdir -p \"$RESULTS_DIR/launcher_logs\";"

  local has_job=0
  for job in "${JOBS[@]}"; do
    IFS="|" read -r job_gpu map_name config seed name_override extra <<< "$job"
    if [ "$job_gpu" != "$gpu" ]; then
      continue
    fi

    has_job=1
    local log_name="${name_override:-$config}"
    local log_file="$RESULTS_DIR/launcher_logs/${map_name}_${log_name}_s${seed}_gpu${gpu}.log"
    local overrides=""
    if [ -n "$name_override" ]; then
      overrides+=" name=\"$name_override\""
    fi
    if [ -n "$extra" ]; then
      overrides+=" $extra"
    fi

    cmd+=" echo \"[\$(date +%F_%T)] Running gpu=$gpu map=$map_name config=$config seed=$seed name=$log_name extra=$extra\";"
    cmd+=" CUDA_VISIBLE_DEVICES=\"$gpu\" CUDA_DEVICE=\"$gpu\" python src/main.py --config=\"$config\" --env-config=sc2 with env_args.map_name=\"$map_name\" seed=\"$seed\" local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=\"$T_MAX\" test_nepisode=\"$TEST_NEPISODE\" save_model=\"$SAVE_MODEL\" save_model_interval=\"$SAVE_MODEL_INTERVAL\"$overrides 2>&1 | tee \"$log_file\";"
  done

  if [ "$has_job" = "0" ]; then
    return
  fi

  cmd+=" echo \"[\$(date +%F_%T)] All queued jobs finished for GPU $gpu\""

  echo "Launching $session"
  tmux new-session -d -s "$session" "bash -lc '$cmd'"
}

for gpu in 0 1 3 4 5 6 7; do
  launch_gpu_queue "$gpu"
done

echo
echo "Launched MARL transfer adaptation queues."
echo "Monitor with:"
echo "  tmux ls"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $RESULTS_DIR/launcher_logs/5m_vs_6m_iql_attnres_l2_s1_gpu4.log"
echo
echo "After all queues finish, summarize with:"
echo "  python scripts/summarize_marl_transfer_adaptation.py --sacred-dir results/sacred --output-dir \"$RESULTS_DIR/diagnostics\""
