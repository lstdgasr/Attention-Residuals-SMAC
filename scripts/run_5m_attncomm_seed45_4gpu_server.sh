#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for queued parallel launching."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; please initialize conda for this shell first."
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-/home/xhl009/MARL/pymarl-results}"
SC2PATH="${SC2PATH:-/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv}"
T_MAX="${T_MAX:-5000000}"
TEST_NEPISODE="${TEST_NEPISODE:-32}"
SAVE_MODEL="${SAVE_MODEL:-True}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-1000000}"
SESSION_PREFIX="${SESSION_PREFIX:-seed45_5m}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-${SESSION_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/launcher_logs/$RUN_ID}"

mkdir -p "$RESULTS_DIR/sacred" "$RESULTS_DIR/models" "$LOG_DIR" "$RESULTS_DIR/run_manifests"

# Fields: gpu|map|config|seed|extra sacred overrides
JOBS=(
  "2|5m_vs_6m|qmix|4|"
  "2|5m_vs_6m|qmix|5|"
  "3|5m_vs_6m|qmix_attnres_l2|4|record_attn_weights=True"
  "3|5m_vs_6m|qmix_attnres_l2|5|record_attn_weights=True"
  "4|5m_vs_6m|qmix_attncomm_l2_other|4|record_comm_attn_weights=True"
  "4|5m_vs_6m|qmix_attncomm_l2_other|5|record_comm_attn_weights=True"
  "6|5m_vs_6m|qmix_attncomm_l2_self|4|record_comm_attn_weights=True"
  "6|5m_vs_6m|qmix_attncomm_l2_self|5|record_comm_attn_weights=True"
)

manifest="$RESULTS_DIR/run_manifests/${RUN_ID}.tsv"
printf "gpu\tmap\tconfig\tseed\tt_max\textra\tsession\tlog_file\n" > "$manifest"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "T_MAX=$T_MAX"
echo "TEST_NEPISODE=$TEST_NEPISODE"
echo "SAVE_MODEL=$SAVE_MODEL"
echo "SAVE_MODEL_INTERVAL=$SAVE_MODEL_INTERVAL"
echo "SESSION_PREFIX=$SESSION_PREFIX"
echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "DRY_RUN=$DRY_RUN"
echo

for job in "${JOBS[@]}"; do
  IFS="|" read -r gpu map_name config seed extra <<< "$job"
  session="${RUN_ID}_gpu${gpu}"
  log_file="$LOG_DIR/${map_name}_${config}_s${seed}_gpu${gpu}_seed45.log"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$gpu" "$map_name" "$config" "$seed" "$T_MAX" "$extra" "$session" "$log_file" >> "$manifest"
done

echo "Wrote manifest: $manifest"
if [ "$DRY_RUN" = "1" ]; then
  column -t -s $'\t' "$manifest" || cat "$manifest"
  exit 0
fi

conda_base="$(conda info --base)"
conda_activate="source \"$conda_base/etc/profile.d/conda.sh\" && conda activate pymarl-sc2"

launch_gpu_queue() {
  local gpu="$1"
  local session="${RUN_ID}_gpu${gpu}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists; refusing to overwrite it."
    exit 1
  fi

  local cmd
  cmd="set -euo pipefail; cd \"$PROJECT_DIR\"; $conda_activate; export SC2PATH=\"$SC2PATH\"; export RESULTS_DIR=\"$RESULTS_DIR\"; mkdir -p \"$RESULTS_DIR/sacred\" \"$RESULTS_DIR/models\" \"$LOG_DIR\";"

  local has_job=0
  for job in "${JOBS[@]}"; do
    IFS="|" read -r job_gpu map_name config seed extra <<< "$job"
    if [ "$job_gpu" != "$gpu" ]; then
      continue
    fi

    has_job=1
    local log_file="$LOG_DIR/${map_name}_${config}_s${seed}_gpu${gpu}_seed45.log"
    local overrides=""
    if [ -n "$extra" ]; then
      overrides+=" $extra"
    fi

    cmd+=" echo \"[\$(date +%F_%T)] Running gpu=$gpu map=$map_name config=$config seed=$seed t_max=$T_MAX extra=$extra\";"
    cmd+=" CUDA_VISIBLE_DEVICES=\"$gpu\" CUDA_DEVICE=\"$gpu\" python src/main.py --config=\"$config\" --env-config=sc2 with env_args.map_name=\"$map_name\" seed=\"$seed\" local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=\"$T_MAX\" test_nepisode=\"$TEST_NEPISODE\" save_model=\"$SAVE_MODEL\" save_model_interval=\"$SAVE_MODEL_INTERVAL\"$overrides 2>&1 | tee \"$log_file\";"
  done

  if [ "$has_job" = "0" ]; then
    return
  fi

  cmd+=" echo \"[\$(date +%F_%T)] All queued jobs finished for GPU $gpu\""

  echo "Launching $session"
  tmux new-session -d -s "$session" "bash -lc '$cmd'"
}

for gpu in 2 3 4 6; do
  launch_gpu_queue "$gpu"
done

echo
echo "Launched optional 5m_vs_6m seed4/5 insurance queues."
echo "Monitor with:"
echo "  tmux ls"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $LOG_DIR/5m_vs_6m_qmix_attncomm_l2_self_s5_gpu6_seed45.log"
echo
echo "After all queues finish, summarize seeds 1-5 with:"
echo "  python scripts/summarize_marl_transfer_adaptation.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/diagnostics\" --maps 5m_vs_6m --primary-configs qmix,qmix_attnres_l2,qmix_attncomm_l2_other,qmix_attncomm_l2_self --seeds 1,2,3,4,5 --include-cross --cross-pairs qmix:qmix_attnres_l2,qmix:qmix_attncomm_l2_other,qmix:qmix_attncomm_l2_self"
echo
echo "Then regenerate figures with:"
echo "  python scripts/plot_marl_transfer_curves.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/figures\" --maps 5m_vs_6m --seeds 1,2,3,4,5"
echo "  python scripts/plot_comm_attn_heatmaps.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/figures\""
echo "  python scripts/plot_attn_weight_heatmaps.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/figures\" --configs qmix_attnres_l2"
