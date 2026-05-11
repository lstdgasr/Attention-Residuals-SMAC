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
T_MAX_5M="${T_MAX_5M:-5000000}"
T_MAX_3S5Z="${T_MAX_3S5Z:-2050000}"
TEST_NEPISODE="${TEST_NEPISODE:-32}"
SAVE_MODEL="${SAVE_MODEL:-True}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-1000000}"
SESSION_PREFIX="${SESSION_PREFIX:-attncomm4}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"
SMOKE_GPU="${SMOKE_GPU:-0}"

mkdir -p "$RESULTS_DIR/sacred" "$RESULTS_DIR/models" "$RESULTS_DIR/launcher_logs" "$RESULTS_DIR/run_manifests"

# Fields: gpu|map|config|seed|t_max|extra sacred overrides
JOBS=(
  "3|5m_vs_6m|qmix|1|$T_MAX_5M|"
  "3|5m_vs_6m|qmix_attncomm_l2_other|1|$T_MAX_5M|record_comm_attn_weights=True"
  "3|5m_vs_6m|qmix_attncomm_l2_self|1|$T_MAX_5M|record_comm_attn_weights=True"
  "3|3s5z|qmix_attncomm_l2_other|1|$T_MAX_3S5Z|record_comm_attn_weights=True"
  "3|3s5z|qmix_attncomm_l2_self|1|$T_MAX_3S5Z|record_comm_attn_weights=True"

  "2|5m_vs_6m|qmix|2|$T_MAX_5M|"
  "2|5m_vs_6m|qmix_attncomm_l2_other|2|$T_MAX_5M|record_comm_attn_weights=True"
  "2|5m_vs_6m|qmix_attncomm_l2_self|2|$T_MAX_5M|record_comm_attn_weights=True"
  "2|3s5z|qmix_attncomm_l2_other|2|$T_MAX_3S5Z|record_comm_attn_weights=True"

  "4|5m_vs_6m|qmix|3|$T_MAX_5M|"
  "4|5m_vs_6m|qmix_attncomm_l2_other|3|$T_MAX_5M|record_comm_attn_weights=True"
  "4|5m_vs_6m|qmix_attncomm_l2_self|3|$T_MAX_5M|record_comm_attn_weights=True"
  "4|3s5z|qmix_attncomm_l2_self|2|$T_MAX_3S5Z|record_comm_attn_weights=True"

  "7|5m_vs_6m|qmix_attnres_l2|1|$T_MAX_5M|record_attn_weights=True"
  "7|5m_vs_6m|qmix_attnres_l2|2|$T_MAX_5M|record_attn_weights=True"
  "7|5m_vs_6m|qmix_attnres_l2|3|$T_MAX_5M|record_attn_weights=True"
  "7|3s5z|qmix_attncomm_l2_other|3|$T_MAX_3S5Z|record_comm_attn_weights=True"
  "7|3s5z|qmix_attncomm_l2_self|3|$T_MAX_3S5Z|record_comm_attn_weights=True"
)

manifest="$RESULTS_DIR/run_manifests/${SESSION_PREFIX}_qmix_l2_$(date +%Y%m%d_%H%M%S).tsv"
printf "gpu\tmap\tconfig\tseed\tt_max\textra\tsession\tlog_file\n" > "$manifest"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SC2PATH=$SC2PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "T_MAX_5M=$T_MAX_5M"
echo "T_MAX_3S5Z=$T_MAX_3S5Z"
echo "TEST_NEPISODE=$TEST_NEPISODE"
echo "SAVE_MODEL=$SAVE_MODEL"
echo "SAVE_MODEL_INTERVAL=$SAVE_MODEL_INTERVAL"
echo "SESSION_PREFIX=$SESSION_PREFIX"
echo "RUN_SMOKE_TEST=$RUN_SMOKE_TEST"
echo "SMOKE_GPU=$SMOKE_GPU"
echo "DRY_RUN=$DRY_RUN"
echo

for job in "${JOBS[@]}"; do
  IFS="|" read -r gpu map_name config seed t_max extra <<< "$job"
  session="${SESSION_PREFIX}_gpu${gpu}"
  log_file="$RESULTS_DIR/launcher_logs/${map_name}_${config}_s${seed}_gpu${gpu}.log"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$gpu" "$map_name" "$config" "$seed" "$t_max" "$extra" "$session" "$log_file" >> "$manifest"
done

echo "Wrote manifest: $manifest"
if [ "$DRY_RUN" = "1" ]; then
  column -t -s $'\t' "$manifest" || cat "$manifest"
  exit 0
fi

if [ "$RUN_SMOKE_TEST" = "1" ]; then
  echo "Running smoke test on GPU $SMOKE_GPU..."
  CUDA_VISIBLE_DEVICES="$SMOKE_GPU" CUDA_DEVICE="$SMOKE_GPU" RESULTS_DIR="$RESULTS_DIR" bash scripts/server_smoke_test.sh
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
  cmd="set -euo pipefail; cd \"$PROJECT_DIR\"; $conda_activate; export SC2PATH=\"$SC2PATH\"; export RESULTS_DIR=\"$RESULTS_DIR\"; mkdir -p \"$RESULTS_DIR/sacred\" \"$RESULTS_DIR/models\" \"$RESULTS_DIR/launcher_logs\";"

  local has_job=0
  for job in "${JOBS[@]}"; do
    IFS="|" read -r job_gpu map_name config seed t_max extra <<< "$job"
    if [ "$job_gpu" != "$gpu" ]; then
      continue
    fi

    has_job=1
    local log_file="$RESULTS_DIR/launcher_logs/${map_name}_${config}_s${seed}_gpu${gpu}.log"
    local overrides=""
    if [ -n "$extra" ]; then
      overrides+=" $extra"
    fi

    cmd+=" echo \"[\$(date +%F_%T)] Running gpu=$gpu map=$map_name config=$config seed=$seed t_max=$t_max extra=$extra\";"
    cmd+=" CUDA_VISIBLE_DEVICES=\"$gpu\" CUDA_DEVICE=\"$gpu\" python src/main.py --config=\"$config\" --env-config=sc2 with env_args.map_name=\"$map_name\" seed=\"$seed\" local_results_path=\"$RESULTS_DIR\" use_cuda=True t_max=\"$t_max\" test_nepisode=\"$TEST_NEPISODE\" save_model=\"$SAVE_MODEL\" save_model_interval=\"$SAVE_MODEL_INTERVAL\"$overrides 2>&1 | tee \"$log_file\";"
  done

  if [ "$has_job" = "0" ]; then
    return
  fi

  cmd+=" echo \"[\$(date +%F_%T)] All queued jobs finished for GPU $gpu\""

  echo "Launching $session"
  tmux new-session -d -s "$session" "bash -lc '$cmd'"
}

for gpu in 2 3 4 7; do
  launch_gpu_queue "$gpu"
done

echo
echo "Launched AttnComm-QMIX-L2 4GPU server queues."
echo "Monitor with:"
echo "  tmux ls"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $RESULTS_DIR/launcher_logs/5m_vs_6m_qmix_attncomm_l2_self_s1_gpu3.log"
echo
echo "After all queues finish, summarize and plot with:"
echo "  python scripts/summarize_marl_transfer_adaptation.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/diagnostics\" --maps 5m_vs_6m,3s5z --primary-configs qmix,qmix_attnres_l2,qmix_attncomm_l2_other,qmix_attncomm_l2_self --seeds 1,2,3 --include-cross --cross-pairs qmix:qmix_attnres_l2,qmix:qmix_attncomm_l2_other,qmix:qmix_attncomm_l2_self"
echo "  python scripts/plot_marl_transfer_curves.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/figures\" --seeds 1,2,3"
echo "  python scripts/plot_comm_attn_heatmaps.py --sacred-dir \"$RESULTS_DIR/sacred\" --output-dir \"$RESULTS_DIR/figures\""
