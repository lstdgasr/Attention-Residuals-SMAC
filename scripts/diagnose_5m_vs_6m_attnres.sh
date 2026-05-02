#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
SACRED_DIR="${SACRED_DIR:-results/sacred}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"
BASELINE="${BASELINE:-qmix}"
CANDIDATE="${CANDIDATE:-qmix_attnres}"
GAP_THRESHOLD="${GAP_THRESHOLD:-0.25}"
T_MAX="${T_MAX:-2050000}"

mkdir -p "$RESULTS_DIR/diagnostics"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "SACRED_DIR=$SACRED_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "MAP_NAME=$MAP_NAME"
echo "BASELINE=$BASELINE"
echo "CANDIDATE=$CANDIDATE"
echo "GAP_THRESHOLD=$GAP_THRESHOLD"
echo

python scripts/diagnose_smac_progress.py \
  --sacred-dir "$SACRED_DIR" \
  --output-dir "$RESULTS_DIR/diagnostics" \
  --map-name "$MAP_NAME" \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --gap-threshold "$GAP_THRESHOLD" \
  --t-max "$T_MAX"

echo
echo "Recent launcher log snippets, if available:"
if [ -d "$RESULTS_DIR/launcher_logs" ]; then
  for pattern in "${MAP_NAME}_${BASELINE}" "${MAP_NAME}_${CANDIDATE}"; do
    echo
    echo "== $pattern =="
    matches=( "$RESULTS_DIR"/launcher_logs/"$pattern"*".log" )
    if [ -e "${matches[0]}" ]; then
      for log_file in "${matches[@]}"; do
        echo "-- $log_file"
        grep -E "Recent Stats|test_battle_won_mean|test_return_mean|t_env" "$log_file" | tail -n 20 || true
      done
    else
      echo "No launcher logs matched $pattern"
    fi
  done
else
  echo "No launcher log directory: $RESULTS_DIR/launcher_logs"
fi

echo
echo "Decision rule:"
echo "  If baseline_minus_candidate_win >= $GAP_THRESHOLD at common_t, keep the run but start a light AttnRes ablation."
echo "  If both are low, let them reach T_MAX=$T_MAX before changing code or claims."
