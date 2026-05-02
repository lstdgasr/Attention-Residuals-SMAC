#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$HOME/MARL/pymarl-results}"
SACRED_DIR="${SACRED_DIR:-results/sacred}"
MAPS="${MAPS:-3s5z 5m_vs_6m}"

mkdir -p "$RESULTS_DIR/figures"

echo "SACRED_DIR=$SACRED_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "MAPS=$MAPS"

for map_name in $MAPS; do
  echo "Plotting $map_name"
  python scripts/plot_smac_results.py \
    --sacred-dir "$SACRED_DIR" \
    --map-name "$map_name" \
    --output-dir "$RESULTS_DIR/figures" \
    --completed-only
done
