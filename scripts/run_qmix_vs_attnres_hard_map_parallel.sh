#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_NAME="${MAP_NAME:-3s5z}"

echo "Hard-map comparison batch"
echo "MAP_NAME=$MAP_NAME"
echo
echo "Recommended order:"
echo "  1) MAP_NAME=3s5z GPUS=\"0 1 4 5\" bash scripts/run_qmix_vs_attnres_hard_map_parallel.sh"
echo "  2) MAP_NAME=5m_vs_6m GPUS=\"0 1 4 5\" bash scripts/run_qmix_vs_attnres_hard_map_parallel.sh"
echo "  3) MAP_NAME=8m_vs_9m GPUS=\"0 1 4 5\" bash scripts/run_qmix_vs_attnres_hard_map_parallel.sh   # optional"
echo

MAP_NAME="$MAP_NAME" bash "$SCRIPT_DIR/run_qmix_vs_attnres_parallel.sh"
