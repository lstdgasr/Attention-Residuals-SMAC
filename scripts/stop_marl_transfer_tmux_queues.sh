#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
PROJECT_DIR="${PROJECT_DIR:-/home/xhl009/MARL/pymarl}"

SESSION_PATTERN='^(hardmix_|followup_|adapt_|diag_).*(gpu[0-9]+)$'

echo "DRY_RUN=$DRY_RUN"
echo "PROJECT_DIR=$PROJECT_DIR"
echo

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not available."
  exit 1
fi

echo "Matching tmux sessions:"
mapfile -t sessions < <(tmux list-sessions -F '#S' 2>/dev/null | grep -E "$SESSION_PATTERN" || true)
if [ "${#sessions[@]}" -eq 0 ]; then
  echo "  none"
else
  printf '  %s\n' "${sessions[@]}"
fi
echo

echo "PyMARL training processes under $PROJECT_DIR:"
ps -u "$USER" -o pid=,ppid=,stat=,etime=,cmd= \
  | grep -F "$PROJECT_DIR/src/main.py" \
  | grep -v grep \
  || echo "  none"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run only. No tmux sessions were killed."
  echo
  echo "If tmux cleanup leaves a stale process, inspect with:"
  echo "  nvidia-smi"
  echo "Then kill only your own matching PyMARL PID, for example:"
  echo "  kill <PID>"
  exit 0
fi

for session in "${sessions[@]}"; do
  echo "Killing tmux session: $session"
  tmux kill-session -t "$session"
done

echo
echo "Remaining PyMARL training processes under $PROJECT_DIR:"
ps -u "$USER" -o pid=,ppid=,stat=,etime=,cmd= \
  | grep -F "$PROJECT_DIR/src/main.py" \
  | grep -v grep \
  || echo "  none"
echo
echo "tmux cleanup complete. Run nvidia-smi to confirm GPU memory is released."
