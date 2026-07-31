#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
SESSION="${TMUX_SESSION:-reviewer_v7_cluster_fidelity}"
LOG="${LOG_PATH:-logs/runs/reviewer-v7-cluster-fidelity.log}"

if [[ "${1:-}" != "--execute" ]]; then
  exec "$PYTHON_BIN" scripts/run_cluster_quality_fidelity_queue.py
fi

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

COMMAND="set -o pipefail; '$PYTHON_BIN' scripts/run_cluster_quality_fidelity_queue.py --execute 2>&1 | tee -a '$LOG'"
tmux new-session -d -s "$SESSION" "bash -lc \"$COMMAND\""
echo "Started $SESSION"
echo "Log: $LOG"
