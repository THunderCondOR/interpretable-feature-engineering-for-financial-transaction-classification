#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
SOCKET="${TMUX_SOCKET:-reviewer-v6-offline}"
SESSION="${TMUX_SESSION:-reviewer-v6-e5-fidelity}"
LOG="${LOG_PATH:-logs/runs/reviewer-v6-e5-fidelity.log}"

if [[ "${1:-}" != "--execute" ]]; then
  exec "$PYTHON_BIN" scripts/run_e5_clustering_fidelity_queue.py
fi
if tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SOCKET/$SESSION" >&2
  exit 2
fi
mkdir -p "$(dirname "$LOG")"
COMMAND="set -o pipefail; '$PYTHON_BIN' scripts/run_e5_clustering_fidelity_queue.py --execute 2>&1 | tee -a '$LOG'"
tmux -L "$SOCKET" new-session -d -s "$SESSION" "bash -lc \"$COMMAND\""
echo "Started tmux -L $SOCKET attach -t $SESSION"
