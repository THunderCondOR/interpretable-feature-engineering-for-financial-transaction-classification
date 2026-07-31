#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
session="reviewer_v10_datafusion_gpu_waiter"
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) execute=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $execute -eq 0 ]]; then
  "$python_bin" scripts/run_datafusion_gpu_priority.py
  exit 0
fi
cd "$repo_root"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2; exit 3
fi
mkdir -p logs/runs/reviewer-v10-datafusion-gpu-priority
tmux new-session -d -s "$session" \
  "cd '$repo_root' && set -o pipefail && '$python_bin' scripts/run_datafusion_gpu_priority.py --execute 2>&1 | tee 'logs/runs/reviewer-v10-datafusion-gpu-priority/waiter.log'"
echo "started tmux=$session"
