#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
run_id="reviewer-v10-stability"
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --execute) execute=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
session="${run_id//[^a-zA-Z0-9_]/_}"
command=("$python_bin" scripts/run_current_stability_queue.py
  --datafusion-run-id reviewer-v10-datafusion-public
  --wait-for-tmux-session reviewer_v10_lora_all_datasets --execute)
if [[ $execute -eq 0 ]]; then
  "$python_bin" scripts/run_current_stability_queue.py
  exit 0
fi
cd "$repo_root"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2; exit 3
fi
mkdir -p "logs/runs/$run_id"
tmux new-session -d -s "$session" \
  "cd '$repo_root' && set -o pipefail && ${command[*]} 2>&1 | tee 'logs/runs/$run_id/stability.log'"
echo "started tmux=$session log=logs/runs/$run_id/stability.log"
