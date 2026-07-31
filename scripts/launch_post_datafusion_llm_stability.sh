#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
run_id="reviewer-v10-llm-stability-waiters"
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --execute) execute=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $execute -eq 0 ]]; then
  "$python_bin" scripts/run_post_datafusion_llm_stability.py --model qwen
  "$python_bin" scripts/run_post_datafusion_llm_stability.py --model gpt_oss
  exit 0
fi
cd "$repo_root"
mkdir -p "logs/runs/$run_id"
for model in qwen gpt_oss; do
  session="${run_id//[^a-zA-Z0-9_]/_}_${model}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2; exit 3
  fi
  tmux new-session -d -s "$session" \
    "cd '$repo_root' && set -a && source .envrc && set +a && set -o pipefail && '$python_bin' scripts/run_post_datafusion_llm_stability.py --model '$model' --execute 2>&1 | tee 'logs/runs/$run_id/${model}.log'"
done
echo "started post-DataFusion LLM stability waiters for Qwen and GPT-OSS"
