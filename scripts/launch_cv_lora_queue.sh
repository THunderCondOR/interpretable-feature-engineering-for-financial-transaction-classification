#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="reviewer-v9-lora-qwen3-8b"
execute=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --execute) execute=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "invalid run ID" >&2
  exit 2
fi

session="${run_id//[^a-zA-Z0-9_]/_}"
log="logs/runs/$run_id/lora.log"
command=(
  /home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python
  scripts/run_cv_lora_queue.py
  --datasets berka,datafusion_education
  --folds 0,1,2,3,4
  --models qwen3_8b
  --output-root "results/v5/lora/$run_id"
  --protocol-map
  datafusion_education=mbd_5fold_seed42,berka=unittab_70_30_5seed
  --execute
)

if [[ $execute -eq 0 ]]; then
  echo "mode=dry-run"
  echo "session=$session"
  echo "models=qwen3_8b"
  echo "jobs=10 (Berka first, then Data Fusion)"
  echo "hf_cache=data/huggingface_cache"
  exit 0
fi

cd "$repo_root"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 3
fi
mkdir -p "logs/runs/$run_id" data/huggingface_cache

tmux new-session -d -s "$session" \
  "cd '$repo_root' && export HF_HOME='$repo_root/data/huggingface_cache' && set -a && source .envrc && set +a && set -o pipefail && nvidia-smi && ${command[*]} 2>&1 | tee '$log'"
echo "started tmux=$session log=$log"
