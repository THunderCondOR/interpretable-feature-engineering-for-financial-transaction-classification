#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="reviewer-v9-grounding-official-ready"
execute=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --execute) execute=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "run ID may contain only letters, digits, dot, underscore and dash" >&2
  exit 2
fi

session="${run_id//[^a-zA-Z0-9_]/_}"
command=(
  /home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python
  scripts/run_grounding_suite.py
  --run-id "$run_id"
  --datasets rosbank berka gender age
  --max-openrouter-cost-usd 5
  --max-concurrent 32
  --execute --execute-api --until-complete
)

if [[ $execute -eq 0 ]]; then
  printf 'mode=dry-run\nsession=%s\nrun_id=%s\nclaims=840\njudgments=1680\nestimated_cost_usd=3.30\n' \
    "$session" "$run_id"
  exit 0
fi

cd "$repo_root"
set -a
source .envrc
set +a
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo 'OPENROUTER_API_KEY is empty; set it in .envrc before --execute.' >&2
  exit 3
fi
# Grounding must use the dedicated local proxy, independently of any system or
# shell-wide proxy configuration.
export OPENROUTER_PROXY_URL="http://127.0.0.1:5300"
if ! "${command[0]}" -c 'import socket; s=socket.create_connection(("127.0.0.1", 5300), 2); s.close()'; then
  echo 'Local OpenRouter proxy is not listening on 127.0.0.1:5300' >&2
  exit 5
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 4
fi

mkdir -p "logs/runs/$run_id"
# The secret is loaded inside the child shell and is never included in tmux's
# command line or logs.
tmux new-session -d -s "$session" \
  "cd '$repo_root' && set -a && source .envrc && set +a && export OPENROUTER_PROXY_URL='http://127.0.0.1:5300' && set -o pipefail && ${command[*]} 2>&1 | tee 'logs/runs/$run_id/grounding.log'"
echo "started tmux=$session log=logs/runs/$run_id/grounding.log"
