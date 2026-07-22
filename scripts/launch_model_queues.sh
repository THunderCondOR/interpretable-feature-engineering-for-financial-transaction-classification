#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
QWEN_CONFIG=""
GPT_CONFIG=""
EXECUTE=0
UNTIL_COMPLETE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --qwen-config) QWEN_CONFIG="$2"; shift 2 ;;
    --gpt-config) GPT_CONFIG="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --until-complete) UNTIL_COMPLETE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" && -n "$QWEN_CONFIG" && -n "$GPT_CONFIG" ]] || {
  echo "Usage: $0 --run-id ID --qwen-config FILE --gpt-config FILE [--execute --until-complete]" >&2
  exit 2
}

SESSION_PREFIX="${RUN_ID//-/_}"
QWEN_SESSION="${SESSION_PREFIX}_qwen"
GPT_SESSION="${SESSION_PREFIX}_gpt_oss"
STATUS_SESSION="${SESSION_PREFIX}_status"
QWEN_CMD="python scripts/run_model_queue.py --run-id $RUN_ID --model-config $QWEN_CONFIG --execute --until-complete 2>&1 | tee -a logs/runs/$RUN_ID/qwen.log"
GPT_CMD="python scripts/run_model_queue.py --run-id $RUN_ID --model-config $GPT_CONFIG --execute --until-complete 2>&1 | tee -a logs/runs/$RUN_ID/gpt_oss.log"
STATUS_CMD="python scripts/pipeline_status.py --run-id $RUN_ID --watch 5"

if [[ "$EXECUTE" -eq 0 ]]; then
  printf 'DRY RUN: tmux new-session -d -s %s %q\n' "$QWEN_SESSION" "$QWEN_CMD"
  printf 'DRY RUN: tmux new-session -d -s %s %q\n' "$GPT_SESSION" "$GPT_CMD"
  printf 'DRY RUN: tmux new-session -d -s %s %q\n' "$STATUS_SESSION" "$STATUS_CMD"
  exit 0
fi

[[ "$UNTIL_COMPLETE" -eq 1 ]] || { echo "--execute requires --until-complete" >&2; exit 2; }
mkdir -p "logs/runs/$RUN_ID"
for session in "$QWEN_SESSION" "$GPT_SESSION" "$STATUS_SESSION"; do
  tmux has-session -t "$session" 2>/dev/null && { echo "Session already exists: $session" >&2; exit 1; }
done
tmux new-session -d -s "$QWEN_SESSION" "$QWEN_CMD"
tmux new-session -d -s "$GPT_SESSION" "$GPT_CMD"
tmux new-session -d -s "$STATUS_SESSION" "$STATUS_CMD"
printf 'Launched independent sessions: %s %s %s\n' "$QWEN_SESSION" "$GPT_SESSION" "$STATUS_SESSION"
