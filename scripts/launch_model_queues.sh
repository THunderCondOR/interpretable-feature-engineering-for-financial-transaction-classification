#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
QWEN_CONFIG=""
GPT_CONFIG=""
PYTHON_BIN=""
EXECUTE=0
UNTIL_COMPLETE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --qwen-config) QWEN_CONFIG="$2"; shift 2 ;;
    --gpt-config) GPT_CONFIG="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --until-complete) UNTIL_COMPLETE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ID" && -n "$QWEN_CONFIG" && -n "$GPT_CONFIG" ]] || {
  echo "Usage: $0 --run-id ID --qwen-config FILE --gpt-config FILE [--python-bin PATH] [--execute --until-complete]" >&2
  exit 2
}

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No Python interpreter found; pass --python-bin PATH" >&2
    exit 2
  fi
fi

SESSION_PREFIX="${RUN_ID//-/_}"
QWEN_SESSION="${SESSION_PREFIX}_qwen"
GPT_SESSION="${SESSION_PREFIX}_gpt_oss"
STATUS_SESSION="${SESSION_PREFIX}_status"
printf -v QWEN_CMD '%q ' "$PYTHON_BIN" scripts/run_model_queue.py --run-id "$RUN_ID" --model-config "$QWEN_CONFIG" --execute --until-complete
printf -v GPT_CMD '%q ' "$PYTHON_BIN" scripts/run_model_queue.py --run-id "$RUN_ID" --model-config "$GPT_CONFIG" --execute --until-complete
printf -v STATUS_CMD '%q ' "$PYTHON_BIN" scripts/pipeline_status.py --run-id "$RUN_ID" --watch 5
printf -v QWEN_LOG '%q' "logs/runs/$RUN_ID/qwen.log"
printf -v GPT_LOG '%q' "logs/runs/$RUN_ID/gpt_oss.log"
QWEN_CMD+="2>&1 | tee -a $QWEN_LOG"
GPT_CMD+="2>&1 | tee -a $GPT_LOG"

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
