#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID=""
QWEN_CONFIG=""
GPT_CONFIG=""
PYTHON_BIN=""
EXECUTE=0
EXECUTE_API=0
UNTIL_COMPLETE=0

usage() {
  cat >&2 <<'EOF'
Usage: scripts/launch_model_queues.sh \
  --run-id ID \
  --qwen-config FILE \
  --gpt-config FILE \
  [--python-bin PATH] \
  [--execute --execute-api --until-complete]

Dry-run is the default and performs no API calls, tmux operations, or writes.
Real execution requires all three guards and an explicit --python-bin.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --qwen-config) QWEN_CONFIG="${2:-}"; shift 2 ;;
    --gpt-config) GPT_CONFIG="${2:-}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --execute-api) EXECUTE_API=1; shift ;;
    --until-complete) UNTIL_COMPLETE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" || -z "$QWEN_CONFIG" || -z "$GPT_CONFIG" ]]; then
  usage
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run ID must contain only letters, digits, dots, underscores, and hyphens" >&2
  exit 2
fi

resolve_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$REPO_ROOT" "$value"
  fi
}

QWEN_CONFIG="$(resolve_path "$QWEN_CONFIG")"
GPT_CONFIG="$(resolve_path "$GPT_CONFIG")"
DISPLAY_PYTHON="${PYTHON_BIN:-python}"
SESSION_PREFIX="$(printf '%s' "$RUN_ID" | tr -cs 'A-Za-z0-9_' '_' | cut -c1-48)"
SOCKET_NAME="reviewer_v2_${SESSION_PREFIX}"
QWEN_SESSION="${SESSION_PREFIX}_qwen"
GPT_SESSION="${SESSION_PREFIX}_gpt_oss"
STATUS_SESSION="${SESSION_PREFIX}_status"
BOOTSTRAP_SESSION="${SESSION_PREFIX}_bootstrap"
QWEN_LOG="logs/runs/${RUN_ID}/qwen.log"
GPT_LOG="logs/runs/${RUN_ID}/gpt_oss.log"

build_commands() {
  local python_bin="$1"
  local qwen_argv gpt_argv status_argv qwen_log_quoted gpt_log_quoted
  printf -v qwen_argv '%q ' \
    "$python_bin" scripts/run_model_queue.py \
    --run-id "$RUN_ID" --model-config "$QWEN_CONFIG" \
    --execute --execute-api --until-complete
  printf -v gpt_argv '%q ' \
    "$python_bin" scripts/run_model_queue.py \
    --run-id "$RUN_ID" --model-config "$GPT_CONFIG" \
    --execute --execute-api --until-complete
  printf -v status_argv '%q ' \
    "$python_bin" scripts/pipeline_status.py --run-id "$RUN_ID" --watch 5
  printf -v qwen_log_quoted '%q' "$QWEN_LOG"
  printf -v gpt_log_quoted '%q' "$GPT_LOG"

  # The outer bash propagates the Python exit code through tee.
  printf -v QWEN_INNER 'cd %q && %s2>&1 | tee -a %s' \
    "$REPO_ROOT" "$qwen_argv" "$qwen_log_quoted"
  printf -v GPT_INNER 'cd %q && %s2>&1 | tee -a %s' \
    "$REPO_ROOT" "$gpt_argv" "$gpt_log_quoted"
  printf -v STATUS_INNER 'cd %q && %s' "$REPO_ROOT" "$status_argv"
  printf -v QWEN_CMD 'bash -o pipefail -lc %q' "$QWEN_INNER"
  printf -v GPT_CMD 'bash -o pipefail -lc %q' "$GPT_INNER"
  printf -v STATUS_CMD 'bash -o pipefail -lc %q' "$STATUS_INNER"
}

print_scope() {
  cat <<EOF
Overnight LLM scope:
  models: 2 (Qwen, GPT-OSS)
  datasets: 3 (gender, age, rosbank)
  split cells: 9 per model
  clients: 43,400 per model
  estimated API requests: 177,200 including three prompt pilots
  tmux socket: ${SOCKET_NAME}
EOF
}

build_commands "$DISPLAY_PYTHON"
print_scope

if [[ "$EXECUTE" -eq 0 ]]; then
  printf 'DRY RUN: tmux new-session via -L %q -d -s %q %q\n' "$SOCKET_NAME" "$QWEN_SESSION" "$QWEN_CMD"
  printf 'DRY RUN: tmux new-session via -L %q -d -s %q %q\n' "$SOCKET_NAME" "$GPT_SESSION" "$GPT_CMD"
  printf 'DRY RUN: tmux new-session via -L %q -d -s %q %q\n' "$SOCKET_NAME" "$STATUS_SESSION" "$STATUS_CMD"
  exit 0
fi

if [[ "$EXECUTE_API" -ne 1 || "$UNTIL_COMPLETE" -ne 1 ]]; then
  echo "--execute requires --execute-api and --until-complete" >&2
  exit 2
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "--execute requires an explicit --python-bin PATH" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for execute mode" >&2
  exit 2
fi

build_commands "$PYTHON_BIN"

# Session collisions are checked before the network probe. A dedicated tmux
# socket keeps this run isolated without copying credentials into commands.
EXISTING_SESSIONS="$(tmux -L "$SOCKET_NAME" list-sessions -F '#{session_name}' 2>/dev/null || true)"
for session in "$QWEN_SESSION" "$GPT_SESSION" "$STATUS_SESSION"; do
  if grep -Fqx "$session" <<<"$EXISTING_SESSIONS"; then
    echo "Session already exists on socket $SOCKET_NAME: $session" >&2
    exit 1
  fi
done

# This is the only pre-launch command allowed to contact the API. It performs
# all local validation, including the clean-worktree guard, before probing.
"$PYTHON_BIN" "$REPO_ROOT/scripts/api_preflight.py" \
  --repo-root "$REPO_ROOT" \
  --run-id "$RUN_ID" \
  --qwen-config "$QWEN_CONFIG" \
  --gpt-config "$GPT_CONFIG" \
  --probe

mkdir -p "$REPO_ROOT/logs/runs/$RUN_ID"

# Keep failed panes inspectable and pass bash argv directly to tmux.  A short
# bootstrap session lets us set remain-on-exit before any worker can fail.
tmux -L "$SOCKET_NAME" new-session -d -s "$BOOTSTRAP_SESSION" "sleep 60"
tmux -L "$SOCKET_NAME" set-option -g remain-on-exit on
tmux -L "$SOCKET_NAME" new-session -d -s "$QWEN_SESSION" -c "$REPO_ROOT" \
  bash -o pipefail -lc "$QWEN_INNER"
tmux -L "$SOCKET_NAME" new-session -d -s "$GPT_SESSION" -c "$REPO_ROOT" \
  bash -o pipefail -lc "$GPT_INNER"
tmux -L "$SOCKET_NAME" new-session -d -s "$STATUS_SESSION" -c "$REPO_ROOT" \
  bash -o pipefail -lc "$STATUS_INNER"
tmux -L "$SOCKET_NAME" kill-session -t "$BOOTSTRAP_SESSION"

sleep 1
for session in "$QWEN_SESSION" "$GPT_SESSION" "$STATUS_SESSION"; do
  if [[ "$(tmux -L "$SOCKET_NAME" display-message -p -t "$session":0.0 '#{pane_dead}')" == "1" ]]; then
    echo "Session exited during launch: $session" >&2
    tmux -L "$SOCKET_NAME" capture-pane -p -t "$session":0.0 -S -80 >&2 || true
    tmux -L "$SOCKET_NAME" kill-server || true
    exit 1
  fi
done

printf 'Launched independent sessions on tmux socket %s: %s %s %s\n' \
  "$SOCKET_NAME" "$QWEN_SESSION" "$GPT_SESSION" "$STATUS_SESSION"
printf 'Attach with: tmux -L %q attach -t %q\n' "$SOCKET_NAME" "$QWEN_SESSION"
