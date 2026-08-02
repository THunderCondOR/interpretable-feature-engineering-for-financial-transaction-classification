#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
RUN_ID=""
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; exit 2; }
if [[ "${EXECUTE}" -ne 1 ]]; then
  exec "${PYTHON_BIN}" scripts/run_isolated_offline_queue.py --run-id "${RUN_ID}"
fi
SESSION="${RUN_ID}_offline"
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Duplicate session: ${SESSION}" >&2; exit 1; }
LOG_DIR="${REPO_ROOT}/logs/runs/${RUN_ID}"
mkdir -p "${LOG_DIR}"
tmux new-session -d -c "${REPO_ROOT}" -s "${SESSION}" \
  bash -lc "set -o pipefail; '${PYTHON_BIN}' scripts/run_isolated_offline_queue.py --run-id '${RUN_ID}' --execute 2>&1 | tee -a '${LOG_DIR}/offline.log'"
sleep 2
tmux has-session -t "${SESSION}" 2>/dev/null || { echo "Offline waiter exited during startup" >&2; exit 1; }
echo "Started ${SESSION}; it waits for API completions and the current LoRA GPU session."
