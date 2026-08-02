#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
RUN_ID=""
DATASET=""
MANIFEST=""
BASE_CONFIG=""
EXECUTE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_ID}" || -z "${DATASET}" || -z "${MANIFEST}" || -z "${BASE_CONFIG}" ]]; then
  echo "--run-id, --dataset, --manifest, and --base-config are required" >&2
  exit 2
fi

COMMON=(--manifest "${MANIFEST}" --base-config "${BASE_CONFIG}" --run-id "${RUN_ID}")
if [[ "${EXECUTE}" -ne 1 ]]; then
  "${PYTHON_BIN}" scripts/run_isolated_model_queue.py --model qwen "${COMMON[@]}"
  "${PYTHON_BIN}" scripts/run_isolated_model_queue.py --model gpt_oss "${COMMON[@]}"
  exit 0
fi

for variable in API_BASE_URL API_KEY; do
  if [[ -z "${!variable:-}" || "${!variable}" == "\${"* ]]; then
    echo "Missing or unresolved ${variable}" >&2
    exit 1
  fi
done
# Avoid inheriting stale credentials from a long-lived tmux server.
tmux set-environment -g API_BASE_URL "${API_BASE_URL}"
tmux set-environment -g API_KEY "${API_KEY}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }

for session in "${RUN_ID}_qwen" "${RUN_ID}_gpt_oss"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi
done

LOG_DIR="${REPO_ROOT}/logs/runs/${RUN_ID}"
mkdir -p "${LOG_DIR}"
tmux new-session -d -c "${REPO_ROOT}" -s "${RUN_ID}_qwen" \
  bash -lc "set -o pipefail; '${PYTHON_BIN}' scripts/run_isolated_model_queue.py --model qwen --manifest '${MANIFEST}' --base-config '${BASE_CONFIG}' --run-id '${RUN_ID}' --execute --execute-api --until-complete 2>&1 | tee -a '${LOG_DIR}/qwen.log'"
tmux new-session -d -c "${REPO_ROOT}" -s "${RUN_ID}_gpt_oss" \
  bash -lc "set -o pipefail; '${PYTHON_BIN}' scripts/run_isolated_model_queue.py --model gpt_oss --manifest '${MANIFEST}' --base-config '${BASE_CONFIG}' --run-id '${RUN_ID}' --execute --execute-api --until-complete 2>&1 | tee -a '${LOG_DIR}/gpt_oss.log'"

sleep 2
for session in "${RUN_ID}_qwen" "${RUN_ID}_gpt_oss"; do
  tmux has-session -t "${session}" 2>/dev/null || {
    echo "Worker exited during startup: ${session}" >&2
    exit 1
  }
done
echo "Started ${DATASET} on the default tmux server: sessions=${RUN_ID}_qwen,${RUN_ID}_gpt_oss"
