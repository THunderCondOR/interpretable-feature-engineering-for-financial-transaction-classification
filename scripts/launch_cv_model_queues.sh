#!/usr/bin/env bash
set -euo pipefail

RUN_ID="reviewer-v5-fixed-new-datasets"
PYTHON_BIN="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
EXECUTE=0
DATASETS="berka,datafusion_education"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "Run ID: ${RUN_ID}"
echo "Datasets: ${DATASETS}"

if [[ "${EXECUTE}" -ne 1 ]]; then
  "${PYTHON_BIN}" scripts/run_cv_llm_queue.py --model qwen --run-id "${RUN_ID}" --datasets "${DATASETS}"
  "${PYTHON_BIN}" scripts/run_cv_llm_queue.py --model gpt_oss --run-id "${RUN_ID}" --datasets "${DATASETS}"
  exit 0
fi

for variable in API_BASE_URL API_KEY; do
  if [[ -z "${!variable:-}" || "${!variable}" == "\${"* ]]; then
    echo "Missing or unresolved ${variable}" >&2
    exit 1
  fi
done

SOCKET="cv_${RUN_ID//[^a-zA-Z0-9_-]/_}"
mkdir -p "logs/runs/${RUN_ID}"
"${PYTHON_BIN}" scripts/cv_preflight.py --execute
"${PYTHON_BIN}" scripts/cv_api_probe.py \
  --model-config configs/v2/qwen.yaml \
  --model-config configs/v2/gpt_oss.yaml \
  --execute-api
for session in "${RUN_ID}_qwen" "${RUN_ID}_gpt_oss" "${RUN_ID}_status"; do
  if tmux -L "${SOCKET}" has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi
done

tmux -L "${SOCKET}" new-session -d -s "${RUN_ID}_qwen" \
  "set -o pipefail; '${PYTHON_BIN}' scripts/run_cv_queue_watchdog.py --python-bin '${PYTHON_BIN}' --model qwen --run-id '${RUN_ID}' --datasets '${DATASETS}' 2>&1 | tee -a 'logs/runs/${RUN_ID}/qwen.cv_queue.log'"
tmux -L "${SOCKET}" new-session -d -s "${RUN_ID}_gpt_oss" \
  "set -o pipefail; '${PYTHON_BIN}' scripts/run_cv_queue_watchdog.py --python-bin '${PYTHON_BIN}' --model gpt_oss --run-id '${RUN_ID}' --datasets '${DATASETS}' 2>&1 | tee -a 'logs/runs/${RUN_ID}/gpt_oss.cv_queue.log'"
tmux -L "${SOCKET}" new-session -d -s "${RUN_ID}_status" \
  "'${PYTHON_BIN}' scripts/pipeline_status.py --run-id '${RUN_ID}' --results-root 'results/v5/runs/${RUN_ID}' --watch 5"

echo "Started tmux socket ${SOCKET}: ${RUN_ID}_qwen, ${RUN_ID}_gpt_oss, ${RUN_ID}_status"
