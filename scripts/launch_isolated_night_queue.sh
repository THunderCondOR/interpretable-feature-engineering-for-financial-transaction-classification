#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python"
RUN_ID=""
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; exit 2; }
COMMON=(--run-id "${RUN_ID}")
if [[ "${EXECUTE}" -ne 1 ]]; then
  "${PYTHON_BIN}" scripts/run_isolated_night_queue.py --model qwen "${COMMON[@]}"
  "${PYTHON_BIN}" scripts/run_isolated_night_queue.py --model gpt_oss "${COMMON[@]}"
  exit 0
fi
for variable in API_BASE_URL API_KEY; do
  [[ -n "${!variable:-}" && "${!variable}" != \$\{* ]] || {
    echo "Missing or unresolved ${variable}" >&2; exit 1;
  }
done
# tmux servers are long-lived and may retain credentials from the shell that
# originally created them.  Synchronize the resolved values before spawning
# workers; never interpolate secrets into the pane command itself.
tmux set-environment -g API_BASE_URL "${API_BASE_URL}"
tmux set-environment -g API_KEY "${API_KEY}"
for manifest in \
  data/isolated_benchmarks/datafusion_default_2023/stratified_60_20_20_seed137/benchmark_manifest.json \
  data/isolated_benchmarks/cofinfad_operational_fidelity/score_activity_stratified_7500_seed137/benchmark_manifest.json; do
  [[ -f "${manifest}" ]] || { echo "Missing prepared manifest: ${manifest}" >&2; exit 1; }
done
for model in qwen gpt_oss; do
  tmux has-session -t "${RUN_ID}_${model}" 2>/dev/null && {
    echo "Duplicate session: ${RUN_ID}_${model}" >&2; exit 1;
  }
done
for spec in \
  "data/isolated_benchmarks/datafusion_default_2023/stratified_60_20_20_seed137/benchmark_manifest.json configs/datafusion_default_2023/base.yaml" \
  "data/isolated_benchmarks/cofinfad_operational_fidelity/score_activity_stratified_7500_seed137/benchmark_manifest.json configs/cofinfad/base.yaml"; do
  read -r manifest config <<<"${spec}"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "${PYTHON_BIN}" scripts/run_isolated_prompt_pilot.py \
      --manifest "${manifest}" --base-config "${config}" --run-id "${RUN_ID}" \
      --stage materialize --execute
done
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "${PYTHON_BIN}" scripts/cv_api_probe.py \
    --model-config configs/v2/qwen.yaml --model-config configs/v2/gpt_oss.yaml --execute-api
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "${PYTHON_BIN}" scripts/isolated_api_canary.py \
    --run-id "${RUN_ID}" --execute-api --until-complete
LOG_DIR="${REPO_ROOT}/logs/runs/${RUN_ID}"
mkdir -p "${LOG_DIR}"
for model in qwen gpt_oss; do
  tmux new-session -d -c "${REPO_ROOT}" -s "${RUN_ID}_${model}" \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    bash -lc "set -o pipefail; '${PYTHON_BIN}' scripts/run_isolated_night_queue.py --model '${model}' --run-id '${RUN_ID}' --execute --execute-api --until-complete 2>&1 | tee -a '${LOG_DIR}/${model}.isolated_night.log'"
done
sleep 2
for model in qwen gpt_oss; do
  tmux has-session -t "${RUN_ID}_${model}" 2>/dev/null || {
    echo "Worker exited during startup: ${RUN_ID}_${model}" >&2; exit 1;
  }
done
echo "Started parallel model queues on the default tmux server; datasets are sequential DF2023 -> COFINFAD."
