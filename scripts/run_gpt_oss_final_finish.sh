#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
run_root="$repo_dir/results/gpt_oss_120b"
marker_dir="$run_root/queue_markers"

common_args=(
  --model "${GPT_OSS_MODEL:-Openai/Gpt-oss-120b}"
  --max-concurrent "${GPT_OSS_CONCURRENCY:-100}"
  --rate-limit-fallback-concurrent "${GPT_OSS_RATE_LIMIT_FALLBACK_CONCURRENCY:-10}"
  --rate-limit-recovery-batches "${GPT_OSS_RATE_LIMIT_RECOVERY_BATCHES:-3}"
  --batch-size "${GPT_OSS_BATCH_SIZE:-100}"
  --max-tokens "${GPT_OSS_MAX_TOKENS:-8192}"
  --claims-max-tokens "${GPT_OSS_CLAIMS_MAX_TOKENS:-4096}"
)

mkdir -p "$marker_dir"

"$python_bin" run_pipeline.py \
  --config configs/age.yaml \
  --output-base-dir "$run_root/age" \
  "${common_args[@]}" \
  --steps ml \
  --splits train,val,test \
  --experiments standard,handcrafted,cot,concat
touch "$marker_dir/cot_age.done"

for dataset in gender rosbank age; do
  echo "Running GPT-OSS baseline/majority for $dataset"
  "$python_bin" run_pipeline.py \
    --config "configs/$dataset.yaml" \
    --output-base-dir "$run_root/$dataset" \
    "${common_args[@]}" \
    --steps llm_eval,majority \
    --splits train,val,test
  touch "$marker_dir/baseline_${dataset}.done"
done

"$python_bin" scripts/build_table1_summary.py \
  --run-root "$run_root" \
  --reference-root "$repo_dir/results" \
  --output "$run_root/table1_metrics.json"

touch "$marker_dir/all.done"
echo "GPT-OSS final finish completed."
