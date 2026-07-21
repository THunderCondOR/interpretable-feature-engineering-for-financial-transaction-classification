#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
model="${GPT_OSS_MODEL:-Openai/Gpt-oss-120b}"
concurrency="${GPT_OSS_CONCURRENCY:-64}"
batch_size="${GPT_OSS_BATCH_SIZE:-64}"
output_dir="$repo_dir/results/gpt_oss_120b/age"

common_args=(
  --config configs/age.yaml
  --output-base-dir "$output_dir"
  --model "$model"
  --max-concurrent "$concurrency"
  --rate-limit-fallback-concurrent "${GPT_OSS_RATE_LIMIT_FALLBACK_CONCURRENCY:-10}"
  --rate-limit-recovery-batches "${GPT_OSS_RATE_LIMIT_RECOVERY_BATCHES:-3}"
  --batch-size "$batch_size"
  --max-tokens "${GPT_OSS_MAX_TOKENS:-8192}"
  --claims-max-tokens "${GPT_OSS_CLAIMS_MAX_TOKENS:-4096}"
)

for pass in 1 2 3; do
  echo "Starting age CoT pass $pass/3"
  "$python_bin" run_pipeline.py "${common_args[@]}" \
    --steps cot \
    --splits train,val,test \
    --experiments cot,concat
done

for pass in 1 2 3; do
  echo "Starting age claims pass $pass/3"
  "$python_bin" run_pipeline.py "${common_args[@]}" \
    --steps claims \
    --splits train,val,test
done

"$python_bin" run_pipeline.py "${common_args[@]}" \
  --steps cot_features,ml \
  --splits train,val,test \
  --experiments standard,handcrafted,cot,concat

touch "$repo_dir/results/gpt_oss_120b/queue_markers/cot_age.done"
