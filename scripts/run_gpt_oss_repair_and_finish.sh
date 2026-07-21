#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
model="${GPT_OSS_MODEL:-Openai/Gpt-oss-120b}"
run_root="$repo_dir/results/gpt_oss_120b"
marker_dir="$run_root/queue_markers"
stamp="$(date +%Y%m%d_%H%M%S)"

common_args=(
  --model "$model"
  --max-concurrent "${GPT_OSS_CONCURRENCY:-100}"
  --rate-limit-fallback-concurrent "${GPT_OSS_RATE_LIMIT_FALLBACK_CONCURRENCY:-10}"
  --rate-limit-recovery-batches "${GPT_OSS_RATE_LIMIT_RECOVERY_BATCHES:-3}"
  --batch-size "${GPT_OSS_BATCH_SIZE:-100}"
  --max-tokens "${GPT_OSS_MAX_TOKENS:-8192}"
  --claims-max-tokens "${GPT_OSS_CLAIMS_MAX_TOKENS:-4096}"
)

mkdir -p "$marker_dir"

echo "Repairing rosbank prompts after stricter behavioral explanation prompt"
"$python_bin" run_pipeline.py \
  --config configs/rosbank.yaml \
  --output-base-dir "$run_root/rosbank" \
  "${common_args[@]}" \
  --steps stats,prompts \
  --splits train,val,test

for pass in 1 2 3; do
  echo "Repairing rosbank CoT pass $pass/3"
  "$python_bin" run_pipeline.py \
    --config configs/rosbank.yaml \
    --output-base-dir "$run_root/rosbank" \
    "${common_args[@]}" \
    --steps cot \
    --splits train,val,test
done

echo "Backing up old rosbank claims so they are rebuilt from repaired explanations"
for split in train val test; do
  claims="$run_root/rosbank/claims_${split}.jsonl"
  stats="$run_root/rosbank/claims_${split}.generation_stats.json"
  [[ -f "$claims" ]] && mv "$claims" "$claims.pre_repair_$stamp"
  [[ -f "$stats" ]] && mv "$stats" "$stats.pre_repair_$stamp"
done

for pass in 1 2 3; do
  echo "Rebuilding rosbank claims pass $pass/3"
  "$python_bin" run_pipeline.py \
    --config configs/rosbank.yaml \
    --output-base-dir "$run_root/rosbank" \
    "${common_args[@]}" \
    --steps claims \
    --splits train,val,test
done

"$python_bin" run_pipeline.py \
  --config configs/rosbank.yaml \
  --output-base-dir "$run_root/rosbank" \
  "${common_args[@]}" \
  --steps cot_features,ml \
  --splits train,val,test \
  --experiments standard,handcrafted,cot,concat
touch "$marker_dir/cot_rosbank.done"

for pass in 1 2 3; do
  echo "Repairing age CoT pass $pass/3"
  "$python_bin" run_pipeline.py \
    --config configs/age.yaml \
    --output-base-dir "$run_root/age" \
    "${common_args[@]}" \
    --steps cot \
    --splits train,val,test
done

for pass in 1 2 3; do
  echo "Repairing age claims pass $pass/3"
  "$python_bin" run_pipeline.py \
    --config configs/age.yaml \
    --output-base-dir "$run_root/age" \
    "${common_args[@]}" \
    --steps claims \
    --splits train,val,test
done

"$python_bin" run_pipeline.py \
  --config configs/age.yaml \
  --output-base-dir "$run_root/age" \
  "${common_args[@]}" \
  --steps cot_features,ml \
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
echo "GPT-OSS repair and finish completed."
