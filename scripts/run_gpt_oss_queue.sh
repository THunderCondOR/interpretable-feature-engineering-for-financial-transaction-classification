#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
model="${GPT_OSS_MODEL:-Openai/Gpt-oss-120b}"
concurrency="${GPT_OSS_CONCURRENCY:-32}"
batch_size="${GPT_OSS_BATCH_SIZE:-32}"
explanation_tokens="${GPT_OSS_MAX_TOKENS:-8192}"
claims_tokens="${GPT_OSS_CLAIMS_MAX_TOKENS:-4096}"
datasets=(gender rosbank age)
run_root="$repo_dir/results/gpt_oss_120b"
marker_dir="$run_root/queue_markers"

mkdir -p "$marker_dir"
cd "$repo_dir"

common_args=(
  --model "$model"
  --max-concurrent "$concurrency"
  --batch-size "$batch_size"
  --max-tokens "$explanation_tokens"
  --claims-max-tokens "$claims_tokens"
)

# Phase 1: generate and distill CoT for every dataset, strictly sequentially.
for dataset in "${datasets[@]}"; do
  if [[ -f "$marker_dir/cot_${dataset}.done" ]]; then
    echo "Skipping CoT phase for $dataset: $marker_dir/cot_${dataset}.done exists"
    continue
  fi

  output_dir="$run_root/$dataset"
  "$python_bin" run_pipeline.py \
    --config "configs/$dataset.yaml" \
    --output-base-dir "$output_dir" \
    "${common_args[@]}" \
    --steps stats,prompts \
    --splits train,val,test \
    --experiments cot,concat

  # The first pass generates the full set. Two cheap resume passes retry only
  # incomplete responses or outputs that could not be parsed.
  for pass in 1 2 3; do
    echo "Starting CoT pass $pass/3 for $dataset"
    "$python_bin" run_pipeline.py \
      --config "configs/$dataset.yaml" \
      --output-base-dir "$output_dir" \
      "${common_args[@]}" \
      --steps cot \
      --splits train,val,test \
      --experiments cot,concat
  done

  for pass in 1 2 3; do
    echo "Starting claims pass $pass/3 for $dataset"
    "$python_bin" run_pipeline.py \
      --config "configs/$dataset.yaml" \
      --output-base-dir "$output_dir" \
      "${common_args[@]}" \
      --steps claims \
      --splits train,val,test
  done

  "$python_bin" run_pipeline.py \
    --config "configs/$dataset.yaml" \
    --output-base-dir "$output_dir" \
    "${common_args[@]}" \
    --steps cot_features,ml \
    --splits train,val,test \
    --experiments standard,handcrafted,cot,concat
  touch "$marker_dir/cot_${dataset}.done"
done

# Phase 2: locally evaluate saved test responses and the majority baseline.
for dataset in "${datasets[@]}"; do
  if [[ -f "$marker_dir/baseline_${dataset}.done" ]]; then
    echo "Skipping baseline phase for $dataset: $marker_dir/baseline_${dataset}.done exists"
    continue
  fi

  output_dir="$run_root/$dataset"
  "$python_bin" run_pipeline.py \
    --config "configs/$dataset.yaml" \
    --output-base-dir "$output_dir" \
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
echo "GPT-OSS queue completed."
