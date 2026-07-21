#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python}"
model="${GPT_OSS_MODEL:-Openai/Gpt-oss-120b}"
run_root="$repo_dir/results/gpt_oss_120b"
marker_dir="$run_root/queue_markers"

echo "Waiting for $marker_dir/cot_age.done"
while [[ ! -f "$marker_dir/cot_age.done" ]]; do
  sleep 60
done

echo "Rebuilding rosbank CoT features/ML after claims repair"
"$python_bin" run_pipeline.py \
  --config configs/rosbank.yaml \
  --output-base-dir "$run_root/rosbank" \
  --model "$model" \
  --steps cot_features,ml \
  --splits train,val,test \
  --experiments standard,handcrafted,cot,concat

for dataset in gender rosbank age; do
  echo "Running GPT-OSS baseline/majority for $dataset"
  "$python_bin" run_pipeline.py \
    --config "configs/$dataset.yaml" \
    --output-base-dir "$run_root/$dataset" \
    --model "$model" \
    --steps llm_eval,majority \
    --splits train,val,test
  touch "$marker_dir/baseline_${dataset}.done"
done

"$python_bin" scripts/build_table1_summary.py \
  --run-root "$run_root" \
  --reference-root "$repo_dir/results" \
  --output "$run_root/table1_metrics.json"

touch "$marker_dir/all.done"
echo "GPT-OSS finish-after-age completed."
