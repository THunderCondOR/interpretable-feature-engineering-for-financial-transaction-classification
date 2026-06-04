# Interpretable feature engineering for financial transaction classification

This repository contains a split-aware pipeline for experiments on financial transaction classification datasets.

Supported datasets:

- `gender`
- `age`
- `rosbank`

All datasets use fixed client-level test identifiers from `data/test_ids`. The test split is never sampled randomly.

## 1. Prepare data

Place the provided files under:

```text
data/test_ids/gender_test_ids.csv
data/test_ids/age_test_ids.csv
data/test_ids/rosbank_test_ids.csv
```

Then run:

```bash
python prepare_data.py --dataset all
python validate_splits.py --dataset all
```

This creates:

```text
data/<dataset>/train.csv
data/<dataset>/val.csv
data/<dataset>/test.csv
data/<dataset>/split_report.json
data/<dataset>/split_summary.txt
```

## 2. Run experiments

Each dataset is controlled by a config in `configs/`.

### Handcrafted classical baselines

```bash
python run_pipeline.py --config configs/gender.yaml --steps ml --experiments handcrafted
python run_pipeline.py --config configs/age.yaml --steps ml --experiments handcrafted
python run_pipeline.py --config configs/rosbank.yaml --steps ml --experiments handcrafted
```

### Direct LLM baseline

```bash
python run_pipeline.py --config configs/gender.yaml --steps stats,prompts,cot,llm_eval --splits test
```

The same command works for `age` and `rosbank` by changing the config path.

### CoT feature pipeline

```bash
python run_pipeline.py --config configs/gender.yaml --steps stats,prompts,cot,claims,cot_features,ml --splits train,val,test --experiments cot,concat
```

The CoT feature builder fits clusters on train claims only and assigns validation/test claims to train-fitted clusters.

### LoRA fine-tuning

```bash
python run_pipeline.py --config configs/gender.yaml --steps lora
```

## 3. Main outputs

```text
results/<dataset>/summary_stats.txt
results/<dataset>/prompts_<split>.jsonl
results/<dataset>/explanations_<split>.jsonl
results/<dataset>/llm_metrics_<split>.json
results/<dataset>/claims_<split>.jsonl
results/<dataset>/cot_features_<split>.parquet
results/<dataset>/cot_clusters.json
results/<dataset>/ml_metrics.json
results/<dataset>/metrics.json
```
