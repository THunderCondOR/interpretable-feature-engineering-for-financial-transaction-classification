# Data Fusion Education and Berka benchmark pipeline

The v5 pipeline adds two transaction-only, fold-evaluated main datasets without
changing legacy Gender, Age, or Rosbank artifacts.

## Protocols

- `datafusion_education / mbd_5fold_seed42`: 8,509 labeled clients, five
  MBD-style folds, transaction-only input, ROC-AUC primary.
- `berka / unittab_70_30_5seed`: 682 loans, five 478/204 repeated splits,
  A/C versus B/D, positive-class F1 primary. Transactions are cut strictly
  before loan origination and `UVER` loan-payment records are removed.

Every outer fold has a separate `inner_train` and `inner_validation`. Qwen
selects zero-shot, factual FS1, or factual FS2 inside that fold. The selected
format is then frozen for both Qwen and GPT-OSS. Outer-test labels are never
used in summaries, demonstrations, clustering, feature selection, or model
tuning.

## Commands

All runners are dry-run by default.

```bash
python scripts/prepare_benchmark_dataset.py \
  --dataset berka --execute

python scripts/prepare_benchmark_dataset.py \
  --dataset datafusion_education \
  --download --split-backend pyspark --execute

bash scripts/launch_cv_model_queues.sh \
  --run-id reviewer-v5-fixed-new-datasets

bash scripts/launch_cv_model_queues.sh \
  --run-id reviewer-v5-fixed-new-datasets --execute

python scripts/run_cv_offline_pipeline.py \
  --datasets berka,datafusion_education \
  --models qwen,gpt_oss --execute

python scripts/summarize_cv_benchmarks.py \
  --run-id reviewer-v5-fixed-new-datasets --execute
```

The exact Data Fusion reference preparation requires `pyspark==3.3.3`.
`--split-backend sklearn_approx` is provided only for diagnostics and its
results must not be presented as an exact MBD reproduction. The Berka
comparison is protocol-matched but not ID-identical because UniTTab does not
publish the five test-ID lists.

## Output boundaries

API artifacts:

```text
results/v5/runs/<run_id>/<dataset>/<protocol>/fold_<n>/<variant>/<model>/seed_17/
```

Fold-fitted clusters and ML outputs:

```text
results/v5/derived/cv_main/<dataset>/<protocol>/fold_<n>/<model>/
```

The API stage writes explanations, direct predictions, and atomic claims only.
The offline stage selects clustering settings on inner validation, refits the
semantic space on the full outer train, and assigns outer-test claims to frozen
outer-train centroids.
