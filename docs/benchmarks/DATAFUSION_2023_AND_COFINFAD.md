# Isolated reviewer benchmarks: provenance and comparison protocol

## Data Fusion Contest 2023 — default prediction

Source dataset: [Data Fusion Contest 2023 — credit default prediction](https://ods.ai/competitions/data-fusion2023-attack/Dataset).
The repository pins every downloaded archive by SHA-256 in
`src/benchmarks/datafusion_default_2023.py`.

The released labelled population contains 7,080 clients and 2,124,000
transactions (300 per client), with 262 positive targets. The original contest
test set was closed and was explicitly described as a new-data evaluation set
on the [competition site](https://ods.ai/competitions/data-fusion2023-defence/dataset).
No public client-ID list with labels was found that would permit an exact
reconstruction of that hidden test evaluation.

We therefore use a fixed, stratified 60/20/20 split with seed 137. The exact
client IDs and hashes are written to the benchmark manifest. Published contest
scores are context, not an exact same-split comparison. Comparable numbers in
the paper must come from baselines reproduced on these same fixed IDs.

Two open reference implementations are recorded:

- the official released RNN checkpoint and preprocessing, downloaded with the
  dataset and reproduced by `scripts/run_datafusion_default_baselines.py`;
- the [KonderLip open solution](https://github.com/KonderLip/data-fusion2023-defence),
  which combines the official RNN prediction with temporal and MCC aggregates
  in CatBoost and reports CV ROC-AUC. Its CV value is not presented as our
  fixed-split test value.

The official model contains functional dropout that remains active in eval
mode. Our wrapper preserves this published behavior and reports the inference
seed as a sensitivity axis.

Data Fusion Contest 2022 Education remains a separate legacy benchmark and its
five-fold infrastructure is not modified by this implementation.

## COFINFAD — natural score fidelity

Source dataset: [COFINFAD on Hugging Face](https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad),
pinned to revision `f7b6a9f45bd75fba9f791238dee85c5028ef5b19` with file
hashes in `src/benchmarks/cofinfad.py`.

`churn_probability` is treated as an already produced model/score output, not
as observed churn ground truth. The primary 7,500-client protocol predicts
train-defined score quartiles from transactions plus operational evidence:
product use, application engagement, satisfaction, and support. Demographics,
the published score, customer lifetime value, and target-adjacent segments are
excluded from all LLM prompts.

The split is 4,500/1,500/1,500, stratified by published score and activity.
Continuous-score surrogate fidelity is then evaluated by
`scripts/run_cofinfad_continuous_fidelity.py`, including R², MAE, RMSE,
Pearson/Spearman agreement and teacher-confidence/coverage curves.

The exact duplicate transaction rows published by the source are retained and
counted in the manifest. Transaction-only features are a negative control;
full-profile features are an offline diagnostic ceiling, not the primary
interpretability result.
