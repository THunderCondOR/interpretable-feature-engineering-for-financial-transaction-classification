# Reproducing the experiments

Start with the Rosbank commands in the [README](README.md). The main comparisons use Qwen3.5-122B-A10B and GPT-OSS-120B, training-derived reference summaries, and zero client demonstrations. Rationale generation uses temperature 0.8 and top-p 0.9. Claim extraction uses the same model with temperature 0 and top-p 1. See the [shared settings](configs/common.yaml) and [model profiles](configs/models).

## What each stage produces

| Stage | Input | Output |
| --- | --- | --- |
| `stats,prompts` | Prepared client splits | Train reference statistics, client profiles, prompts by split |
| `cot,llm_eval` | Prompts | Rationales, direct label predictions, and split metrics |
| `claims` | Rationales | Atomic claims linked to clients and source explanations |
| `ife select-features` | Run manifest and claims for all splits | Selected clusters, medoids, client features, assignments, and ML results |
| `ml` | Prepared splits and any required concept matrices | XGBoost and decision-tree metrics and predictions |

`ife run` records the input, prompt, configuration, package, and Git identities in `manifest.json`. Reuse a run directory to resume the same experiment. Use `--output-base-dir` for a changed configuration. The split-aware stages accept `--splits`. Feature construction and downstream training consume train, validation, and test together.

### Feature construction and baselines

The embedding stage deduplicates normalized claim texts and retains their client-level occurrences. Multilingual E5 receives the `query: ` prefix. The selector fits clusters on training embeddings, compares candidates on validation, and maps test claims to frozen training centroids. The default search uses average-linkage clustering for up to 45,000 unique training claims and mini-batch k-means above that limit. The minimum coverage is five distinct training clients. Use `--assignment-quantile 0.95` for cluster-specific distance caps, bounded by 0.45.

The fixed-configuration `cot_features` stage uses the clustering settings in YAML. For the validation search and multiple encodings use `ife select-features`. Its `--skip-ml` flag writes the selected representation and assignments without training the final classifiers.

| Feature set | Contents |
| --- | --- |
| `standard` / `handcrafted` | Generic aggregates / dataset-aware engineered features |
| `llm_profile` / `standard_profile` | Numeric profile statistics / aggregates plus profile statistics |
| `cot` | Claim-cluster features |
| `concat` / `standard_cot` | Claims plus handcrafted features / claims plus generic aggregates |
| `all_nonclaim` / `all_features` | All three conventional feature families / those families plus claims |

For the incremental-feature comparison, rank concepts against out-of-fold residuals of a conventional classifier, then select the number of added concepts on validation. Run `python -m interpretable_features.analysis.incremental_features --cell DIR --execute`. The candidate grid includes the conventional model alone. Use the `boosters` extra for LightGBM and CatBoost.

For LoRA, install `python -m pip install -e '.[lora,data]'` on a supported GPU system. Use `ife run --config configs/datasets/rosbank.yaml --steps lora --lora-models qwen3_8b --execute`. The model list and training settings are in the shared config.

## Stability and explanation audits

- **Stability.** Refit one randomness source at a time. The study seeds are 17, 101, and 947. Vary `generation.seed` for rationale decoding and give each run its own output directory. Use `--clustering-seed` and separate derived roots for clustering reruns. For matched-centroid cosine, client-level agreement, and exact-text anchors, use `python -m interpretable_features.analysis.compare_runs --left DIR --right DIR --axis generation_seed --output comparison.json`. For a common-anchor CSV with columns `anchor_id,run,cluster_id`, use `ife stability --input assignments.csv --axis generation_seed --output agreement.csv`.
- **Grounding.** The sampling module reads completed generation runs and binds each claim to the client profile and training summary actually used in its prompt. Run `python -m interpretable_features.analysis.sample_grounding --help` for inputs. To judge the sample, use `ife grounding-judge --input sample.jsonl --model-config configs/models/qwen.yaml --judge-name qwen --output qwen.jsonl --execute-api`. Combine independent judges' JSONL rows, then aggregate with `ife grounding --input judgments.jsonl --judges qwen gpt_oss --output grounding.json`. The summary retains disagreement and bootstraps by client.
- **Demographic audit.** The `sample_stereotypes`, `judge_stereotypes`, and `summarize_stereotypes` modules sample matched rationales, collect blinded judgments, and summarize audit rates. Each exposes its arguments via `python -m interpretable_features.analysis.MODULE --help`.
- **Attributions.** The `shap_comparison` and `shap_joint` modules use XGBoost's TreeSHAP contributions for two separate classifiers or one joint model. Pass `--cell DIR --output reports/figure --execute`. Install the `reports` extra for the figure exports.

## Surrogate fidelity

For a classification teacher, `python -m interpretable_features.analysis.fit_surrogate --help` lists the three concept matrices and three teacher-prediction tables it expects. Teacher files contain client ID, label, and one probability column per class (`teacher_prob_0`, `teacher_prob_1`, …). The module fits surrogates on train, selects them on validation, and reports held-out agreement and probability error. If both sets of probabilities are already available, `ife fidelity --input probabilities.npz --output fidelity.json` accepts arrays named `teacher_probabilities`, `surrogate_probabilities` (clients × classes), and `labels`.

For COFINFAD, prepare the data, generate claims with its dataset config, and run `ife select-features --skip-ml`. Then fit the continuous surrogate against the provider score.

```bash
features=outputs/selected/cofinfad_operational_fidelity/qwen/seed_17
python -m interpretable_features.analysis.continuous_fidelity \
  --manifest data/cofinfad_operational_fidelity/score_activity_stratified_7500_seed137/benchmark_manifest.json \
  --train-features "$features/cot_features_train.parquet" \
  --val-features "$features/cot_features_val.parquet" \
  --test-features "$features/cot_features_test.parquet" \
  --output-dir outputs/cofinfad/continuous --execute
```

Read `continuous_fidelity_metrics.json` for the validation-selected regressor and held-out R², MAE, RMSE, and correlations. For the transaction-claim evidence ablation, the `transaction_ablation` module uses cached embeddings to rebuild features from transaction-related claims.
