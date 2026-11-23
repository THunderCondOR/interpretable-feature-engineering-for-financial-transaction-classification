# From Transactions to Concepts

Companion code for *From Transactions to Concepts: Distilling LLM Rationales into Semantic Features for Transaction Modeling* by Mikhail Chaichuk, Elena Tutubalina, Andrey Savchenko, and Alexander Panchenko.

The pipeline turns a client's transaction history into a row of interpretable features. An LLM reads a transaction profile and training-set statistics, predicts a label, and explains its prediction. A second pass extracts short behavioral claims. Similar training claims form a concept, represented by an example claim from its cluster. A downstream classifier can use the concepts on their own or alongside conventional transaction features.

## Quick start

Run the examples from the repository root with Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,reports,dev]'
ife report --synthetic --output reports/demo.html
```

Open `reports/demo.html` to explore a synthetic cluster map and model comparisons. The demo runs offline after installation. For a first experiment on real data, prepare Rosbank and evaluate the training-set majority baseline on its 500 held-out clients.

```bash
ife prepare --dataset rosbank --execute
ife run --config configs/datasets/rosbank.yaml --steps majority --splits test --execute
```

The result is `outputs/rosbank/metrics_majority.json`. The same split supports a conventional ML baseline with `--steps ml --experiments standard --execute`. Dataset preparation may download the public source files. The six datasets, split definitions, and inputs are described in the [data guide](DATASETS.md).

## Reproduce the semantic features

Configure an OpenAI-compatible endpoint that serves the selected model. Copy `.env.example` to `.env`, fill in the endpoint and key, and export them in the current shell.

```bash
set -a
source .env
set +a
```

Generate profiles, rationales, direct predictions, and atomic claims for all three splits. This command sends client profiles to the configured provider. Omit `--execute` to inspect the plan before making requests.

```bash
ife run --config configs/datasets/rosbank.yaml \
  --model-config configs/models/qwen.yaml \
  --steps stats,prompts,cot,llm_eval,claims \
  --execute --until-complete
```

Fit candidate concept spaces on training claims and select the representation on validation clients. The final classifier uses the selected features to predict on held-out clients. This stage reuses the saved claims. The first run may download the embedding model.

```bash
ife select-features --source-root outputs/rosbank/qwen \
  --derived-root outputs/selected --assignment-quantile 0.95 \
  --ml-experiments standard cot all_nonclaim all_features --execute
```

In `outputs/selected/rosbank/qwen/seed_17/`, open `cot_clusters.json` for representative claims, `cluster_selection.json` for the validation decision, and `ml_metrics.json` for downstream results. Use the GPT-OSS profile to repeat generation with the other model. The [experiment guide](EXPERIMENTS.md) covers alternative feature sets, stability, grounding, surrogate fidelity, and TreeSHAP.

## Repository guide

| Location | Contents |
| --- | --- |
| [configs](configs) | Shared protocol, dataset settings, and model profiles |
| [prompts](prompts/common) | Exact rationale-generation and claim-extraction templates |
| [src/interpretable_features](src/interpretable_features) | Data adapters, pipeline, downstream models, and the `ife` CLI |
| [analysis](src/interpretable_features/analysis) | Validation-based feature selection and paper analyses |
| [tests](tests) | Split handling, prompt contracts, train-only transforms, resume behavior, and evaluation |

Run `python -m pytest` and `ruff check .` to check the checkout. See the [citation metadata](CITATION.cff) and [MIT license](LICENSE).
