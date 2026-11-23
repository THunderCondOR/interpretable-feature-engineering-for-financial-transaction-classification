# Data and splits

Run `ife prepare --dataset NAME --execute` from the repository root. Preparation writes under `data/` by default. A dry run prints the planned directories. If you use a different output root, update the input paths in your experiment configuration.

| Dataset | Prediction target | Paper split (train / validation / test) | Primary measure |
| --- | --- | --- | --- |
| Rosbank | Client churn | 4,000 / 500 / 500 | ROC-AUC |
| Data Fusion 2023 | Credit default | 4,248 / 1,416 / 1,416 | ROC-AUC |
| Berka | Loan outcome | 478 outer-train / 204 outer-test, five repetitions | Positive-class F1 |
| Gender | Binary demographic benchmark | 6,720 / 840 / 840 | ROC-AUC |
| Age | Four age groups | 8,000 / 1,000 / 3,000 | Accuracy |
| COFINFAD | Continuous provider churn score | 4,500 / 1,500 / 1,500 | R² |

**Rosbank, Gender, and Age.** Preparation downloads the labelled data at pinned revisions. The bundled holdout definitions store positions in sorted public client IDs. The adapter checks the population hash before applying those positions. It writes separate split CSVs, a split report, and the Age subsample IDs. Age uses a stratified sample of training and validation clients with seed 137 and keeps the full training split for prompt reference statistics. To reproduce a different holdout, pass `--test-ids-dir DIR` with `rosbank_test_ids.csv` (column `cl_id`), `gender_test_ids.csv` (`customer_id`), or `age_test_ids.csv` (`client_id`). [Rosbank source](https://huggingface.co/datasets/pytorch-lifestream/rosbank-churn) · [Gender source](https://huggingface.co/datasets/pytorch-lifestream/transactions-gender) · [Age source](https://huggingface.co/datasets/pytorch-lifestream/age-group-prediction)

**Berka.** Put the comma-separated `loan.csv` and `trans.csv` tables with ISO dates in a raw directory. Run `ife prepare --dataset berka --raw-root data/raw/berka --execute`. Only pre-origination events enter the account history and loan-payment events are excluded. The five repeated 70/30 splits use seeds 9, 17, 101, 137, and 947. Use `ife run --config configs/datasets/berka.yaml --fold 0` through `--fold 4`. Each run draws 100 validation loans from the 478 outer training loans, leaving 378 for the inner fit. [CTU Financial dataset](https://relational.fel.cvut.cz/dataset/Financial)

**Data Fusion 2023 and COFINFAD.** The adapters retrieve checksum-pinned source files and write a shared event table, split ID lists, and a benchmark manifest. The dataset names for preparation are `datafusion_default_2023` and `cofinfad`. Data Fusion retains the original currency for each transaction. COFINFAD combines transactions with operational fields. Its training-score quartiles define the labels for generating rationales. The final fidelity evaluation uses the continuous scores saved separately in `teacher_scores.csv`. [Data Fusion competition](https://ods.ai/competitions/data-fusion2023-attack/Dataset) · [COFINFAD source](https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad)

## Prepared event tables

The loader reads CSV or Parquet. The required fields are `customer_id`, `tr_datetime`, `amount`, `mcc_code_desc`, and `label`. A dataset configuration can map different source column names to these fields. Keep all events for a client in one split. The adapter retains dataset-specific fields such as currency, operation type, account balance, or operational measures. The dataset configuration defines the units, transaction direction, and target classes used in prompts.
