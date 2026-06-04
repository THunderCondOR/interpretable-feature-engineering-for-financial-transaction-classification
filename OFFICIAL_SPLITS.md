# Data preparation workflow

The repository uses fixed client-level test identifiers for all supported datasets.
There is no random local test split.

## Inputs

Place the provided files under:

```text
data/test_ids/gender_test_ids.csv
data/test_ids/age_test_ids.csv
data/test_ids/rosbank_test_ids.csv
```

Expected id columns:

| Dataset | Test id column | Label source |
|---|---|---|
| gender | `customer_id` | `pytorch-lifestream/transactions-gender`, `labels` config |
| age | `client_id` | `pytorch-lifestream/age-group-prediction`, `train_target` config |
| rosbank | `cl_id` | `pytorch-lifestream/rosbank-churn`, labeled train config |

## Prepare data

```bash
python prepare_data.py --dataset all
```

This writes:

```text
data/<dataset>/train.csv
data/<dataset>/val.csv
data/<dataset>/test.csv
data/<dataset>/split_report.json
data/<dataset>/split_summary.txt
```

The prepared CSV files use the internal schema:

```text
customer_id, tr_datetime, amount, mcc_code_desc, label
```

Rosbank keeps extra columns used by handcrafted features:

```text
currency_name, trx_cat_ru, mcc_desc
```

## Validate data

```bash
python validate_splits.py --dataset all
```

The validator checks that train, validation, and test client ids are disjoint and that `test.csv` exactly matches the provided test-id file.

## Split policy

For every dataset:

1. download the labeled source split;
2. normalize columns to the internal schema;
3. move provided test ids to `test.csv`;
4. split all remaining labeled clients into train and validation with stratification by label.
