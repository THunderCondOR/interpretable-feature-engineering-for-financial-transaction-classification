# Official test-id split workflow

This branch replaces random local test splits with fixed client-level test identifiers.

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

The provided ids are removed from the labeled client pool and used as the final test set.
Train and validation are created from the remaining labeled clients with stratification by client-level label.

## Prepare data

```bash
python prepare_official_splits.py --dataset all
```

This writes:

```text
data/<dataset>/train.csv
data/<dataset>/val.csv
data/<dataset>/test.csv
data/<dataset>/split_report.json
data/<dataset>/split_summary.txt
```

The default validation fraction is `1/9` of the non-test clients. Since the provided test ids are approximately 10% of labeled clients, this gives approximately 80/10/10 train/val/test by clients.

## Validate data

```bash
python validate_splits.py --dataset all
```

The validator checks that:

- train, validation, and test client ids are disjoint;
- the prepared test split exactly matches the provided test-id file;
- label counts and majority baselines are printed for each split.

## Downstream pipeline implications

After running the preparation script, existing configs point to the new normalized CSV files:

```text
data/gender/train.csv, val.csv, test.csv
data/age/train.csv, val.csv, test.csv
data/rosbank/train.csv, val.csv, test.csv
```

The next required refactor is to make CoT feature extraction strictly train-fitted:

1. build summary statistics and few-shot examples from train only;
2. generate prompts/explanations/claims for train, validation, and test separately;
3. fit claim clustering and supervised cluster filtering on train claims only;
4. assign validation and test claims to the train-fitted clusters without using their labels.
