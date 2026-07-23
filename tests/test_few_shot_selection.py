import pandas as pd

from src.data.profiles import format_legacy_mean_category_summary
from src.pipeline.prompt_builder import (
    build_few_shot_str,
    build_prompts,
    prompt_length_telemetry,
    representative_medoid_ids,
)


def config(**pipeline):
    return {"dataset": {"name": "gender", "amount_semantics": "signed_cashflow", "label_names": {"0": "zero", "1": "one"}, "category_label": "categories"}, "pipeline": pipeline}


def frame():
    rows=[]
    for label in (0,1):
        for within,count in enumerate((1,2,3,4,10)):
            cid=label*100+within
            for txn in range(count):
                rows.append({"customer_id":cid,"label":label,"amount":-(within+1),"mcc_code_desc":f"category_{within}","tr_datetime":pd.Timestamp("2024-01-01")+pd.Timedelta(days=txn)})
    return pd.DataFrame(rows)


def test_zero_shot_has_truly_empty_demonstration_block():
    assert build_few_shot_str(frame(),config(few_shot_per_class=0))==""


def test_representative_few_shot_uses_class_medoids_deterministically():
    settings=config(few_shot_per_class=1,few_shot_strategy="representative_medoid",few_shot_seed=137)
    first=build_few_shot_str(frame(),settings)
    second=build_few_shot_str(frame(),settings)
    assert first==second
    assert first.count("* Total transactions: 3")==2
    assert "reasoning" not in first.lower()
    assert "cot" not in first.lower()


def test_medoid_ids_are_train_only_and_deterministic():
    train = frame()
    first = representative_medoid_ids(
        train, config(), label_id=0, n_clients=2
    )
    second = representative_medoid_ids(
        train.sample(frac=1.0, random_state=9),
        config(),
        label_id=0,
        n_clients=2,
    )
    assert first == second
    assert set(first) <= set(train.loc[train.label == 0, "customer_id"])


def test_neutral_only_legacy_summary_is_train_scoped_and_has_no_tail_std():
    summary=format_legacy_mean_category_summary(frame(),config())
    assert "Training-split" in summary
    assert "std" not in summary.lower()
    assert "mean count among clients using category" in summary


def test_long_prompts_are_not_blocked_or_truncated_and_have_telemetry(tmp_path):
    (tmp_path / "system.txt").write_text(
        "{TASK_DESCRIPTION}\n{DATASET_GUIDANCE}\n{ALLOWED_LABELS}",
        encoding="utf-8",
    )
    (tmp_path / "user.txt").write_text(
        "{SUMMARY_TRANSACTIONAL_STATS}\n{FEW_SHOT_SECTION}\n{CLIENT_STATS}",
        encoding="utf-8",
    )
    (tmp_path / "claims_system.txt").write_text(
        "Extract English claims.", encoding="utf-8"
    )
    (tmp_path / "claims_user.txt").write_text(
        "Forbidden:\n{FORBIDDEN_LABELS}\nRationale:\n{COT}", encoding="utf-8"
    )
    settings = config(few_shot_per_class=0)
    settings["dataset"].update(
        {
            "prompt_task_description": "Predict a test label.",
            "prompt_dataset_guidance": "Use transaction evidence.",
        }
    )
    settings["prompts"] = {
        "base_dir": str(tmp_path),
        "system": "system.txt",
        "user": "user.txt",
        "claims_system": "claims_system.txt",
        "claims_user": "claims_user.txt",
        "language": "en",
        "category_mapping_version": "en_v1",
    }
    very_long = "summary " * 20_000
    records = build_prompts(frame().query("customer_id == 0"), settings, very_long, "")
    assert very_long in records[0]["user_prompt"]
    assert records[0]["prompt_lengths"]["summary"] == len(very_long)
    telemetry = prompt_length_telemetry(records)
    assert telemetry["components"]["summary"]["max"] == len(very_long)
