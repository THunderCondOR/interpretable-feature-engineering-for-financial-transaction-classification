import pandas as pd

from src.data.profiles import format_legacy_mean_category_summary
from src.pipeline.prompt_builder import build_few_shot_str


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
    settings=config(few_shot_per_class=1,few_shot_strategy="representative",few_shot_seed=137)
    first=build_few_shot_str(frame(),settings)
    second=build_few_shot_str(frame(),settings)
    assert first==second
    assert first.count("* Всего операций: 3")==2


def test_neutral_only_legacy_summary_is_train_scoped_and_has_no_tail_std():
    summary=format_legacy_mean_category_summary(frame(),config())
    assert "Training-split" in summary
    assert "std" not in summary.lower()
    assert "mean count among clients using category" in summary
