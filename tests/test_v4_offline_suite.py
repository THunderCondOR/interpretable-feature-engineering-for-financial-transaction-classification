from scripts.run_v4_offline_suite import CELLS, cell_key


def test_offline_queue_is_smallest_first_and_age_qwen_last():
    keys = [cell_key(dataset, model) for dataset, model, _ in CELLS]
    assert keys == [
        "rosbank:gpt_oss",
        "gender:gpt_oss",
        "gender:qwen",
        "age:gpt_oss",
        "age:qwen",
    ]
    assert len(keys) == len(set(keys))
