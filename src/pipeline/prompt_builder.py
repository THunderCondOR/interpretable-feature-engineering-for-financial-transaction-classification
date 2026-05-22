"""
src/pipeline/prompt_builder.py

Assembles LLM prompts for CoT explanation generation.
Reads prompt templates from files (dataset-specific) and fills in:
    - SUMMARY_TRANSACTIONAL_STATS  (dataset-level averages per label)
    - FEW_SHOT_EXAMPLES            (sampled client profiles with known labels)
    - CLIENT_STATS                 (the target client's user_summary_str)

Output is a list of dicts saved as JSONL — same format consumed by
explanation_gen.py and lora_trainer.py.
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

from src.data.aggregator import get_summary_fn


def _load_template(base_dir: str, relative_path: str) -> str:
    path = Path(base_dir) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_few_shot_str(
    df,
    config: dict,
    n_per_class: int = 1,
    seed: int = 42,
) -> str:
    """
    Sample n_per_class clients per label and format them as few-shot examples.
    Uses the same user_summary_str format as the target client.
    """
    random.seed(seed)
    label_names    = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn     = get_summary_fn(config)

    parts = ["Примеры клиентов:\n"]
    i = 1
    for label_id_str, label_name in label_names.items():
        label_id = int(label_id_str)
        ids = df.loc[df["label"] == label_id, "customer_id"].unique().tolist()
        sampled = random.sample(ids, k=min(n_per_class, len(ids)))

        for cid in sampled:
            client_df = df[df["customer_id"] == cid]
            summary = summary_fn(client_df, category_label)
            parts.append(f"\nКлиент {i} — {label_name}:\n{summary}\n")
            i += 1

    return "\n".join(parts)


def build_prompts(
    df,
    config: dict,
    summary_stats_str: str,
    few_shot_str: str,
) -> list[dict]:
    """
    Build one prompt dict per unique customer_id.

    Each dict contains:
        customer_id, label, label_name,
        client_stats   (user_summary_str),
        system_prompt,
        user_prompt    (fully filled template)

    The user_prompt template must have three placeholders:
        {SUMMARY_TRANSACTIONAL_STATS}
        {FEW_SHOT_EXAMPLES}
        {CLIENT_STATS}
    """
    cfg_prompts   = config["prompts"]
    system_prompt = _load_template(cfg_prompts["base_dir"], cfg_prompts["system"])
    user_template = _load_template(cfg_prompts["base_dir"], cfg_prompts["user"])

    label_names    = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn     = get_summary_fn(config)

    records = []
    for cid in tqdm(df["customer_id"].unique(), desc="Building prompts"):
        client_df    = df[df["customer_id"] == cid]
        label        = int(client_df["label"].iloc[0])
        client_stats = summary_fn(client_df, category_label)

        user_prompt = user_template.format(
            SUMMARY_TRANSACTIONAL_STATS=summary_stats_str,
            FEW_SHOT_EXAMPLES=few_shot_str,
            CLIENT_STATS=client_stats,
        )

        # label_names keys may be int or str depending on YAML loader
        label_name = label_names.get(str(label), label_names.get(label, str(label)))

        records.append({
            "customer_id":  int(cid),
            "label":        label,
            "label_name":   label_name,
            "client_stats": client_stats,
            "system_prompt": system_prompt,
            "user_prompt":  user_prompt,
        })

    return records