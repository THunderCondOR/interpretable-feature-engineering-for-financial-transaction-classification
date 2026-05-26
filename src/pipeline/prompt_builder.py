"""
src/pipeline/prompt_builder.py

Assembles LLM prompts for CoT explanation generation.

Important implementation detail:
    Prompt templates may contain literal braces, for example LaTeX answers like
    \boxed{active client}. Therefore we do not use str.format() on the whole
    template. We replace only the known placeholders explicitly.
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from src.data.aggregator import get_summary_fn


KNOWN_PLACEHOLDERS = {
    "SUMMARY_TRANSACTIONAL_STATS",
    "FEW_SHOT_EXAMPLES",
    "CLIENT_STATS",
}


def _load_template(base_dir: str, relative_path: str) -> str:
    path = Path(base_dir) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _fill_prompt_template(
    template: str,
    *,
    summary_stats_str: str,
    few_shot_str: str,
    client_stats: str,
) -> str:
    """
    Fill only our three supported placeholders.

    This is intentionally not implemented with str.format(), because prompt text
    contains literal braces in strings such as ``\\boxed{отток}``. str.format()
    would interpret ``{отток}`` as a missing formatting key and raise KeyError.
    """
    return (
        template
        .replace("{SUMMARY_TRANSACTIONAL_STATS}", summary_stats_str)
        .replace("{FEW_SHOT_EXAMPLES}", few_shot_str)
        .replace("{CLIENT_STATS}", client_stats)
    )


def build_few_shot_str(
    df,
    config: dict,
    n_per_class: int | None = None,
    seed: int = 42,
) -> str:
    """Sample labeled train clients per class and format them as few-shot examples."""
    random.seed(seed)
    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn = get_summary_fn(config)
    if n_per_class is None:
        n_per_class = config.get("pipeline", {}).get("few_shot_per_class", 1)

    parts = ["Примеры клиентов из обучающей выборки:\n"]
    i = 1
    labeled_df = df[df["label"] >= 0]

    for label_id_str, label_name in sorted(label_names.items(), key=lambda x: int(x[0])):
        label_id = int(label_id_str)
        ids = labeled_df.loc[labeled_df["label"] == label_id, "customer_id"].unique().tolist()
        if not ids:
            continue
        sampled = random.sample(ids, k=min(n_per_class, len(ids)))
        for cid in sampled:
            client_df = labeled_df[labeled_df["customer_id"] == cid]
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
    Build one prompt dict per customer_id.

    The target label is stored for evaluation only and is not inserted into the prompt.
    Blind test rows may have label = -1.
    """
    cfg_prompts = config["prompts"]
    system_prompt = _load_template(cfg_prompts["base_dir"], cfg_prompts["system"])
    user_template = _load_template(cfg_prompts["base_dir"], cfg_prompts["user"])

    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn = get_summary_fn(config)

    records = []
    for cid in tqdm(df["customer_id"].unique(), desc="Building prompts"):
        client_df = df[df["customer_id"] == cid]
        label = int(client_df["label"].iloc[0]) if "label" in client_df.columns else -1
        client_stats = summary_fn(client_df, category_label)
        user_prompt = _fill_prompt_template(
            user_template,
            summary_stats_str=summary_stats_str,
            few_shot_str=few_shot_str,
            client_stats=client_stats,
        )
        label_name = label_names.get(str(label), "unknown") if label >= 0 else "unknown"
        records.append({
            "customer_id": int(cid),
            "label": label,
            "label_name": label_name,
            "client_stats": client_stats,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

    return records
