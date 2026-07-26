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

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.aggregator import get_summary_fn
from src.data.prompt_locale import (
    CATEGORY_MAPPING_VERSION,
    PROMPT_LANGUAGE,
    assert_english_model_text,
)
from src.data.profiles import client_feature_frame
from src.experiments.artifacts import fingerprint
from src.data.entity_ids import canonical_entity_id, entity_sort_key


KNOWN_PLACEHOLDERS = {
    "SUMMARY_TRANSACTIONAL_STATS",
    "FEW_SHOT_SECTION",
    "CLIENT_STATS",
}
SYSTEM_PLACEHOLDERS = {
    "TASK_DESCRIPTION",
    "DATASET_GUIDANCE",
    "ALLOWED_LABELS",
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
    contains literal braces in strings such as ``\\boxed{retained_client}``.
    str.format() would interpret ``{retained_client}`` as a missing formatting
    key and raise KeyError.
    """
    return (
        template
        .replace("{SUMMARY_TRANSACTIONAL_STATS}", summary_stats_str)
        .replace(
            "{FEW_SHOT_SECTION}",
            (
                "\n\nTRAINING EXAMPLES\n\n" + few_shot_str.strip()
                if few_shot_str.strip()
                else ""
            ),
        )
        # Backwards-compatible replacement for historical templates. New v4
        # templates use FEW_SHOT_SECTION so the heading disappears entirely in
        # zero-shot runs.
        .replace("{FEW_SHOT_EXAMPLES}", few_shot_str)
        .replace("{CLIENT_STATS}", client_stats)
    )


def _fill_system_template(template: str, config: dict) -> str:
    dataset = config["dataset"]
    labels = [
        str(value)
        for _, value in sorted(
            dataset["label_names"].items(), key=lambda item: int(item[0])
        )
    ]
    rendered = (
        template.replace(
            "{TASK_DESCRIPTION}", str(dataset["prompt_task_description"]).strip()
        )
        .replace(
            "{DATASET_GUIDANCE}", str(dataset["prompt_dataset_guidance"]).strip()
        )
        .replace(
            "{ALLOWED_LABELS}", "\n".join(f"- {label}" for label in labels)
        )
    )
    for placeholder in SYSTEM_PLACEHOLDERS:
        if "{" + placeholder + "}" in rendered:
            raise ValueError(f"Unresolved system-prompt placeholder: {placeholder}")
    return rendered


def validate_prompt_contract(config: dict) -> None:
    """Fail before API materialization if the English v4 contract is incomplete."""
    prompts = config["prompts"]
    if prompts.get("language") != PROMPT_LANGUAGE:
        raise ValueError(
            f"Expected prompt language {PROMPT_LANGUAGE!r}, got "
            f"{prompts.get('language')!r}"
        )
    if prompts.get("category_mapping_version") != CATEGORY_MAPPING_VERSION:
        raise ValueError(
            "Prompt category mapping version does not match the renderer: "
            f"{prompts.get('category_mapping_version')!r} != "
            f"{CATEGORY_MAPPING_VERSION!r}"
        )
    base_dir = prompts["base_dir"]
    system = _fill_system_template(
        _load_template(base_dir, prompts["system"]), config
    )
    user = _load_template(base_dir, prompts["user"])
    claims_system = _load_template(base_dir, prompts["claims_system"])
    claims_user = _load_template(base_dir, prompts["claims_user"])
    for name, text in (
        ("system prompt", system),
        ("user template", user),
        ("claims system prompt", claims_system),
        ("claims user template", claims_user),
    ):
        assert_english_model_text(text, context=name)
    for placeholder in KNOWN_PLACEHOLDERS:
        if "{" + placeholder + "}" not in user:
            raise ValueError(f"User template is missing placeholder: {placeholder}")
    if "{COT}" not in claims_user:
        raise ValueError("Claims user template is missing {COT}")
    if "{FORBIDDEN_LABELS}" not in claims_user:
        raise ValueError("Claims user template is missing {FORBIDDEN_LABELS}")


def build_few_shot_str(
    df,
    config: dict,
    n_per_class: int | None = None,
    seed: int = 42,
) -> str:
    """Sample labeled train clients per class and format them as few-shot examples."""
    seed = int(config.get("pipeline", {}).get("few_shot_seed", seed))
    rng = random.Random(seed)
    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get(
        "category_label", "transaction categories"
    )
    summary_fn = get_summary_fn(config)
    if n_per_class is None:
        n_per_class = config.get("pipeline", {}).get("few_shot_per_class", 1)
    if int(n_per_class) == 0:
        return ""
    strategy = config.get("pipeline", {}).get("few_shot_strategy", "random")

    parts: list[str] = []
    i = 1
    labeled_df = df[df["label"] >= 0]

    for label_id_str, label_name in sorted(label_names.items(), key=lambda x: int(x[0])):
        label_id = int(label_id_str)
        ids = labeled_df.loc[labeled_df["label"] == label_id, "customer_id"].unique().tolist()
        if not ids:
            continue
        if strategy in {"representative", "representative_medoid"}:
            sampled = representative_medoid_ids(
                labeled_df,
                config,
                label_id=label_id,
                n_clients=n_per_class,
            )
        elif strategy == "random":
            sampled = rng.sample(ids, k=min(n_per_class, len(ids)))
        else:
            raise ValueError(f"Unknown few_shot_strategy: {strategy}")
        for cid in sampled:
            client_df = labeled_df[labeled_df["customer_id"] == cid]
            summary = summary_fn(client_df, category_label)
            parts.append(
                f"Example {i}\n"
                f"Client transaction profile:\n{summary}\n\n"
                f"Correct label: {label_name}"
            )
            i += 1

    rendered = "\n\n".join(parts)
    assert_english_model_text(rendered, context="few-shot demonstrations")
    return rendered


def representative_medoid_ids(
    train_df,
    config: dict,
    *,
    label_id: int,
    n_clients: int,
) -> list[int | str]:
    """Choose deterministic class representatives in robust-scaled profile space."""
    profiles = client_feature_frame(train_df, config)
    numeric = [
        column
        for column in profiles.select_dtypes(include=[np.number]).columns
        if column not in {"customer_id", "label"}
    ]
    if not numeric:
        raise ValueError("Few-shot medoid selection requires numeric client profiles")
    values = profiles[numeric].replace([np.inf, -np.inf], np.nan)
    medians = values.median()
    values = values.fillna(medians).fillna(0.0)
    scale = values.quantile(0.75) - values.quantile(0.25)
    scale = scale.mask(scale.abs() < 1e-12, 1.0)
    scaled = (values - medians) / scale
    class_mask = profiles["label"].astype(int) == int(label_id)
    class_values = scaled.loc[class_mask]
    if class_values.empty:
        return []
    class_center = class_values.median()
    distance = np.sqrt(((class_values - class_center) ** 2).sum(axis=1))
    ranked = (
        pd.DataFrame(
            {
                "customer_id": profiles.loc[class_mask, "customer_id"].map(
                    canonical_entity_id
                ),
                "distance": distance,
            }
        )
        .assign(_entity_sort=lambda frame: frame["customer_id"].map(entity_sort_key))
        .sort_values(["distance", "_entity_sort"], kind="mergesort")
    )
    return ranked["customer_id"].head(int(n_clients)).tolist()


def prompt_length_telemetry(records: list[dict]) -> dict:
    """Summarize prompt component sizes without imposing any validation cap."""
    components = ("summary", "few_shot", "target_profile", "system", "user", "total")
    result = {"n_prompts": len(records), "units": "unicode_characters", "components": {}}
    for component in components:
        values = np.asarray(
            [record["prompt_lengths"][component] for record in records],
            dtype=float,
        )
        result["components"][component] = {
            "min": int(values.min()) if len(values) else 0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "p95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            "max": int(values.max()) if len(values) else 0,
        }
    return result


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
    validate_prompt_contract(config)
    system_prompt = _fill_system_template(
        _load_template(cfg_prompts["base_dir"], cfg_prompts["system"]), config
    )
    user_template = _load_template(cfg_prompts["base_dir"], cfg_prompts["user"])

    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get(
        "category_label", "transaction categories"
    )
    summary_fn = get_summary_fn(config)
    assert_english_model_text(
        summary_stats_str, context="training-split class reference"
    )
    assert_english_model_text(few_shot_str, context="few-shot demonstrations")

    records = []
    grouped = df.groupby("customer_id", sort=False, observed=True)
    for cid, client_df in tqdm(
        grouped,
        total=df["customer_id"].nunique(),
        desc="Building prompts",
    ):
        label = int(client_df["label"].iloc[0]) if "label" in client_df.columns else -1
        client_stats = summary_fn(client_df, category_label)
        user_prompt = _fill_prompt_template(
            user_template,
            summary_stats_str=summary_stats_str,
            few_shot_str=few_shot_str,
            client_stats=client_stats,
        )
        assert_english_model_text(client_stats, context=f"client profile {cid}")
        assert_english_model_text(
            system_prompt, context=f"rendered system prompt {cid}"
        )
        assert_english_model_text(user_prompt, context=f"rendered user prompt {cid}")
        label_name = label_names.get(str(label), "unknown") if label >= 0 else "unknown"
        prompt_hash = fingerprint(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "customer_id": canonical_entity_id(cid),
            }
        )
        records.append({
            "customer_id": canonical_entity_id(cid),
            "label": label,
            "label_name": label_name,
            "client_stats": client_stats,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_hash": prompt_hash,
            "client_stats_hash": fingerprint(client_stats),
            "summary_stats_hash": fingerprint(summary_stats_str),
            "few_shot_hash": fingerprint(few_shot_str),
            "prompt_lengths": {
                "summary": len(summary_stats_str),
                "few_shot": len(few_shot_str),
                "target_profile": len(client_stats),
                "system": len(system_prompt),
                "user": len(user_prompt),
                "total": len(system_prompt) + len(user_prompt),
            },
        })

    return records
