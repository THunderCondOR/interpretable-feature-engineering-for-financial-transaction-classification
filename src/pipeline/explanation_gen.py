"""
src/pipeline/explanation_gen.py

Sends client prompts to an OpenAI-compatible LLM API and saves CoT explanations.
Supports split-specific files such as prompts_val.jsonl -> explanations_val.jsonl.
When resume=True, existing successful rows are reused on restart and reruns send
only missing or failed requests.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_boxed_answer, normalize_text_label


RequestKey = tuple[int, int]


def load_prompts(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_path(config: dict, key: str, split: str | None) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = config["output"][key]
    if split is None:
        return out_dir / base
    stem = Path(base).stem
    suffix = Path(base).suffix or ".jsonl"
    return out_dir / f"{stem}_{split}{suffix}"


def _request_key(record: dict) -> RequestKey:
    return int(record["customer_id"]), int(record.get("sample_id", 0))


def _is_successful(record: dict) -> bool:
    return not record.get("error") and bool(str(record.get("explanation", "")).strip())


def load_successful_existing(path: Path) -> dict[RequestKey, dict]:
    if not path.exists():
        return {}

    successful: dict[RequestKey, dict] = {}
    total = 0
    failed = 0
    malformed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if _is_successful(record):
                successful[_request_key(record)] = record
            else:
                failed += 1

    print(
        f"Existing explanations: {len(successful)} successful, "
        f"{failed} failed/empty, {malformed} malformed, {total} total -> {path}"
    )
    return successful


def build_output_record(meta: dict, result: dict, label_names: dict[str, str]) -> dict[str, Any]:
    text = ""
    error = result.get("error")
    if not error and result.get("response") is not None:
        try:
            text = result["response"].choices[0].message.content or ""
        except Exception as exc:
            error = str(exc)

    boxed = extract_boxed_answer(text)
    return {
        **meta,
        "explanation": text,
        "predicted_raw": boxed,
        "predicted": normalize_text_label(boxed, label_names),
        "execution_time": result.get("execution_time", 0.0),
        "error": error,
    }


def write_records(path: Path, ordered_meta: list[dict], records_by_key: dict[RequestKey, dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_errors = 0
    with open(path, "w", encoding="utf-8") as f:
        for meta in ordered_meta:
            key = _request_key(meta)
            record = records_by_key[key]
            if not _is_successful(record):
                n_errors += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return n_errors


def run_explanation_generation(
    config: dict,
    *,
    split: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume: bool = False,
) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "prompts", split)
    save_path = Path(output_path) if output_path else _split_path(config, "explanations", split)

    n_samples = config.get("pipeline", {}).get("n_explanation_samples", 1)
    model = config["llm"]["default_model"]
    llm_cfg = config["llm"]
    label_names = config["dataset"].get("label_names", {})

    prompt_records = load_prompts(load_path)
    print(f"Loaded {len(prompt_records)} client prompts from {load_path}")

    ordered_meta: list[dict] = []
    dialogues_by_key: dict[RequestKey, list[dict]] = {}
    sys_role = "sys" + "tem"
    usr_role = "user"
    sys_prompt_key = "system" + "_prompt"
    usr_prompt_key = "user" + "_prompt"
    for rec in prompt_records:
        for sample_id in range(n_samples):
            meta = {
                "customer_id": rec["customer_id"],
                "label": rec.get("label", -1),
                "label_name": rec.get("label_name", "unknown"),
                "sample_id": sample_id,
            }
            key = _request_key(meta)
            ordered_meta.append(meta)
            dialogues_by_key[key] = [
                {"role": sys_role, "content": rec[sys_prompt_key]},
                {"role": usr_role, "content": rec[usr_prompt_key]},
            ]

    expected_keys = {_request_key(meta) for meta in ordered_meta}
    existing = load_successful_existing(save_path) if resume else {}
    records_by_key = {key: record for key, record in existing.items() if key in expected_keys}

    keys_in_order = [_request_key(meta) for meta in ordered_meta]
    keys_to_run = [key for key in keys_in_order if key not in records_by_key]
    mode = "resume failed/missing" if resume else "rerun all"
    print(
        f"Explanation generation plan ({mode}): expected={len(ordered_meta)}, "
        f"reuse_successful={len(records_by_key)}, to_run={len(keys_to_run)}"
    )

    if keys_to_run:
        dialogues = [dialogues_by_key[key] for key in keys_to_run]
        meta_by_key = {_request_key(meta): meta for meta in ordered_meta}
        metas_to_run = [meta_by_key[key] for key in keys_to_run]

        print(f"Sending {len(dialogues)} requests to {model} ({n_samples} per client configured)...")
        api_results = asyncio.run(batched_query(dialogues, model, llm_cfg))

        for meta, result in zip(metas_to_run, api_results):
            records_by_key[_request_key(meta)] = build_output_record(meta, result, label_names)
    else:
        print("No requests to send: all expected explanations are already successful.")

    missing = expected_keys - set(records_by_key)
    if missing:
        raise RuntimeError(f"Internal error: missing {len(missing)} expected explanation records")

    n_errors = write_records(save_path, ordered_meta, records_by_key)
    if n_errors:
        print(f"Warning: {n_errors}/{len(ordered_meta)} explanation records are still failed or empty")
    print(f"Saved {len(ordered_meta)} explanations -> {save_path}")
