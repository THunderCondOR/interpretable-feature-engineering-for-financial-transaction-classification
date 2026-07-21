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
from collections import Counter
from pathlib import Path
from typing import Any

from src.experiments.artifacts import fingerprint, prompt_signature
from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_boxed_answer, normalize_text_label


RequestKey = tuple[int, int]


def _behavioral_text(explanation: str) -> str:
    """Return the part of an explanation before the final boxed answer."""
    text = str(explanation or "").strip()
    if not text:
        return ""
    final_idx = text.lower().rfind("final:")
    if final_idx >= 0:
        text = text[:final_idx].strip()
    return text


def _has_behavioral_explanation(record: dict) -> bool:
    """Reject outputs that contain only a final answer and no usable rationale."""
    explanation = str(record.get("explanation", "") or "")
    behavioral = _behavioral_text(explanation)
    min_chars = int(record.get("min_behavioral_explanation_chars", 80) or 80)
    if len(behavioral) < min_chars:
        return False
    # Require at least two sentence-like fragments so a short label/list heading
    # is not treated as a usable behavioral rationale.
    sentence_marks = sum(behavioral.count(mark) for mark in (".", "!", "?", "\n", ";"))
    return sentence_marks >= 2


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
    return (
        not record.get("error")
        and bool(str(record.get("explanation", "")).strip())
        and record.get("predicted") is not None
        and _has_behavioral_explanation(record)
    )


def _record_error_type(record: dict) -> str | None:
    if _is_successful(record):
        return None
    if record.get("error_type"):
        return str(record["error_type"])

    error = str(record.get("error") or "")
    if error.startswith("incomplete response"):
        return "IncompleteResponse"
    if error.startswith("invalid API response"):
        return "InvalidAPIResponse"
    if error.startswith("missing or unrecognized"):
        return "MissingFinalAnswer"
    if error.startswith("empty response") or not str(record.get("response_content", record.get("explanation", ""))).strip():
        return "EmptyResponse"
    if record.get("predicted") is None:
        return "MissingFinalAnswer"
    if not _has_behavioral_explanation(record):
        return "NoBehavioralExplanation"
    if error:
        return error.split(":", 1)[0].strip() or "UnknownError"
    return "UnknownError"


def summarize_records(records: list[dict]) -> dict[str, Any]:
    error_types = Counter(
        error_type
        for record in records
        if (error_type := _record_error_type(record)) is not None
    )
    failed = sum(error_types.values())
    return {
        "total": len(records),
        "successful": len(records) - failed,
        "failed": failed,
        "error_types": dict(sorted(error_types.items())),
    }


def _generation_stats_path(path: Path) -> Path:
    return path.with_suffix(".generation_stats.json")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def load_successful_existing(
    path: Path,
    expected_signatures: dict[RequestKey, str] | None = None,
) -> dict[RequestKey, dict]:
    if not path.exists():
        return {}

    successful: dict[RequestKey, dict] = {}
    total = 0
    failed = 0
    malformed = 0
    incompatible = 0
    error_types: Counter[str] = Counter()

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
            key = _request_key(record)
            expected = expected_signatures.get(key) if expected_signatures is not None else None
            compatible = expected is None or record.get("generation_signature") == expected
            if _is_successful(record) and compatible:
                successful[key] = record
            else:
                failed += 1
                if not compatible:
                    incompatible += 1
                    error_types["IncompatibleGeneration"] += 1
                else:
                    error_types[_record_error_type(record) or "UnknownError"] += 1

    print(
        f"Existing explanations: {len(successful)} successful, "
        f"{failed} failed/empty, {incompatible} incompatible, "
        f"{malformed} malformed, {total} total, "
        f"error_types={dict(error_types)} -> {path}"
    )
    return successful


def build_output_record(meta: dict, result: dict, label_names: dict[str, str]) -> dict[str, Any]:
    text = ""
    error = result.get("error")
    error_type = result.get("error_type")
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_text = ""
    reasoning_chars = 0

    if not error and result.get("response") is not None:
        try:
            response = result["response"]
            choice = response.choices[0]
            message = choice.message
            text = message.content or ""
            finish_reason = choice.finish_reason

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is None:
                reasoning = (getattr(message, "model_extra", None) or {}).get("reasoning_content")
            reasoning_text = reasoning or ""
            reasoning_chars = len(reasoning_text)

            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
        except Exception as exc:
            error = f"invalid API response: {exc}"
            error_type = "InvalidAPIResponse"

    # Provider-specific hidden reasoning is diagnostic metadata, not a feature.
    explanation = text
    boxed = extract_boxed_answer(text)
    predicted = normalize_text_label(boxed, label_names)

    if not error and not text.strip():
        error = "empty response content"
        error_type = "EmptyResponse"
    elif not error and finish_reason not in (None, "stop"):
        error = f"incomplete response: finish_reason={finish_reason}"
        error_type = "IncompleteResponse"
    elif not error and predicted is None:
        error = "missing or unrecognized boxed final answer"
        error_type = "MissingFinalAnswer"
    elif error and not error_type:
        error_type = str(error).split(":", 1)[0].strip() or "UnknownAPIError"

    record = {
        **meta,
        "explanation": explanation,
        "response_content": text,
        "reasoning": reasoning_text,
        "predicted_raw": boxed,
        "predicted": predicted,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_chars": reasoning_chars,
        "execution_time": result.get("execution_time", 0.0),
        "error": error,
        "error_type": error_type,
    }
    if not error:
        min_chars = int(meta.get("min_behavioral_explanation_chars", 80) or 80)
        record["min_behavioral_explanation_chars"] = min_chars
        if not _has_behavioral_explanation(record):
            record["error"] = (
                f"no usable behavioral explanation before final answer "
                f"(min_chars={min_chars})"
            )
            record["error_type"] = "NoBehavioralExplanation"
    return record


def write_records(path: Path, ordered_meta: list[dict], records_by_key: dict[RequestKey, dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    n_errors = 0
    with open(temp_path, "w", encoding="utf-8") as f:
        for meta in ordered_meta:
            key = _request_key(meta)
            record = records_by_key[key]
            if not _is_successful(record):
                n_errors += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)
    return n_errors


def run_explanation_generation(
    config: dict,
    *,
    split: str | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume: bool = True,
) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "prompts", split)
    save_path = Path(output_path) if output_path else _split_path(config, "explanations", split)

    n_samples = config.get("pipeline", {}).get("n_explanation_samples", 1)
    llm_cfg = dict(config["llm"])
    llm_cfg.update(config.get("generation", {}))
    model = str(llm_cfg.get("model", llm_cfg["default_model"]))
    label_names = config["dataset"].get("label_names", {})
    min_behavioral_explanation_chars = int(
        config.get("pipeline", {}).get("min_behavioral_explanation_chars", 80)
    )

    prompt_records = load_prompts(load_path)
    print(f"Loaded {len(prompt_records)} client prompts from {load_path}")

    ordered_meta: list[dict] = []
    dialogues_by_key: dict[RequestKey, list[dict]] = {}
    expected_signatures: dict[RequestKey, str] = {}
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
                "min_behavioral_explanation_chars": min_behavioral_explanation_chars,
            }
            key = _request_key(meta)
            system_prompt = rec[sys_prompt_key]
            user_prompt = rec[usr_prompt_key]
            decoding = {
                "temperature": llm_cfg.get("temperature", 1.0),
                "top_p": llm_cfg.get("top_p", 0.9),
                "max_tokens": llm_cfg.get("max_tokens", 2048),
                "seed": llm_cfg.get("seed"),
                "extra_body": llm_cfg.get("extra_body"),
            }
            signature = prompt_signature(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                decoding=decoding,
                sample_id=sample_id,
            )
            meta["prompt_hash"] = fingerprint(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )
            meta["generation_signature"] = signature
            expected_signatures[key] = signature
            ordered_meta.append(meta)
            dialogues_by_key[key] = [
                {"role": sys_role, "content": system_prompt},
                {"role": usr_role, "content": user_prompt},
            ]

    expected_keys = {_request_key(meta) for meta in ordered_meta}
    existing = (
        load_successful_existing(save_path, expected_signatures)
        if resume
        else {}
    )
    records_by_key = {key: record for key, record in existing.items() if key in expected_keys}

    keys_in_order = [_request_key(meta) for meta in ordered_meta]
    keys_to_run = [key for key in keys_in_order if key not in records_by_key]
    mode = "resume failed/missing" if resume else "rerun all"
    print(
        f"Explanation generation plan ({mode}): expected={len(ordered_meta)}, "
        f"reuse_successful={len(records_by_key)}, to_run={len(keys_to_run)}"
    )

    stats_path = _generation_stats_path(save_path)
    last_batch_summary: dict[str, Any] | None = None

    def save_generation_stats() -> dict:
        available_records = [
            records_by_key[_request_key(meta)]
            for meta in ordered_meta
            if _request_key(meta) in records_by_key
        ]
        available_summary = summarize_records(available_records)
        processed_new_records = [
            records_by_key[key]
            for key in keys_to_run
            if key in records_by_key
        ]
        new_summary = summarize_records(processed_new_records)
        payload = {
            "output_path": str(save_path),
            "expected_records": len(ordered_meta),
            "resume": resume,
            "reused_successful": len(records_by_key) - len(processed_new_records),
            "planned_new_requests": len(keys_to_run),
            "processed_new_requests": len(processed_new_records),
            "pending_new_requests": len(keys_to_run) - len(processed_new_records),
            "new_requests": new_summary,
            "available_records": available_summary,
            "last_batch": last_batch_summary,
        }
        _write_json_atomic(stats_path, payload)
        return payload

    if keys_to_run:
        dialogues = [dialogues_by_key[key] for key in keys_to_run]
        meta_by_key = {_request_key(meta): meta for meta in ordered_meta}
        metas_to_run = [meta_by_key[key] for key in keys_to_run]

        print(f"Sending {len(dialogues)} requests to {model} ({n_samples} per client configured)...")
        checkpointed_keys: set[RequestKey] = set()

        def checkpoint(batch_results: list[tuple[int, dict]]) -> None:
            nonlocal last_batch_summary
            batch_records = []
            for idx, result in batch_results:
                meta = metas_to_run[idx]
                key = _request_key(meta)
                record = build_output_record(meta, result, label_names)
                records_by_key[key] = record
                batch_records.append(record)
                checkpointed_keys.add(key)

            available_meta = [
                meta for meta in ordered_meta
                if _request_key(meta) in records_by_key
            ]
            write_records(save_path, available_meta, records_by_key)
            last_batch_summary = summarize_records(batch_records)
            stats = save_generation_stats()
            print(
                f"[COT BATCH] completed={last_batch_summary['total']} "
                f"successful={last_batch_summary['successful']} "
                f"failed={last_batch_summary['failed']} "
                f"error_types={last_batch_summary['error_types']}"
            )
            print(
                f"[COT CHECKPOINT] processed_new={len(checkpointed_keys)}/{len(keys_to_run)} "
                f"successful_new={stats['new_requests']['successful']} "
                f"failed_new={stats['new_requests']['failed']} "
                f"saved={len(available_meta)}/{len(ordered_meta)} -> {save_path}; "
                f"stats -> {stats_path}"
            )

        api_results = asyncio.run(
            batched_query(
                dialogues,
                model,
                llm_cfg,
                on_batch_complete=checkpoint,
            )
        )

        for meta, result in zip(metas_to_run, api_results):
            records_by_key[_request_key(meta)] = build_output_record(meta, result, label_names)
    else:
        print("No requests to send: all expected explanations are already successful.")

    missing = expected_keys - set(records_by_key)
    if missing:
        raise RuntimeError(f"Internal error: missing {len(missing)} expected explanation records")

    n_errors = write_records(save_path, ordered_meta, records_by_key)
    final_stats = save_generation_stats()
    print(
        f"[COT SUMMARY] successful={final_stats['available_records']['successful']} "
        f"failed={final_stats['available_records']['failed']} "
        f"error_types={final_stats['available_records']['error_types']} "
        f"stats -> {stats_path}"
    )
    if n_errors:
        print(f"Warning: {n_errors}/{len(ordered_meta)} explanation records are still failed or empty")
    print(f"Saved {len(ordered_meta)} explanations -> {save_path}")
