"""
src/pipeline/claims_extractor.py

Extracts atomic behavioral claims from CoT explanations.
Supports split-specific files such as explanations_train.jsonl -> claims_train.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from src.experiments.artifacts import fingerprint, stage_signature
from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_json_list


def _behavioral_text(explanation: str) -> str:
    text = str(explanation or "").strip()
    if not text:
        return ""
    final_idx = text.lower().rfind("final:")
    if final_idx >= 0:
        text = text[:final_idx].strip()
    return text


def _has_behavioral_explanation(explanation: str, min_chars: int = 80) -> bool:
    behavioral = _behavioral_text(explanation)
    if len(behavioral) < min_chars:
        return False
    sentence_marks = sum(behavioral.count(mark) for mark in (".", "!", "?", "\n", ";"))
    return sentence_marks >= 2


def _split_path(config: dict, key: str, split: str | None) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = config["output"][key]
    if split is None:
        return out_dir / base
    stem = Path(base).stem
    suffix = Path(base).suffix or ".jsonl"
    return out_dir / f"{stem}_{split}{suffix}"


def load_explanations(path: str | Path) -> list[list[dict]]:
    by_client: dict[int, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = int(rec["customer_id"])
            by_client.setdefault(cid, []).append(rec)
    return list(by_client.values())


def _claims_successful(record: dict) -> bool:
    return not record.get("error") and bool(record.get("claims"))


def _load_successful_claims(
    path: Path,
    expected_signatures: dict[int, dict[str, str]],
) -> dict[int, dict]:
    if not path.exists():
        return {}
    successful = {}
    error_types: Counter[str] = Counter()
    total = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                error_types["MalformedJSONL"] += 1
                continue
            cid = int(record["customer_id"])
            expected = expected_signatures.get(cid)
            compatible = bool(expected) and all(
                record.get(key) == value for key, value in expected.items()
            )
            if _claims_successful(record) and compatible:
                successful[cid] = record
            else:
                error_type = (
                    str(record.get("error_type"))
                    if record.get("error_type")
                    else "IncompatibleGeneration"
                    if not compatible
                    else "EmptyClaims"
                )
                error_types[error_type] += 1
    print(
        f"Existing claims: {len(successful)} successful, "
        f"{total - len(successful)} failed/empty, error_types={dict(error_types)} -> {path}"
    )
    return successful


def _parse_claim_result(result: dict) -> tuple[list[str], str | None, str | None]:
    if result.get("error") or result.get("response") is None:
        return (
            [],
            str(result.get("error_type") or "UnknownAPIError"),
            str(result.get("error") or "missing API response"),
        )
    try:
        choice = result["response"].choices[0]
        if choice.finish_reason not in (None, "stop"):
            return [], "IncompleteResponse", f"finish_reason={choice.finish_reason}"
        text = choice.message.content or ""
        parsed = [str(item).strip() for item in extract_json_list(text) if str(item).strip()]
        if not parsed:
            return [], "EmptyClaims", "response did not contain a non-empty JSON list"
        return parsed, None, None
    except Exception as exc:
        return [], "InvalidAPIResponse", str(exc)


def _write_claim_records(path: Path, ordered_clients: list[int], records: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        for customer_id in ordered_clients:
            if customer_id in records:
                file.write(json.dumps(records[customer_id], ensure_ascii=False) + "\n")
    temp_path.replace(path)


def _claims_summary(records: list[dict]) -> dict:
    error_types = Counter(
        str(record.get("error_type") or "EmptyClaims")
        for record in records
        if not _claims_successful(record)
    )
    return {
        "total": len(records),
        "successful": sum(_claims_successful(record) for record in records),
        "failed": sum(error_types.values()),
        "error_types": dict(sorted(error_types.items())),
        "total_claims": sum(len(record.get("claims", [])) for record in records),
    }


def run_claims_extraction(config: dict, *, split: str | None = None, input_path: str | Path | None = None, output_path: str | Path | None = None) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "explanations", split)
    save_path = Path(output_path) if output_path else _split_path(config, "claims", split)

    n_samples = config.get("pipeline", {}).get("n_claims_samples", 1)
    llm_cfg = dict(config["llm"])
    llm_cfg.update(config.get("claims_generation", {}))
    model = str(llm_cfg.get("model", llm_cfg["default_model"]))
    llm_cfg["max_tokens"] = int(
        llm_cfg.get(
            "max_tokens",
            config.get("pipeline", {}).get("claims_max_tokens", 2048),
        )
    )

    cfg_prompts = config["prompts"]
    base_dir = Path(cfg_prompts["base_dir"])
    claims_sys = (base_dir / cfg_prompts["claims_system"]).read_text(encoding="utf-8")
    claims_usr_t = (base_dir / cfg_prompts["claims_user"]).read_text(encoding="utf-8")
    generation_signature = stage_signature(
        "claims_extraction",
        inputs={
            "system_prompt": claims_sys,
            "user_prompt": claims_usr_t,
        },
        configuration={
            "model": model,
            "temperature": llm_cfg.get("temperature", 0.0),
            "top_p": llm_cfg.get("top_p", 1.0),
            "max_tokens": llm_cfg["max_tokens"],
            "seed": llm_cfg.get("seed"),
            "extra_body": llm_cfg.get("extra_body"),
            "n_claims_samples": n_samples,
        },
    )

    client_groups = load_explanations(load_path)
    print(f"Loaded explanations for {len(client_groups)} clients from {load_path}")

    ordered_clients = [int(group[0]["customer_id"]) for group in client_groups]
    expected_client_set = set(ordered_clients)
    selected_by_client: dict[int, list[dict]] = {}
    invalid_behavioral_by_client: dict[int, int] = {}
    expected_signatures: dict[int, dict[str, str]] = {}
    for group in client_groups:
        cid = int(group[0]["customer_id"])
        valid = []
        invalid_behavioral = 0
        for record in group:
            if not record.get("explanation") or record.get("error"):
                continue
            min_chars = int(record.get("min_behavioral_explanation_chars", 80) or 80)
            if not _has_behavioral_explanation(record.get("explanation", ""), min_chars):
                invalid_behavioral += 1
                continue
            valid.append(record)
        selected = sorted(valid, key=lambda record: int(record.get("sample_id", 0)))[:n_samples]
        source_hash = fingerprint(
            [
                {
                    "sample_id": record.get("sample_id", 0),
                    "generation_signature": record.get("generation_signature"),
                    "explanation": record.get("explanation", ""),
                }
                for record in selected
            ]
        )
        selected_by_client[cid] = selected
        invalid_behavioral_by_client[cid] = invalid_behavioral
        expected_signatures[cid] = {
            "generation_signature": generation_signature,
            "source_explanation_hash": source_hash,
        }

    existing = _load_successful_claims(save_path, expected_signatures)
    claims_by_client = {
        customer_id: record
        for customer_id, record in existing.items()
        if customer_id in expected_client_set
    }

    all_dialogues, meta = [], []
    for group in client_groups:
        cid = int(group[0]["customer_id"])
        if cid in claims_by_client:
            continue
        selected = selected_by_client[cid]
        source_hash = expected_signatures[cid]["source_explanation_hash"]
        if not selected:
            first = group[0]
            invalid_behavioral = invalid_behavioral_by_client[cid]
            error_type = "NoBehavioralExplanation" if invalid_behavioral else "NoValidExplanation"
            error = (
                "no usable behavioral explanation available"
                if invalid_behavioral
                else "no valid explanation available"
            )
            claims_by_client[cid] = {
                "customer_id": cid,
                "label": first.get("label", -1),
                "label_name": first.get("label_name", "unknown"),
                "claims": [],
                "error": error,
                "error_type": error_type,
                "generation_signature": generation_signature,
                "source_explanation_hash": source_hash,
            }
            continue
        for rec in selected:
            user_prompt = claims_usr_t.format(COT=rec.get("explanation", ""))
            all_dialogues.append([
                {"role": "system", "content": claims_sys},
                {"role": "user", "content": user_prompt},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label": rec.get("label", -1),
                "label_name": rec.get("label_name", "unknown"),
                "source_explanation_hash": source_hash,
            })

    print(
        f"Claims plan: expected_clients={len(ordered_clients)}, "
        f"reuse_successful={len(existing)}, requests_to_run={len(all_dialogues)}, "
        f"max_tokens={llm_cfg['max_tokens']}"
    )

    request_errors: dict[int, list[tuple[str, str]]] = {}

    def checkpoint(batch_results: list[tuple[int, dict]]) -> None:
        batch_records = []
        for idx, result in batch_results:
            item_meta = meta[idx]
            cid = int(item_meta["customer_id"])
            parsed, error_type, error = _parse_claim_result(result)
            record = claims_by_client.setdefault(
                cid,
                {
                    "customer_id": cid,
                    "label": item_meta.get("label", -1),
                    "label_name": item_meta.get("label_name", "unknown"),
                    "claims": [],
                    "error": None,
                    "error_type": None,
                    "generation_signature": generation_signature,
                    "source_explanation_hash": item_meta["source_explanation_hash"],
                },
            )
            record["claims"].extend(parsed)
            if error_type:
                request_errors.setdefault(cid, []).append((error_type, error or ""))
            if record["claims"]:
                seen = set()
                record["claims"] = [
                    claim
                    for claim in record["claims"]
                    if not (claim.lower() in seen or seen.add(claim.lower()))
                ]
                record["error"] = None
                record["error_type"] = None
            else:
                errors = request_errors.get(cid, [])
                record["error_type"] = errors[-1][0] if errors else "EmptyClaims"
                record["error"] = errors[-1][1] if errors else "no claims extracted"
            batch_records.append(record)

        _write_claim_records(save_path, ordered_clients, claims_by_client)
        summary = _claims_summary(batch_records)
        print(
            f"[CLAIMS BATCH] successful={summary['successful']} failed={summary['failed']} "
            f"error_types={summary['error_types']} total_claims={summary['total_claims']}"
        )

    if all_dialogues:
        asyncio.run(
            batched_query(
                all_dialogues,
                model,
                llm_cfg,
                on_batch_complete=checkpoint,
            )
        )

    _write_claim_records(save_path, ordered_clients, claims_by_client)
    final_records = [claims_by_client[cid] for cid in ordered_clients]
    summary = _claims_summary(final_records)
    stats_path = save_path.with_suffix(".generation_stats.json")
    temp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with open(temp_stats, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    temp_stats.replace(stats_path)
    print(
        f"[CLAIMS SUMMARY] successful={summary['successful']} failed={summary['failed']} "
        f"error_types={summary['error_types']} total_claims={summary['total_claims']} "
        f"-> {save_path}; stats -> {stats_path}"
    )
