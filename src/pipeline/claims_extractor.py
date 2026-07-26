"""
src/pipeline/claims_extractor.py

Extracts atomic behavioral claims from CoT explanations.
Supports split-specific files such as explanations_train.jsonl -> claims_train.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.experiments.artifacts import fingerprint, stage_signature
from src.data.prompt_locale import assert_english_model_text, contains_cyrillic
from src.pipeline.semantic_features import normalize_claim
from src.utils.async_api import batched_query
from src.utils.event_log import append_structured_event
from src.utils.prompt_parsing import extract_json_list
from src.data.entity_ids import EntityId, canonical_entity_id


def _behavioral_text(explanation: str) -> str:
    text = str(explanation or "").strip()
    if not text:
        return ""
    final_idx = text.lower().rfind("final:")
    if final_idx >= 0:
        text = text[:final_idx].strip()
    return text


def _has_behavioral_explanation(explanation: str, min_chars: int = 1) -> bool:
    behavioral = _behavioral_text(explanation)
    if len(behavioral) < min_chars:
        return False
    return any(character.isalpha() for character in behavioral)


def _split_path(config: dict, key: str, split: str | None) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = config["output"][key]
    if split is None:
        return out_dir / base
    stem = Path(base).stem
    suffix = Path(base).suffix or ".jsonl"
    return out_dir / f"{stem}_{split}{suffix}"


def load_explanations(path: str | Path) -> list[list[dict]]:
    by_client: dict[EntityId, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = canonical_entity_id(rec["customer_id"])
            by_client.setdefault(cid, []).append(rec)
    return list(by_client.values())


def _claims_successful(record: dict) -> bool:
    return not record.get("error") and bool(record.get("claims"))


def _load_successful_claims(
    path: Path,
    expected_signatures: dict[EntityId, dict[str, str]],
) -> dict[EntityId, dict]:
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
            cid = canonical_entity_id(record["customer_id"])
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


def _validate_claims(
    values: list,
    *,
    forbidden_labels: set[str],
) -> tuple[list[str], str | None, str | None]:
    if not values:
        return [], "EmptyClaims", "response did not contain a non-empty JSON list"
    if any(not isinstance(value, str) for value in values):
        return [], "InvalidClaimsSchema", "every claim must be a JSON string"
    claims = [value.strip() for value in values if value.strip()]
    if not claims:
        return [], "EmptyClaims", "response did not contain a non-empty claim"
    valid_claims: list[str] = []
    rejected: list[tuple[str, str]] = []
    for claim in claims:
        if contains_cyrillic(claim):
            rejected.append(
                ("NonEnglishClaims", f"claim contains Cyrillic: {claim[:120]!r}")
            )
            continue
        normalized = re.sub(r"[\s_-]+", " ", claim.lower())
        if any(
            re.search(
                r"(?<!\w)"
                + re.escape(re.sub(r"[\s_-]+", " ", label.lower()))
                + r"(?!\w)",
                normalized,
            )
            for label in forbidden_labels
        ):
            rejected.append(
                (
                    "TargetLabelLeakage",
                    f"claim contains a target label: {claim[:120]!r}",
                )
            )
            continue
        # Numeric category identifiers are names, not quantitative evidence.
        without_category_ids = re.sub(
            r"\boperation group \d+\b", "operation group", claim, flags=re.I
        )
        if re.search(r"\d", without_category_ids):
            rejected.append(
                ("NumericClaim", f"claim contains an exact number: {claim[:120]!r}")
            )
            continue
        if not re.match(r"^The client(?:'s|\b)", claim, flags=re.I):
            rejected.append(
                (
                    "InvalidClaimSubject",
                    f"claim must use 'The client': {claim[:120]!r}",
                )
            )
            continue
        valid_claims.append(claim)
    if valid_claims:
        return valid_claims, None, None
    if rejected:
        error_type, error = rejected[0]
        return [], error_type, error
    return [], "EmptyClaims", "response did not contain a valid claim"


def _parse_claim_result(
    result: dict,
    *,
    forbidden_labels: set[str] | None = None,
) -> tuple[list[str], str | None, str | None]:
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
        parsed = extract_json_list(text)
        return _validate_claims(
            parsed,
            forbidden_labels=forbidden_labels or set(),
        )
    except Exception as exc:
        return [], "InvalidAPIResponse", str(exc)


def _write_claim_records(
    path: Path,
    ordered_clients: list[EntityId],
    records: dict[EntityId, dict],
) -> None:
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


def _attach_claim_records(record: dict) -> None:
    """Attach stable provenance while retaining the legacy string list."""
    source_hash = str(record.get("source_explanation_hash", ""))
    extractor = str(record.get("generation_signature", ""))
    generation_run = str(record.get("generation_run", extractor[:16]))
    record["claim_records"] = [
        {
            "claim_id": fingerprint({
                "customer_id": canonical_entity_id(record["customer_id"]),
                "source_explanation_hash": source_hash,
                "extractor_signature": extractor,
                "normalized_text": normalize_claim(claim),
                "occurrence": index,
            })[:20],
            "customer_id": canonical_entity_id(record["customer_id"]),
            "source_explanation_hash": source_hash,
            "generation_run": generation_run,
            "extractor_signature": extractor,
            "normalized_text": normalize_claim(claim),
            "original_text": claim,
        }
        for index, claim in enumerate(record.get("claims", []))
        if normalize_claim(claim)
    ]


def run_claims_extraction(config: dict, *, split: str | None = None, input_path: str | Path | None = None, output_path: str | Path | None = None) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "explanations", split)
    save_path = Path(output_path) if output_path else _split_path(config, "claims", split)

    n_samples = config.get("pipeline", {}).get("n_claims_samples", 1)
    llm_cfg = dict(config["llm"])
    llm_cfg.update(config.get("execution", {}))
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
    assert_english_model_text(claims_sys, context="claims system prompt")
    assert_english_model_text(claims_usr_t, context="claims user prompt")
    forbidden_labels = {
        str(value).strip()
        for value in (
            list(config["dataset"].get("label_names", {}).values())
            + list(config["dataset"].get("claim_forbidden_terms", []))
        )
        if str(value).strip()
    }
    generation_signature = stage_signature(
        "claims_extraction",
        inputs={
            "system_prompt": claims_sys,
            "user_prompt": claims_usr_t,
            "forbidden_labels": sorted(forbidden_labels),
        },
        configuration={
            "model": model,
            "temperature": llm_cfg.get("temperature", 0.0),
            "top_p": llm_cfg.get("top_p", 1.0),
            "max_tokens": llm_cfg["max_tokens"],
            "seed": llm_cfg.get("seed"),
            "extra_body": llm_cfg.get("extra_body"),
            "n_claims_samples": n_samples,
            "output_contract": "english_label_free_claims_v4",
        },
    )

    client_groups = load_explanations(load_path)
    print(f"Loaded explanations for {len(client_groups)} clients from {load_path}")

    ordered_clients = [
        canonical_entity_id(group[0]["customer_id"]) for group in client_groups
    ]
    expected_client_set = set(ordered_clients)
    selected_by_client: dict[EntityId, list[dict]] = {}
    invalid_behavioral_by_client: dict[EntityId, int] = {}
    expected_signatures: dict[EntityId, dict[str, str]] = {}
    for group in client_groups:
        cid = canonical_entity_id(group[0]["customer_id"])
        valid = []
        invalid_behavioral = 0
        for record in group:
            if not record.get("explanation") or record.get("error"):
                continue
            min_chars = int(record.get("min_behavioral_explanation_chars", 1) or 1)
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
            "source_prompt_hashes": sorted({
                str(record.get("prompt_hash"))
                for record in selected
                if record.get("prompt_hash")
            }),
            "source_client_stats_hashes": sorted({
                str(record.get("client_stats_hash"))
                for record in selected
                if record.get("client_stats_hash")
            }),
            "source_summary_stats_hashes": sorted({
                str(record.get("summary_stats_hash"))
                for record in selected
                if record.get("summary_stats_hash")
            }),
        }

    existing = _load_successful_claims(save_path, expected_signatures)
    claims_by_client = {
        customer_id: record
        for customer_id, record in existing.items()
        if customer_id in expected_client_set
    }

    all_dialogues, meta = [], []
    for group in client_groups:
        cid = canonical_entity_id(group[0]["customer_id"])
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
                "source_prompt_hashes": expected_signatures[cid]["source_prompt_hashes"],
                "source_client_stats_hashes": expected_signatures[cid]["source_client_stats_hashes"],
                "source_summary_stats_hashes": expected_signatures[cid]["source_summary_stats_hashes"],
            }
            continue
        for rec in selected:
            rationale = _behavioral_text(rec.get("explanation", ""))
            user_prompt = claims_usr_t.format(
                COT=rationale,
                FORBIDDEN_LABELS="\n".join(
                    f"- {label}" for label in sorted(forbidden_labels)
                ),
            )
            assert_english_model_text(
                user_prompt, context=f"claims user prompt {rec['customer_id']}"
            )
            all_dialogues.append([
                {"role": "system", "content": claims_sys},
                {"role": "user", "content": user_prompt},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label": rec.get("label", -1),
                "label_name": rec.get("label_name", "unknown"),
                "source_explanation_hash": source_hash,
                "source_prompt_hashes": expected_signatures[cid]["source_prompt_hashes"],
                "source_client_stats_hashes": expected_signatures[cid]["source_client_stats_hashes"],
                "source_summary_stats_hashes": expected_signatures[cid]["source_summary_stats_hashes"],
            })

    print(
        f"Claims plan: expected_clients={len(ordered_clients)}, "
        f"reuse_successful={len(existing)}, requests_to_run={len(all_dialogues)}, "
        f"max_tokens={llm_cfg['max_tokens']}"
    )

    request_errors: dict[EntityId, list[tuple[str, str]]] = {}
    content_error_counts: Counter[str] = Counter()
    content_error_attempts: Counter[int] = Counter()
    content_error_last_reasons: dict[int, str] = {}
    stats_path = save_path.with_suffix(".generation_stats.json")
    if stats_path.is_file():
        try:
            prior_validation = json.loads(
                stats_path.read_text(encoding="utf-8")
            ).get("content_validation", {})
            content_error_counts.update(
                {
                    str(key): int(value)
                    for key, value in prior_validation.get(
                        "error_types", {}
                    ).items()
                }
            )
            content_error_attempts.update(
                {
                    int(key): int(value)
                    for key, value in prior_validation.get(
                        "attempts_by_request", {}
                    ).items()
                }
            )
            content_error_last_reasons.update(
                {
                    int(key): str(value)
                    for key, value in prior_validation.get(
                        "last_reasons_by_request_index", {}
                    ).items()
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            print(
                f"Warning: ignored malformed prior content-validation "
                f"counters in {stats_path}"
            )

    llm_cfg.setdefault("scheduler_state_dir", str(save_path.parent / ".scheduler" / save_path.stem))
    llm_cfg.setdefault("events_path", str(save_path.parent / ".scheduler" / f"{save_path.stem}.events.jsonl"))
    llm_cfg["event_context"] = {
        "run_id": config.get("experiment", {}).get("run_id"),
        "dataset": config["dataset"].get("name", "unknown"),
        "model": config.get("experiment", {}).get("model_slug", model),
        "stage": "claims",
        "split": split,
    }

    if all_dialogues:
        until_complete = bool(llm_cfg.get("until_complete"))
        pending_indices = list(range(len(all_dialogues)))
        completed_request_indices: set[int] = set()
        repair_pass = 0

        while pending_indices:
            pass_indices = list(pending_indices)
            pass_dialogues = [all_dialogues[index] for index in pass_indices]
            pass_meta = [meta[index] for index in pass_indices]
            llm_cfg["request_keys"] = [
                f"{meta[index]['customer_id']}:{index}" for index in pass_indices
            ]
            llm_cfg["generation_signature"] = fingerprint(
                {
                    "claims_signature": generation_signature,
                    "request_keys": llm_cfg["request_keys"],
                }
            )

            def checkpoint(batch_results: list[tuple[int, dict]]) -> None:
                batch_records = []
                rejected_details = []
                accepted_in_batch = 0
                for idx, result in batch_results:
                    item_meta = pass_meta[idx]
                    original_index = pass_indices[idx]
                    cid = canonical_entity_id(item_meta["customer_id"])
                    parsed, error_type, error = _parse_claim_result(
                        result, forbidden_labels=forbidden_labels
                    )
                    if error_type:
                        reason = str(error or error_type)
                        request_errors.setdefault(cid, []).append(
                            (error_type, reason)
                        )
                        content_error_counts[error_type] += 1
                        content_error_attempts[original_index] += 1
                        content_error_last_reasons[original_index] = reason
                        rejected_details.append(
                            {
                                "request_key": f"{cid}:{original_index}",
                                "customer_id": cid,
                                "request_index": original_index,
                                "error_type": error_type,
                                "reason": reason,
                                "attempt": content_error_attempts[original_index],
                            }
                        )
                        if until_complete:
                            continue

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
                            "source_prompt_hashes": item_meta["source_prompt_hashes"],
                            "source_client_stats_hashes": item_meta["source_client_stats_hashes"],
                            "source_summary_stats_hashes": item_meta["source_summary_stats_hashes"],
                        },
                    )
                    record["claims"].extend(parsed)
                    if record["claims"]:
                        seen = set()
                        record["claims"] = [
                            claim
                            for claim in record["claims"]
                            if not (
                                claim.lower() in seen
                                or seen.add(claim.lower())
                            )
                        ]
                        record["error"] = None
                        record["error_type"] = None
                        completed_request_indices.add(original_index)
                        accepted_in_batch += 1
                    else:
                        errors = request_errors.get(cid, [])
                        record["error_type"] = (
                            errors[-1][0] if errors else "EmptyClaims"
                        )
                        record["error"] = (
                            errors[-1][1] if errors else "no claims extracted"
                        )
                    _attach_claim_records(record)
                    batch_records.append(record)

                if rejected_details:
                    batch_error_types = dict(
                        sorted(
                            Counter(
                                item["error_type"]
                                for item in rejected_details
                            ).items()
                        )
                    )
                    append_structured_event(
                        llm_cfg.get("events_path"),
                        llm_cfg["event_context"],
                        event="content_records_deferred",
                        repair_pass=repair_pass,
                        batch_size=len(batch_results),
                        accepted=accepted_in_batch,
                        deferred=len(rejected_details),
                        error_types=batch_error_types,
                        errors=rejected_details,
                    )
                    print(
                        f"[CLAIMS CONTENT DEFERRED] repair_pass={repair_pass} "
                        f"accepted={accepted_in_batch} "
                        f"deferred={len(rejected_details)} "
                        f"error_types={batch_error_types} "
                        f"request_keys="
                        f"{[item['request_key'] for item in rejected_details]}"
                    )

                _write_claim_records(
                    save_path, ordered_clients, claims_by_client
                )
                summary = _claims_summary(batch_records)
                print(
                    f"[CLAIMS BATCH] successful={summary['successful']} "
                    f"failed={summary['failed']} "
                    f"error_types={summary['error_types']} "
                    f"total_claims={summary['total_claims']}"
                )

            asyncio.run(
                batched_query(
                    pass_dialogues,
                    model,
                    llm_cfg,
                    on_batch_complete=checkpoint,
                )
            )
            pending_indices = [
                index
                for index in pass_indices
                if index not in completed_request_indices
            ]
            if not until_complete or not pending_indices:
                break
            repair_pass += 1
            append_structured_event(
                llm_cfg.get("events_path"),
                llm_cfg["event_context"],
                event="content_repair_pass_started",
                repair_pass=repair_pass,
                pending=len(pending_indices),
                cumulative_error_types=dict(
                    sorted(content_error_counts.items())
                ),
            )
            repair_cooldown = float(
                llm_cfg.get("content_repair_cooldown_seconds", 0.0)
            )
            if repair_cooldown > 0:
                time.sleep(repair_cooldown)

    for record in claims_by_client.values():
        _attach_claim_records(record)
    _write_claim_records(save_path, ordered_clients, claims_by_client)
    final_records = [claims_by_client[cid] for cid in ordered_clients]
    summary = _claims_summary(final_records)
    summary["content_validation"] = {
        "total_rejections": sum(content_error_counts.values()),
        "unique_rejected_requests": len(content_error_attempts),
        "error_types": dict(sorted(content_error_counts.items())),
        "max_attempts_for_one_request": max(
            content_error_attempts.values(), default=0
        ),
        "attempts_by_request": {
            str(index): attempts
            for index, attempts in sorted(content_error_attempts.items())
        },
        "last_reasons_by_request_index": {
            str(index): reason
            for index, reason in sorted(content_error_last_reasons.items())
        },
        "last_reasons_by_request": {
            f"{meta[index]['customer_id']}:{index}": reason
            for index, reason in sorted(content_error_last_reasons.items())
        },
    }
    temp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with open(temp_stats, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    temp_stats.replace(stats_path)
    print(
        f"[CLAIMS SUMMARY] successful={summary['successful']} failed={summary['failed']} "
        f"error_types={summary['error_types']} total_claims={summary['total_claims']} "
        f"-> {save_path}; stats -> {stats_path}"
    )
