"""
src/pipeline/claims_extractor.py

Extracts atomic behavioral claims from CoT explanations.
Supports split-specific files such as explanations_train.jsonl -> claims_train.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_json_list


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


def run_claims_extraction(config: dict, *, split: str | None = None, input_path: str | Path | None = None, output_path: str | Path | None = None) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "explanations", split)
    save_path = Path(output_path) if output_path else _split_path(config, "claims", split)

    n_samples = config.get("pipeline", {}).get("n_claims_samples", 1)
    model = config["llm"]["default_model"]
    llm_cfg = config["llm"]

    cfg_prompts = config["prompts"]
    base_dir = Path(cfg_prompts["base_dir"])
    claims_sys = (base_dir / cfg_prompts["claims_system"]).read_text(encoding="utf-8")
    claims_usr_t = (base_dir / cfg_prompts["claims_user"]).read_text(encoding="utf-8")

    client_groups = load_explanations(load_path)
    print(f"Loaded explanations for {len(client_groups)} clients from {load_path}")

    all_dialogues, meta = [], []
    for group in client_groups:
        # Prefer valid explanations, but keep ordering stable.
        valid = [r for r in group if r.get("explanation")]
        for rec in valid[:n_samples]:
            user_prompt = claims_usr_t.format(COT=rec.get("explanation", ""))
            all_dialogues.append([
                {"role": "system", "content": claims_sys},
                {"role": "user", "content": user_prompt},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label": rec.get("label", -1),
                "label_name": rec.get("label_name", "unknown"),
            })

    print(f"Sending {len(all_dialogues)} claim extraction requests to {model}...")
    api_results = asyncio.run(batched_query(all_dialogues, model, llm_cfg))

    claims_by_client: dict[int, dict] = {}
    n_errors = 0
    for m, result in zip(meta, api_results):
        cid = int(m["customer_id"])
        parsed = []
        if result.get("error") or result.get("response") is None:
            n_errors += 1
        else:
            try:
                text = result["response"].choices[0].message.content or ""
                parsed = [str(x).strip() for x in extract_json_list(text) if str(x).strip()]
            except Exception:
                n_errors += 1
                parsed = []
        claims_by_client.setdefault(cid, {
            "customer_id": cid,
            "label": m.get("label", -1),
            "label_name": m.get("label_name", "unknown"),
            "claims": [],
        })["claims"].extend(parsed)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for rec in claims_by_client.values():
            # stable de-duplication per client
            seen, dedup = set(), []
            for claim in rec["claims"]:
                key = claim.lower()
                if key not in seen:
                    dedup.append(claim)
                    seen.add(key)
            rec["claims"] = dedup
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_claims = sum(len(r["claims"]) for r in claims_by_client.values())
    if n_errors:
        print(f"Warning: {n_errors}/{len(api_results)} requests failed")
    print(f"Saved {total_claims} claims for {len(claims_by_client)} clients -> {save_path}")
