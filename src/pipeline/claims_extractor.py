"""
src/pipeline/claims_extractor.py

Step 3 of the pipeline: extract atomic behavioral claims from CoT explanations.

Reads:  {output.explanations}  (jsonl)
Writes: {output.claims}        (jsonl)

Each output line: {customer_id, label, label_name, claims: [str, ...]}
"""

import asyncio
import json
from pathlib import Path

from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_json_list


def load_explanations(path: str) -> list[list[dict]]:
    """Load and group explanations by customer_id (multiple samples per client)."""
    by_client: dict[int, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["customer_id"]
            by_client.setdefault(cid, []).append(rec)
    return list(by_client.values())


def run_claims_extraction(config: dict) -> None:
    out_dir   = Path(config["output"]["base_dir"])
    load_path = out_dir / config["output"]["explanations"]
    save_path = out_dir / config["output"]["claims"]

    n_samples = config["pipeline"].get("n_claims_samples", 10)
    model     = config["llm"]["default_model"]
    llm_cfg   = config["llm"]

    cfg_prompts  = config["prompts"]
    base_dir     = cfg_prompts["base_dir"]
    claims_sys   = (Path(base_dir) / cfg_prompts["claims_system"]).read_text(encoding="utf-8")
    claims_usr_t = (Path(base_dir) / cfg_prompts["claims_user"]).read_text(encoding="utf-8")

    client_groups = load_explanations(str(load_path))
    print(f"Loaded explanations for {len(client_groups)} clients from {load_path}")

    all_dialogues = []
    meta = []

    for group in client_groups:
        for rec in group[:n_samples]:
            explanation = rec.get("explanation", "")
            if not explanation:
                continue
            user_prompt = claims_usr_t.format(COT=explanation)
            all_dialogues.append([
                {"role": "system", "content": claims_sys},
                {"role": "user",   "content": user_prompt},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label":       rec["label"],
                "label_name":  rec["label_name"],
            })

    print(f"Sending {len(all_dialogues)} claim extraction requests...")
    api_results = asyncio.run(batched_query(all_dialogues, model, llm_cfg))

    claims_by_client: dict[int, dict] = {}
    n_errors = 0
    for m, result in zip(meta, api_results):
        cid = m["customer_id"]

        if result["error"] or result["response"] is None:
            n_errors += 1
            parsed = []
        else:
            try:
                text   = result["response"].choices[0].message.content or ""
                parsed = extract_json_list(text)
            except Exception:
                parsed = []
                n_errors += 1

        if cid not in claims_by_client:
            claims_by_client[cid] = {
                "customer_id": cid,
                "label":       m["label"],
                "label_name":  m["label_name"],
                "claims":      [],
            }
        claims_by_client[cid]["claims"].extend(parsed)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for rec in claims_by_client.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_claims = sum(len(r["claims"]) for r in claims_by_client.values())
    if n_errors:
        print(f"  Warning: {n_errors}/{len(api_results)} requests failed")
    print(f"Saved {total_claims} claims for {len(claims_by_client)} clients → {save_path}")