"""
src/pipeline/explanation_gen.py

Sends client prompts to an OpenAI-compatible LLM API and saves CoT explanations.
Supports split-specific files such as prompts_val.jsonl -> explanations_val.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_boxed_answer, normalize_text_label


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


def run_explanation_generation(config: dict, *, split: str | None = None, input_path: str | Path | None = None, output_path: str | Path | None = None) -> None:
    load_path = Path(input_path) if input_path else _split_path(config, "prompts", split)
    save_path = Path(output_path) if output_path else _split_path(config, "explanations", split)

    n_samples = config.get("pipeline", {}).get("n_explanation_samples", 1)
    model = config["llm"]["default_model"]
    llm_cfg = config["llm"]
    label_names = config["dataset"].get("label_names", {})

    records = load_prompts(load_path)
    print(f"Loaded {len(records)} client prompts from {load_path}")

    all_dialogues, meta = [], []
    for rec in records:
        for sample_id in range(n_samples):
            all_dialogues.append([
                {"role": "system", "content": rec["system_prompt"]},
                {"role": "user", "content": rec["user_prompt"]},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label": rec.get("label", -1),
                "label_name": rec.get("label_name", "unknown"),
                "sample_id": sample_id,
            })

    print(f"Sending {len(all_dialogues)} requests to {model} ({n_samples} per client)...")
    api_results = asyncio.run(batched_query(all_dialogues, model, llm_cfg))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_errors = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for m, result in zip(meta, api_results):
            text = ""
            error = result.get("error")
            if error or result.get("response") is None:
                n_errors += 1
            else:
                try:
                    text = result["response"].choices[0].message.content or ""
                except Exception as exc:
                    error = str(exc)
                    n_errors += 1

            boxed = extract_boxed_answer(text)
            record = {
                **m,
                "explanation": text,
                "predicted_raw": boxed,
                "predicted": normalize_text_label(boxed, label_names),
                "execution_time": result.get("execution_time", 0.0),
                "error": error,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if n_errors:
        print(f"Warning: {n_errors}/{len(api_results)} requests failed")
    print(f"Saved {len(api_results)} explanations -> {save_path}")
