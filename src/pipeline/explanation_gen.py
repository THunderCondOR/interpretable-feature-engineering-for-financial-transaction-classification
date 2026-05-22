"""
src/pipeline/explanation_gen.py

Step 2 of the pipeline: send client prompts to LLM and save CoT explanations.

Reads:  {output.clients_stats}  (from stats step)  OR pre-built prompts JSON
Writes: {output.explanations}   (jsonl)

Each output line: {customer_id, label, label_name, explanation, execution_time}
"""

import asyncio
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.async_api import batched_query
from src.utils.prompt_parsing import extract_boxed_answer


def load_prompts(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_messages(record: dict, n_samples: int = 1) -> list[list[dict]]:
    """Build n_samples independent message lists for one client (for majority voting)."""
    messages = [
        {"role": "system",  "content": record["system_prompt"]},
        {"role": "user",    "content": record["user_prompt"]},
    ]
    return [messages] * n_samples


def run_explanation_generation(config: dict) -> None:
    out_dir  = Path(config["output"]["base_dir"])
    load_path = out_dir / config["output"]["clients_stats"]
    save_path = out_dir / config["output"]["explanations"]

    n_samples = config["pipeline"].get("n_explanation_samples", 8)
    model     = config["llm"]["default_model"]
    llm_cfg   = config["llm"]

    records = load_prompts(str(load_path))
    print(f"Loaded {len(records)} client prompts from {load_path}")

    # Build all dialogue batches: n_samples per client
    all_dialogues = []
    meta = []  # (customer_id, label, label_name) repeated n_samples times
    for rec in records:
        for _ in range(n_samples):
            all_dialogues.append([
                {"role": "system", "content": rec["system_prompt"]},
                {"role": "user",   "content": rec["user_prompt"]},
            ])
            meta.append({
                "customer_id": rec["customer_id"],
                "label":       rec["label"],
                "label_name":  rec["label_name"],
            })

    print(f"Sending {len(all_dialogues)} requests ({n_samples} per client)...")
    api_results = asyncio.run(batched_query(all_dialogues, model, llm_cfg))

    # Write results
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for m, result in zip(meta, api_results):
            response = result["response"]
            text = response.choices[0].message.content if response.choices else ""
            record = {
                **m,
                "explanation":    text,
                "predicted":      extract_boxed_answer(text),
                "execution_time": result["execution_time"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(api_results)} explanations to {save_path}")
