"""Repair saved prediction parsing without repeating LLM requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipeline.explanation_gen import summarize_records
from src.utils.prompt_parsing import extract_boxed_answer, normalize_text_label


def split_path(config: dict, split: str) -> Path:
    base = Path(config["output"]["explanations"])
    return Path(config["output"]["base_dir"]) / f"{base.stem}_{split}{base.suffix}"


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def repair_path(path: Path, label_names: dict[str, str]) -> dict:
    records = []
    repaired = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            boxed = record.get("predicted_raw") or extract_boxed_answer(
                str(record.get("response_content") or record.get("explanation") or "")
            )
            predicted = normalize_text_label(boxed, label_names)
            parsing_failure = (
                record.get("error_type") == "MissingFinalAnswer"
                or str(record.get("error") or "").startswith(
                    "missing or unrecognized boxed final answer"
                )
            )
            if parsing_failure and predicted is not None:
                record["predicted_raw"] = boxed
                record["predicted"] = predicted
                record["error"] = None
                record["error_type"] = None
                repaired += 1
            records.append(record)

    atomic_write_jsonl(path, records)
    summary = summarize_records(records)
    stats_path = path.with_suffix(".generation_stats.json")
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as file:
            stats = json.load(file)
        stats["available_records"] = summary
        if stats.get("processed_new_requests") == len(records):
            stats["new_requests"] = summary
        atomic_write_json(stats_path, stats)
    return {"path": str(path), "repaired": repaired, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="test,train,val")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    label_names = {
        str(key): str(value)
        for key, value in config["dataset"]["label_names"].items()
    }

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        path = split_path(config, split)
        if not path.exists():
            print(f"[SKIP] missing {path}")
            continue
        result = repair_path(path, label_names)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
