#!/usr/bin/env python3
"""Plan or run full GPT-5.5 grounding and three-judge summaries in sequence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    (
        "reviewer-v9-main",
        REPO_ROOT / "results/v2/grounding/reviewer-v9-grounding-official-ready",
    ),
    (
        "datafusion-default-2023",
        REPO_ROOT / "results/isolated/grounding/reviewer-v11-df2023-grounding",
    ),
)
JUDGE_FILES = (
    "judge_openrouter_deepseek_deepseek_v4_pro.jsonl",
    "judge_openrouter_google_gemini_3_flash_preview.jsonl",
)


def commands(python_bin: str, chunk_size: int) -> list[dict[str, object]]:
    jobs = []
    for name, root in CELLS:
        full_dir = root / "full_codex_judge"
        codex_output = full_dir / "judge_codex_gpt_5_5_full.jsonl"
        summary_dir = root / "three_judge"
        judge = [
            python_bin,
            str(REPO_ROOT / "scripts/run_codex_grounding_full_judge.py"),
            "--input", str(root / "grounding_samples.jsonl"),
            "--output", str(codex_output),
            "--model", "gpt-5.5",
            "--chunk-size", str(chunk_size),
            "--execute-codex",
        ]
        summarize = [
            python_bin,
            str(REPO_ROOT / "scripts/summarize_grounding_three_judges.py"),
            "--samples", str(root / "grounding_samples.jsonl"),
            "--initial-inputs", *[str(root / filename) for filename in JUDGE_FILES],
            "--codex-input", str(codex_output),
            "--output-dir", str(summary_dir),
            "--execute",
        ]
        jobs.append({"name": name, "full_judge": judge, "summary": summarize})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-codex", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    jobs = commands(args.python_bin, args.chunk_size)
    print(json.dumps({
        "mode": "execute" if args.execute and args.execute_codex else "dry-run",
        "existing_two_judge_outputs": "read-only",
        "jobs": jobs,
    }, indent=2))
    if not (args.execute and args.execute_codex):
        return
    for job in jobs:
        subprocess.run(job["full_judge"], check=True, cwd=REPO_ROOT)
        subprocess.run(job["summary"], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
