"""Measure interference between three independent dataset LLM processes.

Each dataset is first measured alone, then all three are launched concurrently
as separate subprocesses. Only stress-test outputs are written; explanation
files are never touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


DATASETS = ("gender", "age", "rosbank")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def load_scenario(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        report = json.load(file)
    return report["scenarios"][0]


async def run_dataset(
    *,
    repo_dir: Path,
    dataset: str,
    phase: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
    prompt_mode: str,
    output_dir: Path,
) -> dict:
    output_path = output_dir / f"{phase}_{dataset}.json"
    log_path = output_dir / f"{phase}_{dataset}.log"
    command = [
        sys.executable,
        "scripts/stress_test_llm.py",
        "--config",
        f"configs/{dataset}.yaml",
        "--split",
        "test",
        "--concurrencies",
        str(concurrency),
        "--requests-per-scenario",
        str(requests),
        "--max-tokens",
        str(max_tokens),
        "--prompt-mode",
        prompt_mode,
        "--cooldown-seconds",
        "0",
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=repo_dir,
            env=env,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        return_code = await process.wait()
    wall_seconds = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(
            f"{phase}/{dataset} failed with code {return_code}; see {log_path}"
        )
    scenario = load_scenario(output_path)
    scenario["process_wall_seconds"] = wall_seconds
    scenario["output_path"] = str(output_path)
    scenario["log_path"] = str(log_path)
    return scenario


async def main_async(args: argparse.Namespace) -> None:
    repo_dir = Path(__file__).resolve().parent.parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_dir / "results" / "parallel_llm_stress"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    standalone = {}
    for dataset in DATASETS:
        print(f"[STANDALONE] {dataset}", flush=True)
        standalone[dataset] = await run_dataset(
            repo_dir=repo_dir,
            dataset=dataset,
            phase="standalone",
            concurrency=args.concurrency,
            requests=args.requests,
            max_tokens=args.max_tokens,
            prompt_mode=args.prompt_mode,
            output_dir=output_dir,
        )

    print("[PARALLEL] gender + age + rosbank", flush=True)
    parallel_started = time.monotonic()
    parallel_results = await asyncio.gather(
        *[
            run_dataset(
                repo_dir=repo_dir,
                dataset=dataset,
                phase="parallel",
                concurrency=args.concurrency,
                requests=args.requests,
                max_tokens=args.max_tokens,
                prompt_mode=args.prompt_mode,
                output_dir=output_dir,
            )
            for dataset in DATASETS
        ]
    )
    parallel_wall = time.monotonic() - parallel_started
    parallel = dict(zip(DATASETS, parallel_results))

    comparison = {}
    for dataset in DATASETS:
        solo = standalone[dataset]
        shared = parallel[dataset]
        comparison[dataset] = {
            "standalone_wall_seconds": solo["wall_seconds"],
            "parallel_wall_seconds": shared["wall_seconds"],
            "slowdown_ratio": shared["wall_seconds"] / solo["wall_seconds"],
            "standalone_requests_per_second": solo["requests_per_second"],
            "parallel_requests_per_second": shared["requests_per_second"],
            "throughput_retention": (
                shared["requests_per_second"] / solo["requests_per_second"]
            ),
            "parallel_transport_failed": shared["transport_failed"],
            "parallel_empty_responses": shared["empty_responses"],
        }

    total_requests = args.requests * len(DATASETS)
    report = {
        "datasets": list(DATASETS),
        "concurrency_per_process": args.concurrency,
        "requests_per_process": args.requests,
        "max_tokens": args.max_tokens,
        "prompt_mode": args.prompt_mode,
        "standalone": standalone,
        "parallel": parallel,
        "comparison": comparison,
        "parallel_group": {
            "wall_seconds": parallel_wall,
            "total_requests": total_requests,
            "aggregate_requests_per_second": total_requests / parallel_wall,
            "transport_failed": sum(
                result["transport_failed"] for result in parallel.values()
            ),
            "empty_responses": sum(
                result["empty_responses"] for result in parallel.values()
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, report)

    for dataset, metrics in comparison.items():
        print(
            f"[RESULT] {dataset}: standalone={metrics['standalone_wall_seconds']:.2f}s "
            f"parallel={metrics['parallel_wall_seconds']:.2f}s "
            f"slowdown={metrics['slowdown_ratio']:.2f}x "
            f"throughput_retention={metrics['throughput_retention']:.2%}",
            flush=True,
        )
    print(
        f"[GROUP] wall={parallel_wall:.2f}s "
        f"aggregate_rps={report['parallel_group']['aggregate_requests_per_second']:.3f} "
        f"errors={report['parallel_group']['transport_failed']} "
        f"empty={report['parallel_group']['empty_responses']}",
        flush=True,
    )
    print(f"[SUMMARY] {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standalone vs three-process parallel LLM load"
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=("short", "full"), default="short")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1 or args.max_tokens < 1:
        parser.error("concurrency, requests and max-tokens must be positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
