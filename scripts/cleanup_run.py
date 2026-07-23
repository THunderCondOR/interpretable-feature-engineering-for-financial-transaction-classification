#!/usr/bin/env python3
"""Safely remove artifacts owned by exactly one versioned run.

Dry-run is the default.  Deletion requires ``--execute`` and is refused while
any other process command line contains the exact run identifier.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


def live_processes(run_id: str, *, proc_root: Path = Path("/proc")) -> list[dict]:
    """Return live processes whose argv contains the exact run-id argument."""
    matches: list[dict] = []
    own_pid = os.getpid()
    for entry in proc_root.iterdir() if proc_root.is_dir() else ():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            decoded = [value.decode("utf-8", errors="replace") for value in argv if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if run_id in decoded:
            matches.append({"pid": int(entry.name), "argv": decoded})
    return sorted(matches, key=lambda row: row["pid"])


def matching_result_dirs(results_root: Path, run_id: str) -> list[Path]:
    """Find result directories whose own manifest declares exactly ``run_id``."""
    matches: list[Path] = []
    if not results_root.exists():
        return matches
    for manifest in results_root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        declared = (
            payload.get("run_id")
            or payload.get("identity", {}).get("run_id")
            or payload.get("experiment", {}).get("run_id")
        )
        if declared == run_id:
            matches.append(manifest.parent)
    return sorted(set(matches))


def cleanup_targets(
    run_id: str,
    *,
    results_root: Path,
    logs_root: Path,
    reports_root: Path,
) -> list[Path]:
    targets = matching_result_dirs(results_root, run_id)
    for root in (logs_root, reports_root):
        candidate = root / run_id
        if candidate.exists():
            targets.append(candidate)
    return sorted(set(path.resolve() for path in targets))


def _assert_below(path: Path, allowed_roots: Iterable[Path]) -> None:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
        raise RuntimeError(f"Refusing to delete path outside configured roots: {resolved}")


def remove_run(
    run_id: str,
    *,
    results_root: Path = Path("results/v2"),
    logs_root: Path = Path("logs/runs"),
    reports_root: Path = Path("reports"),
    execute: bool = False,
    proc_root: Path = Path("/proc"),
) -> dict:
    if not run_id.strip() or run_id in {".", ".."}:
        raise ValueError("A non-empty exact run ID is required")
    processes = live_processes(run_id, proc_root=proc_root)
    targets = cleanup_targets(
        run_id,
        results_root=results_root,
        logs_root=logs_root,
        reports_root=reports_root,
    )
    plan = {
        "mode": "execute" if execute else "dry-run",
        "run_id": run_id,
        "live_processes": processes,
        "targets": [str(path) for path in targets],
    }
    if not execute:
        return plan
    if processes:
        raise RuntimeError(
            f"Refusing cleanup: run {run_id!r} still has {len(processes)} live process(es)"
        )
    allowed = [results_root, logs_root, reports_root]
    for target in targets:
        _assert_below(target, allowed)
        shutil.rmtree(target)
    plan["removed"] = plan["targets"]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--logs-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = remove_run(
        args.run_id,
        results_root=args.results_root,
        logs_root=args.logs_root,
        reports_root=args.reports_root,
        execute=args.execute,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
