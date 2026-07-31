#!/usr/bin/env python3
"""Build reviewer-facing grounding diagnostics and a blinded manual audit."""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERDICTS = (
    "supported", "partially_supported", "unsupported", "not_verifiable"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def grouped_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row.get("judge_name")), str(row.get("dataset")),
            str(row.get("run_name")), str(row.get("claim_type")),
        )].append(row)
    result = []
    for key, items in sorted(groups.items()):
        counts = Counter(str(item.get("verdict")) for item in items)
        result.append({
            "judge": key[0], "dataset": key[1], "source_model": key[2],
            "claim_type": key[3], "n": len(items),
            **{f"n_{verdict}": counts[verdict] for verdict in VERDICTS},
            **{
                f"rate_{verdict}": counts[verdict] / len(items)
                for verdict in VERDICTS
            },
        })
    return result


def select_manual(
    by_sample: dict[str, list[dict[str, Any]]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for sample_id, rows in by_sample.items():
        verdicts = {str(row.get("verdict")) for row in rows}
        priority = (
            "unsupported" if "unsupported" in verdicts
            else "partially_supported" if "partially_supported" in verdicts
            else "not_verifiable" if "not_verifiable" in verdicts
            else "supported"
        )
        exemplar = rows[0]
        buckets[(
            str(exemplar.get("dataset")), str(exemplar.get("run_name")), priority
        )].append(sample_id)

    selected: list[str] = []
    limits = {
        "unsupported": None, "partially_supported": 5,
        "not_verifiable": None, "supported": 3,
    }
    for key, identifiers in sorted(buckets.items()):
        rng.shuffle(identifiers)
        limit = limits[key[2]]
        selected.extend(identifiers if limit is None else identifiers[:limit])
    selected = list(dict.fromkeys(selected))

    visible, key_rows = [], []
    for index, sample_id in enumerate(selected, start=1):
        rows = by_sample[sample_id]
        exemplar = rows[0]
        audit_id = f"GA-{index:04d}"
        visible.append({
            "audit_id": audit_id,
            "dataset": exemplar.get("dataset"),
            "field_semantics": exemplar.get("field_semantics"),
            "train_reference_summary": exemplar.get("train_reference_summary"),
            "client_stats": exemplar.get("client_stats"),
            "claim": exemplar.get("claim"),
            "human_verdict": "",
            "human_claim_type": "",
            "human_comment": "",
        })
        key_rows.append({
            "audit_id": audit_id,
            "sample_id": sample_id,
            "source_model": exemplar.get("run_name"),
            "customer_id": exemplar.get("customer_id"),
            "judge_results": [{
                "judge": row.get("judge_name"), "verdict": row.get("verdict"),
                "claim_type": row.get("claim_type"),
                "confidence": row.get("confidence"),
                "evidence": row.get("evidence"), "reason": row.get("reason"),
            } for row in rows],
        })
    return visible, key_rows


def manual_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    options = "".join(
        f'<option value="{value}">{value}</option>'
        for value in ("", *VERDICTS)
    )
    for row in rows:
        esc = lambda value: html.escape(str(value or ""))
        cards.append(f"""
<article data-id="{esc(row['audit_id'])}">
  <h2>{esc(row['audit_id'])} · {esc(row['dataset'])}</h2>
  <h3>Claim</h3><p class="claim">{esc(row['claim'])}</p>
  <details><summary>Field semantics and train reference</summary>
    <pre>{esc(row['field_semantics'])}\n\n{esc(row['train_reference_summary'])}</pre>
  </details>
  <h3>Client transaction summary</h3><pre>{esc(row['client_stats'])}</pre>
  <label>Human verdict <select class="verdict">{options}</select></label>
  <label>Claim type <input class="claim-type"></label>
  <label>Comment <textarea class="comment" rows="3"></textarea></label>
</article>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Blinded grounding validation</title><style>
body{{font:15px/1.45 system-ui;max-width:1180px;margin:24px auto;padding:0 16px;background:#f4f6f8;color:#17202a}}
article{{background:white;border:1px solid #d9e0e6;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 2px 8px #0001}}
pre{{white-space:pre-wrap;background:#f7f8fa;padding:12px;border-radius:8px;max-height:440px;overflow:auto}}
.claim{{font-size:17px;font-weight:650}} label{{display:block;margin:12px 0}} select,input,textarea{{width:100%;box-sizing:border-box;padding:8px;margin-top:4px}}
button{{position:sticky;top:12px;padding:10px 16px;background:#1666c5;color:white;border:0;border-radius:8px}}
</style></head><body><h1>Blinded grounding validation</h1>
<p>The source model and automatic judge verdicts are intentionally hidden. Unsupported cases are oversampled; do not interpret this sheet as a prevalence estimate.</p>
<button onclick="download()">Download annotations JSON</button>{''.join(cards)}
<script>function download(){{const rows=[...document.querySelectorAll('article')].map(x=>({{audit_id:x.dataset.id,human_verdict:x.querySelector('.verdict').value,human_claim_type:x.querySelector('.claim-type').value,human_comment:x.querySelector('.comment').value}}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{{type:'application/json'}}));a.download='grounding_human_annotations.json';a.click();}}</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "inputs": [str(path) for path in args.inputs],
        "judgments": len(rows), "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not rows:
        raise ValueError("No grounding judgments found")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)
    unsupported = [
        row for row in rows if row.get("verdict") == "unsupported"
    ]
    disagreements = []
    for sample_id, items in by_sample.items():
        verdicts = sorted({str(item.get("verdict")) for item in items})
        if len(verdicts) > 1:
            disagreements.append({
                "sample_id": sample_id, "dataset": items[0].get("dataset"),
                "source_model": items[0].get("run_name"),
                "claim": items[0].get("claim"), "verdicts": " | ".join(verdicts),
            })
    breakdown = grouped_breakdown(rows)
    visible, key_rows = select_manual(by_sample, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "judgments.csv", rows)
    write_csv(args.output_dir / "breakdown.csv", breakdown)
    write_csv(args.output_dir / "unsupported_cases.csv", unsupported)
    write_csv(args.output_dir / "disagreements.csv", disagreements)
    atomic_json(args.output_dir / "manual_validation_blinded.json", visible)
    atomic_json(args.output_dir / "manual_validation_key.json", key_rows)
    (args.output_dir / "manual_validation.html").write_text(
        manual_html(visible), encoding="utf-8"
    )
    summary = {
        **plan, "unique_samples": len(by_sample),
        "judges": sorted({str(row.get("judge_name")) for row in rows}),
        "verdicts": dict(Counter(str(row.get("verdict")) for row in rows)),
        "unsupported_judgments": len(unsupported),
        "disagreements": len(disagreements),
        "manual_validation_items": len(visible),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    lines = [
        "# Detailed grounding report", "", f"Judgments: {len(rows)}", "",
        f"Unique claims: {len(by_sample)}", "",
        f"Unsupported judgments: {len(unsupported)}", "",
        f"Manual validation items: {len(visible)}", "", "## Breakdown", "",
        "| Judge | Dataset | Source | Claim type | N | Supported | Partial | Unsupported | Not verifiable |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in breakdown:
        lines.append(
            f"| {row['judge']} | {row['dataset']} | {row['source_model']} | "
            f"{row['claim_type']} | {row['n']} | {row['rate_supported']:.1%} | "
            f"{row['rate_partially_supported']:.1%} | {row['rate_unsupported']:.1%} | "
            f"{row['rate_not_verifiable']:.1%} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved detailed grounding report -> {args.output_dir}")


if __name__ == "__main__":
    main()
