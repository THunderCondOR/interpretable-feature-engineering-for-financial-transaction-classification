#!/usr/bin/env python3
"""Build a preselected blinded human-validation form for stereotype auditing."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_stereotype_audit_sample import read_jsonl


VERDICTS = [
    "no_sensitive_inference",
    "evidence_bounded_sensitive_inference",
    "weakly_grounded_sensitive_inference",
    "unsupported_stereotype",
    "not_assessable",
]
FLAGS = [
    "external_group_generalization", "categorical_personal_attribute",
    "invented_age_boundary", "top_k_absence_as_evidence",
    "essentialist_or_causal_leap",
]


def select_human_items(
    samples: list[dict[str, Any]], private: list[dict[str, Any]],
    *, total: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private_by_id = {str(row["sample_id"]): row for row in private}
    cells = sorted({(str(row["dataset"]), str(row["run_name"])) for row in samples})
    if len(cells) != 4 or total % len(cells):
        raise ValueError(f"human-items={total} must be divisible by four audit cells")
    per_cell = total // len(cells)
    by_dataset_client: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in samples:
        by_dataset_client[str(row["dataset"])][str(row["customer_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for dataset, client_rows in sorted(by_dataset_client.items()):
        paired = {
            cid: rows for cid, rows in client_rows.items()
            if len({str(row["run_name"]) for row in rows}) == 2
        }
        labels: dict[str, list[str]] = defaultdict(list)
        for cid, rows in paired.items():
            labels[str(private_by_id[str(rows[0]["sample_id"])]["true_label"])].append(cid)
        base, remainder = divmod(per_cell, len(labels))
        chosen_clients: list[str] = []
        for index, (label, candidates) in enumerate(sorted(labels.items())):
            count = base + int(index < remainder)
            rng = random.Random(f"{seed}:{dataset}:{label}")
            chosen_clients.extend(rng.sample(sorted(candidates), count))
        for cid in chosen_clients:
            selected.extend(paired[cid])
    if len(selected) != total:
        raise ValueError(f"Selected {len(selected)} human items, expected {total}")
    random.Random(seed).shuffle(selected)
    visible, key = [], []
    for index, row in enumerate(selected, start=1):
        audit_id = f"SA-{index:03d}-" + hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode()
        ).hexdigest()[:8]
        visible.append({
            "audit_id": audit_id,
            "dataset": row["dataset"],
            "generator_instruction": row["generator_system_prompt"],
            "train_reference_and_client_profile": row["generator_user_prompt"],
            "rationale": row["rationale"],
        })
        secret = private_by_id[str(row["sample_id"])]
        key.append({
            "audit_id": audit_id,
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "customer_id": row["customer_id"],
            "source_model": row["run_name"],
            "true_label": secret["true_label"],
            "prediction_correct": secret["prediction_correct"],
        })
    return visible, key


def render(rows: list[dict[str, Any]]) -> str:
    cards = []
    verdict_options = "".join(
        f'<option value="{value}">{value}</option>' for value in ("", *VERDICTS)
    )
    for row in rows:
        esc = lambda value: html.escape(str(value or ""))
        flag_boxes = "".join(
            f'<label><input type="checkbox" class="flag" value="{flag}"> {flag}</label>'
            for flag in FLAGS
        )
        cards.append(f"""
<article data-id="{esc(row['audit_id'])}">
  <h2>{esc(row['audit_id'])} · {esc(row['dataset'])}</h2>
  <h3>Rationale</h3><pre class="rationale">{esc(row['rationale'])}</pre>
  <details><summary>Exact generation instruction and evidence</summary>
    <h3>Instruction</h3><pre>{esc(row['generator_instruction'])}</pre>
    <h3>Train reference and client profile</h3><pre>{esc(row['train_reference_and_client_profile'])}</pre>
  </details>
  <label>Verdict <select class="verdict">{verdict_options}</select></label>
  <fieldset><legend>Flags</legend>{flag_boxes}</fieldset>
  <label>Quoted span <textarea class="quote" rows="2"></textarea></label>
  <label>Comment <textarea class="comment" rows="3"></textarea></label>
</article>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Blinded stereotype audit</title><style>
body{{font:15px/1.45 system-ui;max-width:1180px;margin:24px auto;padding:0 16px;background:#f4f6f8;color:#17202a}}
article{{background:white;border:1px solid #d9e0e6;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 2px 8px #0001}}
pre{{white-space:pre-wrap;background:#f7f8fa;padding:12px;border-radius:8px;max-height:520px;overflow:auto}}
.rationale{{font-size:16px}}label{{display:block;margin:10px 0}}select,textarea{{width:100%;box-sizing:border-box;padding:8px}}
fieldset label{{display:inline-block;margin-right:16px}}button{{position:sticky;top:12px;padding:10px 16px;background:#1666c5;color:white;border:0;border-radius:8px}}
</style></head><body><h1>Blinded stereotype audit</h1>
<p id="progress"></p><p>This random sample was selected before automatic judgments. Source model, true label, correctness, and judge outputs are hidden.</p>
<button onclick="downloadAudit()">Download annotations JSON</button>{''.join(cards)}
<script>
const storageKey='stereotype-human-audit-v1';let saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
function collect(){{return [...document.querySelectorAll('article')].map(card=>({{audit_id:card.dataset.id,human_verdict:card.querySelector('.verdict').value,human_flags:[...card.querySelectorAll('.flag:checked')].map(x=>x.value),human_quoted_span:card.querySelector('.quote').value,human_comment:card.querySelector('.comment').value}}));}}
function persist(){{for(const row of collect())saved[row.audit_id]=row;localStorage.setItem(storageKey,JSON.stringify(saved));progress();}}
function restore(){{for(const card of document.querySelectorAll('article')){{const row=saved[card.dataset.id];if(!row)continue;card.querySelector('.verdict').value=row.human_verdict||'';for(const box of card.querySelectorAll('.flag'))box.checked=(row.human_flags||[]).includes(box.value);card.querySelector('.quote').value=row.human_quoted_span||'';card.querySelector('.comment').value=row.human_comment||'';}}}}
function progress(){{const rows=collect(),done=rows.filter(x=>x.human_verdict).length;document.getElementById('progress').textContent=`Completed ${{done}} / ${{rows.length}}`;}}
function downloadAudit(){{persist();const rows=collect();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{{type:'application/json'}}));a.download='stereotype_human_annotations.json';a.click();}}
document.body.addEventListener('change',persist);document.body.addEventListener('input',persist);restore();progress();
</script></body></html>"""


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-key", type=Path, required=True)
    parser.add_argument("--human-items", type=int, default=40)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "human_items": args.human_items, "output": str(args.output_html),
    }, indent=2))
    if not args.execute:
        return
    visible, key = select_human_items(
        read_jsonl(args.input), json.loads(args.private_key.read_text(encoding="utf-8")),
        total=args.human_items, seed=args.seed,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render(visible), encoding="utf-8")
    atomic_json(args.output_key, key)
    print(f"Saved {len(visible)} blinded human items -> {args.output_html}")


if __name__ == "__main__":
    main()
