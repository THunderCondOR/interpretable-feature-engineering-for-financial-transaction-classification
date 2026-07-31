"""Build a blinded, self-contained HTML form for manual grounding annotation."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from collections import defaultdict
from pathlib import Path


VERDICTS = ["supported", "partially_supported", "unsupported", "not_verifiable"]
CLAIM_TYPES = [
    "direct_observation", "train_relative_comparison",
    "temporal_or_activity_interpretation",
    "higher_level_behavioral_interpretation",
    "demographic_or_social_inference",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_blinded(rows: list[dict], per_dataset: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    cells = defaultdict(list)
    for row in rows:
        cells[(str(row["dataset"]), str(row["run_name"]))].append(row)
    selected = []
    for dataset in sorted({key[0] for key in cells}):
        models = sorted(key[1] for key in cells if key[0] == dataset)
        if len(models) != 2:
            raise ValueError(f"Expected two source models for {dataset}, found {models}")
        allocations = [per_dataset // 2, per_dataset - per_dataset // 2]
        for model, count in zip(models, allocations):
            candidates = cells[(dataset, model)]
            if len(candidates) < count:
                raise ValueError(f"Not enough claims for {dataset}/{model}: {len(candidates)} < {count}")
            selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    visible, key = [], []
    dataset_seen = defaultdict(int)
    calibration_per_dataset = max(1, per_dataset // 3)
    for index, row in enumerate(selected):
        annotation_id = hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode()
        ).hexdigest()[:16]
        dataset = str(row["dataset"])
        phase = "calibration" if dataset_seen[dataset] < calibration_per_dataset else "holdout"
        dataset_seen[dataset] += 1
        visible.append({
            "annotation_id": annotation_id,
            "phase": phase,
            "dataset": dataset,
            "client_stats": row["client_stats"],
            "train_reference_summary": row["train_reference_summary"],
            "field_semantics": row["field_semantics"],
            "claim": row["claim"],
        })
        key.append({
            "annotation_id": annotation_id,
            "sample_id": row["sample_id"],
            "dataset": dataset,
            "run_name": row["run_name"],
            "phase": phase,
        })
    return visible, key


def render(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    verdicts = json.dumps(VERDICTS)
    claim_types = json.dumps(CLAIM_TYPES)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Grounding audit</title>
<style>body{{font:15px system-ui;max-width:1180px;margin:auto;background:#f4f6fa;color:#17213a}}header,.card{{background:white;margin:18px;padding:22px;border-radius:14px}}pre{{white-space:pre-wrap;background:#f7f8fb;padding:12px;max-height:260px;overflow:auto}}.claim{{font-size:19px;font-weight:650}}label{{display:block;margin:7px}}button{{padding:10px 16px;margin:8px}}</style></head><body>
<header><h1>Blinded grounding audit</h1><p id='progress'></p><button onclick='download()'>Download annotations JSON</button></header><div id='app'></div>
<script>const rows={payload}, verdicts={verdicts}, types={claim_types};let answers=JSON.parse(localStorage.getItem('grounding-audit')||'{{}}');
function esc(x){{return String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function radios(name,values,id){{return values.map(v=>`<label><input type='radio' name='${{name}}-${{id}}' value='${{v}}' ${{answers[id]?.[name]===v?'checked':''}} onchange="save('${{id}}','${{name}}','${{v}}')"> ${{v}}</label>`).join('')}}
function render(){{document.getElementById('app').innerHTML=rows.map((r,i)=>`<section class='card'><h2>${{i+1}}. ${{esc(r.dataset)}} · ${{esc(r.phase)}}</h2><div class='claim'>${{esc(r.claim)}}</div><h3>Field semantics</h3><pre>${{esc(r.field_semantics)}}</pre><h3>Client evidence</h3><pre>${{esc(r.client_stats)}}</pre><details><summary>Train-only reference (open only for comparative claims)</summary><pre>${{esc(r.train_reference_summary)}}</pre></details><h3>Verdict</h3>${{radios('verdict',verdicts,r.annotation_id)}}<h3>Claim type</h3>${{radios('claim_type',types,r.annotation_id)}}<label>Notes <input style='width:75%' value='${{esc(answers[r.annotation_id]?.notes||'')}}' oninput="save('${{r.annotation_id}}','notes',this.value)"></label></section>`).join('');progress()}}
function save(id,k,v){{answers[id]=answers[id]||{{annotation_id:id}};answers[id][k]=v;localStorage.setItem('grounding-audit',JSON.stringify(answers));progress()}}
function progress(){{const done=rows.filter(r=>answers[r.annotation_id]?.verdict&&answers[r.annotation_id]?.claim_type).length;document.getElementById('progress').textContent=`Completed ${{done}} / ${{rows.length}}`;}}
function download(){{const out=rows.map(r=>answers[r.annotation_id]||{{annotation_id:r.annotation_id}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='grounding-human-annotations.json';a.click()}}render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-key", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "input": str(args.input), "per_dataset": args.per_dataset}, indent=2))
        return
    visible, key = select_blinded(load_jsonl(args.input), args.per_dataset, args.seed)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render(visible), encoding="utf-8")
    args.output_key.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(visible)} blinded items -> {args.output_html}")


if __name__ == "__main__":
    main()
