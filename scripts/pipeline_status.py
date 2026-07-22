"""Render terminal and atomic static-HTML status from append-only event logs."""
from __future__ import annotations
import argparse
import html
import time
from pathlib import Path

from src.experiments.events import STAGES, read_events, status_snapshot


def event_state(event):
    name = event.get("state") or event.get("event", "pending")
    completed, expected = event.get("completed", 0), event.get("expected", 0)
    progress = f" {completed}/{expected}" if expected else ""
    concurrency = f" c={event['concurrency']}" if event.get("concurrency") else ""
    return f"{name}{progress}{concurrency}"


def terminal_table(snapshot):
    cells = snapshot["cells"]
    keys = sorted({key.split("|", 1)[0] for key in cells}) or ["no-events"]
    display_stages = ["stats", "prompts", "explanations", "claims", "embeddings", "clusters", "features", "ml", "grounding", "reports"]
    widths = [24] + [14] * len(display_stages)
    lines = ["".join(value[:width - 1].ljust(width) for value, width in zip(["dataset/model", *display_stages], widths))]
    for key in keys:
        values = [key] + [event_state(cells.get(f"{key}|{stage}", {})) if f"{key}|{stage}" in cells else "·" for stage in display_stages]
        lines.append("".join(value[:width - 1].ljust(width) for value, width in zip(values, widths)))
    lines.append(f"Errors: {snapshot['errors'] or {}}")
    return "\n".join(lines)


def html_document(snapshot, run_id):
    cells = snapshot["cells"]
    keys = sorted({key.split("|", 1)[0] for key in cells}) or ["no-events"]
    rows = []
    for key in keys:
        stage_cells = []
        for stage in STAGES:
            event = cells.get(f"{key}|{stage}")
            state = event_state(event) if event else "pending"
            css = "done" if event and (event.get("state") == "completed" or event.get("event") == "window_committed" and event.get("completed") == event.get("expected")) else "running" if event else "pending"
            stage_cells.append(f'<td class="{css}"><b>{html.escape(stage)}</b><span>{html.escape(state)}</span></td>')
        rows.append(f"<tr><th>{html.escape(key)}</th>{''.join(stage_cells)}</tr>")
    history = "".join(f"<li>{html.escape(str(item['key']))}: {item['concurrency']} ({html.escape(str(item.get('mode')) )})</li>" for item in snapshot["concurrency_history"][-50:]) or "<li>No samples yet</li>"
    errors = "".join(f"<li>{html.escape(name)}: {count}</li>" for name, count in snapshot["errors"].items()) or "<li>No errors</li>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(run_id)} status</title><style>
body{{font:14px system-ui;background:#f4f6fb;color:#15213a;margin:28px}}h1{{margin-bottom:4px}}.card{{background:white;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 7px 24px #1b2b4b12;overflow:auto}}table{{border-collapse:separate;border-spacing:6px}}th{{text-align:left;white-space:nowrap}}td{{min-width:120px;padding:10px;border-radius:9px;background:#edf0f5}}td span{{display:block;font-size:11px;margin-top:5px;color:#5a6475}}td.done{{background:#dff5e5}}td.running{{background:#fff1c8}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}ul{{margin:0;padding-left:20px}}
</style></head><body><h1>Run {html.escape(run_id)}</h1><p>Updated: {html.escape(str(snapshot['updated_at']))}</p><section class="card"><table>{''.join(rows)}</table></section><div class="grid"><section class="card"><h2>Concurrency history</h2><ul>{history}</ul></section><section class="card"><h2>Errors by type</h2><ul>{errors}</ul></section></div></body></html>"""


def render(run_id, logs_root, results_root, output):
    paths = list((logs_root / run_id).glob("**/*.events.jsonl"))
    paths += list(results_root.glob("**/*.events.jsonl"))
    snapshot = status_snapshot(read_events(paths))
    print(terminal_table(snapshot))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(html_document(snapshot, run_id), encoding="utf-8")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", type=float, default=0)
    args = parser.parse_args()
    output = args.output or Path("reports") / args.run_id / "status.html"
    while True:
        render(args.run_id, args.logs_root, args.results_root, output)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
