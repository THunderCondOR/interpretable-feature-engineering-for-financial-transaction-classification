"""Build offline reviewer-revision HTML and PDF."""
from __future__ import annotations
import argparse
import html
import textwrap
from pathlib import Path

NAVY, BLUE, TEAL, GOLD, RED = "#17253f", "#3867d6", "#16a085", "#f2b134", "#d9534f"

CONCERNS = [
    ("Train-only boundary", "Technically closed", "Hashed train-only summaries, demos, clustering and selection."),
    ("Heavy-tailed statistics", "Technically closed", "Untrimmed observations with P5/Q1/median/Q3/P95."),
    ("Stereotype priming", "Technically closed", "Neutral prompts and evidence/interpretation separation."),
    ("Cluster stability", "Awaiting results", "Seeds, thresholds, K, coverage, encoding, model matching."),
    ("Claim grounding", "Awaiting results", "Blinded judges, kappa, client bootstrap and adjudication."),
    ("Classifier faithfulness", "Awaiting results", "Surrogate fidelity plus classifier occlusion."),
    ("Human usefulness", "Explicit limitation", "No utility claim without a user study."),
]

SECTIONS = [
    ("Reviewer concern matrix", "Code-level closure is separated from empirical evidence still awaiting runs."),
    ("Found defects and corrections", "Amount meanings, negative sorting, Rosbank recency, hidden reasoning leakage and label-filtered clusters are corrected and regression-tested."),
    ("Revision architecture", "Versioned profiles feed neutral prompts, atomic generation, stable claims, label-agnostic clustering, frozen transforms and seeded evaluation."),
    ("Train-only data boundary", "Validation selects prompts and hyperparameters. Validation/test labels never construct summaries, demonstrations, clusters, filters or features."),
    ("Artifact invalidation DAG", "Prompt changes invalidate explanations, claims, embeddings, clusters, features and CoT/concat ML; independent baselines remain reusable."),
    ("Robust transaction statistics", "Each dataset declares amount semantics; untrimmed client-level quantiles and category prevalence/share export to JSON, CSV, Markdown and LaTeX."),
    ("Neutral gender pilot", "Four variants use 400 stratified validation clients (seed 137), balanced accuracy and a predeclared zero-shot equivalence rule."),
    ("Atomic API scheduler", "Any 429 rolls back the whole window; 64 → 60-second cooldown → ten clean windows of 10 → probe 64."),
    ("64 → cooldown → 10×10 → 64 state machine", "Fallback counts only committed clean windows. A 429 at 10 rolls back the window and resets recovery."),
    ("Parallel Qwen / GPT queues", "Workers have independent semaphores, limiter states, locks, events and logs. The launcher is dry-run by default."),
    ("Live status and provenance", "Append-only events drive terminal and HTML progress, error, concurrency, retry, throughput and checkpoint views."),
    ("Claims and semantic clustering", "Stable claim IDs preserve source and extractor hashes; duplicate texts embed once; labels are post-hoc metadata only."),
    ("Publication cluster map", "UMAP is presentation-only: atomic colors, prevalence-scaled points, gray outliers, macro-theme contours and callouts."),
    ("Stability experiments", "Seeds 17/101/947; common 300-client anchors; thresholds, fixed K, coverage, encodings and ML seeds are reported as separate axes."),
    ("Grounding", "Fifty clients and up to three claims per dataset/model cell; exact hashed evidence; two blinded judges; Codex adjudicates selected cases."),
    ("Classifier fidelity", "A validation-selected non-claim teacher is compared with claim-based logistic, XGBoost and shallow-tree surrogates."),
    ("Cluster and tree explorer", "Atomic clusters, macro-theme contours, representative claims, prevalence, compactness, lift, cross-model matches and tree paths."),
    ("Rerun matrix", "Gender gets neutral API reruns; age and Rosbank retain immutable legacy API outputs and receive offline analyses; scoring is excluded."),
    ("Future launch commands", "Computational runners require --execute; API runners additionally require --execute-api and --until-complete."),
    ("Reviewer response checklist", "Stability, grounding and fidelity remain awaiting results; human usefulness remains an explicit limitation."),
]


def flow_svg(labels):
    width, box, gap = 1120, 135, 18
    start = (width - len(labels) * box - (len(labels) - 1) * gap) / 2
    parts = ['<svg viewBox="0 0 1120 180"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#17253f"/></marker></defs>']
    for i, label in enumerate(labels):
        x, color = start + i * (box + gap), [BLUE, TEAL, GOLD, "#8e6bbf"][i % 4]
        parts.append(f'<rect x="{x}" y="55" width="{box}" height="62" rx="13" fill="{color}"/><text x="{x+box/2}" y="91" text-anchor="middle" fill="white" font-size="14" font-weight="700">{html.escape(label)}</text>')
        if i:
            parts.append(f'<path d="M{x-gap+2},86 L{x-6},86" stroke="{NAVY}" stroke-width="3" marker-end="url(#a)"/>')
    parts.append("</svg>")
    return "".join(parts)


def scheduler_svg():
    return f'''<svg viewBox="0 0 1120 260"><defs><marker id="b" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="{NAVY}"/></marker></defs>
<rect x="40" y="90" width="220" height="82" rx="16" fill="{BLUE}"/><text x="150" y="124" text-anchor="middle" fill="white" font-size="20" font-weight="700">HIGH 64</text><text x="150" y="151" text-anchor="middle" fill="white">atomic window 64</text>
<rect x="450" y="90" width="220" height="82" rx="16" fill="{RED}"/><text x="560" y="124" text-anchor="middle" fill="white" font-size="20" font-weight="700">ROLLBACK</text><text x="560" y="151" text-anchor="middle" fill="white">cooldown 60 seconds</text>
<rect x="860" y="90" width="220" height="82" rx="16" fill="{TEAL}"/><text x="970" y="124" text-anchor="middle" fill="white" font-size="20" font-weight="700">FALLBACK 10</text><text x="970" y="151" text-anchor="middle" fill="white">10 clean windows</text>
<path d="M260,131 L440,131" stroke="{NAVY}" stroke-width="4" marker-end="url(#b)"/><text x="350" y="115" text-anchor="middle" fill="{RED}">any 429</text><path d="M670,131 L850,131" stroke="{NAVY}" stroke-width="4" marker-end="url(#b)"/><path d="M970,82 C970,20 150,20 150,82" fill="none" stroke="{NAVY}" stroke-width="4" marker-end="url(#b)"/><text x="560" y="35" text-anchor="middle">probe 64 after exactly 10 clean windows</text></svg>'''


def cluster_svg():
    dots = []
    for ci, (cx, cy, color) in enumerate([(170,130,"#76c66b"),(330,85,"#f39c35"),(520,175,"#5dade2"),(700,95,"#af7ac5"),(800,205,"#ef6f6c")]):
        for j in range(20):
            dots.append(f'<circle cx="{cx+((j*37+ci*13)%72)-36}" cy="{cy+((j*23+ci*17)%52)-26}" r="5" fill="{color}" opacity=".8"/>')
    return f'<svg viewBox="0 0 1120 300"><rect width="1120" height="300" rx="18" fill="#fbfcff"/><path d="M70,55 C200,20 400,40 410,230 C260,280 100,240 70,55Z" fill="none" stroke="{NAVY}" stroke-width="2"/><path d="M440,45 C620,20 875,35 875,245 C680,280 500,260 440,45Z" fill="none" stroke="{NAVY}" stroke-width="2"/>{"".join(dots)}<rect x="900" y="35" width="200" height="225" rx="14" fill="white" stroke="#dfe5ef"/><text x="920" y="68" font-weight="700">Representative claims</text><text x="920" y="108">① medoid</text><text x="920" y="143">② prevalence + lift</text><text x="920" y="178">③ cross-model match</text><text x="920" y="213">④ tree-node links</text></svg>'


def build_html():
    matrix = "".join(f'<tr><td>{html.escape(a)}</td><td><span class="pill">{html.escape(b)}</span></td><td>{html.escape(c)}</td></tr>' for a,b,c in CONCERNS)
    cards = []
    for number, (title, body) in enumerate(SECTIONS, 1):
        visual = ""
        if title == "Reviewer concern matrix": visual = f'<table><tr><th>Concern</th><th>Status</th><th>Action</th></tr>{matrix}</table>'
        elif title == "Revision architecture": visual = flow_svg(["Raw splits","Robust profiles","Neutral prompts","Atomic API","Claims","Clusters","ML"])
        elif title == "Train-only data boundary": visual = flow_svg(["TRAIN fit","Frozen summary","Frozen centroids","VAL tune","TEST once"])
        elif title == "Artifact invalidation DAG": visual = flow_svg(["Prompt","Explanation","Claims","Clusters","Features","CoT ML"])
        elif title in {"Atomic API scheduler", "64 → cooldown → 10×10 → 64 state machine"}: visual = scheduler_svg()
        elif title in {"Claims and semantic clustering", "Publication cluster map", "Cluster and tree explorer"}: visual = cluster_svg()
        cards.append(f'<section><div class="num">{number:02d}</div><h2>{html.escape(title)}</h2><p>{html.escape(body)}</p>{visual}</section>')
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Reviewer revision v2</title><style>
@page{{size:A4 landscape;margin:14mm}}@font-face{{font-family:DejaVu;src:url("file:///usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")}}*{{box-sizing:border-box}}body{{margin:0;background:#edf1f7;color:{NAVY};font-family:DejaVu,Arial,sans-serif}}header{{padding:55px 7%;background:linear-gradient(130deg,{NAVY},{BLUE});color:white}}h1{{font-size:42px;margin:0 0 12px}}header p{{font-size:18px;max-width:850px}}main{{max-width:1180px;margin:auto;padding:28px}}section{{position:relative;background:white;margin:22px 0;padding:30px 34px;border-radius:18px;box-shadow:0 10px 30px #17253f12;break-inside:avoid;page-break-inside:avoid}}.num{{position:absolute;right:28px;top:20px;font-size:38px;font-weight:800;color:#e5eaf2}}h2{{font-size:27px;margin:0 0 12px}}p{{font-size:16px;line-height:1.55;max-width:1000px}}svg{{width:100%;margin-top:12px}}table{{border-collapse:collapse;width:100%;margin-top:18px}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #e5eaf2}}.pill{{background:#e8eefc;border-radius:99px;padding:5px 9px;font-size:12px;font-weight:700}}code{{background:#eef2f7;padding:3px 6px;border-radius:5px}}footer{{padding:30px;text-align:center;color:#6d7788}}</style></head><body><header><h1>Reviewer Revision v2</h1><p>Reproducible architecture, experiment matrix, atomic LLM execution, stability, grounding, surrogate fidelity, and publication reporting.</p></header><main>{''.join(cards)}</main><footer>Generated offline · no experimental result is implied by this design document</footer></body></html>'''


def fallback_pdf(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY})
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis("off"); fig.patch.set_facecolor(NAVY)
        ax.text(.07,.72,"Reviewer Revision v2",fontsize=31,weight="bold",color="white")
        ax.text(.07,.59,"Reproducible architecture · robust statistics · neutral prompts\natomic LLM execution · stability · grounding · surrogate fidelity",fontsize=16,color="#dfe8ff",linespacing=1.7)
        ax.add_patch(plt.Rectangle((.07,.20),.38,.12,color=BLUE,transform=ax.transAxes)); ax.text(.09,.255,"DESIGN DOCUMENT · NO RESULTS IMPLIED",color="white",fontsize=11,weight="bold")
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69,8.27)); ax.axis("off"); ax.set_title("Reviewer concern matrix",loc="left",fontsize=24,weight="bold",pad=20)
        table=ax.table(cellText=CONCERNS,colLabels=["Concern","Status","Action"],cellLoc="left",colLoc="left",bbox=[0,.12,1,.76],colWidths=[.25,.2,.55]); table.auto_set_font_size(False); table.set_fontsize(9)
        for (row,col),cell in table.get_celld().items(): cell.set_edgecolor("#dfe5ef"); cell.set_facecolor(NAVY if row==0 else "#f7f9fc"); cell.get_text().set_color("white" if row==0 else NAVY)
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

        for start in range(1,len(SECTIONS),2):
            fig, axes=plt.subplots(2,1,figsize=(11.69,8.27)); fig.subplots_adjust(hspace=.32,top=.93,bottom=.07)
            for offset,ax in enumerate(axes):
                index=start+offset
                ax.axis("off"); ax.add_patch(plt.Rectangle((0,0),1,1,color="#f7f9fc",transform=ax.transAxes,zorder=-1))
                if index>=len(SECTIONS): continue
                title,body=SECTIONS[index]
                ax.text(.03,.83,f"{index+1:02d}  {title}",fontsize=17,weight="bold",transform=ax.transAxes)
                wrapped="\n".join(textwrap.wrap(body,130))
                ax.text(.03,.63,wrapped,fontsize=10.5,linespacing=1.55,transform=ax.transAxes,va="top")
                if title in {"Revision architecture","Train-only data boundary","Artifact invalidation DAG"}:
                    labels={"Revision architecture":["Profiles","Prompts","API","Claims","Clusters","ML"],"Train-only data boundary":["TRAIN fit","Freeze","VAL tune","TEST once"],"Artifact invalidation DAG":["Prompt","Explanation","Claims","Clusters","Features","ML"]}[title]
                    xs=[.06+i*.86/(len(labels)-1) for i in range(len(labels))]
                    for i,(x,label) in enumerate(zip(xs,labels)):
                        ax.text(x,.19,label,ha="center",va="center",fontsize=8,color="white",bbox=dict(boxstyle="round,pad=.6",fc=[BLUE,TEAL,GOLD,"#8e6bbf"][i%4],ec="none"),transform=ax.transAxes)
                        if i: ax.annotate("",xy=(x-.05,.19),xytext=(xs[i-1]+.05,.19),arrowprops=dict(arrowstyle="->",color=NAVY),xycoords=ax.transAxes)
                if "64" in title or title=="Atomic API scheduler":
                    ax.text(.12,.18,"HIGH 64",color="white",bbox=dict(boxstyle="round,pad=.7",fc=BLUE,ec="none"),transform=ax.transAxes); ax.text(.45,.18,"ROLLBACK + 60s",color="white",bbox=dict(boxstyle="round,pad=.7",fc=RED,ec="none"),transform=ax.transAxes); ax.text(.78,.18,"10 × 10",color="white",bbox=dict(boxstyle="round,pad=.7",fc=TEAL,ec="none"),transform=ax.transAxes)
            pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html",type=Path,default=Path("docs/reviewer_revision_pipeline.html"))
    parser.add_argument("--pdf",type=Path,default=Path("docs/reviewer_revision_pipeline.pdf"))
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        print(f"DRY RUN: build {args.html} and {args.pdf}; pass --execute")
        return
    args.html.parent.mkdir(parents=True,exist_ok=True); args.html.write_text(build_html(),encoding="utf-8")
    try:
        from weasyprint import HTML
        HTML(filename=str(args.html)).write_pdf(str(args.pdf))
        backend="WeasyPrint"
    except ImportError:
        fallback_pdf(args.pdf); backend="Matplotlib PDF fallback (WeasyPrint unavailable)"
    print(f"Built {args.html} and {args.pdf} using {backend}")


if __name__=="__main__":
    main()
