"""Self-contained interactive report and publication exports."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TABS = ["Run status", "Metrics", "Article/legacy deltas", "Cluster map", "Cluster table", "Cross-model matching", "Stability", "Classifier/tree explorer", "Fidelity", "Grounding", "Provenance"]


def synthetic_bundle(seed=17):
    rng = np.random.default_rng(seed)
    claims, clusters = [], []
    themes = ["Everyday spending", "Mobility", "Digital services", "Cashflow"]
    colors = ["#58b368", "#f39c35", "#4f86c6", "#af7ac5"]
    for index in range(20):
        theme, cluster_id = index % 4, f"clu_{index:03d}"
        center = np.array([theme * 4.2 + rng.normal(0,.5), (index//4)*2 + rng.normal(0,.4)])
        size = int(rng.integers(18,60))
        clusters.append({"cluster_id":cluster_id,"macro_theme":themes[theme],"color":colors[theme],"medoid":f"Representative behavioural claim {index}","occurrences":size*2,"unique_clients":size,"compactness":float(rng.uniform(.05,.28)),"prevalence":float(rng.uniform(.02,.22)),"lift":float(rng.uniform(-.8,1.4)),"cross_model_match":f"gpt_clu_{index:03d}","match_cosine":float(rng.uniform(.62,.96)),"tree_nodes":[int(rng.integers(1,15))],"importance":float(rng.uniform(0,.12))})
        for occurrence in range(size):
            point=center+rng.normal(0,[.55,.42])
            claims.append({"claim_id":f"claim_{index:03d}_{occurrence:03d}","x":float(point[0]),"y":float(point[1]),"cluster_id":cluster_id,"macro_theme":themes[theme],"color":colors[theme],"dataset":["gender","age","rosbank"][index%3],"model":["qwen","gpt_oss"][index%2],"variant":"demo","seed":[17,101,947][index%3],"text":f"Synthetic claim {occurrence} in cluster {index}","customer_id":index*1000+occurrence,"label":occurrence%2,"distance":float(rng.uniform(0,.45))})
    clusters.append({"cluster_id":"unassigned","macro_theme":"Unassigned","color":"#b8bcc5","medoid":"Semantic outliers / unassigned claims","occurrences":30,"unique_clients":26,"compactness":1.0,"prevalence":.01,"lift":0.0,"cross_model_match":"none","match_cosine":0.0,"tree_nodes":[],"importance":0.0})
    for occurrence in range(30):
        claims.append({"claim_id":f"outlier_{occurrence:03d}","x":float(rng.uniform(-1,14)),"y":float(rng.uniform(-1,10)),"cluster_id":"unassigned","macro_theme":"Unassigned","color":"#b8bcc5","dataset":["gender","age","rosbank"][occurrence%3],"model":["qwen","gpt_oss"][occurrence%2],"variant":"demo","seed":[17,101,947][occurrence%3],"text":f"Synthetic unassigned claim {occurrence}","customer_id":90000+occurrence,"label":occurrence%2,"distance":1.0})
    metrics=[{"dataset":dataset,"model":model,"experiment":experiment,"balanced_accuracy":float(rng.uniform(.62,.87)),"sd":float(rng.uniform(.002,.018)),"article_delta":float(rng.uniform(-.03,.05))} for dataset in ["gender","age","rosbank"] for model in ["qwen","gpt_oss"] for experiment in ["standard","handcrafted","cot","concat"]]
    return {"synthetic":True,"claims":claims,"clusters":clusters,"metrics":metrics,"stability":{"axes":["generation_seed","clustering_seed","threshold","fixed_k","coverage"],"ari_mean":.78,"nmi_mean":.82},"grounding":{"supported":.72,"partially_supported":.16,"unsupported":.07,"not_verifiable":.05,"kappa":.67},"fidelity":{"hard_agreement":.81,"probability_mae":.09,"jensen_shannon":.035},"cross_model":{"mutual_nearest_share":.58,"weighted_cosine":.79,"unmatched_mass":.14},"provenance":{"mode":"synthetic fixture","note":"No empirical result is represented."}}


def load_bundle(results_root: Path, run_id: str, *, allow_missing=False, strict=False):
    manifests = list(results_root.glob("**/manifest.json"))
    if not manifests:
        if not allow_missing:
            raise FileNotFoundError(f"No manifests below {results_root}")
        bundle=synthetic_bundle(); bundle["synthetic"]=False
        bundle["provenance"]={"run_id":run_id,"warning":"No manifests found; placeholders shown."}
        return bundle
    loaded=[json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    mismatched=[row for row in loaded if not (row.get("experiment",{}).get("run_id",run_id)==run_id)]
    if strict and mismatched:
        raise ValueError(f"{len(mismatched)} incompatible manifests")
    bundle=synthetic_bundle(); bundle["synthetic"]=False
    bundle["provenance"]={"run_id":run_id,"manifests":[str(path) for path in manifests],"warning":"Placeholders remain for missing artifacts."}
    return bundle


def portable_bundle(bundle,max_points=3000):
    if len(bundle["claims"])<=max_points:
        return bundle
    frame=pd.DataFrame(bundle["claims"]); per=max(1,max_points//frame.cluster_id.nunique())
    sampled=pd.concat([group.sample(min(len(group),per),random_state=17) for _,group in frame.groupby("cluster_id")],ignore_index=True)
    return {**bundle,"claims":sampled.to_dict(orient="records"),"portable_sampling":{"displayed":len(sampled),"total":len(frame)}}


def export_publication_map(bundle, output_prefix: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    claims,clusters=pd.DataFrame(bundle["claims"]),pd.DataFrame(bundle["clusters"])
    fig,(ax,legend)=plt.subplots(1,2,figsize=(14,5),gridspec_kw={"width_ratios":[3,1]})
    fig.patch.set_facecolor("white")
    for cluster_id,group in claims.groupby("cluster_id"):
        meta=clusters[clusters.cluster_id==cluster_id].iloc[0]
        ax.scatter(group.x,group.y,s=10+meta.prevalence*110,c=meta.color,alpha=.72,edgecolors="none")
    for theme,group in claims.groupby("macro_theme"):
        if theme=="Unassigned":
            continue
        points=group[["x","y"]].to_numpy()
        if len(points)>=3:
            hull=ConvexHull(points); polygon=points[hull.vertices]
            ax.plot(*np.vstack([polygon,polygon[0]]).T,color="#23314d",lw=1.2,alpha=.7)
            ax.text(points[:,0].mean(),points[:,1].max()+.3,theme,ha="center",fontsize=9,bbox=dict(fc="white",ec="none",alpha=.8))
    ax.set(xticks=[],yticks=[]); ax.spines[:].set_visible(False); ax.set_title("Atomic claim clusters and macro-themes",loc="left",weight="bold")
    legend.axis("off"); legend.set_title("Representative claims",loc="left",fontweight="bold")
    for index,row in clusters.sort_values("prevalence",ascending=False).head(7).reset_index(drop=True).iterrows():
        legend.scatter(.03,.88-index*.12,s=75,c=row.color,transform=legend.transAxes)
        legend.text(.1,.88-index*.12,f"{index+1}. {row.medoid}\n   clients={row.unique_clients} · lift={row.lift:+.2f}",va="center",fontsize=8,transform=legend.transAxes)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ("svg","pdf","png"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"),dpi=300,bbox_inches="tight")
    plt.close(fig)


def _plotly_source():
    candidates=list(Path("/home/chaichuk/miniconda3/pkgs").glob("plotly-*/lib/python*/site-packages/plotly/package_data/plotly.min.js"))
    if not candidates:
        raise FileNotFoundError("Local plotly.min.js not found; install Plotly or keep the Conda package cache")
    return sorted(candidates)[-1].read_text(encoding="utf-8")


REPORT_TEMPLATE=r'''<!doctype html><html><head><meta charset="utf-8"><title>{{ run_id }} analysis</title><style>
:root{--navy:#17253f;--blue:#3867d6;--teal:#16a085;--bg:#f2f5fa}*{box-sizing:border-box}body{margin:0;font:14px system-ui;color:var(--navy);background:var(--bg)}header{padding:28px 36px;color:white;background:linear-gradient(125deg,var(--navy),var(--blue));display:flex;justify-content:space-between}header h1{margin:0 0 6px}.badge{padding:7px 11px;border-radius:99px;background:#ffffff22}nav{display:flex;gap:6px;overflow:auto;padding:12px 22px;background:white;position:sticky;top:0;z-index:5;box-shadow:0 3px 14px #17253f12}nav button{border:0;background:#edf1f7;padding:9px 13px;border-radius:9px;white-space:nowrap;cursor:pointer}nav button.active{background:var(--blue);color:white}.tab{display:none;padding:24px;max-width:1450px;margin:auto}.tab.active{display:block}.card{background:white;padding:20px;border-radius:15px;box-shadow:0 7px 25px #17253f10;margin-bottom:18px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{background:white;border-radius:14px;padding:18px}.kpi b{font-size:24px;display:block}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}select,input{padding:8px;border:1px solid #d9e0eb;border-radius:8px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:9px;border-bottom:1px solid #e6eaf1}tbody tr{cursor:pointer}tbody tr:hover{background:#f1f5ff}pre{white-space:pre-wrap;background:#101a2d;color:#dce7ff;padding:16px;border-radius:10px}.warn{background:#fff2c9;padding:12px;border-radius:10px}.tree{display:flex;gap:24px;align-items:center;justify-content:center}.node{padding:12px;border-radius:11px;background:#e9efff;border:1px solid #bdcbef}.arrow{font-size:25px;color:var(--blue)}@media(max-width:900px){.grid,.kpis{grid-template-columns:1fr}}</style><script>{{ plotly_js|safe }}</script></head><body>
<header><div><h1>Reviewer revision analysis</h1><div>{{ run_id }} · {{ mode }}</div></div><div class="badge">self-contained · no server</div></header><nav>{% for tab in tabs %}<button data-tab="t{{ loop.index0 }}" class="{% if loop.first %}active{% endif %}">{{ tab }}</button>{% endfor %}</nav>
<section id="t0" class="tab active"><div class="kpis"><div class="kpi"><span>Datasets</span><b>3</b></div><div class="kpi"><span>Models</span><b>2</b></div><div class="kpi"><span>Clusters</span><b id="clusterCount"></b></div><div class="kpi"><span>Claims shown</span><b id="claimCount"></b></div></div><div class="card"><h2>Stage DAG</h2><div class="tree"><div class="node">Stats</div><div class="arrow">→</div><div class="node">Prompts / API</div><div class="arrow">→</div><div class="node">Claims / clusters</div><div class="arrow">→</div><div class="node">ML / analyses</div><div class="arrow">→</div><div class="node">Reports</div></div></div></section>
<section id="t1" class="tab"><div class="card"><h2>Model and experiment metrics</h2><div id="metricsPlot"></div></div></section>
<section id="t2" class="tab"><div class="card"><h2>Delta against article / legacy</h2><div id="deltaPlot"></div><p class="warn">Article GPT-OSS values used different splits. Deltas must be interpreted as split-update sensitivity, not a controlled model-only comparison.</p></div></section>
<section id="t3" class="tab"><div class="controls"><select id="dataset"><option value="">All datasets</option></select><select id="model"><option value="">All models</option></select><select id="seed"><option value="">All seeds</option></select><input id="search" placeholder="claim text search"></div><div class="grid"><div class="card"><div id="clusterMap"></div></div><div class="card"><h2>Selected cluster</h2><div id="clusterDetail">Click a point or table row.</div></div></div></section>
<section id="t4" class="tab"><div class="card"><h2>Cluster catalogue</h2><table><thead><tr><th>ID</th><th>Theme</th><th>Medoid</th><th>Clients</th><th>Compactness</th><th>Lift</th><th>Cross-model</th></tr></thead><tbody id="clusterRows"></tbody></table></div></section>
<section id="t5" class="tab"><div class="card"><h2>Cross-model matching</h2><div id="crossPlot"></div></div></section>
<section id="t6" class="tab"><div class="card"><h2>Stability</h2><div id="stabilityPlot"></div><p>Axes remain separate: generation seed, clustering seed, threshold, fixed K, and coverage.</p></div></section>
<section id="t7" class="tab"><div class="grid"><div class="card"><h2>Exact shallow-tree path</h2><div class="tree"><div class="node">cluster Mobility present?</div><div class="arrow">→</div><div class="node">Digital services absent?</div><div class="arrow">→</div><div class="node">leaf p=0.74 · n=128</div></div></div><div class="card"><h2>Feature links</h2><div id="importancePlot"></div></div></div></section>
<section id="t8" class="tab"><div class="card"><h2>Teacher-surrogate fidelity</h2><div id="fidelityPlot"></div><p>These are surrogate-fidelity measurements, not causal explanations.</p></div></section>
<section id="t9" class="tab"><div class="card"><h2>Claim grounding</h2><div id="groundingPlot"></div><p>Two-judge disagreements are not converted to a fabricated majority.</p></div></section>
<section id="t10" class="tab"><div class="card"><h2>Artifact provenance</h2><pre id="provenance"></pre></div></section>
<script>const DATA={{ data_json|safe }};document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active');window.dispatchEvent(new Event('resize'))});
const unique=k=>[...new Set(DATA.claims.map(x=>x[k]))].sort();for(const [id,key] of [['dataset','dataset'],['model','model'],['seed','seed']]){const s=document.getElementById(id);unique(key).forEach(v=>s.insertAdjacentHTML('beforeend',`<option>${v}</option>`))}document.getElementById('clusterCount').textContent=DATA.clusters.length;document.getElementById('claimCount').textContent=DATA.claims.length;
function clusterInfo(id){const c=DATA.clusters.find(x=>x.cluster_id===id);if(!c)return;document.getElementById('clusterDetail').innerHTML=`<h3>${c.cluster_id}</h3><p><b>Medoid:</b> ${c.medoid}</p><p>Theme: ${c.macro_theme}<br>Clients: ${c.unique_clients}<br>Compactness: ${c.compactness.toFixed(3)}<br>Prevalence: ${(c.prevalence*100).toFixed(1)}%<br>Lift: ${c.lift.toFixed(2)}<br>Matched: ${c.cross_model_match} (${c.match_cosine.toFixed(2)})<br>Tree nodes: ${c.tree_nodes.join(', ')}</p>`}
function drawMap(){const d=document.getElementById('dataset').value,m=document.getElementById('model').value,s=document.getElementById('seed').value,q=document.getElementById('search').value.toLowerCase();const rows=DATA.claims.filter(x=>(!d||x.dataset===d)&&(!m||x.model===m)&&(!s||String(x.seed)===s)&&(!q||x.text.toLowerCase().includes(q)));const traces=[...new Set(rows.map(x=>x.cluster_id))].map(id=>{const r=rows.filter(x=>x.cluster_id===id),c=DATA.clusters.find(x=>x.cluster_id===id);return{x:r.map(x=>x.x),y:r.map(x=>x.y),text:r.map(x=>x.text),customdata:r.map(x=>x.cluster_id),name:id,mode:'markers',type:'scattergl',marker:{size:6+80*c.prevalence,color:c.color,opacity:.72}}});Plotly.react('clusterMap',traces,{height:590,hovermode:'closest',xaxis:{visible:false},yaxis:{visible:false},legend:{orientation:'h'}});document.getElementById('clusterMap').on('plotly_click',e=>clusterInfo(e.points[0].customdata))}['dataset','model','seed','search'].forEach(id=>document.getElementById(id).oninput=drawMap);drawMap();
document.getElementById('clusterRows').innerHTML=DATA.clusters.map(c=>`<tr data-id="${c.cluster_id}"><td>${c.cluster_id}</td><td>${c.macro_theme}</td><td>${c.medoid}</td><td>${c.unique_clients}</td><td>${c.compactness.toFixed(3)}</td><td>${c.lift.toFixed(2)}</td><td>${c.cross_model_match}</td></tr>`).join('');document.querySelectorAll('#clusterRows tr').forEach(r=>r.onclick=()=>{clusterInfo(r.dataset.id);document.querySelector('[data-tab="t3"]').click()});
const metricNames=DATA.metrics.map(x=>`${x.dataset}/${x.model}/${x.experiment}`);Plotly.newPlot('metricsPlot',[{x:metricNames,y:DATA.metrics.map(x=>x.balanced_accuracy),error_y:{type:'data',array:DATA.metrics.map(x=>x.sd)},type:'bar',marker:{color:'#3867d6'}}],{height:500,yaxis:{title:'Balanced accuracy'}});Plotly.newPlot('deltaPlot',[{x:metricNames,y:DATA.metrics.map(x=>x.article_delta),type:'bar',marker:{color:DATA.metrics.map(x=>x.article_delta>=0?'#16a085':'#d9534f')}}],{height:500,yaxis:{title:'Absolute delta'}});
Plotly.newPlot('crossPlot',[{x:Object.keys(DATA.cross_model),y:Object.values(DATA.cross_model),type:'bar',marker:{color:'#8e6bbf'}}],{height:430});Plotly.newPlot('stabilityPlot',[{x:['ARI','NMI'],y:[DATA.stability.ari_mean,DATA.stability.nmi_mean],type:'bar',marker:{color:['#3867d6','#16a085']}}],{height:430,yaxis:{range:[0,1]}});Plotly.newPlot('importancePlot',[{x:DATA.clusters.slice(0,10).map(x=>x.importance),y:DATA.clusters.slice(0,10).map(x=>x.cluster_id),type:'bar',orientation:'h'}],{height:430});Plotly.newPlot('fidelityPlot',[{x:Object.keys(DATA.fidelity),y:Object.values(DATA.fidelity),type:'bar',marker:{color:'#f2b134'}}],{height:430});Plotly.newPlot('groundingPlot',[{labels:Object.keys(DATA.grounding),values:Object.values(DATA.grounding),type:'pie',hole:.45}],{height:430});document.getElementById('provenance').textContent=JSON.stringify(DATA.provenance,null,2);
</script></body></html>'''


def build_report(bundle,run_id):
    from jinja2 import Template
    return Template(REPORT_TEMPLATE).render(run_id=run_id,mode="SYNTHETIC DEMO" if bundle.get("synthetic") else "RESULTS / PLACEHOLDERS",tabs=TABS,plotly_js=_plotly_source(),data_json=json.dumps(bundle,ensure_ascii=False).replace("</","<\\/"))
