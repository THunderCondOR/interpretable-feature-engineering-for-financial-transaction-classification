"""Build a self-contained cluster/classifier analysis report."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting.analysis_report import build_report, export_publication_map, load_bundle, portable_bundle, synthetic_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",type=Path,default=Path("results/v2"))
    parser.add_argument("--run-id",default="reviewer-v2")
    parser.add_argument("--output",type=Path,default=Path("reports/reviewer-v2/index.html"))
    parser.add_argument("--demo",action="store_true")
    mode=parser.add_mutually_exclusive_group(); mode.add_argument("--portable",action="store_true"); mode.add_argument("--full",action="store_true")
    validation=parser.add_mutually_exclusive_group()
    validation.add_argument("--allow-missing",action="store_true")
    validation.add_argument("--strict",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"mode":"dry-run","run_id":args.run_id,"source":"synthetic" if args.demo else str(args.results_root),"output":str(args.output),"report_mode":"full" if args.full else "portable","allow_missing":args.allow_missing,"strict":args.strict},indent=2))
        return
    bundle=synthetic_bundle() if args.demo else load_bundle(args.results_root,args.run_id,allow_missing=args.allow_missing,strict=args.strict)
    display=bundle if args.full else portable_bundle(bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp"); temporary.write_text(build_report(display,args.run_id),encoding="utf-8"); temporary.replace(args.output)
    pd.DataFrame(bundle["claims"]).to_csv(args.output.parent/"claims.csv",index=False)
    pd.DataFrame(bundle["clusters"]).to_csv(args.output.parent/"clusters.csv",index=False)
    pd.DataFrame(bundle["metrics"]).to_csv(args.output.parent/"metrics.csv",index=False)
    (args.output.parent/"analysis_bundle.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding="utf-8")
    export_publication_map(bundle,args.output.parent/"publication_cluster_map")
    print(f"Built self-contained report -> {args.output}")


if __name__=="__main__":
    main()
