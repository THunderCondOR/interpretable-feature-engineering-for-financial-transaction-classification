from pathlib import Path
import pytest
from src.reporting.analysis_report import TABS, build_report, export_publication_map, load_bundle, portable_bundle, synthetic_bundle


def test_demo_report_is_self_contained_and_has_every_tab(tmp_path):
    bundle=synthetic_bundle(); document=build_report(bundle,"demo")
    for tab in TABS: assert tab in document
    assert "Plotly.newPlot" in document
    assert "https://cdn.plot.ly" not in document
    assert "No empirical result is represented" in document
    assert "clusterDetail').innerHTML" not in document
    assert "clusterRows').innerHTML" not in document
    export_publication_map(bundle,tmp_path/"map")
    for suffix in ("svg","pdf","png"): assert (tmp_path/f"map.{suffix}").exists()


def test_portable_sampling_preserves_full_exports_separately():
    bundle=synthetic_bundle(); sampled=portable_bundle(bundle,max_points=100)
    assert len(sampled["claims"])<=len(bundle["claims"])
    assert sampled["portable_sampling"]["total"]==len(bundle["claims"])


def test_strict_missing_results_refuses_to_build(tmp_path):
    with pytest.raises(FileNotFoundError): load_bundle(tmp_path,"missing",strict=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_bundle(tmp_path,"missing",strict=True,allow_missing=True)
    placeholder=load_bundle(tmp_path,"missing",allow_missing=True)
    assert "warning" in placeholder["provenance"]
