from src.experiments.dag import DEPENDENCIES, plan_stages


def test_dag_reuses_only_compatible_completed_stages():
    signatures = {stage: f"new-{stage}" for stage in DEPENDENCIES}
    existing = {"stats": {"state": "completed", "signature": "new-stats"}, "prompts": {"state": "completed", "signature": "old-prompts"}}
    states = plan_stages(signatures, existing)
    assert states["stats"]["state"] == "reused"
    assert states["prompts"]["state"] == "incompatible"
    assert states["explanations"]["state"] == "blocked"


def test_empty_run_plans_full_dependency_chain():
    states = plan_stages({stage: stage for stage in DEPENDENCIES})
    assert all(value["state"] == "planned" for value in states.values())
    assert states["reports"]["dependencies"] == ["stability", "fidelity", "grounding"]
