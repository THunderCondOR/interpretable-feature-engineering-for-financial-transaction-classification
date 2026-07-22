"""Stage dependency planning with signature-aware reuse."""
DEPENDENCIES = {
    "stats": (), "prompts": ("stats",), "explanations": ("prompts",),
    "direct_eval": ("explanations",), "claims": ("explanations",),
    "embeddings": ("claims",), "clusters": ("embeddings",), "features": ("clusters",),
    "ml": ("features",), "stability": ("ml",), "fidelity": ("ml",),
    "grounding": ("claims",), "reports": ("stability", "fidelity", "grounding"),
}


def plan_stages(expected_signatures, existing=None):
    existing, states = existing or {}, {}
    for stage, dependencies in DEPENDENCIES.items():
        artifact = existing.get(stage, {})
        compatible = artifact.get("state") == "completed" and artifact.get("signature") == expected_signatures.get(stage)
        if compatible:
            state, reason = "reused", "compatible completed artifact"
        elif any(states.get(dep, {}).get("state") in {"blocked", "incompatible"} for dep in dependencies):
            state, reason = "blocked", "upstream unavailable"
        elif artifact.get("state") == "completed":
            state, reason = "incompatible", "signature mismatch"
        elif all(states.get(dep, {}).get("state") in {"reused", "planned"} for dep in dependencies):
            state, reason = "planned", "dependencies available"
        else:
            state, reason = "blocked", "dependency incomplete"
        states[stage] = {"state": state, "reason": reason, "dependencies": list(dependencies), "signature": expected_signatures.get(stage)}
    return states
