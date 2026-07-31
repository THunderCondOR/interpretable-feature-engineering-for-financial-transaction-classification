#!/usr/bin/env python3
"""Select non-claim teachers and run seeded claim-space fidelity analyses."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import (
    atomic_write_json,
    files_fingerprint,
    fingerprint,
)
from src.experiments.derived_artifacts import (
    compatible_stage,
    complete_stage,
    stage_identity,
)
from src.models.ml_baseline import build_feature_sets, split_xy, xgb_objective
from src.utils.cluster import embed_texts


DATASETS = ("rosbank", "gender", "age")
MODELS = ("qwen", "gpt_oss")
CLAIM_SPACES = (*MODELS, "union", "semantic_union")
TEACHER_SEEDS = (17, 101, 947)
SURROGATE_SEEDS = (17, 101, 947)
SEED_PAIRS = (
    (17, 17), (17, 101), (17, 947),
    (101, 17), (947, 17),
)
NONCLAIM = ("standard", "handcrafted", "standard_profile", "all_nonclaim")
SOURCE_VARIANTS = {
    "rosbank": "guided_zero_shot_v4",
    "gender": "guided_zero_shot_v4",
    "age": "guided_zero_shot_v4__age_opaque",
}


def source_root(dataset: str, model: str) -> Path:
    return (
        Path("results/v2") / dataset / SOURCE_VARIANTS[dataset]
        / model / "seed_17"
    )


def derived_cell(root: Path, dataset: str, model: str) -> Path:
    return root / dataset / model / "seed_17"


def reusable_teacher_selection(
    path: Path,
    expected_identity: dict | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload["selected"]
        paths = selected["paths"]
        expected_hashes = selected["file_hashes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    if set(paths) != {"train", "val", "test"}:
        return False
    if files_fingerprint(paths.values()) != expected_hashes:
        return False
    if expected_identity is None:
        return True
    return compatible_stage(path.parent / "teacher_stage.json", expected_identity)


def load_config(dataset: str, cot_cell: Path) -> dict:
    manifest = json.loads(
        (source_root(dataset, "qwen") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config = copy.deepcopy(manifest["config"])
    config["output"]["base_dir"] = str(cot_cell)
    config.setdefault("input", {})["cot_features_base_dir"] = str(cot_cell)
    return config


def model_specs(config: dict):
    specs = []
    for size, parameters in (
        ("compact", {"n_estimators": 200, "max_depth": 4,
                     "learning_rate": 0.07}),
        ("large", {"n_estimators": 500, "max_depth": 6,
                   "learning_rate": 0.03}),
    ):
        for balanced in (False, True):
            specs.append({
                "backend": "xgboost", "size": size, "balanced": balanced,
                "factory": lambda seed, p=parameters: XGBClassifier(
                    **p, objective=xgb_objective(config), subsample=0.9,
                    colsample_bytree=0.9, random_state=seed, n_jobs=2,
                    tree_method="hist", eval_metric="logloss", verbosity=0,
                ),
            })
    unavailable = {}
    try:
        from lightgbm import LGBMClassifier
        for size, parameters in (
            ("compact", {"n_estimators": 200, "num_leaves": 15,
                         "learning_rate": 0.07}),
            ("large", {"n_estimators": 500, "num_leaves": 31,
                       "learning_rate": 0.03}),
        ):
            for balanced in (False, True):
                specs.append({
                    "backend": "lightgbm", "size": size,
                    "balanced": balanced,
                    "factory": lambda seed, p=parameters: LGBMClassifier(
                        **p, random_state=seed, n_jobs=2, verbosity=-1,
                    ),
                })
    except ImportError:
        unavailable["lightgbm"] = "package not installed"
    try:
        from catboost import CatBoostClassifier
        for size, parameters in (
            ("compact", {"iterations": 200, "depth": 4,
                         "learning_rate": 0.07}),
            ("large", {"iterations": 500, "depth": 6,
                       "learning_rate": 0.03}),
        ):
            for balanced in (False, True):
                specs.append({
                    "backend": "catboost", "size": size,
                    "balanced": balanced,
                    "factory": lambda seed, p=parameters: CatBoostClassifier(
                        **p, random_seed=seed, verbose=False,
                        allow_writing_files=False, thread_count=2,
                    ),
                })
    except ImportError:
        unavailable["catboost"] = "package not installed"
    return specs, unavailable


def fit_teacher_model(spec, model, values, labels):
    fit_kwargs = {}
    if spec["balanced"]:
        fit_kwargs["sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=labels
        )
    model.fit(values, labels, **fit_kwargs)
    return model


def aligned_probabilities(model, values: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(values)
    result = np.zeros((len(values), n_classes), dtype=float)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    return result


def teacher_stage_identity(
    dataset: str,
    config: dict,
) -> dict:
    dataset_config = config.get("dataset", {})
    input_paths = [
        *dataset_config.get("splits", {}).values(),
        *dataset_config.get("client_ids_by_split", {}).values(),
        Path(__file__),
        REPO_ROOT / "src/models/ml_baseline.py",
    ]
    input_paths = [
        path for path in input_paths if isinstance(path, (str, Path))
    ]
    return stage_identity(
        stage=f"fidelity_teacher_v2:{dataset}",
        source={
            "dataset": dataset,
            "source_manifest": str(source_root(dataset, "qwen") / "manifest.json"),
        },
        inputs=files_fingerprint(input_paths),
        configuration={
            "nonclaim_feature_sets": NONCLAIM,
            "teacher_seeds": TEACHER_SEEDS,
            "model_grid_version": 2,
            "dataset_config": dataset_config,
        },
        repo_root=REPO_ROOT,
    )


def train_teacher(
    dataset: str,
    rebuild_root: Path,
    output_dir: Path,
) -> Path:
    config = load_config(dataset, derived_cell(
        rebuild_root, dataset, "qwen"
    ))
    identity = teacher_stage_identity(dataset, config)
    selection_path = output_dir / "teacher_selection.json"
    if reusable_teacher_selection(selection_path, identity):
        return selection_path
    packs = build_feature_sets(config, list(NONCLAIM))
    specs, unavailable = model_specs(config)
    n_classes = int(config["dataset"]["num_labels"])
    candidates = []
    for feature_set, pack in packs.items():
        columns = pack["columns"]
        x_train, y_train = split_xy(pack["train"], columns)
        x_val, y_val = split_xy(pack["val"], columns)
        for spec_index, spec in enumerate(specs):
            model = fit_teacher_model(
                spec, spec["factory"](17), x_train, y_train
            )
            probabilities = aligned_probabilities(model, x_val, n_classes)
            candidates.append({
                "feature_set": feature_set,
                "spec_index": spec_index,
                "backend": spec["backend"],
                "size": spec["size"],
                "balanced": spec["balanced"],
                "balanced_accuracy": float(balanced_accuracy_score(
                    y_val, probabilities.argmax(axis=1)
                )),
                "log_loss": float(log_loss(
                    y_val, probabilities, labels=np.arange(n_classes)
                )),
            })
    selected = sorted(
        candidates,
        key=lambda row: (
            -row["balanced_accuracy"], row["log_loss"],
            row["feature_set"], row["backend"],
        ),
    )[0]
    pack = packs[selected["feature_set"]]
    selected_spec = specs[int(selected["spec_index"])]
    columns = pack["columns"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: output_dir / f"teacher_predictions_{split}.csv"
        for split in ("train", "val", "test")
    }
    records = {split: [] for split in paths}
    x_train, y_train = split_xy(pack["train"], columns)
    for seed in TEACHER_SEEDS:
        model = fit_teacher_model(
            selected_spec, selected_spec["factory"](seed), x_train, y_train
        )
        for split, frame in (
            ("train", pack["train"]),
            ("val", pack["val"]),
            ("test", pack["test"]),
        ):
            values, _ = split_xy(frame, columns)
            probabilities = aligned_probabilities(model, values, n_classes)
            for index, row in frame.reset_index(drop=True).iterrows():
                records[split].append({
                    "seed": seed,
                    "customer_id": row.customer_id,
                    "label": int(row.label),
                    **{
                        f"teacher_prob_{label}": float(
                            probabilities[index, label]
                        )
                        for label in range(n_classes)
                    },
                })
    for split, path in paths.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        pd.DataFrame(records[split]).to_csv(temporary, index=False)
        temporary.replace(path)
    payload = {
        "selection_split": "validation",
        "primary_metric": "balanced_accuracy",
        "selected_teacher": (
            f"{selected['feature_set']}:{selected['backend']}"
        ),
        "selected_candidate": selected,
        "candidate_metrics": candidates,
        "unavailable_backends": unavailable,
        "selected": {
            "paths": {key: str(value) for key, value in paths.items()},
            "filters": {},
            "file_hashes": files_fingerprint(paths.values()),
        },
    }
    atomic_write_json(selection_path, payload)
    complete_stage(
        output_dir / "teacher_stage.json",
        identity,
        outputs=[selection_path, *paths.values()],
        metrics={
            "selected_teacher": payload["selected_teacher"],
            "selected_candidate": selected,
        },
    )
    return selection_path


def union_features(
    left: Path,
    right: Path,
    output: Path,
) -> None:
    first = pd.read_parquet(left)
    second = pd.read_parquet(right)
    keys = ["customer_id", "label"]
    if first[keys].duplicated().any() or second[keys].duplicated().any():
        raise ValueError("Duplicate customer/label rows in union claim space")
    first = first.rename(columns={
        column: f"cot_qwen__{column.removeprefix('cot_')}"
        for column in first if column.startswith("cot_")
    })
    second = second.rename(columns={
        column: f"cot_gpt__{column.removeprefix('cot_')}"
        for column in second if column.startswith("cot_")
    })
    merged = first.merge(second, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("Qwen/GPT claim spaces have different client coverage")
    # A common binary presence contract avoids mixing binary and normalized
    # count geometries in a single surrogate and makes occlusion well-defined.
    feature_columns = [
        column for column in merged if column.startswith("cot_")
    ]
    merged[feature_columns] = (
        merged[feature_columns].to_numpy(dtype=float) > 0
    ).astype(np.float32)
    temporary = output.with_suffix(output.suffix + ".tmp")
    merged.to_parquet(temporary, index=False)
    temporary.replace(output)


def semantic_union_mapping(
    qwen_cell: Path,
    gpt_cell: Path,
    *,
    embedding_model: str = "intfloat/multilingual-e5-large",
    minimum_similarity: float = 0.50,
    embedder=embed_texts,
) -> dict:
    """Match Qwen/GPT medoids in one common centered E5 coordinate space.

    Cluster centroids cannot be compared when model-specific PCA transforms
    were fitted independently.  Re-embedding all medoid texts together keeps
    both sets in exactly the same semantic coordinate system.
    """
    qwen_meta = json.loads(
        (qwen_cell / "selected_clusters/cluster_model.json").read_text(
            encoding="utf-8"
        )
    )
    gpt_meta = json.loads(
        (gpt_cell / "selected_clusters/cluster_model.json").read_text(
            encoding="utf-8"
        )
    )
    qwen_names = list(qwen_meta["selected_feature_names"])
    gpt_names = list(gpt_meta["selected_feature_names"])
    qwen_medoids = {
        row["feature"]: row.get("medoid", row["feature"])
        for row in qwen_meta.get("cluster_meta", [])
    }
    gpt_medoids = {
        row["feature"]: row.get("medoid", row["feature"])
        for row in gpt_meta.get("cluster_meta", [])
    }
    missing = [
        f"qwen:{name}" for name in qwen_names if name not in qwen_medoids
    ] + [
        f"gpt_oss:{name}" for name in gpt_names if name not in gpt_medoids
    ]
    if missing:
        raise ValueError(
            "Selected clusters are missing semantic medoid text: "
            + ", ".join(missing[:10])
        )
    texts = (
        [qwen_medoids[name] for name in qwen_names]
        + [gpt_medoids[name] for name in gpt_names]
    )
    common = np.asarray(
        embedder(texts, model_name=embedding_model),
        dtype=np.float32,
    )
    if common.ndim != 2 or len(common) != len(texts):
        raise ValueError("Cross-model embedder returned an invalid matrix")
    common -= common.mean(axis=0, keepdims=True)
    common /= np.maximum(
        np.linalg.norm(common, axis=1, keepdims=True), 1e-12
    )
    q = common[:len(qwen_names)]
    g = common[len(qwen_names):]
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    similarities = q @ g.T
    q_nearest = similarities.argmax(axis=1)
    g_nearest = similarities.argmax(axis=0)
    pairs = []
    for q_index, g_index in enumerate(q_nearest):
        if int(g_nearest[int(g_index)]) != q_index:
            continue
        similarity = float(similarities[q_index, int(g_index)])
        if similarity < float(minimum_similarity):
            continue
        pairs.append({
            "qwen_feature": qwen_names[q_index],
            "gpt_feature": gpt_names[int(g_index)],
            "qwen_medoid": qwen_medoids.get(
                qwen_names[q_index], qwen_names[q_index]
            ),
            "gpt_medoid": gpt_medoids.get(
                gpt_names[int(g_index)], gpt_names[int(g_index)]
            ),
            "centered_cosine_similarity": similarity,
        })
    paired_qwen = {row["qwen_feature"] for row in pairs}
    paired_gpt = {row["gpt_feature"] for row in pairs}
    return {
        "method": "mutual_nearest_common_centered_e5_medoids",
        "embedding_model": embedding_model,
        "minimum_similarity": float(minimum_similarity),
        "pairs": pairs,
        "unmatched_qwen": [
            {
                "feature": name,
                "medoid": qwen_medoids.get(name, name),
            }
            for name in qwen_names if name not in paired_qwen
        ],
        "unmatched_gpt": [
            {
                "feature": name,
                "medoid": gpt_medoids.get(name, name),
            }
            for name in gpt_names if name not in paired_gpt
        ],
    }


def semantic_union_features(
    left: Path,
    right: Path,
    output: Path,
    mapping: dict,
) -> list[dict]:
    first = pd.read_parquet(left)
    second = pd.read_parquet(right)
    keys = ["customer_id", "label"]
    qwen_columns = {
        column: f"qwen__{column}"
        for column in first if column.startswith("cot_")
    }
    gpt_columns = {
        column: f"gpt__{column}"
        for column in second if column.startswith("cot_")
    }
    first = first.rename(columns=qwen_columns)
    second = second.rename(columns=gpt_columns)
    merged = first.merge(
        second, on=keys, how="inner", validate="one_to_one",
    )
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("Qwen/GPT claim spaces have different client coverage")
    result = merged[keys].copy()
    cluster_meta = []
    for index, pair in enumerate(mapping["pairs"]):
        name = f"cot_shared_{index:04d}"
        result[name] = (
            (merged[qwen_columns[pair["qwen_feature"]]].to_numpy(float) > 0)
            | (merged[gpt_columns[pair["gpt_feature"]]].to_numpy(float) > 0)
        ).astype(np.float32)
        cluster_meta.append({
            "feature": name,
            "medoid": (
                f"{pair['qwen_medoid']} / {pair['gpt_medoid']}"
            ),
            **pair,
        })
    for prefix, entries in (
        ("qwen_only", mapping["unmatched_qwen"]),
        ("gpt_only", mapping["unmatched_gpt"]),
    ):
        for index, entry in enumerate(entries):
            source_name = entry["feature"]
            name = f"cot_{prefix}_{index:04d}"
            renamed = (
                qwen_columns[source_name]
                if prefix == "qwen_only"
                else gpt_columns[source_name]
            )
            result[name] = (
                merged[renamed].to_numpy(float) > 0
            ).astype(np.float32)
            cluster_meta.append({
                "feature": name,
                "medoid": entry["medoid"],
                "source_feature": source_name,
            })
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(output)
    return cluster_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-root", type=Path,
        default=Path("results/v2/derived/reviewer-v6-e5-clustering"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/v2/derived/fidelity-v2"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "datasets": DATASETS,
        "teacher_features": NONCLAIM,
        "teacher_backends": ["xgboost", "lightgbm", "catboost-if-installed"],
        "claim_spaces": CLAIM_SPACES,
        "teacher_seeds": TEACHER_SEEDS,
        "surrogate_seeds": SURROGATE_SEEDS,
        "seed_pairs": SEED_PAIRS,
        "rebuild_root": str(args.rebuild_root),
        "output_root": str(args.output_root),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    state_path = args.output_root / "fidelity_state.json"
    prior_completed = []
    if state_path.is_file():
        prior_completed = json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("completed", [])
    completed = set(prior_completed)
    state = {**plan, "state": "running", "completed": sorted(completed)}
    atomic_write_json(state_path, state)
    for dataset in DATASETS:
        for model in MODELS:
            cell = derived_cell(args.rebuild_root, dataset, model)
            for split in ("train", "val", "test"):
                path = cell / f"cot_features_{split}.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
        dataset_output = args.output_root / dataset
        teacher_output = dataset_output / "teacher"
        selection = train_teacher(
            dataset, args.rebuild_root, teacher_output
        )
        selection_payload = json.loads(selection.read_text(encoding="utf-8"))
        qwen_cell = derived_cell(args.rebuild_root, dataset, "qwen")
        gpt_cell = derived_cell(args.rebuild_root, dataset, "gpt_oss")
        semantic_mapping = semantic_union_mapping(qwen_cell, gpt_cell)
        semantic_dir = dataset_output / "semantic_union_features"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        semantic_mapping_path = semantic_dir / "mapping.json"
        atomic_write_json(semantic_mapping_path, semantic_mapping)
        semantic_metadata_path = semantic_dir / "cluster_metadata.json"
        for teacher_seed, surrogate_seed in SEED_PAIRS:
            seeded = copy.deepcopy(selection_payload)
            seeded["selected"]["filters"] = {"seed": teacher_seed}
            seeded_path = (
                dataset_output / "teacher"
                / f"selection_teacher_seed_{teacher_seed}.json"
            )
            atomic_write_json(seeded_path, seeded)
            for space in CLAIM_SPACES:
                key = (
                    f"{dataset}:{space}:teacher_{teacher_seed}:"
                    f"surrogate_{surrogate_seed}"
                )
                cell_output = (
                    dataset_output / space
                    / f"teacher_seed_{teacher_seed}"
                    / f"surrogate_seed_{surrogate_seed}"
                )
                required = (
                    cell_output / "fidelity_metrics.json",
                    cell_output / "surrogate_selection.json",
                    cell_output / "surrogate_predictions.csv",
                    cell_output / "cluster_occlusion.jsonl",
                    cell_output / "tree_decision_paths.jsonl",
                    cell_output / "fidelity_stage.json",
                )
                feature_paths = {}
                for split in ("train", "val", "test"):
                    if space == "union":
                        union_dir = dataset_output / "union_features"
                        union_dir.mkdir(parents=True, exist_ok=True)
                        feature_path = union_dir / f"cot_features_{split}.parquet"
                        union_features(
                            derived_cell(
                                args.rebuild_root, dataset, "qwen"
                            ) / f"cot_features_{split}.parquet",
                            derived_cell(
                                args.rebuild_root, dataset, "gpt_oss"
                            ) / f"cot_features_{split}.parquet",
                            feature_path,
                        )
                    elif space == "semantic_union":
                        feature_path = (
                            semantic_dir / f"cot_features_{split}.parquet"
                        )
                        cluster_meta = semantic_union_features(
                            qwen_cell / f"cot_features_{split}.parquet",
                            gpt_cell / f"cot_features_{split}.parquet",
                            feature_path,
                            semantic_mapping,
                        )
                        atomic_write_json(
                            semantic_metadata_path,
                            {"cluster_meta": cluster_meta},
                        )
                    else:
                        feature_path = derived_cell(
                            args.rebuild_root, dataset, space
                        ) / f"cot_features_{split}.parquet"
                    feature_paths[split] = feature_path
                command = [
                    sys.executable, "scripts/run_fidelity_analysis.py",
                    "--train-features", str(feature_paths["train"]),
                    "--val-features", str(feature_paths["val"]),
                    "--test-features", str(feature_paths["test"]),
                    "--teacher-selection", str(seeded_path),
                    "--output-dir", str(
                        cell_output
                    ),
                    "--teacher-seed", str(teacher_seed),
                    "--surrogate-seed", str(surrogate_seed),
                    "--encoding", (
                        "binary" if space in {
                            "union", "semantic_union"
                        } else str(
                            json.loads(
                                (
                                    derived_cell(
                                        args.rebuild_root, dataset, space
                                    )
                                    / "selected_clusters/cluster_model.json"
                                ).read_text(encoding="utf-8")
                            ).get("representation", {}).get(
                                "encoding", "binary"
                            )
                        )
                    ),
                    "--execute",
                ]
                if space not in {"union"}:
                    command.extend([
                        "--cluster-metadata",
                        str(
                            semantic_metadata_path
                            if space == "semantic_union"
                            else
                            derived_cell(
                                args.rebuild_root, dataset, space
                            ) / "selected_clusters/cluster_model.json"
                        ),
                    ])
                state["state"] = "running"
                state["current"] = key
                state.pop("error", None)
                atomic_write_json(state_path, state)
                try:
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                except Exception as error:
                    state["state"] = "failed"
                    state["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    atomic_write_json(state_path, state)
                    raise
                if not all(path.is_file() for path in required):
                    error = FileNotFoundError(
                        f"Incomplete fidelity cell {key}: {required}"
                    )
                    state["state"] = "failed"
                    state["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    atomic_write_json(state_path, state)
                    raise error
                completed.add(key)
                state["completed"] = sorted(completed)
                state["current"] = key
                atomic_write_json(state_path, state)
    state["state"] = "completed"
    state["current"] = None
    state.pop("error", None)
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()
