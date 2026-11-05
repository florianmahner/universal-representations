#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.metric.compute.run import compute_universality_for_model
from experiments.universality.shared import METRICS, load_components_for_rank

N_NULL_PERMS = 1000
N_REFERENCE_MODELS = 162

logger = logging.getLogger(__name__)


def metric_names() -> set[str]:
    return set(METRICS) | {"cos2_perm_adj"}


def sigma_token(value: object) -> str:
    return f"{float(value):.1f}"


def part_path(out: Path, model: str) -> Path:
    return out / "universality" / "by_model" / f"{model}.csv"


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def reference_sha256(reference: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for model in sorted(reference):
        value = np.ascontiguousarray(reference[model])
        digest.update(model.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def load_target_w(
    model: str,
    selection: pd.DataFrame,
    embeddings: Path,
    rank: int,
    n_items: int,
) -> tuple[np.ndarray, str]:
    row = selection.loc[model]
    path = (
        embeddings
        / model
        / f"rank_{rank}"
        / f"sigma_{sigma_token(row['sigma_mult'])}"
        / f"seed_{int(row['best_seed'])}"
        / "w.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    w = np.load(path)["w"]
    if w.shape != (n_items, rank):
        raise ValueError(f"{model}: expected W {(n_items, rank)}, got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError(f"{model}: non-finite selected W")
    if float(w.min()) < -1e-10:
        raise ValueError(f"{model}: selected W contains negative values")
    return w, array_sha256(w)


def part_is_current(
    part: pd.DataFrame,
    selection_row: pd.Series | dict[str, object],
    rank: int,
    null_seed: int,
    n_reference_models: int,
    target_w_sha256: str,
    reference_fingerprint: str,
) -> bool:
    """Whether a cached per-model part matches the run it would be reused for."""
    required = {
        "metric",
        "dimension",
        "universality",
        "n_null_perms",
        "null_seed",
        "n_reference_models",
        "sigma_mult",
        "best_seed",
        "target_w_sha256",
        "reference_sha256",
    }
    return required.issubset(part.columns) and (
        len(part) == rank * len(metric_names())
        and int(part["n_null_perms"].iloc[0]) == N_NULL_PERMS
        and int(part["null_seed"].iloc[0]) == null_seed
        and int(part["n_reference_models"].iloc[0]) == n_reference_models
        and np.isclose(float(part["sigma_mult"].iloc[0]), float(selection_row["sigma_mult"]))
        and int(part["best_seed"].iloc[0]) == int(selection_row["best_seed"])
        and part["target_w_sha256"].iloc[0] == target_w_sha256
        and part["reference_sha256"].iloc[0] == reference_fingerprint
    )


def score_one(
    model: str,
    w_target: np.ndarray,
    reference: dict[str, np.ndarray],
    selection_row: dict[str, object],
    training_dataset: str,
    dataset: str,
    rank: int,
    seed: int,
    reference_fingerprint: str,
    out: Path,
) -> pd.DataFrame:
    path = part_path(out, model)
    if path.exists():
        cached = pd.read_csv(path)
        if (
            part_is_current(
                cached,
                selection_row,
                rank=rank,
                null_seed=seed,
                n_reference_models=len(reference),
                target_w_sha256=selection_row["target_w_sha256"],
                reference_fingerprint=reference_fingerprint,
            )
            and set(cached["metric"]) == metric_names()
            and np.isfinite(cached["universality"]).all()
            and cached["universality"].between(0, 1).all()
            and cached.groupby("metric")["dimension"].nunique().eq(rank).all()
        ):
            logger.info("%s: cached scores", model)
            return cached

    rows = compute_universality_for_model(
        model,
        {**reference, model: w_target},
        n_null_perms=N_NULL_PERMS,
        null_seed=seed,
    )
    frame = pd.DataFrame(rows)
    frame["training_dataset"] = training_dataset
    frame["architecture"] = "AlexNet-GN"
    frame["objective"] = "IPCL"
    frame["rank"] = rank
    frame["n_reference_models"] = len(reference)
    frame["reference_dataset"] = dataset
    frame["n_null_perms"] = N_NULL_PERMS
    frame["null_seed"] = seed
    frame["target_w_sha256"] = selection_row["target_w_sha256"]
    frame["reference_sha256"] = reference_fingerprint
    for column in ("sigma_mult", "best_seed", "stability", "evar", "harmonic"):
        frame[column] = selection_row[column]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "universality" / "by_model").mkdir(parents=True, exist_ok=True)
    rank = cfg.factorization.rank

    # The four IPCL checkpoints share one architecture and objective and
    # differ only in the dataset they were trained on.
    training_datasets = {
        "ipcl_alexnet_gn_imagenet1k": "ImageNet-1K",
        "ipcl_alexnet_gn_openimagesv6": "OpenImages",
        "ipcl_alexnet_gn_places2": "Places-256",
        "ipcl_alexnet_gn_vggface2": "VGGFace2",
    }
    selected = list(cfg.models) or list(training_datasets)
    unknown = set(selected).difference(training_datasets)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")
    models = [model for model in training_datasets if model in selected]

    selection_path = out / "meta" / f"bandwidth_selection_rank_{rank}.csv"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path}; run factorize first")
    selection = pd.read_csv(selection_path)
    missing = set(training_datasets).difference(selection["model"])
    if missing:
        raise ValueError(f"selection missing IPCL models: {sorted(missing)}")
    selection = selection.set_index("model")

    reference = load_components_for_rank(rank=rank, dataset=cfg.dataset)
    if len(reference) != N_REFERENCE_MODELS:
        raise ValueError(f"expected {N_REFERENCE_MODELS} reference models, got {len(reference)}")
    overlap = set(reference).intersection(models)
    if overlap:
        raise ValueError(f"target leakage into reference panel: {sorted(overlap)}")
    reference_rows = {int(w.shape[0]) for w in reference.values()}
    if len(reference_rows) != 1:
        raise ValueError(f"reference row counts disagree: {sorted(reference_rows)}")
    n_items = reference_rows.pop()
    for model, w in reference.items():
        if w.shape != (n_items, rank):
            raise ValueError(f"reference {model}: expected {(n_items, rank)}, got {w.shape}")
        if not np.isfinite(w).all():
            raise ValueError(f"reference {model}: non-finite W")
        if float(w.min()) < -1e-10:
            raise ValueError(f"reference {model}: negative W")
    logger.info("hashing fixed reference panel for stale-cache protection")
    reference_fingerprint = reference_sha256(reference)

    # Load and fingerprint every planned target, not just the selected ones, so
    # the aggregate below can keep parts scored by an earlier partial run.
    target_ws = {
        model: load_target_w(model, selection, out / "embeddings", rank, n_items)
        for model in training_datasets
    }
    run_config = {
        "models": models,
        "rank": rank,
        "metric": "cos2_perm_adj",
        "reference_dataset": cfg.dataset,
        "n_reference_models": len(reference),
        "n_null_perms": N_NULL_PERMS,
        "null_calibration": "pair_specific",
        "seed": cfg.seed,
        "n_jobs": cfg.n_jobs,
        "skip_existing": True,
        "reference_sha256": reference_fingerprint,
        "target_w_sha256": {model: sha for model, (_, sha) in target_ws.items()},
    }
    (out / "meta" / "scoring_run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    order = list(training_datasets)
    frames = Parallel(n_jobs=min(cfg.n_jobs, len(models)), verbose=10)(
        delayed(score_one)(
            model,
            target_ws[model][0],
            reference,
            {**selection.loc[model].to_dict(), "target_w_sha256": target_ws[model][1]},
            training_datasets[model],
            cfg.dataset,
            rank,
            cfg.seed + order.index(model),
            reference_fingerprint,
            out,
        )
        for model in models
    )

    # Rebuild the aggregates from every valid completed part so that models
    # scored by an earlier run survive a run over a subset of the targets.
    parts = []
    for index, model in enumerate(order):
        path = part_path(out, model)
        if not path.exists():
            continue
        part = pd.read_csv(path)
        if part_is_current(
            part,
            selection.loc[model],
            rank=rank,
            null_seed=cfg.seed + index,
            n_reference_models=len(reference),
            target_w_sha256=target_ws[model][1],
            reference_fingerprint=reference_fingerprint,
        ):
            parts.append(part)
        else:
            logger.warning("%s: excluded stale/incomplete part from aggregate", path)
    combined = pd.concat(parts or frames, ignore_index=True)
    combined.to_csv(out / "universality" / "per_dimension_all_metrics.csv", index=False)
    primary = combined[combined["metric"] == "cos2_perm_adj"].copy()
    primary.to_csv(out / "per_dimension.csv", index=False)
    logger.info(
        "wrote %s (%d primary rows; %d references per target)",
        out / "per_dimension.csv",
        len(primary),
        len(reference),
    )


if __name__ == "__main__":
    main()
