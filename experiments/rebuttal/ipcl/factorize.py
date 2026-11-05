#!/usr/bin/env python3
"""Factorize the IPCL checkpoints over the bandwidth x seed grid and select sigma.

Each checkpoint gets its own median-distance bandwidth base, the full multiplier
grid, and every seed; the bandwidth is then chosen by the harmonic mean of
cross-seed stability and explained variance, exactly as in the paper's pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.rebuttal.universality_across_layers.factorization.factorize import rbf_frugal
from src.metrics import explained_variance, median_sigma_heuristic
from src.stability import factorization_stability, reference_seed_index

logger = logging.getLogger(__name__)


def seed_dir(out: Path, model: str, rank: int, sigma_mult: str, seed: int) -> Path:
    return (
        out
        / "embeddings"
        / model
        / f"rank_{rank}"
        / f"sigma_{sigma_mult}"
        / f"seed_{seed}"
    )


def final_history_value(estimator: object, key: str) -> float:
    history = getattr(estimator, "history_", None)
    if isinstance(history, dict) and key in history and history[key]:
        return float(history[key][-1])
    return float("nan")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def compatible_cached_metrics(
    metrics: dict[str, object],
    *,
    model: str,
    rank: int,
    median_sigma_value: float,
    sigma_mult: str,
    sigma: float,
    seed: int,
    max_outer: int,
    max_inner: int,
    tol: float,
    feature_sha256: str,
) -> bool:
    expected = {
        "model": model,
        "rank": rank,
        "median_sigma": median_sigma_value,
        "sigma_mult": sigma_mult,
        "sigma": sigma,
        "seed": seed,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "tol": tol,
        "feature_sha256": feature_sha256,
    }
    for key, value in expected.items():
        cached = metrics.get(key)
        if cached is None:
            return False
        if key == "sigma_mult":
            if str(cached) != str(value):
                return False
        elif isinstance(value, float):
            if not np.isclose(float(cached), value):
                return False
        elif cached != value:
            return False
    return True


def sigma_bases(
    models: list[str],
    datasets: dict[str, str],
    feature_paths: dict[str, Path],
    feature_hashes: dict[str, str],
    out: Path,
) -> dict[str, float]:
    """Per-model bandwidth base: the exact median pairwise distance over all images."""
    bases: dict[str, float] = {}
    rows = []
    for model in models:
        features = np.asarray(np.load(feature_paths[model]), dtype=np.float64)
        sigma0 = float(median_sigma_heuristic(features))
        bases[model] = sigma0
        rows.append(
            {
                "model": model,
                "training_dataset": datasets[model],
                "median_sigma": sigma0,
                "n_samples": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "feature_path": str(feature_paths[model]),
                "feature_sha256": feature_hashes[model],
            }
        )
    pd.DataFrame(rows).to_csv(out / "meta" / "median_sigma.csv", index=False)
    return bases


def fit_model_sigma(
    model: str,
    dataset: str,
    feature_path: Path,
    sigma_mult: str,
    sigma0: float,
    out: Path,
    *,
    rank: int,
    seeds: tuple[int, ...],
    max_outer: int,
    max_inner: int,
    tol: float,
    feature_sha256: str,
) -> list[dict[str, object]]:
    paths = {
        seed: (
            seed_dir(out, model, rank, sigma_mult, seed) / "w.npz",
            seed_dir(out, model, rank, sigma_mult, seed) / "metrics.json",
        )
        for seed in seeds
    }
    cached_rows: dict[int, dict[str, object]] = {}
    sigma = sigma0 * float(sigma_mult)
    for seed, (w_path, metrics_path) in paths.items():
        if w_path.exists() and metrics_path.exists():
            metrics = read_json(metrics_path)
            if not compatible_cached_metrics(
                metrics,
                model=model,
                rank=rank,
                median_sigma_value=sigma0,
                sigma_mult=sigma_mult,
                sigma=sigma,
                seed=seed,
                max_outer=max_outer,
                max_inner=max_inner,
                tol=tol,
                feature_sha256=feature_sha256,
            ):
                raise ValueError(
                    f"incompatible cached fit at {metrics_path}; delete it to refit"
                )
            cached_rows[seed] = metrics
    if len(cached_rows) == len(seeds):
        logger.info("%s sigma=%s: all %d seeds cached", model, sigma_mult, len(seeds))
        return [cached_rows[seed] for seed in seeds]

    from src.factorization import get_snmf_class

    SNMF = get_snmf_class()

    features = np.asarray(np.load(feature_path), dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError(f"{model}: non-finite features")
    logger.info(
        "%s sigma=%s: building %d x %d RBF kernel",
        model,
        sigma_mult,
        len(features),
        len(features),
    )
    kernel_start = time.perf_counter()
    similarity = rbf_frugal(features, sigma=sigma)
    kernel_time = time.perf_counter() - kernel_start
    rows: list[dict[str, object]] = []

    for seed in seeds:
        if seed in cached_rows:
            rows.append(cached_rows[seed])
            continue
        w_path, metrics_path = paths[seed]
        w_path.parent.mkdir(parents=True, exist_ok=True)

        # An orphan W has no trustworthy record of its feature/kernel identity,
        # so it is refit rather than silently adopted.
        logger.info("%s sigma=%s seed=%d: fitting SNMF", model, sigma_mult, seed)
        estimator = SNMF(
            rank=rank,
            max_outer=max_outer,
            max_inner=max_inner,
            tol=tol,
            init="random_sqrt",
            random_state=seed,
        )
        fit_start = time.perf_counter()
        estimator.fit(similarity)
        fit_time = time.perf_counter() - fit_start
        w = estimator.w_
        np.savez_compressed(w_path, w=w)

        evar = explained_variance(similarity, w)
        metrics = {
            "model": model,
            "training_dataset": dataset,
            "feature_path": str(feature_path),
            "feature_sha256": feature_sha256,
            "n_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "rank": rank,
            "median_sigma": sigma0,
            "sigma_mult": sigma_mult,
            "sigma": sigma,
            "seed": seed,
            "max_outer": max_outer,
            "max_inner": max_inner,
            "tol": tol,
            "n_iter": int(getattr(estimator, "n_iter_", -1)),
            "final_evar": evar,
            "rec_error": final_history_value(estimator, "rec_error"),
            "kernel_time_s": kernel_time,
            "fit_time_s": fit_time,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        rows.append(metrics)
        logger.info(
            "%s sigma=%s seed=%d: evar=%.4f fit=%.1fs",
            model,
            sigma_mult,
            seed,
            evar,
            fit_time,
        )
    return rows


def collect_seed_metrics(
    models: list[str],
    rank: int,
    sigmas: tuple[str, ...],
    seeds: tuple[int, ...],
    out: Path,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for sigma_mult in sigmas:
            for seed in seeds:
                path = seed_dir(out, model, rank, sigma_mult, seed) / "metrics.json"
                if path.exists():
                    rows.append(read_json(path))
    return pd.DataFrame(rows)


def select_components(
    models: list[str],
    datasets: dict[str, str],
    rank: int,
    sigmas: tuple[str, ...],
    seeds: tuple[int, ...],
    out: Path,
    *,
    expected_feature_hashes: dict[str, str],
    expected_sigma_bases: dict[str, float],
    expected_max_outer: int,
    expected_max_inner: int,
    expected_tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    dimension_rows: list[dict[str, object]] = []
    for model in models:
        for sigma_mult in sigmas:
            ws: list[np.ndarray] = []
            evars: list[float] = []
            complete = True
            for seed in seeds:
                directory = seed_dir(out, model, rank, sigma_mult, seed)
                w_path = directory / "w.npz"
                metrics_path = directory / "metrics.json"
                if not (w_path.exists() and metrics_path.exists()):
                    complete = False
                    break
                metrics = read_json(metrics_path)
                if (
                    metrics.get("model") != model
                    or int(metrics.get("rank", -1)) != rank
                    or str(metrics.get("sigma_mult")) != sigma_mult
                    or int(metrics.get("seed", -1)) != seed
                    or metrics.get("feature_sha256") != expected_feature_hashes[model]
                    or int(metrics.get("max_outer", -1)) != expected_max_outer
                    or int(metrics.get("max_inner", -1)) != expected_max_inner
                    or not np.isclose(float(metrics.get("tol", np.nan)), expected_tol)
                    or not np.isclose(
                        float(metrics.get("median_sigma", np.nan)),
                        expected_sigma_bases[model],
                    )
                ):
                    raise ValueError(f"{metrics_path}: candidate provenance mismatch")
                expected_sigma = float(metrics["median_sigma"]) * float(sigma_mult)
                if not np.isclose(float(metrics["sigma"]), expected_sigma):
                    raise ValueError(f"{metrics_path}: inconsistent sigma/base/multiplier")
                ws.append(np.load(w_path)["w"])
                if ws[-1].ndim != 2 or ws[-1].shape[1] != rank:
                    raise ValueError(f"{w_path}: expected (*, {rank}), got {ws[-1].shape}")
                if not np.isfinite(ws[-1]).all():
                    raise ValueError(f"{w_path}: non-finite selected-candidate W")
                if float(ws[-1].min()) < -1e-10:
                    raise ValueError(f"{w_path}: negative selected-candidate W")
                evars.append(float(metrics["final_evar"]))
            if not complete:
                logger.warning(
                    "%s sigma=%s: incomplete; excluded from selection", model, sigma_mult
                )
                continue
            stability_by_dim = factorization_stability(ws)
            stability = float(np.nanmean(stability_by_dim))
            evar = float(np.mean(evars))
            harmonic = (
                2.0 * stability * evar / (stability + evar)
                if stability > 0 and evar > 0
                else float("nan")
            )
            best_seed = int(seeds[reference_seed_index(ws)])
            candidate_rows.append(
                {
                    "model": model,
                    "training_dataset": datasets[model],
                    "sigma_mult": sigma_mult,
                    "best_seed": best_seed,
                    "n_seeds": len(seeds),
                    "stability": stability,
                    "evar": evar,
                    "harmonic": harmonic,
                }
            )
            dimension_rows.extend(
                {
                    "model": model,
                    "training_dataset": datasets[model],
                    "sigma_mult": sigma_mult,
                    "dimension": dimension,
                    "stability": float(value),
                }
                for dimension, value in enumerate(stability_by_dim)
            )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise ValueError("no complete sigma candidates available for selection")
    selected = (
        candidates.loc[candidates.groupby("model")["harmonic"].idxmax()]
        .sort_values("model")
        .reset_index(drop=True)
    )
    selected["selected_at_grid_boundary"] = selected["sigma_mult"].isin(
        {sigmas[0], sigmas[-1]}
    )
    candidates.to_csv(out / "meta" / f"stability_evar_all_rank_{rank}.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(
        out / "meta" / f"stability_by_dimension_rank_{rank}.csv", index=False
    )
    selected.to_csv(out / "meta" / f"bandwidth_selection_rank_{rank}.csv", index=False)
    return candidates, selected


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)

    datasets = {
        "ipcl_alexnet_gn_imagenet1k": "ImageNet-1K",
        "ipcl_alexnet_gn_openimagesv6": "OpenImages",
        "ipcl_alexnet_gn_places2": "Places-256",
        "ipcl_alexnet_gn_vggface2": "VGGFace2",
    }
    models = list(cfg.models) or list(datasets)
    unknown = set(models).difference(datasets)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")

    rank = cfg.factorization.rank
    sigmas = tuple(str(value) for value in cfg.factorization.sigma_grid)
    seeds = tuple(cfg.factorization.seeds)
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required for stability-based selection")
    feature_paths = {
        model: Path(cfg.data_root) / "features" / cfg.dataset / f"{model}.npy"
        for model in models
    }

    logger.info("hashing %d feature files for stale-cache protection", len(models))
    feature_hashes = {model: sha256_file(path) for model, path in feature_paths.items()}
    (out / "meta" / "factorization_run_config.json").write_text(
        json.dumps(
            {
                "models": models,
                "rank": rank,
                "sigmas": sigmas,
                "seeds": seeds,
                "max_outer": cfg.factorization.max_outer,
                "max_inner": cfg.factorization.max_inner,
                "tol": cfg.factorization.tol,
                "n_jobs": cfg.n_jobs,
                "feature_sha256": feature_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    bases = sigma_bases(models, datasets, feature_paths, feature_hashes, out)

    tasks = [(model, sigma_mult) for model in models for sigma_mult in sigmas]
    logger.info(
        "%d models x %d sigmas x %d seeds = %d fits (%d concurrent kernels)",
        len(models),
        len(sigmas),
        len(seeds),
        len(tasks) * len(seeds),
        cfg.n_jobs,
    )
    Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(fit_model_sigma)(
            model,
            datasets[model],
            feature_paths[model],
            sigma_mult,
            bases[model],
            out,
            rank=rank,
            seeds=seeds,
            max_outer=cfg.factorization.max_outer,
            max_inner=cfg.factorization.max_inner,
            tol=cfg.factorization.tol,
            feature_sha256=feature_hashes[model],
        )
        for model, sigma_mult in tasks
    )

    seed_metrics = collect_seed_metrics(models, rank, sigmas, seeds, out)
    seed_metrics.to_csv(out / "meta" / f"factorization_metrics_rank_{rank}.csv", index=False)
    candidates, selected = select_components(
        models,
        datasets,
        rank,
        sigmas,
        seeds,
        out,
        expected_feature_hashes=feature_hashes,
        expected_sigma_bases=bases,
        expected_max_outer=cfg.factorization.max_outer,
        expected_max_inner=cfg.factorization.max_inner,
        expected_tol=cfg.factorization.tol,
    )
    candidate_counts = candidates.groupby("model")["sigma_mult"].nunique()
    incomplete = {
        model: int(candidate_counts.get(model, 0))
        for model in models
        if int(candidate_counts.get(model, 0)) != len(sigmas)
    }
    if incomplete:
        raise ValueError(
            f"incomplete sigma grid (expected {len(sigmas)} complete candidates/model): "
            f"{incomplete}"
        )
    if len(selected) != len(models):
        missing = set(models).difference(selected["model"])
        raise ValueError(f"no selected factorization for: {sorted(missing)}")
    logger.info("selected components:\n%s", selected.to_string(index=False))


if __name__ == "__main__":
    main()
