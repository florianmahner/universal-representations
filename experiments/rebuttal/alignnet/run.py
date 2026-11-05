#!/usr/bin/env python3
"""Universality of human-aligned (AligNet) models against their base backbones.

Downloads the released AligNet checkpoints, extracts penultimate features on
THINGS, factorizes them with the paper's protocol, scores every dimension
against the reference model pool, and compares Base / Distilled / AligNet.
"""
from __future__ import annotations

import logging
import math
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

from experiments.universality.shared import load_components_for_rank
from experiments.universality.shared.core import (
    METRICS,
    cos2_perm_null_thresholds,
    null_adjust,
)
from src.colors import INDIGO, ROSE, TEAL, soft
from src.factorization import get_snmf_class
from src.metrics import explained_variance
from src.stability import _correlation_matrix, _hungarian_match, factorization_stability

logger = logging.getLogger(__name__)

N_NULL_PERMS = 1000
N_BOOTSTRAP = 10000


def variant_labels() -> dict[str, str]:
    """Display name of each released checkpoint of a backbone."""
    return {"base_model": "Base", "untransformed": "Distilled", "alignet": "AligNet"}


def variant_colors() -> dict[str, str]:
    return {"base_model": INDIGO, "untransformed": ROSE, "alignet": TEAL}


def variants() -> tuple[str, ...]:
    return tuple(variant_labels())


def backbone_variants() -> dict[str, tuple[str, ...]]:
    """Backbones in the AligNet release and the variants published for each."""
    backbones = (
        "SigLIP-B",
        "SigLIP2-B",
        "CLIP-B",
        "CapPa-B",
        "DINOv1-B",
        "DINOv2-B",
        "DINOv2-B224",
        "ViT-B",
        "Scratch-B",
    )
    available = {backbone: variants() for backbone in backbones}
    available["Scratch-B"] = ("base_model", "alignet")
    return available


def normalize_model_part(value: str) -> str:
    return value.lower().replace("-", "_").replace("/", "_")


def model_id(backbone: str, variant: str) -> str:
    return f"alignet_{normalize_model_part(backbone)}_{variant}"


def archive_name(backbone: str, variant: str) -> str:
    return f"{backbone}-{variant}.tar.gz"


def model_dir_name(backbone: str, variant: str) -> str:
    return f"{backbone}-{variant}"


def model_url(backbone: str, variant: str) -> str:
    return f"https://storage.googleapis.com/alignet/models/{archive_name(backbone, variant)}"


def planned_models(cfg: DictConfig) -> pd.DataFrame:
    available = backbone_variants()
    rows = []
    for backbone in list(cfg.models) or list(available):
        published = set(available.get(backbone, variants()))
        for variant in variants():
            if variant not in published:
                logger.warning("Skipping unavailable variant: %s %s", backbone, variant)
                continue
            rows.append({
                "model": model_id(backbone, variant),
                "backbone": backbone,
                "variant": variant,
                "url": model_url(backbone, variant),
                "archive": archive_name(backbone, variant),
                "saved_model_dir": model_dir_name(backbone, variant),
            })
    if not rows:
        raise ValueError("No models selected")
    return pd.DataFrame(rows)


def stage_dirs(out: Path, dataset: str) -> dict[str, Path]:
    """Sub-directories of the experiment's outputs folder, created on demand."""
    dirs = {
        "models": out / "models",
        "features": out / "features" / dataset,
        "embeddings": out / "embeddings" / dataset,
        "meta": out / "meta",
        "universality": out / "universality",
        "figures": out / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_plan(cfg: DictConfig, out: Path) -> pd.DataFrame:
    stage_dirs(out, cfg.dataset)
    plan = planned_models(cfg)
    plan_path = out / "model_registry.csv"
    plan.to_csv(plan_path, index=False)
    logger.info("Selected %d model variants", len(plan))
    logger.info("Backbones: %s", ", ".join(dict.fromkeys(plan["backbone"])))
    logger.info("Variants: %s", ", ".join(dict.fromkeys(plan["variant"])))
    logger.info("Sigmas: %s", ", ".join(map(str, cfg.factorization.sigma_grid)))
    logger.info("Seeds: %s", ", ".join(map(str, cfg.factorization.seeds)))
    logger.info("Dataset: %s", cfg.dataset)
    logger.info("Outputs: %s", out)
    logger.info("Registry: %s", plan_path)
    return plan


def download_file(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("SKIP download %s", out_path.name)
        return

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        chunk_size = 1024 * 1024
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) if tqdm else None
        with out_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
        if progress is not None:
            progress.close()


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tar.extractall(output_dir)


def download_models(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> None:
    models_dir = stage_dirs(out, cfg.dataset)["models"]
    for row in plan.itertuples(index=False):
        archive_path = models_dir / row.archive
        saved_model_dir = models_dir / row.saved_model_dir
        download_file(row.url, archive_path)
        if (saved_model_dir / "saved_model.pb").exists():
            logger.info("SKIP extract %s", saved_model_dir.name)
            continue
        logger.info("Extracting %s", archive_path.name)
        safe_extract_tar(archive_path, models_dir)


def load_image_info(cfg: DictConfig) -> pd.DataFrame:
    image_csv = Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv"
    df = pd.read_csv(image_csv)
    required = {"row", "filename", "category", "path"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {image_csv}: {sorted(missing_columns)}")
    return df


def item_paths(cfg: DictConfig, items: pd.DataFrame) -> list[Path]:
    # The catalog's "path" column holds legacy locations; the images are read
    # from the staged tree below data_root instead.
    return [
        Path(cfg.data_root) / "images" / "things" / str(row.category) / str(row.filename)
        for row in items.itertuples(index=False)
    ]


def item_info(cfg: DictConfig) -> pd.DataFrame:
    df = load_image_info(cfg)
    items = df.assign(item_id=df["row"].astype(str)).reset_index(drop=True)
    missing = [p for p in item_paths(cfg, items)[:100] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing THINGS images, first missing path: {missing[0]}")
    return items


def resolve_saved_model_dir(models_dir: Path, saved_model_dir: str) -> Path:
    direct = models_dir / saved_model_dir
    if (direct / "saved_model.pb").exists():
        return direct
    candidates = sorted((models_dir / saved_model_dir).glob("**/saved_model.pb"))
    if candidates:
        return candidates[0].parent
    raise FileNotFoundError(f"No saved_model.pb found under {direct}")


def extract_features_for_model(
    saved_model_path: Path,
    paths: list[Path],
    feature_key: str = "pre_logits",
    image_size: int = 224,
    batch_size: int = 64,
) -> np.ndarray:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = tf.saved_model.load(str(saved_model_path))
    forward = model.signatures["serving_default"]
    features = []
    dataset = image_dataset(paths, image_size, batch_size, tf)
    n_batches = math.ceil(len(paths) / batch_size)
    progress_every = max(1, n_batches // 10)
    for batch_idx, images in enumerate(dataset, start=1):
        outputs = forward(images=images)
        if feature_key not in outputs:
            keys = ", ".join(sorted(outputs))
            raise KeyError(f"{feature_key!r} not found in SavedModel outputs. Available: {keys}")
        value = outputs[feature_key].numpy()
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        features.append(value.astype(np.float32, copy=False))
        if batch_idx % progress_every == 0 or batch_idx == n_batches:
            logger.info("Processed %d/%d batches from %s", batch_idx, n_batches, saved_model_path.name)
    return np.concatenate(features, axis=0)


def image_dataset(paths: list[Path], image_size: int, batch_size: int, tf: object) -> object:
    path_strings = [str(path) for path in paths]

    def load_one(path: object) -> object:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size], method="bicubic", antialias=True)
        image = tf.clip_by_value(image, 0.0, 255.0)
        return tf.cast(image, tf.float32) / 255.0

    return (
        tf.data.Dataset.from_tensor_slices(path_strings)
        .map(load_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def extract_features(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> None:
    dirs = stage_dirs(out, cfg.dataset)
    items = item_info(cfg)
    paths = item_paths(cfg, items)
    rows = []
    for row in plan.itertuples(index=False):
        out_path = dirs["features"] / f"{row.model}.npy"
        if out_path.exists():
            logger.info("SKIP features %s", row.model)
            arr = np.load(out_path, mmap_mode="r")
            rows.append(feature_record(row, arr.shape, out_path, cfg))
            continue
        saved_model_path = resolve_saved_model_dir(dirs["models"], row.saved_model_dir)
        logger.info("Extracting %s from %s images", row.model, len(paths))
        t0 = time.perf_counter()
        arr = extract_features_for_model(saved_model_path=saved_model_path, paths=paths)
        if arr.shape[0] != len(paths):
            raise ValueError(f"{row.model}: got {arr.shape[0]} rows, expected {len(paths)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{row.model}: non-finite features detected")
        np.save(out_path, arr)
        logger.info("Saved %s shape=%s in %.1fs", out_path, arr.shape, time.perf_counter() - t0)
        rows.append(feature_record(row, arr.shape, out_path, cfg))
    pd.DataFrame(rows).to_csv(out / "feature_metadata.csv", index=False)
    items[["row", "filename", "category", "path", "item_id"]].to_csv(
        out / f"{cfg.dataset}_item_info.csv",
        index=False,
    )


def feature_record(
    row: object,
    shape: tuple[int, ...],
    out_path: Path,
    cfg: DictConfig,
    feature_key: str = "pre_logits",
) -> dict[str, object]:
    return {
        "model": row.model,
        "backbone": row.backbone,
        "variant": row.variant,
        "dataset": cfg.dataset,
        "item_level": "image",
        "feature_key": feature_key,
        "n_items": int(shape[0]),
        "feature_dim": int(math.prod(shape[1:])),
        "path": str(out_path),
    }


def median_sigma(features: np.ndarray) -> float:
    distances = pdist(features, metric="euclidean")
    distances = distances[distances > 0]
    if distances.size == 0:
        raise ValueError("All pairwise feature distances are zero")
    return float(np.median(distances))


def rbf_similarity(features: np.ndarray, sigma: float) -> np.ndarray:
    distances = pairwise_distances(features, features, metric="euclidean")
    distances **= 2
    distances *= -1.0 / (2 * sigma**2)
    np.exp(distances, out=distances)
    return distances


def final_history_value(estimator: object, key: str) -> float:
    history = getattr(estimator, "history_", None)
    if isinstance(history, dict) and key in history and history[key]:
        return float(history[key][-1])
    return float("nan")


def fit_snmf_one(
    similarity: np.ndarray,
    feature_shape: tuple[int, int],
    model: str,
    sigma_mult: str,
    sigma0: float,
    sigma: float,
    seed: int,
    cfg: DictConfig,
    out_dir: Path,
    feature_key: str = "pre_logits",
) -> dict[str, object]:
    SNMF = get_snmf_class()

    estimator = SNMF(
        rank=cfg.factorization.rank,
        max_outer=cfg.factorization.max_outer,
        max_inner=cfg.factorization.max_inner,
        tol=cfg.factorization.tol,
        init="random_sqrt",
        random_state=seed,
        verbose=0,
    )

    logger.info("%s sigma=%s seed=%d: fitting SNMF", model, sigma_mult, seed)
    t0 = time.perf_counter()
    estimator.fit(similarity)
    fit_time = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "w.npz", w=estimator.w_)
    dump(estimator, out_dir / "estimator.joblib")

    evar = final_history_value(estimator, "evar")
    if not np.isfinite(evar):
        evar = explained_variance(similarity, estimator.w_)

    metrics = {
        "model": model,
        "rank": cfg.factorization.rank,
        "n_samples": int(feature_shape[0]),
        "feature_dim": int(feature_shape[1]),
        "feature_key": feature_key,
        "median_sigma": sigma0,
        "sigma_mult": sigma_mult,
        "sigma": sigma,
        "seed": seed,
        "max_outer": cfg.factorization.max_outer,
        "max_inner": cfg.factorization.max_inner,
        "tol": cfg.factorization.tol,
        "n_iter": int(getattr(estimator, "n_iter_", -1)),
        "final_evar": evar,
        "rec_error": final_history_value(estimator, "rec_error"),
        "fit_time_s": fit_time,
    }
    return metrics


def factorize(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> None:
    dirs = stage_dirs(out, cfg.dataset)
    rows = []
    for row in plan.itertuples(index=False):
        feature_path = dirs["features"] / f"{row.model}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found for {row.model}: {feature_path}")
        features = np.load(feature_path)
        sigma0 = median_sigma(features)
        for sigma_mult in map(str, cfg.factorization.sigma_grid):
            sigma = sigma0 * float(sigma_mult)
            logger.info("%s sigma=%s: building RBF kernel", row.model, sigma_mult)
            t0 = time.perf_counter()
            similarity = rbf_similarity(features, sigma)
            kernel_time = time.perf_counter() - t0
            for seed in cfg.factorization.seeds:
                out_dir = (
                    dirs["embeddings"] / row.model / f"rank_{cfg.factorization.rank}"
                    / f"sigma_{sigma_mult}" / f"seed_{seed}"
                )
                metrics_path = out_dir / "metrics.json"
                if (out_dir / "w.npz").exists() and metrics_path.exists():
                    logger.info("SKIP factorization %s sigma=%s seed=%d", row.model, sigma_mult, seed)
                    metrics = pd.read_json(metrics_path, typ="series").to_dict()
                else:
                    metrics = fit_snmf_one(
                        similarity=similarity,
                        feature_shape=features.shape,
                        model=row.model,
                        sigma_mult=sigma_mult,
                        sigma0=sigma0,
                        sigma=sigma,
                        seed=seed,
                        cfg=cfg,
                        out_dir=out_dir,
                    )
                    metrics["kernel_time_s"] = kernel_time
                    pd.Series(metrics).to_json(metrics_path, indent=2)
                metrics["backbone"] = row.backbone
                metrics["variant"] = row.variant
                rows.append(metrics)
    pd.DataFrame(rows).to_csv(out / "factorization_metrics.csv", index=False)


def load_w(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.load(path)["w"]
    return np.load(path)


def load_seed_ws(base_dir: Path, seeds: tuple[int, ...]) -> tuple[list[np.ndarray], list[int]]:
    ws, used = [], []
    for seed in seeds:
        path = base_dir / f"seed_{seed}" / "w.npz"
        if path.exists():
            ws.append(load_w(path))
            used.append(seed)
    return ws, used


def find_reference_seed(ws: list[np.ndarray]) -> int:
    n = len(ws)
    rank = ws[0].shape[1]
    avg = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = _correlation_matrix(ws[i], ws[j])
            perm = _hungarian_match(c)
            avg[i] += c[np.arange(rank), perm].sum()
    avg /= max(n - 1, 1)
    return int(np.argmax(avg))


def read_evar_from_metrics(seed_dir: Path) -> float:
    p = seed_dir / "metrics.json"
    if not p.exists():
        return float("nan")
    return float(pd.read_json(p, typ="series").get("final_evar", float("nan")))


def select_components(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> pd.DataFrame:
    dirs = stage_dirs(out, cfg.dataset)
    seeds = tuple(cfg.factorization.seeds)
    rows = []
    for row in plan.itertuples(index=False):
        for sigma_mult in map(str, cfg.factorization.sigma_grid):
            base_dir = dirs["embeddings"] / row.model / f"rank_{cfg.factorization.rank}" / f"sigma_{sigma_mult}"
            if not base_dir.exists():
                continue
            ws, used_seeds = load_seed_ws(base_dir, seeds)
            if len(ws) < 2:
                continue
            stability = float(np.nanmean(factorization_stability(ws)))
            evars = [read_evar_from_metrics(base_dir / f"seed_{s}") for s in used_seeds]
            mean_evar = float(np.nanmean(evars))
            harmonic = (
                2 * stability * mean_evar / (stability + mean_evar)
                if (np.isfinite(stability) and stability > 0 and mean_evar > 0) else float("nan")
            )
            best_seed = int(used_seeds[find_reference_seed(ws)])
            rows.append({
                "model": row.model,
                "backbone": row.backbone,
                "variant": row.variant,
                "sigma_mult": str(sigma_mult),
                "best_seed": best_seed,
                "n_seeds": len(ws),
                "stability": stability,
                "evar": mean_evar,
                "harmonic": harmonic,
            })
    all_rows = pd.DataFrame(rows)
    if all_rows.empty:
        raise ValueError("No component candidates found")
    all_rows.to_csv(dirs["meta"] / f"stability_evar_all_rank_{cfg.factorization.rank}.csv", index=False)
    best = (
        all_rows.loc[all_rows.groupby("model")["harmonic"].idxmax()]
        .sort_values(["backbone", "variant"])
        .reset_index(drop=True)
    )
    out_path = dirs["meta"] / f"bandwidth_selection_rank_{cfg.factorization.rank}.csv"
    best.to_csv(out_path, index=False)
    logger.info("Saved selection to %s (%d models)", out_path, len(best))
    logger.info("Sigma distribution:\n%s", best["sigma_mult"].value_counts().sort_index().to_string())
    return best


def load_selected_components(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> dict[str, np.ndarray]:
    dirs = stage_dirs(out, cfg.dataset)
    selection_path = dirs["meta"] / f"bandwidth_selection_rank_{cfg.factorization.rank}.csv"
    if not selection_path.exists():
        select_components(cfg, out, plan)
    selection = pd.read_csv(selection_path)
    components = {}
    for row in selection.itertuples(index=False):
        path = (
            dirs["embeddings"] / row.model / f"rank_{cfg.factorization.rank}"
            / f"sigma_{row.sigma_mult}" / f"seed_{int(row.best_seed)}" / "w.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Selected W not found: {path}")
        components[row.model] = load_w(path)
    return components


def score_against_reference(
    target_name: str,
    w_target: np.ndarray,
    reference: dict[str, np.ndarray],
    n_null_perms: int,
    seed: int,
) -> list[dict[str, object]]:
    sources = list(reference.items())
    n_dims = w_target.shape[1]
    all_scores = {name: np.empty((len(sources), n_dims)) for name in METRICS}
    for i, (_, w_source) in enumerate(sources):
        for metric_name, metric_fn in METRICS.items():
            all_scores[metric_name][i] = metric_fn(w_source, w_target)

    thresholds = cos2_perm_null_thresholds(
        w_target,
        [w for _, w in sources],
        n_perms=n_null_perms,
        seed=seed,
    )

    rows = []
    for metric_name, scores in all_scores.items():
        for dim in range(n_dims):
            rows.append(score_row(target_name, dim, metric_name, scores[:, dim], len(sources)))

    cos2_scores = all_scores["cos2_perm"]
    for dim in range(n_dims):
        adjusted = null_adjust(cos2_scores[:, dim], thresholds[:, dim])
        rows.append(score_row(target_name, dim, "cos2_perm_adj", adjusted, len(sources)))
    return rows


def score_row(
    model: str,
    dimension: int,
    metric: str,
    scores: np.ndarray,
    n_reference_models: int,
) -> dict[str, object]:
    return {
        "model": model,
        "dimension": dimension,
        "metric": metric,
        "universality": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "n_reference_models": n_reference_models,
    }


def score_universality(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> None:
    dirs = stage_dirs(out, cfg.dataset)
    target_components = load_selected_components(cfg, out, plan)
    reference = load_components_for_rank(rank=cfg.factorization.rank, dataset=cfg.dataset)
    plan_idx = plan.set_index("model")
    targets = list(target_components.items())
    example_ref = next(iter(reference.values()))
    for model, w_target in targets:
        if w_target.shape[0] != example_ref.shape[0]:
            raise ValueError(
                f"{model}: target has {w_target.shape[0]} rows, reference has {example_ref.shape[0]} rows"
            )

    logger.info("Scoring %d target models against %d references with n_jobs=%d",
                len(targets), len(reference), cfg.n_jobs)
    per_model = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(score_against_reference)(
            target_name=model,
            w_target=w_target,
            reference=reference,
            n_null_perms=N_NULL_PERMS,
            seed=cfg.seed + i,
        )
        for i, (model, w_target) in enumerate(targets)
    )

    rows = []
    for (model, _), model_rows in zip(targets, per_model):
        for score in model_rows:
            score["backbone"] = plan_idx.loc[model, "backbone"]
            score["variant"] = plan_idx.loc[model, "variant"]
        rows.extend(model_rows)

    per_dim = pd.DataFrame(rows)
    per_dim_path = dirs["universality"] / f"per_dimension_rank_{cfg.factorization.rank}.csv"
    per_dim.to_csv(per_dim_path, index=False)
    logger.info("Saved per-dimension scores to %s", per_dim_path)

    summary = summarize_universality(per_dim)
    summary_path = dirs["universality"] / f"model_summary_rank_{cfg.factorization.rank}.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved model summary to %s", summary_path)

    comparison = compare_variants(summary)
    comparison_path = dirs["universality"] / f"variant_comparison_rank_{cfg.factorization.rank}.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info("Saved variant comparison to %s", comparison_path)


def summarize_universality(per_dim: pd.DataFrame) -> pd.DataFrame:
    return (
        per_dim.groupby(["model", "backbone", "variant", "metric"])["universality"]
        .agg(
            mean_universality="mean",
            median_universality="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )


def compare_variants(summary: pd.DataFrame, metric: str = "cos2_perm_adj") -> pd.DataFrame:
    sub = summary[summary["metric"] == metric]
    pivot = sub.pivot(index="backbone", columns="variant", values="mean_universality").reset_index()
    for variant in variants():
        if variant not in pivot.columns:
            pivot[variant] = np.nan
    pivot["delta_alignet_minus_base"] = pivot["alignet"] - pivot["base_model"]
    pivot["delta_alignet_minus_untransformed"] = pivot["alignet"] - pivot["untransformed"]
    pivot["delta_untransformed_minus_base"] = pivot["untransformed"] - pivot["base_model"]
    return pivot[[
        "backbone",
        "base_model",
        "untransformed",
        "alignet",
        "delta_alignet_minus_base",
        "delta_alignet_minus_untransformed",
        "delta_untransformed_minus_base",
    ]]


def _save_compact(fig, out_path: Path, dpi: int = 450) -> None:
    """Save PDF (vector) + PNG at a screen-friendly DPI."""
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _grouped_variant_bars(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    out_path: Path,
    fmt: str = "{:.2f}",
    legend_loc: str = "upper left",
    err_col: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from src.plotting import create_figure

    labels, colors = variant_labels(), variant_colors()
    df = df.dropna(subset=[value_col]).copy()
    present_variants = [v for v in variants() if v in set(df["variant"])]
    df["variant"] = pd.Categorical(df["variant"], categories=present_variants, ordered=True)
    backbones = (
        df.groupby("backbone")[value_col].mean()
        .sort_values(ascending=True).index.tolist()
    )

    n_var = len(present_variants)
    n_bb = len(backbones)
    dense = n_bb > 4
    slot, bar_w = (0.30, 0.27) if dense else (0.18, 0.16)
    group_w = slot * (n_var - 1)
    group_gap = 0.55 if dense else 0.85
    if n_bb > 6:
        fig_w = max(7.0, 0.85 * n_bb)
        fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    else:
        size = "wide" if dense else "single"
        fig, ax = create_figure(size, pad_top=0.55 if dense else 0.35, pad_bottom=0.85 if dense else 0.55)

    centers = np.arange(n_bb) * (group_w + group_gap)
    ymax = float(df[value_col].max())
    pad = ymax * 0.02
    label_fs = plt.rcParams["font.size"] - (2 if dense else 1)
    for i, v in enumerate(present_variants):
        offset = (i - (n_var - 1) / 2) * slot
        heights = np.array([
            float(df[(df["backbone"] == bb) & (df["variant"] == v)][value_col].iloc[0])
            if not df[(df["backbone"] == bb) & (df["variant"] == v)].empty else np.nan
            for bb in backbones
        ])
        errs = None
        if err_col is not None and err_col in df.columns:
            errs = np.array([
                float(df[(df["backbone"] == bb) & (df["variant"] == v)][err_col].iloc[0])
                if not df[(df["backbone"] == bb) & (df["variant"] == v)].empty else 0.0
                for bb in backbones
            ])
        ax.bar(
            centers + offset, heights, width=bar_w,
            color=soft(colors[v]), edgecolor=colors[v],
            linewidth=1.0, label=labels[v], zorder=3,
            yerr=errs, error_kw=dict(ecolor=colors[v], elinewidth=0.8, capsize=1.5) if errs is not None else None,
        )
        for x_, h, e in zip(centers + offset, heights, (errs if errs is not None else np.zeros_like(heights))):
            if np.isfinite(h):
                ax.text(x_, h + pad + e, fmt.format(h), ha="center", va="bottom",
                        fontsize=label_fs, color=colors[v])

    ax.set_xticks(centers)
    rot = 30 if dense else 0
    ha = "right" if dense else "center"
    ax.set_xticklabels(backbones, rotation=rot, ha=ha)
    ax.set_xlim(centers[0] - group_w / 2 - 0.35, centers[-1] + group_w / 2 + 0.35)
    ax.set_ylim(0, ymax * 1.30)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if dense:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                  ncol=n_var, handlelength=1.0, handleheight=1.0,
                  borderaxespad=0.0, columnspacing=1.5, labelspacing=0.3)
    else:
        ax.legend(frameon=False, loc=legend_loc, handlelength=1.0, handleheight=1.0,
                  borderaxespad=0.2, labelspacing=0.3)
    _save_compact(fig, out_path)


def _scatter_base_vs_alignet(
    df: pd.DataFrame, value_col: str, axis_label: str, out_path: Path,
    err_col: str | None = None,
) -> None:
    """Per-backbone Base (x) vs AligNet (y) scatter with y=x diagonal."""
    import matplotlib.pyplot as plt
    from src.plotting import create_figure

    df = df.dropna(subset=[value_col]).copy()
    pivot = df.pivot_table(index="backbone", columns="variant", values=value_col).dropna(subset=["base_model", "alignet"])
    err_pivot = (
        df.pivot_table(index="backbone", columns="variant", values=err_col)
        if err_col is not None and err_col in df.columns else None
    )

    fig, ax = create_figure("square", pad_left=0.7, pad_right=0.25, pad_top=0.35, pad_bottom=0.6)
    lo = float(min(pivot["base_model"].min(), pivot["alignet"].min())) - 0.02
    hi = float(max(pivot["base_model"].max(), pivot["alignet"].max())) + 0.02
    lo = max(lo, 0.0)
    ax.plot([lo, hi], [lo, hi], "--", color="#BBBBBB", lw=0.8, zorder=1)

    texts = []
    for bb, row in pivot.iterrows():
        x, y = row["base_model"], row["alignet"]
        if err_pivot is not None:
            ex = float(err_pivot.loc[bb, "base_model"]) if not pd.isna(err_pivot.loc[bb, "base_model"]) else 0.0
            ey = float(err_pivot.loc[bb, "alignet"]) if not pd.isna(err_pivot.loc[bb, "alignet"]) else 0.0
            ax.errorbar([x], [y], xerr=[ex], yerr=[ey], fmt="none",
                        ecolor=INDIGO, elinewidth=0.8, capsize=1.5, alpha=0.7, zorder=2)
        ax.plot(x, y, "o", ms=8, mfc=soft(INDIGO), mec=INDIGO, mew=1.2, zorder=3)
        texts.append(ax.text(x, y, bb, fontsize=plt.rcParams["font.size"] - 0.5, color=INDIGO))

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4),
            expand=(1.2, 1.4),
        )
    except Exception:
        pass

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    clean = axis_label.replace("(NMF)", "").replace("(raw)", "").replace("(nmf)", "").strip().rstrip(",")
    if clean and clean[0].islower():
        clean = clean[0].upper() + clean[1:]
    ax.set_xlabel(f"{clean} (Base)")
    ax.set_ylabel(f"{clean} (AligNet)")
    ax.grid(True, color="#EEEEEE", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _save_compact(fig, out_path)


def _dumbbell_plot(
    df: pd.DataFrame, value_col: str, xlabel: str, out_path: Path, fmt: str = "{:.2f}",
    err_col: str | None = None,
) -> None:
    """Per-backbone Base→AligNet dumbbell: two dots connected by a colored arrow."""
    import matplotlib.pyplot as plt
    from src.plotting import create_figure

    colors = variant_colors()
    df = df.dropna(subset=[value_col]).copy()
    pivot = df.pivot_table(index="backbone", columns="variant", values=value_col).dropna(subset=["base_model", "alignet"])
    pivot["delta"] = pivot["alignet"] - pivot["base_model"]
    if err_col is not None and err_col in df.columns:
        err_pivot = df.pivot_table(index="backbone", columns="variant", values=err_col)
    else:
        err_pivot = None
    pivot = pivot.sort_values("alignet", ascending=True)

    n = len(pivot)
    if n > 6:
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.45 * n + 0.6)))
        fig.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.12)
    else:
        fig, ax = create_figure("wide", pad_left=0.95, pad_right=0.30, pad_top=0.4, pad_bottom=0.55)
    y = np.arange(n)

    for yi, (bb, row) in zip(y, pivot.iterrows()):
        x0, x1 = row["base_model"], row["alignet"]
        line_color = colors["alignet"] if x1 >= x0 else colors["base_model"]
        ax.plot([x0, x1], [yi, yi], "-", lw=1.2, color=line_color, alpha=0.6, zorder=2)
        if err_pivot is not None:
            e0 = float(err_pivot.loc[bb, "base_model"]) if not pd.isna(err_pivot.loc[bb, "base_model"]) else 0.0
            e1 = float(err_pivot.loc[bb, "alignet"]) if not pd.isna(err_pivot.loc[bb, "alignet"]) else 0.0
            ax.errorbar([x0], [yi], xerr=[e0], fmt="none",
                        ecolor=colors["base_model"], elinewidth=0.8, capsize=1.5, zorder=2)
            ax.errorbar([x1], [yi], xerr=[e1], fmt="none",
                        ecolor=colors["alignet"], elinewidth=0.8, capsize=1.5, zorder=2)
        ax.plot(x0, yi, "o", ms=7, mfc=soft(colors["base_model"]),
                mec=colors["base_model"], mew=1.1, zorder=3)
        ax.plot(x1, yi, "o", ms=7, mfc=soft(colors["alignet"]),
                mec=colors["alignet"], mew=1.1, zorder=3)
        ax.text(max(x0, x1) + 0.012, yi, f"{row['delta']:+.2f}",
                va="center", ha="left", fontsize=plt.rcParams["font.size"] - 1,
                color=line_color)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(pivot["alignet"].max(), pivot["base_model"].max()) * 1.18)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   mfc=soft(colors["base_model"]),
                   mec=colors["base_model"], mew=1.1, ms=7, label="Base"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   mfc=soft(colors["alignet"]),
                   mec=colors["alignet"], mew=1.1, ms=7, label="AligNet"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handlelength=0.5, borderaxespad=0.3)
    _save_compact(fig, out_path)


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _bars_with_bootstrap_ci(
    sig_df: pd.DataFrame,
    out_path: Path,
    ylabel: str = "Universality",
    variants_to_show: tuple[str, ...] = ("base_model", "alignet"),
    p_col: str = "p_value",
    title: str | None = None,
) -> None:
    """Bar plot with 95% bootstrap CI error bars and Base-vs-AligNet significance stars."""
    import matplotlib.pyplot as plt
    from src.plotting import create_figure

    labels, colors = variant_labels(), variant_colors()
    df = sig_df.copy()
    df["mean_alignet_for_sort"] = df["alignet_mean"]
    backbones = df.sort_values("mean_alignet_for_sort", ascending=True)["backbone"].tolist()

    n_var = len(variants_to_show)
    n_bb = len(backbones)
    dense = n_bb > 4
    slot, bar_w = (0.30, 0.27) if dense else (0.22, 0.20)
    group_w = slot * (n_var - 1)
    group_gap = 0.55 if dense else 0.85
    if n_bb > 6:
        fig_w = max(7.0, 0.85 * n_bb)
        fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    else:
        size = "wide" if dense else "single"
        fig, ax = create_figure(size, pad_top=0.65, pad_bottom=0.85 if dense else 0.55)

    centers = np.arange(n_bb) * (group_w + group_gap)
    label_fs = plt.rcParams["font.size"] - (2 if dense else 1)
    star_fs = plt.rcParams["font.size"] - 1

    ymax = float(np.nanmax([df[f"{v}_ci_hi"].max() for v in variants_to_show]))
    pad = ymax * 0.02

    for i, v in enumerate(variants_to_show):
        offset = (i - (n_var - 1) / 2) * slot
        means = np.array([df.set_index("backbone").loc[bb, f"{v}_mean"] for bb in backbones])
        lo = np.array([df.set_index("backbone").loc[bb, f"{v}_ci_lo"] for bb in backbones])
        hi = np.array([df.set_index("backbone").loc[bb, f"{v}_ci_hi"] for bb in backbones])
        err = np.vstack([means - lo, hi - means])
        ax.bar(
            centers + offset, means, width=bar_w,
            color=soft(colors[v]), edgecolor=colors[v],
            linewidth=1.0, label=labels[v], zorder=3,
            yerr=err, error_kw=dict(ecolor=colors[v], elinewidth=0.8, capsize=1.5),
        )
        for x_, m, h_ in zip(centers + offset, means, hi):
            ax.text(x_, h_ + pad, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=label_fs, color=colors[v])

    sig_lookup = df.set_index("backbone")
    for k, bb in enumerate(backbones):
        p = float(sig_lookup.loc[bb, p_col])
        b_hi = float(sig_lookup.loc[bb, "base_model_ci_hi"])
        a_hi = float(sig_lookup.loc[bb, "alignet_ci_hi"])
        bar_top = max(b_hi, a_hi) + pad * 5
        x_b = centers[k] - slot / 2
        x_a = centers[k] + slot / 2
        ax.plot([x_b, x_b, x_a, x_a], [bar_top - pad, bar_top, bar_top, bar_top - pad],
                color="#444444", linewidth=0.7, zorder=4)
        label = _significance_stars(p)
        ax.text((x_b + x_a) / 2, bar_top + pad * 0.6, label,
                ha="center", va="bottom", fontsize=star_fs, color="#222222")

    ax.set_xticks(centers)
    rot = 30 if dense else 0
    ha = "right" if dense else "center"
    ax.set_xticklabels(backbones, rotation=rot, ha=ha)
    ax.set_xlim(centers[0] - group_w / 2 - 0.35, centers[-1] + group_w / 2 + 0.35)
    ax.set_ylim(0, ymax * 1.45)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              ncol=n_var, handlelength=1.0, handleheight=1.0,
              borderaxespad=0.0, columnspacing=1.5, labelspacing=0.3)
    if title is not None:
        ax.set_title(title, fontsize=plt.rcParams["font.size"], pad=20, loc="right", color="#444444")
    _save_compact(fig, out_path)


def _robust_per_variant(cfg: DictConfig, out: Path) -> pd.DataFrame:
    """Cross-seed mean + SEM of universality at the selected sigma, per variant."""
    rob = pd.read_csv(
        out / "universality" / f"robustness_rank_{cfg.factorization.rank}.csv"
    )
    g = rob.groupby(["model", "backbone", "variant"]).agg(
        value=("mean_universality", "mean"),
        std=("mean_universality", "std"),
        n=("mean_universality", "count"),
    ).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    return g


def plot_results(cfg: DictConfig, out: Path) -> None:
    dirs = stage_dirs(out, cfg.dataset)
    rank = cfg.factorization.rank
    plot_df = _robust_per_variant(cfg, out)
    logger.info("Plotting with cross-seed robustness estimates")

    _grouped_variant_bars(
        plot_df, value_col="value", ylabel="Universality",
        out_path=dirs["figures"] / f"variant_universality_rank_{rank}",
        err_col="sem",
    )
    _grouped_variant_bars(
        plot_df[plot_df["variant"].isin(["base_model", "alignet"])],
        value_col="value", ylabel="Universality",
        out_path=dirs["figures"] / f"variant_universality_2way_rank_{rank}",
        err_col="sem",
    )
    _dumbbell_plot(
        plot_df, value_col="value", xlabel="Universality",
        out_path=dirs["figures"] / f"variant_universality_dumbbell_rank_{rank}",
        err_col="sem",
    )
    _scatter_base_vs_alignet(
        plot_df, value_col="value", axis_label="universality",
        out_path=dirs["figures"] / f"variant_universality_scatter_rank_{rank}",
        err_col="sem",
    )

    sig_df = pd.read_csv(dirs["universality"] / f"significance_rank_{rank}.csv")
    _bars_with_bootstrap_ci(
        sig_df, p_col="p_holm",
        out_path=dirs["figures"] / f"variant_universality_significance_rank_{rank}",
    )
    logger.info("Plotted significance bars (Holm-corrected)")

    ds = pd.read_csv(dirs["universality"] / f"downstream_rank_{rank}.csv")
    downstream_panels = [("triplet_accuracy", "Triplet accuracy", "triplet")]
    macaque_cols = [c for c in ("macaque_F_mean_r", "macaque_N_mean_r") if c in ds.columns]
    if macaque_cols:
        ds = ds.assign(macaque_mean_r=ds[macaque_cols].mean(axis=1))
        downstream_panels.append(("macaque_mean_r", "Macaque IT encoding r", "macaque"))

    for col, label, slug in downstream_panels:
        _grouped_variant_bars(
            ds, value_col=col, ylabel=label,
            out_path=dirs["figures"] / f"variant_{slug}_rank_{rank}",
        )
        _grouped_variant_bars(
            ds[ds["variant"].isin(["base_model", "alignet"])],
            value_col=col, ylabel=label,
            out_path=dirs["figures"] / f"variant_{slug}_2way_rank_{rank}",
        )
        _dumbbell_plot(
            ds, value_col=col, xlabel=label,
            out_path=dirs["figures"] / f"variant_{slug}_dumbbell_rank_{rank}",
        )
        _scatter_base_vs_alignet(
            ds, value_col=col, axis_label=label.lower(),
            out_path=dirs["figures"] / f"variant_{slug}_scatter_rank_{rank}",
        )

    logger.info("Saved figures to %s", dirs["figures"])


def load_triplets(data_root: Path, n_categories: int) -> np.ndarray:
    triplet_dir = data_root / "behavior" / "things_triplets"
    parts = [np.loadtxt(triplet_dir / name, dtype=int) for name in ("trainset.txt", "validationset.txt")]
    triplets = np.concatenate(parts, axis=0)
    if not (triplets.min() >= 0 and triplets.max() < n_categories):
        raise ValueError(f"Triplet indices outside [0, {n_categories})")
    return triplets


def triplet_accuracy_from_sim(s: np.ndarray, triplets: np.ndarray) -> float:
    ii, jj, kk = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    s_ij, s_ik, s_jk = s[ii, jj], s[ii, kk], s[jj, kk]
    return float(((s_ij > s_ik) & (s_ij > s_jk)).mean())


def load_macaque_it(data_root: Path, monkey: str, threshold: float = 0.3) -> np.ndarray:
    path = data_root / "features" / "macaque" / "22k" / "processed" / f"{monkey}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True).item()
    it_start, it_end = d["roi_slices"]["it"]
    it_mua = d["train_MUA"][it_start:it_end]
    mean_reliab = d["reliab"][it_start:it_end].mean(axis=1)
    mask = mean_reliab > threshold
    return it_mua[mask].T


def encode_one(features: np.ndarray, y: np.ndarray, n_folds: int, random_state: int) -> float:
    from scipy import stats as scipy_stats
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)
    predictions = np.zeros_like(y)
    for train_idx, test_idx in cv.split(features):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(features[train_idx], y[train_idx])
        predictions[test_idx] = ridge.predict(features[test_idx])
    rs = np.array([
        scipy_stats.pearsonr(y[:, j], predictions[:, j]).statistic
        for j in range(y.shape[1])
    ])
    return float(np.mean(rs))


def _downstream_one(
    model: str,
    w: np.ndarray,
    backbone: str,
    variant: str,
    triplets: np.ndarray,
    exemplary_idx: np.ndarray,
    monkey_targets: dict[str, np.ndarray],
    n_folds: int,
    random_state: int,
) -> dict[str, object]:
    from sklearn.metrics.pairwise import cosine_similarity

    row: dict[str, object] = {
        "model": model, "backbone": backbone, "variant": variant,
        "triplet_accuracy": triplet_accuracy_from_sim(cosine_similarity(w[exemplary_idx]), triplets),
    }
    for monkey, y in monkey_targets.items():
        if w.shape[0] != y.shape[0]:
            continue
        row[f"macaque_{monkey}_mean_r"] = encode_one(w, y, n_folds, random_state)
    return row


def evaluate_downstream(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> pd.DataFrame:
    dirs = stage_dirs(out, cfg.dataset)
    components = load_selected_components(cfg, out, plan)
    plan_idx = plan.set_index("model")
    # THINGS has 1,854 concepts with 12 images each; the triplet task is defined
    # over concepts, so score the first image of every concept.
    n_categories = 1854
    exemplary_idx = np.arange(n_categories) * 12
    triplets = load_triplets(Path(cfg.data_root), n_categories)

    monkey_targets: dict[str, np.ndarray] = {}
    for monkey in ("F", "N"):
        try:
            monkey_targets[monkey] = load_macaque_it(Path(cfg.data_root), monkey)
            logger.info("Macaque %s: %d reliable IT neurons", monkey, monkey_targets[monkey].shape[1])
        except FileNotFoundError:
            logger.warning("Macaque %s data not found, skipping", monkey)

    logger.info("Downstream eval on %d models with n_jobs=%d", len(components), cfg.n_jobs)
    rows = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(_downstream_one)(
            model=model, w=w,
            backbone=plan_idx.loc[model, "backbone"],
            variant=plan_idx.loc[model, "variant"],
            triplets=triplets, exemplary_idx=exemplary_idx,
            monkey_targets=monkey_targets,
            n_folds=5, random_state=cfg.seed,
        )
        for model, w in components.items()
    )

    df = pd.DataFrame(rows)
    out_path = dirs["universality"] / f"downstream_rank_{cfg.factorization.rank}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved downstream metrics to %s", out_path)
    return df


def evaluate_robustness(cfg: DictConfig, out: Path, plan: pd.DataFrame) -> pd.DataFrame:
    """Score universality across all seeds at the selected sigma, per variant.

    Lets us put error bars on Base vs AligNet so the comparison isn't seed-noise.
    """
    dirs = stage_dirs(out, cfg.dataset)
    selection = pd.read_csv(dirs["meta"] / f"bandwidth_selection_rank_{cfg.factorization.rank}.csv")
    reference = load_components_for_rank(rank=cfg.factorization.rank, dataset=cfg.dataset)

    jobs = []
    for row in selection.itertuples(index=False):
        for seed in cfg.factorization.seeds:
            w_path = (
                dirs["embeddings"] / row.model / f"rank_{cfg.factorization.rank}"
                / f"sigma_{row.sigma_mult}" / f"seed_{seed}" / "w.npz"
            )
            if w_path.exists():
                jobs.append((row.model, row.backbone, row.variant, str(row.sigma_mult), int(seed), w_path))

    logger.info("Robustness: %d (variant, seed) jobs at n_jobs=%d", len(jobs), cfg.n_jobs)

    raw_dir = dirs["universality"] / f"raw_pool_scores_rank_{cfg.factorization.rank}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    null_seed = int(cfg.seed)

    def _score_one(model, backbone, variant, sigma_mult, seed, w_path, ref):
        from experiments.universality.shared.core import (
            METRICS, cos2_perm_null_thresholds, null_adjust,
        )
        w = load_w(w_path)
        sources = list(ref.values())
        all_scores = np.empty((len(sources), w.shape[1]))
        for i, w_src in enumerate(sources):
            all_scores[i] = METRICS["cos2_perm"](w_src, w)
        thresholds = cos2_perm_null_thresholds(
            w,
            sources,
            n_perms=N_NULL_PERMS,
            seed=null_seed + seed,
        )
        adjusted = null_adjust(all_scores, thresholds)
        per_dim_adj = adjusted.mean(axis=0)
        np.savez(
            raw_dir / f"{model}__seed_{seed}.npz",
            cos2_perm=all_scores.astype(np.float32),
            thresholds=thresholds.astype(np.float32),
        )
        return {
            "model": model, "backbone": backbone, "variant": variant,
            "sigma_mult": str(sigma_mult), "seed": seed,
            "mean_universality": float(per_dim_adj.mean()),
            "std_universality": float(per_dim_adj.std()),
            "min_universality": float(per_dim_adj.min()),
            "max_universality": float(per_dim_adj.max()),
            "per_dim_adj": per_dim_adj.tolist(),
        }

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(_score_one)(m, b, v, sm, s, p, reference)
        for (m, b, v, sm, s, p) in jobs
    )
    per_dim_rows = []
    for r in results:
        per_dim = r.pop("per_dim_adj")
        for d, val in enumerate(per_dim):
            per_dim_rows.append({
                "model": r["model"], "backbone": r["backbone"], "variant": r["variant"],
                "sigma_mult": r["sigma_mult"], "seed": r["seed"],
                "dimension": d, "universality": float(val),
            })
    df = pd.DataFrame(results)
    out_path = dirs["universality"] / f"robustness_rank_{cfg.factorization.rank}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved robustness CSV to %s", out_path)

    per_dim_df = pd.DataFrame(per_dim_rows)
    per_dim_seed_path = dirs["universality"] / f"per_dimension_per_seed_rank_{cfg.factorization.rank}.csv"
    per_dim_df.to_csv(per_dim_seed_path, index=False)
    logger.info("Saved per-dim per-seed CSV to %s (%d rows)", per_dim_seed_path, len(per_dim_df))
    logger.info("Saved raw pool cos² + threshold .npz files to %s", raw_dir)

    summary = df.groupby(["model", "backbone", "variant"]).agg(
        mean_seed=("mean_universality", "mean"),
        std_seed=("mean_universality", "std"),
        n_seed=("mean_universality", "count"),
    ).reset_index()
    logger.info("\nCross-seed universality (selected sigma)\n%s", summary.to_string(index=False))
    return df


def _summarize_pair(
    backbone: str,
    unit_label: str,
    means: dict[str, float],
    boots: dict[str, np.ndarray],
    n_bootstrap: int,
    paired: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "backbone": backbone, "bootstrap_unit": unit_label,
        "n_bootstrap": n_bootstrap, "paired": paired,
    }
    for v, m in means.items():
        b = boots[v]
        lo, hi = np.percentile(b, [2.5, 97.5])
        row[f"{v}_mean"] = float(m)
        row[f"{v}_ci_lo"] = float(lo)
        row[f"{v}_ci_hi"] = float(hi)
    delta_b = boots["alignet"] - boots["base_model"]
    d_lo, d_hi = np.percentile(delta_b, [2.5, 97.5])
    p_one = float((delta_b <= 0).mean())
    p_value = 2.0 * min(p_one, 1.0 - p_one)
    row["delta"] = float(means["alignet"] - means["base_model"])
    row["delta_ci_lo"] = float(d_lo)
    row["delta_ci_hi"] = float(d_hi)
    row["p_value"] = p_value
    return row


def _bootstrap_pool(cfg: DictConfig, out: Path, rng: np.random.Generator) -> list[dict]:
    """Paired bootstrap over the reference-pool models, averaged across seeds."""
    from experiments.universality.shared.core import EPS

    dirs = stage_dirs(out, cfg.dataset)
    raw_dir = dirs["universality"] / f"raw_pool_scores_rank_{cfg.factorization.rank}"
    if not raw_dir.exists():
        logger.warning("No %s — skip pool bootstrap", raw_dir)
        return []
    selection = pd.read_csv(dirs["meta"] / f"bandwidth_selection_rank_{cfg.factorization.rank}.csv")

    per_bv: dict[tuple[str, str], list[np.ndarray]] = {}
    for sel_row in selection.itertuples(index=False):
        seed_arrays: list[np.ndarray] = []
        for seed in cfg.factorization.seeds:
            p = raw_dir / f"{sel_row.model}__seed_{seed}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            cos2 = d["cos2_perm"].astype(np.float32)
            thr = d["thresholds"].astype(np.float32)
            denom = np.maximum(1 - thr, EPS)
            adj = np.clip((cos2 - thr) / denom, 0, 1)
            seed_arrays.append(adj)
        if seed_arrays:
            per_bv[(sel_row.backbone, sel_row.variant)] = np.stack(seed_arrays, axis=0)

    rows = []
    for backbone in sorted({b for (b, _) in per_bv}):
        present = [v for v in variants() if (backbone, v) in per_bv]
        if not {"base_model", "alignet"}.issubset(present):
            continue
        arrays = {v: per_bv[(backbone, v)] for v in present}
        n_pool = arrays["base_model"].shape[1]
        means = {v: float(arrays[v].mean()) for v in present}

        idx_all = rng.integers(0, n_pool, size=(N_BOOTSTRAP, n_pool))
        boots = {v: np.empty(N_BOOTSTRAP, dtype=np.float64) for v in present}
        chunk = 200
        for start in range(0, N_BOOTSTRAP, chunk):
            end = min(start + chunk, N_BOOTSTRAP)
            idx_chunk = idx_all[start:end]
            for v in present:
                arr = arrays[v]
                sub = arr[:, idx_chunk, :]
                boots[v][start:end] = sub.mean(axis=(0, 2, 3))
        rows.append(_summarize_pair(backbone, "pool", means, boots, N_BOOTSTRAP, paired=True))
    return rows


def _add_multiple_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for unit, grp in df.groupby("bootstrap_unit", sort=False):
        grp = grp.copy().reset_index(drop=True)
        ps = grp["p_value"].to_numpy()
        n = len(ps)
        order = np.argsort(ps)
        sorted_ps = ps[order]
        adj_sorted = np.maximum.accumulate(sorted_ps * (n - np.arange(n)))
        adj_sorted = np.clip(adj_sorted, 0, 1)
        p_holm = np.empty(n)
        p_holm[order] = adj_sorted
        grp["p_bonferroni"] = np.minimum(ps * n, 1.0)
        grp["p_holm"] = p_holm
        grp["significant_raw"] = grp["p_value"] < 0.05
        grp["significant_bonferroni"] = grp["p_bonferroni"] < 0.05
        grp["significant_holm"] = grp["p_holm"] < 0.05
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def compute_significance(cfg: DictConfig, out: Path) -> pd.DataFrame:
    """Bootstrap Base vs AligNet universality over the reference pool.

    Resampling unit: the reference-pool models, paired across variants and
    averaged over seeds. Reports a 95% CI per variant, the delta CI, a two-sided
    bootstrap p, and Bonferroni / Holm corrections across the backbones.
    """
    dirs = stage_dirs(out, cfg.dataset)
    rng = np.random.default_rng(cfg.seed)
    rows = _bootstrap_pool(cfg, out, rng)
    if not rows:
        raise RuntimeError("No bootstrap rows produced — check inputs")

    df = _add_multiple_comparisons(pd.DataFrame(rows))

    out_path = dirs["universality"] / f"significance_rank_{cfg.factorization.rank}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(df))
    cols = ["backbone", "base_model_mean", "alignet_mean", "delta",
            "delta_ci_lo", "delta_ci_hi", "p_value", "p_bonferroni", "p_holm",
            "significant_raw", "significant_holm"]
    logger.info("\nn_bootstrap=%d\n%s", N_BOOTSTRAP, df[cols].to_string(index=False))
    return df


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    plan = write_plan(cfg, out)
    download_models(cfg, out, plan)
    extract_features(cfg, out, plan)
    factorize(cfg, out, plan)
    select_components(cfg, out, plan)
    score_universality(cfg, out, plan)
    evaluate_downstream(cfg, out, plan)
    evaluate_robustness(cfg, out, plan)
    compute_significance(cfg, out)
    plot_results(cfg, out)


if __name__ == "__main__":
    main()
