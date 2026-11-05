from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ThingsDataset

logger = logging.getLogger(__name__)

# Taxonomy dimension -> the metadata column that defines it.
TAXONOMY_DIMS = {
    "brand": "source",
    "architecture": "model_class",
    "objective": "train_task",
    "dataset": "train_data",
}

# Model-name patterns excluded from the analyses (training trajectories, etc.).
EXCLUDE_PATTERNS = ["_epoch", "_dvd_"]


@lru_cache(maxsize=1)
def load_model_info() -> pd.DataFrame:
    """Load model metadata from the canonical CSV via ThingsDataset."""
    data = ThingsDataset()
    df = data.model_info.copy()
    return df.fillna("")


def _find_model_rows(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    identifier_columns = [
        col for col in ("feature_uid", "name", "model_string", "display_name")
        if col in df.columns
    ]

    if not identifier_columns:
        logger.warning("Model metadata has no identifier columns")
        return pd.DataFrame()

    mask = pd.Series(False, index=df.index)
    for col in identifier_columns:
        mask |= df[col].astype(str) == model_name
    return df[mask]


def get_model_info(model_name: str) -> dict[str, str] | None:
    """Get a model metadata row by model identifier."""
    df = load_model_info()
    if df.empty:
        return None

    matches = _find_model_rows(df, model_name)
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {col: str(row.get(col, "")).strip() for col in df.columns}


def _normalize_group_value(value: str) -> str:
    norm = value.strip().lower()
    return norm if norm else "unknown"


def filter_models(models: list[str], exclude_patterns: list[str] | None = None) -> list[str]:
    """
    Filter out models matching exclusion patterns.

    By default excludes training trajectory models (epoch snapshots).
    """
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS

    filtered = []
    for model_name in models:
        exclude = any(pattern in model_name.lower() for pattern in exclude_patterns)
        if not exclude:
            filtered.append(model_name)

    return filtered


def load_components_for_rank(
    rank: int = 50,
    models: list[str] | None = None,
    max_models: int | None = None,
    dataset: str = "things22k",
    exclude_trajectory: bool = True,
    strict: bool = True,
    data_root: Path | str = Path("data"),
) -> dict[str, np.ndarray]:
    """Load optimal W matrices for all models at a given rank.

    Parameters
    ----------
    rank : int
        NMF rank to load.
    models : list[str] | None
        Specific models to load. If None, load all available.
    max_models : int | None
        Limit number of models (useful for testing).
    dataset : str
        Dataset name (subdirectory of <data_root>/embeddings/).
    exclude_trajectory : bool
        If True, exclude training trajectory models (epoch snapshots).
    strict : bool
        If True, fail when a requested selection or W matrix is missing or
        malformed. Set False only for explicitly exploratory partial analyses.
    data_root : Path | str
        Root of the data tree; entry points pass ``cfg.data_root``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from model name to W matrix of shape (n_items, rank).
    """
    base_dir = Path(data_root) / "embeddings" / dataset
    rank_dir = f"rank_{rank}"

    if models is None:
        if not base_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {base_dir}")
        models = sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir() and (d / rank_dir).is_dir()
        )

    if exclude_trajectory:
        models = filter_models(models)

    sel_path = Path(data_root) / "meta" / dataset / f"bandwidth_selection_rank_{rank}.csv"
    if not sel_path.exists():
        raise FileNotFoundError(
            f"No sigma selection for dataset={dataset}, rank={rank} at {sel_path}. "
            f"Run: python -m experiments.factorization.bandwidth_selection.compute_selection "
            f"dataset={dataset} factorization.rank={rank}"
        )
    sel = pd.read_csv(sel_path)
    if max_models:
        models = models[:max_models]

    selected_models = set(sel["model"].astype(str))
    missing_selections = [m for m in models if m not in selected_models]
    if strict and missing_selections:
        preview = ", ".join(missing_selections[:5])
        raise KeyError(
            f"{len(missing_selections)} requested models lack a rank-{rank} "
            f"sigma selection (first: {preview})"
        )
    available_models = [m for m in models if m in selected_models]

    logger.info(f"Loading {len(available_models)} models at rank {rank}")

    sel_idx = sel.set_index("model")
    loaded: dict[str, np.ndarray] = {}
    failures: list[str] = []
    n_items: int | None = None
    for model_name in available_models:
        try:
            row = sel_idx.loc[model_name]
            seed_dir = (
                base_dir / model_name / rank_dir
                / f"sigma_{row['sigma_mult']}" / f"seed_{int(row['best_seed'])}"
            )
            npz_path = seed_dir / "w.npz"
            npy_path = seed_dir / "w.npy"
            if npz_path.exists():
                with np.load(npz_path) as archive:
                    w = archive["w"]
            elif npy_path.exists():
                w = np.load(npy_path)
            else:
                raise FileNotFoundError(f"No W matrix found in {seed_dir}")
            if w.ndim != 2 or w.shape[1] != rank:
                raise ValueError(f"expected (*, {rank}), got {w.shape}")
            if not np.isfinite(w).all():
                raise ValueError("W contains NaN or infinite values")
            if n_items is None:
                n_items = int(w.shape[0])
            elif w.shape[0] != n_items:
                raise ValueError(f"expected {n_items} rows, got {w.shape[0]}")
            loaded[model_name] = w
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")

    if failures and strict:
        preview = "\n  ".join(failures[:10])
        raise RuntimeError(
            f"Failed to load {len(failures)} selected component matrices:\n  {preview}"
        )
    for failure in failures:
        logger.warning("Failed to load %s", failure)

    logger.info(f"Successfully loaded {len(loaded)} models")
    return loaded


def get_model_taxonomy(model_name: str) -> dict[str, str]:
    """Get taxonomy values directly from metadata columns."""
    info = get_model_info(model_name)
    if info is None:
        return {dim: "unknown" for dim in TAXONOMY_DIMS}

    return {
        dim: _normalize_group_value(info.get(column, ""))
        for dim, column in TAXONOMY_DIMS.items()
    }


def get_models_in_group(models: list[str], taxonomy_dim: str, group_value: str) -> list[str]:
    """Get all models belonging to a specific taxonomy group."""
    if taxonomy_dim not in TAXONOMY_DIMS:
        raise ValueError(
            f"Unknown taxonomy dim: {taxonomy_dim}. Expected one of: {list(TAXONOMY_DIMS)}"
        )

    normalized_group = _normalize_group_value(group_value)
    return [model_name for model_name in models if get_model_taxonomy(model_name)[taxonomy_dim] == normalized_group]


def ensure_data_dir() -> Path:
    """Create the directory holding the per-dimension universality tables."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _universality_path(rank: int, dataset: str) -> Path:
    """Return CSV path for universality scores."""
    return Path(__file__).parent / "data" / f"universality_{dataset}_rank_{rank}.csv"


def load_universality(
    rank: int = 50,
    metric: str = "cos2_perm_adj",
    dataset: str = "things22k",
) -> pd.DataFrame:
    """Load per-dimension universality scores for all DNN models.

    Parameters
    ----------
    rank : int
        NMF rank (one of 10, 30, 50, 100, 200).
    metric : str
        Universality metric ("cos2_perm_adj", "cos2_perm", or "max_corr").
    dataset : str
        Dataset name (e.g. "things22k", "objectnet").

    Returns
    -------
    pd.DataFrame
        Columns: model, dimension, metric, universality, std, min, max, rank
    """
    path = _universality_path(rank, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"No universality data for rank {rank}, dataset {dataset} at {path}. "
            f"Run: python -m experiments.universality.metric.compute.run "
            f"dataset={dataset}"
        )
    df = pd.read_csv(path)
    return df[df["metric"] == metric].reset_index(drop=True)


def load_universality_by_model(
    rank: int = 50,
    metric: str = "cos2_perm_adj",
    dataset: str = "things22k",
) -> pd.DataFrame:
    """Load model-level mean universality (averaged across dimensions)."""
    df = load_universality(rank, metric=metric, dataset=dataset)
    return (
        df.groupby("model")["universality"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
