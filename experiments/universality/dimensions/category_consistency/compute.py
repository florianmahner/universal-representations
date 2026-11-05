from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.universality.shared import (
    load_components_for_rank,
    load_model_info,
    load_universality,
)
from src.data import ThingsDataset

log = logging.getLogger(__name__)

LOO_IMPORTANCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reconstruction_importance"
    / "outputs"
)


def load_base_data(
    rank: int = 50,
    max_models: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load W matrices, universality scores, categories, and model metadata."""
    ds = ThingsDataset()
    components = load_components_for_rank(rank=rank, max_models=max_models)

    n_items = next(iter(components.values())).shape[0]
    if n_items == len(ds.all_categories):
        categories = np.array(ds.all_categories)
    else:
        categories = np.array(ds.categories)
    assert len(categories) == n_items, (
        f"Category count {len(categories)} != W row count {n_items}"
    )

    univ = load_universality(rank=rank)
    model_info = load_model_info()

    log.info(
        "Loaded %d models, %d universality rows, %d items (%d categories)",
        len(components), len(univ), n_items, len(np.unique(categories)),
    )
    return components, univ, categories, model_info


def _build_category_encoding(
    categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build integer category labels and group sizes."""
    unique_cats = np.array(sorted(set(categories)))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_labels = np.array([cat_to_idx[c] for c in categories])
    n_cats = len(unique_cats)
    cat_sizes = np.bincount(cat_labels, minlength=n_cats).astype(float)
    return unique_cats, cat_labels, cat_sizes


def compute_category_selectivity(
    components: dict[str, np.ndarray],
    universality: pd.DataFrame,
    categories: np.ndarray,
) -> pd.DataFrame:
    """Compute eta-squared and effective category number for every dimension.

    eta_squared: fraction of loading variance explained by THINGS category
    membership (one-way ANOVA effect size). Asks "are exemplars of the same
    category assigned similar loadings?".

    c_eff: inverse participation ratio (Hill number of order 2) of the
    category-mean loadings. Asks "how many categories does this dimension
    substantially load on?". Ranges in [1, n_cats]: ~1 for a dimension
    concentrated on a single category; ~n_cats for a dimension spread
    uniformly across all categories. Complementary to eta^2 — together they
    distinguish category-specific (high eta^2, low c_eff) from broad
    semantic (high eta^2, high c_eff) dimensions.
    """
    unique_cats, cat_labels, cat_sizes = _build_category_encoding(categories)
    n_cats = len(unique_cats)

    univ_lookup = {}
    for _, row in universality.iterrows():
        univ_lookup[(row["model"], int(row["dimension"]))] = row["universality"]

    records = []
    models = sorted(components.keys())
    for i, model in enumerate(models):
        if i % 40 == 0:
            log.info("  Category selectivity: model %d/%d", i + 1, len(models))
        W = components[model]

        # Vectorized eta^2 for all dims at once
        grand_means = W.mean(axis=0)
        ss_total = np.sum((W - grand_means[np.newaxis, :]) ** 2, axis=0)

        cat_sums = np.zeros((n_cats, W.shape[1]))
        for c in range(n_cats):
            mask = cat_labels == c
            cat_sums[c] = W[mask].sum(axis=0)
        cat_means = cat_sums / cat_sizes[:, np.newaxis]
        ss_between = np.sum(
            cat_sizes[:, np.newaxis] * (cat_means - grand_means[np.newaxis, :]) ** 2,
            axis=0,
        )
        eta_sq = ss_between / (ss_total + 1e-12)

        # Effective number of categories (Hill order-2 / inverse participation ratio)
        # of the per-category mean loadings. NMF loadings are non-negative, so
        # cat_means can be normalized to a probability distribution over
        # categories without sign issues.
        # c_eff      : raw mass concentration. Dominated by the NMF baseline
        #              floor — even strongly selective dimensions look broad.
        # c_eff_excess: subtract per-dimension min before normalizing, so the
        #              measure reflects above-baseline concentration. This is
        #              the version that should be interpreted as "how many
        #              categories does this dimension select for".
        cat_mass = cat_means.sum(axis=0)
        q = cat_means / (cat_mass[np.newaxis, :] + 1e-12)
        c_eff = 1.0 / (np.sum(q ** 2, axis=0) + 1e-12)

        cat_means_excess = cat_means - cat_means.min(axis=0, keepdims=True)
        excess_mass = cat_means_excess.sum(axis=0)
        q_excess = cat_means_excess / (excess_mass[np.newaxis, :] + 1e-12)
        c_eff_excess = 1.0 / (np.sum(q_excess ** 2, axis=0) + 1e-12)

        # Loading-mass summary retained as a separate diagnostic.  It is not
        # the reconstruction importance (delta R²) reported in Figure 4.
        col_norms_sq = (W ** 2).sum(axis=0)
        total_norm_sq = col_norms_sq.sum()
        norm_fractions = col_norms_sq / (total_norm_sq + 1e-12)

        ss_within = ss_total - ss_between

        for d in range(W.shape[1]):
            records.append({
                "model": model,
                "dimension": d,
                "universality": univ_lookup.get((model, d), np.nan),
                "eta_squared": float(eta_sq[d]),
                "c_eff": float(c_eff[d]),
                "c_eff_excess": float(c_eff_excess[d]),
                "ss_between": float(ss_between[d]),
                "ss_within": float(ss_within[d]),
                "norm_fraction": float(norm_fractions[d]),
            })

    df = pd.DataFrame(records)
    log.info(
        "Category selectivity: %d dims, mean eta^2=%.3f (null=%.3f), "
        "mean c_eff=%.1f, mean c_eff_excess=%.1f (max=%d)",
        len(df), df["eta_squared"].mean(),
        (len(unique_cats) - 1) / (len(categories) - 1),
        df["c_eff"].mean(), df["c_eff_excess"].mean(), n_cats,
    )
    return df


def compute_within_model_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_dims: int = 10,
) -> dict:
    """Compute within-model Spearman correlations between two columns."""
    rhos = []
    for _, grp in df.dropna(subset=[x_col, y_col]).groupby("model"):
        if len(grp) >= min_dims:
            r, _ = spearmanr(grp[x_col], grp[y_col])
            if np.isfinite(r):
                rhos.append(r)
    rhos = np.array(rhos)
    return {
        "median_rho": float(np.median(rhos)),
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "frac_positive": float((rhos > 0).mean()),
        "frac_negative": float((rhos < 0).mean()),
        "n_models": len(rhos),
        "q25_rho": float(np.percentile(rhos, 25)),
        "q75_rho": float(np.percentile(rhos, 75)),
    }


def compute_category_bias(
    components: dict[str, np.ndarray],
    universality: pd.DataFrame,
    categories: np.ndarray,
) -> pd.DataFrame:
    """Per-category universality bias.

    For each THINGS category, computes the difference in mean loading between
    the universal half and model-specific half of each model's dimensions
    (split at the per-model median universality score), then averages across
    models. Positive = category loads more on universal dims; negative = more
    on model-specific dims. Balanced by construction, unaffected by the skewed
    universality distribution.
    """
    unique_cats, cat_labels, cat_sizes = _build_category_encoding(categories)
    n_cats = len(unique_cats)

    univ_by_model: dict[str, dict[int, float]] = {}
    for _, row in universality.iterrows():
        univ_by_model.setdefault(row["model"], {})[int(row["dimension"])] = row["universality"]

    diff_accum = np.zeros(n_cats)
    n_models = 0

    for model in sorted(components.keys()):
        W = components[model]
        if model not in univ_by_model:
            continue
        umap = univ_by_model[model]
        scores = np.array([umap.get(d, np.nan) for d in range(W.shape[1])])
        valid = ~np.isnan(scores)
        if valid.sum() < 2:
            continue
        median_u = np.median(scores[valid])
        universal_dims = np.where(valid & (scores >= median_u))[0]
        specific_dims = np.where(valid & (scores < median_u))[0]

        # Per-category mean loading in each half
        cat_mean_universal = np.zeros(n_cats)
        cat_mean_specific = np.zeros(n_cats)
        for c in range(n_cats):
            mask = cat_labels == c
            cat_mean_universal[c] = W[np.ix_(mask, universal_dims)].mean()
            cat_mean_specific[c] = W[np.ix_(mask, specific_dims)].mean()

        diff_accum += cat_mean_universal - cat_mean_specific
        n_models += 1

    mean_diff = diff_accum / n_models

    return pd.DataFrame({
        "category": unique_cats,
        "universality_bias": mean_diff,
    }).sort_values("universality_bias", ascending=False)


def load_importance_data(rank: int = 50) -> pd.DataFrame:
    """Load the manuscript's true leave-one-dimension-out delta R² table."""
    path = LOO_IMPORTANCE_DIR / f"loo_importance_full_rank_{rank}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Reconstruction importance not found at {path}. Run "
            "python -m experiments.universality.dimensions.reconstruction_importance.run "
            f"--rank {rank} before the characterization analysis."
        )
    df = pd.read_csv(path)
    required = {"model", "dimension", "delta_r2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "rank" in df.columns:
        df = df[df["rank"] == rank].copy()
    if df.duplicated(["model", "dimension"]).any():
        raise ValueError(f"{path} contains duplicate model/dimension rows")
    log.info("Loaded true LOO reconstruction importance: %d rows", len(df))
    return df


def compute_importance_universality(
    importance_df: pd.DataFrame,
    selectivity_df: pd.DataFrame,
    model_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Importance x universality orthogonality analysis.

    Reconstruction importance is the leave-one-dimension-out delta R² defined
    in the manuscript. Missing rows are an error; a loading-norm proxy must not
    silently replace the reported quantity.
    """
    loo = importance_df[["model", "dimension", "delta_r2"]].copy()
    base = selectivity_df[
        ["model", "dimension", "universality", "norm_fraction"]
    ].copy()
    df = base.merge(
        loo,
        on=["model", "dimension"],
        how="left",
        validate="one_to_one",
    )
    if df["delta_r2"].isna().any():
        missing = df.loc[
            df["delta_r2"].isna(), ["model", "dimension"]
        ].head().to_dict("records")
        raise ValueError(
            "True LOO reconstruction importance is missing for "
            f"{int(df['delta_r2'].isna().sum())} dimensions; first: {missing}"
        )
    df = df.rename(columns={"delta_r2": "importance"})
    id_col = "feature_uid" if "feature_uid" in model_info.columns else "name"
    meta_sub = model_info[[id_col, "model_class", "train_task"]].rename(
        columns={id_col: "model"}
    ).drop_duplicates(subset="model")
    df = df.merge(meta_sub, on="model", how="left", validate="many_to_one")
    log.info("Using true LOO delta R² as reconstruction importance")

    valid = df.dropna(subset=["universality", "importance"])
    rho_overall, p_overall = spearmanr(valid["universality"], valid["importance"])

    within = compute_within_model_correlations(valid, "universality", "importance")

    # Quadrant analysis
    univ_thresh = float(valid["universality"].quantile(0.25))
    imp_thresh = float(valid["importance"].quantile(0.75))
    low_univ = valid["universality"] < univ_thresh
    high_imp = valid["importance"] >= imp_thresh

    # Per-model idiosyncratic fraction
    id_col = "feature_uid" if "feature_uid" in model_info.columns else "name"
    meta_sub = model_info[[id_col, "model_class", "train_task", "objective_category"]].rename(
        columns={id_col: "model"}).drop_duplicates(subset="model")

    idio_frac = (
        valid.assign(is_idio=low_univ)
        .groupby("model")["is_idio"]
        .mean()
        .reset_index()
        .rename(columns={"is_idio": "frac_idiosyncratic"})
    )
    idio_frac = idio_frac.merge(meta_sub, on="model", how="left")

    stats = {
        "rho_overall": float(rho_overall),
        "p_overall": float(p_overall),
        "within_model": within,
        "universality_threshold": float(univ_thresh),
        "importance_threshold": float(imp_thresh),
        "quadrant_counts": {
            "universal_important": int((~low_univ & high_imp).sum()),
            "universal_unimportant": int((~low_univ & ~high_imp).sum()),
            "idiosyncratic_important": int((low_univ & high_imp).sum()),
            "idiosyncratic_noise": int((low_univ & ~high_imp).sum()),
        },
    }

    log.info(
        "Importance-universality: rho=%.3f, within-model median=%.3f",
        rho_overall, within["median_rho"],
    )
    return df, idio_frac, stats
