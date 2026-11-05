from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from experiments.universality.shared import max_correlation

log = logging.getLogger(__name__)

CONTRAST_LABELS = {
    "compare_training": "Trained vs. Random",
    "compare_architecture": "Architecture (ImageNet classification)",
    "compare_goal_slip": "Objective (ViT + YFCC15M)",
    "compare_goal_taskonomy_tasks": "Task (ResNet-50 + Taskonomy)",
    "compare_goal_taskonomy_cluster": "Task cluster (ResNet-50 + Taskonomy)",
    "compare_goal_expertise": "Expertise domain (BiT)",
    "compare_goal_selfsupervised": "SSL method (ResNet-50 + ImageNet)",
    "compare_goal_contrastive": "Contrastive vs. non-contrastive",
    "compare_diet_imagenetsize": "Dataset scale (ImageNet-1K vs 21K)",
}


def build_grouping(
    meta: pd.DataFrame,
    contrasts: pd.DataFrame,
    grouping_name: str,
    grouping_type: str,
) -> tuple[dict[str, str], str]:
    """Build model->group mapping from metadata column or contrast column."""
    if grouping_type == "metadata":
        if grouping_name not in meta.columns:
            raise ValueError(
                f"Column '{grouping_name}' not in model metadata. "
                f"Available: {list(meta.columns)}"
            )
        sub = meta[["feature_uid", grouping_name]].copy()
        sub = sub.dropna(subset=[grouping_name])
        sub[grouping_name] = sub[grouping_name].astype(str).str.strip()
        sub = sub[sub[grouping_name] != ""]
        grouping = dict(zip(sub["feature_uid"], sub[grouping_name]))
        description = f"Grouped by metadata column: {grouping_name}"

    elif grouping_type == "contrast":
        if grouping_name not in contrasts.columns:
            available = [c for c in contrasts.columns if c.startswith("compare_")]
            raise ValueError(
                f"Contrast '{grouping_name}' not found. Available: {available}"
            )
        contrast_sub = contrasts[["model_string", grouping_name]].dropna(
            subset=[grouping_name]
        )
        contrast_sub = contrast_sub.drop_duplicates(subset=["model_string"])
        merged = meta[["feature_uid", "model_string"]].merge(
            contrast_sub, on="model_string", how="inner",
        )
        merged[grouping_name] = merged[grouping_name].astype(str).str.strip()
        merged = merged[merged[grouping_name] != ""]
        grouping = dict(zip(merged["feature_uid"], merged[grouping_name]))
        description = CONTRAST_LABELS.get(
            grouping_name, f"Controlled contrast: {grouping_name}"
        )

    else:
        raise ValueError(f"Unknown grouping_type: {grouping_type}")

    return grouping, description


def compute_dimension_specificity(
    components: dict[str, np.ndarray],
    target_model: str,
    same_group_models: set[str],
) -> dict[str, np.ndarray | int]:
    """Compute per-dimension within vs between max-correlation for one model."""
    w_target = components[target_model]
    n_dims = w_target.shape[1]

    within_corr = []
    between_corr = []

    for source_name, w_source in components.items():
        if source_name == target_model:
            continue
        corr = max_correlation(w_source, w_target)
        if source_name in same_group_models:
            within_corr.append(corr)
        else:
            between_corr.append(corr)

    if within_corr:
        within = np.mean(within_corr, axis=0)
    else:
        within = np.full(n_dims, np.nan)

    if between_corr:
        between = np.mean(between_corr, axis=0)
    else:
        between = np.full(n_dims, np.nan)

    return {
        "within": within,
        "between": between,
        "specificity": within - between,
        "n_within": len(within_corr),
        "n_between": len(between_corr),
    }


def compute_all_specificity(
    components: dict[str, np.ndarray],
    grouping: dict[str, str],
    min_group_size: int = 3,
    between_pool: str = "group",
) -> pd.DataFrame:
    """Compute specificity for all models and dimensions."""
    available = {m for m in components if m in grouping}
    group_counts = Counter(grouping[m] for m in available)
    valid_groups = {g for g, c in group_counts.items() if c >= min_group_size}

    valid_models = [m for m in available if grouping[m] in valid_groups]
    # Source pool: all components when between_pool="all", else only grouped models.
    source_pool = components if between_pool == "all" else {m: components[m] for m in valid_models}

    log.info(
        f"Using {len(valid_models)} models in {len(valid_groups)} groups "
        f"(min_group_size={min_group_size}, between_pool={between_pool})"
    )
    if between_pool == "all":
        log.info(f"Between pool: all {len(components)} models in zoo")
    for g in sorted(valid_groups):
        n = sum(1 for m in valid_models if grouping[m] == g)
        log.info(f"  {g}: {n} models")

    if len(valid_models) < 2:
        raise ValueError(
            f"Not enough models in valid groups. "
            f"Group sizes: {dict(group_counts)}"
        )

    rows = []
    n_dims = next(iter(source_pool.values())).shape[1]

    for target_model in valid_models:
        target_group = grouping[target_model]
        same_group = {
            m for m in valid_models
            if m != target_model and grouping[m] == target_group
        }
        scores = compute_dimension_specificity(source_pool, target_model, same_group)

        for dim_idx in range(n_dims):
            rows.append({
                "model": target_model,
                "group": target_group,
                "dimension": dim_idx,
                "within": scores["within"][dim_idx],
                "between": scores["between"][dim_idx],
                "specificity": scores["specificity"][dim_idx],
                "n_within": scores["n_within"],
                "n_between": scores["n_between"],
            })

    return pd.DataFrame(rows)


def summarize_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dimension-level results to group-level summary."""
    model_level = df.groupby(["model", "group"]).agg(
        within=("within", "mean"),
        between=("between", "mean"),
        specificity=("specificity", "mean"),
    ).reset_index()

    summary = model_level.groupby("group").agg(
        n_models=("model", "nunique"),
        mean_within=("within", "mean"),
        mean_between=("between", "mean"),
        mean_specificity=("specificity", "mean"),
        std_specificity=("specificity", "std"),
    ).reset_index()

    return summary.sort_values("mean_specificity", ascending=False)


def _precompute_pairwise_max_corr(
    components: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Pre-compute max-correlation for all model pairs."""
    model_names = sorted(components.keys())
    n = len(model_names)
    rank = next(iter(components.values())).shape[1]
    pairwise = np.full((n, n, rank), np.nan)

    for i, source in enumerate(model_names):
        for j, target in enumerate(model_names):
            if i == j:
                continue
            pairwise[i, j, :] = max_correlation(components[source], components[target])

    return pairwise, model_names


def _specificity_from_pairwise(
    pairwise: np.ndarray,
    model_names: list[str],
    grouping: dict[str, str],
    min_group_size: int,
    between_pool: str = "group",
) -> dict[str, float]:
    """Compute group-mean specificity from precomputed pairwise tensor."""
    available = [m for m in model_names if m in grouping]
    group_counts = Counter(grouping[m] for m in available)
    valid_groups = {g for g, c in group_counts.items() if c >= min_group_size}

    idx_lookup = {m: i for i, m in enumerate(model_names)}
    group_specs: dict[str, list[float]] = {g: [] for g in valid_groups}

    source_names = model_names if between_pool == "all" else available

    for target_name in available:
        g = grouping[target_name]
        if g not in valid_groups:
            continue
        j = idx_lookup[target_name]

        within_vals = []
        between_vals = []
        for source_name in source_names:
            if source_name == target_name:
                continue
            i = idx_lookup[source_name]
            corr = pairwise[i, j, :]
            if source_name in grouping and grouping[source_name] == g:
                within_vals.append(corr)
            elif between_pool == "all" or (
                source_name in grouping and grouping[source_name] in valid_groups
            ):
                between_vals.append(corr)

        if within_vals and between_vals:
            within_mean = np.mean(within_vals, axis=0)
            between_mean = np.mean(between_vals, axis=0)
            spec = float(np.mean(within_mean - between_mean))
            group_specs[g].append(spec)

    return {g: float(np.mean(vals)) for g, vals in group_specs.items() if vals}


def run_permutation_test(
    components: dict[str, np.ndarray],
    grouping: dict[str, str],
    min_group_size: int = 3,
    n_permutations: int = 1000,
    seed: int = 42,
    between_pool: str = "group",
) -> pd.DataFrame:
    """Permutation test for group-level specificity significance."""
    available = {m for m in components if m in grouping}
    group_counts = Counter(grouping[m] for m in available)
    valid_groups = {g for g, c in group_counts.items() if c >= min_group_size}
    valid_models = sorted(m for m in available if grouping[m] in valid_groups)

    # When between_pool="all", precompute pairwise over the full zoo so that
    # ungrouped models can serve as between-group sources.
    if between_pool == "all":
        pool = components
    else:
        pool = {m: components[m] for m in valid_models}

    log.info(f"Pre-computing pairwise correlations for {len(pool)} models...")
    pairwise, model_names = _precompute_pairwise_max_corr(pool)

    observed = _specificity_from_pairwise(
        pairwise, model_names, grouping, min_group_size, between_pool,
    )
    log.info(f"Observed specificity: {observed}")

    rng = np.random.default_rng(seed)
    # Only shuffle labels among grouped models; zoo models stay ungrouped.
    labeled_models = [m for m in model_names if m in grouping]
    labels = [grouping[m] for m in labeled_models]

    null_counts = {g: 0 for g in observed}
    for perm_i in range(n_permutations):
        shuffled_labels = rng.permutation(labels).tolist()
        shuffled_grouping = {
            m: lab for m, lab in zip(labeled_models, shuffled_labels)
        }
        perm_spec = _specificity_from_pairwise(
            pairwise, model_names, shuffled_grouping, min_group_size,
            between_pool,
        )
        for g, obs_val in observed.items():
            if g in perm_spec and perm_spec[g] >= obs_val:
                null_counts[g] += 1

    rows = []
    for g in sorted(observed):
        p = (null_counts[g] + 1) / (n_permutations + 1)
        rows.append({"group": g, "observed": observed[g], "p_value": p})

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["group", "observed", "p_value", "p_value_fdr"])
    if len(result) > 1:
        result["p_value_fdr"] = apply_fdr_correction(result["p_value"].values)
    else:
        result["p_value_fdr"] = result["p_value"]
    return result


def run_bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs on group-mean specificity."""
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    model_means = df.groupby(["model", "group"])["specificity"].mean().reset_index()

    rows = []
    for group, gdf in model_means.groupby("group"):
        values = gdf["specificity"].values
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means[b] = sample.mean()

        rows.append({
            "group": group,
            "mean_specificity": float(values.mean()),
            "ci_lower": float(np.percentile(boot_means, 100 * alpha)),
            "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha))),
        })

    return pd.DataFrame(rows)


def compute_effect_sizes(df: pd.DataFrame) -> dict:
    """Compute effect sizes for specificity differences across groups."""
    model_means = df.groupby(["model", "group"])["specificity"].mean().reset_index()
    groups = model_means["group"].unique()
    n_groups = len(groups)

    if n_groups < 2:
        return {
            "effect_size": np.nan, "effect_size_type": "n/a",
            "test_statistic": np.nan, "p_value": np.nan, "n_groups": n_groups,
        }

    group_values = [
        model_means[model_means["group"] == g]["specificity"].values
        for g in groups
    ]

    if n_groups == 2:
        a, b = group_values
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
        stat, p = scipy_stats.ttest_ind(a, b)
        return {
            "effect_size": float(d), "effect_size_type": "cohens_d",
            "test_statistic": float(stat), "p_value": float(p),
            "n_groups": n_groups,
        }

    stat, p = scipy_stats.f_oneway(*group_values)
    all_vals = np.concatenate(group_values)
    grand_mean = all_vals.mean()
    ss_between = sum(len(gv) * (gv.mean() - grand_mean)**2 for gv in group_values)
    ss_total = np.sum((all_vals - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    return {
        "effect_size": float(eta_sq), "effect_size_type": "eta_squared",
        "test_statistic": float(stat), "p_value": float(p),
        "n_groups": n_groups,
    }


def apply_fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)

    adjusted = p * m / ranks
    adjusted = np.minimum(adjusted, 1.0)

    reverse_order = order[::-1]
    adjusted_sorted = adjusted[reverse_order]
    for i in range(1, len(adjusted_sorted)):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted[reverse_order] = adjusted_sorted

    return adjusted
