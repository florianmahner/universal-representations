from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .analysis import DEFAULT_TIER_EDGES, DEFAULT_TIER_LABELS

log = logging.getLogger(__name__)

COVERAGE_METRICS = [
    ("mean_universality", "Mean universality"),
    ("coherence", "Within-cluster coherence"),
    ("task_entropy", "Task provenance entropy"),
    ("gini", "Gini coefficient"),
    ("dominant_task_frac", "Dominant task fraction"),
]


def assign_tiers(
    profiles: pd.DataFrame,
    tier_edges: list[float] | None = None,
    tier_labels: list[str] | None = None,
) -> pd.Series:
    """Assign coverage tier labels to each cluster."""
    if tier_edges is None:
        tier_edges = DEFAULT_TIER_EDGES
    if tier_labels is None:
        tier_labels = DEFAULT_TIER_LABELS
    return pd.cut(
        profiles["model_coverage"],
        bins=tier_edges,
        labels=tier_labels,
        right=False,
    )


def _bootstrap_ci(
    x: np.ndarray, n_boot: int = 5000, ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean."""
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def _dunn_posthoc(
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Dunn's post-hoc test with Bonferroni correction after Kruskal-Wallis.

    Uses rank-based z-scores (standard Dunn formulation).
    """
    all_vals = np.concatenate(list(groups.values()))
    ranks = sp_stats.rankdata(all_vals)

    group_names = list(groups.keys())
    ns = {g: len(groups[g]) for g in group_names}
    N = len(all_vals)

    offset = 0
    mean_ranks = {}
    for g in group_names:
        n = ns[g]
        mean_ranks[g] = ranks[offset : offset + n].mean()
        offset += n

    tied = len(ranks) != len(set(ranks))
    if tied:
        _, counts = np.unique(ranks, return_counts=True)
        C = np.sum(counts**3 - counts) / (12 * (N - 1))
    else:
        C = 0.0

    sigma2 = (N * (N + 1) / 12 - C)

    records = []
    pairs = [(group_names[i], group_names[j])
             for i in range(len(group_names)) for j in range(i + 1, len(group_names))]
    n_comparisons = len(pairs)

    for ga, gb in pairs:
        se = np.sqrt(sigma2 * (1 / ns[ga] + 1 / ns[gb]))
        z = (mean_ranks[ga] - mean_ranks[gb]) / se if se > 0 else 0.0
        p = 2 * sp_stats.norm.sf(abs(z))
        records.append({
            "group_a": ga,
            "group_b": gb,
            "z": float(z),
            "p_raw": float(p),
            "p_bonferroni": min(float(p) * n_comparisons, 1.0),
        })

    return pd.DataFrame(records)


def test_tier_differences(
    profiles: pd.DataFrame,
    n_boot: int = 5000,
) -> dict:
    """Kruskal-Wallis + Dunn post-hoc + bootstrap CIs for metrics across tiers."""
    df = profiles.copy()
    df["tier"] = assign_tiers(df)
    df = df.dropna(subset=["tier"])

    tier_labels = DEFAULT_TIER_LABELS
    results = {}

    for col, label in COVERAGE_METRICS:
        groups = {t: df.loc[df["tier"] == t, col].dropna().values for t in tier_labels}
        groups = {t: v for t, v in groups.items() if len(v) > 0}

        tier_stats = {}
        for t, v in groups.items():
            lo, hi = _bootstrap_ci(v, n_boot=n_boot)
            tier_stats[t] = {
                "n": len(v),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
            }

        group_arrays = [groups[t] for t in tier_labels if t in groups]
        if len(group_arrays) >= 2 and all(len(g) >= 1 for g in group_arrays):
            h_stat, p_kw = sp_stats.kruskal(*group_arrays)
            posthoc = _dunn_posthoc(groups)
        else:
            h_stat, p_kw = float("nan"), float("nan")
            posthoc = pd.DataFrame()

        # Variance homogeneity (Levene — robust to non-normality, unlike Bartlett)
        if len(group_arrays) >= 2 and all(len(g) >= 2 for g in group_arrays):
            lev_stat, p_lev = sp_stats.levene(*group_arrays, center="median")
        else:
            lev_stat, p_lev = float("nan"), float("nan")

        results[col] = {
            "label": label,
            "tier_stats": tier_stats,
            "kruskal_wallis": {"H": float(h_stat), "p": float(p_kw)},
            "levene": {"W": float(lev_stat), "p": float(p_lev)},
            "posthoc": posthoc.to_dict("records") if len(posthoc) > 0 else [],
        }

    return results


def test_correlations(profiles: pd.DataFrame) -> dict:
    """Pearson and Spearman correlations of metrics with coverage, with p-values."""
    cov = profiles["model_coverage"].values
    results = {}

    for col, label in COVERAGE_METRICS:
        y = profiles[col].values
        valid = ~(np.isnan(cov) | np.isnan(y))
        c, yv = cov[valid], y[valid]

        r_p, p_p = sp_stats.pearsonr(c, yv)
        r_s, p_s = sp_stats.spearmanr(c, yv)

        results[col] = {
            "label": label,
            "n": int(valid.sum()),
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
        }

    return results


def test_provenance_shifts(
    profiles: pd.DataFrame,
    dim_df: pd.DataFrame | None = None,
    provenance: pd.DataFrame | None = None,
    n_perm: int = 1000,
) -> dict:
    """Chi-squared test for task composition differences across tiers.

    If dim_df is provided, also runs a permutation test shuffling cluster-tier
    assignments to test whether task composition differs from chance.
    """
    if provenance is None:
        return {}

    task_prov = provenance[
        (provenance["grouping"] == "task") & (provenance["tier"] != "baseline")
    ].copy()
    if len(task_prov) == 0:
        return {}

    df = profiles.copy()
    df["tier"] = assign_tiers(df)
    df = df.dropna(subset=["tier"])

    tier_labels = [t for t in DEFAULT_TIER_LABELS if t in df["tier"].unique()]

    tasks = sorted(task_prov["category"].unique())
    contingency = np.zeros((len(tier_labels), len(tasks)))
    for i, tier in enumerate(tier_labels):
        tier_clusters = df[df["tier"] == tier]
        for j, task in enumerate(tasks):
            match = task_prov[
                (task_prov["tier"] == tier) & (task_prov["category"] == task)
            ]
            n_dims = tier_clusters["size"].sum()
            contingency[i, j] = float(match["fraction"].iloc[0]) * n_dims if len(match) > 0 else 0.0

    contingency = np.round(contingency).astype(int)
    contingency = contingency[:, contingency.sum(axis=0) > 0]

    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi2, dof, _ = sp_stats.chi2_contingency(contingency)
        n_total = contingency.sum()
        k = min(contingency.shape)
        cramers_v = float(np.sqrt(chi2 / (n_total * (k - 1)))) if n_total > 0 and k > 1 else 0.0
    else:
        chi2, p_chi2, dof, cramers_v = float("nan"), float("nan"), 0, 0.0

    return {
        "chi2": float(chi2),
        "p": float(p_chi2),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "tier_labels": tier_labels,
        "tasks": tasks,
    }


def test_self_clustering(
    profiles: pd.DataFrame,
    n_perm: int = 1000,
) -> dict:
    """Test whether clusters are more task-homogeneous than expected by chance.

    For each cluster, the null model draws `size` dimensions from the marginal
    task distribution (proportional to each task's share of all dimensions),
    then computes the expected dominant-task fraction. Comparing observed
    dominant_task_frac to this null tests whether clusters are genuinely
    task-homogeneous rather than reflecting base rates.
    """
    df = profiles.dropna(subset=["dominant_task_frac", "dominant_task", "size"]).copy()
    if len(df) < 5:
        return {"skipped": True, "reason": "too few clusters"}

    # Marginal task distribution (overall fraction of dimensions per task)
    task_counts = df.groupby("dominant_task")["size"].sum()
    total = task_counts.sum()
    task_probs = (task_counts / total).values
    max_prob = task_probs.max()

    observed = float(df["dominant_task_frac"].mean())
    sizes = df["size"].values

    rng = np.random.default_rng(42)
    null_means = np.empty(n_perm)

    for i in range(n_perm):
        null_fracs = np.empty(len(sizes))
        for j, s in enumerate(sizes):
            draws = rng.choice(len(task_probs), size=s, p=task_probs)
            _, counts = np.unique(draws, return_counts=True)
            null_fracs[j] = counts.max() / s
        null_means[i] = null_fracs.mean()

    p_value = float((null_means >= observed).mean())

    return {
        "observed_mean_dominant_frac": observed,
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std()),
        "max_marginal_task_prob": float(max_prob),
        "p_value": p_value,
        "n_clusters": len(df),
        "n_permutations": n_perm,
    }


def test_gini_variance_collapse(profiles: pd.DataFrame) -> dict:
    """Levene and Brown-Forsythe tests for Gini variance differences across tiers."""
    df = profiles.copy()
    df["tier"] = assign_tiers(df)
    df = df.dropna(subset=["tier", "gini"])

    tier_labels = DEFAULT_TIER_LABELS
    groups = {t: df.loc[df["tier"] == t, "gini"].values for t in tier_labels}
    groups = {t: v for t, v in groups.items() if len(v) >= 2}
    group_arrays = [groups[t] for t in tier_labels if t in groups]

    if len(group_arrays) < 2:
        return {"skipped": True}

    # Levene (median-centered, more robust)
    lev_stat, p_lev = sp_stats.levene(*group_arrays, center="median")
    # Bartlett (assumes normality — included for comparison with original claim)
    bart_stat, p_bart = sp_stats.bartlett(*group_arrays)

    tier_variances = {t: {"n": len(groups[t]), "var": float(groups[t].var(ddof=1)),
                          "std": float(groups[t].std(ddof=1))}
                      for t in groups}

    return {
        "levene": {"W": float(lev_stat), "p": float(p_lev)},
        "bartlett": {"T": float(bart_stat), "p": float(p_bart)},
        "tier_variances": tier_variances,
    }


def run_all_stats(
    profiles: pd.DataFrame,
    provenance: pd.DataFrame | None = None,
    n_boot: int = 5000,
    n_perm: int = 1000,
) -> dict:
    """Run all statistical tests, return combined results dict."""
    log.info("Running tier difference tests (Kruskal-Wallis + Dunn + bootstrap CIs)")
    tier_diffs = test_tier_differences(profiles, n_boot=n_boot)

    log.info("Running correlation tests (Pearson + Spearman with p-values)")
    correlations = test_correlations(profiles)

    log.info("Running Gini variance collapse tests (Levene + Bartlett)")
    gini_var = test_gini_variance_collapse(profiles)

    log.info("Running provenance shift test (chi-squared)")
    prov_shift = test_provenance_shifts(profiles, provenance=provenance)

    log.info("Running self-clustering permutation test")
    self_clust = test_self_clustering(profiles, n_perm=n_perm)

    return {
        "tier_differences": tier_diffs,
        "correlations": correlations,
        "gini_variance_collapse": gini_var,
        "provenance_shifts": prov_shift,
        "self_clustering": self_clust,
    }
