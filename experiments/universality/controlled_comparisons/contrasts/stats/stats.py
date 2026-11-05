from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from data import COMPARE_COLS


def assumption_checks(
    group_arrays: list[np.ndarray], group_names: list[str]
) -> dict:
    """Run Shapiro-Wilk per group and Levene's test across groups."""
    normality = {
        name: float(stats.shapiro(arr).pvalue) if len(arr) >= 3 else float("nan")
        for name, arr in zip(group_names, group_arrays)
    }
    can_levene = len(group_arrays) >= 2 and all(len(a) >= 2 for a in group_arrays)
    levene_p = float(stats.levene(*group_arrays).pvalue) if can_levene else float("nan")
    return {"shapiro_wilk": normality, "levene_p": levene_p}


def _contrast_label(col: str) -> str:
    return col.replace("compare_", "").replace("_", " ").title()




def _extract_groups(
    sub: pd.DataFrame, col: str, y_col: str
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Extract per-group arrays and sorted group names from a contrast column."""
    groups = {name: grp[y_col].values for name, grp in sub.groupby(col)}
    group_names = sorted(groups.keys())
    return groups, group_names


def _group_descriptives(
    groups: dict[str, np.ndarray], group_names: list[str]
) -> dict:
    """Compute per-group n, mean, and std."""
    return {
        "group_sizes": {g: len(groups[g]) for g in group_names},
        "group_means": {g: float(groups[g].mean()) for g in group_names},
        "group_stds": {g: float(groups[g].std()) for g in group_names},
    }


def _ttest_two_groups(a: np.ndarray, b: np.ndarray) -> dict:
    """Independent-samples t-test for two groups."""
    t, p = stats.ttest_ind(a, b)
    return {"test": "t-test", "statistic": float(t), "p_value": float(p)}


def _anova_multi_group(group_arrays: list[np.ndarray]) -> dict:
    """One-way ANOVA for three or more groups."""
    f_stat, p_f = stats.f_oneway(*group_arrays)
    return {
        "test": "one-way ANOVA",
        "statistic": float(f_stat) if not np.isnan(f_stat) else float("nan"),
        "p_value": float(p_f) if not np.isnan(p_f) else float("nan"),
    }


def _pairwise_posthoc(
    groups: dict[str, np.ndarray], testable_names: list[str]
) -> list[dict]:
    """Pairwise t-tests with BH-FDR correction."""
    pairs = list(combinations(testable_names, 2))
    results = []
    raw_ps = []
    for gi, gj in pairs:
        t_pw, p_pw = stats.ttest_ind(groups[gi], groups[gj])
        results.append({
            "group_a": gi,
            "group_b": gj,
            "t": float(t_pw),
            "p_raw": float(p_pw),
        })
        raw_ps.append(float(p_pw))

    _, p_fdr, _, _ = multipletests(raw_ps, method="fdr_bh")
    for pw, pf in zip(results, p_fdr):
        pw["p_fdr"] = float(pf)

    return results


def run_contrast_stats(
    sub: pd.DataFrame, col: str, y_col: str = "universality"
) -> dict:
    """Run statistical tests for a single contrast column."""
    groups, group_names = _extract_groups(sub, col, y_col)
    n_groups = len(group_names)
    group_arrays = [groups[g] for g in group_names]

    result = {
        "contrast": col,
        "label": _contrast_label(col),
        "n_total": len(sub),
        "n_groups": n_groups,
        **_group_descriptives(groups, group_names),
        "assumptions": assumption_checks(group_arrays, group_names),
    }

    if n_groups == 2:
        a, b = group_arrays
        if len(a) < 2 or len(b) < 2:
            result.update({"test": "skipped", "reason": "groups too small"})
            return result
        result.update(_ttest_two_groups(a, b))

    elif n_groups > 2:
        testable = [g for g in group_arrays if len(g) > 1]
        if len(testable) < 2:
            all_vals = np.concatenate(group_arrays)
            result.update({
                "test": "descriptive only (singleton groups)",
                "statistic": float("nan"),
                "p_value": float("nan"),
                "range": float(all_vals.max() - all_vals.min()),
            })
            return result

        result.update(_anova_multi_group(group_arrays))

        testable_names = [g for g in group_names if len(groups[g]) > 1]
        if len(testable_names) >= 2:
            result["pairwise"] = _pairwise_posthoc(groups, testable_names)

    return result


def run_all_stats(
    df: pd.DataFrame, output_dir: Path, y_col: str = "universality"
) -> list[dict]:
    """Run statistical tests for all contrasts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for col in COMPARE_COLS:
        sub = df[df[col].notna()].copy()
        if len(sub) < 3:
            print(f"  Skipping {col}: only {len(sub)} models")
            continue
        if sub[col].nunique() < 2:
            print(f"  Skipping {col}: only {sub[col].nunique()} group")
            continue

        result = run_contrast_stats(sub, col, y_col=y_col)
        all_stats.append(result)

        p_str = f"p={result['p_value']:.4f}" if result["p_value"] >= 0.0001 else "p<.0001"
        print(
            f"  {_contrast_label(col):35s} | "
            f"n={result['n_total']:3d}, groups={result['n_groups']:2d} | "
            f"{result['test']:15s}: {result['statistic']:7.2f}, {p_str}"
        )

    return all_stats
