from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .plot import plot_contrast_pvalues

BOOTSTRAP = 10000
METRIC = "cos2_perm_adj"


def load_data(data_root: Path, dataset: str, rank: int) -> pd.DataFrame:
    uni = pd.read_csv(
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    uni = uni[uni["metric"] == METRIC]
    per_model = uni.groupby("model")["universality"].mean().reset_index()
    meta = pd.read_csv(data_root / "meta" / "things22k" / "model_metadata.csv")
    return per_model.merge(meta, left_on="model", right_on="feature_uid", how="left")


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * d)


def _partial_eta_squared(groups: list[np.ndarray]) -> float:
    """Partial eta-squared (equal to eta-squared in a one-factor ANOVA)."""
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    denominator = ss_between + ss_within
    return float(ss_between / denominator) if denominator > 0 else float("nan")


def _omega_squared(groups: list[np.ndarray]) -> float:
    """Classical one-way ANOVA omega-squared, which may be negative."""
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)
    if df_within <= 0:
        return float("nan")
    ms_within = ss_within / df_within
    denominator = ss_between + ss_within + ms_within
    if denominator <= 0:
        return float("nan")
    return float((ss_between - df_between * ms_within) / denominator)


def _bootstrap_effect_ci(
    groups: list[np.ndarray],
    effect,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        samples = [
            rng.choice(group, size=len(group), replace=True) for group in groups
        ]
        estimates[index] = effect(samples)
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < max(100, n_bootstrap // 2):
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def run_contrast(
    df: pd.DataFrame,
    col: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run ANOVA or Welch's t on one contrast, excluding untestable singleton groups."""
    sub = df[df[col].notna()].copy()
    label = col.replace("compare_", "").replace("_", " ").title()
    n_total = len(sub)
    grouped = [
        (str(name), group["universality"].values)
        for name, group in sub.groupby(col)
    ]
    group_values = [values for _, values in grouped]
    testable = [g for g in group_values if len(g) >= 2]

    row = {
        "contrast": col,
        "label": label,
        "n_total": n_total,
        "n_groups": int(sub[col].nunique()),
        "n_groups_testable": len(testable),
        "group_sizes": {name: len(values) for name, values in grouped},
        "group_means": {
            name: float(np.mean(values)) for name, values in grouped
        },
        "group_stds": {
            name: float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            for name, values in grouped
        },
    }

    if len(testable) < 2:
        row.update({
            "test": "skipped (singletons)",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "effect_size": float("nan"),
            "effect_size_type": "",
            "effect_ci_low": float("nan"),
            "effect_ci_high": float("nan"),
            "effect_ci_method": "",
            "bootstrap_reps": n_bootstrap,
            "omega_squared": float("nan"),
            "omega_ci_low": float("nan"),
            "omega_ci_high": float("nan"),
        })
        return row

    if len(grouped) == 2 and len(testable) == 2:
        a, b = testable
        t, p = stats.ttest_ind(a, b, equal_var=False)
        ci_low, ci_high = _bootstrap_effect_ci(
            testable,
            lambda samples: _hedges_g(samples[0], samples[1]),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        row.update({
            "test": "Welch t",
            "statistic": float(t),
            "p_value": float(p),
            "effect_size": _hedges_g(a, b),
            "effect_size_type": "hedges_g",
            "effect_ci_low": ci_low,
            "effect_ci_high": ci_high,
            "effect_ci_method": "stratified percentile bootstrap",
            "bootstrap_reps": n_bootstrap,
            "omega_squared": float("nan"),
            "omega_ci_low": float("nan"),
            "omega_ci_high": float("nan"),
        })
    else:
        f, p = stats.f_oneway(*testable)
        ci_low, ci_high = _bootstrap_effect_ci(
            testable,
            _partial_eta_squared,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        omega_low, omega_high = _bootstrap_effect_ci(
            testable,
            _omega_squared,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        row.update({
            "test": "one-way ANOVA",
            "statistic": float(f),
            "p_value": float(p),
            "effect_size": _partial_eta_squared(testable),
            "effect_size_type": "partial_eta_squared",
            "effect_ci_low": ci_low,
            "effect_ci_high": ci_high,
            "effect_ci_method": "stratified percentile bootstrap",
            "bootstrap_reps": n_bootstrap,
            "omega_squared": _omega_squared(testable),
            "omega_ci_low": omega_low,
            "omega_ci_high": omega_high,
        })
    return row


def apply_corrections(tbl: pd.DataFrame) -> pd.DataFrame:
    tbl = tbl.copy()
    mask = tbl["p_value"].notna().values
    ps = tbl.loc[mask, "p_value"].values
    n_testable = int(mask.sum())
    n_prespecified = len(tbl)

    p_bonf_testable = np.full(len(tbl), np.nan)
    p_bh = np.full(len(tbl), np.nan)
    if n_testable:
        _, bonf_testable, _, _ = multipletests(ps, method="bonferroni")
        _, bh, _, _ = multipletests(ps, method="fdr_bh")
        p_bonf_testable[mask] = bonf_testable
        p_bh[mask] = bh

    # The methods section defines one family of ten pre-specified contrasts.
    # Descriptive singleton-only rows have no p-value, but remain in the
    # pre-specified family denominator.
    p_bonf_prespecified = np.where(
        mask,
        np.minimum(tbl["p_value"].fillna(0).to_numpy() * n_prespecified, 1.0),
        np.nan,
    )
    tbl["p_bonferroni"] = p_bonf_prespecified
    tbl["p_bonferroni_all_prespecified"] = p_bonf_prespecified
    tbl["p_bonferroni_testable"] = p_bonf_testable
    tbl["p_bh_fdr"] = p_bh
    tbl["n_tests_in_family"] = n_prespecified
    tbl["n_testable_contrasts"] = n_testable
    tbl["n_prespecified_contrasts"] = n_prespecified
    tbl["sig_raw"] = tbl["p_value"] < 0.05
    tbl["sig_bonferroni"] = tbl["p_bonferroni"] < 0.05
    tbl["sig_bh_fdr"] = tbl["p_bh_fdr"] < 0.05
    return tbl


def run_clip_vit_convergence(
    df: pd.DataFrame,
    *,
    group_col: str = "compare_diet_vit",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run the paper's one-sided within-group-SD convergence test."""
    in_group = df.loc[df[group_col].notna(), "universality"].to_numpy()
    full_pool = df["universality"].dropna().to_numpy()
    if len(in_group) < 2:
        raise ValueError(f"{group_col} must identify at least two models")
    if len(full_pool) < len(in_group):
        raise ValueError("Full-model null pool is smaller than the target group")

    observed_sd = float(np.std(in_group, ddof=1))
    rng = np.random.default_rng(seed)
    null_sds = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.choice(full_pool, size=len(in_group), replace=False)
        null_sds[index] = np.std(draw, ddof=1)
    p_lower = float((np.count_nonzero(null_sds <= observed_sd) + 1) /
                    (n_bootstrap + 1))
    return {
        "contrast": group_col,
        "test": "one-sided random-set bootstrap on within-group SD",
        "n_group": int(len(in_group)),
        "null_pool_size": int(len(full_pool)),
        "observed_mean": float(np.mean(in_group)),
        "observed_sd": observed_sd,
        "null_sd_median": float(np.median(null_sds)),
        "null_sd_5pct": float(np.percentile(null_sds, 5)),
        "null_sd_95pct": float(np.percentile(null_sds, 95)),
        "p_value_lower_tail": p_lower,
        "n_resamples": n_bootstrap,
        "sampling": "without replacement",
        "seed": seed,
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # The pre-specified family, in the order fixed by the methods section; the
    # position also seeds the per-contrast bootstrap.
    compare_cols = [
        "compare_training",
        "compare_architecture",
        "compare_goal_slip",
        "compare_goal_taskonomy_tasks",
        "compare_goal_taskonomy_cluster",
        "compare_goal_expertise",
        "compare_diet_vit",
        "compare_goal_selfsupervised",
        "compare_goal_contrastive",
        "compare_diet_imagenetsize",
    ]

    print("Loading data")
    df = load_data(Path(cfg.data_root), cfg.dataset, cfg.factorization.rank)

    print(f"Running uniform tests on {len(compare_cols)} contrasts")
    results = [
        run_contrast(
            df,
            col,
            n_bootstrap=BOOTSTRAP,
            seed=cfg.seed + index,
        )
        for index, col in enumerate(compare_cols)
    ]
    tbl = pd.DataFrame(results)
    tbl = apply_corrections(tbl)
    tbl = tbl.sort_values("p_value", na_position="last").reset_index(drop=True)
    tbl.to_csv(out / "contrast_results.csv", index=False)

    convergence = run_clip_vit_convergence(
        df,
        n_bootstrap=BOOTSTRAP,
        seed=cfg.seed,
    )
    (out / "clip_vit_convergence.json").write_text(
        json.dumps(convergence, indent=2)
    )

    print("Figure")
    plot_contrast_pvalues(tbl, out / "contrast_pvalues.pdf")

    (out / "results.json").write_text(json.dumps(tbl.to_dict("records"), indent=2, default=str))

    _print_report(tbl)
    print(
        "\nCLIP ViT-L/14 dataset convergence: "
        f"SD={convergence['observed_sd']:.4f}, "
        f"p={convergence['p_value_lower_tail']:.4f} "
        f"({convergence['n_resamples']:,} random size-"
        f"{convergence['n_group']} sets)"
    )
    print(f"\nOutputs at: {out}")


def _fmt_p(p):
    if not np.isfinite(p):
        return "  nan"
    if p < 1e-4:
        return "<.0001"
    return f"{p:.4f}"


def _print_report(tbl: pd.DataFrame) -> None:
    n_family = int(tbl["n_tests_in_family"].dropna().iloc[0]) if len(tbl) else 0
    n_testable = (
        int(tbl["n_testable_contrasts"].dropna().iloc[0]) if len(tbl) else 0
    )
    print()
    print("=" * 95)
    print(
        f"UNIFORM TESTS — {len(tbl)} pre-specified contrasts, {n_testable} "
        f"testable, Bonferroni denominator={n_family}"
    )
    print("=" * 95)
    print(f"{'Label':30s} {'n':>4} {'test':16s} {'ES [95% CI]':>25} {'raw p':>10} {'Bonf':>10}  verdict")
    print("-" * 95)
    for _, r in tbl.iterrows():
        es = f"{r['effect_size']:+.3f}" if np.isfinite(r['effect_size']) else "  n/a"
        if np.isfinite(r["effect_ci_low"]) and np.isfinite(r["effect_ci_high"]):
            es = f"{es} [{r['effect_ci_low']:+.3f}, {r['effect_ci_high']:+.3f}]"
        if not np.isfinite(r["p_value"]):
            verdict = "skipped"
        elif r["sig_bonferroni"]:
            verdict = "SIG (Bonferroni)"
        elif r["sig_bh_fdr"]:
            verdict = "SIG (BH-FDR only)"
        elif r["sig_raw"]:
            verdict = "raw sig, not corrected"
        else:
            verdict = "ns"
        print(
            f"{r['label'][:30]:30s} {r['n_total']:>4d} {r['test']:16s} "
            f"{es:>25} {_fmt_p(r['p_value']):>10} "
            f"{_fmt_p(r['p_bonferroni']):>10}  {verdict}"
        )


if __name__ == "__main__":
    main()
