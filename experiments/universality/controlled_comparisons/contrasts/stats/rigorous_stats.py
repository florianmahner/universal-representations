from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from experiments.universality.shared import load_universality_by_model
from experiments.universality.controlled_comparisons.contrasts.uniform_tests.run import (
    run_clip_vit_convergence,
)
from src.data import ThingsDataset

N_BOOT = 10_000
ALPHA_LEVENE = 0.05


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Small-sample-corrected standardized mean difference."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * d)


def omega_squared(*groups: np.ndarray) -> float:
    """Omega-squared effect size for multi-group designs."""
    all_vals = np.concatenate(groups)
    k = len(groups)
    n = len(all_vals)
    grand = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_within = ss_within / (n - k) if n > k else np.nan
    denom = ss_between + ss_within + ms_within
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((ss_between - (k - 1) * ms_within) / denom)


def bootstrap_ci(fn, *arrays, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap percentile CI for an effect-size function."""
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        resampled = [rng.choice(a, size=len(a), replace=True) for a in arrays]
        try:
            val = fn(*resampled)
            if np.isfinite(val):
                boot.append(val)
        except Exception:
            pass
    if not boot:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def _descriptives(groups: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "n": int(len(g)),
            "mean": float(g.mean()),
            "sd": float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
            "shapiro_p": float(stats.shapiro(g).pvalue) if len(g) >= 3 else float("nan"),
        }
        for name, g in groups.items()
    }


def test_two_group(
    df: pd.DataFrame,
    col: str,
    label_a: str,
    label_b: str,
    y_col: str = "universality",
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """Independent-samples t-test with Welch fallback, Hedges' g with bootstrap CI."""
    sub = df[df[col].notna()]
    a = sub[sub[col] == label_a][y_col].values
    b = sub[sub[col] == label_b][y_col].values

    if len(a) < 2 or len(b) < 2:
        return {"error": "groups too small", "n_a": len(a), "n_b": len(b)}

    groups = {label_a: a, label_b: b}
    levene_p = float(stats.levene(a, b).pvalue)
    variance_homogeneous = levene_p >= ALPHA_LEVENE

    if variance_homogeneous:
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=True)
        test_name = "Student's t-test"
        df_param = len(a) + len(b) - 2
    else:
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
        test_name = "Welch's t-test"
        # Welch-Satterthwaite df
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
        n1, n2 = len(a), len(b)
        df_param = (v1/n1 + v2/n2) ** 2 / ((v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1))

    g = hedges_g(a, b)
    g_lo, g_hi = bootstrap_ci(hedges_g, a, b, n_boot=n_boot, seed=seed)
    mean_diff = float(a.mean() - b.mean())
    md_lo, md_hi = bootstrap_ci(lambda x, y: x.mean() - y.mean(), a, b, n_boot=n_boot, seed=seed)

    return {
        "contrast": col,
        "design": "two-group",
        "groups": _descriptives(groups),
        "assumptions": {
            "levene_p": levene_p,
            "variance_homogeneous": variance_homogeneous,
        },
        "primary_test": {
            "name": test_name,
            "t": float(t_stat),
            "df": float(df_param),
            "p": float(p_t),
        },
        "effect_size": {
            "type": "hedges_g",
            "value": g,
            "ci95": [g_lo, g_hi],
            "mean_diff": mean_diff,
            "mean_diff_ci95": [md_lo, md_hi],
        },
    }


def _welch_anova_satterthwaite(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Welch's ANOVA (Brown-Forsythe variant via Alexander-Govern)."""
    try:
        result = stats.alexandergovern(*groups)
        return float(result.statistic), float(result.pvalue), float("nan"), float("nan")
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def test_multi_group(
    df: pd.DataFrame,
    col: str,
    group_order: list[str] | None = None,
    y_col: str = "universality",
    drop_singletons: bool = True,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """One-way ANOVA with Welch fallback, omega² with bootstrap CI, BH-FDR post-hoc."""
    sub = df[df[col].notna()]
    all_groups = {name: grp[y_col].values for name, grp in sub.groupby(col)}

    dropped = {}
    if drop_singletons:
        dropped = {name: arr for name, arr in all_groups.items() if len(arr) < 2}
        all_groups = {name: arr for name, arr in all_groups.items() if len(arr) >= 2}

    if group_order:
        ordered = {g: all_groups[g] for g in group_order if g in all_groups}
        extras = {g: arr for g, arr in all_groups.items() if g not in ordered}
        groups_dict = {**ordered, **extras}
    else:
        groups_dict = dict(sorted(all_groups.items()))

    group_arrays = list(groups_dict.values())
    k = len(group_arrays)
    n_total = sum(len(g) for g in group_arrays)

    if k < 2:
        return {"error": "<2 testable groups"}

    levene_p = float(stats.levene(*group_arrays).pvalue) if all(len(g) >= 2 for g in group_arrays) else float("nan")
    variance_homogeneous = np.isfinite(levene_p) and levene_p >= ALPHA_LEVENE

    if variance_homogeneous:
        f_stat, p_primary = stats.f_oneway(*group_arrays)
        test_name = "One-way ANOVA"
    else:
        f_stat, p_primary, _, _ = _welch_anova_satterthwaite(group_arrays)
        test_name = "Welch's ANOVA (Alexander-Govern)"

    omega = omega_squared(*group_arrays)
    o_lo, o_hi = bootstrap_ci(lambda *gs: omega_squared(*gs), *group_arrays,
                              n_boot=n_boot, seed=seed)

    names = list(groups_dict.keys())
    pairs = list(combinations(names, 2))
    raw_ps = []
    pairwise = []
    for ga, gb in pairs:
        a_arr, b_arr = groups_dict[ga], groups_dict[gb]
        t_pw, p_pw = stats.ttest_ind(a_arr, b_arr, equal_var=variance_homogeneous)
        raw_ps.append(float(p_pw))
        pairwise.append({
            "a": ga,
            "b": gb,
            "mean_diff": float(a_arr.mean() - b_arr.mean()),
            "t": float(t_pw),
            "p_raw": float(p_pw),
        })
    if raw_ps:
        _, p_fdr, _, _ = multipletests(raw_ps, method="fdr_bh")
        for pw, pf in zip(pairwise, p_fdr):
            pw["p_fdr"] = float(pf)

    return {
        "contrast": col,
        "design": "multi-group",
        "groups": _descriptives(groups_dict),
        "dropped_singletons": {k_: float(v[0]) for k_, v in dropped.items()},
        "assumptions": {
            "levene_p": levene_p,
            "variance_homogeneous": variance_homogeneous,
        },
        "primary_test": {
            "name": test_name,
            "statistic": float(f_stat) if np.isfinite(f_stat) else None,
            "df_num": k - 1,
            "df_den": n_total - k,
            "p": float(p_primary) if np.isfinite(p_primary) else None,
        },
        "effect_size": {
            "type": "omega_squared",
            "value": omega,
            "ci95": [o_lo, o_hi],
        },
        "pairwise_posthoc_bh_fdr": pairwise,
    }


def describe_singletons(df: pd.DataFrame, col: str, y_col: str = "universality") -> dict:
    """Descriptive summary for contrasts where every group has n=1."""
    sub = df[df[col].notna()]
    groups = sub.groupby(col)[y_col].mean().sort_values(ascending=False).to_dict()
    vals = sub[y_col].values
    return {
        "contrast": col,
        "design": "descriptive-only (singletons)",
        "n": int(len(vals)),
        "range": float(vals.max() - vals.min()),
        "min": {"label": min(groups, key=groups.get), "value": float(min(groups.values()))},
        "max": {"label": max(groups, key=groups.get), "value": float(max(groups.values()))},
        "all_values": {k: float(v) for k, v in groups.items()},
    }


def test_clip_vit_bootstrap_sd(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "universality",
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    """Compatibility wrapper around the canonical manuscript implementation."""
    if y_col != "universality":
        raise ValueError("The manuscript convergence test uses universality")
    canonical = run_clip_vit_convergence(
        df,
        group_col=group_col,
        n_bootstrap=n_boot,
        seed=seed,
    )
    return {
        "contrast": canonical["contrast"],
        "design": "bootstrap on within-group SD",
        "n_group": canonical["n_group"],
        "null_pool_size": canonical["null_pool_size"],
        "observed_sd": canonical["observed_sd"],
        "observed_mean": canonical["observed_mean"],
        "null_sd_median": canonical["null_sd_median"],
        "null_sd_5pct": canonical["null_sd_5pct"],
        "null_sd_95pct": canonical["null_sd_95pct"],
        "primary_test": {
            "name": "Bootstrap test on within-group SD",
            "p_one_sided_convergence": canonical["p_value_lower_tail"],
            "n_resamples": canonical["n_resamples"],
            "seed": canonical["seed"],
            "null_pool": "all 162 models",
            "sidedness": "one-sided (observed SD < null SD)",
        },
    }


def load_data(rank: int, dataset: str = "things22k") -> pd.DataFrame:
    uni = load_universality_by_model(rank=rank, dataset=dataset)
    uni = uni.rename(columns={"mean": "universality"})
    ds = ThingsDataset()
    mi = ds.model_info.copy()
    compare_cols = [c for c in mi.columns if c.startswith("compare_")]
    keep = ["name"] + compare_cols
    df = uni.merge(mi[keep].drop_duplicates(subset=["name"]),
                   left_on="model", right_on="name", how="left")
    return df


def run_all(
    rank: int = 50,
    dataset: str = "things22k",
    output_dir: Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict:
    df = load_data(rank=rank, dataset=dataset)

    results = {
        "rank": rank,
        "n_models_total": int(len(df)),
        "full_set_universality_median": float(df["universality"].median()),
        "full_set_universality_mean": float(df["universality"].mean()),
        "n_boot": n_boot,
        "seed": seed,
        "contrasts": {},
    }

    results["contrasts"]["architecture"] = test_multi_group(
        df, "compare_architecture",
        group_order=["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"],
        n_boot=n_boot, seed=seed,
    )

    results["contrasts"]["taskonomy_cluster"] = test_multi_group(
        df, "compare_goal_taskonomy_cluster",
        group_order=["Semantic", "3D", "Geometric", "2D"],
        drop_singletons=True,
        n_boot=n_boot, seed=seed,
    )

    results["contrasts"]["taskonomy_tasks"] = describe_singletons(
        df, "compare_goal_taskonomy_tasks",
    )

    results["contrasts"]["contrastive"] = test_two_group(
        df, "compare_goal_contrastive", "Contrastive", "Non-Contrastive",
        n_boot=n_boot, seed=seed,
    )

    results["contrasts"]["slip"] = test_multi_group(
        df, "compare_goal_slip",
        group_order=["SimCLR", "CLIP", "SLIP"],
        n_boot=n_boot, seed=seed,
    )

    results["contrasts"]["imagenet_size"] = test_two_group(
        df, "compare_diet_imagenetsize", "imagenet", "imagenet21k",
        n_boot=n_boot, seed=seed,
    )

    results["contrasts"]["clip_vit_convergence"] = test_clip_vit_bootstrap_sd(
        df, "compare_diet_vit", n_boot=n_boot, seed=seed,
    )

    if output_dir is not None:
        (output_dir / f"rigorous_stats_rank_{rank}.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    return results


def format_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.4f}" if p >= 1e-4 else "<.0001"


def print_report(results: dict) -> None:
    print(f"\n{'=' * 80}")
    print(f"RIGOROUS STATS — rank {results['rank']}")
    print(f"Full-set universality: mean={results['full_set_universality_mean']:.4f} "
          f"median={results['full_set_universality_median']:.4f}")
    print(f"n_boot = {results['n_boot']}, seed = {results['seed']}")
    print(f"{'=' * 80}")

    for name, r in results["contrasts"].items():
        print(f"\n--- {name.upper()} ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        if r["design"] == "descriptive-only (singletons)":
            print(f"  n = {r['n']}, range = {r['range']:.4f}")
            print(f"  max: {r['max']['label']} = {r['max']['value']:.4f}")
            print(f"  min: {r['min']['label']} = {r['min']['value']:.4f}")
            continue

        if r["design"] == "bootstrap on within-group SD":
            print(f"  n_group = {r['n_group']}, null pool = {r['null_pool_size']} models")
            print(f"  observed SD = {r['observed_sd']:.4f}, observed mean = {r['observed_mean']:.4f}")
            print(f"  null SD: median = {r['null_sd_median']:.4f}, "
                  f"5-95% range = [{r['null_sd_5pct']:.4f}, {r['null_sd_95pct']:.4f}]")
            pt = r["primary_test"]
            print(f"  p (one-sided, convergence) = {format_p(pt['p_one_sided_convergence'])}")
            continue

        for gname, gd in r["groups"].items():
            sh = f", Shapiro p={gd['shapiro_p']:.3f}" if np.isfinite(gd.get('shapiro_p', np.nan)) else ""
            print(f"  {gname}: n={gd['n']}, mean={gd['mean']:.4f}, sd={gd['sd']:.4f}{sh}")

        if r.get("dropped_singletons"):
            for dname, dval in r["dropped_singletons"].items():
                print(f"  [dropped singleton] {dname}: value = {dval:.4f}")

        ap = r["assumptions"]
        print(f"  Levene p = {ap['levene_p']:.3f} (variance homogeneous: {ap['variance_homogeneous']})")

        pt = r["primary_test"]
        if r["design"] == "two-group":
            print(f"  {pt['name']}: t({pt['df']:.1f}) = {pt['t']:.3f}, p = {format_p(pt['p'])}")
        else:
            print(f"  {pt['name']}: stat({pt['df_num']}, {pt['df_den']}) = {pt['statistic']:.3f}, p = {format_p(pt['p'])}")

        es = r["effect_size"]
        lo, hi = es["ci95"]
        print(f"  {es['type']} = {es['value']:+.3f} [95% CI: {lo:+.3f}, {hi:+.3f}]")

        if r.get("pairwise_posthoc_bh_fdr"):
            sig_pairs = [pw for pw in r["pairwise_posthoc_bh_fdr"] if pw.get("p_fdr", 1) < 0.05]
            if sig_pairs:
                print("  Post-hoc (BH-FDR p<.05):")
                for pw in sig_pairs:
                    print(f"    {pw['a']} vs {pw['b']}: diff={pw['mean_diff']:+.4f}, "
                          f"t={pw['t']:+.3f}, p_fdr={format_p(pw['p_fdr'])}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    results = run_all(
        rank=cfg.factorization.rank,
        dataset=cfg.dataset,
        output_dir=out,
        n_boot=N_BOOT,
        seed=cfg.seed,
    )
    print_report(results)
    print(f"\nSaved to {out}/rigorous_stats_rank_{cfg.factorization.rank}.json")


if __name__ == "__main__":
    main()
