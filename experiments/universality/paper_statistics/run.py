from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats as sp_stats

from experiments.universality.shared.utils import load_universality_by_model


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    pr, pp = sp_stats.pearsonr(x, y)
    sr, sp = sp_stats.spearmanr(x, y)
    return {"n": int(mask.sum()), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp}


def collect_encoding_correlations() -> list[dict]:
    """Overall universality vs encoding across all models (per monkey and mean)."""
    encoding_dir = Path(
        "experiments/universality/downstream_predictions/macaque_encoding/encoding_things/outputs"
    )
    rows = []
    dfs = {}
    for monkey in ("F", "N"):
        path = encoding_dir / f"encoding_results_{monkey}.csv"
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        dfs[monkey] = pd.read_csv(path, usecols=["model", "mean_r", "universality"])

    if not dfs:
        return rows

    for monkey, df in dfs.items():
        c = _corr(df["universality"].values, df["mean_r"].values)
        rows.append({"section": "4.2", "statistic": f"univ_vs_encoding_monkey_{monkey}",
                      **c})

    if len(dfs) == 2:
        merged = dfs["F"].merge(dfs["N"], on="model", suffixes=("_F", "_N"))
        merged["encoding"] = (merged["mean_r_F"] + merged["mean_r_N"]) / 2
        c = _corr(merged["universality_F"].values, merged["encoding"].values)
        rows.append({"section": "4.2", "statistic": "univ_vs_encoding_mean", **c})

    return rows


def collect_accuracy_vs_encoding() -> list[dict]:
    """Pre-computed accuracy/universality/encoding relationships."""
    path = Path(
        "experiments/universality/downstream_predictions/macaque_encoding/encoding_things/outputs"
        "/accuracy_vs_encoding_results.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        monkey = row["monkey"]
        features = row["features"]
        subset = row["subset"]
        comparison = row["comparison"]
        key = f"{monkey}_{features}_{subset}_{comparison}"
        entry = {
            "section": "4.2",
            "statistic": key,
            "n": int(row["n"]),
            "pearson_r": row["pearson_r"],
            "pearson_p": row["pearson_p"],
        }
        if pd.notna(row.get("spearman_rho")):
            entry["spearman_rho"] = row["spearman_rho"]
            entry["spearman_p"] = row["spearman_p"]
        rows.append(entry)
    return rows


def collect_snmf_vs_raw_ranking() -> list[dict]:
    """Spearman correlation of SNMF vs raw encoding rankings per monkey."""
    encoding_dir = Path(
        "experiments/universality/downstream_predictions/macaque_encoding/encoding_things/outputs"
    )
    rows = []
    for monkey in ("F", "N"):
        path = encoding_dir / f"snmf_vs_raw_ranking_{monkey}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sr, sp = sp_stats.spearmanr(df["rank_snmf"], df["rank_raw"])
        rows.append({
            "section": "validation",
            "statistic": f"snmf_vs_raw_ranking_{monkey}",
            "n": len(df),
            "spearman_rho": sr, "spearman_p": sp,
        })
    return rows


def collect_within_contrast_correlations(rank: int) -> list[dict]:
    """Within-contrast universality vs encoding correlations."""
    path = (
        Path("experiments/universality/controlled_comparisons/contrasts/outputs")
        / f"rank_{rank}"
        / "within_contrast_correlations.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "section": "4.2",
            "statistic": f"within_contrast_{row['contrast']}",
            "n": int(row["n"]),
            "pearson_r": row["pearson_r"],
            "pearson_p": row["pearson_p"],
            "spearman_rho": row["spearman_rho"],
            "spearman_p": row["spearman_p"],
        })
    return rows


def collect_effect_sizes() -> list[dict]:
    """Canonical universality tests and effect sizes across contrasts."""
    path = Path(
        "experiments/universality/controlled_comparisons/contrasts/uniform_tests/outputs"
        "/contrast_results.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "section": "4.1",
            "statistic": f"effect_univ_{row['contrast']}",
            "n": int(row["n_total"]),
            "n_groups": int(row["n_groups"]),
            "test": row["test"],
            "effect_size": row["effect_size"],
            "effect_size_type": row["effect_size_type"],
            "effect_ci_low": row["effect_ci_low"],
            "effect_ci_high": row["effect_ci_high"],
            "p_value": row["p_value"],
            "p_bonferroni": row["p_bonferroni"],
            "p_bonferroni_all_prespecified": row.get(
                "p_bonferroni_all_prespecified", np.nan
            ),
            "p_bonferroni_testable": row.get(
                "p_bonferroni_testable", np.nan
            ),
            "p_bh_fdr": row["p_bh_fdr"],
            "omega_squared": row.get("omega_squared", np.nan),
            "omega_ci_low": row.get("omega_ci_low", np.nan),
            "omega_ci_high": row.get("omega_ci_high", np.nan),
        })
    return rows


def collect_clip_vit_convergence() -> list[dict]:
    """Paper's random-set bootstrap for the seven CLIP ViT-L/14 models."""
    path = Path(
        "experiments/universality/controlled_comparisons/contrasts/uniform_tests/outputs"
        "/clip_vit_convergence.json"
    )
    if not path.exists():
        return []
    result = pd.read_json(path, typ="series")
    return [{
        "section": "4.1",
        "statistic": "clip_vit_training_dataset_convergence",
        "n": int(result["n_group"]),
        "null_pool_size": int(result["null_pool_size"]),
        "observed_sd": float(result["observed_sd"]),
        "p_value": float(result["p_value_lower_tail"]),
        "n_resamples": int(result["n_resamples"]),
    }]


def collect_variance_partitioning() -> list[dict]:
    """Variance partitioning from multiple regression on universality."""
    path = Path(
        "experiments/universality/controlled_comparisons/regression/outputs"
        "/variance_partitioning.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "section": "4.1",
            "statistic": f"variance_{row['predictor_group'].lower().replace(' / ', '_').replace(' ', '_')}",
            "r2_alone": row["r2_alone"],
            "r2_incremental": row["r2_incremental"],
            "r2_shared": row["r2_shared"],
        })
    return rows


def collect_behavior_correlations() -> list[dict]:
    """Universality vs human triplet accuracy."""
    path = Path(
        "experiments/universality/downstream_predictions/things_behavior/triplets/outputs"
        "/triplet_results.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []

    c = _corr(df["universality"].values, df["triplet_accuracy"].values)
    rows.append({"section": "4.3", "statistic": "univ_vs_behavior_triplet", **c})

    if "triplet_accuracy_raw" in df.columns:
        c = _corr(df["universality"].values, df["triplet_accuracy_raw"].values)
        rows.append({"section": "4.3", "statistic": "univ_vs_behavior_triplet_raw", **c})

    return rows


def collect_behavior_dimension_subsets() -> list[dict]:
    """Universal vs specific dimension subsets for behavior prediction."""
    path = Path(
        "experiments/universality/downstream_predictions/things_behavior/dimension_prediction"
        "/outputs/universal_vs_specific_subsets.csv"
    )
    if not path.exists():
        return []

    df = pd.read_csv(path)
    univ_mean = df["acc_universal_half"].mean()
    spec_mean = df["acc_specific_half"].mean()
    full_mean = df["acc_full"].mean()

    t_stat, p_val = sp_stats.ttest_rel(df["acc_universal_half"], df["acc_specific_half"])
    d = (df["acc_universal_half"] - df["acc_specific_half"]).mean() / (
        df["acc_universal_half"] - df["acc_specific_half"]).std()

    return [{
        "section": "4.3",
        "statistic": "behavior_universal_vs_specific_halves",
        "n": len(df),
        "universal_half_mean": univ_mean,
        "specific_half_mean": spec_mean,
        "full_mean": full_mean,
        "paired_t": t_stat,
        "p_value": p_val,
        "cohens_d": d,
    }]


def collect_cross_dataset_stability(rank: int) -> list[dict]:
    """Cross-dataset (THINGS vs ObjectNet) model ranking stability."""
    rows = []
    try:
        things = load_universality_by_model(rank, dataset="things22k")
        objectnet = load_universality_by_model(rank, dataset="objectnet")
    except FileNotFoundError as e:
        print(f"  [SKIP] Cross-dataset: {e}")
        return rows

    merged = things.merge(objectnet, on="model", suffixes=("_things", "_objectnet"))
    sr, sp = sp_stats.spearmanr(merged["mean_things"], merged["mean_objectnet"])
    pr, pp = sp_stats.pearsonr(merged["mean_things"], merged["mean_objectnet"])
    rows.append({
        "section": "validation",
        "statistic": "cross_dataset_things_vs_objectnet",
        "n": len(merged),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
    })
    return rows


def collect_rank_stability(rank: int, dataset: str) -> list[dict]:
    """Cross-rank stability of model universality rankings."""
    rows = []
    rank_dfs = {}
    for r in (10, 30, 50, 100, 200):
        try:
            rank_dfs[r] = load_universality_by_model(r, dataset=dataset)
        except FileNotFoundError:
            pass

    if rank not in rank_dfs or len(rank_dfs) < 2:
        return rows

    ref = rank_dfs[rank]
    for r, df in rank_dfs.items():
        if r == rank:
            continue
        merged = ref.merge(df, on="model", suffixes=(f"_{rank}", f"_{r}"))
        sr, sp = sp_stats.spearmanr(merged[f"mean_{rank}"], merged[f"mean_{r}"])
        rows.append({
            "section": "validation",
            "statistic": f"rank_stability_{rank}_vs_{r}",
            "n": len(merged),
            "spearman_rho": sr, "spearman_p": sp,
        })
    return rows


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    rank = cfg.factorization.rank

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    all_rows: list[dict] = []
    collectors = [
        ("Encoding correlations (overall)", collect_encoding_correlations),
        ("Accuracy vs encoding", collect_accuracy_vs_encoding),
        ("SNMF vs raw ranking", collect_snmf_vs_raw_ranking),
        ("Within-contrast correlations", lambda: collect_within_contrast_correlations(rank)),
        ("Contrast effect sizes", collect_effect_sizes),
        ("CLIP ViT-L/14 convergence", collect_clip_vit_convergence),
        ("Variance partitioning", collect_variance_partitioning),
        ("Behavior correlations", collect_behavior_correlations),
        ("Behavior dimension subsets", collect_behavior_dimension_subsets),
        ("Cross-dataset stability", lambda: collect_cross_dataset_stability(rank)),
        ("Rank stability", lambda: collect_rank_stability(rank, cfg.dataset)),
    ]

    for name, fn in collectors:
        print(f"\n{name}")
        rows = fn()
        print(f"  {len(rows)} statistics collected")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = out / "paper_statistics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} statistics to {out_path}")

    print_report(df)


def print_report(df: pd.DataFrame) -> None:
    """Print a structured summary of key paper statistics."""
    if df.empty:
        print("\nNo statistics collected; run the upstream experiments first.")
        return

    print("\n" + "=" * 70)
    print("PAPER STATISTICS REPORT")
    print("=" * 70)

    print("\n-- Universality vs Encoding (all models) --")
    for _, row in df[df["statistic"].str.startswith("univ_vs_encoding_monkey")].iterrows():
        print(f"  {row['statistic']}: r={row['pearson_r']:.3f}, "
              f"rho={row.get('spearman_rho', float('nan')):.3f}, n={row['n']}")
    mean_row = df[df["statistic"] == "univ_vs_encoding_mean"]
    if not mean_row.empty:
        r = mean_row.iloc[0]
        print(f"  Mean of monkeys: r={r['pearson_r']:.3f}, n={r['n']}")

    print("\n-- Within-Contrast Correlations (univ vs encoding) --")
    within = df[df["statistic"].str.startswith("within_contrast_")]
    for _, row in within.iterrows():
        label = row["statistic"].replace("within_contrast_compare_", "")
        sig = "*" if row["pearson_p"] < 0.05 else " "
        print(f"  {sig} {label:30s} r={row['pearson_r']:.3f}  "
              f"p={row['pearson_p']:.1e}  n={row['n']}")

    print("\n-- Controlled Comparison Effect Sizes --")
    effects = df[df["statistic"].str.startswith("effect_")]
    for _, row in effects.iterrows():
        label = row["statistic"].replace("effect_univ_compare_", "")
        print(
            f"  {label:35s} universality={row['effect_size']:.2f} "
            f"[{row['effect_ci_low']:.2f}, {row['effect_ci_high']:.2f}] "
            f"({row.get('effect_size_type', '')}), "
            f"p_Bonf={row['p_bonferroni']:.2g}"
        )

    print("\n-- Variance Partitioning (universality) --")
    vp = df[df["statistic"].str.startswith("variance_")]
    for _, row in vp.iterrows():
        label = row["statistic"].replace("variance_", "")
        print(f"  {label:25s} R2_alone={row['r2_alone']:.3f}  "
              f"R2_incr={row['r2_incremental']:.3f}")

    print("\n-- Behavior --")
    beh = df[df["statistic"].str.startswith("univ_vs_behavior")]
    for _, row in beh.iterrows():
        print(f"  {row['statistic']}: r={row['pearson_r']:.3f}, "
              f"rho={row.get('spearman_rho', float('nan')):.3f}, n={row['n']}")
    subsets = df[df["statistic"] == "behavior_universal_vs_specific_halves"]
    if not subsets.empty:
        s = subsets.iloc[0]
        print(f"  Universal half: {s['universal_half_mean']:.4f}, "
              f"Specific half: {s['specific_half_mean']:.4f}, "
              f"d={s['cohens_d']:.2f}, p={s['p_value']:.1e}")

    print("\n-- Validation --")
    val = df[df["section"] == "validation"]
    for _, row in val.iterrows():
        rho = row.get("spearman_rho", float("nan"))
        r = row.get("pearson_r", float("nan"))
        parts = []
        if pd.notna(r):
            parts.append(f"r={r:.3f}")
        if pd.notna(rho):
            parts.append(f"rho={rho:.3f}")
        parts.append(f"n={row['n']}")
        print(f"  {row['statistic']:45s} {', '.join(parts)}")


if __name__ == "__main__":
    main()
