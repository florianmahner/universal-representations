from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from scipy.stats import f_oneway, spearmanr

from experiments.universality.shared import load_components_for_rank, load_universality
from src.data import ThingsDataset
from src.plotting import setup_plotting, save_figure
from src.colors import ROSE, TEAL, GRAY

from .compute import compute_category_bias

MIN_N = 10  # minimum basic-level categories per superordinate to include


def load_superordinate_map(data_root: Path = Path("data")) -> pd.DataFrame:
    things = pd.read_csv(
        Path(data_root) / "meta" / "things22k" / "concepts-metadata_things.tsv",
        sep="\t",
        usecols=["Word", "Bottom-up Category (Human Raters)"],
    )
    things.columns = ["category", "superordinate"]
    things["category_lower"] = things["category"].str.lower().str.replace(" ", "_")
    return things


def compute_superordinate_eta2(
    components: dict[str, np.ndarray],
    universality: pd.DataFrame,
    image_superordinates: np.ndarray,
) -> pd.DataFrame:
    """Compute eta-squared using superordinate labels as the grouping variable.

    Same formula as basic-level eta^2, but SS_between is computed across
    superordinate group means rather than basic-level category means.
    Returns per-dimension DataFrame with columns: model, dimension, universality, eta2_super.
    """
    unique_super = np.array(sorted(s for s in set(image_superordinates) if s is not None and str(s) != "nan"))
    super_to_idx = {s: i for i, s in enumerate(unique_super)}
    valid_mask = np.array([str(s) != "nan" and s is not None for s in image_superordinates])
    super_labels = np.array([super_to_idx.get(s, -1) for s in image_superordinates])
    n_super = len(unique_super)
    super_sizes = np.bincount(super_labels[valid_mask], minlength=n_super).astype(float)

    univ_lookup = {(row["model"], int(row["dimension"])): row["universality"]
                   for _, row in universality.iterrows()}

    records = []
    for model, W in components.items():
        W_valid = W[valid_mask]
        sl = super_labels[valid_mask]

        grand_means = W_valid.mean(axis=0)
        ss_total = np.sum((W_valid - grand_means) ** 2, axis=0)

        super_sums = np.zeros((n_super, W_valid.shape[1]))
        for s in range(n_super):
            mask = sl == s
            if mask.any():
                super_sums[s] = W_valid[mask].sum(axis=0)
        super_means = super_sums / np.maximum(super_sizes[:, None], 1)
        ss_between = np.sum(super_sizes[:, None] * (super_means - grand_means) ** 2, axis=0)
        eta2 = ss_between / (ss_total + 1e-12)

        for d in range(W.shape[1]):
            records.append({
                "model": model,
                "dimension": d,
                "universality": univ_lookup.get((model, d), np.nan),
                "eta2_super": float(eta2[d]),
            })

    return pd.DataFrame(records)


def compute_within_category_variance(
    components: dict[str, np.ndarray],
    categories: np.ndarray,
) -> pd.DataFrame:
    """For each basic-level category, compute mean within-exemplar and between-category loading variance.

    within_var: mean variance across exemplars within each basic-level category (averaged over dims/models)
    between_var: mean variance of category means within each superordinate — computed at superordinate level later.

    Returns DataFrame with columns: category, mean_within_var, mean_cat_loading (averaged over dims/models).
    """
    unique_cats = np.array(sorted(set(categories)))
    n_cats = len(unique_cats)
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    cat_labels = np.array([cat_to_idx[c] for c in categories])

    within_var_accum = np.zeros(n_cats)
    cat_mean_accum = np.zeros(n_cats)
    n_models = 0

    for W in components.values():
        for c in range(n_cats):
            mask = cat_labels == c
            exemplars = W[mask]  # (n_exemplars, rank)
            within_var_accum[c] += exemplars.var(axis=0).mean()
            cat_mean_accum[c] += exemplars.mean(axis=0).mean()
        n_models += 1

    return pd.DataFrame({
        "category": unique_cats,
        "mean_within_var": within_var_accum / n_models,
        "mean_cat_loading": cat_mean_accum / n_models,
    })


def make_figure(
    bias_super: pd.DataFrame,
    eta2_super: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Two-panel figure: universality bias per superordinate + superordinate eta^2 by decile."""
    setup_plotting()

    # Sort by universality bias
    order = bias_super.sort_values("bias_mean").index
    bias_sorted = bias_super.loc[order]

    n = len(bias_sorted)
    y = np.arange(n)
    labels = bias_sorted.index.tolist()

    median_rho = bias_sorted["bias_mean"].median()
    colors_list = [TEAL if r > median_rho else ROSE for r in bias_sorted["bias_mean"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    fig.subplots_adjust(wspace=0.35)

    # Panel A: universality bias dot plot
    ax = axes[0]
    ax.axvline(bias_sorted["bias_mean"].mean(), color=GRAY, lw=0.8, ls="--", zorder=0)
    ax.errorbar(
        bias_sorted["bias_mean"], y,
        xerr=bias_sorted["bias_se"],
        fmt="none", color=GRAY, lw=0.8, capsize=2, zorder=1,
    )
    ax.scatter(bias_sorted["bias_mean"], y, color=colors_list, s=28, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Universality bias\n(universal − specific mean loading)", fontsize=8)
    ax.set_title("Universality bias by superordinate", fontsize=8, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.5)

    # Panel B: superordinate eta^2 by universality decile (replicating Fig 4a at superordinate level)
    ax2 = axes[1]
    decile_col = pd.qcut(eta2_super["universality"], 10, labels=False)
    eta2_super_valid = eta2_super.assign(decile=decile_col).dropna(subset=["universality", "eta2_super"])
    decile_means = eta2_super_valid.groupby("decile")["eta2_super"].mean()
    decile_se = eta2_super_valid.groupby("decile")["eta2_super"].sem()
    x_dec = decile_means.index.values

    ax2.fill_between(x_dec, (decile_means - decile_se).values, (decile_means + decile_se).values,
                     alpha=0.2, color=TEAL)
    ax2.plot(x_dec, decile_means.values, color=TEAL, lw=1.5, marker="o", ms=4)
    ax2.set_xlabel("Universality decile", fontsize=8)
    ax2.set_ylabel("Superordinate η²", fontsize=8)
    ax2.set_title("Superordinate category selectivity\nvs universality", fontsize=8, pad=6)
    ax2.set_xticks([0, 4, 9])
    ax2.set_xticklabels(["low", "mid", "high"], fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=7.5)

    save_figure(fig, output_dir / "figures" / "superordinate_analysis")
    print(f"Saved to {output_dir / 'figures' / 'superordinate_analysis'}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    print("Loading data...")
    ds = ThingsDataset()
    components = load_components_for_rank(rank=cfg.factorization.rank, dataset=cfg.dataset)

    # Get category labels for 22k images
    n_items = next(iter(components.values())).shape[0]
    categories = np.array(ds.all_categories if n_items == len(ds.all_categories) else ds.categories)
    print(f"  {len(components)} models, {n_items} items, {len(np.unique(categories))} categories")

    # Compute category bias with balanced split measure
    univ = load_universality(rank=cfg.factorization.rank, dataset=cfg.dataset)
    print("Computing category bias (balanced split)...")
    bias = compute_category_bias(components, univ, categories)
    bias.to_csv(out / "category_bias.csv", index=False)
    bias["category_lower"] = bias["category"].str.lower().str.replace(" ", "_")

    # Load superordinate map
    things = load_superordinate_map(Path(cfg.data_root))
    bias = bias.merge(things[["category_lower", "superordinate"]], on="category_lower", how="left")

    # Compute within-category variance
    print("Computing within-category variance (all models)...")
    within_var = compute_within_category_variance(components, categories)
    within_var["category_lower"] = within_var["category"].str.lower().str.replace(" ", "_")
    within_var = within_var.merge(things[["category_lower", "superordinate"]], on="category_lower", how="left")
    within_var.to_csv(out / "within_category_variance.csv", index=False)
    print(f"  Saved {out / 'within_category_variance.csv'}")

    # Aggregate to superordinate level (n >= MIN_N)
    def agg_superordinate(df, val_col, prefix, super_col="superordinate") -> pd.DataFrame:
        grp = df.dropna(subset=[super_col]).groupby(super_col)[val_col]
        agg = grp.agg(["mean", "std", "count"])
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        agg = agg[agg["count"] >= MIN_N]
        agg.columns = [f"{prefix}_mean", f"{prefix}_std", "n", f"{prefix}_se"]
        return agg

    bias_super = agg_superordinate(bias, "universality_bias", "bias")
    within_agg = agg_superordinate(within_var, "mean_within_var", "var_within")

    # Between-category variance per superordinate:
    # variance of basic-level category mean loadings within each superordinate
    within_var_super = within_var.dropna(subset=["superordinate"])
    between_records = []
    for sup, grp in within_var_super.groupby("superordinate"):
        if len(grp) >= MIN_N:
            between_records.append({
                "superordinate": sup,
                "var_between_mean": grp["mean_cat_loading"].var(),
                "n": len(grp),
            })
    between_agg = pd.DataFrame(between_records).set_index("superordinate")

    # Merge within and between into one var_super
    var_super = within_agg.join(between_agg[["var_between_mean"]], how="inner")

    # Align on shared superordinates
    shared = bias_super.index.intersection(var_super.index)
    bias_super = bias_super.loc[shared]
    var_super = var_super.loc[shared]

    # Stats
    rho, p = spearmanr(bias_super["bias_mean"], var_super["var_within_mean"])
    print(f"\nSpearman correlation (universality bias vs within-cat variance): rho={rho:.3f}, p={p:.3e}")

    groups_bias = [g["universality_bias"].values for _, g in
                   bias.dropna(subset=["superordinate"]).groupby("superordinate")
                   if len(g) >= MIN_N]
    f, p_anova = f_oneway(*groups_bias)
    print(f"ANOVA (universality bias across superordinates): F={f:.2f}, p={p_anova:.2e}")

    groups_var = [g["mean_within_var"].values for _, g in
                  within_var.dropna(subset=["superordinate"]).groupby("superordinate")
                  if len(g) >= MIN_N]
    f2, p2_anova = f_oneway(*groups_var)
    print(f"ANOVA (within-cat variance across superordinates): F={f2:.2f}, p={p2_anova:.2e}")

    print("\nTop 5 superordinates by universality bias (most universal):")
    print(bias_super.sort_values("bias_mean", ascending=False).head(5)[["bias_mean", "n"]])
    print("\nBottom 5 (most model-specific):")
    print(bias_super.sort_values("bias_mean").head(5)[["bias_mean", "n"]])

    # Superordinate-level eta^2: assign superordinate label to each of the 22k images
    cat_to_super = dict(zip(things["category_lower"], things["superordinate"]))
    image_superordinates = np.array([cat_to_super.get(c.lower().replace(" ", "_"), None) for c in categories])
    n_labeled = sum(s is not None for s in image_superordinates)
    print(f"Computing superordinate eta^2 ({n_labeled} / {len(image_superordinates)} images with superordinate label)...")
    eta2_super = compute_superordinate_eta2(components, univ, image_superordinates)
    eta2_super.to_csv(out / "superordinate_eta2.csv", index=False)

    rho_super, p_super = spearmanr(eta2_super["universality"].dropna(),
                                    eta2_super["eta2_super"][eta2_super["universality"].notna()])
    print(f"Superordinate eta^2 ~ universality: rho={rho_super:.3f}, p={p_super:.2e}")

    print("\nGenerating figure...")
    make_figure(bias_super, eta2_super, out)


if __name__ == "__main__":
    main()
