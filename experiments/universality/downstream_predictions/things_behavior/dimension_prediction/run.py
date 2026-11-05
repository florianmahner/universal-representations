from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from src.colors import GRAY_DARK, INDIGO, PURPLE, ROSE
from src.data import ThingsDataset
from src.plotting import setup_plotting, plt

log = logging.getLogger(__name__)

# THINGS-22k stores twelve images for each of the 1854 concepts that the
# behavioural triplet task is defined over.
N_IMAGES_PER_CATEGORY = 12
N_CATEGORIES = 1854


def load_triplets(data_root: Path) -> np.ndarray:
    """Load train + validation triplets as (n_triplets, 3) int array."""
    parts = []
    for name in ("trainset.txt", "validationset.txt"):
        t = np.loadtxt(data_root / "behavior" / "things_triplets" / name, dtype=int)
        parts.append(t)
    triplets = np.concatenate(parts, axis=0)
    log.info(f"Loaded {len(triplets)} triplets")
    return triplets


def get_exemplary_indices() -> np.ndarray:
    """Return indices of the first image per category in the 22k array."""
    return np.arange(N_CATEGORIES) * N_IMAGES_PER_CATEGORY


def triplet_accuracy_from_sim(s: np.ndarray, triplets: np.ndarray) -> float:
    """Compute odd-one-out accuracy from a precomputed similarity matrix."""
    ii, jj, kk = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    s_ij = s[ii, jj]
    s_ik = s[ii, kk]
    s_jk = s[jj, kk]
    correct = (s_ij > s_ik) & (s_ij > s_jk)
    return float(correct.mean())


def ablation_single_model(
    model_name: str,
    dataset: ThingsDataset,
    rank: int,
    exemplary_idx: np.ndarray,
    triplets: np.ndarray,
) -> list[dict]:
    """Compute full and per-dimension ablated triplet accuracy for one model."""
    w = dataset.load_components(model_name, rank=rank)
    w_ex = w[exemplary_idx]

    s_full = cosine_similarity(w_ex)
    acc_full = triplet_accuracy_from_sim(s_full, triplets)

    results = []
    for d in range(rank):
        w_abl = w_ex.copy()
        w_abl[:, d] = 0.0
        s_abl = cosine_similarity(w_abl)
        acc_abl = triplet_accuracy_from_sim(s_abl, triplets)
        results.append({
            "model": model_name,
            "dimension": d,
            "acc_full": acc_full,
            "acc_ablated": acc_abl,
            "importance": acc_full - acc_abl,
        })
    return results


def subset_accuracy_single_model(
    model_name: str,
    dataset: ThingsDataset,
    rank: int,
    exemplary_idx: np.ndarray,
    triplets: np.ndarray,
    uni_per_dim: pd.DataFrame,
) -> dict:
    """Compute triplet accuracy from universal vs specific dimension subsets."""
    w = dataset.load_components(model_name, rank=rank)
    w_ex = w[exemplary_idx]

    model_uni = uni_per_dim[uni_per_dim["model"] == model_name].sort_values("dimension")
    if len(model_uni) != rank:
        return {}

    uni_order = model_uni.sort_values("universality", ascending=False)["dimension"].values
    top_half = uni_order[: rank // 2]
    bottom_half = uni_order[rank // 2 :]

    s_full = cosine_similarity(w_ex)
    acc_full = triplet_accuracy_from_sim(s_full, triplets)

    w_top = np.zeros_like(w_ex)
    w_top[:, top_half] = w_ex[:, top_half]
    s_top = cosine_similarity(w_top)
    acc_top = triplet_accuracy_from_sim(s_top, triplets)

    w_bot = np.zeros_like(w_ex)
    w_bot[:, bottom_half] = w_ex[:, bottom_half]
    s_bot = cosine_similarity(w_bot)
    acc_bot = triplet_accuracy_from_sim(s_bot, triplets)

    return {
        "model": model_name,
        "acc_full": acc_full,
        "acc_universal_half": acc_top,
        "acc_specific_half": acc_bot,
    }


def cumulative_accuracy_single_model(
    model_name: str,
    dataset: ThingsDataset,
    rank: int,
    exemplary_idx: np.ndarray,
    triplets: np.ndarray,
    uni_per_dim: pd.DataFrame,
    seed: int,
    n_random: int = 5,
) -> list[dict]:
    """Compute cumulative accuracy curves ordered by universality and random."""
    w = dataset.load_components(model_name, rank=rank)
    w_ex = w[exemplary_idx]

    model_uni = uni_per_dim[uni_per_dim["model"] == model_name].sort_values("dimension")
    if len(model_uni) != rank:
        return []

    uni_order = model_uni.sort_values("universality", ascending=False)["dimension"].values

    results = []
    for k in range(1, rank + 1):
        dims_uni = uni_order[:k]
        w_sub = np.zeros_like(w_ex)
        w_sub[:, dims_uni] = w_ex[:, dims_uni]
        s_sub = cosine_similarity(w_sub)
        acc_uni = triplet_accuracy_from_sim(s_sub, triplets)

        acc_random_runs = []
        rng = np.random.RandomState(seed)
        for _ in range(n_random):
            dims_rand = rng.choice(rank, k, replace=False)
            w_rand = np.zeros_like(w_ex)
            w_rand[:, dims_rand] = w_ex[:, dims_rand]
            s_rand = cosine_similarity(w_rand)
            acc_random_runs.append(triplet_accuracy_from_sim(s_rand, triplets))

        results.append({
            "model": model_name,
            "k": k,
            "acc_universality_order": acc_uni,
            "acc_random_mean": np.mean(acc_random_runs),
            "acc_random_std": np.std(acc_random_runs),
        })

    return results


def load_universality(dataset: str, rank: int) -> pd.DataFrame:
    """Load per-dimension universality scores written by the metric experiment."""
    path = (
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    univ = pd.read_csv(path)
    return univ[univ["metric"] == "cos2_perm_adj"].reset_index(drop=True)


def plot_importance_vs_universality(
    ablation_df: pd.DataFrame,
    uni_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Scatter: dimension universality vs dimension importance (ablation)."""
    setup_plotting()

    merged = ablation_df.merge(
        uni_df[["model", "dimension", "universality"]],
        on=["model", "dimension"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(
        merged["universality"], merged["importance"],
        alpha=0.05, s=5, c=ROSE, rasterized=True,
    )
    mean_by_dim = merged.groupby("dimension")[["universality", "importance"]].mean()
    r, p = stats.pearsonr(mean_by_dim["universality"], mean_by_dim["importance"])
    ax.scatter(
        mean_by_dim["universality"], mean_by_dim["importance"],
        s=40, c=PURPLE, edgecolors="white", linewidth=0.5,
        zorder=3, label=f"Dimension means (r={r:.2f}, p={p:.1e})",
    )
    z = np.polyfit(mean_by_dim["universality"], mean_by_dim["importance"], 1)
    x_line = np.linspace(mean_by_dim["universality"].min(), mean_by_dim["universality"].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY_DARK, lw=1.5)
    ax.set_xlabel("Dimension universality")
    ax.set_ylabel("Dimension importance (accuracy drop)")
    ax.legend(fontsize=8)

    ax = axes[1]
    per_model_r = []
    for model_name, grp in merged.groupby("model"):
        if len(grp) > 5:
            r_m, _ = stats.pearsonr(grp["universality"], grp["importance"])
            per_model_r.append(r_m)
    ax.hist(per_model_r, bins=25, color=ROSE, edgecolor="white", alpha=0.8)
    mean_r = np.mean(per_model_r)
    ax.axvline(mean_r, color=PURPLE, lw=2, ls="--",
               label=f"Mean r={mean_r:.3f}")
    t_stat, t_p = stats.ttest_1samp(per_model_r, 0)
    ax.set_xlabel("Within-model r(universality, importance)")
    ax.set_ylabel("Count")
    ax.set_title(f"t={t_stat:.1f}, p={t_p:.1e}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "dimension_importance_vs_universality.pdf")
    plt.close(fig)
    log.info("Saved dimension_importance_vs_universality.pdf")


def plot_subset_comparison(subset_df: pd.DataFrame, output_dir: Path) -> None:
    """Paired comparison: universal half vs specific half triplet accuracy.

    For each model, the 50 NMF dimensions are split by universality into
    the top-25 (universal) and bottom-25 (specific). Triplet accuracy is
    computed from each half independently. This tests whether the shared
    representational structure drives behavioral prediction.
    """
    setup_plotting()

    sub = subset_df.dropna()
    t_stat, t_p = stats.ttest_rel(sub["acc_universal_half"], sub["acc_specific_half"])
    diff_mean = (sub["acc_universal_half"] - sub["acc_specific_half"]).mean()

    melted = pd.melt(
        sub, id_vars=["model"], value_vars=["acc_universal_half", "acc_specific_half"],
        var_name="subset", value_name="triplet_accuracy",
    )
    melted["subset"] = melted["subset"].map({
        "acc_universal_half": "Universal\n(top 25 dims)",
        "acc_specific_half": "Specific\n(bottom 25 dims)",
    })

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(
        data=melted, x="subset", y="triplet_accuracy", ax=ax,
        width=0.5, palette=[ROSE, INDIGO],
        fliersize=2, linewidth=1,
    )
    sns.stripplot(
        data=melted, x="subset", y="triplet_accuracy", ax=ax,
        size=3, alpha=0.3, color=GRAY_DARK, jitter=0.15,
    )
    for model in sub["model"]:
        ax.plot(
            [0, 1],
            [sub.loc[sub["model"] == model, "acc_universal_half"].values[0],
             sub.loc[sub["model"] == model, "acc_specific_half"].values[0]],
            color="gray", alpha=0.06, lw=0.5,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Triplet accuracy")
    ax.set_title(
        f"Universal dims predict better\n"
        f"(+{diff_mean:.3f}, paired t={t_stat:.1f}, p={t_p:.1e})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "universal_vs_specific_dimensions.pdf")
    plt.close(fig)
    log.info("Saved universal_vs_specific_dimensions.pdf")


def plot_cumulative_curve(cum_df: pd.DataFrame, output_dir: Path) -> None:
    """Cumulative accuracy curve: universality order vs random order."""
    setup_plotting()

    agg = cum_df.groupby("k").agg(
        uni_mean=("acc_universality_order", "mean"),
        uni_se=("acc_universality_order", "sem"),
        rand_mean=("acc_random_mean", "mean"),
        rand_se=("acc_random_mean", "sem"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.fill_between(
        agg["k"],
        agg["uni_mean"] - agg["uni_se"],
        agg["uni_mean"] + agg["uni_se"],
        alpha=0.15, color=ROSE,
    )
    ax.plot(agg["k"], agg["uni_mean"], "-", color=ROSE, lw=2,
            label="Universality order")

    ax.fill_between(
        agg["k"],
        agg["rand_mean"] - agg["rand_se"],
        agg["rand_mean"] + agg["rand_se"],
        alpha=0.15, color="gray",
    )
    ax.plot(agg["k"], agg["rand_mean"], "-", color="gray", lw=2,
            label="Random order")

    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Triplet accuracy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_accuracy_by_universality.pdf")
    plt.close(fig)
    log.info("Saved cumulative_accuracy_by_universality.pdf")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    things = ThingsDataset()
    uni_per_dim = load_universality(cfg.dataset, rank)

    ablation_csv = out / "dimension_ablation.csv"
    subset_csv = out / "universal_vs_specific_subsets.csv"
    cumulative_csv = out / "cumulative_by_universality_order.csv"

    triplets = load_triplets(Path(cfg.data_root))
    exemplary_idx = get_exemplary_indices()
    models = list(cfg.models) or [
        m for m in things.list_component_models() if not m.startswith("macaque")
    ]

    log.info(f"Running ablation analysis for {len(models)} models × {rank} dimensions...")
    ablation_results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(ablation_single_model)(m, things, rank, exemplary_idx, triplets)
        for m in models
    )
    ablation_df = pd.DataFrame([r for batch in ablation_results for r in batch])
    ablation_df.to_csv(ablation_csv, index=False)
    log.info(f"Saved {ablation_csv}")

    log.info("Running subset comparison...")
    subset_results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(subset_accuracy_single_model)(
            m, things, rank, exemplary_idx, triplets, uni_per_dim,
        )
        for m in models
    )
    subset_df = pd.DataFrame([r for r in subset_results if r])
    subset_df.to_csv(subset_csv, index=False)
    log.info(f"Saved {subset_csv}")

    log.info("Running cumulative accuracy curves...")
    cum_results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(cumulative_accuracy_single_model)(
            m, things, rank, exemplary_idx, triplets, uni_per_dim, cfg.seed,
        )
        for m in models
    )
    cum_df = pd.DataFrame([r for batch in cum_results for r in batch])
    cum_df.to_csv(cumulative_csv, index=False)
    log.info(f"Saved {cumulative_csv}")

    ablation_df = pd.read_csv(ablation_csv)
    subset_df = pd.read_csv(subset_csv)
    cum_df = pd.read_csv(cumulative_csv)

    mean_imp = ablation_df.groupby("model")["importance"].mean()
    log.info(f"Mean ablation importance: {mean_imp.mean():.4f} (std={mean_imp.std():.4f})")

    merged_abl = ablation_df.merge(
        uni_per_dim[["model", "dimension", "universality"]],
        on=["model", "dimension"],
    )
    mean_by_dim = merged_abl.groupby("dimension")[["universality", "importance"]].mean()
    r, p = stats.pearsonr(mean_by_dim["universality"], mean_by_dim["importance"])
    log.info(f"Dimension-level: universality vs importance r={r:.3f}, p={p:.1e}")

    sub = subset_df.dropna()
    t_stat, t_p = stats.ttest_rel(sub["acc_universal_half"], sub["acc_specific_half"])
    diff = (sub["acc_universal_half"] - sub["acc_specific_half"]).mean()
    log.info(f"Universal half vs specific half: diff={diff:.4f}, t={t_stat:.2f}, p={t_p:.1e}")

    plot_importance_vs_universality(ablation_df, uni_per_dim, out)
    plot_subset_comparison(subset_df, out)
    plot_cumulative_curve(cum_df, out)


if __name__ == "__main__":
    main()
