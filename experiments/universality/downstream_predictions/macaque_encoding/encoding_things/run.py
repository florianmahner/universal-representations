import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold

from src.data import ThingsDataset
from src.colors import GRAY, GRAY_LIGHT, INDIGO
from src.plotting import (
    CLASS_COLORS,
    create_figure,
    plt,
    save_figure,
    setup_plotting,
)

N_FOLDS = 5
RELIAB_THRESHOLD = 0.3

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_macaque_it(
    monkey: str,
    data_root: Path | str = Path("data"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load macaque IT MUA, filtered to reliable neurons.

    Returns (n_images, n_neurons) MUA and (n_neurons,) mean reliability.
    """
    path = Path(data_root) / "features" / "macaque" / "22k" / "processed" / f"{monkey}.npy"
    d = np.load(path, allow_pickle=True).item()
    mua = d["train_MUA"]
    reliab = d["reliab"]
    it_start, it_end = d["roi_slices"]["it"]
    it_mua = mua[it_start:it_end]
    it_reliab = reliab[it_start:it_end]
    mean_reliab = it_reliab.mean(axis=1)
    mask = mean_reliab > RELIAB_THRESHOLD
    log.info(f"Monkey {monkey} IT: {mask.sum()}/{len(mask)} neurons with reliab > {RELIAB_THRESHOLD}")
    return it_mua[mask].T, mean_reliab[mask]


def encode_single_model(
    features: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    """Ridge regression encoding for one model. Returns per-neuron Pearson r."""
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)
    predictions = np.zeros_like(y)

    for train_idx, test_idx in cv.split(features):
        # The manuscript selects regularization independently for each neuron.
        # RidgeCV otherwise chooses one shared alpha for a multi-output target.
        ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
        ridge.fit(features[train_idx], y[train_idx])
        predictions[test_idx] = ridge.predict(features[test_idx])

    return np.array([
        stats.pearsonr(y[:, j], predictions[:, j]).statistic
        for j in range(y.shape[1])
    ])


def compute_noise_ceiling(reliab: np.ndarray) -> np.ndarray:
    """Upper bound on Pearson r given split-half reliability."""
    return np.sqrt(np.clip(reliab, 0, 1))


def run_snmf_encoding(
    dataset: ThingsDataset,
    monkey: str,
    rank: int,
    n_folds: int,
    n_jobs: int,
    random_state: int,
    data_root: Path | str = Path("data"),
) -> pd.DataFrame:
    """Run CV encoding from SNMF embeddings for all non-macaque models."""
    y, reliab = load_macaque_it(monkey, data_root)
    noise_ceil = compute_noise_ceiling(reliab)
    log.info(f"Noise ceiling: median={np.median(noise_ceil):.3f}")

    models = [m for m in dataset.list_component_models() if not m.startswith("macaque")]
    log.info(f"SNMF encoding: {len(models)} models at rank {rank}")

    def _run_one(model_name: str) -> dict:
        w = dataset.load_components(model_name, rank=rank)
        r_per_neuron = encode_single_model(w, y, n_folds, random_state)
        r_corrected = r_per_neuron / noise_ceil
        return {
            "model": model_name,
            "mean_r": float(np.mean(r_per_neuron)),
            "median_r": float(np.median(r_per_neuron)),
            "mean_r_corrected": float(np.mean(np.clip(r_corrected, -1, 1))),
            "median_r_corrected": float(np.median(np.clip(r_corrected, -1, 1))),
            "n_neurons": len(r_per_neuron),
        }

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(m) for m in models
    )
    return pd.DataFrame(results)


def run_raw_encoding(
    dataset: ThingsDataset,
    monkey: str,
    n_folds: int,
    n_jobs: int,
    random_state: int,
    data_root: Path | str = Path("data"),
) -> pd.DataFrame:
    """Run CV encoding from raw features for all models."""
    y, reliab = load_macaque_it(monkey, data_root)
    noise_ceil = compute_noise_ceiling(reliab)
    log.info(f"Monkey {monkey}: {y.shape[1]} reliable IT neurons, {y.shape[0]} images")

    models = dataset.list_embedding_models()
    log.info(f"Raw encoding: {len(models)} models")

    def _run_one(model_name: str) -> dict | None:
        try:
            features = dataset.load_embedding(model_name)
            if features.shape[0] != y.shape[0]:
                return None
            r_per_neuron = encode_single_model(features, y, n_folds, random_state)
            r_corrected = r_per_neuron / noise_ceil
            return {
                "model": model_name,
                "mean_r": float(np.mean(r_per_neuron)),
                "median_r": float(np.median(r_per_neuron)),
                "mean_r_corrected": float(np.mean(np.clip(r_corrected, -1, 1))),
                "median_r_corrected": float(np.median(np.clip(r_corrected, -1, 1))),
                "n_neurons": y.shape[1],
                "feature_dim": features.shape[1],
            }
        except Exception as e:
            log.warning(f"  Failed {model_name}: {e}")
            return None

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(m) for m in models
    )
    return pd.DataFrame([r for r in results if r is not None])


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Pearson correlation between x and y controlling for z.

    Returns (r, p) from scipy.stats.pearsonr on the residuals.
    """
    z_col = z.reshape(-1, 1)
    resid_x = x - LinearRegression().fit(z_col, x).predict(z_col)
    resid_y = y - LinearRegression().fit(z_col, y).predict(z_col)
    return stats.pearsonr(resid_x, resid_y)


def analyze_accuracy(
    df: pd.DataFrame,
    monkey: str,
    features_label: str,
) -> pd.DataFrame:
    """Run all correlation analyses: accuracy, universality, encoding triangle.

    Returns DataFrame of results rows.
    """
    sub = df.dropna(subset=["universality", "mean_r", "top1_accuracy"])
    n = len(sub)
    log.info(f"\nMonkey {monkey} [{features_label}]: {n} models with encoding + accuracy + universality")

    if n < 5:
        log.warning("Too few models for analysis")
        return pd.DataFrame()

    rows = []

    r_acc, p_acc = stats.pearsonr(sub["top1_accuracy"], sub["mean_r"])
    rho_acc, rho_p_acc = stats.spearmanr(sub["top1_accuracy"], sub["mean_r"])
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "accuracy_vs_encoding",
                 "pearson_r": r_acc, "pearson_p": p_acc,
                 "spearman_rho": rho_acc, "spearman_p": rho_p_acc, "n": n})
    log.info(f"  Accuracy <> Encoding: r={r_acc:.3f} p={p_acc:.3e} | rho={rho_acc:.3f}")

    r_uni, p_uni = stats.pearsonr(sub["universality"], sub["mean_r"])
    rho_uni, rho_p_uni = stats.spearmanr(sub["universality"], sub["mean_r"])
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "universality_vs_encoding",
                 "pearson_r": r_uni, "pearson_p": p_uni,
                 "spearman_rho": rho_uni, "spearman_p": rho_p_uni, "n": n})
    log.info(f"  Universality <> Encoding: r={r_uni:.3f} p={p_uni:.3e} | rho={rho_uni:.3f}")

    r_au, p_au = stats.pearsonr(sub["top1_accuracy"], sub["universality"])
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "accuracy_vs_universality",
                 "pearson_r": r_au, "pearson_p": p_au,
                 "spearman_rho": np.nan, "spearman_p": np.nan, "n": n})
    log.info(f"  Accuracy <> Universality: r={r_au:.3f} p={p_au:.3e}")

    acc = sub["top1_accuracy"].values
    uni = sub["universality"].values
    enc = sub["mean_r"].values

    pc_uni = partial_corr(enc, uni, acc)
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "encoding_vs_universality|accuracy",
                 "pearson_r": pc_uni.statistic, "pearson_p": pc_uni.pvalue,
                 "spearman_rho": np.nan, "spearman_p": np.nan, "n": n})
    log.info(f"  Encoding <> Universality | Accuracy: r={pc_uni.statistic:.3f} p={pc_uni.pvalue:.3e}")

    pc_acc = partial_corr(enc, acc, uni)
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "encoding_vs_accuracy|universality",
                 "pearson_r": pc_acc.statistic, "pearson_p": pc_acc.pvalue,
                 "spearman_rho": np.nan, "spearman_p": np.nan, "n": n})
    log.info(f"  Encoding <> Accuracy | Universality: r={pc_acc.statistic:.3f} p={pc_acc.pvalue:.3e}")

    x_reg = np.column_stack([uni, acc])
    reg = LinearRegression().fit(x_reg, enc)
    r2 = reg.score(x_reg, enc)
    log.info(f"  Multiple regression R2={r2:.3f}, beta_univ={reg.coef_[0]:.4f}, beta_acc={reg.coef_[1]:.6f}")
    rows.append({"monkey": monkey, "features": features_label, "subset": "all",
                 "comparison": "multiple_regression",
                 "pearson_r": np.sqrt(r2), "pearson_p": np.nan,
                 "spearman_rho": reg.coef_[0], "spearman_p": reg.coef_[1], "n": n})

    supervised_tasks = {"classification", "imagenet classification"}
    sup = sub[sub["train_task"].str.lower().isin(supervised_tasks)]
    if len(sup) >= 10:
        n_sup = len(sup)
        r_acc_s, p_acc_s = stats.pearsonr(sup["top1_accuracy"], sup["mean_r"])
        rho_acc_s, rho_p_s = stats.spearmanr(sup["top1_accuracy"], sup["mean_r"])
        rows.append({"monkey": monkey, "features": features_label, "subset": "supervised",
                     "comparison": "accuracy_vs_encoding",
                     "pearson_r": r_acc_s, "pearson_p": p_acc_s,
                     "spearman_rho": rho_acc_s, "spearman_p": rho_p_s, "n": n_sup})
        log.info(f"  Accuracy <> Encoding (supervised, n={n_sup}): r={r_acc_s:.3f} p={p_acc_s:.3e}")

        r_uni_s, p_uni_s = stats.pearsonr(sup["universality"], sup["mean_r"])
        rows.append({"monkey": monkey, "features": features_label, "subset": "supervised",
                     "comparison": "universality_vs_encoding",
                     "pearson_r": r_uni_s, "pearson_p": p_uni_s,
                     "spearman_rho": np.nan, "spearman_p": np.nan, "n": n_sup})
        log.info(f"  Universality <> Encoding (supervised, n={n_sup}): r={r_uni_s:.3f} p={p_uni_s:.3e}")

        acc_s, uni_s, enc_s = sup["top1_accuracy"].values, sup["universality"].values, sup["mean_r"].values
        pc1 = partial_corr(enc_s, uni_s, acc_s)
        rows.append({"monkey": monkey, "features": features_label, "subset": "supervised",
                     "comparison": "encoding_vs_universality|accuracy",
                     "pearson_r": pc1.statistic, "pearson_p": pc1.pvalue,
                     "spearman_rho": np.nan, "spearman_p": np.nan, "n": n_sup})
        log.info(f"  Encoding <> Universality | Accuracy (supervised): r={pc1.statistic:.3f}")

        pc2 = partial_corr(enc_s, acc_s, uni_s)
        rows.append({"monkey": monkey, "features": features_label, "subset": "supervised",
                     "comparison": "encoding_vs_accuracy|universality",
                     "pearson_r": pc2.statistic, "pearson_p": pc2.pvalue,
                     "spearman_rho": np.nan, "spearman_p": np.nan, "n": n_sup})
        log.info(f"  Encoding <> Accuracy | Universality (supervised): r={pc2.statistic:.3f}")

    return pd.DataFrame(rows)


def merge_metadata(
    df: pd.DataFrame,
    data_root: Path | str = Path("data"),
    dataset: str = "things22k",
    rank: int = 50,
) -> pd.DataFrame:
    """Merge encoding results with universality scores and model metadata."""
    merge_cols = ["universality", "model_class", "train_task", "train_data",
                  "train_task_display", "train_data_display",
                  "architecture", "top1_accuracy", "feature_uid"]
    df = df.drop(columns=[c for c in merge_cols if c in df.columns])

    uni = pd.read_csv(
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    uni = uni[uni["metric"] == "cos2_perm_adj"]
    uni = uni.groupby("model")["universality"].mean().reset_index()
    df = df.merge(uni, on="model", how="left")

    meta = pd.read_csv(Path(data_root) / "meta" / "things22k" / "model_metadata.csv")
    meta_cols = ["feature_uid", "model_class", "train_task", "train_data",
                 "train_task_display", "train_data_display",
                 "architecture", "top1_accuracy"]
    df = df.merge(
        meta[meta_cols],
        left_on="model", right_on="feature_uid", how="left",
    )
    return df.drop(columns=["feature_uid"], errors="ignore")


def plot_universality_vs_encoding(
    df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
) -> None:
    """Scatter of universality vs macaque IT encoding."""
    sub = df.dropna(subset=["universality", "mean_r"])
    if len(sub) < 5:
        return

    fig, ax = create_figure("single")
    r, p = stats.pearsonr(sub["universality"], sub["mean_r"])
    rho, _ = stats.spearmanr(sub["universality"], sub["mean_r"])

    ax.scatter(
        sub["universality"], sub["mean_r"],
        s=20, alpha=0.5, color=INDIGO,
        edgecolors="white", linewidth=0.3,
    )
    z = np.polyfit(sub["universality"], sub["mean_r"], 1)
    x_line = np.linspace(sub["universality"].min(), sub["universality"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY, lw=1)

    ax.set_xlabel("Universality (mean across dimensions)")
    ax.set_ylabel("Mean encoding r")
    ax.text(
        0.05, 0.95,
        f"r = {r:.3f}, p = {p:.1e}\nrho = {rho:.3f}\nn = {len(sub)}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    save_figure(fig, output_dir / f"universality_vs_encoding_{monkey}")
    log.info(f"Saved universality_vs_encoding_{monkey}")


def plot_model_ranking(
    df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
    top_n: int = 40,
) -> None:
    """Horizontal bar chart of top/bottom models by encoding accuracy."""
    default_color = GRAY
    sorted_df = df.sort_values("mean_r", ascending=False).reset_index(drop=True)

    half = top_n // 2
    top = sorted_df.head(half)
    bottom = sorted_df.tail(half)
    show = pd.concat([top, bottom]).reset_index(drop=True)
    show["rank_label"] = (
        [f"#{i+1}" for i in range(len(top))]
        + [f"#{len(sorted_df) - len(bottom) + i + 1}" for i in range(len(bottom))]
    )

    bar_colors = [
        CLASS_COLORS.get(row.get("model_class", ""), default_color)
        for _, row in show.iterrows()
    ]

    fig_h = 0.35 * len(show) + 1.5
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(len(show))
    ax.barh(y_pos, show["mean_r"], color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['rank_label']} {row['model']}" for _, row in show.iterrows()],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean encoding r ({N_FOLDS}-fold CV)")
    ax.set_title(f"Monkey {monkey} IT encoding — top/bottom {half}")

    if "model_class" in show.columns:
        for cls, color in CLASS_COLORS.items():
            if (show["model_class"] == cls).any():
                ax.barh([], [], color=color, label=cls)
        ax.legend(fontsize=7, loc="lower right")

    save_figure(fig, output_dir / f"model_ranking_{monkey}")
    log.info(f"Saved model_ranking_{monkey}")


def plot_raw_vs_snmf(
    df_raw: pd.DataFrame,
    df_snmf: pd.DataFrame,
    monkey: str,
    output_dir: Path,
) -> None:
    """Scatter of SNMF encoding (x) vs raw feature encoding (y) with rank stats."""
    both = df_snmf[["model", "mean_r"]].rename(columns={"mean_r": "snmf_encoding"}).merge(
        df_raw[["model", "mean_r"]].rename(columns={"mean_r": "raw_encoding"}),
        on="model", how="inner",
    )
    if len(both) < 10:
        return

    r, p = stats.pearsonr(both["snmf_encoding"], both["raw_encoding"])
    rho, _ = stats.spearmanr(both["snmf_encoding"], both["raw_encoding"])

    both["rank_snmf"] = both["snmf_encoding"].rank(ascending=False).astype(int)
    both["rank_raw"] = both["raw_encoding"].rank(ascending=False).astype(int)
    both["rank_diff"] = (both["rank_snmf"] - both["rank_raw"]).abs()

    log.info(f"  SNMF vs Raw encoding (n={len(both)}): r={r:.3f}, rho={rho:.3f}")
    log.info(f"  Rank difference: mean={both['rank_diff'].mean():.1f}, median={both['rank_diff'].median():.0f}, max={both['rank_diff'].max()}")
    top_movers = both.nlargest(5, "rank_diff")
    for _, row in top_movers.iterrows():
        log.info(f"    {row['model']}: SNMF rank {row['rank_snmf']} -> Raw rank {row['rank_raw']} (delta={row['rank_diff']})")

    fig, ax = create_figure("square")
    ax.scatter(
        both["snmf_encoding"], both["raw_encoding"],
        s=20, alpha=0.5, color=INDIGO,
        edgecolors="white", linewidth=0.3,
    )
    lims = [
        min(both["snmf_encoding"].min(), both["raw_encoding"].min()) - 0.02,
        max(both["snmf_encoding"].max(), both["raw_encoding"].max()) + 0.02,
    ]
    ax.plot(lims, lims, ls="--", color=GRAY_LIGHT, lw=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("SNMF embedding encoding (mean r)")
    ax.set_ylabel("Raw feature encoding (mean r)")
    ax.text(
        0.05, 0.95, f"r = {r:.3f}, rho = {rho:.3f}\nn = {len(both)}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    save_figure(fig, output_dir / f"snmf_vs_raw_encoding_{monkey}")
    log.info(f"Saved snmf_vs_raw_encoding_{monkey}")

    both.to_csv(output_dir / f"snmf_vs_raw_ranking_{monkey}.csv", index=False)
    log.info(f"Saved snmf_vs_raw_ranking_{monkey}.csv")

    fig, ax = create_figure("square")
    ax.scatter(
        both["rank_snmf"], both["rank_raw"],
        s=20, alpha=0.5, color=INDIGO,
        edgecolors="white", linewidth=0.3,
    )
    n = len(both)
    ax.plot([1, n], [1, n], ls="--", color=GRAY_LIGHT, lw=0.8, zorder=0)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("SNMF embedding rank")
    ax.set_ylabel("Raw feature rank")
    ax.text(
        0.05, 0.95, f"rho = {rho:.3f}\nn = {n}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    save_figure(fig, output_dir / f"snmf_vs_raw_rank_{monkey}")
    log.info(f"Saved snmf_vs_raw_rank_{monkey}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    dataset = ThingsDataset()
    setup_plotting()

    rank = cfg.factorization.rank
    data_root = Path(cfg.data_root)
    all_analysis_rows = []

    for monkey in ["F", "N"]:
        snmf_csv = out / f"encoding_results_{monkey}.csv"
        raw_csv = out / f"raw_encoding_results_{monkey}.csv"

        log.info(f"\nMonkey {monkey}: SNMF encoding")
        df_snmf = run_snmf_encoding(
            dataset, monkey, rank, N_FOLDS, cfg.n_jobs, cfg.seed, data_root,
        )
        df_snmf = merge_metadata(df_snmf, data_root, cfg.dataset, rank)
        df_snmf.to_csv(snmf_csv, index=False)
        log.info(f"Saved {snmf_csv} ({len(df_snmf)} models)")

        log.info(f"\nMonkey {monkey}: Raw feature encoding")
        df_raw = run_raw_encoding(
            dataset, monkey, N_FOLDS, cfg.n_jobs, cfg.seed, data_root,
        )
        df_raw = merge_metadata(df_raw, data_root, cfg.dataset, rank)
        df_raw.to_csv(raw_csv, index=False)
        log.info(f"Saved {raw_csv} ({len(df_raw)} models)")

        log.info(f"\nMonkey {monkey} SNMF: mean_r range [{df_snmf['mean_r'].min():.3f}, {df_snmf['mean_r'].max():.3f}]")
        all_analysis_rows.append(analyze_accuracy(df_snmf, monkey, "snmf"))

        plot_universality_vs_encoding(df_snmf, monkey, out)
        plot_model_ranking(df_snmf, monkey, out)

        log.info(f"Monkey {monkey} Raw: mean_r range [{df_raw['mean_r'].min():.3f}, {df_raw['mean_r'].max():.3f}]")
        all_analysis_rows.append(analyze_accuracy(df_raw, monkey, "raw"))

        plot_raw_vs_snmf(df_raw, df_snmf, monkey, out)

    if all_analysis_rows:
        combined = pd.concat(all_analysis_rows, ignore_index=True)
        combined.to_csv(out / "accuracy_vs_encoding_results.csv", index=False)
        log.info(f"\nSaved {out / 'accuracy_vs_encoding_results.csv'}")


if __name__ == "__main__":
    main()
