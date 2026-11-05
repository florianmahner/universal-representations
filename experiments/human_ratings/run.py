from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats

from src.colors import ROSE, TEAL, INDIGO, GRAY_DARK, GRAY_LIGHT
from src.plotting import setup_plotting, create_figure, save_figure

# The four response options shown to raters, in the order given in the
# manuscript; ties in a plurality vote are resolved by this order.
CONTENT_ORDER = ["semantic", "visual", "both", "neither"]


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_trials(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_trials, real_trials)."""
    all_trials = pd.read_csv(path)
    real = all_trials[all_trials["task_type"] == "real"].copy()
    return all_trials, real


def load_universality(dataset: str, rank: int) -> pd.DataFrame:
    """Load per-dimension universality scores written by the metric experiment."""
    path = (
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    univ = pd.read_csv(path)
    univ = univ[univ["metric"] == "cos2_perm_adj"].reset_index(drop=True)
    univ["dim_id"] = univ["model"] + "__d" + univ["dimension"].astype(str)
    return univ[["dim_id", "universality"]]


def quality_check(all_trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the manuscript's attention-check exclusion criterion.

    Catch-trial performance is retained as a descriptive field, but it is not
    an exclusion criterion in the paper.
    """
    catch = all_trials[all_trials["task_type"] == "catch"]
    attn = all_trials[all_trials["task_type"] == "attention_check"]

    catch_pass = catch.groupby("participant")["category"].apply(
        lambda g: g.isin(["semantic", "both"]).any()
    )
    attn_pass = attn.groupby("participant").apply(
        lambda g: (g["category"] == "both").all() & (g["interpretability"] == 4).all()
    )

    pids = all_trials["participant"].unique()
    qc = pd.DataFrame({"participant": pids})
    qc = qc.merge(catch_pass.rename("catch_pass").reset_index(), on="participant", how="left")
    qc = qc.merge(attn_pass.rename("attn_pass").reset_index(), on="participant", how="left")
    qc["catch_pass"] = qc["catch_pass"].fillna(False)
    qc["attn_pass"] = qc["attn_pass"].fillna(False)
    qc["pass"] = qc["attn_pass"]

    n_total = len(qc)
    print(f"QC: {qc['pass'].sum()}/{n_total} passed ({qc['pass'].mean():.0%})")

    clean = all_trials[all_trials["participant"].isin(qc[qc["pass"]]["participant"])]
    return clean, qc


def compute_reliability(real: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension inter-rater agreement (majority label + pairwise agreement)."""
    records = []
    for dim_id, group in real.groupby("dim_id"):
        n = len(group)
        if n < 2:
            continue
        vc = group["category"].value_counts()
        unknown = set(vc.index) - set(CONTENT_ORDER)
        if unknown:
            raise ValueError(f"Unexpected response labels for {dim_id}: {unknown}")
        # Resolve tied plurality votes in the response order stated in the
        # manuscript, rather than relying on pandas' encounter-order behavior.
        majority = max(CONTENT_ORDER, key=lambda label: (vc.get(label, 0), -CONTENT_ORDER.index(label)))
        majority_frac = vc.get(majority, 0) / n
        n_agree = sum(c * (c - 1) for c in vc.values)
        n_pairs = n * (n - 1)
        agreement = n_agree / n_pairs if n_pairs > 0 else 0
        records.append({
            "dim_id": dim_id,
            "model": dim_id.rsplit("__d", 1)[0] if "__d" in dim_id else "",
            "n_raters": n,
            "majority_label": majority,
            "majority_frac": majority_frac,
            "pairwise_agreement": agreement,
            **{f"n_{c}": vc.get(c, 0) for c in CONTENT_ORDER},
        })
    return pd.DataFrame(records)


def _majority(labels: np.ndarray) -> str:
    """Deterministic plurality label using the manuscript response order for ties."""
    counts = {label: int(np.count_nonzero(labels == label)) for label in CONTENT_ORDER}
    return max(
        CONTENT_ORDER,
        key=lambda label: (counts[label], -CONTENT_ORDER.index(label)),
    )


def compute_split_half(real: pd.DataFrame, half_size: int = 4) -> pd.DataFrame:
    """Exact 4+4 split-half agreement for the original eight-rater design.

    Participant exclusion leaves fewer than eight responses for most
    dimensions, so this design-reliability diagnostic uses the original cohort
    before exclusion.  For dimensions with nine responses because of the two
    replacement assignments, every choice of eight responses is included.
    Swapping the two halves is counted only once.
    """
    records = []
    n_required = 2 * half_size
    for dim_id, group in real.groupby("dim_id"):
        labels = group["category"].to_numpy()
        if len(labels) < n_required:
            continue
        agreements: list[bool] = []
        for selected in combinations(range(len(labels)), n_required):
            selected_set = set(selected)
            anchor = selected[0]
            # Requiring the first half to contain the smallest selected index
            # removes the duplicate obtained by swapping the two halves.
            for remainder in combinations(selected[1:], half_size - 1):
                first = {anchor, *remainder}
                second = selected_set - first
                agreements.append(
                    _majority(labels[sorted(first)])
                    == _majority(labels[sorted(second)])
                )
        records.append(
            {
                "dim_id": dim_id,
                "majority_label": _majority(labels),
                "n_raters": len(labels),
                "n_partitions": len(agreements),
                "split_half": float(np.mean(agreements)),
            }
        )
    return pd.DataFrame(records)


def compute_rater_accuracy_by_content(
    real: pd.DataFrame, reliability: pd.DataFrame
) -> pd.DataFrame:
    """Per-rater accuracy (agreement with majority label) broken down by content type.

    Returns a long-form DataFrame:
        participant, majority_label, n_trials, n_correct, accuracy
    """
    dim_labels = reliability[["dim_id", "majority_label"]].copy()
    df = real.merge(dim_labels, on="dim_id", how="inner")
    df["correct"] = df["category"] == df["majority_label"]

    records = (
        df.groupby(["participant", "majority_label"])
        .agg(n_trials=("correct", "count"), n_correct=("correct", "sum"))
        .reset_index()
    )
    records["accuracy"] = records["n_correct"] / records["n_trials"]
    return records


def plot_rater_accuracy_by_content(rater_acc: pd.DataFrame) -> plt.Figure:
    """Violin + box: per-rater accuracy with majority vote, by content type."""
    fig, ax = create_figure("single")

    content_colors = {
        "semantic": INDIGO,
        "visual": ROSE,
        "both": TEAL,
        "neither": GRAY_LIGHT,
    }
    present = [c for c in CONTENT_ORDER if c in rater_acc["majority_label"].values]
    palette = [content_colors[c] for c in present]

    sns.violinplot(
        data=rater_acc, x="majority_label", y="accuracy",
        order=present, hue="majority_label", hue_order=present,
        palette=palette, inner=None, linewidth=0.5,
        cut=0, ax=ax, alpha=0.3, legend=False,
    )
    sns.boxplot(
        data=rater_acc, x="majority_label", y="accuracy",
        order=present, hue="majority_label", hue_order=present,
        palette=palette, width=0.15, linewidth=0.8,
        fliersize=0, ax=ax, boxprops=dict(zorder=3),
        medianprops=dict(color="k", lw=1.2), legend=False,
    )

    for i, cat in enumerate(present):
        sub = rater_acc.loc[rater_acc["majority_label"] == cat, "accuracy"]
        ax.text(
            i, -0.06, f"n={len(sub)}\n$\\mu$={sub.mean():.2f}",
            ha="center", va="top", fontsize=7, color=GRAY_DARK,
            transform=ax.get_xaxis_transform(),
        )

    groups = [
        rater_acc.loc[rater_acc["majority_label"] == c, "accuracy"].values
        for c in present
    ]
    valid = [g for g in groups if len(g) > 0]
    if len(valid) >= 2:
        H, p_kw = stats.kruskal(*valid)
        p_str = "<.001" if p_kw < 0.001 else f"{p_kw:.3f}"
        ax.set_title(f"KW H={H:.1f}, p={p_str}", fontsize=9)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([c.capitalize() for c in present])
    ax.set_xlabel("Majority content label")
    ax.set_ylabel("Rater accuracy (agree with majority)")
    ax.set_ylim(-0.05, 1.05)
    return fig


def plot_disagreement_confusion(real: pd.DataFrame, reliability: pd.DataFrame) -> plt.Figure:
    """Heatmap: what label do raters assign when they disagree with the majority?

    Rows = majority label, columns = rater's label (only disagreeing trials).
    Values = proportion of disagreements from that majority class going to each label.
    """
    dim_labels = reliability[["dim_id", "majority_label"]].copy()
    df = real.merge(dim_labels, on="dim_id", how="inner")
    disagree = df[df["category"] != df["majority_label"]].copy()

    ct = pd.crosstab(disagree["majority_label"], disagree["category"])
    ct = ct.reindex(index=CONTENT_ORDER, columns=CONTENT_ORDER, fill_value=0)
    for c in CONTENT_ORDER:
        ct.loc[c, c] = 0

    row_sums = ct.sum(axis=1).replace(0, np.nan)
    ct_norm = ct.div(row_sums, axis=0).fillna(0)

    fig, ax = create_figure("single")
    im = ax.imshow(ct_norm.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(CONTENT_ORDER)))
    ax.set_yticks(range(len(CONTENT_ORDER)))
    labels = [c.capitalize() for c in CONTENT_ORDER]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rater label")
    ax.set_ylabel("Majority label")
    ax.set_title("Disagreement confusion (row-normalised)", fontsize=9)

    for i in range(len(CONTENT_ORDER)):
        for j in range(len(CONTENT_ORDER)):
            v = ct_norm.values[i, j]
            raw = ct.values[i, j]
            if raw > 0:
                text_color = "white" if v > 0.55 else GRAY_DARK
                ax.text(j, i, f"{v:.2f}\n(n={raw})", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion of disagreements")
    return fig


def plot_label_proportions(merged: pd.DataFrame) -> plt.Figure:
    """Panel d: proportion of each majority label across universality bins (Wilson CI)."""
    fig, ax = create_figure("single")

    m2 = merged.dropna(subset=["universality", "majority_label"]).copy()
    m2["bin"] = pd.qcut(m2["universality"], q=7, labels=False, duplicates="drop")
    bin_centers = m2.groupby("bin")["universality"].mean().values
    n_bins = len(bin_centers)

    for cat, color in [
        ("neither", GRAY_DARK),
        ("visual", ROSE),
        ("semantic", TEAL),
        ("both", INDIGO),
    ]:
        props, lo_arr, hi_arr = [], [], []
        for b in range(n_bins):
            grp = m2[m2["bin"] == b]
            n = len(grp)
            k = int((grp["majority_label"] == cat).sum())
            lo, hi = _wilson_ci(k, n)
            props.append(k / n if n > 0 else 0.0)
            lo_arr.append(lo)
            hi_arr.append(hi)
        ax.fill_between(bin_centers, lo_arr, hi_arr, alpha=0.15, color=color)
        ax.plot(bin_centers, props, "o-", color=color, markersize=3,
                lw=1.0, label=cat.capitalize())

    ax.set_xlabel("Universality")
    ax.set_ylabel("Proportion of dimensions")
    ax.legend(fontsize=8)
    return fig


def print_stats(rater_acc: pd.DataFrame) -> None:
    print("\n=== Rater accuracy by content type ===")
    for cat in CONTENT_ORDER:
        sub = rater_acc.loc[rater_acc["majority_label"] == cat, "accuracy"]
        if len(sub):
            print(f"  {cat}: mean={sub.mean():.2f} ± {sub.std():.2f} (n={len(sub)} rater observations)")

    present = [c for c in CONTENT_ORDER if c in rater_acc["majority_label"].values]
    groups = [rater_acc.loc[rater_acc["majority_label"] == c, "accuracy"].values for c in present]
    valid = [g for g in groups if len(g) > 0]
    if len(valid) >= 2:
        H, p_kw = stats.kruskal(*valid)
        print(f"\nKruskal-Wallis: H={H:.1f}, p={'<.001' if p_kw < 0.001 else f'{p_kw:.3f}'}")

    pairs = list(combinations(present, 2))
    print("\nPost-hoc Mann-Whitney U (Bonferroni corrected):")
    for a, b in pairs:
        u_a = rater_acc.loc[rater_acc["majority_label"] == a, "accuracy"].values
        u_b = rater_acc.loc[rater_acc["majority_label"] == b, "accuracy"].values
        if len(u_a) > 0 and len(u_b) > 0:
            _, p = stats.mannwhitneyu(u_a, u_b, alternative="two-sided")
            p_corr = min(p * len(pairs), 1.0)
            sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
            print(f"  {a} vs {b}: p={p_corr:.4f} {sig}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    print("Loading data...")
    all_trials, all_real = load_trials(out / "parsed_all_trials.csv")
    n_participants = all_trials["participant"].nunique()
    if n_participants != 570:
        raise ValueError(
            f"Expected the paper cohort of 570 participants, "
            f"found {n_participants}. Check that collection waves were not combined."
        )

    clean_trials, qc = quality_check(all_trials)
    n_pass = int(qc["pass"].sum())
    if n_pass != 432:
        raise ValueError(
            f"Expected 432 attention-check passes, found {n_pass}."
        )
    real = clean_trials[clean_trials["task_type"] == "real"].copy()
    print(
        f"  retained {len(real)} real trials from {n_pass}/{n_participants} "
        f"participants across {real['dim_id'].nunique()} dimensions"
    )

    print("Loading universality scores...")
    univ = load_universality(cfg.dataset, cfg.factorization.rank)

    reliability = compute_reliability(real)
    merged = reliability.merge(univ, on="dim_id", how="inner", validate="one_to_one")
    rating_columns = [
        "dim_id",
        "model",
        "n_raters",
        "majority_label",
        "majority_frac",
        "pairwise_agreement",
        *[f"n_{label}" for label in CONTENT_ORDER],
    ]
    merged = merged.sort_values("dim_id").reset_index(drop=True)
    rating_aggregate = (
        reliability[rating_columns].sort_values("dim_id").reset_index(drop=True)
    )
    print(f"  {len(merged)} dims with universality scores")
    if len(merged) != len(reliability):
        raise ValueError(
            f"Universality scores are missing for {len(reliability) - len(merged)} "
            "rated dimensions."
        )

    aggregate_path = (
        Path(cfg.data_root) / "ratings" / "dim_ratings_aggregated_qc_attn_catch.csv"
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the durable human-data artifact independent of factor selection.
    # Universality is recomputed from the selected factors and joined only at
    # analysis time, preventing stale scores from being carried forward after
    # a bandwidth or seed selection changes.
    rating_aggregate.to_csv(aggregate_path, index=False)
    qc.to_csv(out / "participant_qc.csv", index=False)
    real.to_csv(out / "parsed_real_trials_qc_attn.csv", index=False)

    rater_acc = compute_rater_accuracy_by_content(real, reliability)
    split_half = compute_split_half(all_real)
    reliability.to_csv(out / "dimension_reliability_qc_attn.csv", index=False)
    rater_acc.to_csv(out / "rater_accuracy_qc_attn.csv", index=False)
    split_half.to_csv(out / "split_half_all_assigned.csv", index=False)
    print_stats(rater_acc)
    print(
        "\nExact 4+4 split-half agreement (all assigned responses): "
        f"mean={split_half['split_half'].mean():.2f}, "
        f"median={split_half['split_half'].median():.2f}"
    )

    fig = plot_label_proportions(merged)
    save_figure(
        fig,
        out / "label_proportions_by_universality",
        formats=("pdf", "png"),
    )
    plt.close(fig)

    fig = plot_rater_accuracy_by_content(rater_acc)
    save_figure(fig, out / "rater_accuracy_by_content", formats=("pdf", "png"))
    plt.close(fig)

    fig = plot_disagreement_confusion(real, reliability)
    save_figure(fig, out / "disagreement_confusion", formats=("pdf", "png"))
    plt.close(fig)

    print(f"Done! Saved analyses to {out}")
    print(f"Anonymous aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
