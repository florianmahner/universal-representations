"""Figures and statistics for the layer-wise human rating study.

Combines the early/middle/late ratings with the penultimate-layer ratings of the
main experiment and reports content labels, difficulty and their relation to
per-dimension universality across network depth.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from src.plotting import create_figure, save_figure, setup_plotting
from src.colors import GRAY_DARK, INDIGO, ROSE, TEAL

setup_plotting()
plt.rcParams["savefig.bbox"] = None  # fixed canvas: no tight-cropping on save
plt.rcParams["pdf.fonttype"] = 42    # TrueType, not Type 3: text renders like
plt.rcParams["ps.fonttype"] = 42     # the PNG in every viewer, NeurIPS-safe
# house style sets legend 7.5 / titles 10 vs axes.labelsize 9 -> one size
plt.rcParams["legend.fontsize"] = plt.rcParams["axes.labelsize"]
plt.rcParams["legend.title_fontsize"] = plt.rcParams["axes.labelsize"]
plt.rcParams["axes.titlesize"] = plt.rcParams["axes.labelsize"]

# the depth conditions of the study, in depth order
LAYER_ORDER = ["early", "middle", "late", "penultimate"]
# ordinal depth ramp: blue branch of Tol sunset, light -> dark
LAYER_COLORS = {"early": "#98CAE1", "middle": "#6EA6CD",
                "late": "#4A7BB7", "penultimate": "#364B9A"}


def make_figure(ncols: int = 1, **kw):
    # uniform top pad -> all figures share the same pixel height
    return create_figure("single", ncols=ncols, pad_top=0.55, **kw)


def save(fig, path: Path) -> None:
    """Uniform export: fixed canvas (no tight-cropping), 300 dpi PNG + vector
    PDF from the same figure, so sizes are consistent across the set."""
    save_figure(fig, path, formats=("pdf", "png"), tight=False)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def quality_check(all_trials: pd.DataFrame) -> pd.DataFrame:
    """Apply the manuscript's attention-check-only exclusion per session.

    A participant passes when every attention-check response is ``both`` with
    interpretability rating 4. Catch performance is retained descriptively but
    is not an exclusion criterion.
    """
    keys = ["participant", "study"]
    catch = all_trials[all_trials.task_type == "catch"]
    attn = all_trials[all_trials.task_type == "attention_check"]
    catch_pass = catch.groupby(keys)["category"].apply(
        lambda g: g.isin(["semantic", "both"]).any())
    attn_pass = attn.groupby(keys)["category"].apply(
        lambda g: (g == "both").all())
    attn_strict = attn.groupby(keys).apply(
        lambda g: (g["category"] == "both").all() & (g["interpretability"] == 4).all())

    qc = all_trials[keys + ["layer"]].drop_duplicates(subset=keys).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(qc[keys])
    qc["catch_pass"] = catch_pass.reindex(idx).fillna(False).values
    qc["attn_pass"] = attn_pass.reindex(idx).fillna(False).values
    qc["attn_strict"] = attn_strict.reindex(idx).fillna(False).values
    qc["pass"] = qc.catch_pass & qc.attn_pass
    qc["pass_strict"] = qc.catch_pass & qc.attn_strict

    print("=== QC (catch and attention checks, as collected) ===")
    for layer, g in qc.groupby("layer"):
        print(f"  {layer}: {g['pass'].sum()}/{len(g)} passed "
              f"(catch {g.catch_pass.sum()}, attention {g.attn_pass.sum()})")
    return qc


def repeat_consistency(clean: pd.DataFrame) -> None:
    rep = clean[clean.task_type == "repeat"].copy()
    if rep.empty:
        return
    rep["orig_id"] = rep.dim_id.str.removeprefix("R_")
    orig = clean[clean.task_type == "real"][
        ["participant", "dim_id", "category", "interpretability"]]
    m = rep.merge(orig, left_on=["participant", "orig_id"],
                  right_on=["participant", "dim_id"], suffixes=("_rep", "_orig"))
    if m.empty:
        return
    print("\n=== Repeat trials (within-rater reliability, per layer) ===")
    for layer in [l for l in LAYER_ORDER if l in m.layer.unique()]:
        g = m[m.layer == layer]
        if len(g) < 3:
            print(f"  {layer}: n={len(g)} (too few)")
            continue
        match = (g.category_rep == g.category_orig).mean()
        diff = (g.interpretability_rep - g.interpretability_orig).abs()
        r, p = stats.spearmanr(g.interpretability_rep, g.interpretability_orig)
        print(f"  {layer}: n={len(g)}, category match {match:.0%}, "
              f"|rating diff| mean {diff.mean():.2f}, "
              f"test-retest r={r:.2f} (p={p:.3f})")


def rater_agreement(real: pd.DataFrame) -> None:
    """Per-layer inter-rater quality: mean pairwise agreement and majority
    fraction across dimensions with >= 2 raters (penultimate = baseline)."""
    print("\n=== Inter-rater agreement (per layer, dims with >= 2 raters) ===")
    for layer in [l for l in LAYER_ORDER if l in real.layer.unique()]:
        recs = []
        for dim_id, g in real[real.layer == layer].groupby("dim_id"):
            n = len(g)
            if n < 2:
                continue
            vc = g.category.value_counts()
            n_agree = sum(c * (c - 1) for c in vc.values)
            recs.append((n_agree / (n * (n - 1)), vc.iloc[0] / n))
        if not recs:
            continue
        pw = np.mean([r[0] for r in recs])
        mj = np.mean([r[1] for r in recs])
        print(f"  {layer}: pairwise agreement {pw:.2f}, "
              f"majority fraction {mj:.2f} (n = {len(recs)} dims)")


def dim_aggregates(real: pd.DataFrame, rank: int = 50,
                   dataset: str = "things22k") -> pd.DataFrame:
    content = ["semantic", "visual", "both", "neither"]
    rows = []
    for (dim_id, layer), g in real.groupby(["dim_id", "layer"]):
        vc = g.category.value_counts()
        rows.append({
            "dim_id": dim_id,
            "layer": layer,
            "n_raters": len(g),
            "majority_label": vc.index[0],
            "majority_frac": vc.iloc[0] / len(g),
            "mean_interpretability": g.interpretability.mean(),
            **{f"n_{c}": vc.get(c, 0) for c in content},
            **{f"share_{c}": vc.get(c, 0) / len(g) for c in content},
        })
    agg = pd.DataFrame(rows)

    # WITHIN-COHORT universality (cohort_<depth>: each depth scored against
    # the other 161 models at the SAME depth) — the exact analogue of the
    # paper's penultimate metric. NOT the vs_ref162 variant that the same
    # table also carries (that one scores layer dims against the penultimate
    # reference structure — a different question).
    full = pd.read_csv(Path("experiments/rebuttal/universality_across_layers/scoring/outputs")
                       / f"universality_full_rank_{rank}.csv")
    full = full[full.scoring.str.startswith("cohort_d")]
    full["dim_id"] = full["repr"] + "__d" + full["dimension"].astype(str)
    univ_map = dict(zip(full.dim_id, full.universality))
    # penultimate dims: universality at the same rank from the main analysis
    pen = pd.read_csv(Path("experiments/universality/shared/data")
                      / f"universality_{dataset}_rank_{rank}.csv")
    pen = pen[pen.metric == "cos2_perm_adj"]
    pen["dim_id"] = pen["model"] + "__d" + pen["dimension"].astype(str)
    univ_map.update(dict(zip(pen.dim_id, pen.universality)))
    agg["universality"] = agg.dim_id.map(univ_map)
    return agg


def plot_content_by_layer(real: pd.DataFrame) -> plt.Figure:
    """Content label on x, one bar per layer (depth-ordered) within each label,
    so shifts across depth read as a staircase per label."""
    categories = ["visual", "semantic", "both", "neither"]
    fig, ax = make_figure()
    layers = [l for l in LAYER_ORDER if l in real.layer.unique()]
    width = 0.8 / len(layers)
    for j, layer in enumerate(layers):
        g = real[real.layer == layer]
        xs, props, errs = [], [], []
        for i, cat in enumerate(categories):
            k, n = int((g.category == cat).sum()), len(g)
            lo, hi = wilson_ci(k, n)
            xs.append(i + (j - (len(layers) - 1) / 2) * width)
            props.append(k / n)
            errs.append([k / n - lo, hi - k / n])
        ax.bar(xs, props, width=width * 0.9, color=LAYER_COLORS[layer],
               label=layer.capitalize())
        ax.errorbar(xs, props, yerr=np.array(errs).T, fmt="none",
                    ecolor=GRAY_DARK, elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_xlabel("Content label")
    ax.set_ylabel("Proportion of ratings")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, title="Layer")
    return fig


def plot_interpretability_by_layer(real: pd.DataFrame) -> plt.Figure:
    """Mean rated difficulty per content label, per layer. The task question
    is "How easy was it to see what these images have in common?" with
    1 = very easy and 7 = very difficult — i.e. the stored 'interpretability'
    field is a DIFFICULTY rating (verified against the template and the
    dog-grid catch anchor, mean 3.06). Each trial contributes its own
    (category, rating) pair; 95% CI on the mean."""
    categories = ["visual", "semantic", "both", "neither"]
    fig, ax = make_figure()
    layers = [l for l in LAYER_ORDER if l in real.layer.unique()]
    width = 0.8 / len(layers)
    for j, layer in enumerate(layers):
        g = real[real.layer == layer]
        xs, ms, cis = [], [], []
        for i, cat in enumerate(categories):
            v = g.loc[g.category == cat, "interpretability"].dropna()
            xs.append(i + (j - (len(layers) - 1) / 2) * width)
            ms.append(v.mean())
            cis.append(1.96 * v.std() / len(v) ** 0.5)
        ax.bar(xs, ms, width=width * 0.9, color=LAYER_COLORS[layer],
               label=layer.capitalize())
        ax.errorbar(xs, ms, yerr=cis, fmt="none", ecolor=GRAY_DARK,
                    elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_xlabel("Content label")
    ax.set_ylabel("Difficulty rating")
    ax.set_ylim(1, 7)
    fig.legend(loc="upper center", ncol=4, frameon=False)
    return fig


def plot_universality_vs_content(agg: pd.DataFrame) -> plt.Figure:
    """Stacked majority-label proportions across equal-count universality
    bins, one panel per layer."""
    stack_order = ["semantic", "both", "visual", "neither"]
    # same scheme as the penultimate rating study
    colors = {"semantic": TEAL, "visual": ROSE, "both": INDIGO,
              "neither": GRAY_DARK}
    with_univ = agg.dropna(subset=["universality"])
    layers = [l for l in LAYER_ORDER if l in with_univ.layer.unique()]
    fig, axes = make_figure(ncols=len(layers), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, layer in zip(axes, layers):
        m = with_univ[with_univ.layer == layer].copy()
        n_bins = min(7, max(3, len(m) // 10))
        m["bin"] = pd.qcut(m.universality, q=n_bins, labels=False,
                           duplicates="drop")
        wide = (m.groupby("bin").majority_label.value_counts(normalize=True)
                .unstack().reindex(columns=stack_order).fillna(0.0))
        x = np.arange(len(wide))
        bottom = np.zeros(len(wide))
        for label in stack_order:
            vals = wide[label].to_numpy()
            ax.bar(x, vals, bottom=bottom, width=0.84,
                   color=colors[label], label=label.capitalize(),
                   edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_xlim(-0.55, len(wide) - 0.45)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(wide) + 1)])
        ax.set_xlabel("Universality bin (equal count)")
        ax.set_title(f"{layer.capitalize()} (n = {len(m)} dimensions)")
    axes[0].set_ylabel("Proportion of dimensions")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=4, frameon=False)
    return fig


def _kernel_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray,
                  bw: float = 0.08) -> np.ndarray:
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2)
    return (w * y).sum(1) / w.sum(1)


def plot_universality_vs_content_continuous(agg: pd.DataFrame) -> plt.Figure:
    """Binning-free view: Gaussian-kernel regression of per-dimension vote
    shares (all four categories) on the within-layer universality percentile (rank scale, so layers
    are comparable without assuming absolute scores align), with a bootstrap
    95% band over dimensions."""
    stack_order = ["semantic", "both", "visual", "neither"]
    m_all = agg.dropna(subset=["universality"]).copy()
    m_all["pct"] = m_all.groupby("layer").universality.rank(pct=True)
    layers = [l for l in LAYER_ORDER if l in m_all.layer.unique()]
    fig, axes = make_figure(ncols=len(stack_order), sharey=True)
    grid = np.linspace(0.03, 0.97, 60)
    rng = np.random.default_rng(0)
    for ax, cat in zip(axes, stack_order):
        for layer in layers:
            g = m_all[m_all.layer == layer]
            x, y = g["pct"].values, g[f"share_{cat}"].values
            boots = np.stack([
                _kernel_curve(x[i], y[i], grid)
                for i in (rng.integers(0, len(x), len(x)) for _ in range(300))])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            ax.fill_between(grid, lo, hi, color=LAYER_COLORS[layer],
                            alpha=0.15, lw=0)
            ax.plot(grid, _kernel_curve(x, y, grid), color=LAYER_COLORS[layer],
                    lw=1.5, label=layer.capitalize() if cat == stack_order[0] else None)
        ax.set_xlabel("Universality percentile within layer")
        ax.set_title(cat.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("Share of raters' votes")
    fig.legend(loc="upper center", ncol=4, frameon=False)
    return fig


def _semantic_among_interpretable(agg: pd.DataFrame) -> pd.Series:
    """Semantic share counted over interpretable votes only.

    `share_semantic` and `share_neither` are compositional, so semantic share on
    the full vote set is partly just "not neither" — and "neither" means the
    rater saw no theme at all, which is close to the difficulty question itself.
    Renormalising over visual + semantic + both removes that circularity and asks
    the non-trivial question: among dimensions raters COULD read, does being more
    semantic make them easier to read?
    """
    denom = agg.share_semantic + agg.share_visual + agg.share_both
    return agg.share_semantic / denom.replace(0.0, np.nan)


def plot_semantic_vs_difficulty(agg: pd.DataFrame, real: pd.DataFrame) -> plt.Figure:
    """Semantic content against rated difficulty, before and after controlling
    for interpretability.

    Left: difficulty per content label. "Neither" is the outlier in every layer,
    which is the confound — it marks dimensions with no readable theme at all.
    Right: among interpretable dimensions only, difficulty against semantic
    share. Flat, i.e. semantic content and ease of interpretation are largely
    independent once uninterpretable dimensions are accounted for.
    """
    categories = ["visual", "semantic", "both", "neither"]
    # difficulty means occupy ~3.2-4.4 of the 1-7 scale; crop so differences are
    # legible, and state the full scale in the axis label so nothing is oversold
    ylim = (2.8, 4.8)
    layers = [l for l in LAYER_ORDER if l in agg.layer.unique()]
    fig, axes = make_figure(ncols=2)

    # left: the confound, made explicit
    ax = axes[0]
    # points with 95% CIs, not bars: the means occupy a narrow part of the 1-7
    # scale, and a cropped axis is only honest when there is no bar length to
    # misread. Layers are dodged so the four estimates never overlap.
    width = 0.62 / len(layers)
    for j, layer in enumerate(layers):
        g = real[real.layer == layer]
        xs, ms, cis = [], [], []
        for i, cat in enumerate(categories):
            v = g.loc[g.category == cat, "interpretability"].dropna()
            xs.append(i + (j - (len(layers) - 1) / 2) * width)
            ms.append(v.mean())
            cis.append(1.96 * v.std() / max(len(v), 1) ** 0.5)
        ax.errorbar(xs, ms, yerr=cis, fmt="o", ms=4, color=LAYER_COLORS[layer],
                    ecolor=LAYER_COLORS[layer], elinewidth=1.1, capsize=0,
                    mec="white", mew=0.5, label=layer.capitalize())
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_xlabel("Content label")
    ax.set_ylabel("Difficulty rating (1 = easy, 7 = difficult)")
    ax.set_ylim(*ylim)

    # right: the same question with the confound removed
    ax = axes[1]
    m_all = agg.dropna(subset=["mean_interpretability"]).copy()
    m_all["sem_rel"] = _semantic_among_interpretable(m_all)
    m_all = m_all.dropna(subset=["sem_rel"])
    grid = np.linspace(0.03, 0.97, 60)
    rng = np.random.default_rng(0)
    for j, layer in enumerate(layers):
        g = m_all[m_all.layer == layer]
        x, y = g["sem_rel"].to_numpy(), g["mean_interpretability"].to_numpy()
        # Kernel curves run on thin data at high semantic share (early has only
        # 19 dimensions above 0.6), where they wander in a way that reads as
        # structure. Cut each layer's curve at its own 95th percentile so the
        # drawn range is the range the data actually supports.
        gsub = grid[grid <= float(np.quantile(x, 0.95))]
        boots = np.stack([
            _kernel_curve(x[i], y[i], gsub)
            for i in (rng.integers(0, len(x), len(x)) for _ in range(300))])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(gsub, lo, hi, color=LAYER_COLORS[layer], alpha=0.15, lw=0)
        ax.plot(gsub, _kernel_curve(x, y, gsub), color=LAYER_COLORS[layer], lw=1.5)
        # density rug pinned to the panel floor in axes coords, so the cropped
        # y-range cannot hide it: where the dimensions actually are
        ax.plot(x, np.full_like(x, 0.012 + 0.022 * j), "|", ms=3, mew=0.5,
                alpha=0.45, color=LAYER_COLORS[layer],
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel("Semantic share among interpretable votes")
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=len(layers), frameon=False)
    return fig


def print_semantic_difficulty_stats(agg: pd.DataFrame, out: Path) -> None:
    """The raw and the controlled correlation side by side, so the drop from one
    to the other is on record rather than only in the figure."""
    m_all = agg.dropna(subset=["mean_interpretability"]).copy()
    m_all["sem_rel"] = _semantic_among_interpretable(m_all)
    print("\n=== Semantic content vs difficulty (Spearman, per layer) ===")
    rows = []
    for layer in [l for l in LAYER_ORDER if l in m_all.layer.unique()]:
        g = m_all[m_all.layer == layer]
        r_raw, p_raw = stats.spearmanr(g.share_semantic, g.mean_interpretability)
        r_nei, p_nei = stats.spearmanr(g.share_neither, g.mean_interpretability)
        h = g.dropna(subset=["sem_rel"])
        r_ctl, p_ctl = stats.spearmanr(h.sem_rel, h.mean_interpretability)
        print(f"  {layer:12s} n={len(g):4d}  raw semantic r={r_raw:+.3f} (p={p_raw:.1e})"
              f"  | neither r={r_nei:+.3f} (p={p_nei:.1e})"
              f"  | semantic among interpretable r={r_ctl:+.3f} (p={p_ctl:.1e}, n={len(h)})")
        rows.append(dict(layer=layer, n_dims=len(g),
                         raw_semantic_r=r_raw, raw_semantic_p=p_raw,
                         neither_r=r_nei, neither_p=p_nei,
                         semantic_among_interpretable_r=r_ctl,
                         semantic_among_interpretable_p=p_ctl,
                         n_interpretable=len(h)))
    pd.DataFrame(rows).to_csv(out / "semantic_vs_difficulty_stats.csv", index=False)


def print_universality_content_stats(agg: pd.DataFrame, out: Path) -> None:
    """Within each layer, Spearman correlation of
    dimension universality with the share of semantic / visual votes."""
    m_all = agg.dropna(subset=["universality"])
    print("\n=== Universality vs content shares (Spearman, per layer) ===")
    rows = []
    for layer in [l for l in LAYER_ORDER if l in m_all.layer.unique()]:
        m = m_all[m_all.layer == layer]
        line = f"  {layer} (n = {len(m)} dims):"
        for col in ["share_semantic", "share_visual", "share_both",
                    "share_neither"]:
            r, p = stats.spearmanr(m.universality, m[col])
            line += f"  {col.removeprefix('share_')} r={r:.2f} (p={p:.4f})"
            rows.append(dict(layer=layer, share=col.removeprefix("share_"),
                             n_dims=len(m), spearman_r=r, p=p))
        print(line)
    pd.DataFrame(rows).to_csv(out / "universality_content_stats.csv", index=False)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    all_trials = pd.read_csv(out / "parsed_all_trials.csv")
    # penultimate-layer trials from the main experiment (same task and QC)
    penult = pd.read_csv(
        Path("experiments/human_ratings/outputs/parsed_all_trials.csv"))
    penult["layer"] = "penultimate"
    penult["study"] = "penultimate"
    all_trials = pd.concat([all_trials, penult], ignore_index=True)

    qc = quality_check(all_trials)
    clean = all_trials.merge(qc[qc["pass"]][["participant", "study"]],
                             on=["participant", "study"])
    repeat_consistency(clean)

    real = clean[clean.task_type == "real"]
    rater_agreement(real)
    print("\n=== Trial times (real trials, per layer) ===")
    for layer, g in real.groupby("layer"):
        print(f"  {layer}: median {g.time_s.median():.1f} s, "
              f"5th pct {g.time_s.quantile(0.05):.1f} s")

    agg = dim_aggregates(real, rank=cfg.factorization.rank, dataset=cfg.dataset)
    agg.to_csv(out / "pilot_dim_aggregates.csv", index=False)

    print("\n=== Clean data ===")
    for layer, g in real.groupby("layer"):
        raters = g.groupby("dim_id").size()
        print(f"  {layer}: {g.participant.nunique()} participants, "
              f"{g.dim_id.nunique()} dimensions, "
              f"{raters.mean():.1f} raters/dimension (min {raters.min()})")
    print("\n=== Label proportions (trial level) ===")
    for layer, g in real.groupby("layer"):
        vc = g.category.value_counts(normalize=True)
        print(f"  {layer}: " + ", ".join(
            f"{c} {vc.get(c, 0):.0%}"
            for c in ["semantic", "visual", "both", "neither"]))
    print("\n=== Difficulty rating (1=easy, 7=difficult) ===")
    for layer, g in real.groupby("layer"):
        print(f"  {layer}: mean {g.interpretability.mean():.2f} "
              f"± {g.interpretability.std():.2f}")

    save(plot_content_by_layer(real), out / "main" / "content_by_layer")
    save(plot_interpretability_by_layer(real), out / "main" / "difficulty_by_content")
    print_universality_content_stats(agg, out)

    print_semantic_difficulty_stats(agg, out)
    save(plot_semantic_vs_difficulty(agg, real), out / "main" / "semantic_vs_difficulty")
    save(plot_universality_vs_content(agg), out / "main" / "universality_vs_content")
    save(plot_universality_vs_content_continuous(agg),
         out / "main" / "universality_vs_content_continuous")
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
