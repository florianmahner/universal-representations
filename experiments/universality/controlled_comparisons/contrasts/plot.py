from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.colors import (
    ROSE, INDIGO, PURPLE, GRAY, GRAY_DARK, GRAY_LIGHT, GRAY_PALE, CYCLE,
)
from src.plotting import setup_plotting, save_figure, CLASS_COLORS

PALETTE_CATEGORICAL = CYCLE + [INDIGO, PURPLE]

setup_plotting()

HEADER_FACTOR_ORDER = ["Diet", "Architecture", "Task"]
FACTOR_COLUMNS = {
    "Diet": ("train_data", "train_data_display"),
    "Architecture": ("architecture_family", None),
    "Task": ("train_task", "train_task_display"),
}


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 5000, ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def _shorten_model_name(name: str) -> str:
    for suffix in ["-P4-W7"]:
        name = name.replace(suffix, "")
    return name


def _derive_header_factors(df: pd.DataFrame, contrast_col: str) -> dict[str, str]:
    """Auto-derive fixed vs varying factors from the metadata.

    For each factor (Diet, Architecture, Task), checks if the corresponding
    metadata column has a single unique value (fixed) or multiple (varies).
    Goal contrasts (compare_goal_*) force Task to "varies" because the
    comparison IS across objectives even when train_task is constant.
    """
    sub = df[df[contrast_col].notna()]
    result = {}
    for factor_name in HEADER_FACTOR_ORDER:
        raw_col, display_col = FACTOR_COLUMNS[factor_name]
        if raw_col not in sub.columns:
            continue
        unique_vals = sub[raw_col].dropna().unique()
        if len(unique_vals) == 1:
            if display_col and display_col in sub.columns:
                result[factor_name] = sub[display_col].dropna().iloc[0]
            else:
                result[factor_name] = str(unique_vals[0])
        else:
            result[factor_name] = "varies"

    if contrast_col.startswith("compare_goal_") and result.get("Task") != "varies":
        result["Task"] = "varies"
    if contrast_col.startswith("compare_diet_") and result.get("Diet") != "varies":
        result["Diet"] = "varies"
    return result


def _group_display_labels(df: pd.DataFrame, contrast_col: str) -> dict[str, str]:
    """Map raw group values to display labels using metadata display columns.

    For each group value, checks if it matches a raw train_data or train_task
    value and substitutes the corresponding *_display column value.
    """
    result = {}
    for group_val in df[contrast_col].dropna().unique():
        sub = df[df[contrast_col] == group_val]
        if "train_data" in sub.columns and "train_data_display" in sub.columns:
            if (sub["train_data"] == group_val).any():
                result[group_val] = sub.loc[sub["train_data"] == group_val, "train_data_display"].iloc[0]
                continue
        if "train_task" in sub.columns and "train_task_display" in sub.columns:
            if (sub["train_task"] == group_val).any():
                result[group_val] = sub.loc[sub["train_task"] == group_val, "train_task_display"].iloc[0]
                continue
        result[group_val] = str(group_val)
    return result


def _auto_group_colors(group_order: list[str]) -> dict[str, str]:
    """CLASS_COLORS for model_class values, PALETTE_CATEGORICAL cycling otherwise."""
    result = {}
    pal_idx = 0
    for g in group_order:
        if g in CLASS_COLORS:
            result[g] = CLASS_COLORS[g]
        else:
            result[g] = PALETTE_CATEGORICAL[pal_idx % len(PALETTE_CATEGORICAL)]
            pal_idx += 1
    return result


def _is_all_singleton(df: pd.DataFrame, contrast_col: str) -> bool:
    return (df.groupby(contrast_col)["model"].nunique() == 1).all()


def _draw_header(
    fig: plt.Figure,
    fixed_factors: dict[str, str],
    header_rect: tuple[float, float, float, float],
) -> None:
    ax = fig.add_axes(header_rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(HEADER_FACTOR_ORDER)
    box_w, box_h, gap = 0.20, 0.55, 0.12
    total_used = n * box_w + (n - 1) * gap
    x_start = (1.0 - total_used) / 2

    positions = [x_start + i * (box_w + gap) + box_w / 2 for i in range(n)]

    for i, factor in enumerate(HEADER_FACTOR_ORDER):
        cx = positions[i]
        value = fixed_factors.get(factor, "")
        is_varying = "varies" in value.lower() if value else False

        ec = PURPLE if is_varying else GRAY_LIGHT
        fc = "#fff0f0" if is_varying else "#fafafa"
        tc = PURPLE if is_varying else GRAY_DARK
        lw = 1.8 if is_varying else 0.8
        fw = "bold" if is_varying else "normal"

        bbox = mpatches.FancyBboxPatch(
            (cx - box_w / 2, 0.50 - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(bbox)
        ax.text(cx, 0.60, factor, ha="center", va="center",
                fontsize=9, fontweight=fw, color=tc)

        display_val = "varies" if is_varying else value
        vc = tc if is_varying else GRAY
        ax.text(cx, 0.40, display_val, ha="center", va="center",
                fontsize=7, color=vc,
                style="normal" if is_varying else "italic",
                fontweight="bold" if is_varying else "normal")

        if i < n - 1:
            margin = gap * 0.28
            ax.annotate(
                "", xy=(positions[i + 1] - box_w / 2 - margin, 0.50),
                xytext=(cx + box_w / 2 + margin, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=GRAY_DARK, lw=1.3, mutation_scale=11),
            )


def plot_distribution_contrast(
    uni_per_dim: pd.DataFrame,
    contrast_col: str,
    df_merged: pd.DataFrame,
    group_order: list[str] | None = None,
    fixed_factors: dict[str, str] | None = None,
) -> plt.Figure:
    """Violin + strip for singleton contrasts (each group = 1 model).

    Shows the distribution of per-dimension universality for each model.
    Grand mean + CI band shows the overall reference level.
    """
    if group_order is None:
        medians = uni_per_dim.groupby(contrast_col)["universality"].median().sort_values(ascending=False)
        group_order = medians.index.tolist()
    else:
        group_order = [g for g in group_order if g in uni_per_dim[contrast_col].unique()]

    sub = uni_per_dim[uni_per_dim[contrast_col].isin(group_order)].copy()
    n_groups = len(group_order)
    color_map = _auto_group_colors(group_order)
    display_labels = _group_display_labels(df_merged, contrast_col)

    has_header = fixed_factors is not None
    header_h = 0.65 if has_header else 0.0
    fig_w = max(3.5, n_groups * 0.7 + 1.0)
    total_h = 3.2 + header_h
    fig = plt.figure(figsize=(fig_w, total_h))

    main_bottom = 0.28
    main_top = 1.0 - (header_h / total_h) - 0.02
    ax = fig.add_axes([0.12, main_bottom, 0.85, main_top - main_bottom])
    ax.set_facecolor("white")
    ax.grid(False)

    if has_header:
        hdr_bot = main_top + 0.005
        _draw_header(fig, fixed_factors, (0.15, hdr_bot, 0.70, 1.0 - hdr_bot - 0.005))

    palette = {g: color_map[g] for g in group_order}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=sub, x=contrast_col, y="universality",
            order=group_order, hue=contrast_col, palette=palette,
            legend=False, inner=None, cut=0, ax=ax,
            linewidth=0, alpha=0.25,
        )
        sns.boxplot(
            data=sub, x=contrast_col, y="universality",
            order=group_order, hue=contrast_col, palette=palette,
            legend=False, width=0.18,
            boxprops=dict(alpha=0.7, linewidth=0.6),
            whiskerprops=dict(alpha=0.7, linewidth=0.6),
            capprops=dict(alpha=0.7, linewidth=0.6),
            medianprops=dict(color=GRAY_DARK, linewidth=1.5),
            fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x=contrast_col, y="universality",
            order=group_order, color=GRAY_DARK,
            alpha=0.35, size=2, jitter=0.08, ax=ax,
        )

    model_means = sub.groupby(contrast_col)["universality"].mean()
    grand_mean = model_means.mean()
    ci_lo, ci_hi = _bootstrap_ci(model_means.values)
    ax.axhspan(ci_lo, ci_hi, color=GRAY_PALE, alpha=0.5, linewidth=0, zorder=0)
    ax.axhline(grand_mean, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    for i, g in enumerate(group_order):
        ax.plot(i, model_means[g], marker="D", color=GRAY_DARK,
                markersize=4.5, zorder=5, markeredgecolor="white", markeredgewidth=0.4)

    labels = [display_labels.get(g, g) for g in group_order]
    tick_fs = 8 if n_groups <= 8 else (6.5 if n_groups <= 15 else 5)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_xlabel("")
    ax.set_ylabel("Universality", fontsize=9)
    return fig


def plot_conwell_bar(
    model_data: pd.DataFrame,
    contrast_col: str,
    df_merged: pd.DataFrame,
    y_col: str = "mean_univ",
    y_label: str = "Universality",
    y_lo_col: str | None = "ci_lo",
    y_hi_col: str | None = "ci_hi",
    group_order: list[str] | None = None,
    fixed_factors: dict[str, str] | None = None,
    label_col: str = "label",
) -> plt.Figure:
    """Metric-agnostic Conwell-style per-model bar chart.

    Parameters
    ----------
    model_data : DataFrame
        One row per model with contrast_col, y_col, label_col columns.
    y_lo_col, y_hi_col : str or None
        Columns for error bar bounds. None disables model-level error bars.
    """
    model_stats = model_data.copy()
    model_stats["group"] = model_stats[contrast_col]

    if group_order is None:
        group_means = model_stats.groupby("group")[y_col].mean().sort_values(ascending=False)
        group_order = group_means.index.tolist()
    else:
        group_order = [g for g in group_order if g in model_stats["group"].values]

    model_stats = model_stats[model_stats["group"].isin(group_order)]
    model_stats = pd.concat(
        [model_stats[model_stats["group"] == g].sort_values(y_col) for g in group_order],
        ignore_index=True,
    )
    model_stats["x_pos"] = range(len(model_stats))
    n_models = len(model_stats)

    group_color_map = _auto_group_colors(group_order)
    display_labels = _group_display_labels(df_merged, contrast_col)

    has_header = fixed_factors is not None
    header_h = 0.65 if has_header else 0.0
    fig_w = max(4.5, n_models * 0.3 + 1.0)
    total_h = 3.2 + header_h
    fig = plt.figure(figsize=(fig_w, total_h))

    main_bottom = 0.35
    main_top = 1.0 - (header_h / total_h) - 0.02
    ax = fig.add_axes([0.09, main_bottom, 0.89, main_top - main_bottom])
    ax.grid(False)

    if has_header:
        hdr_bot = main_top + 0.005
        _draw_header(fig, fixed_factors, (0.15, hdr_bot, 0.70, 1.0 - hdr_bot - 0.005))

    has_error = (
        y_lo_col is not None
        and y_hi_col is not None
        and y_lo_col in model_stats.columns
        and y_hi_col in model_stats.columns
    )

    for _, row in model_stats.iterrows():
        color = group_color_map[row["group"]]
        ax.bar(row["x_pos"], row[y_col], width=0.72, color=color, alpha=0.82,
               edgecolor="white", linewidth=0.3)
        if has_error:
            ax.errorbar(row["x_pos"], row[y_col],
                        yerr=[[row[y_col] - row[y_lo_col]], [row[y_hi_col] - row[y_col]]],
                        fmt="none", ecolor=GRAY_DARK, elinewidth=0.7, capsize=1.5, capthick=0.7)

    group_ranges = {}
    for g in group_order:
        grp = model_stats[model_stats["group"] == g]
        group_ranges[g] = (grp["x_pos"].min() - 0.45, grp["x_pos"].max() + 0.45)

    for g in group_order:
        grp = model_stats[model_stats["group"] == g]
        group_y = grp[y_col].values
        group_mean = group_y.mean()
        g_lo, g_hi = _bootstrap_ci(group_y) if len(group_y) > 1 else (group_mean, group_mean)
        x_lo, x_hi = group_ranges[g]
        color = group_color_map[g]
        ax.fill_between([x_lo, x_hi], g_lo, g_hi, alpha=0.13, color=color,
                        hatch="///", edgecolor=color, linewidth=0.0)
        ax.hlines(group_mean, x_lo, x_hi, colors=color, linestyles="dashed", linewidth=1.0, alpha=0.7)

    for i in range(len(group_order) - 1):
        x_sep = (group_ranges[group_order[i]][1] + group_ranges[group_order[i + 1]][0]) / 2
        ax.axvline(x_sep, color=GRAY_LIGHT, linestyle=":", linewidth=0.7, alpha=0.6)

    if has_error:
        all_lo = model_stats[y_lo_col].min()
        all_hi = model_stats[y_hi_col].max()
    else:
        all_lo = model_stats[y_col].min()
        all_hi = model_stats[y_col].max()
    y_pad = max((all_hi - all_lo) * 0.25, 0.01)
    ax.set_ylim(all_lo - y_pad, all_hi + y_pad)

    tick_fs = 6.5 if n_models <= 12 else (5.5 if n_models <= 25 else 4.5)
    ax.set_xticks(model_stats["x_pos"].values)
    ax.set_xticklabels(model_stats[label_col].values, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_xlim(-0.6, n_models - 0.4)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="x", length=0, pad=2)

    all_singleton = all(len(model_stats[model_stats["group"] == g]) == 1 for g in group_order)
    if not all_singleton:
        max_label_len = max(len(str(l)) for l in model_stats[label_col].values)
        offset_pt = -(max_label_len * 3.0 * np.sin(np.radians(45)) + 14)
        for g in group_order:
            grp = model_stats[model_stats["group"] == g]
            display_g = display_labels.get(g, g)
            ax.annotate(
                f"{display_g} (n={len(grp)})",
                xy=(grp["x_pos"].mean(), 0), xycoords=("data", "axes fraction"),
                xytext=(0, offset_pt), textcoords="offset points",
                ha="center", va="top", fontsize=7, fontweight="bold",
                color=group_color_map[g], annotation_clip=False,
            )

    return fig


def plot_conwell_contrast(
    uni_per_dim: pd.DataFrame,
    df_merged: pd.DataFrame,
    contrast_col: str,
    group_order: list[str] | None = None,
    fixed_factors: dict[str, str] | None = None,
    name_col: str = "display_name",
    sort_within_group: str = "universality",
) -> plt.Figure:
    """Conwell et al. (2024)-style bar chart for multi-model contrasts."""
    for col_candidate in [name_col, "name", "model"]:
        if col_candidate in uni_per_dim.columns:
            label_col = col_candidate
            break
    else:
        label_col = "model"

    model_stats = (
        uni_per_dim.groupby("model")
        .agg(mean_univ=("universality", "mean"),
             group=(contrast_col, "first"),
             label=(label_col, "first"))
        .reset_index()
    )
    model_stats["label"] = model_stats["label"].apply(_shorten_model_name)
    model_stats[contrast_col] = model_stats["group"]

    model_stats["ci_lo"] = 0.0
    model_stats["ci_hi"] = 0.0
    for idx, row in model_stats.iterrows():
        vals = uni_per_dim.loc[uni_per_dim["model"] == row["model"], "universality"].values
        lo, hi = _bootstrap_ci(vals)
        model_stats.at[idx, "ci_lo"] = lo
        model_stats.at[idx, "ci_hi"] = hi

    return plot_conwell_bar(
        model_data=model_stats,
        contrast_col=contrast_col,
        df_merged=df_merged,
        y_col="mean_univ",
        y_label="Universality",
        y_lo_col="ci_lo",
        y_hi_col="ci_hi",
        group_order=group_order,
        fixed_factors=fixed_factors,
        label_col="label",
    )


def plot_paired_bars(
    uni_per_dim: pd.DataFrame,
    df_merged: pd.DataFrame,
    group_col: str = "train_data",
    group_a: str = "imagenet",
    group_b: str = "imagenet21k",
    match_col: str = "architecture",
    fixed_factors: dict[str, str] | None = None,
) -> plt.Figure:
    """Paired bar chart: two bars per matched architecture, one per group."""
    display_labels = _group_display_labels(df_merged, group_col)
    label_a = display_labels.get(group_a, group_a)
    label_b = display_labels.get(group_b, group_b)

    model_stats = (
        uni_per_dim.groupby("model").agg(mean_univ=("universality", "mean")).reset_index()
    )
    info_cols = [c for c in ["model", "display_name", "architecture", group_col] if c in df_merged.columns]
    model_stats = model_stats.merge(df_merged[info_cols].drop_duplicates(subset=["model"]), on="model")

    stats_a = model_stats[model_stats[group_col] == group_a]
    stats_b = model_stats[model_stats[group_col] == group_b]
    matched_archs = sorted(set(stats_a[match_col]) & set(stats_b[match_col]))

    pairs = []
    for arch in matched_archs:
        ra = stats_a[stats_a[match_col] == arch].iloc[0]
        rb = stats_b[stats_b[match_col] == arch].iloc[0]
        va = uni_per_dim[uni_per_dim["model"] == ra["model"]]["universality"].values
        vb = uni_per_dim[uni_per_dim["model"] == rb["model"]]["universality"].values
        pairs.append({
            "label": _shorten_model_name(ra["display_name"]),
            "mean_a": ra["mean_univ"], "ci_a": _bootstrap_ci(va),
            "mean_b": rb["mean_univ"], "ci_b": _bootstrap_ci(vb),
        })
    pairs.sort(key=lambda p: p["mean_a"])
    n = len(pairs)
    if n == 0:
        return None

    has_header = fixed_factors is not None
    header_h = 0.65 if has_header else 0.0
    fig_w = max(4.5, n * 0.65 + 1.5)
    total_h = 3.2 + header_h
    fig = plt.figure(figsize=(fig_w, total_h))

    main_bottom = 0.30
    main_top = 1.0 - (header_h / total_h) - 0.02
    ax = fig.add_axes([0.10, main_bottom, 0.87, main_top - main_bottom])
    ax.grid(False)

    if has_header:
        hdr_bot = main_top + 0.005
        _draw_header(fig, fixed_factors, (0.15, hdr_bot, 0.70, 1.0 - hdr_bot - 0.005))

    w = 0.35
    color_a, color_b = INDIGO, ROSE
    for i, p in enumerate(pairs):
        ax.bar(i - w / 2, p["mean_a"], w, color=color_a, alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.errorbar(i - w / 2, p["mean_a"],
                    yerr=[[p["mean_a"] - p["ci_a"][0]], [p["ci_a"][1] - p["mean_a"]]],
                    fmt="none", ecolor=GRAY_DARK, elinewidth=0.7, capsize=1.5, capthick=0.7)
        ax.bar(i + w / 2, p["mean_b"], w, color=color_b, alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.errorbar(i + w / 2, p["mean_b"],
                    yerr=[[p["mean_b"] - p["ci_b"][0]], [p["ci_b"][1] - p["mean_b"]]],
                    fmt="none", ecolor=GRAY_DARK, elinewidth=0.7, capsize=1.5, capthick=0.7)

    all_lo = min(min(p["ci_a"][0] for p in pairs), min(p["ci_b"][0] for p in pairs))
    all_hi = max(max(p["ci_a"][1] for p in pairs), max(p["ci_b"][1] for p in pairs))
    y_pad = max((all_hi - all_lo) * 0.25, 0.01)
    ax.set_ylim(all_lo - y_pad, all_hi + y_pad)

    tick_fs = 7 if n <= 8 else (5.5 if n <= 15 else 4.5)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([p["label"] for p in pairs], rotation=45, ha="right", fontsize=tick_fs)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Universality", fontsize=9)
    ax.tick_params(axis="x", length=0, pad=2)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=color_a, alpha=0.82, label=label_a),
                 Patch(facecolor=color_b, alpha=0.82, label=label_b)],
        fontsize=7, loc="upper left", frameon=False,
    )
    return fig


def _prepare_uni_with_info(
    uni_per_dim: pd.DataFrame,
    df_merged: pd.DataFrame,
    contrast_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    cols = ["model", "display_name", contrast_col]
    if extra_cols:
        cols.extend(c for c in extra_cols if c not in cols)
    available = [c for c in cols if c in df_merged.columns]
    info = df_merged[available].drop_duplicates(subset=["model"])
    merged = uni_per_dim.merge(info, on="model")
    return merged[merged[contrast_col].notna()]


def generate_all_deepnsd_conwell(
    uni_per_dim: pd.DataFrame,
    df_merged: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate contrast figures for all DeepNSD compare_* columns.

    Dispatches automatically:
    - Paired bars for diet_imagenetsize (matched architecture pairs)
    - Distribution plot for singleton contrasts (all groups n=1)
    - Conwell bar chart for multi-model contrasts
    Headers are derived from the data, not hardcoded.
    """
    from data import COMPARE_COLS

    for col in COMPARE_COLS:
        sub = _prepare_uni_with_info(uni_per_dim, df_merged, col)
        if sub[col].nunique() < 2:
            continue

        fixed = _derive_header_factors(df_merged, col)

        if col == "compare_diet_imagenetsize":
            sub_info = df_merged[df_merged[col].notna()]
            fixed["Architecture"] = "matched"
            fig_grouped = plot_conwell_contrast(
                uni_per_dim=sub, df_merged=sub_info,
                contrast_col=col, fixed_factors=fixed,
            )
            save_figure(fig_grouped, output_dir / "diet_imagenetsize_grouped", formats=("pdf",))
            print(f"    Saved diet_imagenetsize_grouped")
            fig_paired = plot_paired_bars(
                uni_per_dim=sub, df_merged=sub_info, fixed_factors=fixed,
            )
            if fig_paired is not None:
                save_figure(fig_paired, output_dir / "diet_imagenetsize_paired", formats=("pdf",))
                print(f"    Saved diet_imagenetsize_paired")
            continue
        elif _is_all_singleton(sub, col):
            fig = plot_distribution_contrast(uni_per_dim=sub, contrast_col=col,
                                              df_merged=df_merged, fixed_factors=fixed)
        else:
            fig = plot_conwell_contrast(uni_per_dim=sub, df_merged=df_merged[df_merged[col].notna()],
                                        contrast_col=col, fixed_factors=fixed)

        save_figure(fig, output_dir / col.replace("compare_", ""), formats=("pdf",))
        print(f"    Saved {col.replace('compare_', '')}")


def _build_architecture_clip(df_merged: pd.DataFrame):
    sub = df_merged[df_merged["train_task"] == "clip"].copy()
    return sub, "model_class", "architecture_clip"


def _build_seer_finetuning(df_merged: pd.DataFrame):
    sub = df_merged[df_merged["source"] == "seer"].copy()
    sub["finetune_status"] = sub["display_name"].apply(
        lambda x: "Finetuned" if "INFT" in str(x) else "Pretrained"
    )
    return sub, "finetune_status", "seer_finetuning"


NEW_CONTRAST_BUILDERS = [
    _build_architecture_clip,
    _build_seer_finetuning,
]


def generate_new_contrasts(
    uni_per_dim: pd.DataFrame,
    df_merged: pd.DataFrame,
    output_dir: Path,
) -> list[dict]:
    """Generate figures for custom controlled contrasts."""
    from contrasts.stats.stats import run_contrast_stats

    all_stats = []
    for builder in NEW_CONTRAST_BUILDERS:
        sub_info, group_col, output_name = builder(df_merged)

        if len(sub_info) < 2 or sub_info[group_col].nunique() < 2:
            print(f"    Skipping {output_name}: insufficient data")
            continue

        stat_sub = sub_info[["model", "universality", group_col]].copy()
        stat_result = run_contrast_stats(stat_sub, group_col)
        stat_result["contrast"] = output_name
        all_stats.append(stat_result)

        uni_sub = _prepare_uni_with_info(uni_per_dim, sub_info, group_col)
        if len(uni_sub) == 0:
            continue

        fixed = _derive_header_factors(sub_info, group_col)
        fig = plot_conwell_contrast(
            uni_per_dim=uni_sub, df_merged=sub_info,
            contrast_col=group_col, fixed_factors=fixed,
        )
        save_figure(fig, output_dir / output_name, formats=("pdf",))
        print(f"    Saved {output_name}")

    return all_stats


def generate_encoding_contrasts(
    df: pd.DataFrame,
    df_merged: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate Conwell-style encoding bar charts for all contrasts."""
    from data import COMPARE_COLS

    for col in COMPARE_COLS:
        sub = df[df[col].notna() & df["encoding"].notna()].copy()
        if sub[col].nunique() < 2:
            continue

        name_col = "display_name" if "display_name" in sub.columns else "model"
        sub["label"] = sub[name_col].apply(_shorten_model_name)
        sub["enc_lo"] = sub[["encoding_F", "encoding_N"]].min(axis=1)
        sub["enc_hi"] = sub[["encoding_F", "encoding_N"]].max(axis=1)

        fixed = _derive_header_factors(df_merged, col)

        fig = plot_conwell_bar(
            model_data=sub,
            contrast_col=col,
            df_merged=df_merged,
            y_col="encoding",
            y_label="IT encoding (mean r)",
            y_lo_col="enc_lo",
            y_hi_col="enc_hi",
            fixed_factors=fixed,
            label_col="label",
        )
        safe = col.replace("compare_", "")
        save_figure(fig, output_dir / f"encoding_{safe}", formats=("pdf",))
        print(f"    Saved encoding_{safe}")
