#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.colors import INDIGO
from src.plotting import create_figure, save_figure, setup_plotting

LOG = logging.getLogger(__name__)


def load_encoding_results(monkey: str) -> pd.DataFrame:
    path = Path(
        "experiments/universality/downstream_predictions/macaque_encoding"
        "/encoding_things/outputs"
    ) / f"encoding_results_{monkey}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Encoding results not found: {path}")
    return pd.read_csv(path)


def shorten_label(label: str, max_len: int = 20) -> str:
    if len(label) <= max_len:
        return label
    if "," in label:
        parts = [p.strip() for p in label.split(",")]
        if len(parts) > 2:
            return f"{parts[0]}+{len(parts)-1}more"
    return label[: max_len - 1] + "..."


def plot_encoding_by_group(
    df_f: pd.DataFrame,
    df_n: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    min_n: int = 5,
) -> None:
    """Boxplot + strip of encoding by model group, sorted by median."""
    setup_plotting()
    merged = df_f[["model", "mean_r"]].merge(
        df_n[["model", "mean_r"]], on="model", suffixes=("_F", "_N"),
    )
    merged["mean_r"] = (merged["mean_r_F"] + merged["mean_r_N"]) / 2
    idx = df_f.set_index("model")
    for col in ["model_class", "train_task_display", "train_data_display"]:
        merged[col] = idx.loc[merged["model"], col].values

    properties = [
        ("model_class", "Architecture"),
        ("train_task_display", "Training objective"),
        ("train_data_display", "Training data"),
    ]
    for display_col, prop_label in properties:
        sub = merged.dropna(subset=[display_col, "mean_r"])
        counts = sub[display_col].value_counts()
        valid_groups = counts[counts >= min_n].index
        sub = sub[sub[display_col].isin(valid_groups)]
        order = sub.groupby(display_col)["mean_r"].median().sort_values().index.tolist()
        n_groups = len(order)
        if n_groups < 2:
            continue

        rng = np.random.default_rng(seed)
        data_per_group = [sub[sub[display_col] == g]["mean_r"].values for g in order]

        fig, ax = create_figure("wide")

        ax.boxplot(
            data_per_group, positions=range(n_groups), vert=False, widths=0.5,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=INDIGO, alpha=0.15, edgecolor=INDIGO, lw=0.8),
            medianprops=dict(color=INDIGO, lw=1.5),
            whiskerprops=dict(color="#999999", lw=0.8),
            capprops=dict(color="#999999", lw=0.8),
        )

        for i, vals in enumerate(data_per_group):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                vals, i + jitter, s=12, alpha=0.5, color=INDIGO,
                edgecolors="white", linewidth=0.3, zorder=3,
            )

        labels = [f"{shorten_label(str(g))} ({counts[g]})" for g in order]
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylim(-0.5, n_groups - 0.5)
        ax.set_xlabel("IT encoding (mean r, avg F+N)")
        fig.tight_layout()
        safe_name = prop_label.lower().replace(" ", "_")
        save_figure(fig, output_dir / f"encoding_by_{safe_name}")
        LOG.info("Saved encoding_by_%s", safe_name)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    df_f = load_encoding_results("F")
    df_n = load_encoding_results("N")
    plot_encoding_by_group(df_f, df_n, out, seed=cfg.seed)


if __name__ == "__main__":
    main()
