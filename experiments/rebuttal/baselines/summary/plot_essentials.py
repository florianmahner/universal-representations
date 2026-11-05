from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    keep = [
        ("trained models (reference)", "Trained models"),
        ("untrained networks", "Untrained networks"),
        ("pixel PCA-50", "First 50 PCs of the images"),
        ("random features", "Random noise"),
    ]
    color = {"Trained models": "#1b4965", "Untrained networks": "#5fa8d3"}
    gray = "#aeb6c2"

    dim = pd.read_csv(out / "floor_controls_dimension_level.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    labels, data = [], []
    for key, label in keep:
        labels.append(label)
        data.append(dim.loc[dim.control == key, "universality"].values)

    pos = np.arange(len(labels))[::-1]
    parts = ax.violinplot(data, positions=pos, vert=False, widths=0.8,
                          showextrema=False)
    for pc, label in zip(parts["bodies"], labels):
        pc.set_facecolor(color.get(label, gray))
        pc.set_alpha(0.8)
        pc.set_edgecolor("none")
    for p, v in zip(pos, data):
        ax.plot(np.median(v), p, "o", color="white", ms=5, zorder=3)
        ax.plot(np.median(v), p, "o", color="#22252b", ms=3, zorder=4)

    floor = dim.loc[dim.control == "random features", "universality"].mean()
    ax.axvline(floor, color="#c1121f", lw=1, ls="--", zorder=0)
    ax.text(floor + 0.003, pos[0] + 0.55, "noise floor of the metric",
            color="#c1121f", fontsize=8, va="bottom")

    ax.set_yticks(pos)
    sub = {"Trained models": "162 models",
           "Untrained networks": "79 architectures",
           "First 50 PCs of the images": "one representation, 50 dimensions",
           "Random noise": "5 draws, 250 dimensions"}
    ax.set_yticklabels([f"{l}\n({sub[l]})" for l in labels], fontsize=9)
    ax.set_xlabel("Universality per dimension (cos², permutation-adjusted)")
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"floor_controls_essentials.{ext}", dpi=300,
                    bbox_inches="tight")
    print("wrote", out / "floor_controls_essentials.pdf")


if __name__ == "__main__":
    main()
