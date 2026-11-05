from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def dimension_level(dataset: str, rank: int) -> pd.DataFrame:
    """Pool the per-dimension universality scores of every control condition."""
    # display name of the pseudo-model -> the identifier its scoring table uses
    pseudo_names = {
        "gabor512": "Gabor bank",
        "pixels_rgb224": "raw pixels",
        "pixels_pca50": "pixel PCA-50",
        "pixel_stats18": "pixel statistics",
        "pixels_shuffled224": "pixels, spatially shuffled",
    }
    frames = []

    # trained reference: the paper's canonical penultimate results (depth 1.0),
    # never recomputed in the review experiments
    pen = pd.read_csv(
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    pen = pen[pen.metric == "cos2_perm_adj"]
    frames.append(
        pd.DataFrame(
            {
                "control": "trained models (reference)",
                "model": pen.model,
                "universality": pen.universality,
            }
        )
    )

    # untrained networks, from the layer-generality scoring
    lg = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/outputs")
        / f"universality_full_rank_{rank}.csv"
    )
    lg = lg[(lg.scoring == "vs_ref162") & (lg.metric == "cos2_perm_adj")]
    g = lg[lg.role == "untrained"]
    frames.append(
        pd.DataFrame(
            {
                "control": "untrained networks",
                "model": g["repr"],
                "universality": g.universality,
            }
        )
    )

    # pseudo-models
    p = (
        Path("experiments/rebuttal/baselines/pixel_models/outputs")
        / f"universality_pseudo_rank_{rank}.csv"
    )
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.metric == "cos2_perm_adj"] if "metric" in d else d
        col = "repr" if "repr" in d else "model"
        d["control"] = d[col].map(pseudo_names)
        frames.append(d[["control", col, "universality"]].rename(columns={col: "model"}))
    else:
        log.warning(f"missing {p} - pseudo-models not scored yet")

    # random features
    r = (
        Path("experiments/rebuttal/baselines/random_features/outputs")
        / f"universality_random_rank_{rank}.csv"
    )
    if r.exists():
        d = pd.read_csv(r)
        d = d[d.metric == "cos2_perm_adj"] if "metric" in d else d
        col = "repr" if "repr" in d else "model"
        frames.append(
            pd.DataFrame(
                {
                    "control": "random features",
                    "model": d[col],
                    "universality": d.universality,
                }
            )
        )
    else:
        log.warning(f"missing {r}")

    return pd.concat(frames, ignore_index=True).dropna(subset=["control"])


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # display name -> (what it is, why we ran it); also fixes the report order
    controls = {
        "trained models (reference)": "the paper's 162 trained networks, penultimate layer",
        "untrained networks": "random-init networks, architecture only, no learning",
        "Gabor bank": "handcrafted low-level vision filters, no learning",
        "raw pixels": "centre-cropped RGB pixels, zero free choices",
        "pixel PCA-50": "top 50 principal components of the pixels",
        "pixel statistics": "simple per-image colour/intensity summaries",
        "pixels, spatially shuffled": "per-image pixel permutation: colour kept, spatial structure destroyed",
        "random features": "Gaussian noise, no images involved at all",
    }

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    dim = dimension_level(cfg.dataset, cfg.factorization.rank)
    dim.to_csv(out / "floor_controls_dimension_level.csv", index=False)

    summary = (
        dim.groupby("control")
        .agg(
            n_dimensions=("universality", "size"),
            n_representations=("model", "nunique"),
            mean=("universality", "mean"),
            sd=("universality", "std"),
            median=("universality", "median"),
        )
        .reindex([c for c in controls if c in set(dim.control)])
    )
    summary["what_it_is"] = [controls[c] for c in summary.index]
    ref = summary.loc["trained models (reference)", "mean"]
    summary["pct_of_trained"] = (summary["mean"] / ref * 100).round(1)
    summary.to_csv(out / "floor_controls_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
