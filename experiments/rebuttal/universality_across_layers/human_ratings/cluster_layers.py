"""Cluster the dimensions of one layer depth across models and pick one
representative dimension per cluster, the rating set for the layer-wise study."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist

from experiments.universality.dimensions.clustering.run import compute_linkage

log = logging.getLogger(__name__)

# Same dendrogram cut as the penultimate-layer rating study.
THRESHOLD = 0.75


def load_layer_dimensions(
    layer_tag: str,
    sel: pd.DataFrame,
    embeddings: Path,
    rank: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack all dimensions of one layer: (n_dims, 22248) plus their index."""
    reprs = sorted(r for r in sel.model if r.endswith(f"__{layer_tag}"))
    vectors, records = [], []
    for name in reprs:
        row = sel[sel.model == name].iloc[0]
        w = np.load(embeddings / name / f"rank_{rank}" / f"sigma_{row.sigma_mult}" /
                    f"seed_{int(row.best_seed)}" / "w.npz")["w"]
        base = name.rsplit("__", 1)[0]
        for d in range(w.shape[1]):
            vectors.append(w[:, d])
            records.append({"model": base, "repr": name, "dimension": d})
    W_all = np.asarray(vectors, dtype=np.float32)
    index = pd.DataFrame(records)
    index["dim_id"] = index["repr"] + "__d" + index.dimension.astype(str)
    return W_all, index


def representatives(W_all: np.ndarray, index: pd.DataFrame,
                    labels: np.ndarray) -> pd.DataFrame:
    """One dimension per cluster: the one closest to the cluster centroid."""
    chosen = []
    for cid in sorted(set(labels)):
        member_idx = np.where(labels == cid)[0]
        W_cluster = W_all[member_idx]
        centroid = W_cluster.mean(axis=0, keepdims=True)
        dists = cdist(centroid, W_cluster, metric="correlation").ravel()
        chosen.append(member_idx[int(np.argmin(dists))])
    out = index.iloc[chosen].copy().reset_index(drop=True)
    out["cluster"] = [labels[i] for i in chosen]
    out["cluster_size"] = [int((labels == labels[i]).sum()) for i in chosen]
    return out


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    embeddings = Path("experiments/rebuttal/universality_across_layers/factorization/outputs/embeddings")
    sel = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/factorization/outputs")
        / f"bandwidth_selection_full_rank_{rank}.csv"
    )

    for layer, tag in {"early": "d010", "middle": "d050", "late": "d090"}.items():
        log.info(f"=== {layer} layer ({tag})")
        W_all, index = load_layer_dimensions(tag, sel, embeddings, rank)
        log.info(f"  {W_all.shape[0]} dimensions from {index['repr'].nunique()} models")

        Z, _ = compute_linkage(W_all)
        np.save(out / f"linkage_{layer}.npy", Z)

        labels = fcluster(Z, t=THRESHOLD, criterion="distance")
        index["cluster"] = labels
        index.to_csv(out / f"clusters_{layer}.csv", index=False)

        reps = representatives(W_all, index, labels)
        reps.to_csv(out / f"representatives_{layer}.csv", index=False)
        log.info(f"  {len(set(labels))} clusters -> {len(reps)} representatives "
                 f"(largest cluster {reps.cluster_size.max()})")


if __name__ == "__main__":
    main()
