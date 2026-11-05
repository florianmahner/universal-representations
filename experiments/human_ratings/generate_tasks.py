from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist

from experiments.universality.shared import load_components_for_rank

THRESHOLD = 0.75  # correlation-distance cut used to define dimension clusters

log = logging.getLogger(__name__)


def select_dimensions(
    W_all: np.ndarray,
    dim_index: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Select 1 representative dimension per cluster (re-cut at *threshold*).

    Steps:
        1. Load precomputed linkage matrix and cut at *threshold* (correlation
           distance) to obtain cluster assignments.
        2. For each cluster, compute the centroid (mean W vector) and pick the
           dimension with highest correlation to that centroid.
        3. Merge LLM ratings to attach content labels to each selected dimension.

    Returns a DataFrame with columns: dim_id, model, dimension, cluster, content.
    """
    linkage_path = Path(
        "experiments/universality/dimensions/clustering/outputs/linkage.npy"
    )
    if not linkage_path.exists():
        raise FileNotFoundError(
            f"Linkage matrix not found at {linkage_path}.\n"
            f"Run the clustering pipeline first."
        )

    Z = np.load(linkage_path)
    labels = fcluster(Z, t=threshold, criterion="distance")
    n_clusters = len(set(labels))
    log.info(f"Re-cut linkage at threshold={threshold}: {n_clusters} clusters")

    if len(labels) != len(W_all):
        raise ValueError(
            f"Linkage has {len(labels)} leaves but W_all has {len(W_all)} rows"
        )

    # Pick 1 representative per cluster: closest to centroid (correlation distance)
    selected_indices = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        member_idx = np.where(mask)[0]
        W_cluster = W_all[member_idx]

        centroid = W_cluster.mean(axis=0, keepdims=True)
        dists = cdist(centroid, W_cluster, metric="correlation").ravel()
        best = member_idx[np.argmin(dists)]
        selected_indices.append(best)

    log.info(f"Selected {len(selected_indices)} representative dimensions")

    result = dim_index.iloc[selected_indices].copy().reset_index(drop=True)
    result["cluster"] = [labels[i] for i in selected_indices]

    # Attach content labels from LLM ratings
    ratings_path = Path(
        "experiments/universality/notebooks/exports/llm_dimension_ratings.csv"
    )
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        ratings = ratings.dropna(subset=["visual", "semantic"]).copy()
        ratings["visual"] = ratings["visual"].astype(bool)
        ratings["semantic"] = ratings["semantic"].astype(bool)

        def content_category(row):
            if row["visual"] and row["semantic"]:
                return "both"
            elif row["semantic"]:
                return "semantic"
            elif row["visual"]:
                return "visual"
            else:
                return "neither"

        ratings["content"] = ratings.apply(content_category, axis=1)
        result = result.merge(
            ratings[["dim_id", "content"]],
            on="dim_id",
            how="left",
        )
        result["content"] = result["content"].fillna("")
        log.info(
            f"Content breakdown: "
            + ", ".join(
                f"{c}={int((result['content'] == c).sum())}"
                for c in ["semantic", "visual", "both", "neither"]
            )
            + f", unlabelled={int((result['content'] == '').sum())}"
        )
    else:
        log.warning(f"LLM ratings not found at {ratings_path}, content will be empty")
        result["content"] = ""

    return result


def build_trial(
    dim_id: str,
    task_type: str = "real",
    content: str = "",
    attn_instruction: str = "",
) -> dict:
    return {
        "dim_id": dim_id,
        "task_type": task_type,
        "content": content,
        "attn_instruction": attn_instruction,
        "grid_url": f"grids/{dim_id}.png",
    }


def build_catch_trials() -> list[dict]:
    """Catch trials: all-dogs grid (obvious semantic)."""
    return [build_trial(
        "CATCH_dog",
        task_type="catch", content="semantic",
    )]


def build_attention_check(real_trials: list[dict]) -> dict:
    """Attention check: use grid from a 'neither' trial if available."""
    neither = [t for t in real_trials if t["content"] == "neither"]
    donor = neither[0] if neither else real_trials[0]
    return {
        "dim_id": "ATTN_CHECK",
        "task_type": "attention_check",
        "content": "",
        "attn_instruction": "For this trial, please select 'Both' and select 4 on the scale.",
        "grid_url": donor["grid_url"],
    }


def trials_to_json_row(trials: list[dict], version: int, base_url: str) -> dict:
    """Pack all trials into a single CSV row with a JSON column."""
    raw_json = json.dumps(trials, separators=(",", ":"))
    html_safe_json = raw_json.replace('"', '&quot;')
    return {
        "version": version,
        "n_trials": len(trials),
        "image_base_url": base_url,
        "trials_json": html_safe_json,
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    n_raters = 8
    trials_per_version = 15

    log.info("Loading components...")
    components = load_components_for_rank(
        rank=cfg.factorization.rank,
        models=list(cfg.models) or None,
        dataset=cfg.dataset,
    )

    w_vectors = []
    dim_records = []
    for model, W in sorted(components.items()):
        for d in range(W.shape[1]):
            w_vectors.append(W[:, d])
            dim_records.append({"model": model, "dimension": d})
    W_all = np.array(w_vectors, dtype=np.float32)
    dim_index = pd.DataFrame(dim_records)
    dim_index["dim_id"] = dim_index["model"] + "__d" + dim_index["dimension"].astype(str)

    log.info("\nSelecting cluster representatives...")
    dim_selection = select_dimensions(W_all=W_all, dim_index=dim_index)

    log.info("\nBuilding trials...")
    real_trials = []
    for _, row in dim_selection.iterrows():
        did = row["dim_id"]
        real_trials.append(build_trial(
            did,
            task_type="real", content=row["content"],
        ))
    log.info(f"Built {len(real_trials)} real trials")

    catch_trials = build_catch_trials()
    attn_trial = build_attention_check(real_trials)

    rng = np.random.default_rng(cfg.seed)
    trial_order = rng.permutation(len(real_trials)).tolist()

    versions = []
    for start in range(0, len(trial_order), trials_per_version):
        chunk_indices = trial_order[start : start + trials_per_version]
        version_trials = [real_trials[i] for i in chunk_indices]

        repeat_src = version_trials[0].copy()
        repeat_trial = {**repeat_src, "dim_id": f"R_{repeat_src['dim_id']}", "task_type": "repeat"}

        all_trials = version_trials + catch_trials + [attn_trial, repeat_trial]
        trial_positions = rng.permutation(len(all_trials)).tolist()
        all_trials = [all_trials[i] for i in trial_positions]

        versions.append(all_trials)

    real_counts = [sum(t["task_type"] == "real" for t in version) for version in versions]
    log.info(
        f"\n{len(versions)} versions with {min(real_counts)}–"
        f"{max(real_counts)} unique real trials, plus {len(catch_trials)} catch, "
        "1 attention check, and 1 repeated-real trial per version"
    )

    rows = []
    for v_idx, version_trials in enumerate(versions):
        # The image host URL was removed for anonymity; fill it in before deploying.
        row = trials_to_json_row(version_trials, version=v_idx, base_url=None)
        for _ in range(n_raters):
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    df.to_csv(out / "tasks.csv", index=False)

    log.info(
        f"\nWrote {len(df)} rows ({len(versions)} versions x {n_raters} raters) "
        f"to {out / 'tasks.csv'}"
    )
    log.info(
        f"Design: {len(real_trials)} dimensions, {n_raters} raters each, "
        f"{min(real_counts) + len(catch_trials) + 2}–"
        f"{max(real_counts) + len(catch_trials) + 2} trials per participant"
    )


if __name__ == "__main__":
    main()
