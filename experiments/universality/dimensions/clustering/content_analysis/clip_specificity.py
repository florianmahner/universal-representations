from __future__ import annotations

import logging

import pandas as pd

from experiments.universality.shared import load_model_info

log = logging.getLogger(__name__)

SLIP_OBJECTIVES = {
    "clip": [m for m in [
        "slip_vit_s_clip", "slip_vit_b_clip", "slip_vit_l_clip",
        "slip_vit_b_clip_cc12m",
    ]],
    "slip": [m for m in [
        "slip_vit_s_slip", "slip_vit_b_slip", "slip_vit_l_slip",
        "slip_vit_b_slip_cc12m",
    ]],
    "simclr": [m for m in [
        "slip_vit_s_simclr", "slip_vit_b_simclr", "slip_vit_l_simclr",
    ]],
}


def _objective_label(model: str, info: pd.DataFrame) -> str:
    """Classify a model into broad objective groups."""
    row = info.loc[info["feature_uid"] == model]
    if row.empty:
        return "other"
    task = row.iloc[0]["train_task"]
    if task == "clip":
        return "clip"
    if task == "slip":
        # distinguish sub-objectives within SLIP family
        for obj, models in SLIP_OBJECTIVES.items():
            if model in models:
                return f"slip_{obj}"
        return "slip_other"
    return task


def analyze_clip_specificity(
    dim_df: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Compute over/under-representation of vision-language models per cluster.

    Returns a DataFrame with enrichment scores: observed fraction minus
    expected fraction of vision-language models in each cluster.
    """
    info = load_model_info()

    # label each dimension's model
    dim_df = dim_df.copy()
    task_map = info.set_index("feature_uid")["train_task"].to_dict()
    dim_df["task"] = dim_df["model"].map(task_map).fillna("unknown")
    dim_df["is_vl"] = dim_df["task"].isin(["clip", "slip"])

    n_total = dim_df["model"].nunique()
    n_vl = dim_df.loc[dim_df["is_vl"], "model"].nunique()
    expected_vl_frac = n_vl / n_total
    log.info("Vision-language models: %d/%d (%.1f%%)", n_vl, n_total, 100 * expected_vl_frac)

    records = []
    for _, row in profiles.iterrows():
        cid = str(row["cluster"])
        cluster_dims = dim_df[dim_df["cluster"].astype(str) == cid]
        cluster_models = cluster_dims["model"].unique()
        n_cluster = len(cluster_models)
        if n_cluster == 0:
            continue

        n_cluster_vl = sum(1 for m in cluster_models if task_map.get(m) in ("clip", "slip"))
        obs_frac = n_cluster_vl / n_cluster
        enrichment = obs_frac - expected_vl_frac

        # SLIP family breakdown: among SLIP models in this cluster,
        # which sub-objectives are present?
        slip_breakdown = {}
        for obj, obj_models in SLIP_OBJECTIVES.items():
            present = [m for m in obj_models if m in cluster_models]
            slip_breakdown[f"slip_{obj}_count"] = len(present)

        records.append({
            "cluster": cid,
            "model_coverage": row["model_coverage"],
            "n_vl_models": n_cluster_vl,
            "vl_fraction": obs_frac,
            "vl_expected": expected_vl_frac,
            "vl_enrichment": enrichment,
            **slip_breakdown,
        })

    result = pd.DataFrame(records)

    n_enriched = (result["vl_enrichment"] > 0.1).sum()
    n_depleted = (result["vl_enrichment"] < -0.05).sum()
    log.info("VL-enriched clusters (>10pp): %d, VL-depleted (<-5pp): %d", n_enriched, n_depleted)
    return result


def analyze_slip_family(dim_df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Compare cluster membership across the three SLIP sub-objectives.

    For each cluster, reports how many of the CLIP-objective, SLIP-objective,
    and SimCLR-objective SLIP models contribute dimensions.
    """
    records = []
    for _, row in profiles.iterrows():
        cid = str(row["cluster"])
        cluster_models = set(
            dim_df.loc[dim_df["cluster"].astype(str) == cid, "model"].unique()
        )
        entry = {"cluster": cid, "model_coverage": row["model_coverage"]}
        for obj, obj_models in SLIP_OBJECTIVES.items():
            present = [m for m in obj_models if m in cluster_models]
            entry[f"{obj}_count"] = len(present)
            entry[f"{obj}_frac"] = len(present) / len(obj_models) if obj_models else 0
        records.append(entry)

    return pd.DataFrame(records)
