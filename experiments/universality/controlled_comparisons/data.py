from __future__ import annotations

from pathlib import Path

import pandas as pd

ENCODING_DIR = Path(
    "experiments/universality/downstream_predictions/macaque_encoding/encoding_things/outputs"
)

COMPARE_COLS = [
    "compare_training",
    "compare_architecture",
    "compare_goal_slip",
    "compare_goal_taskonomy_tasks",
    "compare_goal_taskonomy_cluster",
    "compare_goal_expertise",
    "compare_diet_vit",
    "compare_goal_selfsupervised",
    "compare_goal_contrastive",
    "compare_diet_imagenetsize",
]

CONTRAST_DESCRIPTIONS = {
    "compare_training": (
        "Same architectures compared with and without ImageNet classification training. "
        "Isolates the effect of learning from data."
    ),
    "compare_architecture": (
        "All models trained on ImageNet classification. "
        "Varies only architecture family (CNN, Transformer, Hybrid, MLP-Mixer)."
    ),
    "compare_goal_slip": (
        "ViT models (S, B, L) trained with SimCLR, CLIP, or SLIP. "
        "Isolates training objective while controlling architecture and scale."
    ),
    "compare_goal_taskonomy_tasks": (
        "All ResNet-50 trained on Taskonomy with different visual tasks. "
        "Isolates task objective with identical architecture and data."
    ),
    "compare_goal_taskonomy_cluster": (
        "Same as Taskonomy tasks but grouped into functional clusters "
        "(2D, 3D, Semantic, Geometric)."
    ),
    "compare_goal_expertise": (
        "BiT models fine-tuned on different object categories. "
        "Same architecture, varies only the domain of expertise."
    ),
    "compare_diet_vit": (
        "CLIP ViT-L/14 trained on different web-scale datasets "
        "(OpenAI-400M, LAION, DataComp, DFN, MetaCLIP). Same architecture and objective."
    ),
    "compare_goal_selfsupervised": (
        "ResNet-50 trained on ImageNet with different self-supervised methods. "
        "Isolates SSL objective."
    ),
    "compare_goal_contrastive": (
        "ResNet-50 SSL models split into contrastive vs. non-contrastive. "
        "Same architecture, data, and self-supervised paradigm."
    ),
    "compare_diet_imagenetsize": (
        "Same architectures trained on ImageNet-1k vs. ImageNet-21k. "
        "Isolates dataset scale."
    ),
}


def load_merged(
    rank: int = 50,
    include_encoding: bool = False,
    components: str = "default",
) -> pd.DataFrame:
    """Load universality scores merged with model metadata and contrast assignments.

    All model properties and contrast assignments come from model_metadata.csv.
    """
    from src.data import ThingsDataset

    from experiments.universality.shared import load_universality_by_model
    uni = load_universality_by_model(rank=rank)
    uni = uni.rename(columns={"mean": "universality", "std": "universality_std"})

    ds = ThingsDataset()
    mi = ds.model_info.copy()

    keep_cols = [
        "name", "display_name", "model_string", "model_class",
        "train_task", "train_task_display", "train_data", "train_data_display",
        "top1_accuracy", "total_parameter_count", "architecture",
        "architecture_family", "objective_category",
        "task_cluster", "source",
    ] + COMPARE_COLS
    available = [c for c in keep_cols if c in mi.columns]

    df = uni.merge(
        mi[available].drop_duplicates(subset=["name"]),
        left_on="model",
        right_on="name",
        how="left",
    )

    if include_encoding:
        enc = load_encoding()
        df = df.merge(enc, left_on="name", right_on="model", how="left", suffixes=("", "_enc"))
        if "model_enc" in df.columns:
            df = df.drop(columns=["model_enc"])

    return df


def load_encoding() -> pd.DataFrame:
    """Load macaque IT encoding results averaged across monkeys F and N."""
    df_f = pd.read_csv(ENCODING_DIR / "encoding_results_F.csv", usecols=["model", "mean_r"])
    df_n = pd.read_csv(ENCODING_DIR / "encoding_results_N.csv", usecols=["model", "mean_r"])
    df = df_f.merge(df_n, on="model", suffixes=("_F", "_N"))
    df["encoding"] = (df["mean_r_F"] + df["mean_r_N"]) / 2
    df = df.rename(columns={"mean_r_F": "encoding_F", "mean_r_N": "encoding_N"})
    return df[["model", "encoding", "encoding_F", "encoding_N"]]
