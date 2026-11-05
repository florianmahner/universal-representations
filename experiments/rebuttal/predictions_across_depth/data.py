from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ThingsDataset

log = logging.getLogger(__name__)

EXP = Path(__file__).resolve().parent
# Written by layers/factorization/{select_sigma,artifacts}.py into their own
# outputs directory.
LAYER_FACTORIZATION = EXP.parent / "layers" / "factorization" / "outputs"
EMB = LAYER_FACTORIZATION / "embeddings"
SEL_CSV = LAYER_FACTORIZATION / "bandwidth_selection_full_rank_50.csv"
UNI_MODEL_CSV = LAYER_FACTORIZATION / "universality_model_level_rank_50.csv"
ROOT = EXP.parents[2]

RANK = 50
DEPTHS = ["d010", "d050", "d090", "pen"]
DEPTH_NAME = {"d010": "Early", "d050": "Middle", "d090": "Late",
              "pen": "Penultimate", "untrained": "Untrained"}
DEPTH_X = {"d010": 0.1, "d050": 0.5, "d090": 0.9, "pen": 1.0}
COLOR = {"d010": "#9ecae1", "d050": "#4292c6", "d090": "#08519c", "pen": "#08306b",
         "untrained": "#d95f02"}


@cache
def _selection() -> pd.DataFrame:
    return pd.read_csv(SEL_CSV)


@cache
def _dataset() -> ThingsDataset:
    return ThingsDataset()


def base_models() -> list[str]:
    """The 162 models that have all three depths plus a published penultimate."""
    ds = _dataset()
    pen = {m for m in ds.list_component_models() if not m.startswith("macaque")}
    sel = _selection()
    have = {d: set(sel.loc[sel.model.str.endswith(f"__{d}"), "model"].str.replace(f"__{d}", "", regex=False))
            for d in ("d010", "d050", "d090")}
    common = pen & have["d010"] & have["d050"] & have["d090"]
    return sorted(common)


def untrained_models() -> list[str]:
    """Randomly initialised networks (one per architecture), penultimate layer."""
    sel = _selection()
    names = sel.loc[sel.model.str.startswith("untrained__"), "model"]
    return sorted({n[len("untrained__"):-len("__pen")] for n in names})


def load_w(base: str, depth: str) -> np.ndarray:
    """Rank-50 W for one (model, depth) at the E1-selected sigma and seed.

    depth "pen" is the paper's published penultimate factorization; depth
    "untrained" is the random-init penultimate from the E1 experiment.
    """
    if depth == "pen":
        return _dataset().load_components(base, rank=RANK)
    name = f"untrained__{base}__pen" if depth == "untrained" else f"{base}__{depth}"
    sel = _selection()
    row = sel[sel.model == name]
    if row.empty:
        raise FileNotFoundError(f"No sigma selection for {name}")
    r = row.iloc[0]
    path = EMB / name / f"rank_{RANK}/sigma_{r.sigma_mult}/seed_{int(r.best_seed)}/w.npz"
    return np.load(path)["w"]


def universality_table() -> pd.DataFrame:
    """Universality per (base, depth) on both axes.

    `universality_cohort` is the within-cohort score: each representation against
    the other 161 at the SAME depth (early vs early, late vs late), and for the
    untrained networks against the other untrained networks.
    `universality` is the paper's axis (`vs_ref162`): every representation scored
    against the 162 published penultimate references. At the penultimate layer the
    two coincide, which is why depth 1.0 has one number.
    """
    uni = pd.read_csv(UNI_MODEL_CSV).drop_duplicates(subset=["repr", "scoring"])

    ref = uni[uni.scoring == "vs_ref162"][["base", "depth", "role", "mean"]]
    ref = ref.rename(columns={"mean": "universality"})

    cohort = uni[uni.scoring.str.startswith("cohort_")][["base", "depth", "role", "mean"]]
    cohort = cohort.rename(columns={"mean": "universality_cohort"})

    out = ref.merge(cohort, on=["base", "depth", "role"], how="left")
    out["depth"] = np.where(out.role == "untrained", "untrained", out.depth)
    out = out.drop(columns=["role"])

    from experiments.universality.shared import load_universality_by_model

    pen = load_universality_by_model(RANK).rename(columns={"model": "base", "mean": "universality"})
    pen = pen[["base", "universality"]].copy()
    pen["depth"] = "pen"
    pen["universality_cohort"] = pen["universality"]

    return pd.concat([out[out.depth != "pen"], pen], ignore_index=True)


def model_metadata() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data/meta/things22k/model_metadata.csv")
    cols = ["feature_uid", "display_name", "model_class", "architecture",
            "train_task", "train_task_display", "train_data", "train_data_display",
            "objective_category", "top1_accuracy"]
    return meta[[c for c in cols if c in meta.columns]].rename(columns={"feature_uid": "base"})


def merge_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach universality and model metadata to a (base, depth) result frame."""
    df = df.merge(universality_table(), on=["base", "depth"], how="left")
    return df.merge(model_metadata(), on="base", how="left")
