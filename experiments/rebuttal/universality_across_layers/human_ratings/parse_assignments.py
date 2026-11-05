#!/usr/bin/env python3
"""Turn the layer-wise rating exports into per-trial tables."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from experiments.human_ratings.parse_assignments import parse_assignment_file

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# the three sampled network depths of the layer study
TAGS = {"d010": "early", "d050": "middle", "d090": "late"}


def dim_layer(dim_id: str) -> str:
    for tag, layer in TAGS.items():
        if f"__{tag}__" in dim_id:
            return layer
    return ""


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # The participant-level exports are not bundled.
    files = sorted((Path(cfg.data_root) / "ratings" / "layers").glob("assignments_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No assignments_*.csv in {Path(cfg.data_root) / 'ratings' / 'layers'}"
        )
    log.info(f"{len(files)} assignment file(s)")
    rows = []
    for f in files:
        file_rows = parse_assignment_file(f, llm_map={})
        for r in file_rows:
            r["source_file"] = f.name
        rows.extend(file_rows)
    df = pd.DataFrame(rows)
    df["time_s"] = df["time_ms"] / 1000.0

    df["layer"] = df["dim_id"].map(dim_layer)
    # catch/attn rows carry no depth tag: take the participant's layer WITHIN
    # the same export file (each file is one study, and a participant may have
    # taken part in more than one study)
    participant_layer = (df[df.layer != ""]
                         .groupby(["participant", "source_file"])["layer"]
                         .agg(lambda g: g.mode().iat[0]))
    mask = df.layer == ""
    df.loc[mask, "layer"] = pd.MultiIndex.from_frame(
        df.loc[mask, ["participant", "source_file"]]).map(participant_layer)
    df = df.rename(columns={"source_file": "study"})
    df["study"] = df["study"].str.removeprefix("assignments_").str.removesuffix(".csv")
    # ratings of the two hybrid models' layer dims from PRE-FIX studies refer
    # to the defective representations and are dropped structurally here.
    # Post-fix batches (study name contains 'hybridfix') rate the CORRECTED
    # dims under the same dim_ids and are kept.
    hybrid = ("midas_dpt_hybrid__d0", "timm_vit_base_r50_s16_224_orig_in21k__d0")
    stale = (df.dim_id.str.removeprefix("R_").str.startswith(hybrid)
             & ~df.study.str.contains("hybridfix"))
    if stale.any():
        log.info(f"dropped {int(stale.sum())} trials of defective hybrid dims "
                 f"(pre-fix studies)")
    df = df[~stale]

    # within-study exact duplicates (re-exported files)
    df = df.drop_duplicates(subset=["participant", "dim_id", "study"])
    # a participant who took part in two studies of the same layer (top-up
    # exclusion missed them): keep each study's OWN check trials, but drop
    # re-ratings of grids they already saw (real/repeat trials, keep first)
    is_check = df.task_type.isin(["catch", "attention_check"])
    df = pd.concat([df[is_check],
                    df[~is_check].drop_duplicates(subset=["participant", "dim_id", "layer"])],
                   ignore_index=True)
    df["repr"] = df["model"]                       # model column = model__d0X0
    df["base_model"] = df["repr"].str.rsplit("__", n=1).str[0]

    df.to_csv(out / "parsed_all_trials.csv", index=False)
    real = df[df.task_type == "real"]
    real.to_csv(out / "parsed_real_trials.csv", index=False)
    log.info(f"{len(df)} trial rows, {df.participant.nunique()} participants, "
             f"{real.dim_id.nunique()} rated dimensions")
    per_layer = real.groupby("layer").dim_id.nunique().to_dict()
    log.info(f"per layer: {per_layer}")


if __name__ == "__main__":
    main()
