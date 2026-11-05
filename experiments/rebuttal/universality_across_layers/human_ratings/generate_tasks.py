"""Build the rating-task CSVs for the layer-wise dimension rating study.

Trial construction is imported verbatim from the penultimate-layer study
(``experiments.human_ratings.generate_tasks``); only the dimensions differ, and
they come from this experiment's per-layer clustering.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.human_ratings.generate_tasks import (
    build_attention_check,
    build_catch_trials,
    build_trial,
    trials_to_json_row,
)

log = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    n_raters = 8
    trials_per_version = 15

    for layer in ("early", "middle", "late"):
        reps = pd.read_csv(
            Path("experiments/rebuttal/universality_across_layers/human_ratings/outputs")
            / f"representatives_{layer}.csv"
        )
        log.info(f"{layer}: {len(reps)} dimensions to rate")

        real_trials = [
            build_trial(row.dim_id, task_type="real", content="")
            for _, row in reps.iterrows()
        ]
        # layer-specific grid folder
        for t in real_trials:
            t["grid_url"] = f"grids/{layer}/{t['dim_id']}.png"

        catch_trials = build_catch_trials()
        attn_trial = build_attention_check(real_trials)

        rng = np.random.default_rng(cfg.seed)
        trial_order = rng.permutation(len(real_trials)).tolist()

        versions = []
        for start in range(0, len(trial_order), trials_per_version):
            chunk = [real_trials[i] for i in trial_order[start:start + trials_per_version]]
            repeat_src = chunk[0].copy()
            repeat_trial = {**repeat_src, "dim_id": f"R_{repeat_src['dim_id']}",
                            "task_type": "repeat"}
            all_trials = chunk + catch_trials + [attn_trial, repeat_trial]
            order = rng.permutation(len(all_trials)).tolist()
            versions.append([all_trials[i] for i in order])

        # one row per rater, as in the original tasks.csv (71 versions x 8 rows)
        rows = []
        for v, trials in enumerate(versions):
            # The image host URL was removed for anonymity; fill it in before deploying.
            row = trials_to_json_row(trials, v + 1, None)
            rows.extend([row] * n_raters)
        tasks_path = out / f"tasks_{layer}.csv"
        pd.DataFrame(rows).to_csv(tasks_path, index=False)
        log.info(f"{len(versions)} versions ({trials_per_version} real + "
                 f"{len(catch_trials)} catch + 1 attn + 1 repeat each) -> {tasks_path}")
        log.info(f"raters needed for {n_raters} ratings/dim: "
                 f"{len(versions) * n_raters} participants")


if __name__ == "__main__":
    main()
