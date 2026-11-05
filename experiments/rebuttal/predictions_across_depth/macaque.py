from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.downstream_predictions.macaque_encoding.encoding_things.run import (
    compute_noise_ceiling,
    encode_single_model,
    load_macaque_it,
)

from . import data as D

UNTRAINED = False  # set True to encode the untrained networks instead of the depth sweep

N_FOLDS = 5

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _one(base: str, depth: str, y: np.ndarray, noise_ceil: np.ndarray,
         n_folds: int, random_state: int) -> dict:
    w = D.load_w(base, depth)
    r = encode_single_model(w, y, n_folds, random_state)
    r_corr = np.clip(r / noise_ceil, -1, 1)
    return {
        "base": base,
        "depth": depth,
        "mean_r": float(np.mean(r)),
        "median_r": float(np.median(r)),
        "mean_r_corrected": float(np.mean(r_corr)),
        "median_r_corrected": float(np.median(r_corr)),
        "n_neurons": len(r),
    }


def combine_monkeys(
    per_monkey: dict[str, pd.DataFrame],
    monkeys: tuple[str, ...],
    path: Path,
) -> pd.DataFrame:
    """Average the per monkey encoding scores into one IT score per model.

    Matches the paper: encoding = (mean_r_F + mean_r_N) / 2, with the per monkey
    values kept in the same row as mean_r_F and mean_r_N so no separate per monkey
    table is needed.
    """
    # properties of the model or the representation, identical in the F and N tables
    shared = ["universality", "universality_cohort", "display_name", "model_class",
              "architecture", "train_task", "train_task_display", "train_data",
              "train_data_display", "objective_category", "top1_accuracy"]
    # per monkey encoding scores that get averaged into the single IT score
    scores = ["mean_r", "median_r", "mean_r_corrected", "median_r_corrected"]

    parts = {m: df.set_index(["base", "depth"]) for m, df in per_monkey.items()}
    missing = [m for m in monkeys if m not in parts]
    if missing:
        log.warning(f"No results for monkeys {missing}, cannot average")
        return pd.DataFrame()

    keys = sorted(set.intersection(*(set(df.index) for df in parts.values())))
    dropped = {m: len(df) - len(keys) for m, df in parts.items()}
    if any(dropped.values()):
        log.warning(f"Not every model was scored in both monkeys, dropped {dropped}")

    ref = parts[monkeys[0]].loc[keys]
    out = ref[[c for c in shared if c in ref.columns]].copy()

    for col in scores:
        values_per_monkey = [parts[m].loc[keys, col] for m in monkeys]
        out[col] = sum(values_per_monkey) / len(values_per_monkey)
        for monkey, values in zip(monkeys, values_per_monkey):
            out[f"{col}_{monkey}"] = values

    for monkey in monkeys:
        out[f"n_neurons_{monkey}"] = parts[monkey].loc[keys, "n_neurons"]
    out["monkey"] = "mean"

    out = out.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info(f"Saved {path} ({len(out)} rows, mean of monkeys {' and '.join(monkeys)})")
    return out


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    monkeys = ("F", "N")
    per_monkey = {}

    for monkey in monkeys:
        y, reliab = load_macaque_it(monkey)
        noise_ceil = compute_noise_ceiling(reliab)
        log.info(f"Monkey {monkey}: {y.shape[1]} reliable IT neurons, {y.shape[0]} images, "
                 f"noise ceiling median {np.median(noise_ceil):.3f}")

        if UNTRAINED:
            jobs = [(b, "untrained") for b in D.untrained_models()]
        else:
            jobs = [(b, d) for d in D.DEPTHS for b in D.base_models()]
        log.info(f"Encoding {len(jobs)} representations")
        rows = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
            delayed(_one)(b, d, y, noise_ceil, N_FOLDS, cfg.seed)
            for b, d in jobs
        )
        df = D.merge_all(pd.DataFrame(rows))
        per_monkey[monkey] = df

        for depth, g in df.groupby("depth"):
            log.info(f"  {D.DEPTH_NAME[depth]:12s} mean r {g.mean_r.mean():.4f} "
                     f"[{g.mean_r.min():.3f}, {g.mean_r.max():.3f}]")

    combined = combine_monkeys(
        per_monkey, monkeys,
        out / "scores" / "macaque" / ("untrained.csv" if UNTRAINED else "across_depth.csv"),
    )
    if not combined.empty:
        log.info("Averaged across monkeys F and N:")
        for depth, g in combined.groupby("depth"):
            log.info(f"  {D.DEPTH_NAME[depth]:12s} mean r {g.mean_r.mean():.4f} "
                     f"[{g.mean_r.min():.3f}, {g.mean_r.max():.3f}]")


if __name__ == "__main__":
    main()
