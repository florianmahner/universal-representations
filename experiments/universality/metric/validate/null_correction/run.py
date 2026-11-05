#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.shared import load_components_for_rank
from experiments.universality.shared.core import (
    cos2_perm_null_thresholds,
    cos2_permutation,
    null_adjust,
)

N_PERMUTATIONS = 1000

logger = logging.getLogger(__name__)


def process_one_model(
    target_model: str,
    components: dict[str, np.ndarray],
    n_perm: int,
    seed: int,
) -> dict:
    """Compute raw, pair-threshold, and adjusted scores for one target."""
    W_target = components[target_model]
    sources = [
        W for name, W in components.items() if name != target_model
    ]
    raw_by_pair = np.stack(
        [cos2_permutation(source, W_target) for source in sources]
    )
    thresholds = cos2_perm_null_thresholds(
        W_target,
        sources,
        n_perms=n_perm,
        seed=seed,
    )
    adjusted_by_pair = null_adjust(raw_by_pair, thresholds)

    return {
        "model": target_model,
        "raw": raw_by_pair.mean(axis=0),
        "null_threshold": thresholds.mean(axis=0),
        "corrected": adjusted_by_pair.mean(axis=0),
    }


def load_evar(
    models: list[str],
    rank: int,
    data_root: Path,
    dataset: str,
) -> dict[str, float]:
    """Read evar from metrics.json for each model's selected sigma and seed."""
    evar = {}
    components_dir = data_root / "embeddings" / dataset
    selection_path = data_root / "meta" / dataset / f"bandwidth_selection_rank_{rank}.csv"
    if not selection_path.exists():
        return {model: np.nan for model in models}
    selection = pd.read_csv(selection_path).set_index("model")
    for model in models:
        if model not in selection.index:
            evar[model] = np.nan
            continue
        row = selection.loc[model]
        metrics_path = (
            components_dir / model / f"rank_{rank}"
            / f"sigma_{row['sigma_mult']}" / f"seed_{int(row['best_seed'])}"
            / "metrics.json"
        )
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            if "final_evar" in data:
                evar[model] = data["final_evar"]
            elif "evar" in data:
                ev = data["evar"]
                evar[model] = ev[-1] if isinstance(ev, list) else ev
            else:
                evar[model] = np.nan
        else:
            evar[model] = np.nan
    return evar


def main() -> None:
    """Load components, run the pair-specific null correction, save outputs."""
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    components = load_components_for_rank(
        rank=rank,
        models=list(cfg.models) or None,
        dataset=cfg.dataset,
    )
    # The macaque IT recordings are not DNN models; they are scored separately.
    for model in ("macaque_F_it", "macaque_N_it"):
        components.pop(model, None)
    model_names = list(components.keys())
    n_models = len(model_names)

    logger.info(
        f"Running null correction: {n_models} models, rank={rank}, "
        f"n_perm={N_PERMUTATIONS}"
    )

    seeds = [cfg.seed + i for i in range(n_models)]
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(process_one_model)(model, components, N_PERMUTATIONS, s)
        for model, s in zip(model_names, seeds)
    )

    evar_map = load_evar(model_names, rank, Path(cfg.data_root), cfg.dataset)

    per_dim_rows = []
    per_model_rows = []

    for res in results:
        model = res["model"]
        n_dims = len(res["raw"])

        for d in range(n_dims):
            per_dim_rows.append({
                "model": model,
                "dimension": d,
                "raw": float(res["raw"][d]),
                "null_threshold": float(res["null_threshold"][d]),
                "corrected": float(res["corrected"][d]),
            })

        per_model_rows.append({
            "model": model,
            "raw_mean": float(res["raw"].mean()),
            "null_mean": float(res["null_threshold"].mean()),
            "corrected_mean": float(res["corrected"].mean()),
            "evar": evar_map.get(model, np.nan),
        })

    per_dim_df = pd.DataFrame(per_dim_rows)
    per_model_df = pd.DataFrame(per_model_rows)
    per_dim_df.to_csv(out / "per_dimension.csv", index=False)
    per_model_df.to_csv(out / "per_model.csv", index=False)

    params = {
        "rank": rank,
        "n_perm": N_PERMUTATIONS,
        "n_models": n_models,
        "n_jobs": cfg.n_jobs,
        "seed": cfg.seed,
    }
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Saved per_dimension.csv ({len(per_dim_df)} rows)")
    logger.info(f"Saved per_model.csv ({len(per_model_df)} rows)")
    logger.info("Saved params.json")

    logger.info(f"\nRaw mean universality: {per_model_df['raw_mean'].mean():.4f}")
    logger.info(f"Null mean: {per_model_df['null_mean'].mean():.4f}")
    logger.info(f"Corrected mean: {per_model_df['corrected_mean'].mean():.4f}")


if __name__ == "__main__":
    main()
