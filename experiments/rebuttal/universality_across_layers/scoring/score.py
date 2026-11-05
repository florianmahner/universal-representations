from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.metric.compute.run import compute_universality_for_model
from experiments.universality.shared import load_components_for_rank

logger = logging.getLogger(__name__)

# Permutation null size for cos2_perm_adj, as in the paper.
N_PERMS = 1000
# Relative depths that were extracted and factorized (10 %, 50 %, 90 % of the
# network); each one is also scored as a within-depth cohort.
DEPTHS = ["d010", "d050", "d090"]


def manifest(embeddings: Path) -> pd.DataFrame:
    """One row per factorized representation: its role, base model and depth."""
    rows = []
    for d in sorted(p.name for p in embeddings.iterdir() if p.is_dir()):
        untr = d.startswith("untrained__")
        base, depth = (d[len("untrained__"):-len("__pen")], "pen") if untr else d.rsplit("__", 1)
        rows.append(dict(repr=d, role="untrained" if untr else "trained", base=base, depth=depth))
    return pd.DataFrame(rows)


def jobs(man: pd.DataFrame) -> list[tuple[str, str, tuple[str, ...]]]:
    out = []
    for depth in DEPTHS:
        cohort = tuple(man[(man.role == "trained") & (man.depth == depth)]["repr"])
        out += [(f"cohort_{depth}", t, cohort) for t in cohort]
    out += [("vs_ref162", t, ()) for t in man["repr"]]  # () -> reference cohort
    untr = tuple(man[man.role == "untrained"]["repr"])
    out += [("cohort_untrained", t, untr) for t in untr]
    return sorted(out)


def null_seed_for(target: str, meta_csv: Path, seed: int) -> int:
    """The paper's convention: null_seed = seed + index of the model in
    model_metadata.csv (compute/run.py). For a depth or untrained representation
    we use its base model's index, so a representation keeps one stable seed;
    representations with no base model in the metadata fall back to seed."""
    meta = pd.read_csv(meta_csv)
    order = {uid: i for i, uid in enumerate(meta.feature_uid)}
    base = (target[len("untrained__"):-len("__pen")] if target.startswith("untrained__")
            else target.rsplit("__", 1)[0])
    return seed + order.get(base, 0)


def score_one(scoring: str, target: str, cohort: tuple[str, ...], sel: pd.DataFrame,
              embeddings: Path, parts: Path, meta_csv: Path, rank: int, dataset: str,
              seed: int) -> None:
    part = parts / f"{scoring}__{target}.csv"
    if part.exists():
        return

    def load_w(name):
        r = sel[sel.model == name].iloc[0]
        return np.load(
            embeddings / name / f"rank_{rank}/sigma_{r.sigma_mult}/seed_{int(r.best_seed)}/w.npz"
        )["w"]

    if cohort:
        comp = {r: load_w(r) for r in cohort}
    else:
        # vs_ref162: the target is scored against the reference set EXCLUDING its
        # own base model, matching the paper (a model is never a source for itself)
        ref = dict(load_components_for_rank(rank=rank, dataset=dataset))
        base = (target[len("untrained__"):-len("__pen")]
                if target.startswith("untrained__") else target.rsplit("__", 1)[0])
        ref.pop(base, None)
        comp = {**ref, target: load_w(target)}
    rows = compute_universality_for_model(target, comp, n_null_perms=N_PERMS,
                                          null_seed=null_seed_for(target, meta_csv, seed))
    df = pd.DataFrame([r for r in rows if r["metric"] == "cos2_perm_adj"]).assign(scoring=scoring)
    parts.mkdir(parents=True, exist_ok=True)
    df.to_csv(part, index=False)
    logger.info(f"{scoring} {target}: mean adj {df.universality.mean():.4f}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    parts = out / "universality_parts"
    embeddings = Path("experiments/rebuttal/universality_across_layers/factorization/outputs/embeddings")
    meta_csv = Path(cfg.data_root) / "meta" / cfg.dataset / "model_metadata.csv"
    rank = cfg.factorization.rank
    sel = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/factorization/outputs")
        / f"bandwidth_selection_full_rank_{rank}.csv"
    )

    man = manifest(embeddings)
    todo = jobs(man)
    logger.info(f"{len(todo)} tasks")
    Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(score_one)(s, t, c, sel, embeddings, parts, meta_csv, rank, cfg.dataset, cfg.seed)
        for s, t, c in todo
    )

    written = list(parts.glob("*.csv"))
    assert len(written) >= len(todo), f"only {len(written)}/{len(todo)} parts done"
    df = pd.concat(map(pd.read_csv, written)).rename(columns={"model": "repr"})
    out_csv = out / f"universality_full_rank_{rank}.csv"
    df.merge(man, on="repr", how="left").assign(rank=rank).to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
