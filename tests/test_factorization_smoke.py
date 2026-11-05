import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RANK = 3
SEEDS = [0, 1, 2, 3, 4]


def run_entrypoint(module: str, data_root, *overrides: str) -> None:
    """Invoke an entry point the way a reader would, from the repository root."""
    result = subprocess.run(
        [sys.executable, "-m", module, f"data_root={data_root}", *overrides],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"{module} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8))
    feature_dir = root / "features" / "things22k"
    feature_dir.mkdir(parents=True)
    np.save(feature_dir / "toy.npy", features)

    meta_dir = root / "meta" / "things22k"
    meta_dir.mkdir(parents=True)
    pd.DataFrame({"feature_uid": ["toy"]}).to_csv(meta_dir / "model_metadata.csv", index=False)
    return root


def test_run_grid_writes_all_seed_fits(data_root):
    run_entrypoint(
        "experiments.factorization.fit.run_grid",
        data_root,
        'models=[toy]', f"factorization.rank={RANK}",
        'factorization.sigma_grid=["0.4"]',
        "factorization.max_outer=2", "factorization.max_inner=5",
    )

    base = data_root / "embeddings" / "things22k" / "toy" / f"rank_{RANK}" / "sigma_0.4"
    for seed in SEEDS:
        with np.load(base / f"seed_{seed}" / "w.npz") as archive:
            w = archive["w"]
        assert w.shape == (40, RANK)
        assert np.isfinite(w).all() and (w >= 0).all()
        metrics = json.loads((base / f"seed_{seed}" / "metrics.json").read_text())
        assert np.isfinite(metrics["final_evar"])


def test_compute_selection_selects_from_fits(data_root):
    run_entrypoint(
        "experiments.factorization.bandwidth_selection.compute_selection",
        data_root,
        f"factorization.rank={RANK}", 'factorization.sigma_grid=["0.4"]', "n_jobs=1",
    )

    selection = pd.read_csv(
        data_root / "meta" / "things22k" / f"bandwidth_selection_rank_{RANK}.csv"
    )
    assert len(selection) == 1
    row = selection.iloc[0]
    assert row["model"] == "toy"
    assert row["n_seeds"] == len(SEEDS)
    assert row["best_seed"] in SEEDS
    assert not row["manuscript_grid_complete"]
    assert np.isfinite(row["stability"]) and np.isfinite(row["evar"])
