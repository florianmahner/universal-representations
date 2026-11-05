import json

import numpy as np
import pytest

from experiments.factorization.bandwidth_selection.compute_selection import (
    compute_model_sigma,
)


def _write_seed(root, seed: int, w: np.ndarray) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    np.save(seed_dir / "w.npy", w)
    (seed_dir / "metrics.json").write_text(json.dumps({"final_evar": 0.75}))


def test_bandwidth_selection_requires_all_five_seeds(tmp_path):
    sigma_dir = tmp_path / "model" / "rank_2" / "sigma_0.5"
    for seed in range(4):
        _write_seed(sigma_dir, seed, np.eye(4, 2))

    with pytest.raises(FileNotFoundError, match="Incomplete five-seed grid"):
        compute_model_sigma("model", "0.5", 2, tmp_path)


def test_bandwidth_selection_reports_an_actual_seed_id(tmp_path):
    sigma_dir = tmp_path / "model" / "rank_2" / "sigma_0.5"
    base = np.array([[1.0, 0.0], [0.8, 0.1], [0.1, 0.9], [0.0, 1.0]])
    for seed in range(5):
        w = base[:, ::-1] if seed % 2 else base
        _write_seed(sigma_dir, seed, w)

    result = compute_model_sigma("model", "0.5", 2, tmp_path)

    assert result["n_seeds"] == 5
    assert result["best_seed"] in range(5)
