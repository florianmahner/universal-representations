from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.rebuttal.categorical_control.run import (
    MATCH_TOLERANCE,
    build_matched_assignment,
    partial_correlation,
    triplet_accuracy,
)


def test_partial_correlation_removes_controlled_signal() -> None:
    rng = np.random.default_rng(1)
    control = rng.normal(size=500)
    x = control + rng.normal(scale=0.2, size=500)
    y = control + rng.normal(scale=0.2, size=500)
    raw = np.corrcoef(x, y)[0, 1]
    partial, _ = partial_correlation(x, y, control)
    assert raw > 0.8
    assert abs(partial) < 0.1


def test_triplet_accuracy() -> None:
    embedding = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    triplets = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int16)
    assert triplet_accuracy(embedding, triplets, chunk_size=1) == 1.0


def test_matching_constraints() -> None:
    rng = np.random.default_rng(2)
    rows = []
    for model_index in range(2):
        # Repeated pairs guarantee an exactly balanced feasible solution while
        # still spanning all five within-model eta-squared quintiles.
        eta = np.repeat(np.linspace(0.1, 0.9, 25), 2)
        universality = eta + rng.normal(scale=0.1, size=50)
        for dimension in range(50):
            rows.append(
                {
                    "model": f"model_{model_index}",
                    "dimension": dimension,
                    "universality": universality[dimension],
                    "eta_squared": eta[dimension],
                    "c_eff_excess": 10.0,
                    "importance": 1 / 50,
                }
            )
    assignment = build_matched_assignment(pd.DataFrame(rows))
    summary = assignment.groupby(["model", "subset"]).agg(
        n=("dimension", "size"),
        eta=("eta_squared", "mean"),
        universality=("universality", "mean"),
    )
    for model_index in range(2):
        high = summary.loc[(f"model_{model_index}", "higher_universality")]
        low = summary.loc[(f"model_{model_index}", "lower_universality")]
        assert high["n"] == low["n"] == 25
        assert abs(high["eta"] - low["eta"]) <= MATCH_TOLERANCE + 1e-9
        assert high["universality"] > low["universality"]
    bin_counts = assignment.groupby(
        ["model", "subset", "eta_quintile"]
    ).size()
    assert (bin_counts == 5).all()
