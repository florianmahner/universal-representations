import numpy as np

from experiments.universality.metric.validate.cka_comparison.run import (
    model_level_summary,
)


def test_cka_summary_averages_off_diagonal_pairs():
    """Both quantities are per-model means over the other models, as published."""
    cka = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.8],
            [0.4, 0.8, 1.0],
        ]
    )
    universality = np.array(
        [
            [1.0, 0.10, 0.30],
            [0.10, 1.0, 0.50],
            [0.30, 0.50, 1.0],
        ]
    )

    summary = model_level_summary(cka, universality, ["a", "b", "c"])

    np.testing.assert_allclose(summary["mean_cka"], [0.3, 0.5, 0.6])
    np.testing.assert_allclose(summary["mean_universality"], [0.2, 0.3, 0.4])
