import numpy as np
import pandas as pd

from experiments.universality.dimensions.reconstruction_importance.run import (
    loo_importance_for_model,
)
from experiments.universality.dimensions.category_consistency.compute import (
    compute_importance_universality,
)


def test_streamed_leave_one_out_importance_matches_direct_r2():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(9, 4))
    components = np.abs(rng.normal(size=(9, 3))) * 0.25
    sigma = 1.2

    result = loo_importance_for_model(
        "synthetic",
        features,
        components,
        sigma,
    )

    squared_distances = (
        (features[:, None, :] - features[None, :, :]) ** 2
    ).sum(axis=2)
    similarity = np.exp(-squared_distances / (2 * sigma**2))
    upper = np.triu_indices(len(features), k=1)
    observed = similarity[upper]
    total_ss = np.square(observed - observed.mean()).sum()

    def r2(prediction: np.ndarray) -> float:
        residual_ss = np.square(observed - prediction[upper]).sum()
        return 1.0 - residual_ss / total_ss

    reconstruction = components @ components.T
    expected_full = r2(reconstruction)
    expected_delta = np.array(
        [
            expected_full
            - r2(
                reconstruction
                - np.outer(components[:, index], components[:, index])
            )
            for index in range(components.shape[1])
        ]
    )

    np.testing.assert_allclose(result["r2_full"], expected_full)
    np.testing.assert_allclose(result["delta_r2"], expected_delta)


def test_characterization_uses_delta_r2_not_norm_fraction():
    n_dimensions = 10
    selectivity = pd.DataFrame(
        {
            "model": ["m"] * n_dimensions,
            "dimension": np.arange(n_dimensions),
            "universality": np.linspace(0.1, 1.0, n_dimensions),
            "norm_fraction": np.linspace(0.2, 0.02, n_dimensions),
        }
    )
    loo = pd.DataFrame(
        {
            "model": ["m"] * n_dimensions,
            "dimension": np.arange(n_dimensions),
            "delta_r2": np.linspace(0.01, 0.1, n_dimensions),
        }
    )
    metadata = pd.DataFrame(
        {
            "feature_uid": ["m"],
            "model_class": ["Convolutional"],
            "train_task": ["classification"],
            "objective_category": ["Supervised"],
        }
    )

    merged, _, _ = compute_importance_universality(
        loo,
        selectivity,
        metadata,
    )

    np.testing.assert_allclose(merged["importance"], loo["delta_r2"])
    np.testing.assert_allclose(
        merged["norm_fraction"], selectivity["norm_fraction"]
    )
