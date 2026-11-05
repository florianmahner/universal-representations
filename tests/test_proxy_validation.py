import numpy as np

from experiments.universality.metric.validate.proxy_validation.run import ridge_r2


def test_ridge_r2_is_held_out_and_recovers_real_signal():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(120, 5))
    signal = (source[:, 0] - 0.5 * source[:, 1])[:, None]
    target = np.column_stack(
        [
            signal[:, 0] + rng.normal(scale=0.05, size=len(source)),
            rng.normal(size=len(source)),
        ]
    )

    scores = ridge_r2(source, target, n_splits=5, seed=9)

    assert scores[0] > 0.9
    assert scores[1] < 0.2
