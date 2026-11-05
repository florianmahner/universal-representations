import numpy as np

from experiments.universality.shared.core import (
    METRICS,
    cos2_match_indices,
    cos2_permutation,
    cos2_perm_null_threshold,
    cos2_perm_null_thresholds,
    max_correlation,
    null_adjust,
)
from experiments.universality.metric.validate.null_correction.run import (
    process_one_model,
)


def test_cos2_perm_identical_matrices():
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((100, 10)))
    scores = cos2_permutation(w, w)
    assert scores.shape == (10,)
    np.testing.assert_allclose(scores, 1.0, atol=1e-6)


def test_cos2_perm_permuted_columns():
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((100, 8)))
    w_perm = w[:, [3, 7, 1, 5, 0, 4, 6, 2]]
    scores = cos2_permutation(w, w_perm)
    assert scores.shape == (8,)
    np.testing.assert_allclose(scores, 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        cos2_match_indices(w, w_perm),
        [3, 7, 1, 5, 0, 4, 6, 2],
    )


def test_cos2_perm_random_matrices():
    rng = np.random.default_rng(42)
    w1 = np.abs(rng.standard_normal((200, 10)))
    w2 = np.abs(rng.standard_normal((200, 10)))
    scores = cos2_permutation(w1, w2)
    assert scores.shape == (10,)
    assert scores.mean() < 0.6


def test_cos2_perm_more_source_than_target():
    rng = np.random.default_rng(42)
    w_source = np.abs(rng.standard_normal((100, 20)))
    w_target = np.abs(rng.standard_normal((100, 10)))
    scores = cos2_permutation(w_source, w_target)
    assert scores.shape == (10,)


def test_cos2_perm_fewer_source_than_target():
    rng = np.random.default_rng(42)
    w_source = np.abs(rng.standard_normal((100, 5)))
    w_target = np.abs(rng.standard_normal((100, 10)))
    scores = cos2_permutation(w_source, w_target)
    assert scores.shape == (10,)
    assert (scores == 0.0).sum() == 5


def test_metrics_registry():
    assert set(METRICS.keys()) == {"max_corr", "cos2_perm"}
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((100, 5)))
    for name, fn in METRICS.items():
        scores = fn(w, w)
        assert scores.shape == (5,), f"{name} returned wrong shape"


def test_max_corr_unchanged():
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((100, 10)))
    scores = max_correlation(w, w)
    np.testing.assert_allclose(scores, 1.0, atol=1e-6)


def test_null_adjust_formula():
    raw = np.array([0.8, 0.6, 0.3])
    threshold = np.array([0.5, 0.5, 0.5])
    adj = null_adjust(raw, threshold)
    assert adj.shape == (3,)
    np.testing.assert_allclose(adj, [0.6, 0.2, 0.0])


def test_null_adjust_clips_to_unit():
    raw = np.array([1.0, 0.0])
    threshold = np.array([0.5, 0.8])
    adj = null_adjust(raw, threshold)
    assert (adj >= 0).all()
    assert (adj <= 1).all()
    np.testing.assert_allclose(adj[0], 1.0)
    np.testing.assert_allclose(adj[1], 0.0)


def test_cos2_perm_null_threshold_shape_and_bounds():
    rng = np.random.default_rng(42)
    w_target = np.abs(rng.standard_normal((100, 10)))
    w_source = np.abs(rng.standard_normal((100, 10)))
    threshold = cos2_perm_null_threshold(
        w_source, w_target, n_perms=50, seed=42,
    )
    assert threshold.shape == (10,)
    assert (threshold >= 0).all()
    assert (threshold <= 1).all()


def test_adjusted_leq_raw():
    rng = np.random.default_rng(42)
    w_target = np.abs(rng.standard_normal((100, 10)))
    w_source = np.abs(rng.standard_normal((100, 10)))
    raw = cos2_permutation(w_source, w_target)
    threshold = cos2_perm_null_threshold(
        w_source, w_target, n_perms=50, seed=42,
    )
    adj = null_adjust(raw, threshold)
    assert (adj <= raw + 1e-6).all()


def test_null_threshold_below_identical():
    rng = np.random.default_rng(42)
    w = np.abs(rng.standard_normal((200, 8)))
    source = np.abs(rng.standard_normal((200, 8)))
    threshold = cos2_perm_null_threshold(
        source, w, n_perms=100, seed=42,
    )
    raw = cos2_permutation(w, w)
    assert (raw > threshold).all(), "Identical matrix should exceed null threshold"


def test_null_threshold_is_shared_across_sources():
    """The published null estimates one threshold and applies it to every source."""
    rng = np.random.default_rng(5)
    target = np.abs(rng.standard_normal((80, 6)))
    sources = [
        np.abs(rng.standard_normal((80, 6))),
        np.abs(rng.standard_normal((80, 6))) ** 3,
    ]
    thresholds = cos2_perm_null_thresholds(
        target,
        sources,
        n_perms=40,
        seed=11,
    )
    assert thresholds.shape == (2, 6)
    np.testing.assert_array_equal(thresholds[0], thresholds[1])


def test_null_correction_diagnostic_matches_canonical_pairwise_estimator():
    rng = np.random.default_rng(21)
    components = {
        "target": np.abs(rng.standard_normal((60, 5))),
        "source_a": np.abs(rng.standard_normal((60, 5))),
        "source_b": np.abs(rng.standard_normal((60, 5))) ** 2,
    }
    n_perms = 20
    seed = 13

    result = process_one_model("target", components, n_perms, seed)
    sources = [components["source_a"], components["source_b"]]
    raw_by_pair = np.stack(
        [cos2_permutation(source, components["target"]) for source in sources]
    )
    thresholds = cos2_perm_null_thresholds(
        components["target"],
        sources,
        n_perms=n_perms,
        seed=seed,
    )

    np.testing.assert_allclose(result["raw"], raw_by_pair.mean(axis=0))
    np.testing.assert_allclose(
        result["null_threshold"],
        thresholds.mean(axis=0),
    )
    np.testing.assert_allclose(
        result["corrected"],
        null_adjust(raw_by_pair, thresholds).mean(axis=0),
    )
