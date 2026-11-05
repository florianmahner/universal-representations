import numpy as np
from src.stability import (
    align_seeds_to_reference,
    factorization_stability,
    reference_seed_index,
)


def test_identical_matrices():
    w = np.random.default_rng(0).random((200, 10))
    ws = [w.copy() for _ in range(5)]
    s = factorization_stability(ws)
    assert s.shape == (10,)
    assert np.all(s > 0.99)


def test_random_matrices():
    rng = np.random.default_rng(0)
    ws = [rng.random((200, 10)) for _ in range(5)]
    s = factorization_stability(ws)
    assert s.shape == (10,)
    assert np.all(s < 0.15)


def test_permuted_matrices():
    rng = np.random.default_rng(0)
    w = rng.random((200, 8))
    ws = [w[:, rng.permutation(8)] for _ in range(5)]
    s = factorization_stability(ws)
    assert s.shape == (8,)
    assert np.all(s > 0.99)


def test_all_pair_stability_uses_common_dimension_order():
    rng = np.random.default_rng(4)
    w = rng.random((300, 4))
    permutations = [
        np.array([0, 1, 2, 3]),
        np.array([2, 0, 3, 1]),
        np.array([1, 3, 0, 2]),
        np.array([3, 2, 1, 0]),
        np.array([0, 2, 1, 3]),
    ]
    ws = [w[:, p] + rng.normal(0, 0.001, w.shape) for p in permutations]
    ref = reference_seed_index(ws)
    aligned = align_seeds_to_reference(ws, ref)
    for candidate in aligned:
        diagonal = np.diag(np.corrcoef(aligned[ref].T, candidate.T)[:4, 4:])
        assert np.all(diagonal > 0.99)
    assert np.all(factorization_stability(ws) > 0.99)


def test_too_few_seeds():
    w = np.random.default_rng(0).random((200, 10))
    s = factorization_stability([w])
    assert np.all(np.isnan(s))


def test_empty_list():
    s = factorization_stability([])
    assert s.shape == (0,)


def test_two_seeds():
    rng = np.random.default_rng(0)
    w = rng.random((200, 5))
    ws = [w, w + rng.normal(0, 0.1, w.shape)]
    s = factorization_stability(ws)
    assert s.shape == (5,)
    assert np.all(s > 0.5)
