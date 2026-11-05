import numpy as np
import pytest

from src.metrics import explained_variance


def test_explained_variance_uses_centered_total_sum_of_squares():
    similarity = np.array([[1.0, 0.2], [0.2, 1.0]])
    components = np.array([[0.6], [0.4]])

    residual = similarity - components @ components.T
    centered = similarity - similarity.mean()
    expected = 1.0 - np.square(residual).sum() / np.square(centered).sum()

    assert explained_variance(similarity, components) == pytest.approx(expected)


def test_explained_variance_returns_zero_for_constant_similarity():
    similarity = np.ones((3, 3))
    components = np.ones((3, 1))

    assert explained_variance(similarity, components) == 0.0


@pytest.mark.parametrize(
    ("similarity", "components"),
    [
        (np.ones((2, 3)), np.ones((2, 1))),
        (np.eye(2), np.ones((3, 1))),
        (np.eye(2), np.ones(2)),
    ],
)
def test_explained_variance_validates_shapes(similarity, components):
    with pytest.raises(ValueError):
        explained_variance(similarity, components)
