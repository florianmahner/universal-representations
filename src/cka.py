import numpy as np

from .metrics import gaussian_kernel_similarity, median_sigma_heuristic

Array = np.ndarray


def center_gram(gram: Array, unbiased: bool = False) -> Array:
    """Center a symmetric Gram matrix (unbiased: HSIC estimator of Song et al.)."""
    gram = gram.copy()
    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram


def hsic(gram_x: Array, gram_y: Array, unbiased: bool = False) -> float:
    """Hilbert-Schmidt Independence Criterion between two Gram matrices."""
    kc = center_gram(gram_x, unbiased=unbiased)
    lc = center_gram(gram_y, unbiased=unbiased)
    return float(kc.ravel().dot(lc.ravel()))


def cka(gram_x: Array, gram_y: Array, unbiased: bool = False) -> float:
    """Centered Kernel Alignment in [0, 1] between two Gram matrices."""
    gram_x = center_gram(gram_x, unbiased=unbiased)
    gram_y = center_gram(gram_y, unbiased=unbiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    norm_x = np.linalg.norm(gram_x)
    norm_y = np.linalg.norm(gram_y)
    return float(scaled_hsic / (norm_x * norm_y))


def linear_cka(x: Array, y: Array) -> float:
    """Linear CKA computed in feature space (faster when d < n)."""
    x = x - np.mean(x, 0, keepdims=True)
    y = y - np.mean(y, 0, keepdims=True)
    similarity = np.linalg.norm(x.T.dot(y)) ** 2
    norm_x = np.linalg.norm(x.T.dot(x))
    norm_y = np.linalg.norm(y.T.dot(y))
    return float(similarity / (norm_x * norm_y))


def kernel_cka(x: Array, y: Array, sigma: float | None = None) -> float:
    """RBF-kernel CKA; sigma defaults to the median heuristic per matrix."""
    sigma_x = sigma or median_sigma_heuristic(x)
    sigma_y = sigma or median_sigma_heuristic(y)
    gram_x = gaussian_kernel_similarity(x, x, sigma=sigma_x)
    gram_y = gaussian_kernel_similarity(y, y, sigma=sigma_y)
    return cka(gram_x, gram_y)
