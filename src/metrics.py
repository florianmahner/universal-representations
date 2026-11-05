import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances

Array = np.ndarray


AVAILABLE_METRICS = (
    "pearson",
    "cosine",
    "euclidean",
    "gaussian_kernel",
)


def validate_metric(metric: str):
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Metric {metric} not supported.")


def compute_similarity(x: Array, y: Array, metric: str, **kwargs) -> float | Array:
    """Compute similarity between two matrices."""
    validate_metric(metric)
    similarity_functions = {
        "pearson": pearson_similarity,
        "cosine": cosine_similarity,
        "euclidean": euclidean_similarity,
        "gaussian_kernel": gaussian_kernel_similarity,
    }
    return similarity_functions[metric](x, y, **kwargs)


def compute_distance(x: Array, y: Array, metric: str, **kwargs) -> float | Array:
    """Compute distance between two matrices."""
    validate_metric(metric)
    distance_functions = {
        "pearson": pearson_distance,
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
    }
    return distance_functions[metric](x, y, **kwargs)


def pearson_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson similarity between two matrices, returning an n x n similarity matrix.
    NOTE this is the same as np.corrcoef(x, y, rowvar=True)"""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    xy = x_centered @ y_centered.T
    x_norms = np.linalg.norm(x_centered, axis=1, keepdims=True)
    y_norms = np.linalg.norm(y_centered, axis=1, keepdims=True)

    # Safely handle zero norms by setting invalid entries to 0
    norm_product = x_norms @ y_norms.T
    valid = norm_product > 0
    s = np.zeros_like(norm_product)
    s[valid] = xy[valid] / norm_product[valid]
    s = np.nan_to_num(s, nan=0.0)
    s = np.clip(s, a_min=-1, a_max=1)
    np.fill_diagonal(s, 1)
    return s


def pearson_distance(x: Array, y: Array) -> float | Array:
    """Pearson distance between two matrices."""
    return 1 - pearson_similarity(x, y)


def cosine_similarity(x: Array, y: Array) -> float | Array:
    """Cosine similarity between two matrices."""
    return 1 - cosine_distance(x, y)


def cosine_distance(x: Array, y: Array) -> float | Array:
    """Cosine distance between two matrices."""
    distance = pairwise_distances(x, y, metric="cosine")
    return distance


def euclidean_similarity(x: Array, y: Array) -> float | Array:
    """Euclidean similarity between two matrices."""
    distance = euclidean_distance(x, y)
    s = 1 / (1 + distance)
    return s


def euclidean_distance(x: Array, y: Array) -> float | Array:
    """Euclidean distance between two matrices
    see https://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    """
    distance = pairwise_distances(x, y, metric="euclidean")
    return distance


def explained_variance(similarity: Array, components: Array) -> float:
    """Return the mean-centered variance explained by ``components @ components.T``.

    This is the definition used for bandwidth selection in the manuscript.  The
    denominator is the total sum of squares around the mean similarity, rather
    than the uncentered squared Frobenius norm.
    """
    similarity = np.asarray(similarity)
    components = np.asarray(components)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            f"similarity must be a square matrix, got shape {similarity.shape}"
        )
    if components.ndim != 2 or components.shape[0] != similarity.shape[0]:
        raise ValueError(
            "components must be a 2D matrix with one row per similarity item, "
            f"got {components.shape}"
        )

    residual = similarity - components @ components.T
    total_ss = float(np.square(similarity).sum())
    total_ss -= float(similarity.size) * float(similarity.mean()) ** 2
    if total_ss <= 0:
        return 0.0
    return 1.0 - float(np.square(residual).sum()) / total_ss



def median_sigma_heuristic(x: Array, y: Array = None) -> float:
    """
    Return sigma (bandwidth) for RBF kernel using the median heuristic.

    This computes the median of pairwise distances between x and y,
    then returns sigma = median (direct bandwidth parameter).
    """
    if y is None:
        d = pdist(x, metric="euclidean")
    else:
        d = pairwise_distances(x, y).ravel()

    d = d[d > 0]
    if len(d) == 0:
        raise ValueError("All pairwise distances are zero")

    return np.median(d)


def rbf_kernel(x: Array, y: Array, sigma: float) -> Array:
    """RBF kernel between two matrices."""
    S = pairwise_distances(x, y, metric="euclidean")
    S **= 2
    S *= -1.0 / (2 * sigma**2)
    np.exp(S, out=S)
    return S



def gaussian_kernel_similarity(
    x: Array, y: Array, sigma: float | None = None
) -> float | Array:
    """Gaussian kernel similarity between two matrices aka RBF kernel.

    Params:
        x: Array, First input array.
        y: Array, Second input array.
        sigma: float, Controls the width of the kernel. Higher values lead to more
            smooth similarity functions. If not provided, median distance is used.
    """
    if sigma is None:
        sigma = median_sigma_heuristic(x, y)

    similarity = rbf_kernel(x, y, sigma)
    return similarity
