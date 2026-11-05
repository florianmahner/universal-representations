#!/usr/bin/env python3

import numpy as np
from scipy.stats import pearsonr, spearmanr

Array = np.ndarray


def apply_transform(
    x: np.ndarray, transform: str | list[str], axis: int = 0, verbose: bool = False
) -> np.ndarray:
    transform_to_func = {
        "center": center,
        "standardize": standardize,
        "normalize_l2": normalize_l2,
        "normalize_l1": normalize_l1,
        "positive_shift": positive_shift,
        "softmax": softmax,
        "log_softmax": log_softmax,
        "log": log,
    }

    if isinstance(transform, str):
        if transform not in transform_to_func:
            raise ValueError(
                f"Transform {transform} not found. Available transforms: {transform_to_func.keys()}"
            )
        transform = [transform]

    for t in transform:
        if verbose:
            print(f"Applying {t}")
        x = transform_to_func[t](x, axis=axis)
    return x


def center(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Center the input array by subtracting the mean along the specified axis."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    return x - mean


def standardize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize the input array by centering and scaling to unit variance."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True, ddof=0)
    std[std == 0] = 1  # Prevent division by zero
    return (x - mean) / std



def normalize_l2(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L2 norm."""
    x = np.asarray(x)
    norms = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    return x / (norms + eps)


def normalize_l1(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L1 norm."""
    x = np.asarray(x)
    norms = np.sum(np.abs(x), axis=axis, keepdims=True)
    return x / (norms + eps)


def positive_shift(x: Array, axis: int = 0) -> Array:
    """Shift all values in the array to be non-negative."""
    return x - np.min(x)


def softmax(x: Array, axis: int = 1) -> Array:
    """Apply the softmax function along axis 1."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def log(x: Array) -> Array:
    """Log transform the input array."""
    return np.log(x)


def log_softmax(x: Array, axis: int = 1) -> Array:
    """Apply the log softmax function along axis 1."""
    return np.log(softmax(x, axis=axis))



def compute_correlation_coeff(x: Array, y: Array, method: str = "pearson"):
    """Compute the correlation coefficient and p-value between two matrices."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Dictionary mapping method names to functions
    correlation_methods = {
        "pearson": pearsonr,
        "spearman": spearmanr,
    }

    # Validate the method
    if method not in correlation_methods:
        raise ValueError(f"Method {method} not supported.")

    corr, pval = correlation_methods[method](x.flatten(), y.flatten())
    return corr, pval



def fisher_z_transform(pearson_r: Array) -> Array:
    """Perform Fisher Z-transform on Pearson r values, ensuring valid input range."""
    pearson_r = np.asarray(pearson_r, dtype=np.float64)  # Ensure floating point type
    eps = 1e-7  # Slightly larger than before for better numerical stability

    # Replace invalid values (e.g., NaNs or infinite)
    pearson_r = np.where(np.isfinite(pearson_r), pearson_r, np.nan)
    pearson_r = np.clip(pearson_r, -1 + eps, 1 - eps)

    return np.arctanh(pearson_r)


def average_pearson_r(pearson_r: Array, axis: int = 0) -> Array:
    """Compute the average Pearson r after Fisher Z-transform and inverse transformation."""
    fisher_z = fisher_z_transform(pearson_r)
    mean_z = np.nanmean(fisher_z, axis=axis)

    return np.tanh(mean_z)


def _normal_pdf_numpy(x: Array, mean: Array, std: Array) -> Array:
    """Compute the probability density function of a normal distribution."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
