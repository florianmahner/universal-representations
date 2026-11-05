__all__ = []

try:
    from .data import ThingsDataset
except ModuleNotFoundError:
    pass
else:
    __all__.append("ThingsDataset")

try:
    from .monkey import load_macaque
except ModuleNotFoundError:
    pass
else:
    __all__.append("load_macaque")

try:
    from .metrics import (
        gaussian_kernel_similarity,
        compute_similarity,
    )
except ModuleNotFoundError:  # optional runtime dependencies not installed
    pass

try:
    from .plotting import setup_plotting
except ModuleNotFoundError:
    pass

for _name in (
    "gaussian_kernel_similarity",
    "compute_similarity",
    "setup_plotting",
):
    if _name in globals():
        __all__.append(_name)
