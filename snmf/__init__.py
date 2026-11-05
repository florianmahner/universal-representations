from __future__ import annotations

import logging

logging.getLogger("snmf").addHandler(logging.NullHandler())

from .consensus import AlignedConsensus, EnsembleFit
from .cross_validation import CVResult, cross_val_score
from .model import SNMF

__version__ = "0.1.1"


__all__: list[str] = [
    "SNMF",
    "CVResult",
    "cross_val_score",
    "EnsembleFit",
    "AlignedConsensus",
]
