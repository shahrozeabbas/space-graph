from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpaceState:
    """Internal state for one SPACE fit."""

    partial_correlation: np.ndarray
    sig: np.ndarray
    weight: np.ndarray
    outer_iter: int
