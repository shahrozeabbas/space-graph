"""Weight modes for SPACE outer loop (R `space.joint` semantics)."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

WeightInput = Union[
    Literal['uniform'],
    Literal['equal'],
    Literal['sig'],
    Literal['degree'],
    np.ndarray,
]


def resolve_weight(
    weight: WeightInput,
    p: int,
) -> tuple[np.ndarray, bool, int]:
    """
    Returns (weight_vector, update_each_outer_iter, tag).

    tag: 0 uniform, 1 sig-based, 2 degree-based, 3 custom.
    """
    if isinstance(weight, str):
        if weight in ('uniform', 'equal'):
            return np.ones(p, dtype=np.float64), False, 0
        if weight == 'sig':
            return np.ones(p, dtype=np.float64), True, 1
        if weight == 'degree':
            return np.ones(p, dtype=np.float64), True, 2
        raise ValueError(f'unknown weight mode: {weight}')

    w = np.asarray(weight, dtype=np.float64).ravel()
    if w.shape[0] != p:
        raise ValueError('custom weight must have length p')
    if np.any(w <= 0):
        raise ValueError('custom weight must be positive')
    w = w / w.mean()
    return w, False, 3


def rescale_degree_weights(par_cor: np.ndarray) -> np.ndarray:
    """R: temp.w <- row sums of |rho|>1e-6; +max; normalize to mean 1."""
    p = par_cor.shape[0]
    temp_w = np.sum(np.abs(par_cor) > 1e-6, axis=1).astype(np.float64)
    temp_w = temp_w + np.max(temp_w)
    return temp_w / np.sum(temp_w) * p
