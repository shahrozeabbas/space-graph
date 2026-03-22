"""Map sklearn-style ``alpha`` and mix ``gamma`` (γ) to JSRM ``lam1``, ``lam2``."""

from __future__ import annotations

from typing import Tuple


def alpha_to_penalties(alpha: float, gamma: float) -> Tuple[float, float]:
    """
    Penalties for the inner JSRM solver.

    ``lam1 = alpha * gamma``, ``lam2 = alpha * (1 - gamma)``.

    ``alpha`` is overall strength (sklearn-style). ``gamma`` in ``[0, 1]`` is the
    mix γ: ``gamma == 1`` gives ``lam2 = 0``, matching R ``space::space.joint``
    default ``lam2 = 0`` for a given ``alpha`` (R's ``lam1``).
    """
    a = float(alpha)
    if a < 0.0:
        raise ValueError('alpha must be non-negative')
    g = float(gamma)
    if g < 0.0 or g > 1.0:
        raise ValueError('gamma must be in [0, 1]')
    return a * g, a * (1.0 - g)
