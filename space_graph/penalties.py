"""Map sklearn-style strength ``alpha`` and fixed mix λ to JSRM ``lam1``, ``lam2``."""

from __future__ import annotations

from typing import Tuple

# Hardcoded mix λ (not exposed): matches R ``space::space.joint`` defaults
# ``lam1`` from the user and ``lam2=0``. With ``lam1 = alpha * λ`` and
# ``lam2 = alpha * (1 - λ)``, setting ``λ = 1`` gives ``lam2 = 0`` for all ``alpha``.
LAMBDA = 0.75


def alpha_to_penalties(alpha: float) -> Tuple[float, float]:
    """
    Penalties for the inner JSRM solver.

    ``lam1 = alpha * LAMBDA``, ``lam2 = alpha * (1 - LAMBDA)``.

    ``alpha`` is overall strength (sklearn-style). ``LAMBDA`` is the fixed mix
    corresponding to R's default ``lam2 = 0`` when ``LAMBDA == 1``.
    """
    a = float(alpha)
    if a < 0.0:
        raise ValueError('alpha must be non-negative')
    lam = float(LAMBDA)
    return a * lam, a * (1.0 - lam)
