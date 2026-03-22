"""Map public `alpha` to JSRM lam1, lam2."""

from __future__ import annotations

from typing import Tuple


def alpha_to_penalties(alpha: float) -> Tuple[float, float]:
    """
    User contract: alpha in [0, 1] gives lam1 = alpha, lam2 = 1 - alpha.

    This matches the reference elastic-net-style solver where both penalties appear.
    """
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError('alpha must be in [0, 1]')
    return a, 1.0 - a
