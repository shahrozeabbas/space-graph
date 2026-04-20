'''Map sklearn-style ``alpha`` and mix ``gamma`` (γ) to JSRM ``lam1``, ``lam2``.'''

from __future__ import annotations


def alpha_to_penalties(alpha: float, gamma: float) -> tuple[float, float]:
    '''
    Penalties for the inner JSRM solver.

    ``lam1 = alpha * gamma``, ``lam2 = alpha * (1 - gamma)``.

    ``alpha`` is overall strength (sklearn-style). ``gamma`` in ``[0, 1]`` is the
    mix γ: ``gamma == 1`` gives ``lam2 = 0``, matching R ``space::space.joint``
    default ``lam2 = 0`` for a given ``alpha`` (R's ``lam1``). Range checks live
    on ``SPACE.__init__`` (the public boundary).
    '''
    a = float(alpha)
    if a < 0.0:
        raise ValueError('alpha must be non-negative')
    g = float(gamma)
    return a * g, a * (1.0 - g)
