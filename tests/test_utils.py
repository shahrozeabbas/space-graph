"""Unit tests for utilities."""

import numpy as np

from space_graph.penalties import LAMBDA, alpha_to_penalties
from space_graph.utils import beta_coef_from_rho_upper


def test_alpha_to_penalties_matches_r_default_lam2_zero():
    assert LAMBDA == 1.0
    for a in (0.0, 0.35, 1.0, 2.0):
        lam1, lam2 = alpha_to_penalties(a)
        assert lam1 == a
        assert lam2 == 0.0


def test_beta_coef_shape():
    p = 4
    rho = np.eye(p)
    rho[0, 1] = rho[1, 0] = 0.3
    coef = rho[np.triu_indices(p, k=1)]
    sig = np.ones(p)
    b = beta_coef_from_rho_upper(coef, sig)
    assert b.shape == (p, p)
