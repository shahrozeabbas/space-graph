"""Unit tests for utilities."""

import numpy as np

from space_graph.utils import beta_coef_from_rho_upper


def test_beta_coef_shape():
    p = 4
    rho = np.eye(p)
    rho[0, 1] = rho[1, 0] = 0.3
    coef = rho[np.triu_indices(p, k=1)]
    sig = np.ones(p)
    b = beta_coef_from_rho_upper(coef, sig)
    assert b.shape == (p, p)
