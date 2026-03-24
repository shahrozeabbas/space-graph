"""Equivalence tests for column-wise matmul helpers vs dense reference."""

import numpy as np

from space_graph.solver import _ym_times_elementwise
from space_graph.utils import _y_times_beta, inv_sig_diag_new


def test_ym_times_elementwise_matches_dense_w():
    rng = np.random.default_rng(0)
    n, p = 40, 12
    Y_m = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, p))
    beta = (beta + beta.T) / 2.0
    np.fill_diagonal(beta, 0.0)
    sigma_sr = np.abs(rng.standard_normal(p)) + 0.5
    B = sigma_sr[:, None] / sigma_sr[None, :]

    F_helper = _ym_times_elementwise(Y_m, beta, B)
    W = beta * B
    F_dense = Y_m @ W
    np.testing.assert_allclose(F_helper, F_dense, rtol=1e-14, atol=1e-14)


def test_ym_times_elementwise_sparse_matches_dense():
    rng = np.random.default_rng(11)
    n, p = 25, 9
    Y_m = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, p))
    beta = (beta + beta.T) / 2.0
    np.fill_diagonal(beta, 0.0)
    beta[np.abs(beta) < 0.4] = 0.0
    sigma_sr = np.abs(rng.standard_normal(p)) + 0.5
    B = sigma_sr[:, None] / sigma_sr[None, :]

    F_helper = _ym_times_elementwise(Y_m, beta, B)
    F_dense = Y_m @ (beta * B)
    np.testing.assert_allclose(F_helper, F_dense, rtol=1e-14, atol=1e-14)


def test_y_times_beta_matches_dense():
    rng = np.random.default_rng(1)
    n, p = 35, 10
    Y = rng.standard_normal((n, p))
    b = rng.standard_normal((p, p))
    np.fill_diagonal(b, 0.0)

    esti_helper = _y_times_beta(Y, b)
    esti_dense = Y @ b
    np.testing.assert_allclose(esti_helper, esti_dense, rtol=1e-14, atol=1e-14)


def test_y_times_beta_sparse_matches_dense():
    rng = np.random.default_rng(12)
    n, p = 22, 8
    Y = rng.standard_normal((n, p))
    b = rng.standard_normal((p, p))
    np.fill_diagonal(b, 0.0)
    b[np.abs(b) < 0.35] = 0.0

    esti_helper = _y_times_beta(Y, b)
    esti_dense = Y @ b
    np.testing.assert_allclose(esti_helper, esti_dense, rtol=1e-14, atol=1e-14)


def test_inv_sig_diag_new_matches_manual_dense():
    rng = np.random.default_rng(2)
    n, p = 30, 8
    Y = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, p))
    beta = (beta + beta.T) / 2.0

    sig_helper = inv_sig_diag_new(Y, beta)

    b = beta.copy()
    np.fill_diagonal(b, 0.0)
    esti = Y @ b
    sig_ref = 1.0 / np.mean((Y - esti) ** 2, axis=0)
    np.testing.assert_allclose(sig_helper, sig_ref, rtol=1e-14, atol=1e-14)
