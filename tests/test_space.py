"""Tests for SPACE / JSRM."""

from __future__ import annotations

import os

import numpy as np
import pytest

from space_graph.model import SPACE
from space_graph.solver import jsrm


def _spd_cov(p: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((p, p))
    return a @ a.T + p * np.eye(p)


def test_jsrm_matches_c_when_available():
    lib = '/Users/abba5hahroze/Desktop/space-lasso/space/src/libjsrm_test.so'
    if not os.path.isfile(lib):
        pytest.skip('compiled JSRM test library not present')
    import ctypes
    from numpy.ctypeslib import ndpointer

    cdll = ctypes.CDLL(lib)
    fun = cdll.JSRM
    fun.restype = None
    fun.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
    ]

    rng = np.random.default_rng(42)
    n, p = 12, 6
    X = rng.standard_normal((n, p)).astype(np.float32)
    sig = np.ones(p, dtype=np.float32)
    lam1, lam2 = 0.35, 0.0

    n_in = ctypes.c_int(n)
    p_in = ctypes.c_int(p)
    l1 = ctypes.c_float(lam1)
    l2 = ctypes.c_float(lam2)
    sigma_sr = np.sqrt(sig).astype(np.float32)
    n_iter = ctypes.c_int(500)
    iter_out = ctypes.c_int(0)
    beta = np.zeros(p * p, dtype=np.float32)
    y_flat = np.ascontiguousarray(X.ravel(order='C'))
    fun(
        ctypes.byref(n_in),
        ctypes.byref(p_in),
        ctypes.byref(l1),
        ctypes.byref(l2),
        y_flat,
        sigma_sr,
        ctypes.byref(n_iter),
        ctypes.byref(iter_out),
        beta,
    )
    beta_c = beta.reshape(p, p, order='C')

    beta_py = jsrm(
        X.astype(np.float64),
        sigma_sr.astype(np.float64),
        lam1,
        lam2,
        500,
        tol=1e-6,
    )
    np.testing.assert_allclose(beta_c, beta_py, atol=1e-4, rtol=1e-4)


def test_space_fit_symmetric_unit_diagonal():
    rng = np.random.default_rng(0)
    p, n = 8, 25
    cov = _spd_cov(p, rng)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)

    m = SPACE(alpha=1.0, max_outer_iter=2, max_inner_iter=500)
    m.fit(X)

    r = m.partial_correlation_
    assert r.shape == (p, p)
    assert np.allclose(r, r.T)
    assert np.allclose(np.diag(r), 1.0)
    assert m.sig_ is not None
    assert m.precision_ is not None


def test_alpha_strength_runs():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 5))
    m = SPACE(alpha=0.6, max_outer_iter=2)
    m.fit(X)
    assert m.partial_correlation_.shape == (5, 5)


def test_weight_uniform_vs_sig():
    rng = np.random.default_rng(2)
    p, n = 6, 40
    X = rng.multivariate_normal(np.zeros(p), _spd_cov(p, rng), size=n)

    a = SPACE(alpha=0.9, weight='uniform', max_outer_iter=2)
    a.fit(X)
    b = SPACE(alpha=0.9, weight='sig', max_outer_iter=2)
    b.fit(X)
    assert a.partial_correlation_.shape == b.partial_correlation_.shape


def test_weight_equal_alias_matches_uniform():
    rng = np.random.default_rng(3)
    p, n = 5, 30
    X = rng.multivariate_normal(np.zeros(p), _spd_cov(p, rng), size=n)
    u = SPACE(alpha=0.95, weight='uniform', max_outer_iter=2, tol=1e-6)
    e = SPACE(alpha=0.95, weight='equal', max_outer_iter=2, tol=1e-6)
    u.fit(X)
    e.fit(X)
    np.testing.assert_allclose(u.partial_correlation_, e.partial_correlation_)
