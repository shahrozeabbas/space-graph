'''Tests for SPACE / JSRM.'''

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
        backend='auto',
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


def test_alpha_strength_runs():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 5))
    m = SPACE(alpha=0.6, max_outer_iter=2)
    m.fit(X)
    assert m.partial_correlation_.shape == (5, 5)


def test_space_gamma_tunable():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((25, 4))
    a = SPACE(alpha=0.8, gamma=1.0, max_outer_iter=2)
    b = SPACE(alpha=0.8, gamma=0.5, max_outer_iter=2)
    a.fit(X)
    b.fit(X)
    assert not np.allclose(a.partial_correlation_, b.partial_correlation_)


def test_space_gamma_out_of_range_raises():
    with pytest.raises(ValueError, match='gamma must be in'):
        SPACE(alpha=1.0, gamma=-0.1)
    with pytest.raises(ValueError, match='gamma must be in'):
        SPACE(alpha=1.0, gamma=1.1)


def test_space_init_rejects_bad_scalars():
    with pytest.raises(ValueError, match='alpha must be non-negative'):
        SPACE(alpha=-0.1)
    with pytest.raises(ValueError, match='max_outer_iter must be positive'):
        SPACE(alpha=1.0, max_outer_iter=0)
    with pytest.raises(ValueError, match='max_inner_iter must be positive'):
        SPACE(alpha=1.0, max_inner_iter=0)


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


def test_weight_degree_fits_and_reweights():
    rng = np.random.default_rng(13)
    p, n = 6, 60
    X = rng.multivariate_normal(np.zeros(p), _spd_cov(p, rng), size=n)
    m = SPACE(alpha=0.8, weight='degree', max_outer_iter=3).fit(X)
    r = m.partial_correlation_
    assert r.shape == (p, p)
    assert np.allclose(r, r.T)
    assert np.allclose(np.diag(r), 1.0)
    assert m.weight_.shape == (p,)
    assert np.all(np.isfinite(m.weight_))
    assert np.all(m.weight_ > 0)
    assert np.isclose(m.weight_.mean(), 1.0)


def test_weight_custom_mean_one_normalization():
    rng = np.random.default_rng(14)
    p, n = 5, 60
    X = rng.multivariate_normal(np.zeros(p), _spd_cov(p, rng), size=n)
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = SPACE(alpha=0.9, weight=w, max_outer_iter=2).fit(X)
    assert m.weight_.shape == (p,)
    assert np.isclose(m.weight_.mean(), 1.0)
    np.testing.assert_allclose(m.weight_, w / w.mean(), rtol=1e-14, atol=1e-14)


def test_weight_custom_scale_invariant():
    rng = np.random.default_rng(15)
    p, n = 5, 60
    X = rng.multivariate_normal(np.zeros(p), _spd_cov(p, rng), size=n)
    w = np.array([0.5, 1.5, 1.0, 2.0, 1.0])
    a = SPACE(alpha=0.9, weight=w, max_outer_iter=2).fit(X)
    b = SPACE(alpha=0.9, weight=w * 7.3, max_outer_iter=2).fit(X)
    np.testing.assert_allclose(
        a.partial_correlation_, b.partial_correlation_, rtol=1e-10, atol=1e-12
    )


def test_weight_custom_validation():
    X = np.random.default_rng(16).standard_normal((20, 4))
    with pytest.raises(ValueError, match='custom weight must have length p'):
        SPACE(alpha=1.0, weight=np.array([1.0, 2.0])).fit(X)
    with pytest.raises(ValueError, match='custom weight must be positive'):
        SPACE(alpha=1.0, weight=np.array([1.0, 0.0, 1.0, 1.0])).fit(X)
    with pytest.raises(ValueError, match='custom weight must be positive'):
        SPACE(alpha=1.0, weight=np.array([1.0, -1.0, 1.0, 1.0])).fit(X)


def test_standardize_false_with_fit_sig_updates_sigma_sensibly():
    rng = np.random.default_rng(11)
    p, n = 6, 60
    X = rng.standard_normal((n, p)) + np.array([1.0, -0.5, 2.0, 0.3, -1.2, 0.7])
    m = SPACE(alpha=0.5, max_outer_iter=3, standardize=False, fit_sig=True).fit(X)
    assert m.sig_ is not None
    assert np.all(np.isfinite(m.sig_))
    assert np.all(m.sig_ > 0)
    r = m.partial_correlation_
    assert np.allclose(r, r.T)
    assert np.allclose(np.diag(r), 1.0)


def test_standardize_false_raw_matches_precentered():
    rng = np.random.default_rng(12)
    p, n = 5, 80
    X_raw = rng.standard_normal((n, p)) + 3.0
    X_centered = X_raw - X_raw.mean(axis=0)
    a = SPACE(alpha=0.6, max_outer_iter=3, standardize=False, fit_sig=True).fit(X_raw)
    b = SPACE(alpha=0.6, max_outer_iter=3, standardize=False, fit_sig=True).fit(
        X_centered
    )
    np.testing.assert_allclose(a.sig_, b.sig_, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        a.partial_correlation_, b.partial_correlation_, rtol=1e-9, atol=1e-11
    )


def test_select_alpha_returns_best_in_grid():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 5))
    template = SPACE(max_outer_iter=2, max_inner_iter=500, backend='numpy')
    alphas = np.geomspace(0.05, 1.0, 5)
    best = template.select_alpha(X, alphas)
    assert isinstance(best, float)
    assert np.any(np.isclose(alphas, best))


def test_select_alpha_curve_aligned_with_grid():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((60, 5))
    template = SPACE(max_outer_iter=2, max_inner_iter=500, backend='numpy')
    alphas = np.geomspace(0.05, 1.0, 5)
    best, curve = template.select_alpha(X, alphas, return_curve=True)
    assert curve.shape == alphas.shape
    assert np.all(np.isfinite(curve))
    assert int(np.argmin(curve)) == int(np.argmin(np.abs(alphas - best)))


def test_select_alpha_default_criterion_is_bic():
    rng = np.random.default_rng(16)
    X = rng.standard_normal((50, 5))
    template = SPACE(max_outer_iter=2, max_inner_iter=500, backend='numpy')
    alphas = np.geomspace(0.05, 1.0, 5)
    best_default, curve_default = template.select_alpha(
        X, alphas, return_curve=True
    )
    best_bic, curve_bic = template.select_alpha(
        X, alphas, return_curve=True, criterion='bic'
    )
    assert best_default == best_bic
    np.testing.assert_allclose(curve_default, curve_bic, rtol=0, atol=0)


def test_select_alpha_does_not_mutate_template():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 4))
    template = SPACE(alpha=1.0, max_outer_iter=2, backend='numpy')
    _ = template.select_alpha(X, np.array([0.1, 0.5, 0.9]))
    assert template.alpha == 1.0
    assert template.partial_correlation_ is None
    assert template.sig_ is None
    assert template.weight_ is None


def test_jsrm_init_beta_none_matches_omitted():
    rng = np.random.default_rng(17)
    n, p = 20, 5
    X = rng.standard_normal((n, p))
    sr = np.ones(p)
    a = jsrm(X, sr, 0.4, 0.05, 300, tol=1e-6, backend='numpy')
    b = jsrm(X, sr, 0.4, 0.05, 300, tol=1e-6, backend='numpy', init_beta=None)
    np.testing.assert_allclose(a, b)


def test_jsrm_workspace_matches_reference():
    from space_graph.solver import _JsrmWorkspace

    rng = np.random.default_rng(22)
    n, p = 25, 6
    X = rng.standard_normal((n, p))
    sr = np.ones(p)
    lam1, lam2 = 0.35, 0.05
    ref = jsrm(X, sr, lam1, lam2, 400, tol=1e-6, backend='numpy')
    ws = _JsrmWorkspace.for_shape(n, p)
    out = jsrm(
        X,
        sr,
        lam1,
        lam2,
        400,
        tol=1e-6,
        backend='numpy',
        workspace=ws,
    )
    np.testing.assert_allclose(ref, out, rtol=1e-12, atol=1e-12)
    init = np.zeros((p, p), dtype=np.float64)
    ref2 = jsrm(X, sr, lam1, lam2, 400, tol=1e-6, backend='numpy', init_beta=init)
    out2 = jsrm(
        X,
        sr,
        lam1,
        lam2,
        400,
        tol=1e-6,
        backend='numpy',
        init_beta=init,
        workspace=ws,
    )
    np.testing.assert_allclose(ref2, out2, rtol=1e-12, atol=1e-12)


def test_select_alpha_warm_start_matches_cold_argmin():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))
    template = SPACE(max_outer_iter=2, max_inner_iter=500, backend='numpy')
    alphas = np.geomspace(0.05, 1.0, 7)
    best_cold, curve_cold = template.select_alpha(
        X, alphas, return_curve=True, warm_start=False
    )
    best_warm, curve_warm = template.select_alpha(
        X, alphas, return_curve=True, warm_start=True
    )
    assert np.isfinite(curve_cold).all()
    assert np.isfinite(curve_warm).all()
    assert best_cold == best_warm
    np.testing.assert_allclose(curve_cold, curve_warm, rtol=1e-5, atol=2e-4)


def test_select_alpha_bic_rows_match_scalar_grids():
    rng = np.random.default_rng(101)
    X = rng.standard_normal((55, 5))
    t = SPACE(max_outer_iter=2, backend='numpy')
    alphas = np.array([0.8, 0.05, 0.4, 0.2, 1.0])
    _, curve = t.select_alpha(X, alphas, return_curve=True, warm_start=False)
    assert curve.shape == alphas.shape
    for i in range(alphas.size):
        _, ref_curve = t.select_alpha(
            X, np.array([alphas[i]]), return_curve=True, warm_start=False
        )
        np.testing.assert_allclose(curve[i], ref_curve[0], rtol=1e-12, atol=1e-12)


def test_select_alpha_aic_rows_match_scalar_grids():
    rng = np.random.default_rng(102)
    X = rng.standard_normal((55, 5))
    t = SPACE(max_outer_iter=2, backend='numpy')
    alphas = np.array([0.8, 0.05, 0.4, 0.2, 1.0])
    best, curve = t.select_alpha(
        X, alphas, return_curve=True, warm_start=False, criterion='aic'
    )
    assert curve.shape == alphas.shape
    assert np.all(np.isfinite(curve))
    assert np.any(np.isclose(alphas, best))
    for i in range(alphas.size):
        _, ref_curve = t.select_alpha(
            X,
            np.array([alphas[i]]),
            return_curve=True,
            warm_start=False,
            criterion='aic',
        )
        np.testing.assert_allclose(curve[i], ref_curve[0], rtol=1e-12, atol=1e-12)


def test_select_alpha_rejects_bad_grids():
    X = np.random.default_rng(8).standard_normal((20, 3))
    m = SPACE(max_outer_iter=2)
    with pytest.raises(ValueError, match='alphas must be non-empty'):
        m.select_alpha(X, np.array([]))
    with pytest.raises(ValueError, match='alphas must be non-negative'):
        m.select_alpha(X, np.array([0.1, -0.2]))
    with pytest.raises(ValueError, match='criterion must be'):
        m.select_alpha(X, np.array([0.1, 0.2]), criterion='gcv')


def test_select_alpha_recovers_sparse_structure():
    rng = np.random.default_rng(1)
    n, p = 120, 6
    P = np.eye(p) * 2.0
    P[0, 1] = P[1, 0] = -1.0
    P[2, 3] = P[3, 2] = -1.0
    P[1, 2] = P[2, 1] = -0.6
    Sigma = np.linalg.inv(P)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    template = SPACE(max_outer_iter=3, max_inner_iter=1000, backend='numpy')
    best = template.select_alpha(
        X, np.geomspace(0.005, 1.0, 15), warm_start=False
    )
    chosen = SPACE(alpha=best, max_outer_iter=3, backend='numpy').fit(X)
    off = chosen.partial_correlation_[np.triu_indices(p, k=1)]
    assert np.any(np.abs(off) > 1e-6)
