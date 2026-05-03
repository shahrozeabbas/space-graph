'''Rust JSRM shooting loop: parity and dispatch when ``space_graph._rust`` is built.'''

from __future__ import annotations

import numpy as np
import pytest

import space_graph.solver as solver_mod
from space_graph.model import SPACE

try:
    import space_graph._rust as _rust  # type: ignore[import-not-found]
except ImportError:
    _rust = None

HAS_RUST = _rust is not None


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_jsrm_workspace_class_matches_numpy() -> None:
    assert _rust is not None
    X, sr = _small_problem()
    n, p = X.shape
    ws = _rust.JsrmWorkspace(int(n), int(p))
    b_rs = ws.solve(X, sr, 0.35, 0.08, 400, 1e-6, None)
    b_np = solver_mod.jsrm(
        X, sr, 0.35, 0.08, n_iter=400, tol=1e-6, backend='numpy'
    )
    np.testing.assert_allclose(b_rs, b_np, rtol=0, atol=1e-9)


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_jsrm_workspace_reuse_two_penalties_matches_numpy() -> None:
    assert _rust is not None
    X, sr = _small_problem()
    n, p = X.shape
    ws = _rust.JsrmWorkspace(int(n), int(p))
    b1 = ws.solve(X, sr, 0.35, 0.08, 300, 1e-6, None)
    b2 = ws.solve(X, sr, 0.22, 0.04, 300, 1e-6, None)
    r1 = solver_mod.jsrm(
        X, sr, 0.35, 0.08, n_iter=300, tol=1e-6, backend='numpy'
    )
    r2 = solver_mod.jsrm(
        X, sr, 0.22, 0.04, n_iter=300, tol=1e-6, backend='numpy'
    )
    np.testing.assert_allclose(b1, r1, rtol=0, atol=1e-9)
    np.testing.assert_allclose(b2, r2, rtol=0, atol=1e-9)


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_python_workspace_carries_rust_ws_when_available() -> None:
    n, p = 40, 8
    py_ws = solver_mod._JsrmWorkspace.for_shape(n, p)
    if solver_mod.get_rust_jsrm_solve() is not None:
        assert py_ws.rust_ws is not None


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_jsrm_rust_with_python_workspace_matches_numpy() -> None:
    X, sr = _small_problem()
    ws = solver_mod._JsrmWorkspace.for_shape(X.shape[0], X.shape[1])
    if ws.rust_ws is None:
        pytest.skip('Rust workspace not available')
    b_rs = solver_mod.jsrm(
        X,
        sr,
        0.31,
        0.06,
        n_iter=350,
        tol=1e-6,
        backend='rust',
        workspace=ws,
    )
    b_np = solver_mod.jsrm(
        X,
        sr,
        0.31,
        0.06,
        n_iter=350,
        tol=1e-6,
        backend='numpy',
        workspace=ws,
    )
    np.testing.assert_allclose(b_rs, b_np, rtol=0, atol=1e-9)


def _small_problem() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(202)
    n, p = 40, 8
    return rng.standard_normal((n, p)), np.ones(p, dtype=np.float64)


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_jsrm_rust_matches_numpy() -> None:
    X, sr = _small_problem()
    lam1, lam2 = 0.35, 0.08
    b_np = solver_mod.jsrm(
        X, sr, lam1, lam2, n_iter=400, tol=1e-6, backend='numpy'
    )
    b_rs = solver_mod.jsrm(
        X, sr, lam1, lam2, n_iter=400, tol=1e-6, backend='rust'
    )
    np.testing.assert_allclose(b_rs, b_np, rtol=0, atol=1e-9)


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_jsrm_rust_warm_init_matches_numpy() -> None:
    X, sr = _small_problem()
    rng = np.random.default_rng(303)
    p = X.shape[1]
    init = rng.standard_normal((p, p))
    init = 0.5 * (init + init.T)
    np.fill_diagonal(init, 0.0)
    lam1, lam2 = 0.3, 0.06
    b_np = solver_mod.jsrm(
        X,
        sr,
        lam1,
        lam2,
        n_iter=400,
        tol=1e-6,
        backend='numpy',
        init_beta=init,
    )
    b_rs = solver_mod.jsrm(
        X,
        sr,
        lam1,
        lam2,
        n_iter=400,
        tol=1e-6,
        backend='rust',
        init_beta=init,
    )
    np.testing.assert_allclose(b_rs, b_np, rtol=0, atol=1e-9)


def test_jsrm_rust_raises_when_extension_missing(monkeypatch) -> None:
    monkeypatch.setattr(solver_mod, 'get_rust_jsrm_solve', lambda: None)
    X, sr = _small_problem()
    with pytest.raises(ImportError, match='rust'):
        solver_mod.jsrm(
            X, sr, 0.2, 0.0, n_iter=50, tol=1e-6, backend='rust'
        )


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_space_fit_rust_matches_numpy() -> None:
    rng = np.random.default_rng(17)
    X = rng.standard_normal((35, 6))
    m_np = SPACE(
        alpha=0.6, max_outer_iter=2, max_inner_iter=300, backend='numpy'
    )
    m_rs = SPACE(
        alpha=0.6, max_outer_iter=2, max_inner_iter=300, backend='rust'
    )
    m_np.fit(X)
    m_rs.fit(X)
    np.testing.assert_allclose(
        m_rs.partial_correlation_, m_np.partial_correlation_, atol=1e-9
    )
    np.testing.assert_allclose(m_rs.sig_, m_np.sig_, atol=1e-9)


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_select_alpha_rust_matches_numpy() -> None:
    rng = np.random.default_rng(91)
    X = rng.standard_normal((45, 5))
    alphas = np.array([0.15, 0.4, 0.75], dtype=np.float64)
    t_np = SPACE(
        max_outer_iter=2, max_inner_iter=500, backend='numpy'
    )
    t_rs = SPACE(
        max_outer_iter=2, max_inner_iter=500, backend='rust'
    )
    best_np, curve_np = t_np.select_alpha(
        X, alphas, return_curve=True, warm_start=False
    )
    best_rs, curve_rs = t_rs.select_alpha(
        X, alphas, return_curve=True, warm_start=False
    )
    assert best_rs == best_np
    np.testing.assert_allclose(curve_rs, curve_np, rtol=0, atol=1e-6)
    assert np.all(np.isfinite(curve_rs))


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_select_alpha_warm_start_rust_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))
    alphas = np.geomspace(0.05, 1.0, 7)
    t_np = SPACE(
        max_outer_iter=2, max_inner_iter=500, backend='numpy'
    )
    t_rs = SPACE(
        max_outer_iter=2, max_inner_iter=500, backend='rust'
    )
    best_np, curve_np = t_np.select_alpha(
        X, alphas, return_curve=True, warm_start=True
    )
    best_rs, curve_rs = t_rs.select_alpha(
        X, alphas, return_curve=True, warm_start=True
    )
    assert best_rs == best_np
    np.testing.assert_allclose(curve_rs, curve_np, rtol=1e-5, atol=2e-4)
    assert np.all(np.isfinite(curve_rs))


@pytest.mark.skipif(not HAS_RUST, reason='space_graph._rust extension not installed')
def test_backend_auto_prefers_rust_when_present(monkeypatch) -> None:
    solve_fn = _rust.jsrm_solve
    called = {'rust': False}

    def _get_wrapped():
        def _inner(*args, **kwargs):
            called['rust'] = True
            return solve_fn(*args, **kwargs)

        return _inner

    monkeypatch.setattr(solver_mod, 'get_rust_jsrm_solve', _get_wrapped)

    X, sr = _small_problem()
    solver_mod.jsrm(X, sr, 0.25, 0.05, n_iter=200, tol=1e-6, backend='auto')
    assert called['rust']
