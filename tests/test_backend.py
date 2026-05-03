'''Tests for JSRM / SPACE ``backend`` (numpy, rust, auto).'''

from __future__ import annotations

import numpy as np
import pytest

import space_graph.solver as solver_mod
from space_graph.model import SPACE


def _tiny_X(rng: np.random.Generator) -> np.ndarray:
    p, n = 5, 30
    a = rng.standard_normal((p, p))
    cov = a @ a.T + p * np.eye(p)
    return rng.multivariate_normal(np.zeros(p), cov, size=n)


def test_backend_numpy_never_calls_getter(monkeypatch):
    def _fail():
        pytest.fail('Rust getter should not be called for backend=numpy')

    monkeypatch.setattr(solver_mod, 'get_rust_jsrm_solve', _fail)
    rng = np.random.default_rng(7)
    X = _tiny_X(rng)
    m = SPACE(alpha=0.9, max_outer_iter=2, backend='numpy')
    m.fit(X)
    assert m.partial_correlation_.shape == (5, 5)


def test_backend_auto_falls_back_to_numpy_when_rust_unavailable(monkeypatch):
    monkeypatch.setattr(solver_mod, 'get_rust_jsrm_solve', lambda: None)
    rng = np.random.default_rng(8)
    X = _tiny_X(rng)
    m_auto = SPACE(alpha=0.2, max_outer_iter=2, backend='auto')
    m_np = SPACE(alpha=0.2, max_outer_iter=2, backend='numpy')
    m_auto.fit(X)
    m_np.fit(X)
    np.testing.assert_allclose(
        m_auto.partial_correlation_,
        m_np.partial_correlation_,
        atol=1e-12,
    )


def test_jsrm_invalid_backend_raises():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((20, 4))
    sig = np.ones(4)
    with pytest.raises(ValueError, match='backend must be'):
        solver_mod.jsrm(X, sig, 0.5, 0.0, **{'backend': 'cuda'})


def test_space_invalid_backend_raises():
    with pytest.raises(ValueError, match='backend must be'):
        SPACE(**{'backend': 'fortran'})
