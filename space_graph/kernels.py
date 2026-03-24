"""
Numba-compiled JSRM shooting loop (optional).

Install with ``pip install 'space-graph[numba]'`` or ``pip install numba``.
Numba is loaded only when :func:`get_jsrm_shooting_loop` runs (``backend`` ``auto``
or ``numba`` in ``solver.jsrm``). If unavailable, that function returns ``None`` and
the pure NumPy loop is used (for ``auto``) or ``solver`` raises (for ``numba``).

Legacy attribute ``jsrm_shooting_loop`` resolves via module ``__getattr__`` to the
same value as ``get_jsrm_shooting_loop()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_NOT_TRIED = object()
_cached: Any = _NOT_TRIED


def get_jsrm_shooting_loop() -> Callable[..., None] | None:
    """Return the compiled shooting kernel, or ``None`` if Numba is unavailable."""
    global _cached
    if _cached is _NOT_TRIED:
        _cached = _try_build_shooting_loop()
    return _cached


def __getattr__(name: str) -> Any:
    if name == 'jsrm_shooting_loop':
        return get_jsrm_shooting_loop()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _try_build_shooting_loop():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True, inline='always')
    def _jsrm_apply_residual(
        E_m: np.ndarray,
        Y_m: np.ndarray,
        change_i: int,
        change_j: int,
        beta_change: float,
        B: np.ndarray,
        n: int,
    ) -> None:
        c1 = beta_change * B[change_j, change_i]
        c2 = beta_change * B[change_i, change_j]
        for kk in range(n):
            E_m[kk, change_i] += Y_m[kk, change_j] * c1
            E_m[kk, change_j] += Y_m[kk, change_i] * c2

    @njit(cache=True, inline='always')
    def _jsrm_elastic_shrink(
        beta_next: float,
        b_s: float,
        lambda1: float,
        lambda2: float,
    ) -> float:
        temp1 = beta_next
        if beta_next > 0.0:
            temp = beta_next - lambda1 / b_s
        else:
            temp = -beta_next - lambda1 / b_s
        if temp < 0.0:
            return 0.0
        temp = temp / (1.0 + lambda2)
        if temp1 < 0.0:
            temp = -temp
        return temp

    @njit(cache=True, inline='always')
    def _jsrm_one_step(
        Y_m: np.ndarray,
        E_m: np.ndarray,
        beta_new: np.ndarray,
        beta_old: np.ndarray,
        B: np.ndarray,
        B_s: np.ndarray,
        cur_i: int,
        cur_j: int,
        change_i: int,
        change_j: int,
        beta_change: float,
        lambda1: float,
        lambda2: float,
        n: int,
        eps1: float,
        gate_residual: bool,
    ) -> tuple[float, int, int]:
        """
        One coordinate update (eq. 11–12). If ``gate_residual``, apply residual
        update only when ``|beta_change| > eps1`` (full sweep); else always
        (active sweep), matching ``JSRM.c`` / ``solver`` NumPy loop.
        """
        beta_old[change_i, change_j] = beta_new[change_i, change_j]
        beta_old[change_j, change_i] = beta_new[change_j, change_i]

        if (not gate_residual) or (beta_change < -eps1 or beta_change > eps1):
            _jsrm_apply_residual(
                E_m, Y_m, change_i, change_j, beta_change, B, n
            )

        aij = 0.0
        aji = 0.0
        for kk in range(n):
            aij += E_m[kk, cur_j] * Y_m[kk, cur_i]
            aji += E_m[kk, cur_i] * Y_m[kk, cur_j]
        aij *= B[cur_i, cur_j]
        aji *= B[cur_j, cur_i]

        b_s = B_s[cur_i, cur_j]
        beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
        temp = _jsrm_elastic_shrink(beta_next, b_s, lambda1, lambda2)

        beta_new[cur_i, cur_j] = temp
        beta_new[cur_j, cur_i] = temp

        new_change = beta_old[cur_i, cur_j] - temp
        return new_change, cur_i, cur_j

    @njit(cache=True)
    def jsrm_shooting_loop(
        Y_m: np.ndarray,
        E_m: np.ndarray,
        beta_new: np.ndarray,
        beta_old: np.ndarray,
        beta_last: np.ndarray,
        B: np.ndarray,
        B_s: np.ndarray,
        lambda1: float,
        lambda2: float,
        n: int,
        p: int,
        n_iter: int,
        change_i: int,
        change_j: int,
        beta_change: float,
        tol: float,
    ) -> None:
        """In-place shooting iterations (matches ``solver`` Python loop)."""
        eps1 = tol
        maxdif_tol = tol
        for _ in range(n_iter):
            for ii in range(p):
                for jj in range(p):
                    beta_last[ii, jj] = beta_new[ii, jj]

            nrow_pick = 0
            for j in range(p - 1, 0, -1):
                for i in range(j - 1, -1, -1):
                    b = beta_new[i, j]
                    if not (b > eps1 or b < -eps1):
                        continue
                    nrow_pick += 1
                    beta_change, change_i, change_j = _jsrm_one_step(
                        Y_m,
                        E_m,
                        beta_new,
                        beta_old,
                        B,
                        B_s,
                        i,
                        j,
                        change_i,
                        change_j,
                        beta_change,
                        lambda1,
                        lambda2,
                        n,
                        eps1,
                        False,
                    )

            maxdif = -100.0
            if nrow_pick > 0:
                for ii in range(p):
                    for jj in range(p):
                        d = beta_last[ii, jj] - beta_new[ii, jj]
                        if d < 0.0:
                            d = -d
                        if d > maxdif:
                            maxdif = d

            if maxdif < maxdif_tol or nrow_pick < 1:
                for ii in range(p):
                    for jj in range(p):
                        beta_last[ii, jj] = beta_new[ii, jj]

                for cur_i in range(p - 1):
                    for cur_j in range(cur_i + 1, p):
                        beta_change, change_i, change_j = _jsrm_one_step(
                            Y_m,
                            E_m,
                            beta_new,
                            beta_old,
                            B,
                            B_s,
                            cur_i,
                            cur_j,
                            change_i,
                            change_j,
                            beta_change,
                            lambda1,
                            lambda2,
                            n,
                            eps1,
                            True,
                        )

                maxdif = -100.0
                for ii in range(p):
                    for jj in range(p):
                        d = beta_last[ii, jj] - beta_new[ii, jj]
                        if d < 0.0:
                            d = -d
                        if d > maxdif:
                            maxdif = d

                if maxdif < maxdif_tol:
                    return

    return jsrm_shooting_loop
