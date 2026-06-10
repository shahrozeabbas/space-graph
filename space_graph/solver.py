'''
JSRM inner solver: faithful port of `space/src/JSRM.c` active-shooting logic.

Y layout: `Y[k, j]` = sample k, variable j (same as C row-major `Y_m[k*p+j]`).

Performance: ``_ym_times_elementwise`` dispatches on ``beta`` density
(dense dgemm above ``MATMUL_DENSE_NNZ_FRACTION``, column GEMVs below);
column dots for ``Aij``/``Aji``; in-place residual updates.
'''

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, Literal, Optional

import numpy as np

Backend = Literal['auto', 'numpy', 'rust']

_DEFAULT_TOL = 1e-6
MATMUL_DENSE_NNZ_FRACTION = 0.2


@cache
def get_rust_jsrm_solve() -> Callable[..., np.ndarray] | None:
    '''Return Rust ``jsrm_solve`` if the compiled ``_rust`` module exists.'''
    try:
        from . import _rust

        return _rust.jsrm_solve
    except ImportError:
        return None


@dataclass
class _JsrmWorkspace:
    '''Preallocated buffers for repeated ``jsrm`` calls at fixed (n, p).'''

    n: int
    p: int
    Y_m: np.ndarray
    normx: np.ndarray
    B: np.ndarray
    B_sq: np.ndarray
    B_s: np.ndarray
    G: np.ndarray
    E_m: np.ndarray
    F_fit: np.ndarray
    beta_new: np.ndarray
    beta_old: np.ndarray
    ui: np.ndarray
    uj: np.ndarray
    i_ut: np.ndarray
    j_ut: np.ndarray
    temp1_ut: np.ndarray
    tt_ut: np.ndarray
    b_s_ij_ut: np.ndarray
    bet_ut: np.ndarray
    rust_ws: Any | None = None

    @classmethod
    def for_shape(cls, n: int, p: int) -> '_JsrmWorkspace':
        rust_ws = None
        if get_rust_jsrm_solve() is not None:
            try:
                from . import _rust

                rust_ws = _rust.JsrmWorkspace(int(n), int(p))
            except (ImportError, AttributeError):
                rust_ws = None
        Y_m = np.empty((n, p), dtype=np.float64, order='C')
        normx = np.empty(p, dtype=np.float64)
        B = np.empty((p, p), dtype=np.float64, order='C')
        B_sq = np.empty((p, p), dtype=np.float64, order='C')
        B_s = np.empty((p, p), dtype=np.float64, order='C')
        G = np.empty((p, p), dtype=np.float64, order='C')
        E_m = np.empty((n, p), dtype=np.float64, order='C')
        F_fit = np.empty((n, p), dtype=np.float64, order='C')
        beta_new = np.empty((p, p), dtype=np.float64, order='C')
        beta_old = np.empty((p, p), dtype=np.float64, order='C')
        ui, uj = np.triu_indices(p, k=1)
        i_ut, j_ut = _upper_tri_ij_jsrm_order(p)
        n_ut = ui.shape[0]
        temp1_ut = np.empty(n_ut, dtype=np.float64)
        tt_ut = np.empty(n_ut, dtype=np.float64)
        b_s_ij_ut = np.empty(n_ut, dtype=np.float64)
        bet_ut = np.empty(n_ut, dtype=np.float64)
        return cls(
            n=n,
            p=p,
            Y_m=Y_m,
            normx=normx,
            B=B,
            B_sq=B_sq,
            B_s=B_s,
            G=G,
            E_m=E_m,
            F_fit=F_fit,
            beta_new=beta_new,
            beta_old=beta_old,
            ui=ui,
            uj=uj,
            i_ut=i_ut,
            j_ut=j_ut,
            temp1_ut=temp1_ut,
            tt_ut=tt_ut,
            b_s_ij_ut=b_s_ij_ut,
            bet_ut=bet_ut,
            rust_ws=rust_ws,
        )


def _upper_tri_ij_jsrm_order(p: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Row/col indices for upper triangle (i < j) in the same order as ``JSRM.c``
    scans: ``j = p-1, ..., 1`` and for each ``j``, ``i = j-1, ..., 0``.
    '''
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for j in range(p - 1, 0, -1):
        rows.append(np.arange(j - 1, -1, -1, dtype=np.int32))
        cols.append(np.full(j, j, dtype=np.int32))
    if not rows:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.concatenate(rows), np.concatenate(cols)


def _elastic_net_shrink(
    beta_next: float,
    b_s: float,
    lambda1: float,
    lambda2: float,
) -> float:
    '''One coordinate elastic-net shrink (matches JSRM.c).'''
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


def _aij_aji(
    E_m: np.ndarray,
    Y_m: np.ndarray,
    cur_i: int,
    cur_j: int,
    B: np.ndarray,
) -> tuple[float, float]:
    '''``Aij``, ``Aji`` as in JSRM (BLAS dot on columns).'''
    aij = B[cur_i, cur_j] * float(np.dot(E_m[:, cur_j], Y_m[:, cur_i]))
    aji = B[cur_j, cur_i] * float(np.dot(E_m[:, cur_i], Y_m[:, cur_j]))
    return aij, aji


def _update_e_pair(
    E_m: np.ndarray,
    Y_m: np.ndarray,
    change_i: int,
    change_j: int,
    beta_change: float,
    B: np.ndarray,
) -> None:
    '''Residual update equation (11) in-place.'''
    c1 = beta_change * B[change_j, change_i]
    c2 = beta_change * B[change_i, change_j]
    if c1 != 0.0:
        E_m[:, change_i] += Y_m[:, change_j] * c1
    if c2 != 0.0:
        E_m[:, change_j] += Y_m[:, change_i] * c2


def _ym_times_elementwise(
    Y_m: np.ndarray, beta: np.ndarray, B: np.ndarray
) -> np.ndarray:
    '''
    Return ``Y_m @ (beta * B)``.

    Dispatches on ``beta`` density: dense ``dgemm`` when nonzero fraction
    exceeds ``MATMUL_DENSE_NNZ_FRACTION``, otherwise column-wise GEMVs that
    skip zero rows. Both produce the same value in exact arithmetic;
    summation order may differ by a few ULPs between paths.
    '''
    n, p = Y_m.shape
    if p == 0:
        return np.zeros((n, p), dtype=np.float64)
    if np.count_nonzero(beta) > MATMUL_DENSE_NNZ_FRACTION * beta.size:
        return Y_m @ (beta * B)
    F = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        nz = np.flatnonzero(beta[:, j])
        if nz.size:
            F[:, j] = Y_m[:, nz] @ (beta[nz, j] * B[nz, j])
    return F


def _ym_times_elementwise_into(
    Y_m: np.ndarray,
    beta: np.ndarray,
    B: np.ndarray,
    out: np.ndarray,
) -> None:
    '''Write ``Y_m @ (beta * B)`` into ``out`` (same dispatch as above).'''
    n, p = Y_m.shape
    if p == 0:
        return
    if np.count_nonzero(beta) > MATMUL_DENSE_NNZ_FRACTION * beta.size:
        np.matmul(Y_m, beta * B, out=out)
        return
    out.fill(0.0)
    for j in range(p):
        nz = np.flatnonzero(beta[:, j])
        if nz.size:
            out[:, j] = Y_m[:, nz] @ (beta[nz, j] * B[nz, j])


def _jsrm_return_beta(
    beta_new: np.ndarray,
    workspace: Optional[_JsrmWorkspace],
) -> np.ndarray:
    if workspace is None:
        return beta_new
    return beta_new.copy('C')


def _jsrm_prepare_y_b(
    Y_data: np.ndarray,
    sigma_sr: np.ndarray,
    workspace: Optional[_JsrmWorkspace],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if workspace is None:
        Y_m = Y_data.copy()
        Y_m -= Y_m.mean(axis=0)
        normx = np.sum(Y_m * Y_m, axis=0)
        B = sigma_sr[:, None] / sigma_sr[None, :]
        B_sq = B * B
        B_s = B_sq * normx[:, None] + B_sq.T * normx[None, :]
        return Y_m, normx, B, B_sq, B_s
    ws = workspace
    Y_m = ws.Y_m
    np.copyto(Y_m, Y_data)
    Y_m -= Y_m.mean(axis=0)
    np.sum(Y_m * Y_m, axis=0, out=ws.normx)
    normx = ws.normx
    B = ws.B
    np.divide(sigma_sr[:, None], sigma_sr[None, :], out=B)
    B_sq = ws.B_sq
    np.multiply(B, B, out=B_sq)
    B_s = ws.B_s
    B_s[:, :] = B_sq * normx[:, None] + B_sq.T * normx[None, :]
    return Y_m, normx, B, B_sq, B_s


def _jsrm_closed_form_beta(
    Y_m: np.ndarray,
    B: np.ndarray,
    B_s: np.ndarray,
    lambda1: float,
    lambda2: float,
    p: int,
    workspace: Optional[_JsrmWorkspace],
) -> np.ndarray:
    if workspace is None:
        G = Y_m.T @ Y_m
        ui, uj = np.triu_indices(p, k=1)
        temp1_vec = G[ui, uj] * (B[uj, ui] + B[ui, uj])
        tt = np.abs(temp1_vec) - lambda1
        b_s_ij = B_s[ui, uj] * (1.0 + lambda2)
        bet = np.zeros(ui.shape[0], dtype=np.float64)
        m = tt >= 0.0
        bet[m] = tt[m] / b_s_ij[m]
        bet[m] *= np.sign(temp1_vec[m])
        beta_new = np.zeros((p, p), dtype=np.float64)
        beta_new[ui, uj] = bet
        beta_new[uj, ui] = bet
        return beta_new
    ws = workspace
    ui, uj = ws.ui, ws.uj
    G = ws.G
    np.matmul(Y_m.T, Y_m, out=G)
    temp1 = ws.temp1_ut
    np.multiply(B[uj, ui] + B[ui, uj], G[ui, uj], out=temp1)
    tt = ws.tt_ut
    np.abs(temp1, out=tt)
    tt -= lambda1
    b_s_ij = ws.b_s_ij_ut
    np.multiply(B_s[ui, uj], 1.0 + lambda2, out=b_s_ij)
    bet = ws.bet_ut
    bet.fill(0.0)
    m = tt >= 0.0
    bet[m] = tt[m] / b_s_ij[m]
    bet[m] *= np.sign(temp1[m])
    beta_new = ws.beta_new
    beta_new.fill(0.0)
    beta_new[ui, uj] = bet
    beta_new[uj, ui] = bet
    return beta_new


def _jsrm_warm_start_beta(
    init_beta_arr: np.ndarray,
    workspace: Optional[_JsrmWorkspace],
) -> np.ndarray:
    if workspace is None:
        beta_new = 0.5 * (init_beta_arr + init_beta_arr.T)
    else:
        beta_new = workspace.beta_new
        np.add(init_beta_arr, init_beta_arr.T, out=beta_new)
        beta_new *= 0.5
    np.fill_diagonal(beta_new, 0.0)
    return beta_new


def _jsrm_init_residuals(
    Y_m: np.ndarray,
    beta_new: np.ndarray,
    B: np.ndarray,
    workspace: Optional[_JsrmWorkspace],
) -> tuple[np.ndarray, np.ndarray]:
    if workspace is None:
        F_fit = _ym_times_elementwise(Y_m, beta_new, B)
        E_m = Y_m - F_fit
        beta_old = beta_new.copy()
        return E_m, beta_old
    ws = workspace
    _ym_times_elementwise_into(Y_m, beta_new, B, ws.F_fit)
    E_m = ws.E_m
    np.subtract(Y_m, ws.F_fit, out=E_m)
    beta_old = ws.beta_old
    np.copyto(beta_old, beta_new)
    return E_m, beta_old


def jsrm(
    Y_data: np.ndarray,
    sigma_sr: np.ndarray,
    lam1: float,
    lam2: float,
    n_iter: int = 500,
    tol: float = _DEFAULT_TOL,
    backend: Backend = 'auto',
    init_beta: Optional[np.ndarray] = None,
    workspace: Optional[_JsrmWorkspace] = None,
) -> np.ndarray:
    '''
    Joint sparse regression model (SPACE inner problem).

    Parameters
    ----------
    Y_data : ndarray, shape (n, p)
        Data; columns centered to mean 0 inside (C behavior).
    sigma_sr : ndarray, shape (p,)
        sqrt(sig^{ii}) per variable (R `sig.use^0.5`).
    lam1, lam2 : float
        L1 and elastic-net L2 penalties.
    n_iter : int
        Max inner iterations (R `jsrm` uses 500).
    tol : float
        Convergence tolerance: stop when max coordinate change between sweeps
        is below ``tol`` (also used as the active-set threshold for nonzero
        ``beta``, matching the reference ``1e-6`` scale).
    backend : {'auto', 'numpy', 'rust'}
        Inner solve: ``numpy`` uses pure NumPy; ``auto`` tries the Rust extension
        for the full inner solve, then NumPy; ``rust`` requires the compiled
        ``space_graph._rust`` extension.
    init_beta : ndarray of shape (p, p) or None
        Optional warm start: symmetric zero-diagonal regression coefficients.
        When ``None``, uses the default closed-form initialization (current
        behavior). When set, ``Y_data``, ``sigma_sr``, and penalties must match
        the scaling implied by this beta.
    workspace : _JsrmWorkspace or None
        Optional reused buffers for fixed ``(n, p)``. When set, returns a copy
        of ``beta`` so later workspace reuse does not alias user arrays.

    Returns
    -------
    beta_new : ndarray, shape (p, p)
        Symmetric estimates; diagonal 0.
    '''
    Y_data = np.asarray(Y_data, dtype=np.float64, order='C')
    sigma_sr = np.asarray(sigma_sr, dtype=np.float64).ravel()
    n, p = Y_data.shape
    if sigma_sr.shape[0] != p:
        raise ValueError('sigma_sr must have length p')
    if workspace is not None:
        if workspace.n != n or workspace.p != p:
            raise ValueError('workspace shape mismatch')

    lambda1 = float(lam1)
    lambda2 = float(lam2)
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError('tol must be positive')
    if backend not in ('auto', 'numpy', 'rust'):
        raise ValueError("backend must be 'auto', 'numpy', or 'rust'")
    eps1 = tol
    maxdif_tol = tol

    init_beta_arr: Optional[np.ndarray] = None
    if init_beta is not None:
        b0 = np.asarray(init_beta, dtype=np.float64, order='C')
        if b0.shape != (p, p):
            raise ValueError('init_beta must have shape (p, p)')
        init_beta_arr = b0

    rust_solve: Optional[Callable[..., np.ndarray]] = None
    use_rust = False

    if backend == 'numpy':
        pass
    elif backend == 'rust':
        rust_solve = get_rust_jsrm_solve()
        if rust_solve is None:
            raise ImportError(
                "backend='rust' requires the space_graph._rust extension; "
                "install a Rust-built wheel or run maturin develop"
            )
        use_rust = True
    elif backend == 'auto':
        rust_solve = get_rust_jsrm_solve()
        if rust_solve is not None:
            use_rust = True

    if use_rust:
        assert rust_solve is not None
        rw = workspace.rust_ws if workspace is not None else None
        if rw is not None:
            out = rw.solve(
                Y_data,
                sigma_sr,
                lambda1,
                lambda2,
                n_iter,
                tol,
                init_beta_arr,
            )
        else:
            out = rust_solve(
                Y_data,
                sigma_sr,
                lambda1,
                lambda2,
                n_iter,
                tol,
                init_beta_arr,
            )
        return _jsrm_return_beta(
            np.asarray(out, dtype=np.float64, order='C'), workspace
        )

    Y_m, _normx, B, _B_sq, B_s = _jsrm_prepare_y_b(Y_data, sigma_sr, workspace)

    if init_beta_arr is None:
        beta_new = _jsrm_closed_form_beta(
            Y_m, B, B_s, lambda1, lambda2, p, workspace
        )
    else:
        beta_new = _jsrm_warm_start_beta(init_beta_arr, workspace)

    E_m, beta_old = _jsrm_init_residuals(Y_m, beta_new, B, workspace)

    i_ut, j_ut = (
        (workspace.i_ut, workspace.j_ut)
        if workspace is not None
        else _upper_tri_ij_jsrm_order(p)
    )
    if i_ut.size == 0:
        return _jsrm_return_beta(beta_new, workspace)
    vals_ut = beta_new[i_ut, j_ut]
    first_act = np.flatnonzero((vals_ut > eps1) | (vals_ut < -eps1))
    if first_act.size == 0:
        return _jsrm_return_beta(beta_new, workspace)

    cur_i = int(i_ut[first_act[0]])
    cur_j = int(j_ut[first_act[0]])

    aij, aji = _aij_aji(E_m, Y_m, cur_i, cur_j, B)
    b_s = B_s[cur_i, cur_j]
    beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
    temp = _elastic_net_shrink(beta_next, b_s, lambda1, lambda2)

    beta_change = beta_old[cur_i, cur_j] - temp
    beta_new[cur_i, cur_j] = temp
    beta_new[cur_j, cur_i] = temp

    change_i = cur_i
    change_j = cur_j

    for _ in range(n_iter):
        vals_ut = beta_new[i_ut, j_ut]
        act = (vals_ut > eps1) | (vals_ut < -eps1)
        nrow_pick = int(np.count_nonzero(act))
        maxdif = -100.0

        if nrow_pick > 0:
            pi = i_ut[act]
            pj = j_ut[act]
            max_delta = 0.0
            for t in range(nrow_pick):
                cur_i = int(pi[t])
                cur_j = int(pj[t])
                beta_old[change_i, change_j] = beta_new[change_i, change_j]
                beta_old[change_j, change_i] = beta_new[change_j, change_i]

                _update_e_pair(E_m, Y_m, change_i, change_j, beta_change, B)

                aij, aji = _aij_aji(E_m, Y_m, cur_i, cur_j, B)
                b_s = B_s[cur_i, cur_j]
                beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
                temp = _elastic_net_shrink(beta_next, b_s, lambda1, lambda2)

                beta_new[cur_i, cur_j] = temp
                beta_new[cur_j, cur_i] = temp

                beta_change = beta_old[cur_i, cur_j] - temp
                d = abs(beta_change)
                if d > max_delta:
                    max_delta = d
                change_i = cur_i
                change_j = cur_j

            maxdif = max_delta

        if maxdif < maxdif_tol or nrow_pick < 1:
            max_delta = 0.0
            for cur_i in range(p - 1):
                for cur_j in range(cur_i + 1, p):
                    beta_old[change_i, change_j] = beta_new[
                        change_i, change_j
                    ]
                    beta_old[change_j, change_i] = beta_new[
                        change_j, change_i
                    ]

                    if beta_change < -eps1 or beta_change > eps1:
                        _update_e_pair(
                            E_m,
                            Y_m,
                            change_i,
                            change_j,
                            beta_change,
                            B,
                        )

                    aij, aji = _aij_aji(E_m, Y_m, cur_i, cur_j, B)
                    b_s = B_s[cur_i, cur_j]
                    beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
                    temp = _elastic_net_shrink(beta_next, b_s, lambda1, lambda2)

                    beta_new[cur_i, cur_j] = temp
                    beta_new[cur_j, cur_i] = temp

                    beta_change = beta_old[cur_i, cur_j] - temp
                    d = abs(beta_change)
                    if d > max_delta:
                        max_delta = d
                    change_i = cur_i
                    change_j = cur_j

            maxdif = max_delta

            if maxdif < maxdif_tol:
                break

    return _jsrm_return_beta(beta_new, workspace)
