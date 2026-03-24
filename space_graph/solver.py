"""
JSRM inner solver: faithful port of `space/src/JSRM.c` active-shooting logic.

Y layout: `Y[k, j]` = sample k, variable j (same as C row-major `Y_m[k*p+j]`).

Performance: fitted values ``Y_m @ (beta * B)`` without materializing full ``W``
(column GEMVs); column dots for ``Aij``/``Aji``; in-place residual updates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .kernels import get_jsrm_shooting_loop

Backend = Literal['auto', 'numpy', 'numba']

_DEFAULT_TOL = 1e-6


def _upper_tri_ij_jsrm_order(p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Row/col indices for upper triangle (i < j) in the same order as ``JSRM.c``
    scans: ``j = p-1, ..., 1`` and for each ``j``, ``i = j-1, ..., 0``.
    """
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
    """One coordinate elastic-net shrink (matches JSRM.c)."""
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
    """``Aij``, ``Aji`` as in JSRM (BLAS dot on columns)."""
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
    """Residual update equation (11) in-place."""
    c1 = beta_change * B[change_j, change_i]
    c2 = beta_change * B[change_i, change_j]
    if c1 != 0.0:
        E_m[:, change_i] += Y_m[:, change_j] * c1
    if c2 != 0.0:
        E_m[:, change_j] += Y_m[:, change_i] * c2


def _ym_times_elementwise(
    Y_m: np.ndarray, beta: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    Return ``Y_m @ (beta * B)`` without allocating the full ``p * p`` product matrix.

    Per column ``j``, only rows ``k`` with ``beta[k, j] != 0`` contribute (same as
    dense ``Y_m @ W`` with ``W = beta * B`` in exact arithmetic).
    """
    n, p = Y_m.shape
    F = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        nz = np.flatnonzero(beta[:, j])
        if nz.size:
            F[:, j] = Y_m[:, nz] @ (beta[nz, j] * B[nz, j])
    return F


def jsrm(
    Y_data: np.ndarray,
    sigma_sr: np.ndarray,
    lam1: float,
    lam2: float,
    n_iter: int = 500,
    tol: float = _DEFAULT_TOL,
    backend: Backend = 'auto',
) -> np.ndarray:
    """
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
    backend : {'auto', 'numpy', 'numba'}
        Inner shooting loop: ``numpy`` always uses pure NumPy; ``auto`` uses
        Numba when installed (lazy on first call), else NumPy; ``numba``
        requires Numba and raises ``ImportError`` if the kernel cannot be built.

    Returns
    -------
    beta_new : ndarray, shape (p, p)
        Symmetric estimates; diagonal 0.
    """
    Y_data = np.asarray(Y_data, dtype=np.float64, order='C')
    sigma_sr = np.asarray(sigma_sr, dtype=np.float64).ravel()
    n, p = Y_data.shape
    if sigma_sr.shape[0] != p:
        raise ValueError('sigma_sr must have length p')

    lambda1 = float(lam1)
    lambda2 = float(lam2)
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError('tol must be positive')
    if backend not in ('auto', 'numpy', 'numba'):
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
    eps1 = tol
    maxdif_tol = tol

    Y_m = Y_data.copy()
    Y_m -= Y_m.mean(axis=0)
    normx = np.sum(Y_m * Y_m, axis=0)

    B = sigma_sr[:, None] / sigma_sr[None, :]
    B_sq = B * B
    B_s = B_sq * normx[:, None] + B_sq.T * normx[None, :]

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
    np.fill_diagonal(beta_new, 0.0)

    F_fit = _ym_times_elementwise(Y_m, beta_new, B)
    E_m = Y_m - F_fit

    beta_old = beta_new.copy()
    beta_last = np.empty((p, p), dtype=np.float64)

    i_ut, j_ut = _upper_tri_ij_jsrm_order(p)
    if i_ut.size == 0:
        return beta_new
    vals_ut = beta_new[i_ut, j_ut]
    first_act = np.flatnonzero((vals_ut > eps1) | (vals_ut < -eps1))
    if first_act.size == 0:
        return beta_new

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

    use_numba = False
    if backend == 'numpy':
        loop = None
    elif backend == 'auto':
        loop = get_jsrm_shooting_loop()
        use_numba = loop is not None
    else:
        loop = get_jsrm_shooting_loop()
        if loop is None:
            raise ImportError(
                "backend='numba' requires numba; install with "
                "'pip install space-graph[numba]' or 'pip install numba'"
            )
        use_numba = True

    if use_numba:
        assert loop is not None
        loop(
            Y_m,
            E_m,
            beta_new,
            beta_old,
            beta_last,
            B,
            B_s,
            lambda1,
            lambda2,
            n,
            p,
            n_iter,
            change_i,
            change_j,
            beta_change,
            tol,
        )
        return beta_new

    for _ in range(n_iter):
        beta_last[:] = beta_new

        vals_ut = beta_new[i_ut, j_ut]
        act = (vals_ut > eps1) | (vals_ut < -eps1)
        nrow_pick = int(np.count_nonzero(act))
        maxdif = -100.0

        if nrow_pick > 0:
            pi = i_ut[act]
            pj = j_ut[act]
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
                change_i = cur_i
                change_j = cur_j

            maxdif = float(np.max(np.abs(beta_last - beta_new)))

        if maxdif < maxdif_tol or nrow_pick < 1:
            beta_last[:] = beta_new

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
                    change_i = cur_i
                    change_j = cur_j

            maxdif = float(np.max(np.abs(beta_last - beta_new)))

            if maxdif < maxdif_tol:
                break

    return beta_new
