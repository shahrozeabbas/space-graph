"""
JSRM inner solver: faithful port of `space/src/JSRM.c` active-shooting logic.

Y layout: `Y[k, j]` = sample k, variable j (same as C row-major `Y_m[k*p+j]`).

Performance: vectorized BLAS-friendly ops (``Y @ W`` for fitted values, column
dot products for ``Aij``/``Aji``, in-place column updates for residuals).
"""

from __future__ import annotations

import numpy as np

from .kernels import jsrm_shooting_loop

_DEFAULT_TOL = 1e-6


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


def jsrm(
    Y_data: np.ndarray,
    sigma_sr: np.ndarray,
    lam1: float,
    lam2: float,
    n_iter: int = 500,
    tol: float = _DEFAULT_TOL,
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

    W = beta_new * B
    E_m = Y_m - (Y_m @ W)

    beta_old = beta_new.copy()
    beta_last = np.empty((p, p), dtype=np.float64)

    found = False
    pick_i = pick_j = 0
    for j in range(p - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            b = beta_new[i, j]
            if b > eps1 or b < -eps1:
                pick_i, pick_j = i, j
                found = True
                break
        if found:
            break

    if not found:
        return beta_new

    cur_i, cur_j = pick_i, pick_j

    aij, aji = _aij_aji(E_m, Y_m, cur_i, cur_j, B)
    b_s = B_s[cur_i, cur_j]
    beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
    temp = _elastic_net_shrink(beta_next, b_s, lambda1, lambda2)

    beta_change = beta_old[cur_i, cur_j] - temp
    beta_new[cur_i, cur_j] = temp
    beta_new[cur_j, cur_i] = temp

    change_i = cur_i
    change_j = cur_j

    if jsrm_shooting_loop is not None:
        jsrm_shooting_loop(
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

    nbeta = p * (p - 1) // 2
    pair_buf = np.empty((nbeta, 2), dtype=np.int32)

    for _ in range(n_iter):
        beta_last[:] = beta_new

        k = 0
        for j in range(p - 1, 0, -1):
            for i in range(j - 1, -1, -1):
                b = beta_new[i, j]
                if b > eps1 or b < -eps1:
                    pair_buf[k, 0] = i
                    pair_buf[k, 1] = j
                    k += 1
        nrow_pick = k
        maxdif = -100.0

        if nrow_pick > 0:
            for t in range(nrow_pick):
                cur_i = int(pair_buf[t, 0])
                cur_j = int(pair_buf[t, 1])
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
