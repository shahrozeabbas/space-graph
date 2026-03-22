"""
Numba-compiled JSRM shooting loop (optional).

Install with ``pip install 'space-graph[numba]'`` or ``pip install numba``.
If Numba is not installed, ``jsrm_shooting_loop`` is ``None`` and ``solver.jsrm``
uses the pure NumPy loop.
"""

from __future__ import annotations

import numpy as np


def _try_build_shooting_loop():
    try:
        from numba import njit
    except ImportError:
        return None

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
                    if b > eps1 or b < -eps1:
                        nrow_pick += 1

            maxdif = -100.0

            if nrow_pick > 0:
                for j in range(p - 1, 0, -1):
                    for i in range(j - 1, -1, -1):
                        cur_i = i
                        cur_j = j
                        b = beta_new[cur_i, cur_j]
                        if not (b > eps1 or b < -eps1):
                            continue

                        beta_old[change_i, change_j] = beta_new[
                            change_i, change_j
                        ]
                        beta_old[change_j, change_i] = beta_new[
                            change_j, change_i
                        ]

                        c1 = beta_change * B[change_j, change_i]
                        c2 = beta_change * B[change_i, change_j]
                        for kk in range(n):
                            E_m[kk, change_i] += Y_m[kk, change_j] * c1
                            E_m[kk, change_j] += Y_m[kk, change_i] * c2

                        aij = 0.0
                        aji = 0.0
                        for kk in range(n):
                            aij += E_m[kk, cur_j] * Y_m[kk, cur_i]
                            aji += E_m[kk, cur_i] * Y_m[kk, cur_j]
                        aij *= B[cur_i, cur_j]
                        aji *= B[cur_j, cur_i]

                        b_s = B_s[cur_i, cur_j]
                        beta_next = (aij + aji) / b_s + beta_old[cur_i, cur_j]
                        temp1 = beta_next
                        if beta_next > 0.0:
                            temp = beta_next - lambda1 / b_s
                        else:
                            temp = -beta_next - lambda1 / b_s
                        if temp < 0.0:
                            temp = 0.0
                        else:
                            temp = temp / (1.0 + lambda2)
                            if temp1 < 0.0:
                                temp = -temp

                        beta_new[cur_i, cur_j] = temp
                        beta_new[cur_j, cur_i] = temp

                        beta_change = beta_old[cur_i, cur_j] - temp
                        change_i = cur_i
                        change_j = cur_j

                maxdif = -100.0
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
                        beta_old[change_i, change_j] = beta_new[
                            change_i, change_j
                        ]
                        beta_old[change_j, change_i] = beta_new[
                            change_j, change_i
                        ]

                        if beta_change < -eps1 or beta_change > eps1:
                            c1 = beta_change * B[change_j, change_i]
                            c2 = beta_change * B[change_i, change_j]
                            for kk in range(n):
                                E_m[kk, change_i] += Y_m[kk, change_j] * c1
                                E_m[kk, change_j] += Y_m[kk, change_i] * c2

                        aij = 0.0
                        aji = 0.0
                        for kk in range(n):
                            aij += E_m[kk, cur_j] * Y_m[kk, cur_i]
                            aji += E_m[kk, cur_i] * Y_m[kk, cur_j]
                        aij *= B[cur_i, cur_j]
                        aji *= B[cur_j, cur_i]

                        b_s = B_s[cur_i, cur_j]
                        beta_next = (aij + aji) / b_s + beta_old[
                            cur_i, cur_j
                        ]
                        temp1 = beta_next
                        if beta_next > 0.0:
                            temp = beta_next - lambda1 / b_s
                        else:
                            temp = -beta_next - lambda1 / b_s
                        if temp < 0.0:
                            temp = 0.0
                        else:
                            temp = temp / (1.0 + lambda2)
                            if temp1 < 0.0:
                                temp = -temp

                        beta_new[cur_i, cur_j] = temp
                        beta_new[cur_j, cur_i] = temp

                        beta_change = beta_old[cur_i, cur_j] - temp
                        change_i = cur_i
                        change_j = cur_j

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


jsrm_shooting_loop = _try_build_shooting_loop()
