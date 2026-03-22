"""Public SPACE estimator."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .penalties import alpha_to_penalties
from .solver import jsrm
from .utils import (
    beta_coef_from_rho_upper,
    inv_sig_diag_new,
    partial_corr_to_precision,
    standardize_columns_l2,
)
from .weights import WeightInput, rescale_degree_weights, resolve_weight


class SPACE:
    """
    Sparse partial correlation estimation (SPACE), joint sparse regression.

    Parameters
    ----------
    alpha : float >= 0
        Regularization strength (sklearn-style).
    gamma : float in [0, 1]
        Mix γ between L1-like and L2-like terms: ``lam1 = alpha * gamma``,
        ``lam2 = alpha * (1 - gamma)``. Default ``1`` matches R ``space::space.joint``
        default ``lam2 = 0`` (pure L1 scaling of ``lam1`` at strength ``alpha``).
    weight : {'uniform', 'equal', 'sig', 'degree'} or ndarray of shape (p,)
        Node weights for the joint loss (see Peng et al. and R package).
        ``uniform`` and ``equal`` both mean unit weights (no reweighting).
    max_outer_iter : int
        Outer alternations for ``sig`` / weights (R ``iter``).
    max_inner_iter : int
        Max iterations for the inner JSRM solver.
    tol : float
        Inner solver tolerance: convergence and active-set threshold (default
        ``1e-6``, same scale as the reference C implementation).
    standardize : bool
        If True, center columns and scale to unit L2 norm before fitting.
    fit_sig : bool
        If True, estimate diagonal ``sig^{ii}`` each outer step (when not fixed).
    sig : ndarray of shape (p,) or None
        Initial or fixed ``sig^{ii}``. If provided and ``fit_sig`` is False, held fixed.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        weight: WeightInput = 'uniform',
        max_outer_iter: int = 5,
        max_inner_iter: int = 1000,
        tol: float = 1e-6,
        standardize: bool = True,
        fit_sig: bool = True,
        sig: Optional[np.ndarray] = None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError('gamma must be in [0, 1]')
        self.weight = weight
        self.max_outer_iter = int(max_outer_iter)
        self.max_inner_iter = int(max_inner_iter)
        self.tol = float(tol)
        self.standardize = standardize
        self.fit_sig = fit_sig
        self.sig_init = None if sig is None else np.asarray(sig, dtype=np.float64)
        if self.tol <= 0.0:
            raise ValueError('tol must be positive')

        self.partial_correlation_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.sig_: Optional[np.ndarray] = None
        self.weight_: Optional[np.ndarray] = None
        self._mean_: Optional[np.ndarray] = None
        self._scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'SPACE':
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        lam1, lam2 = alpha_to_penalties(self.alpha, self.gamma)

        if self.standardize:
            Xw, self._mean_, self._scale_ = standardize_columns_l2(X)
        else:
            Xw = X.copy()
            self._mean_ = np.zeros(p)
            self._scale_ = np.ones(p)

        w_vec, w_update, w_tag = resolve_weight(self.weight, p)

        if self.sig_init is not None:
            sig = np.asarray(self.sig_init, dtype=np.float64).ravel()
            if sig.shape[0] != p:
                raise ValueError('sig must have length p')
            sig_update = self.fit_sig
        else:
            sig = np.ones(p, dtype=np.float64)
            sig_update = self.fit_sig

        for _ in range(self.max_outer_iter):
            if w_tag == 1:
                w_vec = sig.copy()
            Y_u = Xw * np.sqrt(w_vec)[None, :]
            sig_u = sig / w_vec

            sigma_sr = np.sqrt(np.maximum(sig_u, 1e-15))
            par_cor = jsrm(
                Y_u,
                sigma_sr,
                lam1,
                lam2,
                self.max_inner_iter,
                tol=self.tol,
            )
            np.fill_diagonal(par_cor, 1.0)

            coef = par_cor[np.triu_indices(p, k=1)]
            beta_cur = beta_coef_from_rho_upper(coef, sig)

            if not w_update and not sig_update:
                break

            if sig_update:
                sig = inv_sig_diag_new(Xw, beta_cur)

            if w_update:
                if w_tag == 2:
                    w_vec = rescale_degree_weights(par_cor)

        self.partial_correlation_ = par_cor
        self.sig_ = sig
        self.weight_ = w_vec
        self.precision_ = partial_corr_to_precision(par_cor, sig)
        return self
