'''Public SPACE estimator.'''

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

from .penalties import alpha_to_penalties
from .solver import Backend, _JsrmWorkspace, get_rust_jsrm_solve, jsrm
from .utils import (
    beta_coef_from_rho_upper,
    inv_sig_diag_new,
    standardize_columns_l2,
)
from .weights import (
    WeightInput,
    WeightTag,
    rescale_degree_weights,
    resolve_weight,
)

AlphaCriterion = Literal['bic', 'aic']


class SPACE:
    '''
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
    backend : {'auto', 'numpy', 'rust'}
        Inner JSRM solve: ``numpy`` is pure NumPy; ``auto`` tries the Rust extension
        for the full inner solve, else NumPy; ``rust`` requires the compiled
        ``space_graph._rust`` extension.
    '''

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
        backend: Backend = 'auto',
    ):
        self.alpha = float(alpha)
        if self.alpha < 0.0:
            raise ValueError('alpha must be non-negative')
        self.gamma = float(gamma)
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError('gamma must be in [0, 1]')
        self.weight = weight
        self.max_outer_iter = int(max_outer_iter)
        if self.max_outer_iter <= 0:
            raise ValueError('max_outer_iter must be positive')
        self.max_inner_iter = int(max_inner_iter)
        if self.max_inner_iter <= 0:
            raise ValueError('max_inner_iter must be positive')
        self.tol = float(tol)
        if self.tol <= 0.0:
            raise ValueError('tol must be positive')
        self.standardize = standardize
        self.fit_sig = fit_sig
        self.sig_init = None if sig is None else np.asarray(sig, dtype=np.float64)
        if backend not in ('auto', 'numpy', 'rust'):
            raise ValueError(
                "backend must be 'auto', 'numpy', or 'rust'"
            )
        self.backend: Backend = backend

        self.partial_correlation_: Optional[np.ndarray] = None
        self.sig_: Optional[np.ndarray] = None
        self.weight_: Optional[np.ndarray] = None

    def _fit_loops(
        self,
        Xw: np.ndarray,
        p: int,
        lam1: float,
        lam2: float,
        *,
        init_jsrm_beta: Optional[np.ndarray] = None,
        workspace: Optional[_JsrmWorkspace] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w_vec, w_update, w_tag = resolve_weight(self.weight, p)

        sig_update = self.fit_sig
        if self.sig_init is not None:
            sig = np.asarray(self.sig_init, dtype=np.float64).ravel()
            if sig.shape[0] != p:
                raise ValueError('sig must have length p')
        else:
            sig = np.ones(p, dtype=np.float64)

        first_jsrm = True
        if w_tag == WeightTag.UNIFORM:
            Y_u = Xw
        elif w_tag == WeightTag.CUSTOM:
            Y_u = Xw * np.sqrt(w_vec)[None, :]

        for _ in range(self.max_outer_iter):
            if w_tag == WeightTag.SIG:
                w_vec = sig.copy()
                Y_u = Xw * np.sqrt(w_vec)[None, :]
            elif w_tag == WeightTag.DEGREE:
                Y_u = Xw * np.sqrt(w_vec)[None, :]

            sig_u = sig if w_tag == WeightTag.UNIFORM else sig / w_vec
            sigma_sr = np.sqrt(np.maximum(sig_u, 1e-15))
            beta0 = init_jsrm_beta if first_jsrm else None
            par_cor = jsrm(
                Y_u,
                sigma_sr,
                lam1,
                lam2,
                self.max_inner_iter,
                tol=self.tol,
                backend=self.backend,
                init_beta=beta0,
                workspace=workspace,
            )
            first_jsrm = False
            np.fill_diagonal(par_cor, 1.0)

            coef = par_cor[np.triu_indices(p, k=1)]
            beta_cur = beta_coef_from_rho_upper(coef, sig)

            if not w_update and not sig_update:
                break

            if sig_update:
                sig = inv_sig_diag_new(Xw, beta_cur, center=not self.standardize)

            if w_update:
                if w_tag == WeightTag.DEGREE:
                    w_vec = rescale_degree_weights(par_cor)

        return par_cor, sig, w_vec

    def fit(self, X: np.ndarray) -> 'SPACE':
        X = np.asarray(X, dtype=np.float64)
        lam1, lam2 = alpha_to_penalties(self.alpha, self.gamma)

        if self.standardize:
            Xw, _, _ = standardize_columns_l2(X)
        else:
            Xw = X

        n, p = Xw.shape
        fit_ws = None
        if self.backend in ('rust', 'auto') and get_rust_jsrm_solve() is not None:
            fit_ws = _JsrmWorkspace.for_shape(n, p)

        par_cor, sig, w_vec = self._fit_loops(
            Xw,
            p,
            lam1,
            lam2,
            init_jsrm_beta=None,
            workspace=fit_ws,
        )
        self.partial_correlation_ = par_cor
        self.sig_ = sig
        self.weight_ = w_vec
        return self

    def select_alpha(
        self,
        X: np.ndarray,
        alphas: np.ndarray,
        return_curve: bool = False,
        warm_start: bool = True,
        criterion: AlphaCriterion = 'bic',
    ) -> Union[float, tuple[float, np.ndarray]]:
        '''
        Information-criterion selection of ``alpha``.

        Fits SPACE at each candidate ``alpha`` (inheriting all other hyperparameters
        from ``self``) and scores
        ``IC(alpha) = sum_i [ n * log(RSS_i) + penalty * k_i ]``,
        where ``RSS_i`` is the residual sum of squares of the i-th regression and
        ``k_i = #{j != i : rho_ij != 0}`` (threshold ``self.tol``). For BIC
        (Peng et al. 2009, Sec. 2.4 / eq. 6), ``penalty = log(n)``; for AIC,
        ``penalty = 2``. Returns the ``alpha`` minimizing the requested criterion.
        Does not mutate ``self`` — pass the returned ``alpha`` into a fresh
        ``SPACE(...).fit(X)``.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Data. Standardized once up front (``standardize_columns_l2`` if
            ``self.standardize`` else column-centered) and reused across the
            grid; inner fits run with ``standardize=False`` on that matrix.
            The paper's BIC is defined on zero-mean regressions, so RSS is
            always evaluated on centered data regardless of ``self.standardize``.
        alphas : array-like of non-negative floats
            Candidate regularization strengths to score.
        return_curve : bool, default False
            If True, also return the per-alpha score vector aligned with ``alphas``.
        warm_start : bool, default True
            If True, fit alphas from high to low, reuse fixed-shape solver
            workspace, and use the previous partial-correlation matrix as the
            inner JSRM initializer. If False, each alpha is fit cold (matches
            independent fits per grid point).
        criterion : {'bic', 'aic'}, default 'bic'
            Information criterion to minimize. ``bic`` is more conservative;
            ``aic`` uses a smaller complexity penalty and tends to select denser
            graphs.

        Returns
        -------
        best_alpha : float
        score_curve : ndarray, shape (len(alphas),), optional
        '''
        alphas = np.asarray(alphas, dtype=np.float64).ravel()
        if alphas.size == 0:
            raise ValueError('alphas must be non-empty')
        if np.any(alphas < 0):
            raise ValueError('alphas must be non-negative')
        if criterion not in ('bic', 'aic'):
            raise ValueError("criterion must be 'bic' or 'aic'")

        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        log_eps = np.finfo(np.float64).tiny
        penalty = np.log(n) if criterion == 'bic' else 2.0

        if self.standardize:
            Xw, _, _ = standardize_columns_l2(X)
        else:
            Xw = X - X.mean(axis=0)

        score_curve = np.empty(alphas.size, dtype=np.float64)
        prev_jsrm: Optional[np.ndarray] = None
        js_workspace = _JsrmWorkspace.for_shape(n, p) if warm_start else None

        if warm_start:
            order = np.argsort(-alphas, kind='stable')
        else:
            order = np.arange(alphas.size)

        for step_idx in order:
            a = float(alphas[step_idx])
            lam1, lam2 = alpha_to_penalties(a, self.gamma)
            rho, sig, _ = self._fit_loops(
                Xw,
                p,
                lam1,
                lam2,
                init_jsrm_beta=prev_jsrm if warm_start else None,
                workspace=js_workspace if warm_start else None,
            )
            if warm_start:
                prev_jsrm = rho.copy()
                np.fill_diagonal(prev_jsrm, 0.0)

            coef = rho[np.triu_indices(p, k=1)]
            beta = beta_coef_from_rho_upper(coef, sig)
            np.fill_diagonal(beta, 0.0)

            residue = Xw - Xw @ beta
            rss = np.sum(residue * residue, axis=0)

            nz = np.abs(rho) > self.tol
            np.fill_diagonal(nz, False)
            k = nz.sum(axis=1)

            score_curve[step_idx] = float(
                n * np.sum(np.log(np.maximum(rss, log_eps)))
                + penalty * k.sum()
            )

        best = int(np.argmin(score_curve))
        best_alpha = float(alphas[best])
        if return_curve:
            return best_alpha, score_curve
        return best_alpha
