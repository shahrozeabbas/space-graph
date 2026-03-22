"""Standardization and matrix transforms matching `space` R code."""

from __future__ import annotations

import numpy as np


def standardize_columns_l2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center each column to mean 0 and scale to L2 norm 1 (per column).
    Returns (X_std, mean, scale) where X_std = (X - mean) / scale.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    scale = np.sqrt(np.sum(Xc**2, axis=0))
    scale = np.where(scale < 1e-15, 1.0, scale)
    return Xc / scale, mean, scale


def beta_coef_from_rho_upper(coef: np.ndarray, sig_fit: np.ndarray) -> np.ndarray:
    """
    `Beta.coef` from space/R/space.R: coef is upper-triangle rho^{ij}, sig_fit is sigma^{ii}.
    """
    p = sig_fit.shape[0]
    result = np.zeros((p, p), dtype=np.float64)
    result[np.triu_indices(p, k=1)] = coef
    result = result + result.T
    inv_sqrt = 1.0 / np.sqrt(sig_fit)
    sqrt_sig = np.sqrt(sig_fit)
    result = (inv_sqrt[:, None] * result) @ np.diag(sqrt_sig)
    return result.T


def inv_sig_diag_new(Y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    `InvSig.diag.new`: 1 / colMeans((Y - Y @ Beta0)^2) with diag(Beta)=0.
    """
    b = beta.copy()
    np.fill_diagonal(b, 0.0)
    esti = Y @ b
    residue = Y - esti
    return 1.0 / np.mean(residue**2, axis=0)


def partial_corr_to_precision(
    parcor: np.ndarray, sig: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a precision-like matrix from partial correlations and sig^{ii},
    following NITK/R style: Theta_ij related to rho_ij via scaling by sig.
    """
    p = parcor.shape[0]
    ind = np.triu_indices(p, k=1)
    coef = parcor[ind]
    result = np.zeros((p, p), dtype=np.float64)
    result[ind] = coef
    result = result + result.T
    inv_sqrt = 1.0 / np.sqrt(sig)
    sqrt_sig = np.sqrt(sig)
    result = inv_sqrt[:, None] * result * sqrt_sig[None, :]
    return result.T
