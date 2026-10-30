"""Small numerical helpers shared by the models: finite-difference Hessians,
Wald intervals, information criteria."""
from __future__ import annotations

import numpy as np
from scipy import stats


def numerical_hessian(f, x, rel_step=1e-4, abs_step=1e-5):
    """Central-difference Hessian of a scalar function ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    d = x.size
    h = np.maximum(abs_step, rel_step * np.abs(x))
    H = np.empty((d, d))
    f0 = f(x)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = h[j]
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    return H


def wald_intervals(nll, theta_hat, level=0.95):
    """Wald standard errors and intervals from the observed information
    (Hessian of the negative log-likelihood) at the MLE.

    Returns (se, lo, hi, cov). ``se`` is NaN where the Hessian is not
    positive definite (flat or boundary solutions)."""
    H = numerical_hessian(nll, theta_hat)
    z = stats.norm.ppf(0.5 + level / 2)
    try:
        cov = np.linalg.inv(H)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(se)) or np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan)
        se = np.full(theta_hat.size, np.nan)
    return se, theta_hat - z * se, theta_hat + z * se, cov


def aic(loglik, k):
    return 2 * k - 2 * loglik


def bic(loglik, k, n):
    return k * np.log(n) - 2 * loglik
