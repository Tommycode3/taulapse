r"""Ex-Gaussian RT model: X = N + E, N ~ Normal(mu, sigma^2), E ~ Exp(mean tau).

.. math::
    f(x \mid \mu,\sigma,\tau) = \frac{1}{\tau}
    \exp\!\Big(\frac{\mu-x}{\tau} + \frac{\sigma^2}{2\tau^2}\Big)
    \,\Phi\!\Big(\frac{x-\mu}{\sigma} - \frac{\sigma}{\tau}\Big)

``tau`` is the exponential ("slow tail") component that ADHD studies interpret
as the signature of occasional attentional lapses. Units follow the input
(seconds or milliseconds); the fitter is scale-equivariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special

from ._stats import aic, bic, wald_intervals

PARAMS = ("mu", "sigma", "tau")


def logpdf(x, mu, sigma, tau):
    """Numerically stable log density (uses log Phi via ``log_ndtr``)."""
    x = np.asarray(x, dtype=float)
    return (-np.log(tau) + (mu - x) / tau + sigma**2 / (2 * tau**2)
            + special.log_ndtr((x - mu) / sigma - sigma / tau))


def pdf(x, mu, sigma, tau):
    return np.exp(logpdf(x, mu, sigma, tau))


def cdf(x, mu, sigma, tau):
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    # F = Phi(z) - exp(-(x-mu)/tau + sigma^2/(2 tau^2)) Phi(z - sigma/tau), in log space
    log_term = (mu - x) / tau + sigma**2 / (2 * tau**2) + special.log_ndtr(z - sigma / tau)
    return special.ndtr(z) - np.exp(log_term)


def simulate(n, mu, sigma, tau, rng=None):
    rng = np.random.default_rng(rng)
    return rng.normal(mu, sigma, n) + rng.exponential(tau, n)


def moment_start(x):
    """Method-of-moments starting values (Lacouture & Cousineau 2008 style)."""
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    g = ((x - m) ** 3).mean() / s**3
    tau = s * (g / 2) ** (1 / 3) if g > 0 else 0.5 * s
    tau = float(np.clip(tau, 0.05 * s, 0.95 * s))
    sigma = float(np.sqrt(max(s**2 - tau**2, (0.05 * s) ** 2)))
    return np.array([m - tau, sigma, tau])


@dataclass
class ExGaussFit:
    mu: float
    sigma: float
    tau: float
    loglik: float
    n: int
    se: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)
    converged: bool = True
    k: int = 3

    @property
    def params(self):
        return {"mu": self.mu, "sigma": self.sigma, "tau": self.tau}

    @property
    def aic(self):
        return aic(self.loglik, self.k)

    @property
    def bic(self):
        return bic(self.loglik, self.k, self.n)


def _nll_unc(theta, x):
    mu, ls, lt = theta
    return -np.sum(logpdf(x, mu, np.exp(ls), np.exp(lt)))


def fit(x, ci=True, level=0.95):
    """Maximum-likelihood fit. Optimisation is on (mu, log sigma, log tau);
    Wald CIs are computed on that scale and exponentiated, which keeps them
    positive and much closer to nominal coverage for skewed sigma/tau."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("need at least 10 RTs for an ex-Gaussian fit")
    m0 = moment_start(x)
    starts = [np.array([m0[0], np.log(m0[1]), np.log(m0[2])])]
    s = x.std()
    starts.append(np.array([np.median(x) - 0.3 * s, np.log(0.6 * s), np.log(0.5 * s)]))
    best = None
    for st in starts:
        r = optimize.minimize(_nll_unc, st, args=(x,), method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        r = optimize.minimize(_nll_unc, r.x, args=(x,), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    out = ExGaussFit(mu=float(th[0]), sigma=float(np.exp(th[1])), tau=float(np.exp(th[2])),
                     loglik=float(-best.fun), n=int(x.size), converged=bool(np.isfinite(best.fun)))
    if ci:
        se, lo, hi, _ = wald_intervals(lambda t: _nll_unc(t, x), th, level)
        out.se = {"mu": se[0], "log_sigma": se[1], "log_tau": se[2]}
        out.ci = {"mu": (lo[0], hi[0]),
                  "sigma": (np.exp(lo[1]), np.exp(hi[1])),
                  "tau": (np.exp(lo[2]), np.exp(hi[2]))}
    return out


def fit_bayes(x, n_iter=6000, burn=2000, rng=None, level=0.95):
    """Lightweight Bayesian fit: random-walk Metropolis on (mu, log sigma, log tau)
    with a proposal covariance taken from the MLE's inverse Hessian.

    Priors (weakly informative, data-scaled so they are unit-free):
      mu ~ Normal(median(x), (5*sd(x))^2); log sigma, log tau ~ Normal(log sd(x), 2^2).
    Returns dict(samples, mean, ci, accept)."""
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    sd, med = x.std(ddof=1), np.median(x)
    mle = fit(x, ci=False)
    th = np.array([mle.mu, np.log(mle.sigma), np.log(mle.tau)])

    def logpost(t):
        lp = (-0.5 * ((t[0] - med) / (5 * sd)) ** 2
              - 0.5 * ((t[1] - np.log(sd)) / 2) ** 2 - 0.5 * ((t[2] - np.log(sd)) / 2) ** 2)
        return lp - _nll_unc(t, x)

    from ._stats import numerical_hessian
    H = numerical_hessian(lambda t: _nll_unc(t, x), th)
    try:
        cov = np.linalg.inv(H)
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = np.diag([0.01 * sd, 0.05, 0.05]) ** 2
    L = np.linalg.cholesky(cov * (2.38**2 / 3))
    cur, cur_lp = th.copy(), logpost(th)
    chain = np.empty((n_iter, 3))
    acc = 0
    for i in range(n_iter):
        prop = cur + L @ rng.standard_normal(3)
        lp = logpost(prop)
        if np.log(rng.uniform()) < lp - cur_lp:
            cur, cur_lp = prop, lp
            acc += 1
        chain[i] = cur
    post = chain[burn:].copy()
    post[:, 1:] = np.exp(post[:, 1:])
    a = (1 - level) / 2
    q = np.quantile(post, [a, 1 - a], axis=0)
    return {"samples": post,
            "mean": dict(zip(PARAMS, post.mean(0))),
            "ci": {p: (q[0, i], q[1, i]) for i, p in enumerate(PARAMS)},
            "accept": acc / n_iter}
