r"""Shifted lognormal RT model: X = theta + exp(Y), Y ~ Normal(m, s^2).

.. math::
    f(x\mid\theta,m,s)=\frac{1}{(x-\theta)\,s\sqrt{2\pi}}
    \exp\!\Big(-\frac{(\ln(x-\theta)-m)^2}{2s^2}\Big),\quad x>\theta

The shift is parameterised as ``theta = min(x) - exp(eta)`` so it always sits
below the fastest RT (the three-parameter lognormal likelihood is unbounded as
``theta -> min(x)``; we report the interior local maximum, as is standard).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ._stats import aic, bic

PARAMS = ("shift", "m", "s")


def logpdf(x, shift, m, s):
    x = np.asarray(x, float)
    y = x - shift
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(y) - np.log(s) - 0.5 * np.log(2 * np.pi) - (np.log(y) - m) ** 2 / (2 * s**2)
    return np.where(y > 0, out, -np.inf)


def pdf(x, shift, m, s):
    return np.exp(logpdf(x, shift, m, s))


def simulate(n, shift, m, s, rng=None):
    rng = np.random.default_rng(rng)
    return shift + np.exp(rng.normal(m, s, n))


@dataclass
class LognormalFit:
    shift: float
    m: float
    s: float
    loglik: float
    n: int
    k: int = 3

    @property
    def params(self):
        return {"shift": self.shift, "m": self.m, "s": self.s}

    @property
    def aic(self):
        return aic(self.loglik, self.k)

    @property
    def bic(self):
        return bic(self.loglik, self.k, self.n)


def fit(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    xmin, sd = x.min(), x.std()

    def unpack(t):
        return xmin - np.exp(t[0]), t[1], np.exp(t[2])

    def nll(t):
        v = -np.sum(logpdf(x, *unpack(t)))
        return v if np.isfinite(v) else 1e12

    best = None
    for frac in (0.5, 0.2, 1.0):
        shift0 = xmin - frac * sd
        ly = np.log(x - shift0)
        t0 = np.array([np.log(xmin - shift0), ly.mean(), np.log(ly.std() + 1e-6)])
        r = optimize.minimize(nll, t0, method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    sh, m, s = unpack(best.x)
    return LognormalFit(shift=float(sh), m=float(m), s=float(s), loglik=float(-best.fun), n=int(x.size))
