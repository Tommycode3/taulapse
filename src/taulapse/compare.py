"""Model comparison on two-choice data (rt, choice).

To compare the DDM (which jointly explains choice and RT) with purely
descriptive RT distributions on equal footing, the descriptive models are
extended with an independent Bernoulli choice: L = p^k (1-p)^(n-k) * prod f(rt).
All three candidates then have 4 free parameters, so AIC and BIC rank them
identically; we report BIC.
"""
from __future__ import annotations

import numpy as np

from . import ddm, exgauss, lognormal
from ._stats import bic

MODELS = ("exgauss", "lognormal", "ddm")


def _bern_ll(choice):
    k, n = np.sum(choice == 1), choice.size
    p = np.clip(k / n, 1e-9, 1 - 1e-9)
    return k * np.log(p) + (n - k) * np.log(1 - p)


def fit_all(rt, choice, ddm_starts=2):
    rt, choice = np.asarray(rt, float), np.asarray(choice).astype(int)
    n = rt.size
    out = {}
    eg = exgauss.fit(rt, ci=False)
    out["exgauss"] = {"loglik": eg.loglik + _bern_ll(choice), "k": 4, "params": eg.params}
    ln = lognormal.fit(rt)
    out["lognormal"] = {"loglik": ln.loglik + _bern_ll(choice), "k": 4, "params": ln.params}
    dd = ddm.fit(rt, choice, n_starts=ddm_starts, ci=False)
    out["ddm"] = {"loglik": dd.loglik, "k": dd.k, "params": dd.params}
    for m in out.values():
        m["bic"] = float(bic(m["loglik"], m["k"], n))
    best = min(out, key=lambda m: out[m]["bic"])
    return best, out
