r"""Ratcliff drift-diffusion model (Wiener first-passage time), from scratch.

Evidence X_t obeys  dX_t = v dt + s dW_t,  X_0 = w a,  absorbing bounds 0 and a,
with the scaling convention s = 1 (as in Navarro & Fuss 2009 / HDDM; multiply
v, a by 0.1 to compare with the classic s = 0.1 convention).

choice == 1  <-> upper bound a ("correct" by convention)
choice == 0  <-> lower bound 0

Lower-bound first-passage density (Navarro & Fuss 2009, eqs. 1-7):

    f(t | v, a, w) = a^-2 exp(-v a w - v^2 t / 2) f(t / a^2 | 0, 1, w)

where the standardised density f(u | 0, 1, w) is evaluated with whichever of
the small-time series
    (2 pi u^3)^(-1/2) sum_k (w + 2k) exp(-(w + 2k)^2 / (2u))
or the large-time series
    pi sum_{k>=1} k exp(-k^2 pi^2 u / 2) sin(k pi w)
needs fewer terms for a target absolute error ``eps``. The upper-bound
density follows by symmetry: f_upper(t | v, a, w) = f_lower(t | -v, a, 1 - w).

Across-trial drift variability v_i ~ N(v, sv^2) is integrated analytically:

    f(t | v, sv, a, w) = a^-2 f(u|0,1,w) (1 + sv^2 t)^(-1/2)
                         exp[(sv^2 a^2 w^2 - 2 a v w - v^2 t) / (2 (1 + sv^2 t))]

Across-trial non-decision variability t0_i ~ U(t0 - st/2, t0 + st/2) is
integrated numerically with Gauss-Legendre quadrature over the part of the
support where the integrand is non-zero (so the kink at t = t0_i is never
inside the quadrature interval).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from ._stats import aic, bic, wald_intervals

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


# --------------------------------------------------------------------------- density
def _std_lower(u, w, eps=1e-12):
    """f(u | v=0, a=1, w) for u > 0 (arrays broadcast). Navarro & Fuss term rules."""
    u = np.asarray(u, float)
    w = np.broadcast_to(np.asarray(w, float), u.shape)
    out = np.zeros(u.shape)
    pos = u > 0
    if not np.any(pos):
        return out
    up, wp = u[pos], w[pos]
    # number of terms, large-time
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_l = np.pi * up * eps < 1
        kl = np.where(cond_l, np.sqrt(np.maximum(-2 * np.log(np.pi * up * eps) / (np.pi**2 * up), 0)), 0)
        kl = np.maximum(kl, 1 / (np.pi * np.sqrt(up)))
        # number of terms, small-time
        cond_s = 2 * np.sqrt(2 * np.pi * up) * eps < 1
        ks = np.where(cond_s, 2 + np.sqrt(np.maximum(-2 * up * np.log(2 * np.sqrt(2 * np.pi * up) * eps), 0)), 2)
        ks = np.maximum(ks, np.sqrt(up) + 1)
    small = ks < kl
    res = np.empty(up.shape)
    if np.any(small):
        us, ws = up[small], wp[small]
        K = int(np.ceil(ks[small].max()))
        k = np.arange(-((K - 1) // 2), (K - 1) // 2 + (K - 1) % 2 + 1)[:, None]
        wk = ws[None, :] + 2 * k
        res[small] = (wk * np.exp(-wk**2 / (2 * us[None, :]))).sum(0) / np.sqrt(2 * np.pi * us**3)
    if np.any(~small):
        ul, wl = up[~small], wp[~small]
        K = int(min(np.ceil(kl[~small].max()), 500))
        k = np.arange(1, K + 1)[:, None]
        res[~small] = np.pi * (k * np.exp(-(k**2) * np.pi**2 * ul[None, :] / 2)
                               * np.sin(k * np.pi * wl[None, :])).sum(0)
    out[pos] = np.maximum(res, 0.0)
    return out


def _density_dec(t, choice, v, a, w, sv):
    """Decision-time density (no t0) for given choice; t may be <= 0 (-> 0)."""
    t = np.asarray(t, float)
    choice = np.broadcast_to(np.asarray(choice), t.shape)
    v = np.broadcast_to(np.asarray(v, float), t.shape)
    up = choice == 1
    vv = np.where(up, -v, v)            # upper = lower with (-v, 1-w)
    ww = np.where(up, 1 - w, w)
    tp = np.where(t > 0, t, 1.0)
    base = _std_lower(tp / a**2, ww) / a**2
    s2 = sv**2
    expo = (s2 * a**2 * ww**2 - 2 * a * vv * ww - vv**2 * tp) / (2 * (1 + s2 * tp))
    dens = base * np.exp(expo) / np.sqrt(1 + s2 * tp)
    return np.where(t > 0, dens, 0.0)


def pdf(rt, choice, v, a, t0, w=0.5, sv=0.0, st=0.0):
    """Joint density of (rt, choice). ``v`` may be an array (per-trial drift)."""
    rt = np.asarray(rt, float)
    if st <= 1e-8:
        return _density_dec(rt - t0, choice, v, a, w, sv)
    lo = t0 - st / 2
    hi = np.minimum(t0 + st / 2, rt)
    width = np.maximum(hi - lo, 0.0)
    # quadrature nodes for the non-decision time tau in [lo, hi]
    tau = lo + (hi[:, None] - lo) * (_GL_X[None, :] + 1) / 2
    vv = np.broadcast_to(np.asarray(v, float), rt.shape)[:, None]
    dens = _density_dec(rt[:, None] - tau, np.asarray(choice)[:, None] if np.ndim(choice) else choice,
                        vv, a, w, sv)
    integral = (dens * _GL_W[None, :]).sum(1) * width / 2
    return integral / st


def loglik(rt, choice, v, a, t0, w=0.5, sv=0.0, st=0.0):
    d = pdf(rt, choice, v, a, t0, w, sv, st)
    return np.log(np.maximum(d, 1e-300))


def prob_upper(v, a, w=0.5):
    """P(hit upper bound) for sv = 0 (closed form)."""
    if abs(v) < 1e-12:
        return w
    return (1 - np.exp(-2 * v * a * w)) / (1 - np.exp(-2 * v * a))


# --------------------------------------------------------------------------- simulation
def simulate(n, v, a, t0, w=0.5, sv=0.0, st=0.0, dt=1e-4, max_t=10.0, rng=None, block=256):
    """Euler-Maruyama simulation, independent of the analytic density (so it can be
    used to validate it). Steps are taken in vectorised blocks; trials not absorbed
    by ``max_t`` are returned with rt = NaN. ``v`` may be scalar or length-n."""
    rng = np.random.default_rng(rng)
    vi = np.broadcast_to(np.asarray(v, float), (n,)).copy()
    if sv > 0:
        vi = vi + sv * rng.standard_normal(n)
    x = np.full(n, w * a)
    t = np.zeros(n)
    rt = np.full(n, np.nan)
    ch = np.full(n, -1)
    active = np.arange(n)
    sq = np.sqrt(dt)
    steps_done = 0
    while active.size and steps_done * dt < max_t:
        inc = vi[active, None] * dt + sq * rng.standard_normal((active.size, block))
        path = x[active, None] + np.cumsum(inc, 1)
        hit_u = path >= a
        hit_l = path <= 0
        hit = hit_u | hit_l
        any_hit = hit.any(1)
        first = np.argmax(hit, 1)
        idx = active[any_hit]
        f = first[any_hit]
        rt[idx] = t[idx] + (f + 1) * dt
        ch[idx] = hit_u[any_hit, f].astype(int)
        keep = ~any_hit
        x[active[keep]] = path[keep, -1]
        t[active[keep]] += block * dt
        active = active[keep]
        steps_done += block
    tnd = t0 + (st * (rng.uniform(size=n) - 0.5) if st > 0 else 0.0)
    return rt + tnd, ch


# --------------------------------------------------------------------------- fitting
def ez_start(rt, choice):
    """EZ-diffusion (Wagenmakers et al. 2007) closed-form estimates, s = 1 scale."""
    rt, choice = np.asarray(rt, float), np.asarray(choice)
    n = rt.size
    pc = np.clip(choice.mean(), 0.5 + 1 / (2 * n), 1 - 1 / (2 * n))
    rc = rt[choice == 1] if np.sum(choice == 1) > 2 else rt
    vrt, mrt = rc.var(ddof=1), rc.mean()
    L = np.log(pc / (1 - pc))
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt
    v = x ** 0.25
    a = L / v
    y = -v * a
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    t0 = mrt - mdt
    t0 = float(np.clip(t0, 0.05, rt.min() - 0.02)) if rt.min() > 0.07 else 0.5 * rt.min()
    return float(v), float(np.clip(a, 0.4, 4.5)), t0


@dataclass
class DDMFit:
    params: dict
    loglik: float
    n: int
    k: int
    se: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)
    converged: bool = True

    @property
    def aic(self):
        return aic(self.loglik, self.k)

    @property
    def bic(self):
        return bic(self.loglik, self.k, self.n)


def fit(rt, choice, cond=None, fit_sv=False, fit_st=False, fit_w=True, n_starts=3,
        ci=True, rng=0, level=0.95):
    """Maximum-likelihood DDM fit with L-BFGS-B on bounded parameters.

    cond : optional array of condition labels -> one drift rate per condition,
           all other parameters shared (e.g. congruent vs incongruent flanker).
    Internally t0 is parameterised by ``tmin = t0 - st/2`` (bounded below min RT)
    so that every candidate gives a finite likelihood.
    """
    rt = np.asarray(rt, float)
    choice = np.asarray(choice).astype(int)
    ok = np.isfinite(rt) & (choice >= 0)
    rt, choice = rt[ok], choice[ok]
    if cond is None:
        cond = np.zeros(rt.size, int)
        labels = [None]
    else:
        cond = np.asarray(cond)[ok]
        labels = list(np.unique(cond))
        cond = np.searchsorted(labels, cond)
    nc = len(labels)
    rmin = rt.min()
    names = [f"v[{l}]" if l is not None else "v" for l in labels] + ["a", "tmin"]
    bounds = [(-8, 8)] * nc + [(0.2, 5.0), (1e-3, rmin - 1e-4)]
    if fit_w:
        names.append("w"); bounds.append((0.05, 0.95))
    if fit_sv:
        names.append("sv"); bounds.append((0.0, 5.0))
    if fit_st:
        names.append("st"); bounds.append((0.0, 0.8))

    def unpack(th):
        d = dict(zip(names, th))
        v = np.asarray(th[:nc])[cond]
        return v, d["a"], d["tmin"], d.get("w", 0.5), d.get("sv", 0.0), d.get("st", 0.0)

    def nll(th):
        v, a, tmin, w, sv, st = unpack(th)
        val = -np.sum(loglik(rt, choice, v, a, tmin + st / 2, w, sv, st))
        return val if np.isfinite(val) else 1e10

    rng = np.random.default_rng(rng)
    v0, a0, t00 = ez_start(rt, choice)
    base = [v0] * nc + [a0, min(t00, rmin - 0.01)]
    if fit_w:
        base.append(0.5)
    if fit_sv:
        base.append(0.5)
    if fit_st:
        base.append(0.05)
    starts = [np.array(base)]
    for _ in range(n_starts - 1):
        s = np.array(base) * rng.uniform(0.7, 1.3, len(base))
        starts.append(s)
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    best = None
    for s in starts:
        s = np.clip(s, lb + 1e-6, ub - 1e-6)
        r = optimize.minimize(nll, s, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    p = dict(zip(names, map(float, th)))
    st = p.get("st", 0.0)
    p["t0"] = p.pop("tmin") + st / 2
    out = DDMFit(params=p, loglik=float(-best.fun), n=int(rt.size), k=len(names),
                 converged=bool(best.success))
    if ci:
        se, lo, hi, _ = wald_intervals(nll, th, level)
        at_bound = (np.abs(th - lb) < 1e-4) | (np.abs(th - ub) < 1e-4)
        se = np.where(at_bound, np.nan, se)
        for i, nm in enumerate(names):
            key = "t0" if nm == "tmin" else nm
            shift = st / 2 if nm == "tmin" else 0.0
            out.se[key] = float(se[i])
            out.ci[key] = (float(lo[i] + shift) if np.isfinite(se[i]) else np.nan,
                           float(hi[i] + shift) if np.isfinite(se[i]) else np.nan)
    return out
