"""Verifier 1: is the from-scratch Wiener first-passage density correct?

(a) joint density integrates to 1 over (t, choice), incl. sv and st
(b) P(upper) matches the closed form (sv = 0)
(c) agrees with an independent Euler-Maruyama simulator (KS distance on each boundary)
(d) agrees with PyDDM (Shinn et al. 2020), a Fokker-Planck solver, if installed
"""
import numpy as np
from _common import FIG, BLUE, GREY, ORANGE, mpl, save_json

from taulapse import ddm, exgauss, lognormal

out = {"normalisation": [], "monte_carlo": [], "pyddm": []}
g = np.linspace(1e-6, 25, 250001)
grid = [(1.0, 1.5, 0.3, 0.5, 0, 0), (-0.5, 2.0, 0.2, 0.3, 0, 0), (2.5, 0.8, 0.25, 0.7, 0, 0),
        (2.0, 1.0, 0.3, 0.6, 1.0, 0), (1.0, 1.5, 0.3, 0.5, 0.8, 0.2), (0.0, 2.5, 0.4, 0.5, 0, 0.3)]
for v, a, t0, w, sv, st in grid:
    m = [np.trapezoid(ddm.pdf(g, np.full(g.size, c), v, a, t0, w, sv, st), g) for c in (0, 1)]
    out["normalisation"].append({"v": v, "a": a, "t0": t0, "w": w, "sv": sv, "st": st,
                                 "total": m[0] + m[1], "p_upper": m[1],
                                 "p_upper_closed": ddm.prob_upper(v, a, w) if sv == 0 else None})
for name, f, args in [("exgauss", exgauss.pdf, (0.4, 0.05, 0.15)), ("lognormal", lognormal.pdf, (0.2, -1.2, 0.5))]:
    gg = np.linspace(-1, 30, 400001)
    out["normalisation"].append({"model": name, "total": float(np.trapezoid(f(gg, *args), gg))})

# (c) Monte Carlo
plt = mpl()
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, (v, a, t0, w, sv, st) in zip(axes, [(1.0, 1.5, 0.3, 0.5, 0, 0), (0.6, 2.0, 0.25, 0.35, 0, 0),
                                            (1.5, 1.4, 0.3, 0.5, 1.0, 0.15)]):
    rt, ch = ddm.simulate(20000, v, a, t0, w, sv, st, dt=1e-4, rng=7)
    row = {"v": v, "a": a, "t0": t0, "w": w, "sv": sv, "st": st, "n": 20000}
    tt = np.linspace(1e-4, 8, 16001)
    for c, sign in ((1, 1), (0, -1)):
        d = ddm.pdf(tt, np.full(tt.size, c), v, a, t0, w, sv, st)
        F = np.cumsum(d) * (tt[1] - tt[0])
        x = np.sort(rt[ch == c])
        Femp = np.arange(1, x.size + 1) / rt.size  # defective empirical CDF
        row[f"ks_{c}"] = float(np.max(np.abs(np.interp(x, tt, F) - Femp)))
        row[f"p_sim_{c}"] = float(np.mean(ch == c))
        row[f"p_ana_{c}"] = float(F[-1])
        h, e = np.histogram(x, bins=80, range=(0, 3))
        ax.stairs(sign * h / (rt.size * (e[1] - e[0])), e, color=GREY, fill=True, alpha=0.6,
                  label="Euler sim" if c == 1 else None)
        ax.plot(tt[tt < 3], sign * d[tt < 3], color=BLUE if c else ORANGE, lw=1.4,
                label=("analytic upper" if c else "analytic lower"))
    out["monte_carlo"].append(row)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"v={v}, a={a}, w={w}, sv={sv}, st={st}")
    ax.set_xlabel("RT (s)")
axes[0].set_ylabel("density (lower bound mirrored)")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/ddm_density_vs_simulation.png")

# (c') Euler step-size convergence: the residual KS on the upper bound should shrink
# with dt if it is discretisation bias (boundary overshoot ~ 0.58 sqrt(dt)), not a density bug.
out["euler_dt"] = []
tt = np.linspace(1e-4, 8, 16001)
d = ddm.pdf(tt, np.ones(tt.size, int), 1.0, 1.5, 0.3, 0.5)
F = np.cumsum(d) * (tt[1] - tt[0])
for dt in (1e-3, 1e-4, 2e-5):
    rt, ch = ddm.simulate(20000, 1.0, 1.5, 0.3, 0.5, dt=dt, rng=11)
    x = np.sort(rt[ch == 1])
    ks = float(np.max(np.abs(np.interp(x, tt, F) - np.arange(1, x.size + 1) / rt.size)))
    out["euler_dt"].append({"dt": dt, "ks_upper": ks, "mean_rt_upper_sim": float(x.mean()),
                            "mean_rt_upper_analytic": float(np.sum(tt * d) / np.sum(d))})

# (d) PyDDM
try:
    from pyddm import BoundConstant, DriftConstant, ICPoint, Model, NoiseConstant, OverlayNonDecision
    for v, a, t0, w in [(1.2, 1.6, 0.3, 0.5), (1.2, 1.6, 0.3, 0.35), (-0.4, 2.2, 0.2, 0.6)]:
        dx = 0.001
        m = Model(drift=DriftConstant(drift=v), noise=NoiseConstant(noise=1), bound=BoundConstant(B=a / 2),
                  IC=ICPoint(x0=(2 * w - 1) * a / 2), overlay=OverlayNonDecision(nondectime=t0),
                  dx=dx, dt=dx, T_dur=6)
        s = m.solve_numerical_cn() if w != 0.5 else m.solve_analytical()
        t = m.t_domain()
        mc = ddm.pdf(t, np.ones(t.size, int), v, a, t0, w)
        me = ddm.pdf(t, np.zeros(t.size, int), v, a, t0, w)
        out["pyddm"].append({"v": v, "a": a, "t0": t0, "w": w,
                             "solver": "analytical" if w == 0.5 else "Crank-Nicolson dx=dt=1e-3",
                             "max_abs_diff_upper": float(np.abs(s.pdf("correct") - mc).max()),
                             "max_abs_diff_lower": float(np.abs(s.pdf("error") - me).max()),
                             "peak_density": float(max(mc.max(), me.max()))})
except ImportError:
    out["pyddm"] = "pyddm not installed"

save_json("density_checks.json", out)
for k, v in out.items():
    print(k)
    for r in (v if isinstance(v, list) else [v]):
        print("  ", {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r.items()})
