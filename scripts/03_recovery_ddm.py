"""Verifier 3: DDM parameter recovery.

A. 4-parameter DDM (v, a, t0, w), 100 random parameter sets, n = 500 trials,
   with 95 % Wald CI coverage.
B. 6-parameter DDM (+ sv, st), 40 random sets, n = 1000 trials.
C. Identifiability at one fixed truth: 30 replicate datasets (n = 1000) fitted
   with the 6-parameter model -> correlation of estimates across replicates,
   plus a profile likelihood over sv on one large dataset (n = 3000).
Data are simulated with the Euler scheme (dt = 1e-4), never with the density
that is being fitted.
"""
import time

import numpy as np
from _common import BLUE, FIG, GREY, ORANGE, RES, mpl, save_json

from taulapse import ddm
from taulapse.recovery import summarize

rng = np.random.default_rng(54)
out = {}
plt = mpl()

# ---------------------------------------------------------------- A
P4 = ["v", "a", "t0", "w"]
LO4 = np.array([0.5, 0.8, 0.15, 0.35]); HI4 = np.array([3.0, 2.5, 0.45, 0.65])
NA, nA = 100, 500
TA = LO4 + (HI4 - LO4) * rng.uniform(size=(NA, 4))
EA = np.empty_like(TA); LA = np.empty_like(TA); HA = np.empty_like(TA)
NERR = np.zeros(NA, int)
t0_ = time.time()
for i, th in enumerate(TA):
    rt, ch = ddm.simulate(nA, *th, rng=rng)
    NERR[i] = int(np.sum(ch == 0))
    f = ddm.fit(rt, ch)
    for j, p in enumerate(P4):
        EA[i, j] = f.params[p]
        LA[i, j], HA[i, j] = f.ci[p]
out["A_4param"] = {"n_sets": NA, "n_trials": nA, "seconds": time.time() - t0_,
                   "ranges": dict(zip(P4, zip(LO4, HI4))),
                   "summary": {p: summarize(TA[:, j], EA[:, j], LA[:, j], HA[:, j]) for j, p in enumerate(P4)}}
# Identifiability: with zero errors only the distance to the upper bound, (1-w)a,
# is constrained by the data; a and w separately are not.
ok = NERR >= 5
out["A_4param"]["n_sets_zero_errors"] = int(np.sum(NERR == 0))
out["A_4param"]["n_sets_ge5_errors"] = int(ok.sum())
out["A_4param"]["summary_ge5_errors"] = {p: summarize(TA[ok, j], EA[ok, j], LA[ok, j], HA[ok, j]) for j, p in enumerate(P4)}
z = NERR == 0
out["A_4param"]["zero_error_sets_upper_distance"] = summarize((1 - TA[z, 3]) * TA[z, 1], (1 - EA[z, 3]) * EA[z, 1])
out["A_4param"]["zero_error_sets_a"] = summarize(TA[z, 1], EA[z, 1], LA[z, 1], HA[z, 1])
np.savez(f"{RES}/recovery_ddm_A_raw.npz", truth=TA, est=EA, lo=LA, hi=HA, n_err=NERR)
print("A done", out["A_4param"]["seconds"])

# ---------------------------------------------------------------- B
P6 = ["v", "a", "t0", "w", "sv", "st"]
LO6 = np.array([0.5, 0.8, 0.2, 0.4, 0.0, 0.0]); HI6 = np.array([3.0, 2.2, 0.45, 0.6, 2.0, 0.3])
NB, nB = 40, 1000
TB = LO6 + (HI6 - LO6) * rng.uniform(size=(NB, 6))
EB = np.empty_like(TB)
t0_ = time.time()
for i, th in enumerate(TB):
    rt, ch = ddm.simulate(nB, *th, rng=rng)
    f = ddm.fit(rt, ch, fit_sv=True, fit_st=True, ci=False, n_starts=2)
    EB[i] = [f.params[p] for p in P6]
out["B_6param"] = {"n_sets": NB, "n_trials": nB, "seconds": time.time() - t0_,
                   "ranges": dict(zip(P6, zip(LO6, HI6))),
                   "summary": {p: summarize(TB[:, j], EB[:, j]) for j, p in enumerate(P6)}}
print("B done", out["B_6param"]["seconds"])

# ---------------------------------------------------------------- C
truthC = dict(v=1.5, a=1.5, t0=0.3, w=0.5, sv=1.0, st=0.15)
EC = np.empty((30, 6))
for i in range(30):
    rt, ch = ddm.simulate(1000, truthC["v"], truthC["a"], truthC["t0"], truthC["w"], truthC["sv"], truthC["st"], rng=rng)
    f = ddm.fit(rt, ch, fit_sv=True, fit_st=True, ci=False, n_starts=2)
    EC[i] = [f.params[p] for p in P6]
corrC = np.corrcoef(EC.T)
rt, ch = ddm.simulate(3000, truthC["v"], truthC["a"], truthC["t0"], truthC["w"], truthC["sv"], truthC["st"], rng=rng)
full = ddm.fit(rt, ch, fit_sv=True, fit_st=True, ci=False)
prof = []
from scipy import optimize
for sv in np.linspace(0.0, 2.0, 11):
    def nll(th):
        v, a, tmin, w, st = th
        val = -np.sum(ddm.loglik(rt, ch, v, a, tmin + st / 2, w, sv, st))
        return val if np.isfinite(val) else 1e10
    p = full.params
    x0 = [p["v"], p["a"], p["t0"] - p["st"] / 2, p["w"], p["st"]]
    r = optimize.minimize(nll, x0, method="L-BFGS-B",
                          bounds=[(-8, 8), (0.2, 5), (1e-3, rt.min() - 1e-4), (0.05, 0.95), (0, 0.8)])
    prof.append({"sv": float(sv), "nll": float(r.fun), "v": float(r.x[0]), "a": float(r.x[1])})
out["C_identifiability"] = {"truth": truthC, "n_reps": 30, "n_trials": 1000,
                            "mean_est": dict(zip(P6, EC.mean(0))), "sd_est": dict(zip(P6, EC.std(0, ddof=1))),
                            "corr_estimates": corrC.round(3).tolist(), "param_order": P6,
                            "profile_sv_n3000": prof, "full_fit_n3000": full.params}
save_json("recovery_ddm.json", out)

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
for j, p in enumerate(P4):
    ax = axes[0, j]
    ax.scatter(TA[NERR > 0, j], EA[NERR > 0, j], s=8, color=BLUE, alpha=0.7, lw=0, label=">=1 error")
    ax.scatter(TA[NERR == 0, j], EA[NERR == 0, j], s=14, color="k", marker="x", lw=0.8, label="0 errors (a, w not separable)")
    if j == 1:
        ax.legend(fontsize=6)
    lim = [LO4[j], HI4[j]]; ax.plot(lim, lim, color=GREY, lw=1)
    s = out["A_4param"]["summary"][p]
    ax.set_title(f"4-par {p}: r={s['r']:.2f} cov={s['coverage']:.2f}")
    ax.set_xlabel(f"true {p}"); ax.set_ylabel(f"est. {p}")
for ax in axes[0, 4:]:
    ax.axis("off")
for j, p in enumerate(P6):
    ax = axes[1, j]
    ax.scatter(TB[:, j], EB[:, j], s=10, color=ORANGE, alpha=0.8, lw=0)
    lim = [LO6[j], HI6[j]]; ax.plot(lim, lim, color=GREY, lw=1)
    s = out["B_6param"]["summary"][p]
    ax.set_title(f"6-par {p}: r={s['r']:.2f} RMSE={s['rmse']:.3f}")
    ax.set_xlabel(f"true {p}"); ax.set_ylabel(f"est. {p}")
fig.suptitle("DDM parameter recovery: top n=500 (v,a,t0,w); bottom n=1000 with across-trial sv, st", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/recovery_ddm.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
im = axes[0].imshow(corrC, vmin=-1, vmax=1, cmap="RdBu_r")
axes[0].set_xticks(range(6), P6); axes[0].set_yticks(range(6), P6)
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f"{corrC[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(corrC[i, j]) > 0.6 else "black")
axes[0].set_title("correlation of estimates across 30 replicates")
fig.colorbar(im, ax=axes[0], fraction=0.046)
sv = [r["sv"] for r in prof]; nl = np.array([r["nll"] for r in prof])
ax = axes[1]
ax.plot(sv, nl - nl.min(), "o-", color=BLUE, label="profile NLL (others re-fit)")
ax.axhline(1.92, color=GREY, ls="--", lw=1, label="95% profile threshold (1.92)")
ax.axvline(truthC["sv"], color=ORANGE, lw=1, label="true sv")
ax2 = ax.twinx(); ax2.plot(sv, [r["a"] for r in prof], "s:", color="#555555", ms=3, label="a at profile")
ax2.set_ylabel("re-fitted a", color="#555555")
ax.set_xlabel("sv fixed at"); ax.set_ylabel("NLL - min")
ax.set_title("profile likelihood of sv (n=3000)")
ax.legend(fontsize=7, loc="upper center")
fig.tight_layout()
fig.savefig(f"{FIG}/ddm_identifiability.png")

print("A ge5 errors", out["A_4param"]["summary_ge5_errors"], "zero-err", out["A_4param"]["n_sets_zero_errors"],
      out["A_4param"]["zero_error_sets_upper_distance"])
for k in ("A_4param", "B_6param"):
    print(k)
    for p, s in out[k]["summary"].items():
        print("  ", p, {kk: round(vv, 4) for kk, vv in s.items()})
print("C corr\n", np.array2string(corrC, precision=2))
print("C mean", out["C_identifiability"]["mean_est"], "sd", out["C_identifiability"]["sd_est"])
print("profile", [(r["sv"], round(r["nll"] - nl.min(), 2), round(r["a"], 3)) for r in prof])
