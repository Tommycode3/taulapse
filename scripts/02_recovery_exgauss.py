"""Verifier 2: ex-Gaussian parameter recovery, CI coverage (Wald and Bayesian),
and the estimation-error correlation structure (identifiability)."""
import numpy as np
from _common import BLUE, FIG, GREY, mpl, save_json

from taulapse import exgauss
from taulapse.recovery import summarize

rng = np.random.default_rng(2026)
N_SETS = 150
LO = np.array([0.30, 0.02, 0.03])   # mu, sigma, tau (seconds)
HI = np.array([0.60, 0.10, 0.30])
truth = LO + (HI - LO) * rng.uniform(size=(N_SETS, 3))
out = {"design": {"n_sets": N_SETS, "ranges_s": {"mu": [0.30, 0.60], "sigma": [0.02, 0.10], "tau": [0.03, 0.30]}},
       "mle": {}, "bayes": {}, "error_corr": {}}
est = {}
for n in (100, 500):
    E = np.empty((N_SETS, 3)); L = np.empty((N_SETS, 3)); H = np.empty((N_SETS, 3))
    for i, th in enumerate(truth):
        x = exgauss.simulate(n, *th, rng=rng)
        f = exgauss.fit(x)
        E[i] = [f.mu, f.sigma, f.tau]
        for j, p in enumerate(exgauss.PARAMS):
            L[i, j], H[i, j] = f.ci[p]
    est[n] = E
    out["mle"][n] = {p: summarize(truth[:, j], E[:, j], L[:, j], H[:, j]) for j, p in enumerate(exgauss.PARAMS)}
    err = E - truth
    out["error_corr"][n] = np.corrcoef(err.T).round(3).tolist()

# Bayesian credible-interval coverage, small n where Wald is least trustworthy
nb = 60
cov = np.zeros(3); cov_wald = np.zeros(3); acc = []
for i in range(nb):
    th = truth[i]
    x = exgauss.simulate(100, *th, rng=rng)
    b = exgauss.fit_bayes(x, rng=i)
    w = exgauss.fit(x)
    acc.append(b["accept"])
    for j, p in enumerate(exgauss.PARAMS):
        cov[j] += b["ci"][p][0] <= th[j] <= b["ci"][p][1]
        cov_wald[j] += w.ci[p][0] <= th[j] <= w.ci[p][1]
out["bayes"] = {"n_trials": 100, "n_sets": nb, "coverage_95": dict(zip(exgauss.PARAMS, cov / nb)),
                "wald_coverage_same_sets": dict(zip(exgauss.PARAMS, cov_wald / nb)),
                "mean_accept": float(np.mean(acc))}
save_json("recovery_exgauss.json", out)

plt = mpl()
fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
for r, n in enumerate((100, 500)):
    for j, p in enumerate(exgauss.PARAMS):
        ax = axes[r, j]
        ax.scatter(truth[:, j] * 1000, est[n][:, j] * 1000, s=9, color=BLUE, alpha=0.7, lw=0)
        lim = [LO[j] * 1000 * 0.8, HI[j] * 1000 * 1.15]
        ax.plot(lim, lim, color=GREY, lw=1)
        s = out["mle"][n][p]
        ax.set_title(f"{p}, n={n}: r={s['r']:.2f}, bias={s['bias']*1000:+.1f} ms, cov={s['coverage']:.2f}")
        ax.set_xlabel(f"true {p} (ms)")
        ax.set_ylabel(f"estimated {p} (ms)")
fig.suptitle("Ex-Gaussian parameter recovery (MLE, 95% Wald coverage)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/recovery_exgauss.png")

for n in (100, 500):
    for p, s in out["mle"][n].items():
        print(n, p, {k: round(v, 4) for k, v in s.items()})
    print("error corr (mu,sigma,tau)", out["error_corr"][n])
print(out["bayes"])
