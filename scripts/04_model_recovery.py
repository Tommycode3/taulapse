"""Verifier 4: model recovery. Simulate (rt, choice) from each of
ex-Gaussian+Bernoulli, shifted-lognormal+Bernoulli and DDM; fit all three;
select by BIC; report the confusion matrix P(selected | generating)."""
import numpy as np
from _common import FIG, mpl, save_json

from taulapse import compare, ddm, exgauss, lognormal

rng = np.random.default_rng(404)
N_PER = 40
N_TRIALS = 300  # reassigned in the loop below
U = lambda lo, hi: rng.uniform(lo, hi)  # noqa: E731


def gen(model):
    p = U(0.75, 0.97)
    if model == "exgauss":
        th = dict(mu=U(0.30, 0.50), sigma=U(0.03, 0.08), tau=U(0.05, 0.25))
        return exgauss.simulate(N_TRIALS, **th, rng=rng), (rng.uniform(size=N_TRIALS) < p).astype(int), th
    if model == "lognormal":
        th = dict(shift=U(0.15, 0.30), m=U(-1.6, -0.8), s=U(0.3, 0.7))
        return lognormal.simulate(N_TRIALS, **th, rng=rng), (rng.uniform(size=N_TRIALS) < p).astype(int), th
    th = dict(v=U(0.8, 2.5), a=U(1.0, 2.0), t0=U(0.20, 0.35), w=U(0.4, 0.6))
    rt, ch = ddm.simulate(N_TRIALS, th["v"], th["a"], th["t0"], th["w"], rng=rng)
    return rt, ch, th


M = compare.MODELS
out = {"n_per_model": N_PER, "models": M, "by_n": {}}
confs = {}
for N_TRIALS in (300, 1000):
    conf = np.zeros((3, 3), int)
    dbic = {m: [] for m in M}
    for i, gm in enumerate(M):
        for _ in range(N_PER):
            rt, ch, _th = gen(gm)
            ok = np.isfinite(rt)
            best, fits = compare.fit_all(rt[ok], ch[ok])
            conf[i, M.index(best)] += 1
            b = sorted(f["bic"] for f in fits.values())
            dbic[gm].append(b[1] - b[0])
    confs[N_TRIALS] = conf
    out["by_n"][N_TRIALS] = {"confusion_counts": conf.tolist(), "confusion_prob": (conf / N_PER).round(3).tolist(),
                             "median_bic_margin": {m: float(np.median(v)) for m, v in dbic.items()}}
save_json("model_recovery.json", out)

plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
labels = ["ex-Gauss", "lognormal", "DDM"]
for ax, (n, conf) in zip(axes, confs.items()):
    P = conf / N_PER
    ax.imshow(P, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("selected by BIC"); ax.set_ylabel("generating model")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", color="white" if P[i, j] > 0.5 else "black")
    ax.set_title(f"{N_PER} datasets/model, {n} trials each")
fig.suptitle("Model recovery confusion matrix P(selected | generating)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/model_recovery.png")
print(out)
