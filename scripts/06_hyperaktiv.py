"""Real data 1: HYPERAKTIV Conners' CPT-II, adult ADHD vs clinical controls.

Trial coding (inferred from the data, not documented in the release; see README):
  TrialK  = stimulus letter code; code 0 occurs on exactly 10 % of trials
            (36/360 for every participant) and is the only code withheld on most
            trials -> code 0 = 'X' (no-go); all other codes = go.
  ResponseK = RT in ms, -1 = no response.
Go-trial RTs < 100 ms are treated as anticipations and dropped (CPT-II convention).
Sessions with < 100 valid go RTs are excluded (2 sessions contain no responses at all).

Pre-specified primary outcome: ex-Gaussian tau (go RTs). Secondary: mu, sigma,
RT SD, RT CV, omission %, commission %, spectral slope beta.
Group test: Mann-Whitney U; effect size = Cliff's delta with a percentile
bootstrap 95 % CI (2000 resamples); Holm correction over the secondaries.
"""
import numpy as np
import pandas as pd
from _common import BLUE, DATA, FIG, GREY, ORANGE, mpl, save_json
from scipy import stats

from taulapse import exgauss, lognormal, spectral

rng = np.random.default_rng(31)
cpt = pd.read_csv(f"{DATA}/hyperaktiv/CPT_II_ConnersContinuousPerformanceTest.csv", sep=";", encoding="utf-8-sig")
info = pd.read_csv(f"{DATA}/hyperaktiv/patient_info.csv", sep=";")
T = cpt[[f"Trial{k}" for k in range(1, 361)]].to_numpy()
R = cpt[[f"Response{k}" for k in range(1, 361)]].to_numpy().astype(float)
sanity = {"x_per_subject_unique": np.unique((T == 0).sum(1)).tolist(),
          "resp_rate_code0": float((R[T == 0] >= 0).mean()), "resp_rate_other": float((R[T != 0] >= 0).mean())}

rows, spectra, excluded = [], {}, []
freqs = np.linspace(1 / 360, 0.5, 180)  # cycles per trial
for i, sid in enumerate(cpt["ID"]):
    go = T[i] != 0
    r = R[i]
    rt = np.where(go & (r >= 100), r / 1000.0, np.nan)
    x = rt[np.isfinite(rt)]
    if x.size < 100:  # pre-specified: sessions with < 100 valid go RTs are invalid
        excluded.append({"ID": int(sid), "n_go_rt": int(x.size)})
        continue
    f = exgauss.fit(x)
    ln = lognormal.fit(x)
    s = spectral.analyse_series(np.arange(360.0), rt, n_surr=199, rng=rng, freqs=freqs)
    spectra[sid] = s["power"]
    rows.append({"ID": sid, "n_go_rt": x.size, "mean_rt": x.mean(), "sd_rt": x.std(ddof=1),
                 "cv_rt": x.std(ddof=1) / x.mean(),
                 "mu": f.mu, "sigma": f.sigma, "tau": f.tau,
                 "tau_lo": f.ci["tau"][0], "tau_hi": f.ci["tau"][1],
                 "bic_exg_minus_ln": f.bic - ln.bic,
                 "omission_pct": 100 * np.mean(go & (r < 0)) / np.mean(go),
                 "commission_pct": 100 * np.mean(~go & (r >= 0)) / np.mean(~go),
                 "anticip_n": int(np.sum(go & (r >= 0) & (r < 100))),
                 "beta": s["beta"], "p_beta": s["p_beta"]})
df = pd.DataFrame(rows).merge(info[["ID", "ADHD", "SEX", "AGE", "ASRS", "WURS", "MED_Stimulants"]], on="ID")
df.to_csv(f"{DATA}/hyperaktiv_derived_per_subject.csv", index=False)


def cliffs(a, b):
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def boot_cliffs(a, b, n=2000):
    v = [cliffs(rng.choice(a, a.size), rng.choice(b, b.size)) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


outcomes = ["tau", "mu", "sigma", "sd_rt", "cv_rt", "omission_pct", "commission_pct", "beta"]
A = df[df.ADHD == 1]; C = df[df.ADHD == 0]
res = {}
for o in outcomes:
    a, c = A[o].to_numpy(), C[o].to_numpy()
    u = stats.mannwhitneyu(a, c, alternative="two-sided")
    res[o] = {"median_adhd": float(np.median(a)), "median_ctrl": float(np.median(c)),
              "U": float(u.statistic), "p": float(u.pvalue),
              "cliffs_delta": cliffs(a, c), "cliffs_ci95": boot_cliffs(a, c)}
sec = [o for o in outcomes if o != "tau"]
ps = np.array([res[o]["p"] for o in sec])
order = np.argsort(ps)
holm = np.empty_like(ps)
running = 0
for k, idx in enumerate(order):
    running = max(running, min(1, (len(ps) - k) * ps[idx]))
    holm[idx] = running
for o, h in zip(sec, holm):
    res[o]["p_holm"] = float(h)

dim = {}
for o in ("tau", "sd_rt", "cv_rt"):
    for scale in ("ASRS", "WURS"):
        m = df[[o, scale]].dropna()
        rho = stats.spearmanr(m[o], m[scale])
        dim[f"{o}~{scale}"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": int(len(m))}

out = {"sanity": sanity, "excluded_sessions": excluded, "n_adhd": int(len(A)), "n_ctrl": int(len(C)),
       "stimulant_medicated_adhd": int(np.nansum(A.MED_Stimulants == 1)),
       "go_rts_per_subject_median": float(df.n_go_rt.median()),
       "exg_preferred_over_lognormal_by_bic": int(np.sum(df.bic_exg_minus_ln < 0)),
       "spectral_p_beta_lt_05": {"adhd": int(np.sum(A.p_beta < 0.05)), "ctrl": int(np.sum(C.p_beta < 0.05))},
       "group_tests": res, "dimensional": dim}
save_json("hyperaktiv.json", out)
for o, r in res.items():
    print(f"{o:15s} adhd {r['median_adhd']:.4f} ctrl {r['median_ctrl']:.4f} p={r['p']:.4f} "
          f"holm={r.get('p_holm', float('nan')):.3f} delta={r['cliffs_delta']:+.2f} {np.round(r['cliffs_ci95'], 2)}")
print({k: v for k, v in out.items() if k not in ("group_tests",)})

# ---------------------------------------------------------------- figures
plt = mpl()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, (lab, q) in zip(axes[:2], [("low-tau participant", 0.1), ("high-tau participant", 0.9)]):
    sid = df.iloc[(df.tau - df.tau.quantile(q)).abs().argsort().iloc[0]]
    i = int(np.where(cpt["ID"] == sid.ID)[0][0])
    r = R[i]; x = r[(T[i] != 0) & (r >= 100)] / 1000
    ax.hist(x, bins=35, density=True, color=GREY, edgecolor="white")
    g = np.linspace(0.1, x.max(), 400)
    ax.plot(g, exgauss.pdf(g, sid.mu, sid.sigma, sid.tau), color=BLUE, lw=2)
    ax.set_title(f"{lab} ({'ADHD' if sid.ADHD else 'clin. ctrl'})\n"
                 f"mu={sid.mu*1000:.0f}  sigma={sid.sigma*1000:.0f}  tau={sid.tau*1000:.0f} ms")
    ax.set_xlabel("go RT (s)"); ax.set_ylabel("density")
ax = axes[2]
for k, (grp, col, lab) in enumerate([(C, GREY, "clinical controls"), (A, ORANGE, "ADHD")]):
    y = grp.tau.to_numpy() * 1000
    ax.scatter(k + rng.uniform(-0.12, 0.12, y.size), y, s=12, color=col, alpha=0.8, lw=0)
    ax.hlines(np.median(y), k - 0.25, k + 0.25, color="k", lw=2)
ax.set_xticks([0, 1], [f"clinical ctrl (n={len(C)})", f"ADHD (n={len(A)})"])
ax.set_ylabel("ex-Gaussian tau (ms)")
r = res["tau"]
ax.set_title(f"tau (go RTs), Mann-Whitney p={r['p']:.1e}\nCliff's delta={r['cliffs_delta']:+.2f} [{r['cliffs_ci95'][0]:+.2f}, {r['cliffs_ci95'][1]:+.2f}]")
fig.tight_layout()
fig.savefig(f"{FIG}/hyperaktiv_exgauss.png")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for grp, col, lab in [(C, GREY, "clinical controls"), (A, ORANGE, "ADHD")]:
    P = np.array([spectra[s] / spectra[s].sum() for s in grp.ID])
    m = np.log10(P).mean(0); se = np.log10(P).std(0, ddof=1) / np.sqrt(len(P))
    ax.plot(freqs, m, color=col, lw=1.5, label=f"{lab} (n={len(P)})")
    ax.fill_between(freqs, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.25, lw=0)
ax.set_xscale("log")
ax.set_xlabel("frequency (cycles / trial)  [real onsets not released -> no Hz axis]")
ax.set_ylabel("log10 relative power")
ax.set_title("CPT-II go-RT spectra (Lomb-Scargle, mean +/- 95% CI)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/hyperaktiv_spectra.png")
