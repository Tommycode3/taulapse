"""Verifier 5: does the spectral pipeline (a) keep its false-positive rate under
white and red (AR(1)) noise and (b) detect a genuine 0.05 Hz modulation of the
lapse rate? Series mimic a 360-trial task with irregular onsets
(SOA uniform 1-4 s, i.e. ~14 min, comparable to CPT-II) and 5 % omissions.

Generative model for the positive case: each trial's RT is ex-Gaussian, but the
exponential tail is switched on with a probability that oscillates at f0:
    p_lapse(t) = p0 * (1 + m sin(2 pi f0 t)),  tail ~ Exp(tau_lapse)
so 'amplitude' m is a modulation depth of the lapse probability, not of RT.
"""
import numpy as np
from _common import BLUE, FIG, GREY, ORANGE, mpl, save_json

from taulapse import spectral

rng = np.random.default_rng(5)
N_TR, F0, BAND = 360, 0.05, (0.03, 0.07)
N_SIM, N_SURR = 60, 99


def onsets():
    t = np.cumsum(rng.uniform(1.0, 4.0, N_TR))
    return t - t[0]


def series(kind, m=0.0):
    t = onsets()
    base = rng.normal(0.40, 0.05, N_TR)
    if kind == "white":
        x = base + rng.exponential(0.10, N_TR)
    elif kind == "ar1":
        e = np.zeros(N_TR)
        for i in range(1, N_TR):
            e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.06)
        x = base + e + rng.exponential(0.06, N_TR)
    else:
        p = np.clip(0.25 * (1 + m * np.sin(2 * np.pi * F0 * t + rng.uniform(0, 2 * np.pi))), 0, 1)
        x = base + rng.exponential(0.03, N_TR) + (rng.uniform(size=N_TR) < p) * rng.exponential(0.35, N_TR)
    x[rng.uniform(size=N_TR) < 0.05] = np.nan
    return t, x


out = {"design": {"n_trials": N_TR, "soa_s": [1, 4], "band_hz": BAND, "f0": F0, "n_sim": N_SIM, "n_surr": N_SURR}}
for kind in ("white", "ar1"):
    pb, pe = [], []
    for _ in range(N_SIM):
        t, x = series(kind)
        r = spectral.analyse_series(t, x, band=BAND, n_surr=N_SURR, rng=rng)
        pb.append(r["p_beta"]); pe.append(r["p_excess"])
    out[kind] = {"fpr_beta_0.05": float(np.mean(np.array(pb) < 0.05)),
                 "fpr_excess_0.05": float(np.mean(np.array(pe) < 0.05))}
    print(kind, out[kind])

power, power_mean = {}, {}
example = None
for m in (0.0, 0.25, 0.5, 1.0):
    pe, pm = [], []
    for k in range(N_SIM):
        t, x = series("osc", m)
        r = spectral.analyse_series(t, x, band=BAND, n_surr=N_SURR, rng=rng)
        pe.append(r["p_excess"])
        # the first (rejected) design: band-*mean* residual instead of the peak
        pm.append(spectral.analyse_series(t, x, band=BAND, n_surr=N_SURR, rng=rng, band_stat="mean")["p_excess"])
        if m == 1.0 and k == 0:
            example = (r["freqs"], r["power"])
    power[m] = float(np.mean(np.array(pe) < 0.05))
    power_mean[m] = float(np.mean(np.array(pm) < 0.05))
    print("modulation", m, "detection rate (peak)", power[m], "(band mean)", power_mean[m])
out["detection_rate_by_modulation"] = power
out["detection_rate_by_modulation_bandmean_stat"] = power_mean
save_json("spectral_validation.json", out)

plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
f, p = example
axes[0].loglog(f, p, color=BLUE, lw=1)
axes[0].axvspan(*BAND, color=ORANGE, alpha=0.2, label="0.03-0.07 Hz band")
axes[0].set_xlabel("frequency (Hz)"); axes[0].set_ylabel("Lomb-Scargle power")
axes[0].set_title("one simulated series, lapse prob. modulated at 0.05 Hz (m=1)")
axes[0].legend(fontsize=7)
ms = list(power)
axes[1].bar([str(m) for m in ms], [power[m] for m in ms], color=BLUE)
axes[1].axhline(0.05, color=GREY, ls="--", lw=1)
axes[1].set_xlabel("modulation depth m of lapse probability")
axes[1].set_ylabel("P(band-excess p < .05)")
axes[1].set_title(f"sensitivity, {N_SIM} sims per level, 360 trials")
fig.tight_layout()
fig.savefig(f"{FIG}/spectral_validation.png")
