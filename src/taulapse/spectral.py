r"""Temporal / spectral structure of an RT series.

Two estimators:

* Lomb-Scargle periodogram for unevenly sampled series (missing trials,
  omissions, or real onset times). Works in Hz if real timestamps are given,
  otherwise in cycles/trial.
* Welch PSD for evenly sampled (interpolated) series.

And two summary statistics, each with a surrogate null:

* aperiodic slope beta from a log-log fit P(f) ~ f^-beta (beta = 0 for white
  noise; beta > 0 = slow drifts / long-range dependence). Null = trial-order
  shuffles, which destroy all serial structure but keep the RT distribution.
* band excess: mean log-power residual (above the aperiodic fit) inside a
  target band, e.g. 0.02-0.08 Hz around the 0.05 Hz "default-mode
  interference" frequency (peak = max residual in band). Null = AR(1) surrogates with matched lag-1
  autocorrelation and variance, i.e. "is there a *peak* beyond generic
  red-noise slowness?". Without real timestamps this band cannot be mapped to
  Hz, and the module refuses to pretend otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _prep(t, x, detrend=True):
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if detrend and x.size > 2:
        x = x - np.polyval(np.polyfit(t, x, 1), t)
    else:
        x = x - x.mean()
    return t, x


def lomb_scargle(t, x, freqs=None, detrend=True):
    """Lomb-Scargle power (units of variance per frequency bin; normalised so that
    the sum over the default grid approximates the series variance)."""
    t, x = _prep(t, x, detrend)
    T = t.max() - t.min()
    if freqs is None:
        med_dt = np.median(np.diff(np.sort(t)))
        fmax = 0.5 / med_dt
        freqs = np.linspace(1 / T, fmax, int(np.floor(T * fmax)))
    p = signal.lombscargle(t, x, 2 * np.pi * freqs)
    return freqs, p


def welch(x, fs=1.0, nperseg=None):
    x = np.asarray(x, float)
    x = signal.detrend(x)
    nperseg = nperseg or min(len(x), 128)
    return signal.welch(x, fs=fs, nperseg=nperseg)


def aperiodic_slope(freqs, power, fmin=None, fmax=None):
    """beta in P ~ f^-beta by OLS in log-log; returns (beta, intercept)."""
    m = (power > 0) & np.isfinite(power)
    if fmin is not None:
        m &= freqs >= fmin
    if fmax is not None:
        m &= freqs <= fmax
    b, c = np.polyfit(np.log10(freqs[m]), np.log10(power[m]), 1)
    return -b, c


def band_excess(freqs, power, band, stat="max"):
    """Log10-power residual above the aperiodic fit inside ``band``.
    ``stat='max'`` (default) is a peak statistic: a narrow-band oscillation
    occupies 1-2 frequency bins, so averaging over the band dilutes it
    (our first version used the mean and had ~no power; see README)."""
    beta, c = aperiodic_slope(freqs, power)
    resid = np.log10(power) - (c - beta * np.log10(freqs))
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return np.nan
    return float(resid[m].max() if stat == "max" else resid[m].mean())


def ar1_surrogate(x, rng):
    x = np.asarray(x, float) - np.nanmean(x)
    r = np.corrcoef(x[:-1], x[1:])[0, 1]
    e = rng.standard_normal(x.size) * np.std(x) * np.sqrt(max(1 - r**2, 1e-6))
    e[0] = rng.standard_normal() * np.std(x)
    return signal.lfilter([1.0], [1.0, -r], e)


def analyse_series(t, x, band=None, n_surr=200, rng=0, freqs=None, band_stat="max"):
    """Spectral summary + surrogate p-values for one RT series.

    Returns dict with freqs, power, beta, p_beta (shuffle null),
    and if ``band`` is given: excess, p_excess (AR(1) null)."""
    rng = np.random.default_rng(rng)
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    f, p = lomb_scargle(t, x, freqs)
    beta, _ = aperiodic_slope(f, p)
    out = {"freqs": f, "power": p, "beta": float(beta)}
    null_b = np.empty(n_surr)
    for i in range(n_surr):
        _, ps = lomb_scargle(t, rng.permutation(x), f)
        null_b[i] = aperiodic_slope(f, ps)[0]
    out["p_beta"] = float((1 + np.sum(null_b >= beta)) / (n_surr + 1))
    if band is not None:
        ex = band_excess(f, p, band, band_stat)
        null_e = np.empty(n_surr)
        for i in range(n_surr):
            _, ps = lomb_scargle(t, ar1_surrogate(x, rng), f)
            null_e[i] = band_excess(f, ps, band, band_stat)
        out["excess"] = ex
        out["p_excess"] = float((1 + np.sum(null_e >= ex)) / (n_surr + 1))
    return out
