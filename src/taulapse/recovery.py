"""Summaries for parameter-recovery studies."""
from __future__ import annotations

import numpy as np


def summarize(true, est, lo=None, hi=None):
    """bias, RMSE, Pearson r, relative bias and (if CIs given) coverage, per column."""
    true, est = np.asarray(true, float), np.asarray(est, float)
    ok = np.isfinite(est) & np.isfinite(true)
    err = est[ok] - true[ok]
    out = {
        "n": int(ok.sum()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r": float(np.corrcoef(true[ok], est[ok])[0, 1]) if np.ptp(true[ok]) > 0 else float("nan"),
        "median_rel_err": float(np.median(np.abs(err) / np.maximum(np.abs(true[ok]), 1e-9))),
    }
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        okc = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(true)
        out["coverage"] = float(np.mean((lo[okc] <= true[okc]) & (true[okc] <= hi[okc])))
        out["n_ci"] = int(okc.sum())
    return out
