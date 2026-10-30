"""Command-line interface.

    taulapse fit trials.csv --out report/            # ex-Gaussian (+ DDM if a choice column exists)
    taulapse fit trials.csv --rt-col rt_ms --subject-col id --models exgauss,lognormal,ddm
    taulapse simulate --model ddm --n 400 --out demo.csv

Input: one row per trial. RTs in ms are detected (median > 20) and converted
to seconds. Rows with missing RT (omissions) are dropped from distributional
fits but kept (as gaps) for the spectral summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import __version__, ddm, exgauss, lognormal, spectral


def _load(path, rt_col, min_rt, max_rt):
    df = pd.read_csv(path, sep=None, engine="python")
    if rt_col not in df.columns:
        sys.exit(f"column '{rt_col}' not in {list(df.columns)}")
    rt = pd.to_numeric(df[rt_col], errors="coerce")
    unit = "ms" if np.nanmedian(rt) > 20 else "s"
    if unit == "ms":
        rt = rt / 1000.0
    rt = rt.where((rt >= min_rt) & (rt <= max_rt))
    df["_rt"] = rt
    return df, unit


def _plot(sub, rt, fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(rt, bins=40, density=True, color="#bbbbbb", edgecolor="white", label="data")
    g = np.linspace(rt.min() * 0.9, np.quantile(rt, 0.995) * 1.1, 400)
    if "exgauss" in fits:
        p = fits["exgauss"]
        ax.plot(g, exgauss.pdf(g, p["mu"], p["sigma"], p["tau"]), color="#1f5fa8", lw=2, label="ex-Gaussian")
    if "lognormal" in fits:
        p = fits["lognormal"]
        ax.plot(g, lognormal.pdf(g, p["shift"], p["m"], p["s"]), color="#c4572a", lw=1.5, ls="--", label="shifted lognormal")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("density")
    ax.set_title(f"subject {sub}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_fit(a):
    df, unit = _load(a.csv, a.rt_col, a.min_rt, a.max_rt)
    models = [m.strip() for m in a.models.split(",")]
    if "ddm" in models and a.choice_col not in df.columns:
        print(f"[taulapse] no '{a.choice_col}' column: skipping DDM", file=sys.stderr)
        models.remove("ddm")
    groups = df.groupby(a.subject_col) if a.subject_col in df.columns else [("all", df)]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for sub, d in groups:
        rt = d["_rt"].to_numpy()
        ok = np.isfinite(rt)
        row = {"subject": sub, "n_trials": len(d), "n_rt": int(ok.sum()),
               "mean_rt": np.nanmean(rt), "sd_rt": np.nanstd(rt, ddof=1)}
        row["cv_rt"] = row["sd_rt"] / row["mean_rt"]
        fits = {}
        if ok.sum() < 30:
            print(f"[taulapse] subject {sub}: only {ok.sum()} RTs, skipped", file=sys.stderr)
            rows.append(row)
            continue
        if "exgauss" in models:
            f = exgauss.fit(rt[ok])
            fits["exgauss"] = f.params
            for k, v in f.params.items():
                row[f"exg_{k}"] = v
                row[f"exg_{k}_lo"], row[f"exg_{k}_hi"] = f.ci[k]
            row["exg_bic"] = f.bic
        if "lognormal" in models:
            f = lognormal.fit(rt[ok])
            fits["lognormal"] = f.params
            for k, v in f.params.items():
                row[f"ln_{k}"] = v
            row["ln_bic"] = f.bic
        if "ddm" in models:
            ch = pd.to_numeric(d[a.choice_col], errors="coerce").to_numpy()
            okd = ok & np.isfinite(ch)
            cond = d[a.condition_col].to_numpy()[okd] if a.condition_col in d.columns else None
            f = ddm.fit(rt[okd], ch[okd].astype(int), cond=cond, fit_sv=a.sv, fit_st=a.st)
            for k, v in f.params.items():
                row[f"ddm_{k}"] = v
                lo, hi = f.ci.get(k, (np.nan, np.nan))
                row[f"ddm_{k}_lo"], row[f"ddm_{k}_hi"] = lo, hi
            row["ddm_loglik"] = f.loglik
        if ok.sum() >= 64:
            t = (d[a.onset_col].to_numpy(float) if a.onset_col in d.columns else np.arange(len(d), dtype=float))
            s = spectral.analyse_series(t, rt, n_surr=a.n_surr)
            row["spec_beta"], row["spec_p_beta"] = s["beta"], s["p_beta"]
            row["spec_units"] = "Hz" if a.onset_col in d.columns else "cycles/trial"
        _plot(sub, rt[ok], fits, os.path.join(a.out, f"fit_{sub}.png"))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "params.csv"), index=False)
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write(f"# taulapse {__version__} report\n\nInput: `{a.csv}` (RT unit detected: {unit}; "
                 f"kept {a.min_rt}-{a.max_rt} s)\n\nModels: {', '.join(models)}\n\n")
        try:
            fh.write(res.round(4).to_markdown(index=False))
        except ImportError:  # tabulate not installed
            fh.write("```\n" + res.round(4).to_string(index=False) + "\n```")
        fh.write("\n\nNot a diagnostic instrument. See README, 'What this does NOT claim'.\n")
    print(res.round(4).to_string(index=False))
    print(f"\n[taulapse] wrote {a.out}/params.csv, report.md, fit_*.png")


def cmd_simulate(a):
    rng = np.random.default_rng(a.seed)
    if a.model == "ddm":
        rt, ch = ddm.simulate(a.n, a.v, a.a, a.t0, a.w, rng=rng)
        df = pd.DataFrame({"subject": "sim", "rt": rt, "correct": ch})
    else:
        rt = exgauss.simulate(a.n, a.mu, a.sigma, a.tau, rng)
        df = pd.DataFrame({"subject": "sim", "rt": rt})
    df.to_csv(a.out, index=False)
    print(f"[taulapse] wrote {len(df)} trials to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="taulapse", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit", help="fit models to a CSV of trials")
    f.add_argument("csv")
    f.add_argument("--out", default="taulapse_report")
    f.add_argument("--rt-col", default="rt")
    f.add_argument("--choice-col", default="correct")
    f.add_argument("--subject-col", default="subject")
    f.add_argument("--condition-col", default="condition")
    f.add_argument("--onset-col", default="onset")
    f.add_argument("--models", default="exgauss,lognormal,ddm")
    f.add_argument("--min-rt", type=float, default=0.15)
    f.add_argument("--max-rt", type=float, default=3.0)
    f.add_argument("--sv", action="store_true", help="fit across-trial drift variability")
    f.add_argument("--st", action="store_true", help="fit across-trial non-decision variability")
    f.add_argument("--n-surr", type=int, default=200)
    f.set_defaults(func=cmd_fit)
    s = sp.add_parser("simulate", help="write a synthetic CSV")
    s.add_argument("--model", choices=["exgauss", "ddm"], default="ddm")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--out", default="sim.csv")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--v", type=float, default=1.5)
    s.add_argument("--a", type=float, default=1.5)
    s.add_argument("--t0", type=float, default=0.3)
    s.add_argument("--w", type=float, default=0.5)
    s.add_argument("--mu", type=float, default=0.40)
    s.add_argument("--sigma", type=float, default=0.05)
    s.add_argument("--tau", type=float, default=0.12)
    s.set_defaults(func=cmd_simulate)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
