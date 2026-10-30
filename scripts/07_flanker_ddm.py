"""Real data 2: two-choice DDM on OpenNeuro ds000102 (Eriksen flanker, healthy
adults, 26 participants x 2 runs x 24 trials, PDDL). This is a *demonstration*
on a small, near-ceiling dataset (accuracy ~ 98.6 %), not a strong test.

Coding: upper bound = correct, lower = error ('accuracy coding'); drift varies
by condition (congruent / incongruent); a, t0 shared; w fixed at 0.5 (with
accuracy coding a start-point bias is not interpretable).
RT < 0.15 s (incl. RT = 0 = no response) excluded.
Note: the SOA here is 10-14 s, so the RT series' Nyquist frequency is
~0.04-0.05 Hz; the 0.05 Hz hypothesis cannot be tested in this dataset.
"""
import glob
import os

import numpy as np
import pandas as pd
from _common import BLUE, DATA, FIG, GREY, ORANGE, mpl, save_json
from scipy import stats

from taulapse import ddm

frames = []
for p in sorted(glob.glob(f"{DATA}/ds000102/sub-*_events.tsv")):
    d = pd.read_csv(p, sep="\t")
    d["subject"] = os.path.basename(p)[:6]
    d["run"] = os.path.basename(p)[7:12]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["rt"] = pd.to_numeric(df.response_time, errors="coerce")
df["correct"] = (df.correctness == "correct").astype(int)
n_all = len(df)
df = df[df.rt >= 0.15].copy()
soa = df.groupby(["subject", "run"]).onset.apply(lambda s: np.diff(np.sort(s.unique()))).explode().astype(float)

rows = []
for sid, d in df.groupby("subject"):
    f = ddm.fit(d.rt.to_numpy(), d.correct.to_numpy(), cond=d.Stimulus.to_numpy(), fit_w=False, n_starts=4)
    rows.append({"subject": sid, "n": len(d), "acc": d.correct.mean(),
                 "v_con": f.params["v[congruent]"], "v_inc": f.params["v[incongruent]"],
                 "a": f.params["a"], "t0": f.params["t0"],
                 "v_con_ci": f.ci.get("v[congruent]"), "v_inc_ci": f.ci.get("v[incongruent]"),
                 "a_ci": f.ci.get("a"), "at_a_bound": bool(f.params["a"] > 4.99),
                 "mrt_con": d[d.Stimulus == "congruent"].rt.mean(), "mrt_inc": d[d.Stimulus == "incongruent"].rt.mean()})
R = pd.DataFrame(rows)
w = stats.wilcoxon(R.v_con, R.v_inc)
wr = stats.wilcoxon(R.mrt_inc, R.mrt_con)
pooled = ddm.fit(df.rt.to_numpy(), df.correct.to_numpy(), cond=df.Stimulus.to_numpy(), fit_w=False, n_starts=4)
ci_width_v = np.array([c[1] - c[0] for c in R.v_inc_ci if c is not None and np.all(np.isfinite(c))])
out = {"n_subjects": int(R.shape[0]), "trials_total": int(n_all), "trials_kept": int(len(df)),
       "accuracy": float(df.correct.mean()), "errors": int((df.correct == 0).sum()),
       "soa_s": {"min": float(soa.min()), "median": float(soa.median()), "max": float(soa.max())},
       "flanker_effect_ms": {"median": float(np.median(R.mrt_inc - R.mrt_con) * 1000),
                             "wilcoxon_p": float(wr.pvalue)},
       "v_con_median": float(R.v_con.median()), "v_inc_median": float(R.v_inc.median()),
       "v_diff_wilcoxon": {"stat": float(w.statistic), "p": float(w.pvalue),
                           "n_v_inc_lt_v_con": int(np.sum(R.v_inc < R.v_con))},
       "a_median": float(R.a.median()), "t0_median": float(R.t0.median()),
       "n_a_at_upper_bound": int(R.at_a_bound.sum()),
       "median_ci_width_v_inc": float(np.median(ci_width_v)) if ci_width_v.size else None,
       "n_finite_ci_v_inc": int(ci_width_v.size),
       "pooled_fit_illustrative": pooled.params, "pooled_ci": pooled.ci}
R.to_csv(f"{DATA}/ds000102_ddm_per_subject.csv", index=False)
save_json("flanker_ddm.json", out)
print(out)

# Figure: pooled RT distributions with the pooled DDM fit (illustration only)
plt = mpl()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
g = np.linspace(0.15, 1.6, 500)
for ax, cond in zip(axes, ["congruent", "incongruent"]):
    d = df[df.Stimulus == cond]
    h, e = np.histogram(d[d.correct == 1].rt, bins=40, range=(0.15, 1.6))
    ax.stairs(h / (len(d) * (e[1] - e[0])), e, color=GREY, fill=True, alpha=0.7, label="correct RTs")
    v = pooled.params[f"v[{cond}]"]
    ax.plot(g, ddm.pdf(g, np.ones(g.size, int), v, pooled.params["a"], pooled.params["t0"], 0.5),
            color=BLUE, lw=2, label="DDM (pooled fit)")
    ax.set_title(f"{cond}: v={v:.2f}, a={pooled.params['a']:.2f}, t0={pooled.params['t0']*1000:.0f} ms "
                 f"({int((d.correct == 0).sum())} errors)")
    ax.set_xlabel("RT (s)"); ax.set_ylabel("defective density")
    ax.legend(fontsize=7)
fig.suptitle("ds000102 flanker (26 healthy adults). Pooling across people is for display only.", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/flanker_ddm.png")

fig, ax = plt.subplots(figsize=(3.6, 3.4))
for _, r in R.iterrows():
    ax.plot([0, 1], [r.v_con, r.v_inc], color=GREY, lw=0.8)
ax.plot([0, 1], [R.v_con.median(), R.v_inc.median()], color=ORANGE, lw=2.5, marker="o")
ax.set_xticks([0, 1], ["congruent", "incongruent"]); ax.set_ylabel("drift rate v (per-subject MLE)")
ax.set_title(f"Wilcoxon p={w.pvalue:.3g}, {out['v_diff_wilcoxon']['n_v_inc_lt_v_con']}/{len(R)} lower v when incongruent", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/flanker_drift_by_condition.png")
