"""Small, fast recovery checks (the full studies live in scripts/)."""
import numpy as np

from taulapse import compare, ddm, exgauss, lognormal, spectral


def test_exgauss_recovery_large_n():
    x = exgauss.simulate(4000, 0.42, 0.045, 0.16, rng=1)
    f = exgauss.fit(x)
    assert abs(f.mu - 0.42) < 0.01
    assert abs(f.sigma - 0.045) < 0.008
    assert abs(f.tau - 0.16) < 0.012
    lo, hi = f.ci["tau"]
    assert lo < f.tau < hi


def test_exgauss_scale_equivariance():
    x = exgauss.simulate(800, 0.42, 0.045, 0.16, rng=2)
    f1, f2 = exgauss.fit(x), exgauss.fit(1000 * x)
    assert np.allclose([f1.mu, f1.sigma, f1.tau], np.array([f2.mu, f2.sigma, f2.tau]) / 1000, rtol=1e-3)


def test_exgauss_bayes_runs_and_brackets_mle():
    x = exgauss.simulate(300, 0.4, 0.05, 0.12, rng=3)
    b = exgauss.fit_bayes(x, n_iter=2500, burn=800, rng=0)
    assert 0.1 < b["accept"] < 0.7
    lo, hi = b["ci"]["tau"]
    assert lo < exgauss.fit(x, ci=False).tau < hi


def test_lognormal_recovery():
    x = lognormal.simulate(4000, 0.25, -1.3, 0.45, rng=4)
    f = lognormal.fit(x)
    assert abs(f.shift - 0.25) < 0.03 and abs(f.m + 1.3) < 0.08 and abs(f.s - 0.45) < 0.05


def test_ddm_recovery_4param():
    rng = np.random.default_rng(5)
    errs = []
    for th in [(1.5, 1.4, 0.3, 0.5), (0.8, 2.0, 0.25, 0.4), (2.5, 1.0, 0.35, 0.6)]:
        rt, ch = ddm.simulate(2000, *th, rng=rng)
        f = ddm.fit(rt, ch, ci=False)
        errs.append([f.params["v"] - th[0], f.params["a"] - th[1], f.params["t0"] - th[2], f.params["w"] - th[3]])
    errs = np.abs(np.array(errs))
    assert np.all(errs[:, 0] < 0.25)
    assert np.all(errs[:, 1] < 0.12)
    assert np.all(errs[:, 2] < 0.03)
    assert np.all(errs[:, 3] < 0.05)


def test_ddm_condition_specific_drift():
    rng = np.random.default_rng(6)
    r1, c1 = ddm.simulate(1500, 2.2, 1.5, 0.3, rng=rng)
    r2, c2 = ddm.simulate(1500, 1.0, 1.5, 0.3, rng=rng)
    rt = np.r_[r1, r2]; ch = np.r_[c1, c2]; cond = np.r_[["easy"] * 1500, ["hard"] * 1500]
    f = ddm.fit(rt, ch, cond=cond, fit_w=False, ci=False)
    assert abs(f.params["v[easy]"] - 2.2) < 0.25 and abs(f.params["v[hard]"] - 1.0) < 0.2


def test_model_selection_picks_ddm_for_ddm_data():
    rt, ch = ddm.simulate(1500, 1.2, 1.8, 0.3, 0.5, rng=7)
    best, _ = compare.fit_all(rt, ch)
    assert best == "ddm"


def test_spectral_detects_strong_slow_drift_and_not_white():
    rng = np.random.default_rng(8)
    t = np.arange(400.0)
    white = rng.normal(0.5, 0.1, 400)
    drift = white + 0.1 * np.sin(2 * np.pi * t / 80)
    assert spectral.analyse_series(t, drift, n_surr=99, rng=1)["p_beta"] < 0.05
    assert spectral.analyse_series(t, white, n_surr=99, rng=1)["beta"] < 0.6
