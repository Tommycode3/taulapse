import numpy as np
import pytest
from scipy import integrate, stats

from taulapse import ddm, exgauss, lognormal


@pytest.mark.parametrize("mu,sigma,tau", [(0.4, 0.05, 0.1), (0.3, 0.02, 0.4), (500, 40, 80)])
def test_exgauss_integrates_to_one(mu, sigma, tau):
    lo, hi = mu - 10 * sigma, mu + 40 * tau
    val, _ = integrate.quad(lambda x: exgauss.pdf(x, mu, sigma, tau), lo, hi, limit=200)
    assert abs(val - 1) < 1e-6


def test_exgauss_matches_scipy_exponnorm():
    x = np.linspace(0.1, 2, 50)
    mu, sigma, tau = 0.4, 0.05, 0.15
    ref = stats.exponnorm.logpdf(x, tau / sigma, loc=mu, scale=sigma)
    assert np.allclose(exgauss.logpdf(x, mu, sigma, tau), ref, atol=1e-9)


def test_exgauss_cdf_consistent():
    x = np.linspace(0.2, 1.5, 30)
    ref = stats.exponnorm.cdf(x, 0.15 / 0.05, loc=0.4, scale=0.05)
    assert np.allclose(exgauss.cdf(x, 0.4, 0.05, 0.15), ref, atol=1e-9)


def test_lognormal_integrates_to_one():
    val, _ = integrate.quad(lambda x: lognormal.pdf(x, 0.2, -1.2, 0.5), 0.2, 20, limit=200)
    assert abs(val - 1) < 1e-6


@pytest.mark.parametrize("v,a,t0,w,sv,st", [
    (1.0, 1.5, 0.3, 0.5, 0, 0), (-0.7, 2.0, 0.2, 0.3, 0, 0), (3.0, 0.7, 0.2, 0.7, 0, 0),
    (1.5, 1.2, 0.3, 0.5, 1.2, 0), (1.0, 1.5, 0.3, 0.4, 0.8, 0.25)])
def test_ddm_joint_density_integrates_to_one(v, a, t0, w, sv, st):
    g = np.linspace(1e-6, 25, 200001)
    tot = sum(np.trapezoid(ddm.pdf(g, np.full(g.size, c), v, a, t0, w, sv, st), g) for c in (0, 1))
    assert abs(tot - 1) < 1e-4


@pytest.mark.parametrize("v,a,w", [(1.0, 1.5, 0.5), (-0.5, 2.0, 0.3), (2.0, 1.0, 0.65), (0.0, 1.0, 0.4)])
def test_ddm_choice_probability_closed_form(v, a, w):
    g = np.linspace(1e-6, 25, 200001)
    pu = np.trapezoid(ddm.pdf(g, np.ones(g.size, int), v, a, 0.0, w), g)
    assert abs(pu - ddm.prob_upper(v, a, w)) < 1e-5


def test_small_and_large_time_series_agree():
    # force both series on the same u and compare
    u = np.array([0.3, 0.6, 1.0])
    w = 0.37
    k = np.arange(-30, 31)[:, None]
    small = ((w + 2 * k) * np.exp(-(w + 2 * k) ** 2 / (2 * u))).sum(0) / np.sqrt(2 * np.pi * u**3)
    kk = np.arange(1, 200)[:, None]
    large = np.pi * (kk * np.exp(-kk**2 * np.pi**2 * u / 2) * np.sin(kk * np.pi * w)).sum(0)
    assert np.allclose(small, large, rtol=1e-10)
    assert np.allclose(ddm._std_lower(u, w), small, rtol=1e-8)


def test_ddm_density_matches_euler_simulation():
    rt, ch = ddm.simulate(6000, 1.2, 1.4, 0.3, 0.45, rng=3)
    tt = np.linspace(1e-4, 8, 8001)
    for c in (0, 1):
        F = np.cumsum(ddm.pdf(tt, np.full(tt.size, c), 1.2, 1.4, 0.3, 0.45)) * (tt[1] - tt[0])
        x = np.sort(rt[ch == c])
        ks = np.max(np.abs(np.interp(x, tt, F) - np.arange(1, x.size + 1) / rt.size))
        assert ks < 0.03
