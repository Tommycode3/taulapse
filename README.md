# taulapse

![HYPERAKTIV ex-Gaussian](figures/hyperaktiv_exgauss.png)

[![CI](https://github.com/Tommycode3/taulapse/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Ex-Gaussian, drift-diffusion and spectral models of reaction-time variability, written from scratch in numpy/scipy, each checked by parameter recovery, model recovery and CI coverage on simulated data before being fit to real data. The motivation is ADHD, where the most consistent behavioural finding across 319 studies is variable responding ([Kofler et al. 2013](https://doi.org/10.1016/j.cpr.2013.06.001)). On adult CPT-II data the ADHD group has a heavier slow tail (median τ 95.8 vs 69.3 ms, Cliff's δ = +0.42) while its Gaussian component is faster.

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# one row per trial; rt in s or ms (auto-detected); optional: correct, subject, condition, onset
taulapse simulate --model ddm --n 400 --out demo.csv
taulapse fit demo.csv --out report/                       # ex-Gaussian + lognormal + DDM + spectrum
taulapse fit my.csv --rt-col rt_ms --subject-col id --models exgauss --min-rt 0.1
taulapse fit flanker.csv --condition-col congruency --sv --st   # condition-specific drift, full DDM
```

This writes `report/params.csv` (one row per subject, estimates and 95 % CIs), `report/report.md` and `report/fit_<subject>.png`.

```python
from taulapse import exgauss, ddm
fit = exgauss.fit(rts)                  # .mu .sigma .tau .ci['tau'] .bic
dd  = ddm.fit(rt, correct, fit_sv=True)  # .params .ci .loglik
```

To reproduce every number and figure, run `bash scripts/run_all.sh` (sequential, ~15–40 min on a laptop, < 300 MB RAM) and `pytest -q` (25 tests). `scripts/fetch_data.py` downloads the datasets from their original hosts with SHA-256 checksums; nothing is redistributed here.

## Adult ADHD on the CPT-II

HYPERAKTIV ([Hicks et al. 2021](https://doi.org/10.1145/3458305.3478454), OSF [3agwr](https://osf.io/3agwr/)) has 103 adult psychiatric outpatients: 51 with ADHD and 52 clinical controls with other diagnoses, not healthy volunteers. The stimulus coding isn't documented. Code `0` occurs on exactly 36/360 trials for everyone and gets a response on 44.5 % of trials versus 95.7 % for the rest, so it is the no-go "X". After dropping go RTs < 100 ms and two sessions with no responses, that leaves 50 ADHD vs 47 controls, median 320 go RTs each. The primary outcome, ex-Gaussian τ of go RTs, was pre-specified; the 7 secondary outcomes are Holm-corrected.

| outcome | median ADHD | median ctrl | Cliff's δ [95 % bootstrap CI] | Mann–Whitney p | Holm p |
|---|---|---|---|---|---|
| τ (ms) | 95.8 | 69.3 | +0.42 [+0.21, +0.62] | 0.0004 | (primary) |
| μ (ms) | 255.6 | 302.6 | −0.41 [−0.62, −0.19] | 0.0005 | 0.002 |
| σ (ms) | 38.4 | 39.8 | −0.03 [−0.27, +0.19] | 0.77 | 1.0 |
| RT SD (ms) | 114.2 | 92.1 | +0.41 [+0.20, +0.60] | 0.0005 | 0.002 |
| RT CV | 0.307 | 0.233 | +0.55 [+0.35, +0.72] | < 10⁻⁴ | < 0.001 |
| omissions (%) | 1.70 | 0.31 | +0.49 [+0.29, +0.67] | < 10⁻⁴ | < 0.001 |
| commissions (%) | 56.9 | 33.3 | +0.46 [+0.24, +0.66] | 0.0001 | < 0.001 |
| spectral slope β | 0.40 | 0.39 | +0.07 [−0.16, +0.31] | 0.56 | 1.0 |

The ADHD group isn't slower; with more commissions, it looks like a less cautious speed-accuracy setting, and the excess variability lives in τ and the CV. Across all participants τ correlates with ASRS self-report (ρ = 0.29, p = 0.006, n = 90) and WURS (ρ = 0.41, p < 10⁻⁴, n = 92); RT CV correlates more strongly (ρ = 0.46 and 0.49). The fast-μ, high-τ pattern partly reflects the negative μ–τ estimation-error correlation found in recovery, so the two effects aren't independent evidence.

Both groups' RT spectra show peaks at ≈ 0.017, 0.05 and 0.1 cycles/trial. The CPT-II changes its inter-stimulus interval every 20 trials within 60-trial blocks, which gives 1/60, 1/20 and a harmonic, so these are most plausibly a task artefact, not a brain rhythm. Trial onsets aren't released, so this couldn't be confirmed and the 0.05 Hz default-mode hypothesis ([Sonuga-Barke & Castellanos 2007](https://doi.org/10.1016/j.neubiorev.2007.02.005)) can't be tested in Hz. [Karalunas et al. 2013](https://doi.org/10.1111/jcpp.12028) also found the ADHD excess isn't confined to low frequencies.

The DDM is demonstrated on OpenNeuro [ds000102](https://openneuro.org/datasets/ds000102) ([Kelly et al. 2008](https://doi.org/10.1016/j.neuroimage.2007.08.008)), a flanker task with 26 healthy adults × 48 trials at 98.6 % accuracy. Drift is lower for incongruent flankers in 26/26 participants (median 3.70 vs 4.55, Wilcoxon p = 3·10⁻⁸). With ~0–2 errors per person, though, $a$ is unidentified in 7/26 participants (estimate at the bound a = 5). Its 10–22 s SOA puts the Nyquist frequency at ≤ 0.05 Hz, so it can't test the rhythm either.

## How it works

The ex-Gaussian treats an RT as $X = N + E$ with $N\sim\mathcal N(\mu,\sigma^2)$ and $E\sim\mathrm{Exp}(1/\tau)$:

$$
f(x\mid\mu,\sigma,\tau)=\frac{1}{\tau}\exp\!\left(\frac{\mu-x}{\tau}+\frac{\sigma^2}{2\tau^2}\right)\Phi\!\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\tau}\right),
\qquad \mathbb E[X]=\mu+\tau .
$$

It is fit by maximum likelihood over $(\mu,\log\sigma,\log\tau)$ with $\log\Phi$ from `log_ndtr` for tail stability, starting from method-of-moments ([Lacouture & Cousineau 2008](https://doi.org/10.20982/tqmp.04.1.p035)), with Wald intervals from the observed information or an optional Metropolis sampler. τ is descriptive; reading it as "lapses" needs a process model ([Matzke & Wagenmakers 2009](https://doi.org/10.3758/PBR.16.5.798); [Heathcote et al. 1991](https://doi.org/10.1037/0033-2909.109.2.340)).

That process model is the Ratcliff DDM ([Ratcliff 1978](https://doi.org/10.1037/0033-295X.85.2.59); [Ratcliff & McKoon 2008](https://doi.org/10.1162/neco.2008.12-06-420)): $dX_t = v\,dt + dW_t$ from $X_0 = w\,a$ until $X_t\in\{0,a\}$, plus non-decision time $t_0$ (scaling $s=1$; multiply $v$ and $a$ by 0.1 for the $s=0.1$ convention). The lower-bound first-passage density ([Navarro & Fuss 2009](https://doi.org/10.1016/j.jmp.2009.02.003)) is

$$
f_-(t\mid v,a,w)=\frac{1}{a^2}\exp\!\left(-v a w-\frac{v^2 t}{2}\right) g\!\left(\tfrac{t}{a^2}\,\middle|\,w\right),
$$

$$
g_{\text{small}}(u\mid w)=\frac{1}{\sqrt{2\pi u^3}}\sum_{k=-\infty}^{\infty}(w+2k)e^{-(w+2k)^2/2u},
\qquad
g_{\text{large}}(u\mid w)=\pi\sum_{k=1}^{\infty}k\,e^{-k^2\pi^2u/2}\sin(k\pi w),
$$

using whichever series needs fewer terms for $\varepsilon=10^{-12}$; the upper bound is $f_+(t\mid v,a,w)=f_-(t\mid -v,a,1-w)$. Drift variability $v_i\sim\mathcal N(v,s_v^2)$ integrates in closed form,

$$
f_-(t\mid v,s_v,a,w)=\frac{g(t/a^2\mid w)}{a^2\sqrt{1+s_v^2t}}\exp\!\left(\frac{s_v^2a^2w^2-2avw-v^2t}{2(1+s_v^2t)}\right),
$$

and uniform $t_0$ variability $s_t$ uses 24-point Gauss–Legendre restricted to the non-zero support. Fits are L-BFGS-B from EZ-diffusion starts ([Wagenmakers et al. 2007](https://doi.org/10.3758/BF03194023)), with $t_0 - s_t/2$ kept below the fastest RT so every candidate has finite likelihood. The joint density integrates to 1.000000 across 6 parameter sets, and its KS distance to an Euler–Maruyama simulator falls 0.0144, 0.0083, 0.0045 as dt shrinks from 10⁻³ to 2·10⁻⁵, so the residual is the simulator's. Against PyDDM's Crank–Nicolson solver ([Shinn et al. 2020](https://doi.org/10.7554/eLife.56938)) the max abs difference is 3·10⁻⁵ to 4·10⁻⁴.

For spectra, the detrended RT series goes through a Lomb–Scargle periodogram ([Lomb 1976](https://doi.org/10.1007/BF00648343); [Scargle 1982](https://doi.org/10.1086/160554)), which handles gaps from omissions and no-go trials without interpolation; Welch ([Welch 1967](https://doi.org/10.1109/TAU.1967.1161901)) is there for even sampling. With $\log_{10}P(f)\approx c-\beta\log_{10}f$, the slope $\beta$ is tested against trial-order shuffles, and a band peak $E_{\text{band}}=\max_{f\in\text{band}}[\log_{10}P(f)-(c-\beta\log_{10}f)]$ against AR(1) surrogates with matched lag-1 autocorrelation, the null that keeps generic red-noise slowness. The band statistic was first an average over the band and had power 0.08 even for a full-depth 0.05 Hz modulation, because a narrow rhythm fills 1–2 of ~40 bins; the max replaced it.

## What recovery showed

All of this is in `results/*.json` and `figures/`, simulated with Euler–Maruyama at dt = 10⁻⁴ s, never from the density being fitted.

The ex-Gaussian recovers well: at n = 100, r(true, est) is 0.97 / 0.80 / 0.95 for μ / σ / τ with Wald coverage 0.93 / 0.94 / 0.96. But estimation errors in μ and τ correlate at about −0.64 to −0.70, so group differences in μ and τ aren't independent.

For the 4-parameter DDM at n = 500, all damage to $a$ and $w$ comes from the 8 of 100 datasets with zero errors. There the likelihood depends almost only on $(1-w)a$, which is still recovered (RMSE 0.056, r = 0.94), and the Wald interval for $a$ widens to ≈ 40–390, so it flags the problem. In the 6-parameter model at n = 1000, drift variability $s_v$ is the weakest parameter (r = 0.59, RMSE 0.66). It trades off against $v$ and $a$ (estimate correlations 0.89 and 0.78), and on one n = 3000 dataset its 95 % profile interval is still roughly [0.65, 1.3]. "More drift variability" is a popular mechanistic reading of ADHD variability, but at clinical trial counts $s_v$ alone shouldn't carry a group comparison.

In model recovery by BIC at n = 300, data generated by an ex-Gaussian is labelled ex-Gaussian 0.50 of the time and shifted lognormal the other 0.50; at n = 1000 it is labelled correctly 0.83 of the time. In simulated 15-minute, 360-trial sessions, a 0.05 Hz rhythm in lapse probability is detected 0.02 / 0.12 / 0.80 of the time at modulation depth 0.25 / 0.5 / 1.0, against a 0.05 false-positive rate for white noise and 0.00 for AR(1).

## Limitations

- Not diagnostic. δ ≈ 0.4 means heavily overlapping distributions, from one dataset, against psychiatric controls.
- No age or sex adjustment on HYPERAKTIV. The CPT-II may have been part of the assessment that produced the ADHD label. Only one ADHD participant is coded as on stimulants, in a `patient_info.csv` with many blanks.
- The HYPERAKTIV OSF metadata says CC-BY 4.0, but the wiki requires prior written permission for competitions and commercial use.
- No dataset here can test a 0.05 Hz rhythm, and real task structure produces spectral peaks of its own.
- The standard DDM misfits the incongruent leading edge in the flanker data; flanker-specific models (shrinking spotlight, DSTP) aren't implemented.
- Fits are per-participant MLE. A hierarchical model ([HDDM, Wiecki et al. 2013](https://doi.org/10.3389/fninf.2013.00014)) is the natural next step.
- No contaminant mixture ([Ratcliff & Tuerlinckx 2002](https://doi.org/10.3758/BF03196302)); outliers are handled by RT windows.
- Still missing: an ADHD two-choice dataset with enough errors per person for a DDM comparison, and an ADHD dataset with real trial onsets.

## References

All DOIs were verified against Crossref by `scripts/verify_dois.py` (`results/dois.json`).

- Hicks, S. A. et al. (2021). HYPERAKTIV: An activity dataset from patients with ADHD. *MMSys '21*. https://doi.org/10.1145/3458305.3478454
- Huang-Pollock, C. L. et al. (2012). Evaluating vigilance deficits in ADHD: a meta-analysis of CPT performance. *J Abnorm Psychol*. https://doi.org/10.1037/a0027205
- Karalunas, S. L., Huang-Pollock, C. L., & Nigg, J. T. (2012). Decomposing ADHD-related effects in response speed and variability. *Neuropsychology*. https://doi.org/10.1037/a0029936
- Karalunas, S. L., Huang-Pollock, C. L., & Nigg, J. T. (2013). Is reaction time variability in ADHD mainly at low frequencies? *J Child Psychol Psychiatry*. https://doi.org/10.1111/jcpp.12028
- Karalunas, S. L. et al. (2014). Annual Research Review: Reaction time variability in ADHD and autism. *J Child Psychol Psychiatry*. https://doi.org/10.1111/jcpp.12217
- Kelly, A. M. C. et al. (2008). Competition between functional brain networks mediates behavioral variability. *NeuroImage*. https://doi.org/10.1016/j.neuroimage.2007.08.008
- Kofler, M. J. et al. (2013). Reaction time variability in ADHD: a meta-analytic review of 319 studies. *Clin Psychol Rev*. https://doi.org/10.1016/j.cpr.2013.06.001
- Castellanos, F. X. et al. (2005). Varieties of ADHD-related intra-individual variability. *Biol Psychiatry*. https://doi.org/10.1016/j.biopsych.2004.12.005
- Heathcote, A., Popiel, S. J., & Mewhort, D. J. K. (1991). Analysis of response time distributions: an example using the Stroop task. *Psychol Bull*. https://doi.org/10.1037/0033-2909.109.2.340
- Lacouture, Y., & Cousineau, D. (2008). How to use MATLAB to fit the ex-Gaussian. *TQMP*. https://doi.org/10.20982/tqmp.04.1.p035
- Leth-Steensen, C., Elbaz, Z. K., & Douglas, V. I. (2000). Mean response times, variability, and skew in the responding of ADHD children. *Acta Psychol*. https://doi.org/10.1016/S0001-6918(00)00019-6
- Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophys Space Sci*. https://doi.org/10.1007/BF00648343
- Matzke, D., & Wagenmakers, E.-J. (2009). Psychological interpretation of the ex-Gaussian and shifted Wald parameters. *Psychon Bull Rev*. https://doi.org/10.3758/PBR.16.5.798
- Mulder, M. J. et al. (2010). Basic impairments in regulating the speed-accuracy tradeoff predict symptoms of ADHD. *Biol Psychiatry*. https://doi.org/10.1016/j.biopsych.2010.07.031
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *J Math Psychol*. https://doi.org/10.1016/j.jmp.2009.02.003
- Gondan, M., Blurton, S. P., & Kesselmeier, M. (2014). Even faster and even more accurate first-passage time densities for the Wiener diffusion model. *J Math Psychol*. https://doi.org/10.1016/j.jmp.2014.05.002
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychol Rev*. https://doi.org/10.1037/0033-295X.85.2.59
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model. *Neural Comput*. https://doi.org/10.1162/neco.2008.12-06-420
- Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model. *Psychon Bull Rev*. https://doi.org/10.3758/BF03196302
- Scargle, J. D. (1982). Studies in astronomical time series analysis II. *ApJ*. https://doi.org/10.1086/160554
- Shinn, M., Lam, N. H., & Murray, J. D. (2020). A flexible framework for simulating and fitting generalized drift-diffusion models (PyDDM). *eLife*. https://doi.org/10.7554/eLife.56938
- Sonuga-Barke, E. J. S., & Castellanos, F. X. (2007). Spontaneous attentional fluctuations in impaired states and pathological conditions: a neurobiological hypothesis. *Neurosci Biobehav Rev*. https://doi.org/10.1016/j.neubiorev.2007.02.005
- Vaurio, R. G., Simmonds, D. J., & Mostofsky, S. H. (2009). Increased intra-individual RT variability in ADHD across response inhibition tasks. *Neuropsychologia*. https://doi.org/10.1016/j.neuropsychologia.2009.01.022
- Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychon Bull Rev*. https://doi.org/10.3758/BF03194023
- Welch, P. (1967). The use of FFT for the estimation of power spectra. *IEEE Trans Audio Electroacoust*. https://doi.org/10.1109/TAU.1967.1161901
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the DDM in Python. *Front Neuroinform*. https://doi.org/10.3389/fninf.2013.00014
- Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*. https://doi.org/10.7554/eLife.49547
- Ziegler, S. et al. (2016). Modelling ADHD: a review of ADHD theories through their predictions for computational models. *Neurosci Biobehav Rev*. https://doi.org/10.1016/j.neubiorev.2016.09.002

MIT © 2026 Tommaso Maggiolo. See [LICENSE](LICENSE).
