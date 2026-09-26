# Phase A: Statistical Anomaly Detector

Script: [`phase_a.py`](../phase_a.py) · Method: [1] Sec. III and Alg. 2, restating [3] Sec. VII

## Objective

Build a detector that answers one question at every time step:

> Do the last $T$ residual samples still look like **independent, standard normal** noise?

A simple threshold on $|z|$ only catches residuals that get *bigger*. A careful
attacker can keep the size and spread of $z$ normal and still inject
correlation over time. This detector compares the joint distribution of
$L$ consecutive samples against the nominal one, so it also catches
changes in correlation.

Phase A tests the detector on **synthetic** residuals only, where the right
answer is known, before it is connected to the plant (Phase C).

## Symbols

| Symbol | Code name | Value | Meaning |
|---|---|---|---|
| $z(k)$ | `residual` | — | Normalised residual at time $k$. Nominally i.i.d. $\mathcal N(0,1)$. |
| $L$ | `BLOCK_LEN` | 3 | Block length: how many consecutive samples are tested together. |
| $T$ | `WINDOW_LEN` | 800 | Window length: how many recent blocks the statistic averages over. |
| $I$ | `N_CODEBOOK` | 100 | Number of codebook points (and degrees of freedom of the $\chi^2$ test). |
| $D$ | — | 1 | Residual dimension. Each output channel is tested separately. |
| $\alpha$ | `ALPHA` | 0.99 | Alarm threshold on $\psi$. Nominal false-alarm rate $\approx 1-\alpha$. |
| $\rho_m$ | `codebook[m]` | $(I, L)$ | Codebook point $m$: a representative block of $L$ values. |
| $\Phi$ | `norm.cdf` | — | Standard normal CDF. |
| $u^*$ | `u_star` | $(I,)$ | Nominal probability that a block lies below each $\rho_m$. |
| $u_{k,T}$ | `count / T` | $(I,)$ | Observed fraction of the last $T$ blocks below each $\rho_m$. |
| $\Sigma$ | `Sigma` | $(I, I)$ | Covariance of $\sqrt{T}(u_{k,T}-u^*)$ under nominal conditions. |
| $v_{k,T}$ | `v` | scalar | Test statistic. About $\chi^2_I$ when nominal, large when anomalous. |
| $\psi_{k,T}$ | `psi` | $[0, 1]$ | $\chi^2_I$ CDF of $v$: how extreme $v$ is. |

## Method

**1. Codebook** (`build_codebook`). Run Lloyd's algorithm (equivalent to k-means)
on samples of $\mathcal N(0, I_L)$ to place $I$ points $\rho_m$ where normal
blocks are likely to fall.

**2. Nominal CDF vector** (`nominal_cdf_vector`). For i.i.d. $\mathcal N(0,1)$ samples:

$$[u^*]_m = \Pr(\text{block} \le \rho_m) = \prod_{l=0}^{L-1} \Phi(\rho_{m,l})$$

**3. Covariance** (`nominal_covariance`). Consecutive blocks overlap in $L-1$
samples, so their indicators are correlated. [1] eq. (20) sums the
cross-covariances over lags $|k| < L$:

$$\Sigma = \sum_{k=-(L-1)}^{L-1} \big(\mathbb E[\xi_k \xi_0^\top] - u^* u^{*\top}\big), \qquad [\xi_k]_m = \mathbf 1\{\text{block}_k \le \rho_m\}$$

`nominal_covariance_mc` estimates the same matrix by simulation. It is used
only to check the closed form.

**4. Online detector** (`run_detector`). At each step $k$:

$$u_{k,T} = \frac1T \sum_{\text{last } T \text{ blocks}} \xi, \qquad
v_{k,T} = T\,(u_{k,T}-u^*)^\top \Sigma^{-1} (u_{k,T}-u^*), \qquad
\psi_{k,T} = F_{\chi^2_I}(v_{k,T})$$

and raises an alarm when $\psi_{k,T} \ge \alpha$. A ring buffer keeps $u_{k,T}$
up to date in $O(IL)$ per step. The output is `NaN` until the first full window
at $k = T + L - 2$.

## Test signals

| Function | What it generates | Why |
|---|---|---|
| `ar1_residual` | $z_t = \phi z_{t-1} + \sqrt{1-\phi^2}\,e_t$ | Still $\mathcal N(0,1)$ at each step but correlated over time. A threshold on $\lvert z \rvert$ cannot see it. |
| `intricate_attack` | Attack (27) of [1]: $r(k) = \bar v\, r(k-\bar\tau) + \sqrt{1-\bar v^2}\,z(k)$, $z_a = \gamma r$ | Stealthy attack from the papers. With $\gamma \in \{-1,+1\}$ it passes variance, autocorrelation and KS tests. |

## Reading the output

Values below are from a run with the default seed.

| Check | What is tested | Pass condition | Typical result |
|---|---|---|---|
| [1] Codebook | Shape is $(I, L)$; centred near 0 | shape `(100, 3)` | centre ≈ `[0, -0.003, 0.005]` |
| [2] $u^*$ | Entries in $(0,1)$, mean near $0.5^3$ | mean ∈ (0.08, 0.18) | 0.124 |
| [3] $\Sigma$ | Symmetric; lag-0 diagonal = $u^*(1-u^*)$; positive semi-definite; matches Monte Carlo | MC error < 3 % | 0.45 % |
| [4] Nominal input | $v$ should follow $\chi^2_{100}$ (mean 100, std 14.1) | mean $v$ ∈ (90, 115) | mean 99.0, std 16.4, alarm rate 3.2 % |
| [5] AR(1) after $k_0$ | Same variance, added correlation → $v$ must rise | mean $v$ after > 150 | $\phi$=0.3: 193, 0.5: 394, 0.8: 2181 |
| [6] Attack (27) | $\{-1,+1\}$ vs $\{0,1\}$ attacker, $T = 100$ | — | $v$: 96 → 125 and 96 → 1247 |

How to interpret the results:

- **Alarm rate 3.2 % instead of 1 %, std 16.4 instead of 14.1.** $v \sim \chi^2_I$ only holds as
  $T \to \infty$. With $T = 800$ the distribution has a slightly heavier tail. This is a known
  finite-window effect, not a bug.
- **Check 5.** The variance before and after the change is ≈ 1.00 in all cases,
  so an amplitude test would see nothing, yet $v$ grows with the correlation strength.
- **Check 6.** The $\{-1,+1\}$ attack has variance 1.01, lag-1 autocorrelation −0.003 and
  KS statistic 0.003. It looks perfectly normal to simple tests, but $v$ still rises.
  The $\{0,1\}$ version printed in the papers zeroes half the samples and is trivially detectable.

## Output file: `detector_stats.npz`

Load with `phase_a.load_stats(path)`, which returns a dict.

| Key | Shape | Meaning |
|---|---|---|
| `codebook` | (100, 3) | $\rho_m$ |
| `u_star` | (100,) | $u^*$ |
| `Sigma` | (100, 100) | $\Sigma$ |
| `Sigma_inv` | (100, 100) | Pseudo-inverse of $\Sigma$ used in $v$ |
| `L`, `T`, `I` | scalar | Parameters the stats were built with |

Pass the dict to `run_detector(residual, stats)` to run the detector on any residual sequence.
