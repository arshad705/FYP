"""
phase_a.py — Phase A: statistical anomaly detector, tested on synthetic residuals.

Implements [1] Sec. III / Alg. 2 (Yan, Fu & Seron, ICCA 2024), which restates
[3] Sec. VII (Marelli, Sui, Fu & Lu, IEEE TAC 2021). Validated against known
synthetic inputs before being connected to the plant in Phase C.

Pipeline
    1. Codebook     rho_m in R^L via Lloyd's algorithm    [1] Alg. 2 init
    2. Nominal CDF  [u*]_m = prod_l Phi(rho_{m,l})         [1] eq. (19)
    3. Covariance   Sigma = sum_{|k|<L} Sigma(k)           [1] eq. (20)
    4. Detector     u, v, psi, alarm                        [1] Alg. 2 steps 3-7

Parameters [1] Sec. V: L = 3, T = 800, I = 100, D = 1.
Output: detector_stats.npz (codebook, u_star, Sigma, Sigma_inv, L, T, I).
"""

import numpy as np
from pathlib import Path
from scipy.stats import norm, chi2, kstest
from sklearn.cluster import KMeans

BLOCK_LEN:   int   = 3       # L
WINDOW_LEN:  int   = 800     # T
N_CODEBOOK:  int   = 100     # I
ALPHA:       float = 0.99    # alarm if psi >= ALPHA (false-alarm rate ~ 1 - ALPHA)
SEED:        int   = 20260906
N_TRAIN:     int   = 100_000 # Lloyd training samples

RESULTS_DIR = Path(__file__).parent


# Step 1 — Codebook  [1] Alg. 2 init
def build_codebook(
    n_points: int = N_CODEBOOK,
    block_len: int = BLOCK_LEN,
    seed: int = SEED,
    n_train: int = N_TRAIN,
) -> np.ndarray:
    """Lloyd's algorithm (= k-means) on samples of N(0, I_L). Returns (I, L); row m is rho_m."""
    rng = np.random.default_rng(seed)
    training_samples = rng.standard_normal((n_train, block_len))
    km = KMeans(n_clusters=n_points, n_init=10, random_state=seed).fit(training_samples)
    return np.asarray(km.cluster_centers_, dtype=float)


# Step 2 — Nominal CDF vector  [1] eq. (19)
def nominal_cdf_vector(codebook: np.ndarray) -> np.ndarray:
    """[u*]_m = prod_l Phi(rho_{m,l}) = P(block of L i.i.d. N(0,1) samples <= rho_m)."""
    return norm.cdf(codebook).prod(axis=1)


# Step 3 — Covariance  [1] eq. (20), Prop. 3.1
def nominal_covariance(codebook: np.ndarray) -> np.ndarray:
    """
    Sigma = sum_{|lag|<L} (E[xi_lag xi_0^T] - u* u*^T),  [xi_k]_m = 1{samples k..k+L-1 <= rho_m}.

    E[[xi_lag]_m [xi_0]_n] is a product over each sample t covered by either block:
        in both blocks   -> Phi(min(rho_{m,t-lag}, rho_{n,t}))
        block 0 only     -> Phi(rho_{n,t})
        block lag only   -> Phi(rho_{m,t-lag})
    This merges the three cases of [1] eq. (20) into one rule. Returns (I, I).
    """
    n_points, block_len = codebook.shape
    cdf = norm.cdf(codebook)                      # Phi(rho_{m,l}), (I, L)
    u_star = cdf.prod(axis=1)
    Sigma = np.zeros((n_points, n_points))
    for lag in range(-(block_len - 1), block_len):
        joint = np.ones((n_points, n_points))     # joint[m, n] = E[[xi_lag]_m [xi_0]_n]
        for t in range(min(0, lag), max(block_len, block_len + lag)):
            in_block0 = 0 <= t < block_len
            in_blocklag = 0 <= t - lag < block_len
            if in_block0 and in_blocklag:
                bound = np.minimum(codebook[:, t - lag][:, None], codebook[:, t][None, :])
                joint *= norm.cdf(bound)
            elif in_block0:
                joint *= cdf[:, t][None, :]
            else:
                joint *= cdf[:, t - lag][:, None]
        Sigma += joint - np.outer(u_star, u_star)
    return Sigma


def nominal_covariance_mc(
    codebook: np.ndarray,
    n_samples: int = 600_000,
    seed: int = 1,
) -> np.ndarray:
    """Monte Carlo Sigma, used only to cross-check nominal_covariance() (should agree within a few %)."""
    n_points, block_len = codebook.shape
    residual = np.random.default_rng(seed).standard_normal(n_samples)
    blocks = np.lib.stride_tricks.sliding_window_view(residual, block_len)
    xi = np.empty((len(blocks), n_points), dtype=np.float32)
    for a in range(0, len(blocks), 50_000):
        xi[a:a + 50_000] = np.all(blocks[a:a + 50_000, None, :] <= codebook[None], axis=2)
    xi -= xi.mean(axis=0)
    n_blocks = len(xi)
    Sigma_mc = np.zeros((n_points, n_points))
    for lag in range(block_len):
        cov = (xi[lag:].T @ xi[:n_blocks - lag]) / n_blocks
        Sigma_mc += cov if lag == 0 else cov + cov.T
    return Sigma_mc


# Step 4 — Online detector  [1] Alg. 2 steps 3-7
def run_detector(
    residual: np.ndarray,
    stats: dict,
    window_len: int = WINDOW_LEN,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    At each k, using the newest block residual[k-L+1..k] (causal):
        u     = fraction of the last T blocks with block <= rho_m
        v     = T (u - u*)^T Sigma^{-1} (u - u*)
        psi   = chi-square CDF of v with I d.o.f.;  alarm = psi >= alpha

    u is updated with a ring buffer, O(I L) per step instead of O(T I L).
    stats needs 'codebook', 'u_star', 'Sigma_inv'.
    Returns v, psi, alarm, each NaN/False until k = T + L - 2.
    """
    codebook, u_star, Sigma_inv = stats["codebook"], stats["u_star"], stats["Sigma_inv"]
    n_points, block_len = codebook.shape
    n = len(residual)
    v = np.full(n, np.nan)
    ring = np.zeros((window_len, n_points), dtype=np.int32)
    count = np.zeros(n_points, dtype=np.int64)
    for k in range(block_len - 1, n):
        block = residual[k - block_len + 1: k + 1]
        indicator = np.all(block[None, :] <= codebook, axis=1).astype(np.int32)
        slot = (k - block_len + 1) % window_len
        count += indicator - ring[slot]
        ring[slot] = indicator
        if k >= window_len + block_len - 2:
            d = count / window_len - u_star
            v[k] = window_len * d @ Sigma_inv @ d
    psi = chi2.cdf(v, df=n_points)
    return v, psi, psi >= alpha


def save_stats(path: Path, codebook, u_star, Sigma, Sigma_inv) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, codebook=codebook, u_star=u_star, Sigma=Sigma, Sigma_inv=Sigma_inv,
             L=BLOCK_LEN, T=WINDOW_LEN, I=N_CODEBOOK)


def load_stats(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# Synthetic residuals for validation
def ar1_residual(n: int, phi: float, rng) -> np.ndarray:
    """Unit-variance AR(1): N(0,1) marginal but correlated, so a simple threshold can't see it."""
    z = rng.standard_normal(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
    return z


def intricate_attack(nominal: np.ndarray, v_bar: float, tau_bar: int, rng, sign_set: str = "pm1") -> np.ndarray:
    """
    Attack (27) of [1] / (15) of [3]:  r(k) = v_bar r(k-tau_bar) + sqrt(1-v_bar^2) z(k),  z_a = gamma r.
    sign_set 'pm1': gamma in {-1,+1}, E[gamma] = 0 as [3]'s proof needs.
    sign_set '01':  gamma in {0,1} as printed in the papers; not stealthy.
    """
    n = len(nominal)
    r = np.zeros(n)
    r[:tau_bar] = rng.standard_normal(tau_bar)
    for k in range(tau_bar, n):
        r[k] = v_bar * r[k - tau_bar] + np.sqrt(1 - v_bar ** 2) * nominal[k]
    gamma = rng.choice([-1.0, 1.0], n) if sign_set == "pm1" else rng.integers(0, 2, n).astype(float)
    return gamma * r


if __name__ == "__main__":
    print("=" * 65)
    print("Phase A Validation — Statistical Detector (L=3, T=800, I=100)")
    print("=" * 65)

    print("\n[1] Codebook (Lloyd's algorithm)")
    codebook = build_codebook()
    print(f"  shape = {codebook.shape}, centre of mass = {np.round(codebook.mean(axis=0), 3)}  "
          f"[{'OK' if codebook.shape == (N_CODEBOOK, BLOCK_LEN) else 'WRONG'}]")

    print("\n[2] Nominal CDF vector u*")
    u_star = nominal_cdf_vector(codebook)
    in_range = bool(np.all((u_star > 0) & (u_star < 1)))
    print(f"  all entries in (0,1): {in_range}   mean = {u_star.mean():.3f}  (theory 0.5^3 = 0.125)  "
          f"[{'OK' if in_range and 0.08 < u_star.mean() < 0.18 else 'WRONG'}]")

    # Check 3: symmetry, Bernoulli diagonal, PSD, Monte Carlo agreement
    print("\n[3] Covariance Sigma")
    Sigma = nominal_covariance(codebook)
    sym = bool(np.allclose(Sigma, Sigma.T, atol=1e-12))
    lag0 = np.ones_like(Sigma)
    for t in range(BLOCK_LEN):
        lag0 *= norm.cdf(np.minimum(codebook[:, t][:, None], codebook[:, t][None, :]))
    lag0 -= np.outer(u_star, u_star)
    bern = float(np.abs(np.diag(lag0) - u_star * (1 - u_star)).max())
    eig = np.linalg.eigvalsh(Sigma)
    Sigma_mc = nominal_covariance_mc(codebook)
    rel_err = float(np.linalg.norm(Sigma - Sigma_mc) / np.linalg.norm(Sigma))
    print(f"  symmetric: {sym}  [{'OK' if sym else 'FAIL'}]")
    print(f"  lag-0 diagonal vs u*(1-u*): max diff = {bern:.1e}  [{'OK' if bern < 1e-12 else 'FAIL'}]")
    print(f"  eigenvalues: min = {eig.min():.2e}, cond = {eig.max()/eig.min():.1e}  [{'OK' if eig.min() > -1e-10 else 'FAIL'}]")
    print(f"  closed form vs Monte Carlo: rel. error = {rel_err:.4f}  [{'OK' if rel_err < 0.03 else 'FAIL — do not proceed'}]")
    Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-10, hermitian=True)
    stats = dict(codebook=codebook, u_star=u_star, Sigma_inv=Sigma_inv)

    # Check 4: nominal input -> v ~ chi2_I
    print("\n[4] Detector on i.i.d. N(0,1) residual (100 000 steps)")
    rng = np.random.default_rng(1)
    v, psi, alarm = run_detector(rng.standard_normal(100_000), stats)
    ok = ~np.isnan(v)
    far = float(alarm[ok][::WINDOW_LEN].mean())
    print(f"  first valid k = {int(np.argmax(ok))}  (expected T+L-2 = {WINDOW_LEN + BLOCK_LEN - 2})")
    print(f"  mean v = {v[ok].mean():.1f}  (chi2_100 mean = 100)   std v = {v[ok].std():.1f}  (chi2_100 std = 14.1)")
    print(f"  alarm rate at alpha = {ALPHA}: {far:.3f}  (asymptotic 1-alpha = {1-ALPHA:.2f}; a few % is the finite-T effect)"
          f"  [{'OK' if 90 < v[ok].mean() < 115 else 'FAIL'}]")

    # Check 5: same marginal but correlated -> v should rise
    print("\n[5] Detector on AR(1) residual after k0 = 50 000 (N(0,1) marginal, not i.i.d.)")
    n, k0 = 100_000, 50_000
    for phi in [0.3, 0.5, 0.8]:
        z = rng.standard_normal(n)
        z[k0:] = ar1_residual(n - k0, phi, rng)
        v, psi, alarm = run_detector(z, stats)
        after = np.arange(n) >= k0 + WINDOW_LEN
        print(f"  phi = {phi}: v before {np.nanmean(v[:k0]):6.1f}, after {np.nanmean(v[after]):7.1f}; "
              f"alarm rate after {alarm[after].mean():.2f}; variance before/after {z[:k0].var():.2f}/{z[k0:].var():.2f}"
              f"  [{'OK' if np.nanmean(v[after]) > 150 else 'FAIL'}]")

    # Check 6: attack (27) with gamma in {-1,+1} vs {0,1}
    print("\n[6] Intricate attack (27), tau_bar = 1, v_bar = 1/sqrt(2), T = 100 as in [3] Sec. VIII")
    n, k0 = 50_000, 25_000
    nominal = rng.standard_normal(n)
    for sign_set in ["pm1", "01"]:
        z = nominal.copy()
        z[k0:] = intricate_attack(nominal, 1 / np.sqrt(2), 1, np.random.default_rng(3), sign_set)[k0:]
        v, psi, alarm = run_detector(z, stats, window_len=100)
        post = z[k0:]
        print(f"  gamma in {'{-1,+1}' if sign_set == 'pm1' else '{0,1}  '}: v before {np.nanmean(v[:k0]):6.1f}, after {np.nanmean(v[k0+100:]):7.1f} | "
              f"post-attack var {post.var():.2f}, zeros {(post == 0).mean():.2f}, "
              f"lag-1 acf {np.corrcoef(post[:-1], post[1:])[0, 1]:+.3f}, KS vs N(0,1) {kstest(post, 'norm').statistic:.3f}")
    print("  -> {-1,+1} passes every simple check and is still detected; {0,1} is not stealthy at all.")

    out = RESULTS_DIR / "detector_stats.npz"
    save_stats(out, codebook, u_star, Sigma, Sigma_inv)
    print(f"\nStats saved -> {out}")
    print("\n[Phase A validation complete]")