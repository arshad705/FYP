"""
phase_c.py — Phase C: connect Phase B (plant + estimator) to Phase A (detector).

Reproduces [1] Sec. V, Figs. 2 and 3 (Yan, Fu & Seron, ICCA 2024):
HyCon2 Scenario 3, fault H5: 10 -> 2 at k_bar = 5e4, 1e5 steps.

Pipeline
    1. Load         detector_stats.npz (Phase A), estimator_gains.npz (Phase B)
    2. Simulate     plant [1] eq. (30); area 5 uses post-fault matrices for k > k_bar ([1] Prop. 4.1)
    3. Estimate     distributed estimator [1] eq. (4), nominal (pre-fault) model throughout
    4. Normalise    z_check_i = Gamma_i^{-1/2} (z_i - C x_hat_i)       [1] eq. (10)
    5. Detect       v_{k,T} per area and per output channel             [1] Alg. 2, D = 1
    6. Plot         Fig. 2 (v_{k,T}) and Fig. 3 (|z_check| vs threshold)

Assumptions not stated in [1] (to confirm with supervisor):
    - u_i = 0: [1] eqs. (1), (4), (21) have no input term, and no load profile covers 1e5 steps.
    - Discretisation: Phase B's per-area ZOH; post-fault area 5 re-discretised on its own.
    - Fig. 3 baseline: |z_check_{i,l}| against a constant alpha_1 = 4.57 ([1] Fig. 3 caption),
      with [2]'s 1% value 2.57 shown dashed.
    - x(0) = x_hat(0) = 0 as in [2] Sec. V (residual variance settles within 3 steps).

Run phase_a.py and phase_b.py first.
Output: phase_c_results.npz, fig2_statistical.png, fig3_residual.png.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_a import run_detector, load_stats, WINDOW_LEN, BLOCK_LEN, ALPHA
from phase_b import AREAS, NEIGHBOURS, FAULT_K

N_STEPS:    int   = 100_000      # [1] Fig. 2 x-axis runs to 10 x 10^4
SEED:       int   = 20260926
ALPHA_1:    float = 4.57         # [1] Fig. 3 caption
ALPHA_BOEM: float = 2.57         # [2] Sec. V, 1% false-alarm threshold

RESULTS_DIR = Path(__file__).parent
FIRST_V = WINDOW_LEN + BLOCK_LEN - 2         # first k with a valid v_{k,T} (Phase A check [4])


# Step 1 — Load Phase B output back into the dict layout of phase_b.py
def load_gains(path: Path) -> dict:
    """Returns A[i][j], A_fault[i][j], L[i][j], Gamma_inv_sqrt[i] for j in N_i U {i}, plus C, Q, R."""
    d = np.load(path)
    A, L, G = {}, {}, {}
    for i in AREAS:
        A[i] = {j: d[f"A_{i}_{j}"] for j in [i] + NEIGHBOURS[i]}
        L[i] = {j: d[f"L_{i}_{j}"] for j in [i] + NEIGHBOURS[i]}
        G[i] = d[f"Gamma_inv_sqrt_{i}"]
    A_fault = {i: dict(A[i]) for i in AREAS}
    for j in [5] + NEIGHBOURS[5]:                     # H5 appears only in area 5's row (Phase B check [3])
        A_fault[5][j] = d[f"Afault_5_{j}"]
    return dict(A=A, A_fault=A_fault, L=L, Gamma_inv_sqrt=G, C=d["C"], Q=d["Q"], R=d["R"])


# Steps 2-4 — Simulate plant and estimators, return normalised residuals
def simulate(gains: dict, n_steps: int = N_STEPS, fault_k: int = FAULT_K, seed: int = SEED) -> dict:
    """
    At each k, in this order:
        y_i(k)       = C x_i(k) + v_i(k)                                   [1] eq. (30)
        z~_i(k)      = y_i(k) - C x_hat_i(k)                               local residual, z_i = y_i
        z_check_i(k) = Gamma_i^{-1/2} z~_i(k)                              [1] eq. (10)
        x_hat_i(k+1) = sum_{j in N_i U {i}} A_ij x_hat_j(k) + L_ij z~_j(k)   [1] eq. (4), nominal A
        x_i(k+1)     = sum_{j in N_i U {i}} A_ij(k) x_j(k) + w_i(k)          A_ij(k) post-fault for k > fault_k
    The estimator keeps the pre-fault model: that mismatch is the anomaly.
    Returns z_check[i], shape (n_steps, 2), for each area.
    """
    A, A_fault, L = gains["A"], gains["A_fault"], gains["L"]
    C, G = gains["C"], gains["Gamma_inv_sqrt"]
    n_out, n_state = C.shape

    rng = np.random.default_rng(seed)
    w_std = np.sqrt(gains["Q"][0, 0])                           # Q_i = 1e-6 I  ->  std 1e-3
    v_std = np.sqrt(gains["R"][0, 0])                           # R_i = 1e-6 I
    w = {i: w_std * rng.standard_normal((n_steps, n_state)) for i in AREAS}
    v = {i: v_std * rng.standard_normal((n_steps, n_out)) for i in AREAS}

    x = {i: np.zeros(n_state) for i in AREAS}
    x_hat = {i: np.zeros(n_state) for i in AREAS}
    z_check = {i: np.zeros((n_steps, n_out)) for i in AREAS}

    for k in range(n_steps):
        A_plant = A_fault if k > fault_k else A

        # residuals of every area first: each estimator update needs its neighbours' residuals
        resid = {}
        for i in AREAS:
            y = C @ x[i] + v[i][k]
            resid[i] = y - C @ x_hat[i]
            z_check[i][k] = G[i] @ resid[i]

        new_x_hat, new_x = {}, {}
        for i in AREAS:
            est = np.zeros(n_state)                             # [1] eq. (4)
            plant = w[i][k].copy()                              # [1] eq. (30), u_i = 0
            for j in [i] + NEIGHBOURS[i]:
                est = est + A[i][j] @ x_hat[j] + L[i][j] @ resid[j]
                plant = plant + A_plant[i][j] @ x[j]
            new_x_hat[i] = est
            new_x[i] = plant
        x, x_hat = new_x, new_x_hat

    return z_check


# Step 5 — Detector per area and per output channel  [1] Alg. 2 with D = 1
def detect_all(z_check: dict, stats: dict) -> dict:
    """Phase A's detector on each channel separately. Returns dicts keyed (area, channel), channel = 1, 2."""
    out = {"v": {}, "psi": {}, "alarm": {}}
    for i in AREAS:
        for l in [1, 2]:
            v, psi, alarm = run_detector(z_check[i][:, l - 1], stats)
            out["v"][(i, l)] = v
            out["psi"][(i, l)] = psi
            out["alarm"][(i, l)] = alarm
    return out


# Step 6 — Figures  [1] Figs. 2 and 3
def plot_fig2(v: dict, path: Path, fault_k: int = FAULT_K) -> None:
    """v_{k,T} per area (rows) and channel (columns), log y-axis as in [1] Fig. 2."""
    fig, axes = plt.subplots(len(AREAS), 2, figsize=(10, 9), sharex=True)
    for r, i in enumerate(AREAS):
        for l in [1, 2]:
            ax = axes[r, l - 1]
            ax.plot(np.arange(len(v[(i, l)])), v[(i, l)], lw=0.6)
            ax.axhline(100, color="grey", ls=":", lw=1)           # chi2_100 mean, for reference
            ax.axvline(fault_k, color="red", ls="--", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(40, 1000)                                 # same range on every panel
            ax.set_ylabel(f"$v_{{k,T}}({i},{l})$")
    for ax in axes[-1]:
        ax.set_xlabel("Time step $k$")
    fig.suptitle("Statistical approach: $v_{k,T}$, fault $H_5$: 10 → 2 at $k$ = 5×10⁴  ([1] Fig. 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fig3(z_check: dict, path: Path, fault_k: int = FAULT_K) -> None:
    """|z_check_{i,l}| with alpha_1 = 4.57 (red) and 2.57 (dashed), log y-axis as in [1] Fig. 3."""
    fig, axes = plt.subplots(len(AREAS), 2, figsize=(10, 9), sharex=True)
    for r, i in enumerate(AREAS):
        for l in [1, 2]:
            ax = axes[r, l - 1]
            ax.plot(np.arange(z_check[i].shape[0]), np.abs(z_check[i][:, l - 1]), lw=0.3)
            ax.axhline(ALPHA_1, color="red", lw=1.2, label=f"$\\alpha_1$ = {ALPHA_1}")
            ax.axhline(ALPHA_BOEM, color="red", ls="--", lw=0.8, label=f"{ALPHA_BOEM} ([2], 1%)")
            ax.axvline(fault_k, color="black", ls=":", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 20)
            ax.set_ylabel(f"|ž$_{{{i}{l}}}$|")
    axes[0, 1].legend(loc="lower right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time step $k$")
    fig.suptitle("Threshold-based residual approach: |ž|  ([1] Fig. 3)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def first_alarm_after(alarm: np.ndarray, start: int) -> int | None:
    """Index of the first alarm at or after start, or None."""
    hits = np.flatnonzero(alarm[start:])
    return int(start + hits[0]) if len(hits) else None


if __name__ == "__main__":
    print("=" * 65)
    print("Phase C Validation — Power network + statistical detector ([1] Sec. V)")
    print("=" * 65)

    stats_path, gains_path = RESULTS_DIR / "detector_stats.npz", RESULTS_DIR / "estimator_gains.npz"
    for p, script in [(stats_path, "phase_a.py"), (gains_path, "phase_b.py")]:
        if not p.exists():
            raise SystemExit(f"Missing {p.name}: run {script} first.")
    stats, gains = load_stats(stats_path), load_gains(gains_path)
    if int(stats["T"]) != WINDOW_LEN or int(stats["L"]) != BLOCK_LEN:
        raise SystemExit("detector_stats.npz was built with different T or L; re-run phase_a.py.")

    print(f"\n[1] Simulating {N_STEPS} steps, fault for k > {FAULT_K}, u_i = 0")
    z_check = simulate(gains)
    pre = slice(FIRST_V, FAULT_K + 1)                   # healthy data only
    post = slice(FAULT_K + 1, N_STEPS)

    # Check 2: healthy z_check must be white with identity covariance, or eq. (11) and the test don't apply
    print("\n[2] Normalised residual before the fault (expected: covariance I, lag-1 acf 0)")
    for i in AREAS:
        z = z_check[i][pre]
        cov = np.cov(z.T)
        acf = [np.corrcoef(z[:-1, l], z[1:, l])[0, 1] for l in [0, 1]]
        ok = np.abs(cov - np.eye(2)).max() < 0.05 and max(abs(a) for a in acf) < 0.05
        print(f"  area {i}: cov = [[{cov[0,0]:.3f}, {cov[0,1]:+.3f}], [{cov[1,0]:+.3f}, {cov[1,1]:.3f}]], "
              f"lag-1 acf ch1 {acf[0]:+.3f}, ch2 {acf[1]:+.3f}  [{'OK' if ok else 'FAIL'}]")

    print(f"\n[3] Detector per channel (D = 1, L = {BLOCK_LEN}, T = {WINDOW_LEN}, I = {stats['codebook'].shape[0]})")
    det = detect_all(z_check, stats)
    for i in AREAS:
        for l in [1, 2]:
            v = det["v"][(i, l)]
            v_pre = np.nanmean(v[pre])
            v_post = np.nanmean(v[FAULT_K + FIRST_V:])          # window holds post-fault data only
            far = det["alarm"][(i, l)][pre].mean()
            print(f"  area {i}, ch {l}: mean v before {v_pre:6.1f}, after {v_post:6.1f} (ratio {v_post / v_pre:4.2f}), "
                  f"alarm rate before {far:.3f}")
    print(f"  Expected: v ~ 100 when healthy (chi2_100 mean); alarm rate ~ 1-alpha = {1 - ALPHA:.2f} plus the finite-T effect")
    print("  (Phase A check [4]); only area 5 changes after the fault.")

    # Check 4: a single alarm can be chance, so require the alarm to stay on
    print(f"\n[4] Detection of the fault in area 5, psi >= {ALPHA}")
    for l in [1, 2]:
        alarm = det["alarm"][(5, l)]
        k_hit = first_alarm_after(alarm, FAULT_K + 1)
        on_rate = alarm[FAULT_K + FIRST_V:].mean()
        verdict = "DETECTED" if on_rate > 0.5 else "not reliably detected"
        print(f"  ch {l}: first alarm " + (f"at k = {k_hit} (+{k_hit - FAULT_K})" if k_hit is not None else "none")
              + f", alarm on {on_rate:.2f} of the time once the window is post-fault  [{verdict}]")

    print("\n[5] Threshold baseline ([1] Fig. 3): fraction of steps with |z_check| > threshold")
    for i in AREAS:
        for l in [1, 2]:
            z = np.abs(z_check[i][:, l - 1])
            print(f"  area {i}, ch {l}: > {ALPHA_BOEM}: before {np.mean(z[pre] > ALPHA_BOEM):.4f}, after {np.mean(z[post] > ALPHA_BOEM):.4f}"
                  f"  |  > {ALPHA_1}: before {np.mean(z[pre] > ALPHA_1):.5f}, after {np.mean(z[post] > ALPHA_1):.5f}")
    print("  Theory for N(0,1): P(|z| > 2.57) = 0.0102, P(|z| > 4.57) = 4.9e-6.")

    fig2, fig3 = RESULTS_DIR / "fig2_statistical.png", RESULTS_DIR / "fig3_residual.png"
    plot_fig2(det["v"], fig2)
    plot_fig3(z_check, fig3)
    np.savez(RESULTS_DIR / "phase_c_results.npz",
             **{f"zcheck_{i}": z_check[i] for i in AREAS},
             **{f"v_{i}_{l}": det["v"][(i, l)] for i in AREAS for l in [1, 2]},
             fault_k=FAULT_K, n_steps=N_STEPS, seed=SEED)
    print(f"\nFigures saved -> {fig2.name}, {fig3.name}")
    print("\n[Phase C validation complete]")