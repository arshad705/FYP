"""
phase_c_attacks.py — Phase C, attack scenarios: residual-level experiment with [1]'s attack constructions.

[1] Sec. V says simulations of attack (27) and of replay attacks give results "akin to" Figs. 2 and 3
(statistical test detects, threshold on |z_check| does not), but gives no details.  This script applies
the paper's two constructions to residuals and runs both detectors.

Pipeline
    1. Simulate     healthy network (no fault), same plant and estimator as phase_c.py, u_i = 0
    2. Attack       replace area 5's STORED normalised residual for k >= k0                  [1] Sec. IV
                      intricate: r(k) = v_bar r(k - tau_bar) + sqrt(1 - v_bar^2) z(k), z_a = gamma r   [1] eq. (27)
                      replay:    z_check(k) = z_check(k - T_bar) for k >= k0, periodic          [1] Remark 4.2
    3. Detect       v_{k,T} per area and channel, as in phase_c.py                          [1] Alg. 2, D = 1
    4. Plot         Fig. 2 / Fig. 3 equivalents for each attack

Scope: a residual-level experiment. The estimators are never fed attacked measurements ([1] Sec. II defines
attacks on the received measurements; (27) and Remark 4.2 describe the resulting residual). Areas 1-3 are
unchanged because their residual streams are untouched, which says nothing about how a measurement attack
would propagate through the distributed estimators.

Assumptions not stated in [1] (to confirm with supervisor):
    - Attacked area 5, both channels, from k0 = 5e4 (same area and time as the H5 fault); no fault in these runs.
    - tau_bar = 1, v_bar = 1/sqrt(2) from [3] Sec. VIII.
    - gamma in {-1,+1}: our reading of the printed model, since (28) needs E[gamma] = 0 ([3] Sec. VI-A).
      gamma in {0,1}, as printed, is run for comparison (Phase A check [6]).
    - Replay recording: the last T_bar healthy residuals before k0, repeated; T_bar = 400 for the figures,
      T_bar in {100, 400, 800, 1600} in the sweep, all with this one recording position.

Detector (Phase A): L = 3 sets the time span inside each tested block (dependence at lags <= 2); T = 800 sets
how many block indicators are averaged. Neither gives a replay-period boundary: check [5] reports one
recording per period, and check [6] shows independent recordings on either side of the threshold.
One run's alarm share is a fraction of overlapping windows, not a detection probability.

Seeds: SEED (phase_c.py) for the network, ATTACK_SEED for gamma and r(0); the codebook ("checkpoint bank")
is identified by the SHA-256 printed at the start and saved in the .npz.

Run phase_a.py and phase_b.py first.  Usage: python phase_c_attacks.py
Output: fig2_attack_<name>.png, fig3_attack_<name>.png, phase_c_attacks_results.npz
"""

import hashlib

import numpy as np
from scipy.stats import chi2, kstest

from phase_a import run_detector, load_stats, intricate_attack, WINDOW_LEN, BLOCK_LEN, ALPHA
from phase_b import AREAS
from phase_c import (load_gains, simulate, detect_all, plot_fig2, plot_fig3, first_alarm_after,
                     N_STEPS, SEED, ALPHA_1, ALPHA_BOEM, RESULTS_DIR, FIRST_V)

ATTACK_K:     int   = 50_000                 # k0, same as the fault time in phase_c.py
ATTACK_AREA:  int   = 5
TAU_BAR:      int   = 1                      # [3] Sec. VIII
V_BAR:        float = 1 / np.sqrt(2)         # [3] Sec. VIII
REPLAY_T:     int   = 400                    # replay period for the figures
REPLAY_SWEEP: list[int] = [100, 400, 800, 1600]
ATTACK_SEED:  int   = 27
REPLAY_CHECKS: list[tuple[int, int]] = [(800, 9), (400, 60)]   # (period, recording seed) cited in review


# Step 2 — Attacks on the stored normalised residual of one area
def apply_intricate(z_check: dict, sign_set: str, k0: int = ATTACK_K, area: int = ATTACK_AREA,
                    seed: int = ATTACK_SEED) -> dict:
    """Attack (27) on each channel of area's z_check for k >= k0, driven by the healthy residual z(k)."""
    rng = np.random.default_rng(seed)
    out = {i: z.copy() for i, z in z_check.items()}
    for l in range(out[area].shape[1]):
        out[area][k0:, l] = intricate_attack(z_check[area][k0:, l], V_BAR, TAU_BAR, rng, sign_set)
    return out


def apply_replay(z_check: dict, period: int, k0: int = ATTACK_K, area: int = ATTACK_AREA) -> dict:
    """Replay: the last `period` healthy residuals before k0 are repeated, z(k) = z(k - period) for k >= k0."""
    out = {i: z.copy() for i, z in z_check.items()}
    n = out[area].shape[0]
    record = z_check[area][k0 - period:k0]
    out[area][k0:] = np.resize(record, (n - k0, record.shape[1]))
    return out


def replay_score(period: int, seed: int, stats: dict) -> tuple[float, bool]:
    """
    v_{k,T} and alarm at the last step of a periodic replay of the Gaussian recording
    default_rng(seed).standard_normal(period), repeated for at least 2T samples. When T is a multiple of
    the period, every fully replayed window holds the same blocks, so this is v for the whole replay.
    """
    record = np.random.default_rng(seed).standard_normal(period)
    v, psi, alarm = run_detector(np.tile(record, -(-2 * WINDOW_LEN // period)), stats)
    return float(v[-1]), bool(alarm[-1])


def residual_stats(z: np.ndarray) -> tuple[float, float, float, float]:
    """Variance, lag-1 autocorrelation of z and of z^2, KS distance to N(0,1)."""
    return (z.var(), np.corrcoef(z[:-1], z[1:])[0, 1], np.corrcoef(z[:-1] ** 2, z[1:] ** 2)[0, 1],
            kstest(z, "norm").statistic)


def report(z_check: dict, det: dict, k0: int = ATTACK_K) -> None:
    """Checks [2]-[4] for one attacked run (a single realisation)."""
    pre, post = slice(FIRST_V, k0), slice(k0, N_STEPS)
    full = slice(k0 + FIRST_V, N_STEPS)                       # window holds attacked data only

    print(f"\n  Residual statistics, area {ATTACK_AREA}: var, lag-1 acf, lag-1 acf of z^2, KS distance to N(0,1)")
    for l in [1, 2]:
        after, before = residual_stats(z_check[ATTACK_AREA][post, l - 1]), residual_stats(z_check[ATTACK_AREA][pre, l - 1])
        print(f"    ch {l}: after k0  {after[0]:.3f}, {after[1]:+.3f}, {after[2]:+.3f}, {after[3]:.4f}"
              f"  |  before k0 (healthy)  {before[0]:.3f}, {before[1]:+.3f}, {before[2]:+.3f}, {before[3]:.4f}")
    print("    Reported values, not test results: attacked samples are dependent, so i.i.d. significance levels don't apply.")

    print("  Statistical test, this run: mean v before -> after; alarm share = fraction of overlapping windows in alarm after k0")
    for i in AREAS:
        for l in [1, 2]:
            v, alarm = det["v"][(i, l)], det["alarm"][(i, l)]
            v_pre, v_post = np.nanmean(v[pre]), np.nanmean(v[full])
            share = alarm[full].mean()
            if i == ATTACK_AREA:
                k_hit = first_alarm_after(alarm, k0)
                tag = f", first alarm +{k_hit - k0}" if k_hit is not None else ", no alarm"
            else:
                tag = "  (residual untouched)"
            print(f"    area {i}, ch {l}: {v_pre:6.1f} -> {v_post:7.1f} (ratio {v_post / v_pre:5.2f}), "
                  f"alarm share {share:.2f}{tag}")
    print("  An alarm share from one run is not a detection probability: consecutive windows share T - 1 blocks.")

    print(f"  Threshold baseline, area {ATTACK_AREA}: fraction of steps with |z_check| > threshold, before -> after")
    for l in [1, 2]:
        z = np.abs(z_check[ATTACK_AREA][:, l - 1])
        print(f"    ch {l}: > {ALPHA_BOEM}: {np.mean(z[pre] > ALPHA_BOEM):.4f} -> {np.mean(z[post] > ALPHA_BOEM):.4f}"
              f"  |  > {ALPHA_1}: {np.mean(z[pre] > ALPHA_1):.5f} -> {np.mean(z[post] > ALPHA_1):.5f}")


if __name__ == "__main__":
    print("=" * 65)
    print("Phase C Attacks — residual-level experiment, [1]'s intricate (27) and replay constructions")
    print("=" * 65)
    print("Area 5's stored residual is replaced after a healthy simulation; the estimators never see attacked data.")

    stats_path, gains_path = RESULTS_DIR / "detector_stats.npz", RESULTS_DIR / "estimator_gains.npz"
    for p, script in [(stats_path, "phase_a.py"), (gains_path, "phase_b.py")]:
        if not p.exists():
            raise SystemExit(f"Missing {p.name}: run {script} first.")
    stats, gains = load_stats(stats_path), load_gains(gains_path)
    if int(stats["T"]) != WINDOW_LEN or int(stats["L"]) != BLOCK_LEN:
        raise SystemExit("detector_stats.npz was built with different T or L; re-run phase_a.py.")
    codebook_sha = hashlib.sha256(np.ascontiguousarray(stats["codebook"]).tobytes()).hexdigest()
    threshold = chi2.ppf(ALPHA, stats["codebook"].shape[0])             # psi >= alpha  <=>  v >= threshold
    print(f"Codebook (checkpoint bank): I = {stats['codebook'].shape[0]}, L = {BLOCK_LEN}, SHA-256 {codebook_sha}")
    print(f"T = {WINDOW_LEN}, alpha = {ALPHA} (alarm when v >= {threshold:.2f}); seeds: network {SEED}, attack {ATTACK_SEED}")

    print(f"\n[1] Healthy simulation, {N_STEPS} steps, no fault, u_i = 0")
    z_healthy = simulate(gains, fault_k=N_STEPS)
    z = z_healthy[ATTACK_AREA][FIRST_V:ATTACK_K]
    cov = np.cov(z.T)
    print(f"  area {ATTACK_AREA} before k0: cov diag ({cov[0, 0]:.3f}, {cov[1, 1]:.3f}), "
          f"off-diag {cov[0, 1]:+.3f}  [{'OK' if np.abs(cov - np.eye(2)).max() < 0.05 else 'FAIL'}]")

    scenarios = {
        "intricate_pm1": (apply_intricate(z_healthy, "pm1"),
                          "intricate attack (27), $\\gamma \\in \\{-1,+1\\}$, area 5 from $k$ = 5×10⁴"),
        "intricate_01":  (apply_intricate(z_healthy, "01"),
                          "intricate attack (27), $\\gamma \\in \\{0,1\\}$, area 5 from $k$ = 5×10⁴"),
        "replay":        (apply_replay(z_healthy, REPLAY_T),
                          f"periodic replay, period {REPLAY_T}, area 5 from $k$ = 5×10⁴"),
    }

    results = {}
    for n, (name, (z_att, title)) in enumerate(scenarios.items(), start=2):
        print(f"\n[{n}] Scenario '{name}', one run: area {ATTACK_AREA}'s stored residual replaced from k0 = {ATTACK_K}")
        det = detect_all(z_att, stats)
        report(z_att, det)
        plot_fig2(det["v"], RESULTS_DIR / f"fig2_attack_{name}.png", fault_k=ATTACK_K,
                  title=f"Statistical approach: $v_{{k,T}}$, residual-level {title}")
        plot_fig3(z_att, RESULTS_DIR / f"fig3_attack_{name}.png", fault_k=ATTACK_K,
                  title=f"Threshold approach: |ž|, residual-level {title}")
        results.update({f"{name}_zcheck_{ATTACK_AREA}": z_att[ATTACK_AREA]})
        results.update({f"{name}_v_{i}_{l}": det["v"][(i, l)] for i in AREAS for l in [1, 2]})

    # One recording per period: what this recording does, not a rule about the period (see [6])
    n += 1
    print(f"\n[{n}] Replay period sweep, area {ATTACK_AREA}, T = {WINDOW_LEN}, default recording "
          "(last T_bar healthy residuals before k0)")
    for period in REPLAY_SWEEP:
        z_att = apply_replay(z_healthy, period)
        line = []
        for l in [1, 2]:
            v, psi, alarm = run_detector(z_att[ATTACK_AREA][:, l - 1], stats)
            v_pre = np.nanmean(v[FIRST_V:ATTACK_K])
            v_post = np.nanmean(v[ATTACK_K + FIRST_V:])
            share = alarm[ATTACK_K + FIRST_V:].mean()
            line.append(f"ch {l}: v {v_pre:5.1f} -> {v_post:7.1f}, alarm share {share:.2f}")
        print(f"  T_bar = {period:5d}: " + " | ".join(line))
    print("  Observations for this recording and these settings only; other recordings can fall on either side.")

    n += 1
    print(f"\n[{n}] Replay of independent Gaussian recordings, same codebook, T = {WINDOW_LEN}, alarm when v >= {threshold:.2f}")
    for period, seed in REPLAY_CHECKS:
        v_rep, alarm_rep = replay_score(period, seed, stats)
        print(f"  period {period:4d}, recording default_rng({seed}).standard_normal({period}): "
              f"v = {v_rep:.2f} -> {'alarm' if alarm_rep else 'no alarm'}")
    print("  Whether a replay crosses the threshold depends on the recording; L and T give no period boundary.")

    np.savez(RESULTS_DIR / "phase_c_attacks_results.npz", **results, attack_k=ATTACK_K, n_steps=N_STEPS,
             tau_bar=TAU_BAR, v_bar=V_BAR, replay_T=REPLAY_T, network_seed=SEED, attack_seed=ATTACK_SEED,
             codebook_sha256=codebook_sha)
    print(f"\nFigures saved -> fig2_attack_<name>.png, fig3_attack_<name>.png for {', '.join(scenarios)}")
    print("\n[Phase C attacks complete]")
