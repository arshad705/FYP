"""
phase_c_attacks.py — Phase C, attack scenarios: intricate attack (27) and replay attack.

[1] Sec. V says simulations of attack (27) and of replay attacks give results "akin to" Figs. 2 and 3
(statistical test detects, threshold on |z_check| does not), but gives no details.  This script runs them.

Pipeline
    1. Simulate     healthy network (no fault), same plant and estimator as phase_c.py, u_i = 0
    2. Attack       replace area 5's normalised residual for k >= k0                        [1] Sec. IV
                      intricate: r(k) = v_bar r(k - tau_bar) + sqrt(1 - v_bar^2) z(k), z_a = gamma r   [1] eq. (27)
                      replay:    z_check(k) = z_check(k - T_bar) for k >= k0                [1] Remark 4.2
    3. Detect       v_{k,T} per area and channel, as in phase_c.py                          [1] Alg. 2, D = 1
    4. Plot         Fig. 2 / Fig. 3 equivalents for each attack

Assumptions not stated in [1] (to confirm with supervisor):
    - Attacked area 5, both channels, from k0 = 5e4 (same area and time as the H5 fault); no fault in these runs.
    - The attack acts on the normalised residual after simulation, as [1] defines both attacks; the
      estimators are not fed the attacked data, so the attack does not spread to neighbouring areas.
    - tau_bar = 1, v_bar = 1/sqrt(2) from [3] Sec. VIII.
    - gamma in {-1,+1} as main run ({0,1} as printed in [1] is not stealthy, Phase A check [6]); {0,1} also run.
    - Replay period T_bar = 400 for the figures; T_bar in {100, 400, 800, 1600} swept (window T = 800).

Run phase_a.py and phase_b.py first.  Usage: python phase_c_attacks.py
Output: fig2_attack_<name>.png, fig3_attack_<name>.png, phase_c_attacks_results.npz
"""

import numpy as np
from scipy.stats import kstest

from phase_a import run_detector, load_stats, intricate_attack, WINDOW_LEN, BLOCK_LEN, ALPHA
from phase_b import AREAS
from phase_c import (load_gains, simulate, detect_all, plot_fig2, plot_fig3, first_alarm_after,
                     N_STEPS, ALPHA_1, ALPHA_BOEM, RESULTS_DIR, FIRST_V)

ATTACK_K:     int   = 50_000                 # k0, same as the fault time in phase_c.py
ATTACK_AREA:  int   = 5
TAU_BAR:      int   = 1                      # [3] Sec. VIII
V_BAR:        float = 1 / np.sqrt(2)         # [3] Sec. VIII
REPLAY_T:     int   = 400                    # replay period for the figures, < T = 800
REPLAY_SWEEP: list[int] = [100, 400, 800, 1600]
ATTACK_SEED:  int   = 27


# Step 2 — Attacks on the normalised residual of one area
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


def report(name: str, z_check: dict, det: dict, k0: int = ATTACK_K) -> None:
    """Checks [2]-[4] for one attacked run."""
    pre, post = slice(FIRST_V, k0), slice(k0, N_STEPS)
    full = slice(k0 + FIRST_V, N_STEPS)                       # window holds attacked data only

    print(f"\n  Stealthiness, area {ATTACK_AREA} after k0 (healthy: var 1, acf 0, KS ~ 0.005)")
    for l in [1, 2]:
        z = z_check[ATTACK_AREA][post, l - 1]
        print(f"    ch {l}: var {z.var():.3f}, lag-1 acf {np.corrcoef(z[:-1], z[1:])[0, 1]:+.3f}, "
              f"KS vs N(0,1) {kstest(z, 'norm').statistic:.3f}")

    print("  Statistical test: mean v before -> after, alarm-on fraction after")
    for i in AREAS:
        for l in [1, 2]:
            v, alarm = det["v"][(i, l)], det["alarm"][(i, l)]
            v_pre, v_post = np.nanmean(v[pre]), np.nanmean(v[full])
            on = alarm[full].mean()
            tag = ""
            if i == ATTACK_AREA:
                k_hit = first_alarm_after(alarm, k0)
                tag = (f", first alarm +{k_hit - k0}" if k_hit is not None else ", no alarm") \
                      + f"  [{'DETECTED' if on > 0.5 else 'not reliably detected'}]"
            print(f"    area {i}, ch {l}: {v_pre:6.1f} -> {v_post:7.1f} (ratio {v_post / v_pre:5.2f}), "
                  f"alarm on {on:.2f}{tag}")

    print(f"  Threshold baseline, area {ATTACK_AREA}: fraction of steps with |z_check| > threshold, before -> after")
    for l in [1, 2]:
        z = np.abs(z_check[ATTACK_AREA][:, l - 1])
        print(f"    ch {l}: > {ALPHA_BOEM}: {np.mean(z[pre] > ALPHA_BOEM):.4f} -> {np.mean(z[post] > ALPHA_BOEM):.4f}"
              f"  |  > {ALPHA_1}: {np.mean(z[pre] > ALPHA_1):.5f} -> {np.mean(z[post] > ALPHA_1):.5f}")


if __name__ == "__main__":
    print("=" * 65)
    print("Phase C Attacks — intricate attack (27) and replay ([1] Sec. V)")
    print("=" * 65)

    stats_path, gains_path = RESULTS_DIR / "detector_stats.npz", RESULTS_DIR / "estimator_gains.npz"
    for p, script in [(stats_path, "phase_a.py"), (gains_path, "phase_b.py")]:
        if not p.exists():
            raise SystemExit(f"Missing {p.name}: run {script} first.")
    stats, gains = load_stats(stats_path), load_gains(gains_path)
    if int(stats["T"]) != WINDOW_LEN or int(stats["L"]) != BLOCK_LEN:
        raise SystemExit("detector_stats.npz was built with different T or L; re-run phase_a.py.")

    print(f"\n[1] Healthy simulation, {N_STEPS} steps, no fault, u_i = 0")
    z_healthy = simulate(gains, fault_k=N_STEPS)
    z = z_healthy[ATTACK_AREA][FIRST_V:ATTACK_K]
    cov = np.cov(z.T)
    print(f"  area {ATTACK_AREA} before k0: cov diag ({cov[0, 0]:.3f}, {cov[1, 1]:.3f}), "
          f"off-diag {cov[0, 1]:+.3f}  [{'OK' if np.abs(cov - np.eye(2)).max() < 0.05 else 'FAIL'}]")

    scenarios = {
        "intricate_pm1": (apply_intricate(z_healthy, "pm1"),
                          "Intricate attack (27), $\\gamma \\in \\{-1,+1\\}$, area 5 from $k$ = 5×10⁴"),
        "intricate_01":  (apply_intricate(z_healthy, "01"),
                          "Intricate attack (27), $\\gamma \\in \\{0,1\\}$, area 5 from $k$ = 5×10⁴"),
        "replay":        (apply_replay(z_healthy, REPLAY_T),
                          f"Replay attack, period {REPLAY_T}, area 5 from $k$ = 5×10⁴"),
    }

    results = {}
    for n, (name, (z_att, title)) in enumerate(scenarios.items(), start=2):
        print(f"\n[{n}] Scenario '{name}': area {ATTACK_AREA} attacked from k0 = {ATTACK_K}")
        det = detect_all(z_att, stats)
        report(name, z_att, det)
        plot_fig2(det["v"], RESULTS_DIR / f"fig2_attack_{name}.png", fault_k=ATTACK_K,
                  title=f"Statistical approach: $v_{{k,T}}$, {title}")
        plot_fig3(z_att, RESULTS_DIR / f"fig3_attack_{name}.png", fault_k=ATTACK_K,
                  title=f"Threshold approach: |ž|, {title}")
        results.update({f"{name}_zcheck_{ATTACK_AREA}": z_att[ATTACK_AREA]})
        results.update({f"{name}_v_{i}_{l}": det["v"][(i, l)] for i in AREAS for l in [1, 2]})

    # Replay is a periodic copy: a window of T = 800 only holds repeated blocks when the period is shorter
    n += 1
    print(f"\n[{n}] Replay period sweep, area {ATTACK_AREA} (window T = {WINDOW_LEN})")
    for period in REPLAY_SWEEP:
        z_att = apply_replay(z_healthy, period)
        line = []
        for l in [1, 2]:
            v, psi, alarm = run_detector(z_att[ATTACK_AREA][:, l - 1], stats)
            v_pre = np.nanmean(v[FIRST_V:ATTACK_K])
            v_post = np.nanmean(v[ATTACK_K + FIRST_V:])
            on = alarm[ATTACK_K + FIRST_V:].mean()
            line.append(f"ch {l}: v {v_pre:5.1f} -> {v_post:7.1f}, alarm on {on:.2f}")
        print(f"  T_bar = {period:5d}: " + " | ".join(line))
    print(f"  Expected: detected for T_bar < T, not for T_bar >= T (window then holds no repeated block).")

    np.savez(RESULTS_DIR / "phase_c_attacks_results.npz", **results, attack_k=ATTACK_K, n_steps=N_STEPS,
             tau_bar=TAU_BAR, v_bar=V_BAR, replay_T=REPLAY_T)
    print(f"\nFigures saved -> fig2_attack_<name>.png, fig3_attack_<name>.png for {', '.join(scenarios)}")
    print("\n[Phase C attacks complete]")
