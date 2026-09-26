"""
phaseB.py — Phase B: Plant Model and Estimator Design
=====================================================
Sources: Yan, Fu & Seron, IEEE ICCA 2024 [1], Sec. V eq. (29)-(30) and Algorithm 1
         Boem et al., IEEE TCNS 6(2), 2019 [2], Sec. II and Sec. IV
         Riverso & Ferrari-Trecate, arXiv:1207.2000 [5], eq. (2), Table 2, Scenario 3

Part 1 — Plant (HyCon2 Scenario 3, areas 1, 2, 3, 5; area 4 disconnected)
    continuous matrices A^c_ii, A^c_ij, Bbar^c_i    [5] eq. (2), [1] eq. (29)
    Dss discretisation at Ts = 1 s                  [5] Sec. 1   -> A_ii, A_ij, Bbar_i   ([1] eq. 30)
    post-fault plant with H5: 10 -> 2               [1] Sec. V

Part 2 — Estimator (Algorithm 1 of [1], in the required order)
    step 1  L_ij      = argmin ||A_ij - L_ij C_j||        [2] Alg. 2 step (i)
            Q~_i      = Q_i + sum_j L_ij R_j L_ij^T       [1] Alg. 1
    step 2  Phi_i     from the filter Riccati equation    [1] Alg. 1
            L_ii      = A_ii Phi_i C_i^T (C_i Phi_i C_i^T + R_i)^{-1}   [1] eq. (8)
            Gamma_i   = C_i Phi_i C_i^T + R_i             [1] eq. (10)
    checks  rho(F~_ii) < 1 and beta_i < 1                 [2] Prop. 1, [1] Prop. 2.1

Standalone: imports only numpy and scipy.  Run:  python phaseB.py   (instant)

Saved output
------------
  estimator_gains.npz (next to this script), loaded by Phase C.
  Keys: A_<i>_<j>, Afault_5_<j>, Bbar_<i>, Bbarfault_5, C, Q, R, L_<i>_<j>, Phi_<i>,
        Gamma_<i>, Gamma_inv_sqrt_<i>, areas, Ts, fault_k
"""

import numpy as np
from pathlib import Path
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are, fractional_matrix_power

RESULTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Part 1 parameters  [5] Table 2, Scenario 3
# ---------------------------------------------------------------------------
TS: float = 1.0
AREAS: list[int] = [1, 2, 3, 5]
NEIGHBOURS: dict[int, list[int]] = {1: [2], 2: [1, 3, 5], 3: [2], 5: [2]}   # N_i, no self

INERTIA    = {1: 12.0, 2: 10.0,   3: 8.0,  5: 10.0}     # H_i
DROOP      = {1: 0.05, 2: 0.0625, 3: 0.08, 5: 0.05}     # R_i  (speed regulation)
DAMPING    = {1: 0.70, 2: 0.90,   3: 0.90, 5: 0.86}     # D_i
T_TURBINE  = {1: 0.65, 2: 0.40,   3: 0.30, 5: 0.80}     # T_t_i
T_GOVERNOR = {1: 0.10, 2: 0.10,   3: 0.10, 5: 0.15}     # T_g_i
P_TIE      = {(1, 2): 4.0, (2, 3): 2.0, (2, 5): 3.0}    # P_ij; P34, P45 belong to area 4 and are NOT used

N_STATE, N_OUT = 4, 2
C_MAT = np.array([[1.0, 0.0, 0.0, 0.0],                # measures rotor angle and frequency deviation
                  [0.0, 1.0, 0.0, 0.0]])
Q_MAT = 1e-6 * np.eye(N_STATE)                          # Q_i  ([1] Sec. V)
R_MAT = 1e-6 * np.eye(N_OUT)                            # R_i

FAULT_K: int = 50_000                                   # k_bar
FAULT_H5: float = 2.0                                   # H5: 10 -> 2


def p_tie(i: int, j: int) -> float:
    """Symmetric tie-line coefficient P_ij."""
    return P_TIE[(i, j)] if (i, j) in P_TIE else P_TIE[(j, i)]


# ---------------------------------------------------------------------------
# Part 1, step 1 — continuous-time matrices  [5] eq. (2), [1] eq. (29)
# ---------------------------------------------------------------------------
def A_cont_self(i: int, H: float | None = None) -> np.ndarray:
    """
    A^c_ii for area i.  State x_i = (d_theta, d_omega, d_Pm, d_Pv).
        row 0: [ 0,            1,          0,      0     ]
        row 1: [-sumP/(2H),   -D/(2H),     1/(2H), 0     ]
        row 2: [ 0,            0,         -1/Tt,   1/Tt  ]
        row 3: [ 0,           -1/(R Tg),   0,     -1/Tg  ]
    sumP runs over the ACTIVE tie lines of area i only: 4, 9, 2, 3 for areas 1, 2, 3, 5.
    """
    H = INERTIA[i] if H is None else H
    sumP = sum(p_tie(i, j) for j in NEIGHBOURS[i])
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 0] = -sumP / (2 * H)
    A[1, 1] = -DAMPING[i] / (2 * H)
    A[1, 2] = 1.0 / (2 * H)
    A[2, 2] = -1.0 / T_TURBINE[i]
    A[2, 3] = 1.0 / T_TURBINE[i]
    A[3, 1] = -1.0 / (DROOP[i] * T_GOVERNOR[i])
    A[3, 3] = -1.0 / T_GOVERNOR[i]
    return A


def A_cont_cross(i: int, j: int, H: float | None = None) -> np.ndarray:
    """A^c_ij: single entry P_ij/(2 H_i) at (1,0).  Denominator is H_i, so A_ij and A_ji are not transposes."""
    H = INERTIA[i] if H is None else H
    A = np.zeros((4, 4))
    A[1, 0] = p_tie(i, j) / (2 * H)
    return A


def Bbar_cont(i: int, H: float | None = None) -> np.ndarray:
    """Bbar^c_i = B^c_i + L^c_i, since [1] (following [2]) sets u_i = dP_ref_i = dP_L_i."""
    H = INERTIA[i] if H is None else H
    B = np.zeros((4, 1))
    B[1, 0] = -1.0 / (2 * H)
    B[3, 0] = 1.0 / T_GOVERNOR[i]
    return B


# ---------------------------------------------------------------------------
# Part 1, step 2 — Dss discretisation  [5] Sec. 1
# ---------------------------------------------------------------------------
def discretise_area(i: int, H: float | None = None, Ts: float = TS):
    """
    System-by-system exact ZOH discretisation: area i is discretised alone, with its input u_i and
    its neighbours' states x_j treated as exogenous inputs.  Gives
        A_ii = expm(A^c_ii Ts),     A_ij = int_0^Ts expm(A^c_ii s) ds  A^c_ij,
    and keeps A_ij nonzero only in column 0, as A^c_ij is.
    (Scheme D, expm of the assembled network, fills every block and must NOT be used.)

    Returns A_ii (4x4), {j: A_ij} (4x4 each), Bbar_i (4x1).
    """
    E_c = np.hstack([Bbar_cont(i, H)] + [A_cont_cross(i, j, H) for j in NEIGHBOURS[i]])
    A_d, E_d, _, _, _ = cont2discrete((A_cont_self(i, H), E_c, np.eye(4), np.zeros((4, E_c.shape[1]))),
                                      dt=Ts, method="zoh")
    Bbar = E_d[:, :1]
    A_cross = {j: E_d[:, 1 + 4 * n: 5 + 4 * n] for n, j in enumerate(NEIGHBOURS[i])}
    return A_d, A_cross, Bbar


def build_plant(h_override: dict[int, float] | None = None):
    """
    Discrete blocks for all areas.  h_override = {5: 2.0} builds the post-fault plant.
    Returns A[i][i], A[i][j], Bbar[i].
    """
    h_override = h_override or {}
    A, Bbar = {}, {}
    for i in AREAS:
        A_ii, A_cross, Bbar_i = discretise_area(i, h_override.get(i))
        A[i] = {i: A_ii, **A_cross}
        Bbar[i] = Bbar_i
    return A, Bbar


# ---------------------------------------------------------------------------
# Part 2 — Estimator design  [1] Algorithm 1
# ---------------------------------------------------------------------------
def coupling_gain(A_ij: np.ndarray, C_j: np.ndarray) -> np.ndarray:
    """
    Step 1: L_ij = argmin ||A_ij - L_ij C_j||_inf   ([2] Alg. 2 step (i), an LP in general).
    Here A_ij is nonzero only in column 0 and C_j = [I_2 0], so L_ij = A_ij[:, :2] gives
    A_ij - L_ij C_j = 0 exactly; the least-squares solution A_ij C_j^T (C_j C_j^T)^{-1} is the same.
    The sqrt(varsigma) scaling of the tilde matrices factors out of the argmin.
    """
    return A_ij @ C_j.T @ np.linalg.inv(C_j @ C_j.T)


def local_gain(A_ii: np.ndarray, C_i: np.ndarray, Q_tilde: np.ndarray, R_i: np.ndarray):
    """
    Step 2: solve the FILTER Riccati equation
        Phi = A Phi A^T - A Phi C^T (C Phi C^T + R)^{-1} C Phi A^T + Q~
    scipy's solve_discrete_are(a, b, q, r) solves the control form a^T X a ..., so pass a = A^T, b = C^T.
    Then L_ii = A Phi C^T Gamma^{-1},  Gamma = C Phi C^T + R  ([1] eq. 8 and eq. 10).
    """
    Phi = solve_discrete_are(A_ii.T, C_i.T, Q_tilde, R_i)
    Gamma = C_i @ Phi @ C_i.T + R_i
    L_ii = A_ii @ Phi @ C_i.T @ np.linalg.inv(Gamma)
    return L_ii, Phi, Gamma


def design_estimator(A: dict):
    """Run Algorithm 1 for every area in order.  Returns L, Q_tilde, Phi, Gamma, Gamma_inv_sqrt."""
    L, Q_tilde, Phi, Gamma, Gamma_inv_sqrt = {}, {}, {}, {}, {}
    for i in AREAS:
        L[i] = {}
        for j in NEIGHBOURS[i]:                                   # step 1: coupling gains first
            L[i][j] = coupling_gain(A[i][j], C_MAT)
        Qt = Q_MAT.copy()                                         # Q~_i needs the coupling gains
        for j in NEIGHBOURS[i]:
            Qt = Qt + L[i][j] @ R_MAT @ L[i][j].T
        Q_tilde[i] = Qt
        L[i][i], Phi[i], Gamma[i] = local_gain(A[i][i], C_MAT, Qt, R_MAT)   # step 2
        Gamma_inv_sqrt[i] = fractional_matrix_power(Gamma[i], -0.5).real     # symmetric root = Gamma^{-1/2} in eq. (10)
    return L, Q_tilde, Phi, Gamma, Gamma_inv_sqrt


def varsigma_boem() -> dict[int, int]:
    """
    varsigma_i = card(S_i) with S_i = {j : i in N_j} UNION {i}   ([2] Sec. II-A, self-inclusive).
    [1] prints the definition without self; the scaling cancels in the L_ij design, and [2]'s theorems
    (used for rho(F~_ii) and B_i(k)) are proven under the self-inclusive version, so that is used here.
    Scenario 3: {1: 2, 2: 4, 3: 2, 5: 2}.
    """
    return {i: 1 + sum(1 for j in AREAS if i in NEIGHBOURS[j]) for i in AREAS}


def stability_metrics(A: dict, L: dict, n_terms: int = 500):
    """
    F~_ii = sqrt(varsigma_i) (A_ii - L_ii C),   F~_ij = sqrt(varsigma_j) (A_ij - L_ij C)
    beta_i = sum_{j in N_i} sum_{n>=0} ||F~_ii^n F~_ij||_inf^2          ([2] eq. 12, [1] eq. 9)
    Returns per area: rho(A_ii - L_ii C), rho(F~_ii), beta_i.
    """
    vs = varsigma_boem()
    out = {}
    for i in AREAS:
        F_plain = A[i][i] - L[i][i] @ C_MAT
        F_ii = np.sqrt(vs[i]) * F_plain
        beta, power = 0.0, np.eye(4)
        for _ in range(n_terms):
            for j in NEIGHBOURS[i]:
                F_ij = np.sqrt(vs[j]) * (A[i][j] - L[i][j] @ C_MAT)
                beta += np.linalg.norm(power @ F_ij, ord=np.inf) ** 2
            power = power @ F_ii
        out[i] = dict(rho_plain=float(max(abs(np.linalg.eigvals(F_plain)))),
                      rho_tilde=float(max(abs(np.linalg.eigvals(F_ii)))),
                      beta=float(beta), varsigma=vs[i])
    return out


# ---------------------------------------------------------------------------
# __main__ — Phase B validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("=" * 65)
    print("Phase B Validation — HyCon2 Scenario 3 plant + Algorithm 1 estimator")
    print("=" * 65)

    A, Bbar = build_plant()
    A_f, Bbar_f = build_plant(h_override={5: FAULT_H5})

    # ------------------------------------------------------------------
    # Check 1: continuous-time structure
    # ------------------------------------------------------------------
    print("\n[1] Continuous-time matrices")
    expected_nz = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 1), (3, 3)}
    for i in AREAS:
        nz = {tuple(x) for x in np.argwhere(np.abs(A_cont_self(i)) > 0)}
        sumP = sum(p_tie(i, j) for j in NEIGHBOURS[i])
        print(f"  area {i}: sum P_ij = {sumP:.0f}, nonzero pattern {'OK' if nz == expected_nz else 'WRONG'}, "
              f"eig(A^c_ii) real parts max = {np.linalg.eigvals(A_cont_self(i)).real.max():.3f} (<0 OK)")
    print(f"  A^c_12[1,0] = P12/(2 H1) = {A_cont_cross(1, 2)[1, 0]:.4f},  A^c_21[1,0] = P12/(2 H2) = {A_cont_cross(2, 1)[1, 0]:.4f}  (not transposes)")

    # ------------------------------------------------------------------
    # Check 2: discretisation
    # ------------------------------------------------------------------
    print("\n[2] Dss discretisation (Ts = 1 s)")
    for i in AREAS:
        rho = max(abs(np.linalg.eigvals(A[i][i])))
        cols_ok = all(np.abs(A[i][j][:, 1:]).max() < 1e-13 for j in NEIGHBOURS[i])
        print(f"  area {i}: rho(A_ii) = {rho:.4f} [{'OK' if rho < 1 else 'UNSTABLE'}],  A_ij nonzero only in column 0: {cols_ok}")
    print("  A_25 =\n", A[2][5])

    # ------------------------------------------------------------------
    # Check 3: fault matrices
    # ------------------------------------------------------------------
    print(f"\n[3] Fault H5: {INERTIA[5]:.0f} -> {FAULT_H5:.0f} at k = {FAULT_K}")
    print(f"  |dA_55|max = {np.abs(A_f[5][5] - A[5][5]).max():.3f}   |dA_52|max = {np.abs(A_f[5][2] - A[5][2]).max():.3f}   "
          f"|dBbar_5|max = {np.abs(Bbar_f[5] - Bbar[5]).max():.3f}   (all three change; A_25 does not: {np.abs(A_f[2][5] - A[2][5]).max():.1e})")

    # ------------------------------------------------------------------
    # Check 4: estimator design
    # ------------------------------------------------------------------
    print("\n[4] Algorithm 1: coupling gains, Q~, Riccati, local gains")
    L, Q_tilde, Phi, Gamma, Gamma_inv_sqrt = design_estimator(A)
    for i in AREAS:
        for j in NEIGHBOURS[i]:
            r = np.linalg.norm(A[i][j] - L[i][j] @ C_MAT, ord=np.inf)
            print(f"  ||A_{i}{j} - L_{i}{j} C||_inf = {r:.1e}  [{'OK, coupling cancelled' if r < 1e-12 else 'NONZERO'}]")
    for i in AREAS:
        rhs = (A[i][i] @ Phi[i] @ A[i][i].T
               - A[i][i] @ Phi[i] @ C_MAT.T @ np.linalg.inv(Gamma[i]) @ C_MAT @ Phi[i] @ A[i][i].T + Q_tilde[i])
        res = np.abs(Phi[i] - rhs).max()
        eig_min = np.linalg.eigvalsh(Phi[i]).min()
        print(f"  area {i}: Riccati residual {res:.1e} [{'OK' if res < 1e-10 else 'FAIL'}],  Phi > 0: {eig_min > 0},  "
              f"Q~ != Q: {not np.allclose(Q_tilde[i], Q_MAT)},  Gamma = diag({Gamma[i][0, 0]:.2e}, {Gamma[i][1, 1]:.2e})")

    # ------------------------------------------------------------------
    # Check 5: stability
    # ------------------------------------------------------------------
    print("\n[5] Stability ([2] sufficient condition; varsigma self-inclusive as in [2])")
    for i, m in stability_metrics(A, L).items():
        print(f"  area {i}: varsigma = {m['varsigma']}, rho(A_ii - L_ii C) = {m['rho_plain']:.3f}, "
              f"rho(F~_ii) = {m['rho_tilde']:.3f} [{'OK' if m['rho_tilde'] < 1 else '>=1'}], beta_i = {m['beta']:.1e} [{'OK' if m['beta'] < 1 else 'FAIL'}]")
    print("  With A_ij - L_ij C = 0 exactly, the error dynamics are block-diagonal and stable by Kalman-filter theory")
    print("  regardless of the sufficient condition; rho(F~_ii) < 1 additionally means [2]'s B_i(k) bound converges.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = RESULTS_DIR / "estimator_gains.npz"
    np.savez(out,
             areas=np.array(AREAS), Ts=TS, fault_k=FAULT_K, C=C_MAT, Q=Q_MAT, R=R_MAT,
             **{f"A_{i}_{j}": A[i][j] for i in AREAS for j in A[i]},
             **{f"Afault_5_{j}": A_f[5][j] for j in A_f[5]},
             **{f"Bbar_{i}": Bbar[i] for i in AREAS}, Bbarfault_5=Bbar_f[5],
             **{f"L_{i}_{j}": L[i][j] for i in AREAS for j in L[i]},
             **{f"Phi_{i}": Phi[i] for i in AREAS},
             **{f"Gamma_{i}": Gamma[i] for i in AREAS},
             **{f"Gamma_inv_sqrt_{i}": Gamma_inv_sqrt[i] for i in AREAS})
    print(f"\nGains saved -> {out}")
    print("\n[Phase B validation complete]")