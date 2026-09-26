# Phase B: Plant Model and Distributed Estimator

Script: [`phase_b.py`](../phase_b.py) · Model: [5] Scenario 3, [1] Sec. V · Estimator: [1] Alg. 1, [2]

## Objective

1. **Model the plant.** Build a discrete-time model of the HyCon2 power network
   (Scenario 3: areas 1, 2, 3, 5, with area 4 disconnected), both healthy and after a fault.
2. **Design the estimators.** Design one state estimator per area, each using only local
   measurements and its neighbours' measurements (distributed, not centralised).

These matrices are what [Phase C](phase_c.md) simulates. Under nominal conditions each
estimator's normalised residual $\Gamma_i^{-1/2}(y_i - C\hat x_i)$ should be
white $\mathcal N(0, I)$, which is exactly what the Phase A detector tests.

## The network

```
   [1] ──P12=4── [2] ──P23=2── [3]
                  │
                P25=3
                  │
                 [5]
```

Each area is a load-frequency-control model with 4 states, and areas are coupled
through tie lines. The **fault** is a drop in area 5's inertia, $H_5$: 10 → 2,
at step $\bar k$ = 50 000 (for example, generation tripping offline).

## Symbols

### Plant

| Symbol | Code name | Meaning |
|---|---|---|
| $x_i = (\Delta\theta, \Delta\omega, \Delta P_m, \Delta P_v)$ | — | Rotor angle deviation, frequency deviation, mechanical power deviation, valve position deviation |
| $y_i = C x_i$ | `C_MAT` | Measured outputs: $\Delta\theta$ and $\Delta\omega$ |
| $u_i$ | — | Input $\Delta P_{ref,i} = \Delta P_{L,i}$ (setpoint equals load, as in [1], [2]) |
| $H_i$ | `INERTIA` | Inertia constant (s). Lower means frequency swings faster. |
| $R_i$ | `DROOP` | Governor speed regulation (droop) |
| $D_i$ | `DAMPING` | Load damping |
| $T_{t,i}$, $T_{g,i}$ | `T_TURBINE`, `T_GOVERNOR` | Turbine and governor time constants (s) |
| $P_{ij}$ | `P_TIE`, `p_tie()` | Tie-line coefficient between areas $i$ and $j$ |
| $\mathcal N_i$ | `NEIGHBOURS` | Areas directly connected to $i$ |
| $T_s$ | `TS` | Sampling period, 1 s |
| $A^c_{ii}, A^c_{ij}, \bar B^c_i$ | `A_cont_self`, `A_cont_cross`, `Bbar_cont` | Continuous-time own, coupling and input matrices |
| $A_{ii}, A_{ij}, \bar B_i$ | `A[i][i]`, `A[i][j]`, `Bbar[i]` | Discrete-time versions |
| $Q_i$, $R_i$ | `Q_MAT`, `R_MAT` | Process and measurement noise covariances ($10^{-6} I$). This $R_i$ is not the droop above; the papers reuse the letter. |

Discrete model of area $i$:

$$x_i(k+1) = A_{ii} x_i(k) + \sum_{j \in \mathcal N_i} A_{ij} x_j(k) + \bar B_i u_i(k) + w_i(k), \qquad y_i(k) = C x_i(k) + n_i(k)$$

### Estimator

| Symbol | Code name | Meaning |
|---|---|---|
| $L_{ij}$ | `L[i][j]` | Coupling gain: cancels the neighbour's effect using the neighbour's measurement |
| $\tilde Q_i$ | `Q_tilde[i]` | $Q_i + \sum_j L_{ij} R_j L_{ij}^\top$: process noise plus the neighbours' measurement noise brought in through $L_{ij}$ |
| $\Phi_i$ | `Phi[i]` | Steady-state error covariance (Riccati solution) |
| $L_{ii}$ | `L[i][i]` | Local Kalman gain |
| $\Gamma_i$ | `Gamma[i]` | Residual (innovation) covariance $C\Phi_i C^\top + R_i$ |
| $\Gamma_i^{-1/2}$ | `Gamma_inv_sqrt[i]` | Whitening matrix that turns the residual into $\mathcal N(0, I)$ for Phase A |
| $\varsigma_i$ | `varsigma_boem()` | Number of areas that use area $i$'s data, including $i$ itself |
| $\tilde F_{ii}, \tilde F_{ij}$ | — | $\sqrt{\varsigma}$-scaled error dynamics used in the stability condition |
| $\beta_i$ | `beta` | Coupling strength in the error dynamics; stability needs $\beta_i < 1$ |

## Method

**Part 1: plant** (`A_cont_self`, `A_cont_cross`, `Bbar_cont`, `discretise_area`, `build_plant`)

1. Build the continuous-time matrices from [5] eq. (2).
2. Discretise **each area separately** with exact zero-order hold, treating
   $u_i$ and the neighbours' states $x_j$ as inputs. This keeps $A_{ij}$ nonzero
   only in column 0 (only $\Delta\theta_j$ couples in), which is what makes the
   coupling cancel exactly in step 3. Discretising the whole network at once
   would fill every block.
3. `build_plant({5: 2.0})` builds the post-fault plant.

**Part 2: estimator** (`coupling_gain`, `local_gain`, `design_estimator`), following the order in [1] Alg. 1:

1. $L_{ij} = \arg\min \lVert A_{ij} - L_{ij} C \rVert$. Because $A_{ij}$ is nonzero
   only in a measured column, least squares makes $A_{ij} - L_{ij}C = 0$ exactly.
2. $\tilde Q_i = Q_i + \sum_j L_{ij} R_j L_{ij}^\top$.
3. Solve the filter Riccati equation
   $\Phi = A\Phi A^\top - A\Phi C^\top (C\Phi C^\top + R)^{-1} C\Phi A^\top + \tilde Q$,
   then $L_{ii} = A\Phi C^\top \Gamma^{-1}$ and $\Gamma = C\Phi C^\top + R$.

**Stability check** (`stability_metrics`): [2] Prop. 1 requires $\rho(\tilde F_{ii}) < 1$ and
$\beta_i = \sum_{j} \sum_{n \ge 0} \lVert \tilde F_{ii}^n \tilde F_{ij} \rVert_\infty^2 < 1$.

## Reading the output

| Check | What is tested | Result |
|---|---|---|
| [1] Continuous-time | Correct sparsity; each area stable on its own (all eigenvalues have negative real part) | All OK. $\sum P_{ij}$ = 4, 9, 2, 3. $A^c_{12} \ne A^{c\top}_{21}$ (0.167 vs 0.200) because they divide by different $H$. |
| [2] Discretisation | $\rho(A_{ii}) < 1$; $A_{ij}$ only in column 0 | $\rho$ = 0.81, 0.49, 0.84, 0.87; column structure kept |
| [3] Fault | Which matrices change when $H_5$: 10 → 2 | $A_{55}$, $A_{52}$, $\bar B_5$ change; $A_{25}$ does not (area 2's dynamics don't depend on $H_5$) |
| [4] Estimator | Coupling fully cancelled; Riccati solved; $\Phi > 0$ | $\lVert A_{ij} - L_{ij}C \rVert = 0$; Riccati residual ~$10^{-15}$; $\Gamma \approx 2$–$3 \times 10^{-6}$ |
| [5] Stability | $\rho(\tilde F_{ii}) < 1$, $\beta_i < 1$ | $\rho(\tilde F_{ii})$ = 0.49, 0.56, 0.44, 0.54; $\beta_i = 0$ |

What this means:

- $\beta_i = 0$ because the coupling is cancelled exactly, so the areas' estimation
  errors are fully decoupled. Each one is an ordinary stable Kalman filter
  ($\rho(A_{ii} - L_{ii}C)$ = 0.28–0.38).
- $\Gamma_i$ is the size of a healthy residual. After the fault, residuals in area 5
  (and its neighbour, area 2) are no longer consistent with $\Gamma_i$, which is
  what the Phase A detector picks up.

## Output file: `estimator_gains.npz`

Load with `np.load("estimator_gains.npz")`. Here `i` and `j` are area numbers
(1, 2, 3, 5), and `j` includes `i` for the diagonal blocks.

| Key | Shape | Meaning |
|---|---|---|
| `A_<i>_<j>` | (4, 4) | Healthy discrete $A_{ij}$ (`A_2_2`, `A_2_5`, …) |
| `Afault_5_<j>` | (4, 4) | Post-fault rows for area 5 ($A_{55}$, $A_{52}$) |
| `Bbar_<i>`, `Bbarfault_5` | (4, 1) | Input matrix, healthy and post-fault |
| `C`, `Q`, `R` | (2,4), (4,4), (2,2) | Output matrix and noise covariances |
| `L_<i>_<j>` | (4, 2) | Estimator gains ($L_{ii}$ local, $L_{ij}$ coupling) |
| `Phi_<i>` | (4, 4) | Error covariance $\Phi_i$ |
| `Gamma_<i>`, `Gamma_inv_sqrt_<i>` | (2, 2) | Residual covariance and its whitening matrix |
| `areas`, `Ts`, `fault_k` | — | `[1, 2, 3, 5]`, 1.0, 50000 |
