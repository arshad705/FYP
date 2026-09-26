# Phase A and Phase B Code Summary

This document summarizes the purpose, algorithm, variables, and function-level behavior of the two main scripts in this project:

- `phase_a.py` — Phase A: statistical anomaly detector
- `phase_b.py` — Phase B: plant model and estimator design

---

## 1. Overall project structure

The project is implemented as a two-stage pipeline:

1. Phase A builds and validates a statistical anomaly detector using synthetic residual data.
2. Phase B builds the plant model and designs the state estimator for the multi-area system.

The two phases are complementary:

- Phase A answers: “Is the residual abnormal?”
- Phase B answers: “What is the system model and how do we estimate its state?”

---

## 2. Phase A: statistical anomaly detector

File: `phase_a.py`

### Purpose

This script validates a statistical anomaly detector for a residual signal $z_{check}(k)$ before it is connected to a real plant model.

The detector is built from a codebook of normal residual blocks. It compares current residual blocks against this nominal profile and raises an alarm when the statistic is too large.

This phase assumes the residual is already available as an input signal, and it tests the detector on synthetic data only.

### Main algorithm

The algorithm follows the paper’s detector construction:

1. Construct a codebook using Lloyd’s algorithm (k-means)
2. Compute the nominal CDF vector $u^*$
3. Compute the covariance matrix $\Sigma$
4. Run the online detector over a sliding window
5. Compute the test statistic $v_{k,T}$
6. Convert to a chi-square probability $\psi_{k,T}$
7. Trigger alarm when $\psi_{k,T} \ge \alpha$

The main detector statistic is:

$$
v_{k,T} = T (u_{k,T} - u^*)^T \Sigma^{-1} (u_{k,T} - u^*)
$$

and the alarm decision is based on the chi-square CDF value:

$$
\psi_{k,T} = H_I(v_{k,T})
$$

---

## 3. Function-by-function breakdown for Phase A

### `build_codebook(...)`

Purpose:
- Generate the codebook $\rho_m$ using Lloyd’s algorithm.

What it does:
- Samples many Gaussian vectors of dimension $L$
- Fits a k-means model with $I$ clusters
- Returns the cluster centers as a matrix of shape $(I, L)$

Variables:
- `n_points`: number of cluster centers, $I$
- `block_len`: block length, $L$
- `seed`: random seed
- `n_train`: number of Gaussian samples used for training

Meaning:
- Each row of `codebook` is a representative normal block center.

---

### `nominal_cdf_vector(codebook)`

Purpose:
- Compute the nominal CDF vector $u^*$

Mathematically:

$$
[u^*]_m = \prod_{l=0}^{L-1} \Phi(\rho_{m,l})
$$

Variables:
- `codebook`: codebook matrix
- `norm.cdf`: standard normal CDF $\Phi(\cdot)$

Meaning:
- This gives the probability that a nominal Gaussian block is component-wise below the codebook vector.

---

### `nominal_covariance(codebook)`

Purpose:
- Compute the analytical covariance matrix $\Sigma$

What it does:
- Considers overlapping blocks at different lags
- Computes the expected indicator covariance
- Sums over all relevant lags and subtracts $u^*u^{*T}$

Variables:
- `n_points`: number of codebook entries
- `block_len`: block length $L$
- `cdf`: Gaussian CDF values for all codebook entries
- `u_star`: nominal vector

Meaning:
- This matrix defines the correlation structure of the nominal block statistics, which is needed in the detector.

---

### `nominal_covariance_mc(...)`

Purpose:
- Monte Carlo estimate of $\Sigma$ for verification.

What it does:
- Generates many Gaussian residual samples
- Forms sliding blocks of length $L$
- Computes empirical indicators over the codebook
- Estimates covariance numerically

Meaning:
- This is a sanity-check function to verify the analytical covariance formula is correct.

---

### `run_detector(residual, stats, window_len, alpha)`

Purpose:
- Run the detector on an entire residual sequence.

What it does:
- For each time index $k$, create a new block of residual samples
- Compare it against the codebook
- Convert the comparison to an indicator vector
- Maintain a sliding window of the last $T$ indicator vectors
- Compute the empirical mean vector $u_{k,T}$
- Compute the detector statistic $v_{k,T}$
- Convert it to a chi-square probability $\psi_{k,T}$
- Set an alarm if $\psi_{k,T} \ge \alpha$

Variables:
- `residual`: residual signal $z_{check}(k)$
- `stats`: contains `codebook`, `u_star`, `Sigma_inv`
- `window_len`: $T$
- `alpha`: alarm threshold
- `ring`: sliding window memory of recent indicator vectors
- `count`: running count over the window
- `d`: difference between empirical and nominal vector, $u_{k,T} - u^*$
- `v`: detection statistic
- `psi`: chi-square CDF value
- `alarm`: Boolean alarm signal

Meaning:
- This is the main online anomaly-detection algorithm.

---

### `save_stats(...)` and `load_stats(...)`

Purpose:
- Cache basic detector parameters and reload them later.

Saved values:
- `codebook`
- `u_star`
- `Sigma`
- `Sigma_inv`
- `L`
- `T`
- `I`

Meaning:
- These are the precomputed statistical detector parameters.

---

### `ar1_residual(...)`

Purpose:
- Generate an AR(1) correlated residual for testing.

Meaning:
- This creates a residual with Gaussian marginal distribution but time correlation, so the detector can be evaluated under non-i.i.d. behavior.

---

### `intricate_attack(...)`

Purpose:
- Generate a structured attack signal to test robustness.

Meaning:
- This simulates a deliberately crafted attack that may evade naive detection, allowing the detector to be tested under adversarial conditions.

---

### `__main__`

Purpose:
- Run validation checks and save detector statistics.

Checks performed:
1. Codebook generation
2. Nominal CDF vector validity
3. Covariance symmetry and PSD behavior
4. Closed-form covariance vs Monte Carlo comparison
5. Normal i.i.d. input check
6. Correlated input check
7. Stealth-attack check
8. Save output file `detector_stats.npz`

---

## 4. Key variables in Phase A

- `BLOCK_LEN = L`: length of each sliding residual block
- `WINDOW_LEN = T`: number of blocks in the detector window
- `N_CODEBOOK = I`: number of codebook entries
- `ALPHA`: detector alarm threshold on the chi-square CDF
- `codebook`: representative normal residual blocks
- `u_star`: nominal probability vector
- `Sigma`: covariance matrix of the indicator process
- `Sigma_inv`: inverse covariance used in the quadratic statistic
- `residual`: observed residual signal
- `v`: detector statistic value
- `psi`: transformed confidence value
- `alarm`: final detection decision

---

## 5. Phase B: plant model and estimator design

File: `phase_b.py`

### Purpose

This script builds the multi-area power-system plant and designs an estimator for each area. It is not a detector in itself; rather, it prepares the system model and state-estimator matrices for later use in a fault-detection framework.

The system is a network of interconnected areas with local dynamics and inter-area coupling.

### Main algorithm

The script does the following:

1. Define the continuous-time model of each area and its neighbors
2. Build the continuous-time matrices $A_{ii}^c$, $A_{ij}^c$, and $B_i^c$
3. Discretize with zero-order hold (ZOH)
4. Build the plant model for all areas
5. Compute inter-area coupling gains $L_{ij}$
6. Compute modified covariance $Q_i^\sim$
7. Solve the discrete Riccati equation for each area
8. Compute local gains $L_{ii}$ and innovation covariance $\Gamma_i$
9. Check stability via spectral radius and $\beta_i$

---

## 6. Function-by-function breakdown for Phase B

### Parameter definitions

These constants define the model parameters:

- `AREAS`: which areas are included
- `NEIGHBOURS`: network connectivity
- `INERTIA`, `DROOP`, `DAMPING`: area physical parameters
- `T_TURBINE`, `T_GOVERNOR`: turbine and governor dynamics
- `P_TIE`: tie-line strengths
- `C_MAT`: measurement matrix
- `Q_MAT`, `R_MAT`: process and measurement noise matrices
- `FAULT_H5`: post-fault inertia reduction in area 5
- `FAULT_K`: fault time index

---

### `p_tie(i, j)`

Purpose:
- Return the symmetric tie-line coefficient between areas.

---

### `A_cont_self(i, H=None)`

Purpose:
- Build the continuous-time self-dynamics matrix $A_{ii}^c$.

Meaning:
- This models the internal dynamics of area $i$.

---

### `A_cont_cross(i, j, H=None)`

Purpose:
- Build the continuous-time coupling matrix $A_{ij}^c$.

Meaning:
- This captures how area $j$ affects area $i$ through the tie-line.

---

### `Bbar_cont(i, H=None)`

Purpose:
- Build the continuous-time input matrix $B_i^c$.

Meaning:
- This defines how control input enters the dynamics.

---

### `discretise_area(i, H=None, Ts=TS)`

Purpose:
- Convert each area model from continuous time to discrete time using ZOH discretization.

Outputs:
- `A_ii`
- `A_ij` for neighbor areas
- `Bbar_i`

Meaning:
- This creates the discrete-time plant for the actual estimator design.

---

### `build_plant(h_override=None)`

Purpose:
- Build the full multi-area discrete plant model.

Meaning:
- This assembles all local and inter-area dynamic matrices for the entire system.

---

### `coupling_gain(A_ij, C_j)`

Purpose:
- Compute the coupling gain $L_{ij}$.

Mathematically, this is implemented as:

$$
L_{ij} = A_{ij} C_j^T (C_j C_j^T)^{-1}
$$

Meaning:
- This is the first step in removing the effect of neighboring-area coupling from the estimation process.

---

### `local_gain(A_ii, C_i, Q_tilde, R_i)`

Purpose:
- Solve the discrete Riccati equation and compute local estimator gain.

Mathematically:

$$
\Phi_i = A_{ii} \Phi_i A_{ii}^T - A_{ii} \Phi_i C_i^T (C_i \Phi_i C_i^T + R_i)^{-1} C_i \Phi_i A_{ii}^T + Q_i^\sim
$$

and then:

$$
L_{ii} = A_{ii} \Phi_i C_i^T (C_i \Phi_i C_i^T + R_i)^{-1}
$$

Variables:
- `Phi_i`: Riccati solution
- `Gamma_i`: innovation covariance
- `L_ii`: local observer gain

Meaning:
- This produces the local estimator for each area.

---

### `design_estimator(A)`

Purpose:
- Run the full estimator-design algorithm for each area.

What it does:
- Compute all inter-area coupling gains
- Update the modified process covariance:
  $$
  Q_i^\sim = Q_i + \sum_j L_{ij} R_j L_{ij}^T
  $$
- Solve the Riccati equation for each area
- Compute local gains and inverse-square-root innovation scaling

Meaning:
- This is the core estimator design routine.

---

### `varsigma_boem()`

Purpose:
- Compute the self-inclusive neighborhood size $\varsigma_i$.

Meaning:
- This parameter is used in the stability analysis of the estimator.

---

### `stability_metrics(A, L, n_terms=500)`

Purpose:
- Check whether the estimator is stable.

It computes:
- spectral radius of the closed-loop matrix
- effective matrices $\tilde F_{ii}$
- a beta quantity $\beta_i$

Meaning:
- This determines whether the designed observer satisfies the theoretical stability conditions.

---

### `__main__`

Purpose:
- Validate the plant and estimator and save the output matrices.

Checks performed:
- continuous-time matrix structure
- discretization stability
- fault effect on area 5
- coupling gain cancellation
- Riccati residual
- stability metrics
- save `estimator_gains.npz`

---

## 7. Key variables in Phase B

- `A_ii`: local area dynamics matrix
- `A_ij`: inter-area coupling matrix
- `Bbar_i`: control input matrix
- `C_MAT`: measurement matrix
- `Q_MAT`: process-noise covariance
- `R_MAT`: measurement-noise covariance
- `L_ij`: coupling gain between areas
- `L_ii`: local estimator gain
- `Phi_i`: Riccati solution
- `Gamma_i`: innovation covariance
- `Gamma_inv_sqrt_i`: inverse square-root of innovation covariance
- `FAULT_H5`: post-fault inertia change in area 5
- `FAULT_K`: fault activation time index
- `Ts`: sampling period

---

## 8. Summary

The project is structured as a two-stage control and detection pipeline:

- Phase A: build and validate the statistical residual detector
- Phase B: build the plant model and design the observer used to generate residuals and estimate system states

In practical terms:

- `phase_a.py` is the detector component
- `phase_b.py` is the system-model and estimator component

Together, they form the preparation needed for a full fault-detection framework.

---

## 9. Overall conclusion

From the code structure and validation behavior, the logic is consistent with the intended academic formulation:

- Phase A implements a block-based statistical anomaly detector.
- Phase B implements a multi-area power-system model and estimator design.

The main concern is not mathematical logic but execution environment compatibility for the Phase A script, especially the dependency stack involving `scikit-learn`, `numpy`, and `pandas`.
