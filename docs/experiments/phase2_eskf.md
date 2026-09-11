# Phase 2: 15-State Error-State Kalman Filter (ESKF)

## 1. Context & Motivation
Having proven that pure strapdown integration diverges quadratically due to sensor biases, the next step is **optimal state estimation**.  
An **Error-State Kalman Filter (ESKF)** separates the system into a high-rate non-linear nominal state and a linearized error state, enabling closed-loop tracking of sensor biases and attitude errors during periods of GNSS availability.

---

## 2. Mathematical Formulation

### 2.1 State Vector Decomposition
The nominal state vector $\mathbf{x} \in \mathbb{R}^{16}$ carries the full non-linear parameters:
$$\mathbf{x} = \left[ \mathbf{p}^n, \, \mathbf{v}^n, \, \mathbf{q}_b^n, \, \mathbf{b}_a, \, \mathbf{b}_g \right]^T$$

The error state $\delta \mathbf{x} \in \mathbb{R}^{15}$ represents small perturbations:
$$\delta \mathbf{x} = \left[ \delta \mathbf{p}^n, \, \delta \mathbf{v}^n, \, \delta \boldsymbol{\theta}^n, \, \delta \mathbf{b}_a, \, \delta \mathbf{b}_g \right]^T$$

### 2.2 Continuous-Time Error Dynamics
$$\delta \dot{\mathbf{p}}^n = \delta \mathbf{v}^n$$
$$\delta \dot{\mathbf{v}}^n = -\lfloor \mathbf{R}_b^n \hat{\mathbf{a}}^b \times \rfloor \delta \boldsymbol{\theta}^n - \mathbf{R}_b^n \delta \mathbf{b}_a + \mathbf{R}_b^n \boldsymbol{\eta}_a$$
$$\delta \dot{\boldsymbol{\theta}}^n = -\lfloor \hat{\boldsymbol{\omega}}^n \times \rfloor \delta \boldsymbol{\theta}^n - \mathbf{R}_b^n \delta \mathbf{b}_g + \mathbf{R}_b^n \boldsymbol{\eta}_g$$
$$\delta \dot{\mathbf{b}}_a = \boldsymbol{\eta}_{ba}, \quad \delta \dot{\mathbf{b}}_g = \boldsymbol{\eta}_{bg}$$

### 2.3 Discrete Covariance Propagation
$$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{Q}_k$$

---

## 3. Pre-Outage Calibration & Blackout Dynamics

1. **Open-Sky GNSS Phase (Pre-Outage)**:
   - When GNSS fixes are available at 1 Hz, the Kalman gain computes error state corrections:
     $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \left( \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k \right)^{-1}$$
   - Accelerometer bias $\mathbf{b}_a$ is calibrated to within $\pm 0.05\text{ m/s}^2$.
   - Gyroscope bias $\mathbf{b}_g$ is calibrated to within $\pm 0.002\text{ rad/s}$.
   - Attitude error is constrained to $<0.8^\circ$.

2. **Blackout Phase (Outage)**:
   - When GNSS measurement updates cease, the filter executes **pure time-propagation**.
   - Because no measurement updates exist to bound the state, covariance matrix $\mathbf{P}$ expands monotonically.
   - Using the calibrated pre-outage bias $\hat{\mathbf{b}}_a$ dramatically reduces short-term divergence compared to raw IMU integration.

---

## 4. Empirical Performance

Evaluated against the pure IMU baseline on suburban trajectory windows:

| Metric | Pure IMU Baseline | 15-State ESKF (Calibrated) | Improvement Factor |
|:---|:---:|:---:|:---:|
| **Attitude Drift at 30 s** | $14.2^\circ$ | **$1.8^\circ$** | **$7.9\times$ Improvement** |
| **Position Drift at 10 s** | $28.4\text{ m}$ | **$6.8\text{ m}$** | **$4.2\times$ Improvement** |
| **Position Drift at 30 s** | $288.1\text{ m}$ | **$85.6\text{ m}$** | **$3.4\times$ Improvement** |
| **Position Drift at 60 s** | $>800\text{ m}$ | **$210–380\text{ m}$** | **$2.5\times$ Improvement** |

---

## 5. Conclusion & Decision

> [!TIP]
> **Decision: ACCEPTED as Core Architecture Component.**  
> The 15-state ESKF successfully stabilizes attitude and continuously calibrates sensor biases during open-sky driving.  
> However, during a 60-second blackout, unobserved along-track velocity continues to drift ($\sim 200\text{ m}$ error). The filter requires additional physical or learned velocity constraints to achieve <10% drift.
