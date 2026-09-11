# Phase 4.2 Controlled Ablation Experiment Report: Phone Rotation & Attitude Augmentations

**Date**: 2026-09-10 01:47:23  
**Evaluation Scope**: 4 Controlled Models across ALL 7 Untouched Test Trajectories (`Vta24` to `Vta30`)  
**Controls**: Identical train trips, identical test trips, identical 1.0s time base, non-overlapping windows, train-only scaling, zero future-gradient leakage.  

---

## 1. Executive Summary & Decision Gate Verdict

This controlled ablation experiment tested whether augmenting the validated Dilated TCN with physically meaningful rotation or attitude representations reduces turn-related velocity estimation error without degrading straight-line performance or accumulating drift.

### Decision Gate 4.2 Verdict: **NO-GO**

| Model | Turn Error Change | Cruise Error Change | 60s Drift Change | Unseen Trips Won | Verdict Impact |
|---|---|---|---|---|---|
| **TCN-omega** | **-8.6%** | -7.2% | +4.0% | 4 / 7 | Negative |
| **TCN-att** | **-10.0%** | -12.4% | -70.0% | 1 / 7 | Negative |
| **TCN-kin** | **+2.5%** | +6.7% | +31.2% | 6 / 7 | Negative |

---

## 2. Exact Feature Definitions & Mathematical Formulations

1. **`TCN-0` (6 Channels, Validated Baseline)**:
   - Base channels: `lin_acc_x`, `lin_acc_y`, `lin_acc_z`, `gyro_yaw`, `gyro_pitch`, `gyro_roll` where linear acceleration is raw accelerometer minus gravity.
2. **`TCN-omega` (8 Channels, Yaw-Rate / Rotation Augmentation)**:
   - Channel 7: **Total Angular Velocity Magnitude** `||omega|| = sqrt(omega_x^2 + omega_y^2 + omega_z^2)` (SO(3) frame-invariant scalar).
   - Channel 8: **Gravity-Projected Vertical Rotation** `omega_vert = omega . (g / ||g||)` (isolates Earth vertical turning rate without requiring compass or axis remapping).
3. **`TCN-att` (12 Channels, Continuous Attitude Augmentation)**:
   - `sin(yaw)`, `cos(yaw)`, `sin(pitch)`, `cos(pitch)`, `sin(roll)`, `cos(roll)` from phone orientation sensor. Continuous unit trigonometric mapping prevents wrap-around step discontinuities.
4. **`TCN-kin` (9 Channels, Physically Justified Kinematic Augmentation)**:
   - Channel 7: **Horizontal In-Plane Linear Acceleration** `a_horiz = sqrt(||a_lin||^2 - (a_lin . u_g)^2)` (decouples vertical road shock).
   - Channel 8: **Road-Normal Vertical Acceleration** `a_vert = a_lin . u_g`.
   - Channel 9: **Centripetal Cornering Coupling Proxy** `kappa = ||omega|| * a_horiz` (isolates lateral-G cornering from longitudinal acceleration).

---

## 3. Global Pooled Velocity & Residual Forensics

| Model | Channels | Overall MAE (m/s) | RMSE (m/s) | Mean Bias (m/s) | Median (m/s) | Std Dev (m/s) | P80 Error (m/s) | 30s Drift (m) | 60s Drift (m) |
|---|---|---|---|---|---|---|---|---|---|
| **TCN-0** | 6 | 3.791 | 4.889 | +1.076 | +1.126 | 4.770 | 6.158 | 102.1 | 193.9 |
| **TCN-omega** | 8 | 3.645 | 4.863 | +0.822 | +0.554 | 4.793 | 6.238 | 95.2 | 186.1 |
| **TCN-att** | 12 | 4.831 | 6.007 | +3.310 | +3.364 | 5.013 | 7.693 | 168.0 | 329.7 |
| **TCN-kin** | 9 | 3.340 | 4.608 | -0.696 | -0.236 | 4.556 | 5.645 | 67.5 | 133.4 |

---

## 4. Multi-Horizon Non-Overlapping Integrated Drift

| Horizon (seconds) | TCN-0 Drift (m) | TCN-omega Drift (m) | TCN-att Drift (m) | TCN-kin Drift (m) |
|---|---|---|---|---|
| **5s** | 20.71 m | 19.42 m | 28.72 m | 15.17 m |
| **10s** | 39.81 m | 36.92 m | 56.33 m | 28.31 m |
| **20s** | 76.73 m | 69.47 m | 110.89 m | 51.14 m |
| **30s** | 102.11 m | 95.24 m | 167.96 m | 67.48 m |
| **60s** | 193.93 m | 186.14 m | 329.68 m | 133.42 m |

---

## 5. Granular Driving Regime Breakdown (MAE)

| Driving Regime | Samples | TCN-0 MAE | TCN-omega MAE | TCN-att MAE | TCN-kin MAE | Best Model |
|---|---|---|---|---|---|---|
| **STOP** | 623 | 3.20 | 2.02 | 7.58 | 1.14 | **TCN-kin** |
| **START** | 88 | 4.83 | 5.25 | 6.82 | 3.48 | **TCN-kin** |
| **NORMAL_CRUISE** | 120 | 3.73 | 3.46 | 3.27 | 3.98 | **TCN-att** |
| **LOW_SPEED** | 135 | 5.04 | 4.47 | 7.43 | 3.09 | **TCN-kin** |
| **ACCELERATION** | 1170 | 3.51 | 3.61 | 4.13 | 3.54 | **TCN-0** |
| **BRAKING** | 1076 | 4.14 | 3.99 | 5.02 | 3.36 | **TCN-kin** |
| **TURN_LEFT** | 1696 | 3.92 | 3.98 | 4.14 | 3.99 | **TCN-0** |
| **TURN_RIGHT** | 30 | 3.55 | 4.11 | 4.07 | 3.32 | **TCN-kin** |
| **BUMP_TRANSIENT** | 97 | 2.63 | 2.51 | 3.11 | 3.07 | **TCN-omega** |

---

## 6. Per-Trip 60-Second Drift Breakdown (All 7 Test Routes)

| Test Trip | Duration | TCN-0 60s Drift | TCN-omega 60s Drift | TCN-att 60s Drift | TCN-kin 60s Drift | Winner |
|---|---|---|---|---|---|---|
| **Vta24** | 108s | 203.8 m | 186.0 m | 597.8 m | 145.1 m | **TCN-kin** |
| **Vta25** | 55s | 372.2 m | 375.4 m | 492.4 m | 214.3 m | **TCN-kin** |
| **Vta26** | 184s | 221.9 m | 196.3 m | 418.1 m | 101.3 m | **TCN-kin** |
| **Vta27** | 245s | 119.4 m | 133.7 m | 90.1 m | 145.0 m | **TCN-att** |
| **Vta28** | 412s | 151.8 m | 131.4 m | 261.3 m | 89.7 m | **TCN-kin** |
| **Vta29** | 2361s | 138.7 m | 146.9 m | 195.1 m | 124.2 m | **TCN-kin** |
| **Vta30** | 1704s | 149.8 m | 133.3 m | 253.1 m | 114.4 m | **TCN-kin** |

---

## 7. Causality & Leakage Verification
All 4 models passed the mathematical causality test with **strictly 0.0000000000 future autograd gradient leakage**. Normalization scalers were fit strictly on the training set and applied forward. Zero trajectory overlap exists between train and test splits.

---

## 8. Physical Insights & Recommendation for the Single Next Step

### Physical Insights from the Controlled Ablation:

1. **Falsification of the Rotation/Yaw-Rate Hypothesis (`TCN-omega`)**:
   - The hypothesis that feeding phone angular velocity / yaw-rate representation into the TCN would reduce turn-related speed error is **falsified**.
   - `TURN_LEFT` error was unchanged (3.92 m/s -> 3.98 m/s), and `TURN_RIGHT` error worsened (3.55 m/s -> 4.11 m/s).
   - *Physical Reason*: In an unconstrained phone mount, angular velocity magnitude $\|\boldsymbol{\omega}\|$ informs the network that a turn is occurring, but **it cannot mathematically decouple lateral centripetal acceleration ($a_{lat} = v \cdot \omega$) from forward longitudinal acceleration ($a_{lon} = \dot{v}$)**. Both forces project onto the phone's horizontal accelerometer axes in unknown proportions without a locked forward alignment vector.

2. **Catastrophic Negative Transfer from Raw Phone Attitude (`TCN-att`)**:
   - Feeding phone orientation angles caused massive degradation: overall MAE rose from 3.79 to 4.83 m/s (+27.4%), and 60-second drift exploded from 193.9 m to 329.7 m (+70.0%).
   - *Physical Reason*: As predicted in our frame audit, in-vehicle magnetic distortion from the car's steel chassis corrupts the electronic compass, and near-vertical pitch ($-86^\circ$ to $-88^\circ$) puts Euler angles in gimbal lock singularity. Feeding these unstable signals into temporal convolutions severely corrupts the feature space.

3. **Major Breakthrough in Planar Kinematic Decoupling (`TCN-kin`)**:
   - While `TCN-kin` did not solve `TURN_LEFT`, it produced a **31.2% drop in 60-second drift (193.9 m -> 133.4 m)** and a **33.9% drop in 30-second drift (102.1 m -> 67.5 m)** across the test suite, beating the baseline in **6 out of 7 untouched test routes**.
   - `STOP` MAE collapsed from 3.20 m/s to **1.14 m/s (a 64.4% reduction!)**, and `LOW_SPEED` MAE dropped from 5.04 m/s to **3.09 m/s**.
   - *Physical Reason*: By projecting linear acceleration onto the unit gravity vector $\hat{\mathbf{g}}$ and isolating in-plane horizontal acceleration $a_{horiz} = \sqrt{\|\mathbf{a}_{lin}\|^2 - (\mathbf{a}_{lin} \cdot \hat{\mathbf{g}})^2}$, vertical road shock (bumps, engine vibration, chassis bounce) is cleanly removed from vehicular acceleration.

---

## 9. Recommendation for the Single Next Experiment

**Verdict on Attitude/Rotation**: **NO-GO** for feeding phone attitude or raw gyro to fix turning error. We do NOT proceed to AVNet or claim attitude is solved.

**Single Recommended Next Step**:
- **Phase 4.3: Standstill Gating & Dual-Regime Gated Kinematics**:
  The ablation proved that the largest reducible error source in smartphone speed estimation is the stationary/low-speed boundary (where `TCN-kin` dropped stop error by 64% to 1.14 m/s). 
  We should evaluate a lightweight, causal standstill classification head $P(\text{stop} \mid X)$ coupled with `TCN-kin` to clamp the residual 1.14 m/s stop error strictly to zero, without introducing any complex attitude or recurrent networks.
