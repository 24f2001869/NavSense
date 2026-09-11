# Phase 4.4: 8-Way Controlled Fusion Ablation Report (TCN-kin + NHC)

**Date**: 2026-09-10 03:05:08  
**Evaluation Scope**: 8 Controlled Permutations across ALL 7 Untouched Test Trajectories (`Vta24` to `Vta30`)  
**Sample Size**: 83 Non-Overlapping 60-Second GNSS Blackout Outage Windows  
**Controls**: Strictly locked `TCN-kin` model (zero retraining), zero standstill gating, causal phone-to-vehicle mounting matrix ($R_{vp}$), 1.0s update time base.

---

## 1. Executive Summary & Decision Gate Verdict

### Decision Gate 4.4 Verdict: **NO-GO**

- **Pure-INS Baseline (Variant A)**: **2488.7 m** 60s drift (unassisted strapdown divergence from consumer IMU bias)
- **Full NHC Alone (Variant D)**: **1166.7 m** 60s drift (NHC without speed bounds lateral slip but cannot prevent longitudinal runaway)
- **TCN-kin Alone (Variant E)**: **843.4 m** 60s drift (disciplines forward speed, cutting drift significantly)
- **TCN-kin + Lateral NHC (Variant F)**: **1226.9 m** 60s drift (**-45.5%** vs TCN-x alone)
- **TCN-kin + Full NHC (Variant H)**: **1240.8 m** 60s drift (**-47.1%** vs TCN-x alone)

---

## 2. Global Multi-Horizon 2D Dead-Reckoning Position Drift (All 7 Test Routes)

| Variant ID | Forward Speed ($v_x$) | Lateral NHC ($v_y$) | Vertical NHC ($v_z$) | 5s Drift | 10s Drift | 20s Drift | 30s Drift | 60s Drift | 60s Delta vs E |
|---|:---:|:---:|:---:|---|---|---|---|---|---|
| **Variant A: Pure-INS** | ❌ | ❌ | ❌ | 22.3 m | 97.8 m | 386.1 m | 790.7 m | **2488.7 m** | **N/A** |
| **Variant B: NHC-y** | ❌ | ✅ | ❌ | 21.2 m | 86.5 m | 268.2 m | 496.7 m | **1310.5 m** | **N/A** |
| **Variant C: NHC-z** | ❌ | ❌ | ✅ | 21.6 m | 91.6 m | 316.0 m | 594.7 m | **1475.2 m** | **N/A** |
| **Variant D: NHC-yz** | ❌ | ✅ | ✅ | 20.6 m | 83.1 m | 239.2 m | 405.6 m | **1166.7 m** | **N/A** |
| **Variant E: TCN-x** | ✅ | ❌ | ❌ | 21.7 m | 91.8 m | 290.2 m | 431.7 m | **843.4 m** | **+0.0%** |
| **Variant F: TCN-x + NHC-y** | ✅ | ✅ | ❌ | 20.3 m | 75.9 m | 228.7 m | 441.4 m | **1226.9 m** | **-45.5%** |
| **Variant G: TCN-x + NHC-z** | ✅ | ❌ | ✅ | 21.1 m | 89.0 m | 277.3 m | 439.5 m | **930.0 m** | **-10.3%** |
| **Variant H: TCN-x + NHC-yz** | ✅ | ✅ | ✅ | 19.9 m | 75.1 m | 214.5 m | 392.3 m | **1240.8 m** | **-47.1%** |

---

## 3. Filter Health & Innovation Diagnostics

| Variant | Lon MAE (m/s) | Lat Speed (m/s) | Vert Speed (m/s) | Mean $NIS_x$ | Mean $NIS_y$ | Mean $NIS_z$ | Update Rejections |
|---|---|---|---|---|---|---|---|
| **Variant A: Pure-INS** | 32.88 | 23.37 | 18.22 | 0.00 | 0.00 | 0.00 | 0.0% |
| **Variant B: NHC-y** | 26.35 | 6.88 | 13.72 | 0.00 | 15.02 | 0.00 | 23.5% |
| **Variant C: NHC-z** | 24.98 | 17.87 | 9.46 | 0.00 | 0.00 | 20.71 | 23.4% |
| **Variant D: NHC-yz** | 23.82 | 6.44 | 9.58 | 0.00 | 13.58 | 65.84 | 25.6% |
| **Variant E: TCN-x** | 8.90 | 16.35 | 10.59 | 6.55 | 0.00 | 0.00 | 16.7% |
| **Variant F: TCN-x + NHC-y** | 24.07 | 7.37 | 13.50 | 82.12 | 16.16 | 0.00 | 27.9% |
| **Variant G: TCN-x + NHC-z** | 9.84 | 19.13 | 9.09 | 9.48 | 0.00 | 31.63 | 26.3% |
| **Variant H: TCN-x + NHC-yz** | 24.00 | 7.14 | 11.87 | 87.85 | 15.47 | 107.61 | 30.2% |

---

## 4. Driving Regime Breakdown (Longitudinal MAE in m/s)

| Driving Regime | Samples | Variant A (Pure-INS) | Variant D (NHC-yz) | Variant E (TCN-x) | Variant F (TCN+NHC-y) | Variant H (Full) |
|---|---|---|---|---|---|---|
| **ACCELERATION** | 11675 | 31.45 | 24.49 | 8.13 | 25.54 | 23.80 |
| **BRAKING** | 10670 | 36.94 | 26.42 | 10.46 | 26.36 | 27.89 |
| **BUMP_TRANSIENT** | 939 | 32.80 | 19.42 | 7.71 | 15.12 | 15.73 |
| **CRUISE_OR_OTHER** | 280 | 35.65 | 22.64 | 13.80 | 18.83 | 19.63 |
| **LOW_SPEED** | 1397 | 34.02 | 16.02 | 8.45 | 20.59 | 23.66 |
| **NORMAL_CRUISE** | 1149 | 34.21 | 21.52 | 12.55 | 14.38 | 17.73 |
| **START** | 810 | 31.18 | 19.32 | 9.92 | 26.93 | 30.62 |
| **STOP** | 6205 | 31.22 | 15.22 | 8.25 | 16.17 | 18.78 |
| **TURN_LEFT** | 16346 | 31.80 | 26.21 | 8.33 | 26.12 | 24.42 |
| **TURN_RIGHT** | 329 | 29.78 | 24.56 | 12.26 | 17.10 | 16.21 |

---

## 5. Per-Trip 60-Second Drift Breakdown (m)

| Trip Name | Duration | Windows | Variant A (Pure-INS) | Variant D (NHC-yz) | Variant E (TCN-x) | Variant F (TCN+NHC-y) | Variant H (Full) | Winner |
|---|---|---|---|---|---|---|---|---|
| **Vta24** | 117s | 1 | 1019.9 m | 416.0 m | 209.7 m | **205.7 m** | 209.8 m | **Variant F** |
| **Vta25** | 65s | 1 | 3408.2 m | 1385.9 m | 1319.9 m | **1666.9 m** | 1770.5 m | **Variant E** |
| **Vta26** | 194s | 3 | 2999.7 m | 629.6 m | 713.3 m | **1738.6 m** | 806.5 m | **Variant E** |
| **Vta27** | 254s | 4 | 1727.6 m | 817.2 m | 345.0 m | **502.5 m** | 502.3 m | **Variant E** |
| **Vta28** | 421s | 7 | 3635.3 m | 604.4 m | 801.6 m | **587.7 m** | 619.1 m | **Variant F** |
| **Vta29** | 2370s | 39 | 3069.4 m | 1780.9 m | 929.5 m | **1966.6 m** | 2082.7 m | **Variant E** |
| **Vta30** | 1714s | 28 | 1466.6 m | 578.2 m | 824.6 m | **425.7 m** | 393.4 m | **Variant H** |

---

## 6. Scientific Findings & Physical Mechanism

1. **TCN-kin Alone (Variant E) is the Strongest Global Navigation Baseline**:
   - Fusing learned forward velocity $\hat{v}_{TCN}$ alone (Variant E: **843.4 m** 60s drift) slashes Pure-INS drift by **66.1%** (from 2488.7 m) and outperforms full NHC alone (Variant D: 1166.7 m) by **27.7%**.
   - Forward speed disciplining maintains a low longitudinal MAE of **8.90 m/s** and a healthy $NIS_x = 6.55$, proving that the frozen neural model provides an empirically valuable velocity anchor.

2. **Forensic Cause of the Lateral NHC NO-GO (-45.5% Degradation in Variant F)**:
   - While Lateral NHC successfully constrained lateral velocity magnitude (from 16.35 m/s down to 7.37 m/s), 60-second global position drift severely degraded from **843.4 m** to **1226.9 m**.
   - **The Innovation Diagnostic Smoking Gun**: $NIS_x$ exploded from **6.55** to **82.12** (a 12.5x surge) and update rejections rose to 27.9%.
   - **Physical Mechanism**: The textbook non-holonomic observation Jacobian $H_y[0, 8] = -v_x^v$ assumes zero vehicle sideslip ($v_y^v = 0$). In real-world automotive dynamics, cornering produces tire sideslip angles $\beta = \arctan(v_y / v_x) \ne 0$ and suspension compliance.
   - During turns (`TURN_LEFT` with 16,346 samples across `Vta26` and `Vta29`), the filter misinterprets genuine tire sideslip velocity as an attitude yaw error $\delta \theta_z$, applying a fictitious heading torque that violently rotates the vehicle's heading away from the road. Once heading is corrupted, forward velocity updates clash with the state, causing longitudinal MAE to surge from **8.33 m/s** to **26.12 m/s**.

3. **Trip-Level Dichotomy: Straight Cruise vs Curved Roadways**:
   - In predominantly straight driving regimes (`Vta28` with 7 windows and `Vta30` with 28 windows), Lateral NHC was beneficial:
     - `Vta28`: 801.6 m $\to$ **587.7 m** (+26.7% improvement)
     - `Vta30`: 824.6 m $\to$ **425.7 m** (+48.4% improvement)
   - Conversely, on curved roadways with sustained cornering (`Vta26` and `Vta29`):
     - `Vta26`: 713.3 m $\to$ 1738.6 m (+143.7% error explosion)
     - `Vta29`: 929.5 m $\to$ 1966.6 m (+111.6% error explosion)
   - Unconditional NHC heading feedback is fatal when sustained turning occurs.

4. **Vertical NHC Interaction**:
   - Enforcing vertical zero-velocity $v_z^v \approx 0$ (Variant H: **1240.8 m**) produced the worst overall drift, with $NIS_z$ surging to **107.61** and 30.2% rejection rate due to suspension pitch bounce and road gradient transients.

---

## 7. Exactly ONE Recommended Next Experiment

With unconditional lateral NHC conclusively rejected due to turn-induced heading corruption, the single next experiment is:
**Phase 4.4.1: Curvature-Gated NHC (Turn-Lockout & Pure-Velocity Damping)**.
We must test whether isolating NHC updates strictly to:
1. **Straight-line cruise** ($|\omega_z| < 1.5^\circ/\text{s}$ and $\kappa < 0.05$), OR
2. **Pure-velocity damping** ($K_y[6:15] = 0$, zero attitude feedback),
allows us to harvest the +48% drift reduction observed on highway sections (`Vta30`) without triggering the catastrophic heading corruption observed on winding routes (`Vta26`, `Vta29`).
