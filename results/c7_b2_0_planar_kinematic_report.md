# Stage C7-B2-0: Offline Planar Kinematic Characterization Audit ($a_y \approx v_x \omega_z$)

**Date**: September 6, 2026  
**Status**: COMPLETED — OFFLINE PHYSICS & SENSOR AUDIT  
**Script**: [`experiments/audit_planar_kinematics_c7_b2_0.py`](../experiments/audit_planar_kinematics_c7_b2_0.py)  
**Master Data**: [`results/c7_b2_0_planar_kinematics.json`](c7_b2_0_planar_kinematics.json)  
**Diagnostic Figure**: [`results/figures/c7_b2_0_planar_kinematic_audit.png`](figures/c7_b2_0_planar_kinematic_audit.png)

---

## Executive Summary & Final Verdict

Following the formal approval of Stage C7-B2-0, we executed a **purely offline physics and sensor characterization audit** to evaluate whether the planar kinematic relationship $a_y \approx v_x \omega_z$ contains usable lateral information to justify designing an ESKF measurement update.

### 🚦 Decision: 🔴 REJECT FOR CONTINUOUS ESKF / 🟡 CONDITIONAL REGIME-GATED ONLY WITH FILTER RE-ARCHITECTING

1. **Physical Validity on Vehicle Instrumentation (Track A: PASS 🟢)**:  
   On CAN/VBOX ground truth, the planar kinematic relationship $e_{\text{kin}}^{\text{VBOX}} = a_y^{\text{VBOX}} - v_x^{\text{VBOX}} \omega_z^{\text{VBOX}}$ holds remarkably tightly:
   - `Vta02` (Suburban, 10,991 epochs): Correlation $r = \mathbf{+0.911}$, $\sigma(e_{\text{kin}}) = \mathbf{0.279\text{ m/s}^2}$, P90 $= 0.433\text{ m/s}^2$.
   - `Vta04` (Highway, 1,789 epochs): Correlation $r = \mathbf{+0.930}$, $\sigma(e_{\text{kin}}) = \mathbf{0.237\text{ m/s}^2}$, P90 $= 0.369\text{ m/s}^2$.
   The vehicle-level physical model is valid across speed and curvature regimes.

2. **Smartphone Feasibility Degradation (Track B: FAIL 🔴)**:  
   On smartphone sensors, lateral acceleration $a_y^{\text{phone}}$ is dominated by non-kinematic transient disturbances (while the exact physical decomposition between chassis vibration, mounting compliance, and roll tilt remains unresolved):
   - Noise multiplier is **$8.5\times\text{--}9.7\times$ higher than vehicle ground truth**: $\sigma(e_{\text{phone,ref}}) = \mathbf{2.375\text{ m/s}^2}$ on `Vta02` and $\mathbf{2.028\text{ m/s}^2}$ on `Vta04` (with peak spikes reaching $15.2\text{ m/s}^2$).
   - Autonomous deployable residual $e_{\text{phone,deploy}} = a_y^{\text{phone}} - \hat{v}_x^{\text{phone}} \hat{\omega}_z^{\text{phone}}$ degrades further to $\sigma = \mathbf{2.572\text{ m/s}^2}$ (`Vta02`) and $\mathbf{2.290\text{ m/s}^2}$ (`Vta04`).

3. **Causal Filtering Tradeoff: Massive Latency (Phase D: CAUTION 🔴)**:  
   Causal low-pass filtering (EMA, $f_c = 0.5\text{--}1.0\text{ Hz}$) reduces residual variance to $\sigma \approx 0.96\text{--}1.32\text{ m/s}^2$, but injects a massive **$-700\text{ to } -800\text{ ms}$ group delay**. As established in C5.5.4, a 700 ms lag causes severe transient overshoot during cornering and braking, degrading dead-reckoning performance.

4. **The Heading Observability Trap (Phase F: THE CRUCIAL RESULT 🔴)**:  
   **A low residual $a_y \approx v_x \omega_z$ does NOT imply heading observability.**  
   Mathematical Jacobian analysis and numerical sweeps reveal:
   - The measurement residual $r_{\text{kin}} = a_y - \hat{v}_x \omega_z$ has **purely second-order sensitivity to heading error $\delta \psi$**:
     $$\left.\frac{\partial r_{\text{kin}}}{\partial \psi}\right|_{\delta\psi=0} \equiv \mathbf{0.000000}$$
   - At a significant heading error of $\delta \psi = 5^\circ$, the kinematic innovation signal is only $\mathbf{0.00119\text{ m/s}^2}$, while smartphone lateral sensor noise is $2.38\text{ m/s}^2$ ($\text{SNR} = \mathbf{-66.0\text{ dB}}$)!
   - In contrast, the standard Non-Holonomic Constraint (NHC, $v_y \approx 0$) has first-order sensitivity $\frac{\partial r_{\text{NHC}}}{\partial \psi} \approx \hat{v}_x = \mathbf{11.8\text{ to } 20.0\text{ m/s}}$, generating a $1.03\text{ m/s}$ signal at $5^\circ$ ($\text{SNR} = \mathbf{+20.2\text{ dB}}$) — **$865\times$ more sensitive than the kinematic constraint**.
   - Putting $a_y \approx v_x \omega_z$ into an ESKF provides **zero first-order heading observability**, while pumping $2.38\text{ m/s}^2$ of high-frequency sensor noise directly into the state covariance.

---

## Diagnostic Overview

*[Stage C7-B2-0 Diagnostic Dashboard — Diagnostic chart]*

---

## Phase A: Track A — VBOX Ground Truth Physical Model Validity

We evaluate the reference kinematic residual:
$$e_{\text{kin}}^{\text{VBOX}} = a_y^{\text{VBOX}} - v_x^{\text{VBOX}} \omega_z^{\text{VBOX}}$$
across all 10,991 epochs of `Vta02` (suburban) and 1,789 epochs of `Vta04` (highway).

### Detailed Distributional Statistics by Driving Regime

#### Journey `Vta02` (Primary Suburban Diagnostic, $T = 1099.1\text{ s}$, $N = 10,991$)
| Regime | Count | Mean [m/s²] | Median [m/s²] | Std [m/s²] | MAE [m/s²] | RMSE [m/s²] | P90 [m/s²] | P95 [m/s²] | Max [m/s²] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 10,991 | **-0.0149** | **-0.0067** | **0.2793** | **0.1954** | **0.2796** | **0.4332** | **0.5905** | **2.2496** |
| Low Speed ($v < 5\text{ m/s}$) | 1,725 | +0.0229 | +0.0900 | 0.3394 | 0.2076 | 0.3402 | 0.5497 | 0.7539 | 2.2496 |
| Medium Speed ($5 \le v < 15\text{ m/s}$) | 7,676 | -0.0098 | -0.0157 | 0.2645 | 0.1888 | 0.2647 | 0.4189 | 0.5558 | 1.7431 |
| Highway Speed ($v \ge 15\text{ m/s}$) | 1,590 | -0.0801 | -0.0858 | 0.2654 | 0.2141 | 0.2773 | 0.4408 | 0.5606 | 1.1741 |
| Straight Driving ($|\omega_z| < 0.02\text{ rad/s}$) | 6,280 | -0.0247 | -0.0079 | **0.1995** | 0.1445 | 0.2010 | **0.3142** | **0.4042** | 1.6506 |
| Moderate Turn ($0.02 \le |\omega_z| < 0.05$) | 2,343 | -0.0173 | -0.0170 | 0.2617 | 0.2019 | 0.2623 | 0.4121 | 0.5139 | 1.6740 |
| Sharp Turn ($|\omega_z| \ge 0.05\text{ rad/s}$) | 2,368 | +0.0137 | +0.0130 | 0.4331 | 0.3240 | 0.4333 | 0.6967 | 0.8654 | 2.2496 |
| Severe Turn ($|\omega_z| \ge 0.10\text{ rad/s}$) | 999 | +0.0114 | +0.0129 | 0.5178 | 0.3869 | 0.5179 | 0.8473 | 1.0450 | 2.2496 |
| Acceleration ($a_x > 1.0\text{ m/s}^2$) | 655 | -0.0953 | -0.0631 | 0.4163 | 0.2981 | 0.4271 | 0.6645 | 0.8519 | 2.0902 |
| Severe Braking ($a_x < -1.5\text{ m/s}^2$) | 367 | +0.0269 | +0.0135 | 0.2671 | 0.1875 | 0.2685 | 0.4221 | 0.5800 | 1.0858 |
| Rough Road (Top 15% Vertical Jitter) | 1,482 | -0.0041 | +0.0000 | 0.3796 | 0.2618 | 0.3796 | 0.6082 | 0.7951 | 2.2496 |

#### Journey `Vta04` (Continuous Highway Confirmation, $T = 178.9\text{ s}$, $N = 1,789$)
| Regime | Count | Mean [m/s²] | Median [m/s²] | Std [m/s²] | MAE [m/s²] | RMSE [m/s²] | P90 [m/s²] | P95 [m/s²] | Max [m/s²] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 1,789 | **-0.0344** | **-0.0329** | **0.2374** | **0.1750** | **0.2399** | **0.3685** | **0.4848** | **1.3238** |
| Low Speed ($v < 5\text{ m/s}$) | 81 | -0.1306 | -0.2178 | 0.4754 | 0.3896 | 0.4930 | 0.7621 | 1.0105 | 1.3238 |
| Medium Speed ($5 \le v < 15\text{ m/s}$) | 1,708 | -0.0299 | -0.0279 | 0.2187 | 0.1649 | 0.2208 | 0.3472 | 0.4506 | 1.1878 |
| Highway Speed ($v \ge 15\text{ m/s}$) | 0 | — | — | — | — | — | — | — | — |
| Straight Driving ($|\omega_z| < 0.02\text{ rad/s}$) | 865 | -0.0453 | -0.0375 | **0.1750** | 0.1350 | 0.1808 | **0.2816** | **0.3600** | 1.0401 |
| Moderate Turn ($0.02 \le |\omega_z| < 0.05$) | 534 | +0.0028 | -0.0074 | 0.2347 | 0.1797 | 0.2347 | 0.3636 | 0.4756 | 0.8883 |
| Sharp Turn ($|\omega_z| \ge 0.05\text{ rad/s}$) | 390 | -0.0614 | -0.0712 | 0.3350 | 0.2574 | 0.3405 | 0.5389 | 0.6518 | 1.3238 |
| Severe Turn ($|\omega_z| \ge 0.10\text{ rad/s}$) | 164 | -0.0653 | -0.0773 | 0.4115 | 0.3133 | 0.4166 | 0.6457 | 0.9612 | 1.3238 |
| Acceleration ($a_x > 1.0\text{ m/s}^2$) | 133 | -0.1702 | -0.1568 | 0.3823 | 0.3019 | 0.4185 | 0.6397 | 0.9836 | 1.3238 |
| Severe Braking ($a_x < -1.5\text{ m/s}^2$) | 61 | -0.0371 | -0.1177 | 0.3462 | 0.2938 | 0.3482 | 0.5802 | 0.6566 | 0.7158 |
| Rough Road (Top 15% Vertical Jitter) | 212 | -0.0826 | -0.0741 | 0.3453 | 0.2527 | 0.3550 | 0.5848 | 0.7637 | 1.3238 |

### Key Physical Validity Takeaways
1. **Model Holds Consistently**: Across both trips, the ground truth physical residual standard deviation is under $0.28\text{ m/s}^2$ ($0.279\text{ m/s}^2$ on Vta02, $0.237\text{ m/s}^2$ on Vta04).
2. **Straight Driving Dispersion**: During straight driving ($|\omega_z| < 0.02\text{ rad/s}$), the residual collapses to $\sigma \approx 0.175\text{--}0.200\text{ m/s}^2$ (P90 $< 0.31\text{ m/s}^2$).
3. **Curvature Tightness**: Even in severe turns ($|\omega_z| \ge 0.10\text{ rad/s}$), the 95th percentile error remains $\le 1.05\text{ m/s}^2$. The vehicle chassis physical model is verified.

---

## Phase B: Track B — Smartphone Feasibility Audit

We now evaluate the two distinct smartphone formulations:
1. **Diagnostic Reference Version**: $e_{\text{phone,ref}} = a_y^{\text{phone}} - v_x^{\text{VBOX}} \omega_z^{\text{VBOX}}$  
   (Isolates phone lateral accelerometer quality against true kinematics).
2. **Autonomous Deployable Version**: $e_{\text{phone,deploy}} = a_y^{\text{phone}} - \hat{v}_x^{\text{phone}} \hat{\omega}_z^{\text{phone}}$  
   (Uses phone-only GPS speed and phone gyroscope).

### Comparative Distributional Statistics

#### Diagnostic Reference Version ($e_{\text{phone,ref}} = a_y^{\text{phone}} - v_x^{\text{VBOX}} \omega_z^{\text{VBOX}}$)
| Trip | Regime | Count | Mean [m/s²] | Median [m/s²] | Std [m/s²] | MAE [m/s²] | RMSE [m/s²] | P90 [m/s²] | Max [m/s²] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vta02** | **Overall** | 10,991 | **-0.0424** | **-0.0027** | **2.3748** | **1.6820** | **2.3752** | **3.7792** | **15.2086** |
| Vta02 | Low Speed ($<5\text{ m/s}$) | 1,725 | +0.0023 | +0.0893 | 1.0409 | 0.6989 | 1.0409 | 1.6571 | 10.2784 |
| Vta02 | Med Speed ($5\text{--}15\text{ m/s}$) | 7,676 | -0.0002 | -0.0183 | 2.3754 | 1.7344 | 2.3754 | 3.7221 | 15.2086 |
| Vta02 | High Speed ($\ge 15\text{ m/s}$) | 1,590 | -0.2946 | -0.3240 | 3.2397 | 2.4958 | 3.2530 | 5.1981 | 14.8178 |
| Vta02 | Straight ($|\omega_z| < 0.02$) | 6,280 | -0.0797 | -0.0304 | 2.2528 | 1.5737 | 2.2543 | 3.5476 | 15.2086 |
| Vta02 | Sharp Turn ($|\omega_z| \ge 0.05$) | 2,368 | +0.1251 | +0.1336 | 2.5812 | 1.8546 | 2.5843 | 4.0149 | 14.4439 |
| **Vta04** | **Overall** | 1,789 | **-0.1780** | **-0.1844** | **2.0284** | **1.5697** | **2.0362** | **3.1682** | **10.7829** |
| Vta04 | Straight ($|\omega_z| < 0.02$) | 865 | -0.1158 | -0.1494 | 1.7697 | 1.3692 | 1.7735 | 2.8593 | 7.4633 |
| Vta04 | Sharp Turn ($|\omega_z| \ge 0.05$) | 390 | -0.3053 | -0.4025 | 2.3332 | 1.8736 | 2.3531 | 3.6993 | 9.6410 |

#### Deployable Autonomous Version ($e_{\text{phone,deploy}} = a_y^{\text{phone}} - \hat{v}_x^{\text{phone}} \hat{\omega}_z^{\text{phone}}$)
| Trip | Regime | Count | Mean [m/s²] | Median [m/s²] | Std [m/s²] | MAE [m/s²] | RMSE [m/s²] | P90 [m/s²] | Max [m/s²] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vta02** | **Overall** | 10,991 | **-0.0724** | **-0.0396** | **2.5724** | **1.7497** | **2.5735** | **4.0055** | **21.3336** |
| Vta02 | Low Speed ($<5\text{ m/s}$) | 1,725 | -0.0179 | +0.0884 | 0.9897 | 0.6686 | 0.9899 | 1.5314 | 10.3595 |
| Vta02 | Med Speed ($5\text{--}15\text{ m/s}$) | 7,676 | -0.0208 | -0.0556 | 2.4921 | 1.7808 | 2.4921 | 3.8915 | 16.0125 |
| Vta02 | High Speed ($\ge 15\text{ m/s}$) | 1,590 | -0.3807 | -0.3937 | 3.8195 | 2.7725 | 3.8384 | 6.3737 | 21.3336 |
| Vta02 | Straight ($|\omega_z| < 0.02$) | 6,280 | -0.1054 | -0.0669 | 2.4445 | 1.6601 | 2.4468 | 3.8831 | 14.9253 |
| Vta02 | Sharp Turn ($|\omega_z| \ge 0.05$) | 2,368 | +0.0752 | +0.1021 | 2.7467 | 1.8564 | 2.7477 | 4.0251 | 21.3336 |
| **Vta04** | **Overall** | 1,789 | **-0.1409** | **-0.0857** | **2.2904** | **1.7406** | **2.2948** | **3.7120** | **13.3163** |
| Vta04 | Straight ($|\omega_z| < 0.02$) | 865 | -0.1030 | -0.0794 | 1.9950 | 1.5294 | 1.9977 | 3.3514 | 7.3064 |
| Vta04 | Sharp Turn ($|\omega_z| \ge 0.05$) | 390 | -0.1444 | -0.1810 | 2.4226 | 1.8562 | 2.4269 | 3.9235 | 9.2807 |

### Why Does the Smartphone Residual Degrade by an Order of Magnitude?
1. **Raw Lateral Acceleration Noise Floor**:  
   The smartphone lateral acceleration has standard deviation $\sigma(a_y^{\text{phone}}) = 2.39\text{ m/s}^2$ (`Vta02`) and $1.95\text{ m/s}^2$ (`Vta04`), whereas the actual vehicle lateral acceleration standard deviation is only $0.677\text{ m/s}^2$ and $0.640\text{ m/s}^2$.
   The smartphone sensor has a **$3.5\times$ higher total standard deviation**, dominated by high-frequency chassis vibration.
2. **Correlation Deficit**:  
   The linear correlation between aligned smartphone $a_y^{\text{phone}}$ and CAN $a_y^{\text{VBOX}}$ is only **$r = +0.153$** on `Vta02` and **$r = +0.042$** on `Vta04`. Unfiltered, the phone lateral channel contains very little vehicle lateral kinematic signal.

---

## Phase C: Dataset Timing & Alignment Audit

Before drawing dynamical conclusions, we audited the time synchronization across all three journeys by computing cross-correlations across lags $\tau \in [-25\text{ s}, +25\text{ s}]$.

| Trip | Duration | Speed Optimal Lag $\tau^*$ | Peak Speed Corr $r_{\max}$ | Zero-Lag Speed Corr $r_0$ | Accel Optimal Lag $\tau^*$ | Peak Accel Corr $r_{\max}$ | Status & Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vta02** | 1099.1 s | **-0.2 s** | **0.996** | **0.996** | **-0.8 s** | +0.211 | 🟢 **SYNCHRONIZED CONTROL**: Timing offset is minimal ($\le 0.2\text{ s}$). Valid for dynamic analysis. |
| **Vta03** | 64.5 s | **-21.5 s** | **0.930** | -0.704 | **+10.9 s** | -0.288 | 🔴 **EXPLICITLY FLAGGED & REJECTED**: Speed lag ($-21.5\text{ s}$) and accel lag ($+10.9\text{ s}$) have contradictory signs. Trip length is truncated to $43\text{ s}$. Unusable for dynamic interpretation. |
| **Vta04** | 178.9 s | **+4.9 s** | **0.860** | 0.488 | **-2.0 s** | +0.148 | 🟡 **TIMING MISMATCH FLAGGED**: Speed cross-correlation exhibits $+4.9\text{ s}$ lag, but accelerometer exhibits $-2.0\text{ s}$ lag. Signal decoupling is structural. |

> [!WARNING]
> **Audit Conclusion on Vta03 & Vta04**: As required, `Vta03` is explicitly flagged and excluded from fine-grained dynamic claims due to dataset-level timing corruption. On `Vta04`, the disagreement between GPS speed lag ($+4.9\text{ s}$) and IMU acceleration lag ($-2.0\text{ s}$) confirms that timing correction cannot fix the smartphone lateral sensor noise.

---

## Phase D: Causal Filtering Audit (Strictly Raw First)

To evaluate whether filtering can recover the lateral signal without corrupting estimation, we test strictly causal filters on `Vta02`:
- **Causal 1st-Order IIR / Exponential Moving Average (EMA)**: $f_c \in \{0.5, 1.0, 2.0, 5.0\}\text{ Hz}$
- **Causal Rolling Median Filter**: $W \in \{0.3, 0.5, 1.0\}\text{ s}$

We record: attenuation ratio ($\operatorname{Var}(y)/\operatorname{Var}(x)$), empirical group delay (via cross-correlation peak), correlation with vehicle ground truth, residual variance, and transient cornering error.

### Strictly Sequential Filtering Tradeoff Table (`Vta02`)

| Filter Type | Parameter | Attenuation (Var Ratio) | Empirical Group Delay | Correlation with VBOX $a_y$ | Residual Std $\sigma(e)$ | Residual MAE | Turn Transient MAE ($|\omega_z| \ge 0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAW BASELINE** | — | **1.000** | **0 ms** | **0.153** | **2.375 m/s²** | **1.682 m/s²** | **1.855 m/s²** |
| Causal EMA | $f_c = 5.0\text{ Hz}$ | 0.933 | -800 ms | 0.158 | 2.293 m/s² | 1.627 m/s² | 1.799 m/s² |
| Causal EMA | $f_c = 2.0\text{ Hz}$ | 0.591 | -800 ms | 0.204 | 1.814 m/s² | 1.302 m/s² | 1.470 m/s² |
| Causal EMA | $f_c = 1.0\text{ Hz}$ | 0.322 | -700 ms | 0.287 | 1.318 m/s² | 0.967 m/s² | 1.146 m/s² |
| Causal EMA | $f_c = 0.5\text{ Hz}$ | 0.183 | **-700 ms** | **0.398** | **0.958 m/s²** | **0.718 m/s²** | **0.894 m/s²** |
| Causal Median | $W = 0.3\text{ s}$ (3 pts) | 0.552 | -700 ms | 0.217 | 1.749 m/s² | 1.256 m/s² | 1.400 m/s² |
| Causal Median | $W = 0.5\text{ s}$ (5 pts) | 0.300 | -500 ms | 0.307 | 1.264 m/s² | 0.930 m/s² | 1.086 m/s² |
| Causal Median | $W = 1.0\text{ s}$ (10 pts) | 0.182 | **-300 ms** | **0.434** | **0.931 m/s²** | **0.699 m/s²** | **0.850 m/s²** |

### Critical Filtering Takeaways
1. **Variance vs. Latency Tradeoff**: Causal filtering reduces instantaneous residual variance (residual standard deviation drops from $2.38\text{ m/s}^2 \to 0.93\text{ m/s}^2$, $-61\%$) but introduces temporal delay and transient distortion; therefore lower residual variance does not necessarily translate into lower navigation drift.
2. **The Severe Latency Penalty**: This filtering introduces a **$-300\text{ to } -700\text{ ms}$ group delay**. In C5.5.4, we proved that feeding a 300–700 ms delayed signal into an in-loop Kalman filter destabilizes dead-reckoning during dynamic maneuvers because the correction arrives after the vehicle has already exited the turn.
3. **Causal Median Outperforms EMA**: The 1.0s causal median achieves slightly lower residual variance ($0.931\text{ m/s}^2$) with only $-300\text{ ms}$ delay, compared to $-700\text{ ms}$ for the 0.5 Hz EMA.

---

## Phase E: Regime Dependence & Conditional SNR Analysis

We analyze whether the lateral kinematic constraint is useless during straight driving but becomes informative during turns, by computing the conditional Signal-to-Noise Ratio:
$$\text{SNR} = 10 \log_{10}\left(\frac{\operatorname{Var}(v_x \omega_z)}{\operatorname{Var}(e_{\text{kin}})}\right)$$

### Conditional Statistics across Curvature and Speed Regimes (`Vta02`)

| Driving Regime | Count | True Kinematic Signal Std [m/s²] | VBOX GT Residual Std [m/s²] | Phone Raw Residual Std [m/s²] | Phone Causal 1Hz Residual Std [m/s²] | VBOX GT SNR [dB] | Phone Raw SNR [dB] | Phone Causal 1Hz SNR [dB] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Straight** ($|\omega_z| < 0.02\text{ rad/s}$) | 6,280 | 0.110 | 0.199 | 2.253 | 1.236 | -5.1 dB | **-26.2 dB** | **-21.0 dB** |
| **Mild Turn** ($0.02 \le |\omega_z| < 0.05$) | 2,343 | 0.381 | 0.262 | 2.466 | 1.324 | +3.3 dB | **-16.2 dB** | **-10.8 dB** |
| **Sharp Turn** ($|\omega_z| \ge 0.05\text{ rad/s}$) | 2,368 | 1.254 | 0.433 | 2.581 | 1.501 | +9.2 dB | **-6.3 dB** | **-1.6 dB** |
| **Severe Turn** ($|\omega_z| \ge 0.10\text{ rad/s}$) | 999 | 1.686 | 0.518 | 2.524 | 1.596 | +10.3 dB | **-3.5 dB** | **+0.5 dB** |
| Low Speed, Straight ($v < 5$, $|\omega_z| < 0.02$) | 1,131 | 0.016 | 0.233 | 0.985 | 0.733 | -23.4 dB | **-35.8 dB** | **-33.4 dB** |
| High Speed, Straight ($v \ge 15$, $|\omega_z| < 0.02$) | 904 | 0.180 | 0.182 | 3.125 | 1.558 | -0.1 dB | **-24.8 dB** | **-18.8 dB** |
| High Speed, Turn ($v \ge 15$, $|\omega_z| \ge 0.05$) | 309 | 1.568 | 0.377 | 3.328 | 1.904 | +12.4 dB | **-6.5 dB** | **-1.7 dB** |
| Med Speed, Turn ($5 \le v < 15$, $|\omega_z| \ge 0.05$) | 1,623 | 1.259 | 0.428 | 2.526 | 1.496 | +9.4 dB | **-6.1 dB** | **-1.5 dB** |

### Key Regime Findings
1. **Straight Driving is Completely Submerged**: In straight driving (which constitutes $57\%$ of the entire journey), the kinematic signal $v_x \omega_z$ has standard deviation of only $0.11\text{ m/s}^2$. The smartphone noise floor is $2.25\text{ m/s}^2$ ($\text{SNR} = \mathbf{-26.2\text{ dB}}$). Any attempt to apply this constraint during straight driving will inject pure white/colored sensor vibration into the Kalman filter.
2. **Sharp Turns Bring Signal Near Parity**: During severe turns ($|\omega_z| \ge 0.10\text{ rad/s}$), the kinematic signal rises to $1.69\text{ m/s}^2$. The filtered phone residual achieves $\text{SNR} = \mathbf{+0.5\text{ dB}}$.
3. **Regime-Gating is Physically Mandatory**: If this constraint were ever used, it could **never** be run continuously. It could only be evaluated when $|\omega_z| \ge 0.05\text{ rad/s}$ and $v \ge 5\text{ m/s}$.

---

## Phase F: The Heading-Error Observability Diagnostic

This is the central test required by the user:
> **"If the estimated heading is wrong, does $r_{\text{kin}}$ contain information that distinguishes the correct heading from the wrong one? Do not call a low residual 'heading observability' automatically."**

### Mathematical Observability Derivation
In the 15-state ESKF, the vehicle velocity is tracked in the navigation frame: $\mathbf{v}^n = [v_N, v_E, v_D]^T$.  
Let the vehicle's true forward ground speed be $v$, moving with true navigation heading $\psi_{\text{true}}$. The true horizontal velocity is:
$$v_N = v \cos\psi_{\text{true}}, \quad v_E = v \sin\psi_{\text{true}}$$

Now suppose the filter's estimated heading has an error $\delta\psi = \hat{\psi} - \psi_{\text{true}}$.  
Projecting $\mathbf{v}^n$ into the estimated vehicle body frame via $\hat{C}_n^v(\hat{\psi})$ yields:
$$\hat{v}_x^v = \cos\hat{\psi} v_N + \sin\hat{\psi} v_E = v \cos(\hat{\psi} - \psi_{\text{true}}) = v \cos(\delta\psi)$$
$$\hat{v}_y^v = -\sin\hat{\psi} v_N + \cos\hat{\psi} v_E = -v \sin(\hat{\psi} - \psi_{\text{true}}) = -v \sin(\delta\psi)$$

#### Comparison 1: The Non-Holonomic Constraint (NHC)
NHC measures the lateral velocity in the vehicle body frame: $z_{\text{NHC}} = 0 \approx \hat{v}_y^v$.  
The measurement innovation is:
$$r_{\text{NHC}}(\delta\psi) = 0 - \hat{v}_y^v = v \sin(\delta\psi) \approx v \cdot \delta\psi$$
The measurement Jacobian with respect to heading error $\delta\psi$ is:
$$\left.\frac{\partial r_{\text{NHC}}}{\partial \psi}\right|_{\delta\psi=0} = v \cos(0) = \mathbf{v}$$
**NHC has a direct, first-order, linear heading sensitivity proportional to forward velocity ($10\text{--}30\text{ m/s}$).**

#### Comparison 2: The Planar Kinematic Constraint ($r_{\text{kin}} = a_y - \hat{v}_x^v \omega_z$)
The true lateral acceleration is $a_y = v \omega_z$.  
When estimated heading has error $\delta\psi$, the predicted forward velocity is $\hat{v}_x^v = v \cos(\delta\psi)$.  
The measurement innovation is:
$$r_{\text{kin}}(\delta\psi) = a_y - \hat{v}_x^v \omega_z = v \omega_z - (v \cos(\delta\psi)) \omega_z = v \omega_z \left(1 - \cos(\delta\psi)\right)$$
Using the Taylor expansion $\cos(\delta\psi) \approx 1 - \frac{\delta\psi^2}{2}$:
$$r_{\text{kin}}(\delta\psi) \approx v \omega_z \cdot \frac{\delta\psi^2}{2}$$
Now compute the measurement Jacobian evaluated at zero heading error:
$$\left.\frac{\partial r_{\text{kin}}}{\partial \psi}\right|_{\delta\psi=0} = \left. v \omega_z \sin(\delta\psi) \right|_{\delta\psi=0} = \mathbf{0.000000}$$

> [!CRITICAL]
> **The Mathematical Proof**:  
> In an Extended Kalman Filter, the update step computes corrections using the **linearized measurement Jacobian** $H = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}}$.  
> Because the first derivative of $r_{\text{kin}}$ with respect to heading $\psi$ is **identically zero**, the measurement matrix entry is:
> $$H_\psi \equiv 0$$
> **The planar kinematic constraint provides ZERO first-order heading observability in an ESKF.**

### Empirical Numerical Innovation Sweep
We simulated heading errors $\delta\psi \in [-15^\circ, +15^\circ]$ across turning regimes on `Vta02` ($v \approx 12\text{ m/s}$, $|\omega_z| \ge 0.05\text{ rad/s}$):

| Injected Heading Error $\delta\psi$ | NHC Innovation $r_{\text{NHC}}$ [m/s] | Kinematic Innovation $r_{\text{kin}}$ [m/s²] | Smartphone Lateral Noise Floor $\sigma(a_y)$ | Kinematic Signal-to-Noise Ratio |
| :---: | :---: | :---: | :---: | :---: |
| $-15.0^\circ$ | -3.054 m/s | +0.01053 m/s² | 2.375 m/s² | -47.1 dB |
| $-10.0^\circ$ | -2.049 m/s | +0.00472 m/s² | 2.375 m/s² | -54.0 dB |
| $-5.0^\circ$ | -1.028 m/s | +0.00119 m/s² | 2.375 m/s² | -66.0 dB |
| $-2.0^\circ$ | -0.412 m/s | +0.00019 m/s² | 2.375 m/s² | -81.9 dB |
| **$0.0^\circ$** | **0.000 m/s** | **0.00000 m/s²** | **2.375 m/s²** | **$-\infty$** |
| $+2.0^\circ$ | +0.412 m/s | +0.00019 m/s² | 2.375 m/s² | -81.9 dB |
| $+5.0^\circ$ | +1.028 m/s | +0.00119 m/s² | 2.375 m/s² | -66.0 dB |
| $+10.0^\circ$ | +2.049 m/s | +0.00472 m/s² | 2.375 m/s² | -54.0 dB |
| $+15.0^\circ$ | +3.054 m/s | +0.01053 m/s² | 2.375 m/s² | -47.1 dB |

### Numerical Interpretation
1. **The Signal is Submerged by $66\text{ dB}$**: Even at a substantial heading error of $5^\circ$ ($0.087\text{ rad}$), the total innovation signal produced by $a_y - v_x \omega_z$ is only **$0.00119\text{ m/s}^2$**. The smartphone lateral sensor noise is $2.375\text{ m/s}^2$. The signal is **2,000 times smaller than the sensor noise**.
2. **Symmetry Prevents Directional Correction**: Because the error is quadratic ($\delta\psi^2$), a $+5^\circ$ heading error and a $-5^\circ$ heading error produce the exact same positive residual ($+0.00119\text{ m/s}^2$). The linear Kalman gain cannot determine whether to turn left or turn right to correct heading!
3. **NHC is $865\times$ More Informative**: NHC generates a $1.028\text{ m/s}$ signed, directional error signal that directly points the filter toward the correct heading.

---

## Synthesis: Epistemic Classification

### 🟢 WHAT WE KNOW (Empirically & Mathematically Proven)
1. **Vehicle-Level Model Validity**: On automotive CAN/VBOX instrumentation, $a_y \approx v_x \omega_z$ is physically valid across all regimes ($\sigma \approx 0.24\text{--}0.28\text{ m/s}^2$, $r \ge 0.91$).
2. **Smartphone Sensor Noise Dominance**: Raw smartphone lateral acceleration is $8.5\times\text{--}9.7\times$ noisier than true vehicle dynamics ($\sigma = 2.03\text{--}2.38\text{ m/s}^2$), with linear correlation $r < 0.16$.
3. **Filtering Latency Hazard**: Causal low-pass filtering can reduce residual variance by $61\%$, but injects $-300\text{ to } -700\text{ ms}$ of group delay, which risks destabilizing dynamic dead-reckoning during turns.
4. **Zero First-Order Heading Observability**: Because $a_y$ and $\omega_z$ are body-frame quantities and $\hat{v}_x = v \cos(\delta\psi)$, the measurement Jacobian with respect to heading is identically zero at $\delta\psi = 0$ ($\left.\frac{\partial r}{\partial \psi}\right|_0 = 0$). The residual depends on heading error only at second order ($\delta\psi^2$), producing a minuscule signal ($0.0012\text{ m/s}^2$ at $5^\circ$) that is submerged $66\text{ dB}$ beneath smartphone sensor noise.
5. **C7-B1-E Correction Confirmed**: The remaining cross-track error is consistent with heading-related unobservability during continuous motion, but the yaw-bias sweep does not identify a constant gyro bias as its dominant cause.

### 🟡 WHAT WE THINK (Strong Plausibility, Not Closed)
1. Rather than observing heading, the only states that $a_y \approx v_x \omega_z$ could theoretically observe are:
   - Accelerometer lateral bias $b_{a,y}$ (if forward velocity and yaw rate are trusted), or
   - Vehicle body roll angle $\phi$ (via gravity leakage $g \sin\phi$).
2. A regime-gated check ($|\omega_z| \ge 0.05\text{ rad/s}$ and $v \ge 10\text{ m/s}$) could potentially serve as an **integrity fault detector** (flagging severe sensor detachment or roll instability), but cannot serve as a continuous heading stabilizer.

### 🔴 WHAT WE DON'T KNOW (Unresolved / Unverified)
1. Whether an attitude-free, scalar speed-consistency update (e.g., estimating speed $\hat{v} = a_y / \omega_z$ during turns) can provide any along-track benefit without contaminating cross-track drift.
2. The exact degree to which suspension roll stiffness contributes to the remaining $a_y$ discrepancy during sharp turns.

---

## Final Recommendation on Stage C7-B2 ESKF Implementation

### Verdict: 🔴 DO NOT INSERT $a_y \approx v_x \omega_z$ INTO THE ESKF FILTER
Based on the six-phase audit:
1. It does **not** provide first-order observability of the heading error that drives lateral drift.
2. It would inject $\sigma \approx 2.38\text{ m/s}^2$ of vibration noise directly into the filter.
3. Filtering it causes $-700\text{ ms}$ latency that harms transient response.
4. The Non-Holonomic Constraint (NHC) already provides $865\times$ higher sensitivity to heading error without requiring noisy lateral accelerometer updates.

**Scientifically, rejecting this constraint preserves the integrity of our filter and prevents adding an unstable, noisy update that cannot physically solve the remaining continuous-motion cross-track drift.**
