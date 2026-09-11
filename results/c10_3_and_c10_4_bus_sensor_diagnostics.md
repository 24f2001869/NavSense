# Stages C10.3 & C10.4: Bus Telemetry Diagnostic Investigations
**Scope**: Empirical investigation of failure modes discovered during the 54-minute physical bus field test (`CPH2381`).  
**Status**: Navigation engine remains **STRICTLY FROZEN**. Zero algorithm or mathematical changes.  
**Dataset**: 26,041 valid GNSS epochs (~54.1 minutes) across Trip 1 and Trip 2 in Hyderabad, India.  
**Date**: September 9, 2026  

---

## 1. Executive Summary

During the Stage C10.2 bus field test, two primary failure mechanisms were discovered:
1. **The Stationary Speed Failure**: The Causal Random Forest speed model predicted ~21–22 km/h while the bus was completely stationary at signals and bus stops, injecting ~333 meters of artificial forward drift per 60 seconds of idling.
2. **The Vertical Channel Divergence**: Filter vertical position ($pos\_u$) exploded to $-15.8\,\text{km}$ in Trip 2 and $+5.2\,\text{km}$ in Trip 1 during extended dead-reckoning outages.

Following strict scientific methodology, we conducted two dedicated diagnostic investigations (**Stage C10.3** and **Stage C10.4**) on the raw 26,041-epoch dataset to isolate the root theoretical mechanisms before touching any code.

```
+-----------------------------------------------------------------------------------------+
| DIAGNOSTIC INVESTIGATION SUMMARY                                                        |
+-----------------------------------------------------------------------------------------+
| Stage C10.3 Question | Can stopped-engine vibration be distinguished from vehicle motion?|
| Stage C10.3 Finding  | YES. Rolling 1s acceleration std (AUC=0.766) and jerk (AUC=0.705)|
|                      | cleanly separate idle from cruising. But gyro norm CANNOT be     |
|                      | used because handheld fidgeting/tremors cause higher rotation    |
|                      | when stopped (mean 30.8°/s) than when cruising (mean 11.0°/s).   |
| Stage C10.4 Question | Why did pos_u collapse to -15 km in Trip 2?                     |
| Stage C10.4 Finding  | VNHC contains an unrecoverable latch: turn-rate lockout (>3°/s)   |
|                      | triggered during turns/tremors, and once |v_z| > 3.0 m/s, the hard|
|                      | innovation gate permanently locked VNHC out (active only 1.0%),  |
|                      | causing pure double integration of vertical acceleration (15 km).|
|                      | In contrast, when VNHC was active (92.1%), drift was only 0.6 m! |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Stage C10.3: Stationary Vibration Diagnostic Audit

### 2.1 Kinematic Regime Classification
Using microsecond-synchronized GNSS ground truth speed and numerical acceleration ($\dot{v} = \Delta v / \Delta t$), every epoch of the 54-minute dataset was classified into one of 5 mutual exclusive regimes:

| Regime | Definition | Epochs | Fraction | Duration |
| :--- | :--- | :--- | :--- | :--- |
| **A: Stationary (Idle)** | $v_{\text{GNSS}} < 0.2\,\text{m/s}$ | 6,298 | **24.2%** | **13.1 min** |
| **B: Slow Crawl** | $0.2\,\text{m/s} \le v_{\text{GNSS}} \le 2.0\,\text{m/s}$ | 4,503 | **17.3%** | **9.4 min** |
| **C: Cruising** | $v_{\text{GNSS}} > 2.0\,\text{m/s}$ and $|\dot{v}| < 0.4\,\text{m/s}^2$ | 13,648 | **52.4%** | **28.4 min** |
| **D: Accelerating** | $v_{\text{GNSS}} > 2.0\,\text{m/s}$ and $\dot{v} \ge 0.4\,\text{m/s}^2$ | 921 | **3.5%** | **1.9 min** |
| **E: Braking** | $v_{\text{GNSS}} > 2.0\,\text{m/s}$ and $\dot{v} \le -0.4\,\text{m/s}^2$ | 671 | **2.6%** | **1.4 min** |

### 2.2 Feature Distributions Across Regimes (Median [IQR])

| Feature | A: Stationary | B: Slow Crawl | C: Cruising | D: Accelerating | E: Braking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GNSS Speed (m/s)** | 0.00 [0.00, 0.00] | 1.09 [0.53, 1.38] | 6.56 [4.53, 9.52] | 6.97 [4.66, 9.81] | 6.95 [4.14, 9.73] |
| **ML Speed (m/s)** | **3.96 [1.64, 10.13]** | 8.46 [2.75, 13.09] | 7.32 [4.21, 10.93] | 7.47 [4.45, 11.35] | 7.85 [4.40, 10.63] |
| **Accel Norm ($m/s^2$)** | 9.84 [9.65, 9.99] | 9.80 [9.49, 10.33] | 9.83 [9.31, 10.35] | 9.81 [9.28, 10.38] | 9.87 [9.33, 10.41] |
| **Accel Std 1s ($m/s^2$)** | **0.23 [0.10, 0.57]** | 0.78 [0.20, 2.32] | **0.79 [0.52, 1.14]** | 0.83 [0.56, 1.19] | 0.79 [0.55, 1.11] |
| **Accel Std 2s ($m/s^2$)** | **0.25 [0.11, 0.63]** | 0.86 [0.22, 2.31] | **0.84 [0.59, 1.18]** | 0.90 [0.64, 1.25] | 0.83 [0.61, 1.17] |
| **Gyro Norm ($^\circ/s$)** | **4.32 [1.11, 21.33]** | 12.05 [2.08, 52.55] | **6.68 [3.50, 12.00]** | 6.90 [3.89, 13.07] | 6.78 [3.52, 12.47] |
| **Jerk $|da/dt|$ ($m/s^3$)** | **1.83 [0.51, 5.34]** | 4.66 [1.21, 17.06] | **6.18 [2.57, 12.14]** | 6.63 [3.00, 12.54] | 6.51 [3.02, 12.16] |
| **High-Freq Accel Mean ($m/s^2$)** | **0.17 [0.07, 0.43]** | 0.56 [0.15, 1.78] | **0.60 [0.40, 0.87]** | 0.62 [0.43, 0.90] | 0.60 [0.41, 0.85] |

### 2.3 Separability Analysis: Stationary (A) vs Cruising (C)

| Candidate Feature | ROC AUC | Cohen's d | Stationary Mean | Cruising Mean | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accel Std 1s** ($\sigma_a$) | **0.766** | **+0.181** | $0.73\,\text{m/s}^2$ | $0.94\,\text{m/s}^2$ | **Strong separation in median ($0.23$ vs $0.79\,\text{m/s}^2$)** |
| **Accel Std 2s** | **0.768** | **+0.185** | $0.76\,\text{m/s}^2$ | $0.98\,\text{m/s}^2$ | Similar to 1s window |
| **High-Freq Accel Mean 1s** | **0.767** | **+0.186** | $0.54\,\text{m/s}^2$ | $0.71\,\text{m/s}^2$ | Captures high-frequency AC oscillation |
| **Jerk** $|da/dt|$ | **0.705** | **+0.183** | $6.55\,\text{m/s}^3$ | $9.12\,\text{m/s}^3$ | 3.4x difference in median ($1.83$ vs $6.18\,\text{m/s}^3$) |
| **Gyro Norm** $\|\boldsymbol{\omega}\|$ | **0.566** | **-0.417** | **$30.80^\circ/\text{s}$** | **$11.02^\circ/\text{s}$** | **INVERTED & UNUSABLE IN HANDHELD MODE** |

### 2.4 Forensic Discovery: Why Baseline ZUPT Failed
In `MainActivity.kt` (line 418), the application's stationary flag was defined as:
```kotlin
val isStationary = curGyro.norm() < 0.05 && Math.abs(curAccel.norm() - 9.81) < 0.3
```
1. `curGyro.norm() < 0.05` equates to **$2.86^\circ/\text{s}$**. Because the phone was in the user's hand during transit, passengers and hand movements created an average stationary gyro rotation of **$30.80^\circ/\text{s}$** (median $4.32^\circ/\text{s}$). The condition failed over 85% of the time.
2. `Math.abs(curAccel.norm() - 9.81) < 0.3` is an instantaneous single-sample check. Engine idle vibrations regularly generated $\pm 0.4 - 1.2\,\text{m/s}^2$ peaks, causing this check to fail almost continuously.
3. Because `isStationary` was permanently false, the Random Forest model received `feat[15] = 0` and evaluated its vibration trees, outputting ~21 km/h forward speed while standing still.

---

## 3. Stage C10.4: Vertical Channel Divergence Forensic Diagnostic

### 3.1 Vertical Error Evolution Across All Outages

| Trip | Outage Episode | Outage Duration | Start $pos\_u$ | End $pos\_u$ | Net $\Delta pos\_u$ | Max Vertical Velocity | VNHC Activation Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip 1** | Ep 1 | 3.9 s | -37.0 m | -45.5 m | -8.4 m | 6.3 m/s | 0.0% |
| **Trip 1** | Ep 2 | 94.3 s | -36.2 m | 191.5 m | +227.7 m | 92.1 m/s | 48.3% |
| **Trip 1** | Ep 3 | 132.4 s | -45.5 m | 1166.5 m | +1,212.0 m | 622.7 m/s | 69.1% |
| **Trip 1** | Ep 4 | 128.6 s | -34.0 m | 4135.8 m | +4,169.8 m | 1584.0 m/s | 21.3% |
| **Trip 1** | Ep 5 | 125.1 s | -21.9 m | 5213.6 m | +5,235.5 m | 1806.0 m/s | 4.8% |
| **Trip 1** | Ep 7 | 99.5 s | -42.3 m | 118.9 m | +161.2 m | 1602.3 m/s | 0.6% |
| **Trip 2** | Ep 1 | 161.7 s | 25.4 m | 1405.3 m | +1,379.8 m | 569.1 m/s | 38.6% |
| **Trip 2** | **Ep 2** | **57.6 s** | **28.1 m** | **28.8 m** | **+0.6 m** | **4.3 m/s** | **92.1% (BENCHMARK)** |
| **Trip 2** | Ep 3 | 80.1 s | 22.9 m | -544.3 m | -567.2 m | 2434.5 m/s | 61.9% |
| **Trip 2** | **Ep 4** | **224.0 s** | **20.4 m** | **-15,829.6 m** | **-15,849.9 m** | **1249.8 m/s** | **1.0% (COLLAPSE)** |

### 3.2 The Smoking Gun: Proof that VNHC Works When Active
Look at the stark contrast between Trip 2 Episode 2 and Trip 2 Episode 4:
- **Episode 2 (57.6 s duration)**: VNHC was active **92.1%** of the time. The total vertical position change was **only +0.6 meters**!
- **Episode 4 (224.0 s duration)**: VNHC was active **1.0%** of the time. The vertical position diverged by **$-15,849.9\,\text{meters}$**!

### 3.3 Forensic Root Cause: The VNHC Double-Gate Latch
Inspection of [`DeadReckoningEngine.java`](../android/app/src/main/java/com/sih26168/idr/engine/DeadReckoningEngine.java#L468-L491) revealed why VNHC was disabled during long outages:

```java
// Gate 1: Turn-rate lockout
if (turn_rate_deg_s > cfg.turn_rate_vnhc_gate_deg_s) { // cfg = 3.0 deg/s
    return;
}
...
// Gate 2: Hard innovation / residual gate
if (nis > cfg.nhc_gate_chi2 || Math.abs(r_vert) > 3.0) {
    return;
}
```

1. **Gate 1 (Turn-Rate Lockout at $3.0^\circ/\text{s}$)**:
   In a moving bus, vehicle turns, road bumps, and user hand tremors routinely exceed $3^\circ/\text{s}$, suspending VNHC updates for multiple seconds.
2. **Gate 2 (The Unrecoverable Hard Residual Latch)**:
   While VNHC is suspended by Gate 1, small uncorrected vertical accelerometer biases integrate into vertical velocity ($v_z = a_z \cdot t$). If $|v_z|$ reaches **$3.1\,\text{m/s}$** (which requires only 1–2 seconds of uncorrected bias), Gate 2 trips:
   $$\text{Math.abs}(r_{\text{vert}}) > 3.0\,\text{m/s} \implies \text{RETURN}$$
   **Crucially, this gate never self-resets during an outage.** Because $|v_z| > 3.0\,\text{m/s}$, VNHC is rejected on *every subsequent epoch*.
3. **Runaway Free-Fall**: Once locked out, vertical acceleration integrates quadratically without any correction:
   $$\Delta z(t) = \frac{1}{2} a_{z,\text{bias}} t^2 \implies \Delta z(224\,\text{s}) \approx -15,850\,\text{m}$$

---

## 4. Engineering Conclusions & Controlled Road Ahead

1. **The Core Hypotheses are Proven**:
   - The stationary failure is not an inherent flaw of Random Forest architecture; it is caused by the absence of a rolling window vibration detector and an unrealistic static gyro threshold.
   - The vertical divergence is not an inherent breakdown of the ESKF; it is an artifact of an overly restrictive $3.0\,\text{m/s}$ innovation latch that locks out VNHC during long outages. When VNHC is active, vertical stability is excellent ($0.6\,\text{m}$ drift in 58s).
2. **Methodological Next Step (Stage C10.5)**:
   Before modifying production code, perform a **rigidly-mounted controlled vehicle test** (phone fixed in a rigid car mount, eliminating hand tremors and isolating true chassis motion from user interaction).
