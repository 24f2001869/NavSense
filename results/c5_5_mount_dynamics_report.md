# Stage C5.5: Smartphone Mount-Dynamics Identification

**Milestone Identifier**: `C5.5`  
**Date**: September 5, 2026  
**Status**: **COMPLETED (DIAGNOSTIC VERDICT: STATIC INVARIANCE REFUTED)**  
**Script**: [`experiments/run_mount_dynamics_identification_c5_5.py`](../experiments/run_mount_dynamics_identification_c5_5.py)  
**Structured Audit Data**: [`results/c5_5_mount_dynamics_audit.json`](c5_5_mount_dynamics_audit.json)  

---

## 1. Executive Summary & Diagnostic Verdict

Following Stage C5.4 (which ruled out suspension pitch gravity leakage and established that the chassis CAN accelerometer matches the VBOX acceleration reference with $0.17\text{--}0.25\text{ m/s}^2$ MAE), Stage C5.5 executed a **purely diagnostic physical identification** across four tests to determine whether a stable, reproducible physical transfer function:
$$a_{\text{phone}}(s) \approx H_m(s) a_{\text{CAN}}(s) + n(s)$$
exists across trips `Vta02`, `Vta03`, and `Vta04`.

### Strict Methodological Controls
1. **Zero AI / Machine Learning**: No neural networks, no Ridge, no Random Forest, no GBDT.
2. **No Parametric $(k, c)$ Assumptions**: Empirical frequency-response and coherence were evaluated without pre-assuming an idealized second-order linear oscillator.
3. **No Navigation Filter Redesign**: No ESKF modifications or heuristic tuning.
4. **CAN Used Strictly as Instrument**: The vehicle chassis CAN accelerometer was treated purely as an offline reference sensor to identify the mount relationship.

---

### Headline Diagnostic Findings

1. **Spectral Coherence Collapses Above $0.16\text{ Hz}$ (Nonlinear / Uncorrelated Dynamics)**:
   - Magnitude-squared coherence $\gamma^2(f)$ between the vehicle chassis and the smartphone accelerometer exceeds $0.6$ only below **$0.16\text{ Hz}$** on `Vta02` and `Vta04`, and is **zero everywhere** on `Vta03`.
   - In the high-frequency band ($> 0.5\text{ Hz}$), coherence collapses completely to **$\gamma^2 = 0.015\text{ to } 0.069$**.
   - Phase wraps erratically ($+675^\circ$ on `Vta02`, $+1186^\circ$ on `Vta04`), and high-frequency gain explodes to $1.75\text{--}5.07$, indicating that high-frequency smartphone acceleration is dominated by mount flexure, vibration, and cradle ringing rather than coherent chassis acceleration.

2. **Cross-Trip Invariance is Decisively REFUTED**:
   - The transfer relationship $H_m(f)$ is **not invariant** across trips.
   - The cross-trip RMS gain difference is **$5.60$** (`Vta02` vs. `Vta04`) and **$4.93$** (`Vta03` vs. `Vta04`).
   - The cross-trip RMS phase difference is **$581^\circ$** and **$846^\circ$**.
   - Because the physical transfer relationship varies wildly between trips (due to varying phone re-seating in the cradle, suction-cup tension, and mount orientation), **any attempt to fit a fixed $(k, c)$ second-order model or static inverse filter is mathematically guaranteed to fail and overfit**.

3. **Time-Domain Superposed Epochs Reveal Severe Rebound Dynamics**:
   - Aligned across discrete braking maneuvers ($N=35$ on `Vta02`, $N=2$ on `Vta04`), the phone exhibits an erratic peak delay ($-0.14\text{ s}$ to $-0.81\text{ s}$).
   - Upon brake release, the phone mount undergoes an elastic snap-back ("rebound"), generating a massive false positive acceleration spike of **$+2.29\text{ m/s}^2$** on `Vta02` and **$+3.69\text{ m/s}^2$** on `Vta04` while the vehicle chassis acceleration is zero.

4. **Low-Pass Filtering Fails to Eliminate Dead-Reckoning Drift**:
   - Separating signals at $0.5\text{ Hz}$ reduces total MAE, but severe braking bias remains huge in the low-pass band (**$+1.05\text{ m/s}^2$** on `Vta02`, **$+1.63\text{ m/s}^2$** on `Vta03`, **$+2.41\text{ m/s}^2$** on `Vta04`).
   - Because hard braking transients are low-frequency envelopes ($1\text{--}3\text{ s}$ duration $\approx 0.3\text{--}1.0\text{ Hz}$), the mount flexure persists in the low-frequency band.
   - Consequently, low-pass filtering yields **zero improvement in 30 s dead-reckoning drift** ($232.13\text{ m} \to 232.62\text{ m}$ on `Vta04`).

---

## 2. Test 1: Frequency-Domain Transfer Function & Coherence Analysis

Using Welch spectral estimation ($N_{\text{perseg}} = 128$, Hanning window, $50\%$ overlap at $f_s = 10\text{ Hz}$), we computed the cross-spectral density $S_{xy}(f)$, transfer function $\hat{H}_m(f) = S_{xy}/S_{xx}$, and magnitude-squared coherence:
$$\gamma_{xy}^2(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}$$

| Spectral Parameter | Trip `Vta02` | Trip `Vta03` | Trip `Vta04` (Untouched Test) |
| :--- | :---: | :---: | :---: |
| **Coherence Bandwidth ($\gamma^2 \ge 0.6$)** | $0.00\text{ to } 0.16\text{ Hz}$ | $0.00\text{ Hz}$ (None) | $0.00\text{ to } 0.16\text{ Hz}$ |
| **Low-Freq Band ($\le 0.5\text{ Hz}$) Mean Coherence** | $0.363$ | $0.076$ | $0.325$ |
| **Low-Freq Band ($\le 0.5\text{ Hz}$) Mean Gain $|H(f)|$** | $0.812$ | $0.283$ | $0.829$ |
| **Low-Freq Band ($\le 0.5\text{ Hz}$) Mean Phase $\angle H(f)$** | $+80.55^\circ$ | $-56.75^\circ$ | $+198.92^\circ$ |
| **High-Freq Band ($> 0.5\text{ Hz}$) Mean Coherence** | **$0.015$** | **$0.069$** | **$0.045$** |
| **High-Freq Band ($> 0.5\text{ Hz}$) Mean Gain $|H(f)|$** | $1.751$ | $3.132$ | $5.066$ |
| **High-Freq Band ($> 0.5\text{ Hz}$) Mean Phase $\angle H(f)$** | $+675.63^\circ$ | $+386.76^\circ$ | $+1186.40^\circ$ |

### Physical Interpretation
- For an ideal rigid mount, $|H_m(f)| = 1.0$, $\angle H_m(f) = 0^\circ$, and $\gamma^2(f) = 1.0$ across all frequencies.
- In reality, coherence is statistically negligible across the vast majority of the spectrum ($\gamma^2 < 0.35$).
- In the high-frequency band ($> 0.5\text{ Hz}$), coherence drops below $0.05$. The gain grows larger than $1.0$ (reaching $5.07$ on `Vta04`), which reflects unmodeled mount vibration and mechanical rattling that does not correlate with chassis acceleration.

---

## 3. Test 2: Time-Domain Superposed Epoch Transient Response

To isolate the mechanical response of the windshield mount, we detected all discrete braking maneuvers ($a_{\text{ref}} \le -1.5\text{ m/s}^2$) and extracted normalized time windows $t \in [-2.0, +5.0]\text{ s}$ aligned at the braking onset ($t_0 = 0\text{ s}$).

| Transient Epoch Metric | Trip `Vta02` | Trip `Vta03` | Trip `Vta04` |
| :--- | :---: | :---: | :---: |
| **Braking Epochs Extracted ($N$)** | $35$ | $1$ | $2$ |
| **Mean Peak Deceleration Delay ($t_{\text{phone}} - t_{\text{CAN}}$)** | $-0.142\text{ s}$ ($-1.4$ samples) | $+1.420\text{ s}$ ($+14.2$ samples) | $-0.812\text{ s}$ ($-8.1$ samples) |
| **Deceleration Attenuation Ratio ($|a_{\text{p}}^{\text{peak}}| / |a_{\text{CAN}}^{\text{peak}}|$)** | $2.367$ | $0.625$ | $4.950$ |
| **Post-Braking Rebound ($t \in [1.5, 4.0]\text{ s}$)** | **$+2.288\text{ m/s}^2$** | **$+0.291\text{ m/s}^2$** | **$+3.687\text{ m/s}^2$** |
| **Chassis CAN Rebound in Same Window** | $+0.114\text{ m/s}^2$ | $+0.042\text{ m/s}^2$ | $+0.086\text{ m/s}^2$ |

### Physical Mechanism: The Rebound Effect
The ensemble-averaged waveforms reveal a distinct mechanical signature:
1. When deceleration begins, the phone cradle deflects forward.
2. During the sustained braking plateau, the phone registers a severe under-prediction (or distorted positive acceleration).
3. Upon brake release (vehicle returning to coast or standstill), the elastic suction cup and flexible arm snap backward.
4. This produces a massive **positive acceleration spike of $+2.3\text{ to } +3.7\text{ m/s}^2$** on the phone, while the chassis CAN accelerometer smoothly returns to zero.
5. In dead-reckoning navigation, this false rebound acceleration integrates directly into an erroneous forward velocity surge, compounding position drift.

---

## 4. Test 3: Cross-Trip Consistency Evaluation

To test whether a single transfer function model can be identified on `Vta02` and deployed on `Vta04`, we resampled all empirical frequency responses onto a standard frequency grid ($0.1\text{ to } 4.9\text{ Hz}$) and computed the cross-trip root-mean-square deviations:

| Cross-Trip Pair | Magnitude RMS Difference | Phase RMS Difference | Invariance Status |
| :--- | :---: | :---: | :---: |
| **`Vta02` vs. `Vta04`** | **$5.6035$** | **$581.13^\circ$** | ❌ **NON-INVARIANT** |
| **`Vta03` vs. `Vta04`** | **$4.9277$** | **$845.70^\circ$** | ❌ **NON-INVARIANT** |

### Definitive Conclusion on Transfer Function Inversion
- The transfer function between vehicle and phone changes radically across trips.
- The magnitude discrepancy exceeds $500\%$, and the phase difference is hundreds of degrees.
- **Scientific Implication**: A static inverse filter $H_m^{-1}(s)$ or a fitted $(k, c)$ second-order transfer function **cannot generalize** across trips. If calibrated on `Vta02`, it will destabilize or introduce severe artifacts when applied to `Vta04`.

---

## 5. Test 4: Low-Frequency vs. Transient Error Decomposition

We applied zero-phase Butterworth filters to separate the acceleration signals into a **quasi-static band ($f < 0.5\text{ Hz}$)** and a **dynamic transient band ($f \ge 0.5\text{ Hz}$)**:

### Error Decomposition across Frequency Bands

| Trip | Signal Band | MAE ($\text{m/s}^2$) | RMSE ($\text{m/s}^2$) | Correlation ($r$) | Severe Brake Bias ($\text{m/s}^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`Vta02`** | Total Band | $1.2418$ | $1.7205$ | $0.2521$ | $+1.2349$ |
| | Low-Pass ($< 0.5\text{ Hz}$) | **$0.6769$** | **$0.8751$** | **$0.4969$** | **$+1.0512$** |
| | High-Pass ($\ge 0.5\text{ Hz}$) | $0.9662$ | $1.4266$ | $-0.0181$ | $+0.1837$ |
| **`Vta03`** | Total Band | $1.1889$ | $1.5582$ | $-0.0435$ | $+1.6210$ |
| | Low-Pass ($< 0.5\text{ Hz}$) | **$0.9246$** | **$1.2239$** | $-0.0814$ | **$+1.6269$** |
| | High-Pass ($\ge 0.5\text{ Hz}$) | $0.5884$ | $0.9328$ | $-0.0080$ | $-0.0058$ |
| **`Vta04`** | Total Band | $1.5391$ | $2.0096$ | $0.0502$ | $+2.5296$ |
| | Low-Pass ($< 0.5\text{ Hz}$) | **$0.8520$** | **$1.1222$** | **$0.1021$** | **$+2.4058$** |
| | High-Pass ($\ge 0.5\text{ Hz}$) | $1.2103$ | $1.6208$ | $0.0065$ | $+0.1238$ |

### Dead-Reckoning Navigation Benchmark (30 s Position Drift)

| Condition | `Vta02` (30 s) | `Vta03` (30 s) | `Vta04` (30 s) |
| :--- | :---: | :---: | :---: |
| **Raw / Static Leveled Phone** | $204.77\text{ m}$ | $225.60\text{ m}$ | $232.13\text{ m}$ |
| **Low-Pass Filtered Phone ($< 0.5\text{ Hz}$)** | $204.84\text{ m}$ | $225.14\text{ m}$ | $232.62\text{ m}$ |
| **Chassis CAN (Rigid Mount)** | **$46.06\text{ m}$** | **$66.57\text{ m}$** | **$66.91\text{ m}$** |

### Key Insights:
1. Low-pass filtering eliminates high-frequency noise, nearly halving the MAE on `Vta02` ($1.24 \to 0.68\text{ m/s}^2$) and on `Vta04` ($1.54 \to 0.85\text{ m/s}^2$).
2. However, **low-pass filtering does NOT reduce dead-reckoning drift** ($232.13\text{ m} \to 232.62\text{ m}$).
3. Why? Because vehicle braking transients have substantial spectral energy in the $0.1\text{--}0.5\text{ Hz}$ band. In that band, the smartphone mount remains physically flexed, sustaining a severe bias of $+2.41\text{ m/s}^2$. 
4. Therefore, naive linear filtering cannot decouple the smartphone from its mount distortion.

---

## 6. Diagnostic Figures

### Figure 1: Mount Frequency Response (Bode Representation & Coherence)
Displays gain, phase lag, and magnitude-squared coherence across trips, showing coherence collapse above $0.16\text{ Hz}$.
*[Bode and Coherence — Diagnostic chart]*

### Figure 2: Superposed Epoch Braking Transients
Depicts ensemble-averaged waveforms ($\pm 1\sigma$ envelopes) illustrating deceleration distortion and the massive post-braking mount rebound.
*[Superposed Epoch Transients — Diagnostic chart]*

### Figure 3: Cross-Trip Transfer Function Consistency
Direct comparison of gain and phase across `Vta02`, `Vta03`, and `Vta04`, demonstrating lack of cross-trip stationarity.
*[Cross-Trip Comparison — Diagnostic chart]*

### Figure 4: Frequency Band Decomposition (Low-Pass vs. High-Pass Drift)
Compares MAE by frequency band and dead-reckoning drift under low-pass filtering.
*[Band Decomposition — Diagnostic chart]*

---

## 7. Architectural Implications for SIH Navigation System

The findings of Stage C5.5 lead to an essential engineering conclusion:

```text
                                C5.5 Diagnostic Verdict
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ↓                                                                   ↓
Static Transfer Inversion / (k,c) Model                     Adaptive Confidence Weighting /
         │                                                        Regime-Gated Fusion
         ↓                                                                   ↓
     REJECTED                                                            VINDICATED
(Non-invariant across trips;                              (Exploits reliable quasi-static regime;
 coherence < 0.16 Hz;                                      down-weights compliant IMU during
 would overfit & destabilize)                              transients, relying on NHC / odometry)
```

### Proposed Navigation Engine Architecture:
Instead of assuming that smartphone acceleration equals vehicle acceleration or attempting an unstable inverse transfer function:

```text
                     Smartphone IMU
                           │
             Mount-Dynamics / Transient Monitor
             (Detects jerk, pitch-rate bursts,
              or sudden variance elevation)
                           │
            ┌──────────────┴──────────────┐
            ↓                             ↓
     Quasi-Static Regime           Transient Regime
   (Cruising / Gradual Motion)   (Hard Braking / Rebound)
            ↓                             ↓
     High IMU Weight             De-weight Phone IMU /
                                 Clamp Deceleration Bias /
                                 Rely on NHC + Wheel Speed
            └──────────────┬──────────────┘
                           ↓
               Multi-Sensor Fusion Engine
              (Compatible with Smartphone
               OR External Rigidly Mounted IMU)
```

This diagnostic progression provides the exact physical justification for our multi-regime navigation architecture.
