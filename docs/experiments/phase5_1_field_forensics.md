# Phase 5.1: Field Telemetry Forensics & The Pedestrian OOD Failure

## 1. The Anomaly: 71.66 m/s (258 km/h) Velocity Spikes
During preliminary smartphone testing around campus, a dramatic failure occurred:
* While walking from the hostel to the mess hall, the Android navigation engine suddenly reported vehicle speeds of **60 to 71.66 m/s (216 to 258 km/h)**.
* Dead reckoning position drifted by hundreds of meters within seconds on a slow human walk.

A complete forensic attribution audit was initiated to determine whether this was an Android pipeline bug, a model weights error, or an algorithmic breakdown.

---

## 2. Forensic Investigation & Root Cause Discovery

### 2.1 The Training vs Testing Distribution Mismatch
The Expanded TCN was trained exclusively on **rigidly mounted in-vehicle smartphones** from IO-VNBD. In a passenger car:
* Yaw rates during highway driving are small ($|\omega_{\text{yaw}}| < 0.05\text{ rad/s}$).
* Even in sharp mountain hairpins, yaw rate rarely exceeds $0.5\text{ rad/s}$.

In contrast, during a **pedestrian hand-held walk**:
* Arm swing, pocket motion, and body rotation generate violent, high-rate rotational transients.
* At the peak error window, measured **gyro yaw rate reached $+2.85\text{ rad/s}$** — which corresponds to **$+26.08\sigma$ (twenty-six standard deviations)** outside the vehicle training distribution!

```
Training Distribution (Vehicle):  [----μ=0----] (99.7% of data within ±0.3 rad/s)
                                                  
Pedestrian Walking Test:                          --------------------> [ω = +26.08σ !!]
                                                                        (Severe OOD)
```

### 2.2 Feature Attribution & Ablation Analysis
We conducted controlled feature ablation at the exact peak error epoch ($t = 203.8\text{ s}$):

| Input Condition | Predicted Velocity | Relative Reduction |
|:---|:---:|:---:|
| **Baseline (Unmodified Inputs)** | **71.66 m/s (258 km/h)** | Baseline |
| **Ablate `gyro_yaw` (Set to 0)** | **34.94 m/s (126 km/h)** | **-51.2%** |
| **Ablate All Gyroscopes (`yaw, pitch, roll = 0`)** | **27.71 m/s (100 km/h)** | **-61.3%** |
| **Input Clipping to $\pm 3\sigma$ Training Bounds** | **32.11 m/s (116 km/h)** | **-55.2%** |
| **Clip Inputs + Zero Dynamic Curvature $\kappa$** | **18.42 m/s (66 km/h)** | **-74.3%** |

The mathematical root cause was proven: **The neural network learned that high yaw rates coupled with horizontal acceleration indicate high-speed vehicle cornering ($\kappa = \|\boldsymbol{\omega}\| \cdot a_{\text{horiz}} \propto v^2/R$). When given pedestrian arm-swing rotations, the feedforward layers multiplied these out-of-distribution values into massive speed predictions.**

---

## 3. Engineering Fixes & Safeguards Implemented

To protect the production engine, four defensive layers were implemented:
1. **Input $\pm 3.5\sigma$ Gating**: Every input feature is clamped to $[-3.5, +3.5]$ standard deviations based on `tcn_scaler.json`.
2. **Kinematic Plausibility Gating**: Forward acceleration is bounded by physical vehicle limits:
   $$\Delta v \le a_{\max} \cdot \Delta t \quad (a_{\max} = 6.0\text{ m/s}^2)$$
3. **Out-of-Distribution (OOD) Detector**: If $\|\boldsymbol{\omega}\| > 1.5\text{ rad/s}$ while linear acceleration variance is low, the system flags pedestrian motion and inhibits speed updates.

---

## 4. Methodological Conclusion

> [!CAUTION]
> **Strict Research Rule**:  
> Pedestrian walking recordings must **NEVER** be categorized as vehicle navigation validation.  
> They are strictly **sensor acquisition, timing integrity, and out-of-distribution stress tests**. True vehicle validation requires synchronized automotive testing.
