# Stage C5.2-C2: Accelerometer Bias Estimation & Bounded Correction

## Research Question
> **Can physically defensible accelerometer-bias correction extend the useful kinematic integration horizon beyond the ~10 s regime demonstrated in C5.2-C1?**

---

## Executive Summary

Stage C5.2-C2 tested four bias-correction methods on the Vta04 standardized outage ($t_0 = 25.1\text{ s}$, $v_0 = 11.75\text{ m/s}$):

| Method | Description | Bias Applied |
|---|---|:---:|
| **C2-A** | No correction (frozen C5.2-C1 baseline) | $0.000\text{ m/s}^2$ |
| **C2-B** | Stationary bias from verified stopped intervals | $0.000\text{ m/s}^2$ |
| **C2-C** | Quasi-static bias update during low-dynamics outage windows | $0.000 \to$ variable |
| **C2-D** | Bounded bias correction ($|b_a| \le 0.5\text{ m/s}^2$) | $0.000\text{ m/s}^2$ |

#### Central Result

**All four methods produced identical results over the 30 s standardized outage**, because:

1. **No intervals satisfying the predefined IMU stationarity criteria were detected in Vta04** during the pre-outage window ($t \in [0, 25.1]\text{ s}$) or the entire trip. The strict criteria ($\text{std}(|\mathbf{a}|) < 0.15\text{ m/s}^2$, $\text{mean}(|\boldsymbol{\omega}|) < 0.05\text{ rad/s}$) found **zero** qualifying samples. This does not prove the vehicle never stopped; rather, it establishes that under these specific thresholds, smartphone MEMS vibration and angular noise precluded IMU-only stationary classification.
2. **Without detected stationarity, no pre-outage IMU sensor bias could be estimated**. Methods C2-B and C2-D correctly defaulted to $b_a = 0.000\text{ m/s}^2$ rather than estimating from non-stationary, motion-contaminated data.
3. **C2-C's quasi-static updates only triggered at $t \ge 75.7\text{ s}$** (well outside the standardized 30 s outage), producing a modest improvement only over the 60 s horizon ($15.89 \to 14.45\text{ m/s}$ MAE, a $9.1\%$ reduction). Forensic inspection reveals that this update absorbed real vehicle longitudinal acceleration into the bias estimate during a straight-line throttle event, violating the Anti-C3 Rule.
4. **Over the SIH-relevant 5–30 s horizon, C2 bias handling has essentially zero effect**: all four error curves sit on top of each other, reaching $\sim 158\text{ m}$ position drift by 30 s.

**C5.2-C2 has failed as a solution to drift, but succeeded as a diagnostic.**

---

## 1. Stationarity Detection Diagnostic

### IMU-Only Detection Criteria

| Parameter | Strict Stationarity | Low-Dynamics (Quasi-Static) |
|---|:---:|:---:|
| Window length | 2.0 s (20 samples @ 10 Hz) | 2.0 s (20 samples @ 10 Hz) |
| $\text{std}(|\mathbf{a}|)$ threshold | $< 0.15\text{ m/s}^2$ | $< 0.50\text{ m/s}^2$ |
| $\text{mean}(|\boldsymbol{\omega}|)$ threshold | $< 0.05\text{ rad/s}$ ($2.86^\circ/\text{s}$) | $< 0.15\text{ rad/s}$ ($8.59^\circ/\text{s}$) |

### Detection Results

| Interval | Strict Stationary Samples | Low-Dynamics Samples |
|---|:---:|:---:|
| Pre-outage ($0\text{--}25.1\text{ s}$) | **0 / 251 samples** ($0.0\%$) | 27 / 251 samples ($10.8\%$) |
| Full trip ($0\text{--}178.9\text{ s}$) | **0 / 1789 samples** ($0.0\%$) | 126 / 1789 samples ($7.0\%$) |

**Offline validation (VBOX reference, not available to estimator)**:
The 27 low-dynamics pre-outage samples had true VBOX speed:
$$\text{mean}(v) = 11.89\text{ m/s} \quad (42.8\text{ km/h}), \quad \max(v) = 11.97\text{ m/s}$$
The vehicle was cruising at highway speeds. Had a detector labeled this as "stationary", the estimated "bias" would have absorbed cruise pitch and road grade.

---

## 2. Bias Estimation Equations & Audit

### C2-A: No Correction (Baseline)
$$a_{\text{corrected}}(t) = a_x^{\text{level}}(t)$$

### C2-B: Stationary Bias
$$b_a = \frac{1}{N_{\text{stat}}} \sum_{i \in \mathcal{S}_{\text{stat}}} a_x^{\text{level}}(t_i), \qquad a_{\text{corrected}}(t) = a_x^{\text{level}}(t) - b_a$$
where $\mathcal{S}_{\text{stat}}$ is the set of IMU-detected stationary samples before $t_0$.
**On Vta04**: $N_{\text{stat}} = 0 \Rightarrow b_a = 0.000\text{ m/s}^2$.

### C2-C: Quasi-Static Bias Update
$$b(t_i) = \begin{cases} b_{\text{stat}} & \text{if } i < k_{\text{outage}} \\ \frac{1}{W} \sum_{k=i-W}^{i-1} a_x^{\text{level}}(t_k) & \text{if trailing window } [i-W, i-1] \text{ meets low-dynamics} \\ b(t_{i-1}) & \text{otherwise (sample-and-hold)} \end{cases}$$
$$a_{\text{corrected}}(t) = a_x^{\text{level}}(t) - b(t)$$

- **Window**: $W = 20$ samples (2.0 s at 10 Hz), causal trailing window excluding the current sample $i$.
- **Trigger**: $\text{std}(|\mathbf{a}|) < 0.5\text{ m/s}^2$ AND $\text{mean}(|\boldsymbol{\omega}|) < 0.15\text{ rad/s}$.
- **Bounding**: Unbounded (no clamping applied).
- **Behavior on Vta04**: Zero updates during $t \in [25.1, 55.1]\text{ s}$ (30 s outage). 61 updates occurred between $75.7\text{ s}$ and $77.6\text{ s}$, where $b(t)$ climbed to $+2.87\text{ m/s}^2$, absorbing true vehicle acceleration during a straight-line throttle event ($v: 2.74 \to 11.27\text{ m/s}$).

### C2-D: Bounded Bias Correction
$$b_{\text{raw}} = b_a \text{ (from C2-B)}, \qquad b_{\text{clamped}} = \text{clip}(b_{\text{raw}}, -b_{\max}, +b_{\max})$$
$$a_{\text{corrected}}(t) = a_x^{\text{level}}(t) - b_{\text{clamped}}$$
**Bound**: $b_{\max} = 0.5\text{ m/s}^2$ based on MEMS zero-g offset limits (MPU-6500 spec: $\pm 60\text{ mg} \approx \pm 0.59\text{ m/s}^2$).
**On Vta04**: $b_{\text{raw}} = 0 \Rightarrow b_{\text{clamped}} = 0.000\text{ m/s}^2$.

---

## 3. Benchmark Results

### Table 1: Velocity MAE (m/s) by Duration
| Method | 5 s | 10 s | 20 s | 30 s | 60 s |
|---|---:|---:|---:|---:|---:|
| **C2-A: No Correction** | **0.47** | **0.46** | 2.45 | 5.57 | 15.89 |
| **C2-B: Stationary Bias** | **0.47** | **0.46** | 2.45 | 5.57 | 15.89 |
| **C2-C: Quasi-Static Update** | **0.47** | **0.46** | 2.45 | 5.57 | **14.45** |
| **C2-D: Bounded Correction** | **0.47** | **0.46** | 2.45 | 5.57 | 15.89 |

### Table 2: Final Velocity Error (m/s) and Position Drift (m)
| Method | 5 s Vel / Pos | 10 s Vel / Pos | 20 s Vel / Pos | 30 s Vel / Pos | 60 s Vel / Pos |
|---|---:|---:|---:|---:|---:|
| **C2-A** | +0.03 / -2.2 | +0.53 / -4.2 | +7.52 / +40.2 | +17.21 / +158.0 | +37.06 / +944.6 |
| **C2-B** | +0.03 / -2.2 | +0.53 / -4.2 | +7.52 / +40.2 | +17.21 / +158.0 | +37.06 / +944.6 |
| **C2-C** | +0.03 / -2.2 | +0.53 / -4.2 | +7.52 / +40.2 | +17.21 / +158.0 | **+16.71 / +859.1** |
| **C2-D** | +0.03 / -2.2 | +0.53 / -4.2 | +7.52 / +40.2 | +17.21 / +158.0 | +37.06 / +944.6 |

---

## 4. Offline Specific-Force Residual Diagnostic

The true longitudinal specific-force residual during the 30 s outage is computed offline via VBOX:
$$r(t) = a_x^{\text{level}}(t) - \frac{dv_{\text{VBOX}}}{dt}(t)$$

Conceptually, this residual can be decomposed as:
$$r(t) = b + \epsilon(t)$$
where $b$ is a constant offset and $\epsilon(t)$ represents time-varying error components.

| Metric | Value |
|---|---:|
| Mean ($\mu_r$) | $+0.569\text{ m/s}^2$ |
| Standard Deviation ($\sigma_r$) | $1.708\text{ m/s}^2$ |
| Minimum | $-4.716\text{ m/s}^2$ |
| Maximum | $+5.008\text{ m/s}^2$ |
| Dynamic Range | $9.724\text{ m/s}^2$ |

### Physical Interpretation & Error Mechanism Candidates

The residual is **not adequately explained by a constant bias alone; substantial time-varying specific-force error remains after accounting for its mean.**

Inspection of the residual time-series reveals clear structured excursions:
- **$25\text{--}55\text{ s}$**: Persistent positive offset regions interspersed with high-frequency oscillation.
- **$65\text{--}68\text{ s}$**: Large structured excursion ($\pm 4\text{ m/s}^2$).
- **$74\text{--}80\text{ s}$**: Several large non-random deviations corresponding to vehicle maneuvers.

These structured excursions indicate that the error is far richer than simple sensor bias plus white noise. The primary **candidate mechanisms** (which must be distinguished from proven decomposition) include:
1. **Attitude / Road Grade Coupling**: Imperfect phone leveling and time-varying road incline coupling gravity into $a_x^{\text{level}}$.
2. **Suspension Dynamics**: Chassis pitch during throttle and brake transients.
3. **Vehicle Longitudinal Dynamics**: Centripetal or transient acceleration leakage across body axes.
4. **MEMS Sensor Bias**: Run-to-run zero-g offset drift.
5. **Sensor Noise and Vibration**: Engine harmonics and pavement texture.

Because $\sigma_r = 1.708\text{ m/s}^2$ exceeds $\mu_r = 0.569\text{ m/s}^2$ by a factor of 3, static bias correction cannot arrest integration drift over 30 s.

---

## 5. Leakage Audit

| Method | VBOX in Estimator? | IMU-Only Stationarity? | Future Samples? | Causal? |
|---|:---:|:---:|:---:|:---:|
| C2-A | **No** | N/A | **No** | **Yes** |
| C2-B | **No** | **Yes** | **No** | **Yes** |
| C2-C | **No** | **Yes** | **No** | **Yes** |
| C2-D | **No** | **Yes** | **No** | **Yes** |

---

## 6. Frozen Scientific Conclusion

> **C5.2-C2 shows that the tested IMU-only bias-correction strategies do not materially reduce velocity integration error over the 30-s outage horizon. No intervals satisfying the predefined stationarity criteria were detected in Vta04, preventing stationary-bias estimation for this trip. A causal quasi-static update produced only a modest improvement at 60 s, reducing MAE from 15.89 to 14.45 m/s. The offline longitudinal specific-force residual contained both a non-zero mean component and substantial time-varying variation, indicating that a constant accelerometer bias alone is insufficient to explain the observed integration drift. These results motivate testing explicit zero-velocity updates during genuine vehicle stops and subsequently learning time-varying residual errors rather than relying on static bias correction alone.**
