# Stage C5.2-C1: Pure Kinematic Velocity Integration Diagnostic Report

## Research Question
> **Can smartphone MEMS longitudinal acceleration physically reconstruct forward vehicle velocity over short intervals via pure kinematic integration?**
>
> $$v(t) = v(t_0) + \int_{t_0}^t a_{\text{long}}^v(\tau)\,d\tau$$

---

## Executive Summary

Stage C5.2-C1 establishes the physical baseline for velocity estimation without any statistical machine learning, without ZUPT, without dynamic bounds, and without filtering loops. 

Following the forensic finding of Stage C5.2-C0—which proved that physical forward acceleration is naturally aligned with the smartphone longitudinal axis ($a_x^p$ or leveled $a_{x,\text{level}}$ within $3^\circ\text{--}5^\circ$)—we integrated forward acceleration directly on test journey `Vta04` starting from ground-truth initial velocity $v(t_0)$ at the standardized outage inception point ($t = 25.1\text{ s}$).

### Key Empirical Findings
1. **High Precision Over Short Horizons ($\le 10\text{ s}$)**:
   - Over a **5.0 s** outage, pure acceleration integration achieves a velocity MAE of **$0.47\text{ m/s}$** ($1.7\text{ km/h}$) and a final velocity error of only **$+0.03\text{ m/s}$**. The resulting final position drift is only **$-2.25\text{ m}$**.
   - Over a **10.0 s** outage, velocity MAE remains **$0.46\text{ m/s}$** ($1.65\text{ km/h}$), with a final velocity error of **$+0.53\text{ m/s}$** and position drift of **$-4.17\text{ m}$** (a **$1.2\%$ drift rate** over $344.5\text{ m}$ distance).
   - This vastly outperforms all statistical ML regressors tested in C5.1, C5.2-A, and C5.2-B (which suffered MAEs of $2.1\text{--}2.7\text{ m/s}$ and severe instantaneous oscillations).
2. **Monotonic Drift Divergence Over Extended Horizons ($> 10\text{ s}$)**:
   - Over the full **30.0 s** standardized outage ($t = 25.1\text{--}55.1\text{ s}$), open-loop uncalibrated integration accumulates uncompensated acceleration, resulting in a velocity MAE of **$5.57\text{ m/s}$**, final velocity error of **$+17.20\text{ m/s}$**, and position error of **$+157.90\text{ m}$** ($45.8\%$ drift rate).
   - Over **60.0 s**, final velocity error reaches **$+37.05\text{ m/s}$** and position error reaches **$+944.32\text{ m}$**.
3. **Pre-Outage Calibration Failure During Maneuvers**:
   - Estimating static bias from the pre-outage window ($0\text{--}25\text{ s}$) yielded an apparent bias of $-0.328\text{ m/s}^2$ because the vehicle was decelerating before $t = 25\text{ s}$.
   - Subtracting this static value during the cruising outage actually accelerated the drift ($+27.03\text{ m/s}$ final velocity error at 30 s), demonstrating that **static bias subtraction across dynamic maneuvers is physically invalid without stationary zero-velocity calibration or dynamic pitch/tilt compensation**.

---

## 1. Experimental Setup & Protocol

- **Dataset**: IO-VNBD Journey `Vta04` (Driver E, iPhone 6 in dashboard cradle).
- **Outage Inception**: $t_0 = 25.1\text{ s}$ ($v(t_0) = 11.75\text{ m/s} = 42.3\text{ km/h}$).
- **Integration Method**: Trapezoidal numerical integration:
  $$v(t_k) = v(t_{k-1}) + \frac{1}{2}\left(a_x(t_k) + a_x(t_{k-1})\right)\Delta t$$
- **Position Mechanization**:
  $$p(t_k) = p(t_{k-1}) + \frac{1}{2}\left(v(t_k) + v(t_{k-1})\right)\Delta t$$
- **Evaluated Horizons**: $5\text{ s}$, $10\text{ s}$, $20\text{ s}$, $30\text{ s}$ (standardized C2/C3/C4 outage), $60\text{ s}$.
- **Zero-Leakage & Isolation**: Initial velocity $v(t_0)$ is taken at outage boundary (as would be provided by GNSS prior to loss). No ground-truth velocity is seen during integration.

### Signal Variant Definitions

The experiment tested the following acceleration signals. **None** of the uncalibrated variants subtract any static offset:

| Variant Name | Signal Definition | Bias Subtracted? | VBOX Used? | Deployed Feasibility |
|---|---|---|---|---|
| **Raw Phone $a_x^p$ (Uncalibrated)** | Direct phone-frame longitudinal accelerometer (`accel_x` from IO-VNBD). | **No** — zero correction applied. | **No** | ✅ Fully deployable |
| **Leveled $a_{x,\text{level}}$ (Uncalibrated)** | Phone $a_x^p$ after static gravity-leveling rotation $R_{\text{level}}$ computed from the mean Android `GRAVITY` sensor vector (IMU-only, full-trip average of `grav_x`, `grav_y`, `grav_z`). | **No** — zero correction applied beyond coordinate rotation. | **No** | ✅ Fully deployable |
| **Pre-Calibrated Bias** | $a_x - b_a$ where $b_a = \text{mean}(a_x[0{:}25\text{s}] - \frac{dv_{\text{VBOX}}}{dt}[0{:}25\text{s}])$. The bias was estimated over the **entire pre-outage window** ($t = 0\text{--}25\text{ s}$), which included active driving maneuvers (acceleration, deceleration, turns). | **Yes** — $b_a \approx -0.328\text{ m/s}^2$ subtracted. | **Yes** — $\frac{dv_{\text{VBOX}}}{dt}$ used to compute bias. | ❌ Not deployable (requires ground-truth velocity reference). Used as **offline diagnostic only**. |
| **Leveled + 0.5 Hz Filter (Pre-Calibrated)** | Leveled $a_{x,\text{level}}$ with pre-calibrated bias subtracted, then 4th-order Butterworth low-pass at 0.5 Hz (zero-phase `filtfilt`). | **Yes** — same $b_a$ as above. | **Yes** | ❌ Not deployable. |

**Critical Note on Pre-Calibrated Bias**: The pre-outage interval ($0\text{--}25\text{ s}$) was **not stationary**. The vehicle was actively driving (mean speed $\approx 10\text{ m/s}$), with braking and acceleration events. Therefore the estimated $b_a = -0.328\text{ m/s}^2$ conflates true sensor bias with the average dynamic vehicle acceleration during that interval. This is why subtracting it during the subsequent cruising outage **worsened** performance.

---

## 2. Benchmark Results Across Outage Durations

### Table 1: Pure Acceleration Integration Performance on `Vta04`

| Outage Duration | Signal Configuration | Velocity MAE (m/s) | Velocity RMSE (m/s) | Final Velocity Error | Drift Rate | Resulting Pos Error (m) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **5.0 s** | Raw Phone $a_x^p$ (Uncalibrated) | **0.47** | **0.62** | **+0.03 m/s** | $0.005\text{ m/s}^2$ | **-2.25 m** |
| | Leveled $a_{x,\text{level}}$ (Uncalibrated) | **0.47** | **0.62** | **+0.03 m/s** | $0.005\text{ m/s}^2$ | **-2.25 m** |
| | Pre-Calibrated Bias ($b_a = -0.33\text{ m/s}^2$) | 0.76 | 0.93 | +1.66 m/s | $0.333\text{ m/s}^2$ | +1.85 m |
| **10.0 s** | Raw Phone $a_x^p$ (Uncalibrated) | **0.46** | **0.58** | **+0.53 m/s** | $0.053\text{ m/s}^2$ | **-4.17 m** |
| | Leveled $a_{x,\text{level}}$ (Uncalibrated) | **0.46** | **0.58** | **+0.53 m/s** | $0.053\text{ m/s}^2$ | **-4.16 m** |
| | Pre-Calibrated Bias ($b_a = -0.33\text{ m/s}^2$) | 1.42 | 1.68 | +3.80 m/s | $0.380\text{ m/s}^2$ | +12.21 m |
| **20.0 s** | Raw Phone $a_x^p$ (Uncalibrated) | 2.45 | 3.47 | +7.52 m/s | $0.376\text{ m/s}^2$ | +40.12 m |
| | Leveled $a_{x,\text{level}}$ (Uncalibrated) | 2.45 | 3.47 | +7.52 m/s | $0.376\text{ m/s}^2$ | +40.15 m |
| | Pre-Calibrated Bias ($b_a = -0.33\text{ m/s}^2$) | 5.39 | 7.04 | +14.07 m/s | $0.704\text{ m/s}^2$ | +105.65 m |
| **30.0 s** *(Std Outage)* | Raw Phone $a_x^p$ (Uncalibrated) | **5.57** | **7.62** | **+17.20 m/s** | $0.573\text{ m/s}^2$ | **+157.90 m** |
| | Leveled $a_{x,\text{level}}$ (Uncalibrated) | **5.57** | **7.62** | **+17.21 m/s** | $0.574\text{ m/s}^2$ | **+157.97 m** |
| | Pre-Calibrated Bias ($b_a = -0.33\text{ m/s}^2$) | 10.25 | 13.13 | +27.03 m/s | $0.901\text{ m/s}^2$ | +305.34 m |
| **60.0 s** | Raw Phone $a_x^p$ (Uncalibrated) | 15.89 | 19.87 | +37.05 m/s | $0.617\text{ m/s}^2$ | +944.32 m |
| | Leveled $a_{x,\text{level}}$ (Uncalibrated) | 15.89 | 19.88 | +37.06 m/s | $0.618\text{ m/s}^2$ | +944.59 m |
| | Pre-Calibrated Bias ($b_a = -0.33\text{ m/s}^2$) | 25.60 | 31.13 | +56.71 m/s | $0.945\text{ m/s}^2$ | +1534.05 m |

---

## 3. Physical Diagnosis of the Error Growth

### 1. Why Short Horizons ($\le 10\text{ s}$) Succeed
Over short intervals, the IMU specific force accurately captures transient vehicle dynamics (throttle application and braking). High-frequency engine and chassis vibrations cancel out naturally during integration. The velocity tracking error is remarkably low ($< 0.5\text{ m/s}$), producing a positional accuracy ($-4.17\text{ m}$) that easily satisfies the SIH $<10\%$ drift threshold over $10\text{ s}$.

### 2. Why Long Horizons ($> 10\text{ s}$) Diverge
Over longer intervals, open-loop integration suffers from linear velocity drift:
$$v_{\text{err}}(t) = \int_{t_0}^t \left(b_a + g\sin\theta_{\text{pitch}}(\tau)\right) d\tau \approx \bar{a}_{\text{res}} \cdot (t - t_0)$$
In `Vta04`, the average uncompensated longitudinal acceleration during the outage is approximately $+0.57\text{ m/s}^2$. This is caused by:
- Road incline / vehicle pitch under steady cruising.
- Sensor thermal zero-offset.
A constant residual of $+0.57\text{ m/s}^2$ would produce approximately $+17.1\text{ m/s}$ velocity error and $+256.5\text{ m}$ position error over 30 s. The observed velocity error ($+17.20\text{ m/s}$) is consistent in scale with this simple constant-bias model, whereas the lower observed position error ($+157.90\text{ m}$ vs $256.5\text{ m}$) indicates that the actual residual is time-varying and/or partially changes sign during the outage interval. This is physically expected: the uncompensated specific-force residual includes road grade changes, suspension pitch dynamics, and vehicle throttle variations, all of which are time-dependent.

### 3. Lesson on Pre-Outage Bias Estimation
Attempting to calibrate a static accelerometer bias by taking the mean difference $\bar{a}_x - \frac{dv}{dt}$ over a pre-outage window without accounting for pitch/tilt dynamics fails whenever the vehicle is maneuvering. Because the vehicle experienced deceleration phases during $0\text{--}25\text{ s}$, the estimated bias was negative ($-0.33\text{ m/s}^2$). Applying this negative offset to an already positive cruising residual accelerated the error growth ($+27.03\text{ m/s}$ vs $+17.20\text{ m/s}$).

---

## 4. Next Step in Roadmap: Stage C5.2-C2 (Bias Estimation & Bounded Correction)

Now that the pure kinematic baseline is established and its failure mode is mathematically pinpointed, the next controlled step is:

> **Stage C5.2-C2: Accelerometer Bias Estimation & Dynamic Bounded Correction**
> * Formulate a disciplined bias estimator that decouples gravity pitch tilt from inertial bias.
> * Implement dynamic velocity bounds $[v_{\min}, v_{\max}]$ derived from vehicle physical limits.
> * Evaluate whether bounded kinematic correction constrains velocity drift across the standardized 30s outage.
