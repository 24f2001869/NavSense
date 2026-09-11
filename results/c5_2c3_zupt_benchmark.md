# Stage C5.2-C3: Stop Detection, Pure Kinematic ZUPT, and Event Validity Audit

## Executive Summary

Stage C5.2-C3 evaluates whether genuine vehicle stops can reset accumulated velocity integration drift through an IMU-triggered **Zero-Velocity Update (ZUPT)**. In strict accordance with the one-change-at-a-time methodology, this stage incorporates:
- **Zero AI / machine learning**
- **Zero ESKF / Kalman filtering**
- **Zero Non-Holonomic Constraints (NHC)**
- **Zero map matching / OSM**
- **Zero GNSS / GPS speed / GPS bearing fusion**

The benchmark was executed on trip **Vta02** ($1099.0\text{ s} \approx 18.3\text{ min}$, $11.05\text{ km}$ total travel), which contains multiple verified vehicle stops at traffic signals and intersections.

Following the initial benchmark, a controlled **Event Validity Audit (C3-C)** and **Oracle Decomposition (C3-B2)** were conducted to rigorously separate the benefit of genuine stationary updates from artificial error truncation caused by premature and false-positive detections.

---

### Core Decomposition: Raw ZUPT vs Oracle Valid ZUPT

| Evaluation Regime | Duration | C3-B0: No ZUPT MAE | C3-B1: Raw IMU ZUPT MAE | C3-B2: Oracle Valid ZUPT MAE | Genuine Stop Reduction (B0 $\to$ B2) | Total Apparent Reduction (B0 $\to$ B1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Complete Vta02 Mission** | $1099.0\text{ s}$ | $191.42\text{ m/s}$ | **$50.52\text{ m/s}$** | **$115.19\text{ m/s}$** | **$-39.8\%$** | **$-73.6\%$** |
| **Segment 1: Early Stops** | $40.0\text{ s}$ | $2.40\text{ m/s}$ | $1.28\text{ m/s}$ | **$1.87\text{ m/s}$** | **$-22.1\%$** | **$-46.6\%$** |
| **Segment 2: Major Stop** | $100.0\text{ s}$ | $19.10\text{ m/s}$ | $6.29\text{ m/s}$ | **$6.71\text{ m/s}$** | **$-64.9\%$** | **$-67.1\%$** |
| **Segment 3: Late Stops** | $60.0\text{ s}$ | $3.18\text{ m/s}$ | $4.89\text{ m/s}$ | **$4.77\text{ m/s}$** | $+50.0\%$* | $+53.7\%$* |
| **Highway Cruise 1 (No Stops)**| $320.0\text{ s}$ | $83.05\text{ m/s}$ | $83.05\text{ m/s}$ | **$83.05\text{ m/s}$** | **$0.0\%$** | **$0.0\%$** |
| **Highway Cruise 2 (No Stops)**| $220.0\text{ s}$ | $79.61\text{ m/s}$ | $79.61\text{ m/s}$ | **$79.61\text{ m/s}$** | **$0.0\%$** | **$0.0\%$** |

*\*Note on Segment 3: In Segment 3, the uncorrected baseline had opposite-sign errors (positive pre-stop drift canceling subsequent negative post-stop acceleration under-reporting). Both Raw ZUPT and Oracle Valid ZUPT cleanly reset velocity at the stop, unmasking the post-stop uphill grade error.*

---

## 1. Exact Detector Equations

The deployed stationarity detector uses **only** smartphone inertial measurements and timestamps. At discrete sample step $k$ ($t_k = k \cdot \Delta t$, $\Delta t = 0.1\text{ s}$), total acceleration magnitude and gyroscope angular rate norm are:
$$\|\mathbf{a}(k)\| = \sqrt{a_x^2(k) + a_y^2(k) + a_z^2(k)}, \qquad \|\boldsymbol{\omega}(k)\| = \sqrt{g_x^2(k) + g_y^2(k) + g_z^2(k)}$$

A strictly causal trailing window of length $W$ covers past and current samples: $\mathcal{W}_k = \{k - W + 1, \, k - W + 2, \, \dots, \, k\}$. The causal window statistics are:
$$\bar{a}_k = \frac{1}{W}\sum_{j \in \mathcal{W}_k} \|\mathbf{a}(j)\|, \qquad \sigma_a(k) = \sqrt{\frac{1}{W}\sum_{j \in \mathcal{W}_k} \left(\|\mathbf{a}(j)\| - \bar{a}_k\right)^2}, \qquad \mu_\omega(k) = \frac{1}{W}\sum_{j \in \mathcal{W}_k} \|\boldsymbol{\omega}(j)\|$$

The binary IMU stationarity decision is:
$$\mathcal{S}_{\text{IMU}}(k) = \begin{cases} \text{True} & \text{if } k \ge W - 1 \text{ and } \sigma_a(k) < \gamma_a \text{ and } \mu_\omega(k) < \gamma_\omega \\ \text{False} & \text{otherwise} \end{cases}$$

---

## 2. Exact Detector Thresholds

Inherited directly from Stage C5.2-C2 without modification:

| Parameter | Exact Value | Physical Justification |
|---|:---:|---|
| **Window Length ($W$)** | $20\text{ samples} = 2.0\text{ s}$ | Sufficient duration to average out engine idle harmonics while remaining responsive. |
| **Causality** | Strictly Causal Trailing | Slices $[k - W + 1 : k + 1]$. Zero access to future samples $j > k$. |
| **Acceleration Threshold ($\gamma_a$)** | $0.15\text{ m/s}^2$ | Smartphone MEMS idling vibration variance on passenger vehicle dashboard. |
| **Gyroscope Threshold ($\gamma_\omega$)** | $0.05\text{ rad/s}$ ($2.86^\circ/\text{s}$) | Distinguishes motionless platform from cornering or road pitch/roll disturbance. |

---

## 3. Offline Ground-Truth Stop Reference Definition

Constructed offline from VBOX ground truth speed $v_{\text{VBOX}}(t)$ (never provided to the deployed detector or integrator):
1. $v_{\text{VBOX}}(t) < 0.10\text{ m/s}$ ($0.36\text{ km/h}$)
2. Contiguous duration $\ge 1.0\text{ s}$ ($10\text{ consecutive samples}$)

This identified **5 reference stop episodes** totaling **$63.6\text{ s}$ ($636\text{ samples}$)** on Vta02:
- **Stop 1**: $t = 14.4\text{--}18.9\text{ s}$ ($4.6\text{ s}$, mean $v = 0.025\text{ m/s}$)
- **Stop 2**: $t = 20.2\text{--}23.1\text{ s}$ ($3.0\text{ s}$, mean $v = 0.029\text{ m/s}$)
- **Stop 3**: $t = 707.3\text{--}753.1\text{ s}$ ($45.9\text{ s}$, mean $v = 0.018\text{ m/s}$) — *major traffic light stop*
- **Stop 4**: $t = 994.7\text{--}1000.6\text{ s}$ ($6.0\text{ s}$, mean $v = 0.021\text{ m/s}$)
- **Stop 5**: $t = 1000.8\text{--}1004.8\text{ s}$ ($4.1\text{ s}$, mean $v = 0.023\text{ m/s}$)

---

## 4. Detector Validation & Corrected Latency Analysis (C3-A)

### Sample-Level Confusion Matrix ($N = 10,991\text{ samples}$)

| Confusion Matrix Element | Count | Duration | Percentage |
|---|:---:|:---:|:---:|
| **True Positives (TP)** | $631\text{ samples}$ | $63.1\text{ s}$ | $5.74\%$ |
| **False Positives (FP)** | **$154\text{ samples}$** | **$15.4\text{ s}$** | **$1.40\%$** |
| **False Negatives (FN)** | $5\text{ samples}$ | $0.5\text{ s}$ | $0.05\%$ |
| **True Negatives (TN)** | $10,201\text{ samples}$ | $1020.1\text{ s}$ | $92.81\%$ |

- **Precision**: $\mathbf{80.38\%}$
- **Recall**: $\mathbf{99.21\%}$
- **F1 Score**: $\mathbf{88.81\%}$
- **IoU**: $\mathbf{79.87\%}$
- **Overall Accuracy**: $\mathbf{98.55\%}$

### Corrected Event-Level Timing & Latency Metrics
In a causal detector, latency relative to the start of a ground-truth stop is defined as:
$$\text{lead\_time} = t_{\text{reference stop start}} - t_{\text{first detection}}$$
Where **positive lead time represents premature detection** (triggering before the vehicle has actually stopped):

| Reference Stop | VBOX Stop Interval | First IMU Trigger | Coverage | Timing Characterization |
|:---:|:---:|:---:|:---:|---|
| **Stop 1** | $14.4\text{--}18.9\text{ s}$ | $13.1\text{ s}$ | $100.0\%$ | **Premature trigger**: $+1.3\text{ s}$ early ($v = 0.66\text{ m/s}$) |
| **Stop 2** | $20.2\text{--}23.1\text{ s}$ | $13.1\text{ s}$ | $100.0\%$ | **Continuous bridge**: overlaps via Episode 1 creep |
| **Stop 3** | $707.3\text{--}753.1\text{ s}$ | $704.6\text{ s}$ | $100.0\%$ | **Premature trigger**: $+2.7\text{ s}$ early ($v = 1.50\text{ m/s}$) |
| **Stop 4** | $994.7\text{--}1000.6\text{ s}$ | $995.2\text{ s}$ | $91.7\%$ | **Delayed trigger**: $-0.5\text{ s}$ latency ($0.5\text{ s}$ detection lag) |
| **Stop 5** | $1000.8\text{--}1004.8\text{ s}$ | $995.2\text{ s}$ | $100.0\%$ | **Continuous bridge**: overlaps via Episode 10 creep |

---

## 5. Sub-Experiment C3-C: ZUPT Event Validity Audit

A complete forensic audit was conducted on every contiguous IMU ZUPT activation episode across the entire 18.3-minute trip. There were **11 distinct ZUPT episodes**:

| Ep | Start Time | End Time | Duration | VBOX Min | VBOX Max | Overlaps Stop? | Lead / Lag | Event Validity Classification |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | $13.1\text{ s}$ | $24.8\text{ s}$ | $11.8\text{ s}$ | $0.00\text{ m/s}$ | $1.08\text{ m/s}$ | Stops 1 & 2 | Lead $+1.3\text{ s}$, Lag $+1.7\text{ s}$ | **EARLY TRIGGER / LATE RELEASE** (Pre-stop braking & post-stop creep) |
| **2** | $369.8\text{ s}$ | $371.4\text{ s}$ | $1.7\text{ s}$ | $2.27\text{ m/s}$ | **$2.63\text{ m/s}$** | None | None | **FALSE ZUPT** (Queue crawling at $9.5\text{ km/h}$) |
| **3** | $374.5\text{ s}$ | $375.8\text{ s}$ | $1.4\text{ s}$ | $0.06\text{ m/s}$ | $0.55\text{ m/s}$ | None | None | **FALSE ZUPT** (Rolling deceleration dip without full stop) |
| **4** | $611.3\text{ s}$ | $612.2\text{ s}$ | $1.0\text{ s}$ | $0.10\text{ m/s}$ | $0.31\text{ m/s}$ | None | None | **FALSE ZUPT** (Rolling creep) |
| **5** | $651.9\text{ s}$ | $653.7\text{ s}$ | $1.9\text{ s}$ | $1.18\text{ m/s}$ | **$1.87\text{ m/s}$** | None | None | **FALSE ZUPT** (Queue crawl at $6.7\text{ km/h}$) |
| **6** | $656.0\text{ s}$ | $656.3\text{ s}$ | $0.4\text{ s}$ | $2.16\text{ m/s}$ | **$2.49\text{ m/s}$** | None | None | **FALSE ZUPT** (Queue crawl at $9.0\text{ km/h}$) |
| **7** | $703.7\text{ s}$ | $703.7\text{ s}$ | $0.1\text{ s}$ | $2.59\text{ m/s}$ | $2.59\text{ m/s}$ | None | None | **FALSE ZUPT** (Transient low variance sample) |
| **8** | $704.6\text{ s}$ | $753.1\text{ s}$ | $48.6\text{ s}$ | $0.00\text{ m/s}$ | **$1.50\text{ m/s}$** | Major Stop 3 | **Lead $+2.7\text{ s}$**, Lag $0.0\text{ s}$ | **EARLY TRIGGER** (Triggered at $1.50\text{ m/s}$ during gentle braking) |
| **9** | $889.6\text{ s}$ | $890.7\text{ s}$ | $1.2\text{ s}$ | $1.74\text{ m/s}$ | **$2.08\text{ m/s}$** | None | None | **FALSE ZUPT** (Queue crawl at $7.5\text{ km/h}$) |
| **10** | $995.2\text{ s}$ | $1005.1\text{ s}$ | $10.0\text{ s}$ | $0.00\text{ m/s}$ | $0.15\text{ m/s}$ | Stops 4 & 5 | Lag $-0.5\text{ s}$, Lag $+0.3\text{ s}$ | **LATE START / LATE RELEASE** (Stop queue bridge) |
| **11** | $1098.7\text{ s}$ | $1099.0\text{ s}$ | $0.4\text{ s}$ | $0.02\text{ m/s}$ | $0.05\text{ m/s}$ | None | None | **VALID (Unindexed)** (Parking at trip termination) |

---

## 6. The Premature Trigger at Major Stop 3 ($t \approx 704\text{--}753\text{ s}$)

Tracing the exact dynamics around Stop 3 exposes the precise origin of the $+150.2\text{ m/s}$ error reset:
- True VBOX stop interval: $t = 707.3\text{ s}$ to $753.1\text{ s}$ ($v < 0.10\text{ m/s}$).
- Detector trigger time: $t = 704.6\text{ s}$ (**$2.7\text{ s}$ early**).
- VBOX speed at trigger: **$v = 1.50\text{ m/s}$ ($5.4\text{ km/h}$)**.
- Over the interval $t = 704.6\text{ s}$ to $707.2\text{ s}$, the vehicle was rolling smoothly to a halt from $1.50\text{ m/s}$ down to $0.14\text{ m/s}$.
- Because the driver braked very smoothly in a straight line, $\sigma_a < 0.15\text{ m/s}^2$ and $\mu_\omega < 0.05\text{ rad/s}$ were satisfied $2.7\text{ s}$ before true zero velocity was reached.
- The algorithm set $v_{\text{est}} = 0.0\text{ m/s}$ while the car was still rolling at $1.5\text{ m/s}$. While this wiped out $+150.2\text{ m/s}$ of accumulated dead reckoning drift, it simultaneously injected a small instantaneous error of $-1.5\text{ m/s}$.

---

## 7. Decomposition: Genuine Stop Benefit vs False-Trigger Truncation

To isolate the genuine physical benefit of vehicle stops from the artifact of false/premature resets, we benchmarked three distinct pipelines across the entire 1099.0 s mission:

1. **C3-B0 (No ZUPT Baseline)**: Open-loop pure integration.
2. **C3-B1 (Raw IMU Detector ZUPT)**: Directly forces $v_{\text{est}} = 0$ whenever $\mathcal{S}_{\text{IMU}} = \text{True}$ (includes premature and false crawl triggers).
3. **C3-B2 (Oracle Valid-ZUPT-Only Diagnostic)**: Resets $v_{\text{est}} = 0$ **strictly and exclusively during genuine VBOX-confirmed stops** ($v_{\text{VBOX}} < 0.10\text{ m/s}$).

### Complete Mission Benchmark Comparison
| Metric | C3-B0: No ZUPT | C3-B1: Raw IMU ZUPT | C3-B2: Oracle Valid ZUPT |
|---|:---:|:---:|:---:|
| **Velocity MAE** | $191.42\text{ m/s}$ | **$50.52\text{ m/s}$** | **$115.19\text{ m/s}$** |
| **Velocity RMSE** | $230.15\text{ m/s}$ | $66.78\text{ m/s}$ | $145.22\text{ m/s}$ |
| **Final Position Error** | $+210,198.8\text{ m}$ ($210.2\text{ km}$) | $+42,497.5\text{ m}$ ($42.5\text{ km}$) | $+111,552.1\text{ m}$ ($111.6\text{ km}$) |
| **Drift Percentage** | $1902.4\%$ | $384.6\%$ | $1009.6\%$ |
| **Net Error Reduction** | Baseline | **$-73.6\%$** | **$-39.8\%$** |

### Physical Meaning of the Decomposition

$$\begin{aligned}
\text{Total Apparent Reduction (B0 } \to \text{ B1)} &= 191.42 \to 50.52\text{ m/s} \quad (\mathbf{-73.6\%}) \\
\text{Genuine Stop Benefit (B0 } \to \text{ B2)} &= 191.42 \to 115.19\text{ m/s} \quad (\mathbf{-39.8\%}) \\
\text{Non-Stationary Activation Truncation (B2 } \to \text{ B1)} &= 115.19 \to 50.52\text{ m/s} \quad (\mathbf{-33.8\%})
\end{aligned}$$

1. **Genuine Stops Provide a True ~40% Full-Mission Reduction**: When resets are strictly confined to true stops ($v < 0.10\text{ m/s}$), the oracle pipeline eliminates $+150\text{ m/s}$ of accumulated drift, reducing overall mission velocity MAE by **$39.8\%$** and reducing cumulative downstream position error by **$98.6\text{ km}$ (nearly $100\text{ km}$)** across the full mission ($210.2 \to 111.6\text{ km}$). In the immediate segment containing Stop 3 (Segment 2), genuine stops deliver a **$64.9\%$** MAE reduction ($19.10 \to 6.71\text{ m/s}$).
2. **The Long Inter-Stop Stretch Explains the Remaining Drift**: Between Stop 2 ($t = 23.1\text{ s}$) and Stop 3 ($t = 707.3\text{ s}$), there are **684 seconds (~11.4 minutes) of continuous driving without a single stop**. During this stretch, the Oracle pipeline (C3-B2) executes zero resets, allowing error to grow monotonically to $+150\text{ m/s}$.
3. **Attribution of the Remaining 33.8%**: The remaining **$33.8\%$ reduction** between the oracle-valid-ZUPT and raw-IMU-ZUPT pipelines is attributable to additional detector activations outside the VBOX-defined stationary intervals, including premature braking triggers and low-speed crawl false positives.

---

## 8. Anti-Leakage Audit

| Verification Item | Status | Evidence |
|---|:---:|---|
| **VBOX used in deployed detector?** | **NO** | `run_stationarity_detector` operates strictly on `(ax, ay, az, gx, gy, gz)`. |
| **VBOX used to trigger deployed ZUPT (C3-B1)?** | **NO** | C3-B1 triggers exclusively on the binary IMU mask `is_stationary`. |
| **Oracle C3-B2 clearly labeled?** | **YES** | C3-B2 is explicitly documented as an offline forensic diagnostic, not a deployable method. |
| **GPS speed / bearing used?** | **NO** | Never indexed or loaded into the integration loop. |
| **Future samples accessed?** | **NO** | Causal trailing window $[k - W + 1 : k + 1]$ strictly enforces past-only context. |
| **AI / Machine Learning used?** | **NO** | Zero ML components. |

---

## 9. Scientific Interpretation & Frozen Conclusion

> **“The IMU-triggered ZUPT pipeline substantially reduces integrated velocity drift on Vta02, but part of the apparent improvement is associated with detector activations that precede the VBOX-defined stationary interval and false activations during low-speed traffic crawling. A controlled oracle decomposition demonstrates that genuine stationary updates alone reduce full-mission velocity MAE by 39.8% (191.42 to 115.19 m/s), reducing cumulative position error by approximately 98.6 km across the mission, and reducing segment MAE by 64.9% during the primary stop. The remaining 33.8% reduction between the oracle-valid-ZUPT and raw-IMU-ZUPT pipelines is attributable to additional detector activations outside the VBOX-defined stationary intervals, including premature braking triggers and low-speed crawl false positives. These findings confirm that while genuine vehicle stops provide a powerful opportunistic drift reset, simple kinematic thresholding cannot reliably distinguish low-speed cruising from true stationarity, and provides zero correction during continuous motion.”**

---

## 10. Concrete Recommendation for Stage C5.2-C4

Stage C5.2-C3 establishes two clear physical boundaries:
1. **At genuine stops**: ZUPT resets velocity error to zero.
2. **Between stops**: Drift grows monotonically at $\sim 0.2\text{--}0.5\text{ m/s}^2$ over continuous motion horizons (e.g. the 11.4-minute highway stretch between Stops 2 and 3).

The logical next step is:
> **Stage C5.2-C4: Cross-Trip Kinematic Horizon & Drift Characterization**
> Quantify the exact error growth rate during continuous motion intervals between stops across all three benchmark trips (`Vta02`, `Vta03`, `Vta04`). This defines the precise operational bounds and error budget for **Stage C5.3**, where learned residual acceleration models ($\hat{r}_a(t)$) must constrain velocity propagation when no stationary opportunities exist.
