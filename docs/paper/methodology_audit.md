# Methodology, Dataset, and Integrity Audit

This document details the experimental protocols, dataset partitions, causality verification, reference hierarchy, and metric definitions governing all evaluations reported in the manuscript.

**Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)

---

## 1. Dataset Partition & The Driver Isolation Resolution

### 1.1 The IO-VNBD Benchmark Dataset
The empirical evaluations in this research utilize the public **Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD)** (Onyekpe, Palade, Kanarachos, & Szkolnik, *Data in Brief*, 2021, DOI: `10.1016/j.dib.2021.106885`).
* **Total Public Archive**: ~2.34 GB uncompressed (~2.07 GiB compressed LFS archive).
* **Synchronized Passenger Car Subset**: 64 whole trips, 32.85 hours, 1,182,661 raw synchronized epochs at 10.0 Hz.
* **Sensor Channels**: Tri-axial smartphone accelerometer, gyroscope, magnetometer, gravity vector, and GNSS position/speed sampled at 10.0 Hz synchronized with vehicle CAN bus diagnostic logging.
* **Vehicle Platform**: Standard passenger car driven across multiple road topologies in the United Kingdom.

### 1.2 Resolution of the "Driver Isolation" Inconsistency
Earlier research logs informally referred to "unseen driver evaluation." A forensic inspection of the primary dataset metadata in `scripts/experiments/run_expanded_tcn_forensics.py` and `scripts/data/audit_speed_distribution.py` reveals the exact driver assignment:
```python
car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
```
* **Audit Finding**: **100% of the 64 passenger vehicle trips evaluated in this research belong to a single human operator: "Driver E".**
* Other driver identifiers in the broader IO-VNBD repository correspond to municipal bus trials (`Driver A`, `Driver B`, `Driver C`, `Driver D`) or non-passenger platforms.
* **Methodological Verdict**:
  * ✅ **Trip-Level Separation**: Strictly maintained. Zero window, sequence, or temporal overlap exists between training, validation, and test partitions. Whole trips were segregated.
  * ❌ **Driver-Level Isolation**: **NOT PRESENT.** Because all passenger car trips were operated by Driver E, claims of "cross-driver generalization" are scientifically unsupported and prohibited in the manuscript.

### 1.3 Canonical Multi-Trip Partition (64 Car Trips)
Note: The total usable model training/validation/test samples ($650,661$) is lower than the raw synchronized epoch count ($1,182,661$) due to the 100-sample pre-buffer initialization window exclusion for each trip, non-driving standstill trims, and temporal alignment boundaries.
* **Training Partition (39 Trips, 478,210 usable model samples, ~73.5% of model data)**:
  * Suburban Town: `Vta01a`, `Vta01b`, `Vta02`, `Vta03`, `Vta04`, `Vta05`, `Vta06`, `Vta07`, `Vta08`, `Vta09`, `Vta10`, `Vta11`, `Vta12`, `Vta13`, `Vta14`, `Vta15`, `Vta16`, `Vta17`, `Vta18`, `Vta19`, `Vta20`.
  * Dense Urban: `Vtb01`, `Vtb02`, `Vtb03`, `Vtb04`, `Vtb05`, `Vtb06`, `Vtb07`, `Vtb08`.
  * Mountain / Winding: `Vw01`, `Vw02`, `Vw03`, `Vw04`, `Vw05`, `Vw06`, `Vw07`, `Vw08`, `Vw09`, `Vw10`, `Vw11`.
* **Validation Partition (6 Trips, ~58,912 epochs, ~8.7% of data)**:
  * Suburban: `Vta18b`, `Vta19b`.
  * Motorway: `V-Vfa01`.
  * Mountain: `Vw11b`, `Vw11c`.
  * Dense Urban: `Vtb08b`.
* **Held-Out Test Partition (19 Trips, 113,539 epochs, 238.1 km, 3.15 h, ~20.8% of data)**:
  * Motorway: `V-Vfa02` (163.4 km, 67,474 epochs, 112.5 min).
  * Suburban Town: `Vta21`, `Vta22`, `Vta23`, `Vta24`, `Vta25`, `Vta26`, `Vta27`, `Vta28`.
  * Dense Urban: `Vtb09`, `Vtb10`, `Vtb11`, `Vtb12`.
  * Winding Mountain: `Vw12`, `Vw13`, `Vw14a`, `Vw14b`, `Vw16a`.
  * Stationary Control: `Vw15` (128.1s vehicle standstill control).

---

## 2. Reference Signals vs. Physical Ground Truth

In satellite-denied navigation research, imprecise terminology regarding "ground truth" undermines credibility. We formalize the actual instrumentation hierarchy provided in the IO-VNBD benchmark dataset (Onyekpe et al., *Data in Brief*, 2021):

```
                     IO-VNBD REFERENCE INSTRUMENTATION HIERARCHY
┌──────────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Racelogic VBOX Video HD2 Data Logger (10 Hz)                         │
│ - External rooftop GPS antenna; logged synchronized position & GNSS velocity │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Tier 2: Ford Fiesta ECU CAN Bus Diagnostic Stream (10 Hz)                    │
│ - Direct vehicle wheel speed diagnostic stream from vehicle powertrain ECU;  │
│   Phase delay identified and compensated: exactly -100 ms (0.10s lag)        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ Comparison: Consumer Smartphone Single-Frequency GNSS Speed (10 Hz)          │
│ - In-cabin Samsung Galaxy GNSS; subject to multipath, urban canyon dropout,  │
│   and dilution-of-precision jitter (0.4–1.2 m/s noise floor)                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

* **Physical Ground Truth Clarification**: Physical optical track ground truth or cm-level RTK dual-frequency carrier-phase GNSS was **not available** in the public IO-VNBD dataset.
* **Engineering Reference Label**: Ford Fiesta ECU CAN bus velocity is used as the primary training and evaluation target. To correct for ECU digital filter latency, all CAN reference velocity timestamps were shifted by $-100\text{ ms}$ relative to raw IMU streams.
* **Terminology Constraint**: CAN speed is designated as the **engineering reference velocity**, never "absolute ground truth."

---

## 3. Mathematical Causality & Leakage Verification

To guarantee that the machine learning models and navigation filters are strictly causal and implementable in real time on mobile edge hardware, four automated verification gates were implemented in `tests/test_causality_and_leakage.py`:

### Gate 1: Gradient Perturbation Causality Test
For an input sequence $\mathbf{X}_{1:T} = [\mathbf{x}_1, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times C}$ and any evaluation timestep $t < T$:
$$\frac{\partial f(\mathbf{X}_{1:T})_t}{\partial \mathbf{x}_{t+k}} = \mathbf{0}, \quad \forall k > 0$$
* **Verification Result**: Perturbing future inputs $t+k$ ($k \in [1, 25]$) yielded exact floating-point zero gradients ($\nabla_{\text{future}} = 0.0000000000$) across intermediate timesteps $t \in \{30, 50, 75\}$. Zero future information enters intermediate convolutional layers.

### Gate 2: Trailing-Edge Window Alignment
The target forward velocity scalar $y_t$ is extracted strictly at the trailing edge of the observation window:
$$y_t = v_{\text{CAN}}(t_{\text{end}}), \quad t_{\text{end}} = \text{start\_idx} + W - 1$$
No centered windows ($t_{\text{target}} \ne t_{\text{mid}}$) or bilateral smoothing filters were permitted.

### Gate 3: Feature Standardization Isolation
The standard z-score normalization parameters (mean $\boldsymbol{\mu}$, standard deviation $\boldsymbol{\sigma}$) stored in `models/metadata/tcn_scaler.json` were computed strictly across the 39 training trips:
$$\boldsymbol{\mu} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{Train}} \mathbf{x}_i, \quad \boldsymbol{\sigma} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{Train}} (\mathbf{x}_i - \boldsymbol{\mu})^2}$$
Zero test-set statistics contaminated model weights or scalers.

---

## 4. Metric Definitions

All quantitative evaluations follow rigorous mathematical definitions:

1. **Pointwise Velocity Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{K} \sum_{k=1}^K |\hat{v}_k - v_k^*|$$
2. **Pointwise Velocity Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\hat{v}_k - v_k^*)^2}$$
3. **Signed Velocity Bias**:
   $$\text{Bias} = \frac{1}{K} \sum_{k=1}^K (\hat{v}_k - v_k^*)$$
4. **2D Horizontal Dead Reckoning Position Error (Drift at Horizon $T$)**:
   $$e_{2D}(T) = \|\mathbf{p}(t_0 + T) - \mathbf{p}^*(t_0 + T)\| = \sqrt{(x - x^*)^2 + (y - y^*)^2}$$
5. **Relative Normalized Position Drift (%)**:
   $$\text{Drift}_{\%} = \frac{e_{2D}(T)}{d_{\text{travel}}(T)} \times 100\% = \frac{\|\mathbf{p}(t_0 + T) - \mathbf{p}^*(t_0 + T)\|}{\int_{t_0}^{t_0 + T} v^*(t) \, dt} \times 100\%$$
   *If a vehicle remains completely stationary ($d_{\text{travel}} < 5.0\text{ m}$), relative percentage drift is mathematically undefined; in such cases, absolute drift in meters is reported to prevent division by zero.*
6. **Strict SIH Benchmark Trip Pass Rate**:
   $$\text{Pass Rate}_{<10\%} = \frac{1}{N_{\text{usable}}} \sum_{i=1}^{N_{\text{usable}}} \mathbb{I}\left(\text{Drift}_{\%, i}(60\text{s}) < 10.0\%\right)$$
   Where a trip passes if and only if its mean relative drift across all non-overlapping 60s blackout windows within that trip is strictly $< 10.0\%$.

---

## 5. Physical Mobile Testing Scope (Android 14)

Field recordings collected directly by the author used a **OnePlus Nord CE 2 Lite 5G** smartphone running Android 14.
* **Recordings Documented**:
  * Stationary bench baseline (`test_a_stationary_*.csv`): Evaluated sensor bias drift at zero velocity.
  * Rooftop pedestrian walking with GNSS (`test_b_walking_rooftop_*.csv`): Sensor callback rate and timing verification.
  * Campus walking with software-injected GNSS blackout (`walk_hostel_mess_*.csv`): Exposed the 71.66 m/s pedestrian OOD artifact.
* **Strict Methodological Boundary**:
  * **Zero Automotive Road Testing Performed on Physical Phone**: The author did not possess an instrumented motor vehicle to conduct synchronized road driving tests.
  * **Scope**: The Android recordings serve strictly as mobile OS scheduling, sensor HAL callback rate, dynamic timestamp interpolation, and out-of-distribution stress tests.
  * Vehicle navigation results reported in the paper originate exclusively from the synchronized IO-VNBD dataset.
