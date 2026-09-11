# Stage C10.2 Real-World Bus Field Test Telemetry Audit
**Platform**: Physical Android Smartphone (OnePlus Nord CE 2 / CPH2381 / Device `68afd407`)  
**Route**: Transit Bus Route between Hostel and Department, Hyderabad, India  
**Total Telemetry**: 26,062 Epochs (~54.13 Minutes of continuous recording; nominal 10-Hz pipeline with observed effective rate ~7.5–8.5 Hz)  
**Date**: September 9, 2026  

---

## 1. Executive Summary & Scientific Scope

This report documents the empirical audit of 54.13 minutes of real-world multi-kilometer transit bus telemetry collected on physical target smartphone hardware (`CPH2381`).

The dataset consists of two complete passenger transit journeys:
1. **Trip 1 (Hostel $\rightarrow$ Department)**: 15,730 epochs, 1,860.0 s (31.00 min), Effective rate: 8.46 Hz ($\Delta t = 118.3\,\text{ms}$)
2. **Trip 2 (Department $\rightarrow$ Hostel)**: 10,332 epochs, 1,387.6 s (23.13 min), Effective rate: 7.45 Hz ($\Delta t = 134.3\,\text{ms}$)

During both journeys, the user initiated multiple simulated GNSS outages (14 distinct outage episodes spanning over 20 minutes of GNSS-denied navigation) while the vehicle was in transit. Because Android background location was simultaneously recorded, continuous, microsecond-synchronized **GNSS Ground Truth** was captured throughout every second of dead reckoning.

```
+-----------------------------------------------------------------------------------------+
| REAL-WORLD FIELD TEST MILESTONE SUMMARY                                                |
+-----------------------------------------------------------------------------------------+
| Target Device        | OnePlus Nord CE 2 (CPH2381), MediaTek Dimensity 900             |
| Operating System     | Android 13 (API Level 33)                                       |
| Telemetry Files      | idr_telemetry_20260909_103010.csv (3.09 MB, 15,730 rows)        |
|                      | idr_telemetry_20260909_125631.csv (2.03 MB, 10,332 rows)        |
| Total Route Duration | 54.13 Minutes (3,247.6 Seconds)                                 |
| Sensor Logging Rate  | Nominal 10 Hz; Effective observed rate: ~7.5–8.5 Hz              |
| Max Vehicle Speed    | 57.2 km/h (Trip 1), 55.2 km/h (Trip 2)                          |
| Causal ML Speed      | Moving scale recovered (+2.0 km/h in Trip 2, +5.4 km/h Trip 1)   |
| Stationary Failure   | ~21–22 km/h predicted during complete zero-speed bus stops      |
| Long-Outage Drift    | 85% to 1262% drift (Far above the SIH <10% benchmark)          |
| Compass Gating Rate  | 99.99% Rejection (Protected filter; zero heading aid inside)    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Quantitative Dataset Breakdown

| Metric | Trip 1 (Hostel $\rightarrow$ Dept) | Trip 2 (Dept $\rightarrow$ Hostel) | Combined Dataset |
| :--- | :--- | :--- | :--- |
| **Total Rows / Epochs** | 15,730 | 10,332 | **26,062** |
| **Duration (min / sec)** | 31.00 min / 1,860.0 s | 23.13 min / 1,387.6 s | **54.13 min / 3,247.6 s** |
| **Effective Sampling Rate** | 8.46 Hz ($\Delta t = 118.3\,\text{ms}$) | 7.45 Hz ($\Delta t = 134.3\,\text{ms}$) | **8.03 Hz ($\Delta t = 124.5\,\text{ms}$)** |
| **GNSS Locked Epochs** | 10,016 (63.7%) | 7,320 (70.8%) | **17,336 (66.5%)** |
| **Dead Reckoning Epochs** | 5,705 (36.3%) | 3,008 (29.1%) | **8,713 (33.4%)** |
| **Reacquiring Epochs** | 9 (0.1%) | 4 (0.0%) | **13 (0.05%)** |
| **Max GNSS Speed** | 57.17 km/h | 55.22 km/h | **57.17 km/h** |
| **Mean Moving GNSS Speed** | 20.54 km/h | 22.66 km/h | **21.33 km/h** |
| **Stationary Time ($v < 0.2\,\text{m/s}$)** | 20.1% (3,156 epochs) | 30.4% (3,142 epochs) | **24.2% (6,298 epochs)** |
| **Geodetic Bounds** | $[17.4427^\circ\text{N}, 78.3150^\circ\text{E}]$ to $[17.4661^\circ\text{N}, 78.3386^\circ\text{E}]$ | $[17.4427^\circ\text{N}, 78.3150^\circ\text{E}]$ to $[17.4640^\circ\text{N}, 78.3374^\circ\text{E}]$ | Hyderabad Transit Corridor |

> [!IMPORTANT]
> **Sampling Timing Reality:** The Android system does not deliver fixed 100 ms callbacks. Effective rates were 8.46 Hz and 7.45 Hz. The application code correctly ingests dynamic, timestamp-derived $\Delta t$ rather than assuming a fixed 0.1 s.

---

## 3. Causal ML Forward Speed Estimator Analysis

### 3.1 Moving Transit Performance ($v_{\text{GNSS}} > 2.0\,\text{m/s}$)
The causal Random Forest model transferred sufficiently to recover the approximate speed scale during moving bus operation, but exhibited substantial trip-dependent bias:

- **Trip 2 (Cruising Concordance)**:
  - Ground Truth GNSS Mean Speed: **25.17 km/h**
  - Causal ML Model Mean Speed: **27.18 km/h**
  - **Net Bias: +2.01 km/h (+8.0% error)**
- **Trip 1 (Over-prediction Bias)**:
  - Ground Truth GNSS Mean Speed: **25.12 km/h**
  - Causal ML Model Mean Speed: **30.53 km/h**
  - **Net Bias: +5.41 km/h (+21.5% error)**

### 3.2 The Stationary Vibration Failure Mode
When the transit bus came to a complete halt at bus stations or traffic signals ($v_{\text{GNSS}} < 0.2\,\text{m/s}$), the ML speed predictor exhibited a severe **stationary failure**:
- **Trip 1 Stationary ML Speed**: Mean = **22.22 km/h** (Median: 20.45 km/h, 90th pct: 49.18 km/h)
- **Trip 2 Stationary ML Speed**: Mean = **21.02 km/h** (Median: 13.75 km/h, 90th pct: 45.85 km/h)

#### Root Cause Mechanism:
1. **Engine Idle Rumble**: The bus diesel engine produces continuous structural vibrations ($10\,\text{Hz} - 25\,\text{Hz}$) through the chassis.
2. **Elevated Sensor Noise**: During zero-speed stops, phone accelerometer variance was $\sigma_a \approx 1.22 - 2.35\,\text{m/s}^2$ and passenger movement/hand tremors produced gyroscope rotation of $28.7^\circ/\text{s} - 32.9^\circ/\text{s}$.
3. **ZUPT Suppression**: Because the baseline stationary detection gate requires quiet sensors ($\sigma_a < \text{threshold}$, $\|\boldsymbol{\omega}\| < \text{threshold}$), the bus engine rumble prevented `isStationary` from triggering.
4. **False Displacement Accumulation**: The Random Forest regressor mapped the high vibration energy to cruising speeds (~21 km/h). A 60-second stop at a traffic light under outage conditions directly injects $20\,\text{km/h} \times 60\,\text{s} \approx 333\,\text{meters}$ of purely artificial forward displacement.

---

## 4. Dead Reckoning Outage Drift Breakdown

Across both journeys, 14 simulated GNSS outages were evaluated. The table below details every multi-second blackout:

| Trip | Episode | Outage Duration | GT Distance Traveled | Bus Avg Speed | Start Horiz Error | End Horiz Error | Net Outage Drift | Drift Rate | Drift % of Dist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip 1** | Ep 2 | **94.3 s** | **682.8 m** | 26.1 km/h | 53.9 m | 581.0 m | **+527.1 m** | **6.16 m/s** | **85.1%** |
| **Trip 1** | Ep 3 | **132.4 s** | **510.6 m** | 13.9 km/h | 60.4 m | 2538.3 m | **+2477.9 m** | **19.17 m/s** | **497.1%** |
| **Trip 1** | Ep 4 | **128.6 s** | **657.9 m** | 18.4 km/h | 71.7 m | 1187.6 m | **+1115.9 m** | **9.24 m/s** | **180.5%** |
| **Trip 1** | Ep 5 | **125.1 s** | **876.1 m** | 25.2 km/h | 40.7 m | 1674.7 m | **+1634.0 m** | **13.39 m/s** | **191.2%** |
| **Trip 1** | Ep 7 | **99.5 s** | **112.4 m** | 4.1 km/h | 52.5 m | 426.0 m | **+373.5 m** | **4.28 m/s** | **379.2%** |
| **Trip 2** | Ep 2 | **161.7 s** | **610.7 m** | 13.6 km/h | 20.9 m | 1025.4 m | **+1004.5 m** | **6.34 m/s** | **167.9%** |
| **Trip 2** | Ep 3 | **57.6 s** | **23.1 m** | 1.4 km/h | 2.3 m | 115.3 m | **+113.0 m** | **2.00 m/s** | **498.8%** |
| **Trip 2** | Ep 4 | **80.1 s** | **320.2 m** | 14.4 km/h | 4.7 m | 4040.7 m | **+4036.0 m** | **50.45 m/s** | **1262.1%** |

### Scientific Verdict on Dead Reckoning Performance:
> [!WARNING]
> **Numerical Execution $\neq$ Navigation Stability.**  
> The engine remained numerically executable throughout the extended outages, but **navigation accuracy degraded severely**, particularly when stationary intervals and vertical drift were present. A drift rate of 6 to 50 m/s is far outside acceptable navigation limits, and the SIH <10% drift requirement was not satisfied on any of these multi-minute real-world outages.

---

## 5. Constraint & Environmental Diagnostics

### 5.1 Magnetometer Gating (Protection Mechanism, Not Heading Aid)
- Inside the bus chassis, the magnetic field fluctuated between **$4.5\,\mu\text{T}$ and $135.7\,\mu\text{T}$** due to structural steel girders, passenger seats, and high-current electrical systems.
- The temporal gradient and innovation gates rejected **100.0%** of magnetic updates in Trip 1 and **99.97%** in Trip 2.
- **Physical Interpretation**: The gate successfully acted as a **protection mechanism**, preventing corrupted magnetic measurements from instantly distorting the filter attitude. However, this also means that the magnetometer provided **zero usable heading information** inside the bus; heading observability was entirely dependent on open-loop gyroscope integration.

### 5.2 Vertical Channel Divergence ($pos\_u$)
- In Trip 2, the vertical position ($pos\_u$) diverged to roughly **$-15\,\text{km}$**, and in Trip 1 to **$+5.2\,\text{km}$**.
- **Coupling Mechanism**: Small residual vertical accelerometer biases integrated quadratically ($z \approx \frac{1}{2} b_z t^2$) during 2- to 3-minute outages. Because Virtual NHC (VNHC) was active only 9% to 13% of the time, the unconstrained vertical velocity and position drifted unchecked. Through attitude cross-coupling, large vertical velocity errors leak directly into horizontal navigation states.

---

## 6. Next Experimental Roadmap (Diagnostics First, No Premature Math Changes)

To systematically address the root failure modes revealed by this dataset, the navigation engine remains **strictly frozen** while diagnostic investigations are conducted:

1. **Stage C10.3 — Stationary Vibration Diagnostic**:
   - **Research Question**: *Can we detect "bus stopped but engine vibrating" using only smartphone sensors?*
   - Classify the 26,062 epochs into stationary vs moving regimes using GNSS labels and evaluate feature separability (frequency-band energy, jerk, low- vs high-frequency acceleration variance).
2. **Stage C10.4 — Vertical Channel Diagnostic**:
   - Determine the exact kinematic mechanism of the vertical channel divergence and evaluate whether strict vertical clamping or flat-plane projection stabilizes $pos\_u$ without degrading horizontal trajectory.
3. **Stage C10.5 — Controlled Vehicle-Mounted Benchmark**:
   - Evaluate dead reckoning under rigid mounting with controlled 10s, 30s, and 60s GNSS outages.

