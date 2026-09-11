<div align="center">

# NavSense: Inertial Dead Reckoning for GNSS-Denied Ground Vehicles

### Continuous Dead Reckoning on Consumer Smartphones via Causal Temporal Convolutional Networks and 15-State Error-State Kalman Filtering
**Smart India Hackathon (SIH) — Problem Statement 26168**

[![GitHub Repository](https://img.shields.io/badge/GitHub-24f2001869%2FNavSense-181717.svg?logo=github)](https://github.com/24f2001869/NavSense)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Target: SIH26168](https://img.shields.io/badge/SIH-Problem%2026168-purple.svg)](docs/problem_statement.md)
[![Platform: Android](https://img.shields.io/badge/Platform-Android%2014-green.svg?logo=android)](android/README.md)
[![Runtime: ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(162%20KB)-orange.svg?logo=onnx)](https://onnxruntime.ai/)
[![Benchmark Status](https://img.shields.io/badge/Status-Research%20Prototype-amber.svg)](docs/current_status.md)

[**Quickstart (5 Min)**](docs/START_HERE.md) • [**System Architecture**](docs/architecture.md) • [**Methodology**](docs/methodology.md) • [**Evidence Map**](docs/evidence_map.md) • [**Failed Experiments Catalog**](docs/failed_experiments_catalog.md) • [**Reproducibility Guide**](docs/reproducibility.md)

</div>

---

## 1. Problem Formulation

When a ground vehicle enters a tunnel, subterranean road corridor, or dense urban canyon, satellite positioning (GNSS/GPS) becomes unavailable.

In automotive factory-fitted navigation, dead reckoning is sustained via **wheel speed encoders (CAN bus wheel ticks)** and calibrated chassis inertial measurement units (IMUs). On a standalone consumer smartphone, **wheel odometry does not exist**. Positioning must rely entirely on low-cost, noisy MEMS inertial sensors (accelerometer, gyroscope, magnetometer).

### The Double-Integration Divergence

Open-loop double integration of uncompensated accelerometer bias $b_a$ leads to quadratic velocity error and cubic position divergence:

$$\Delta d(t) = \frac{1}{2} b_a t^2 \quad \text{(velocity integration)} \implies \Delta p(t) \propto \mathcal{O}(t^3)$$

Even with a modest residual accelerometer bias of $b_a \approx 0.05\text{ m/s}^2$:
* At $t = 10\text{ s}$: $\Delta d \approx 2.5\text{ m}$ (manageable lane-level error).
* At $t = 30\text{ s}$: $\Delta d \approx 22.5\text{ m}$ (lane departure).
* At $t = 60\text{ s}$: $\Delta d \approx 90\text{--}300+\text{ m}$ (complete navigation breakdown).

Without an independent velocity constraint or physical motion gating, mathematical open-loop integration invariably fails.

---

## 2. SIH Problem Statement 26168 Target Benchmark

Smart India Hackathon Problem Statement **26168** defines strict empirical criteria for standalone smartphone dead reckoning:
1. **60-Second Continuous Blackout Survival:** Maintain vehicle position estimates for 60 seconds without GNSS fixes.
2. **Strict Distance-Drift Threshold:** Positional error must remain **$< 10\%$ of total distance traveled** over the blackout window.
3. **Smartphone Hardware Only:** No OBD-II adapters, wheel tick dongles, or infrastructure beacons.

---

## 3. System Architecture

NavSense implements an end-to-end 5-layer pipeline combining deep sequence models with classical estimation theory:

<div align="center">

![NavSense System Architecture](assets/architecture/system_architecture.svg)

</div>

### Pipeline Overview
1. **Sensor Ingestion Layer:** Ingests high-frequency tri-axial linear acceleration, angular velocity, and gravity vectors at 100 Hz; applies causal anti-aliasing filtering and resamples to a synchronized 10.0 Hz clock.
2. **Coordinate Alignment Layer:** Transforms raw phone-body measurements into a leveled navigation frame (horizontal plane vs. gravity vector) with dynamic forward axis alignment via Principal Component Analysis (PCA).
3. **Triple-Branch Velocity Engine:**
   - **Stateful Kinematics Branch:** Integrates longitudinal acceleration ($v_t = v_{t-1} + a_{\text{lon}}\Delta t$) initialized from pre-outage GNSS speed.
   - **Causal Dilated TCN Branch:** 65k-parameter Temporal Convolutional Network with a 6.1s receptive field predicting forward speed directly from motion dynamics.
   - **Causal Zero-Velocity Detector (ZVD):** Energy and variance gating detecting complete vehicle halts and enforcing $v = 0$.
4. **Adaptive Dynamic Regime Blender:** Causally modulates kinematic momentum vs. neural predictions based on real-time maneuver intensity, preventing urban stop-and-go runaway while preserving highway cruise momentum.
5. **15-State Error-State Kalman Filter (ESKF):** Propagates 3D kinematics; fuses virtual velocity pseudo-measurements and zero-velocity updates (ZUPT) with Non-Holonomic Constraints (NHC) during satellite blackouts.
6. **Mobile Edge Deployment:** Fully operational native Android app running local ONNX Runtime inference ($<8\text{ ms}$ latency), Java ESKF, and offline OSMDroid vector map rendering.

---

## 4. Comprehensive Multi-Horizon Benchmark

Evaluated across rolling blackouts on the **19 held-out test trips** of the IO-VNBD dataset (zero data leakage; completely disjoint whole trips):

| Outage Horizon | Evaluated Architecture | Usable Trips | Mean Drift (m) | Relative Drift (%) | SIH Passes (<10%) | Pass Rate | Primary Metric Proof |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **10 Seconds** | **Pure Kinematics** | 18 | **23.66 m** | **83.54%** | **6 / 18** | **33.3%** | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L3) |
| | Fixed Damped Momentum | 18 | 23.49 m | 111.37% | 6 / 18 | 33.3% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L4) |
| | **NavSense Adaptive Fusion**| 18 | 25.06 m | 111.53% | 5 / 18 | 27.8% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L5) |
| | Static Dilated TCN | 18 | 26.83 m | 106.44% | 5 / 18 | 27.8% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L2) |
| **20 Seconds** | **Static Dilated TCN** | 17 | **45.73 m** | **123.95%** | **5 / 17** | **29.4%** | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L6) |
| | **NavSense Adaptive Fusion**| 17 | 49.27 m | 135.51% | 4 / 17 | 23.5% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L9) |
| | Pure Kinematics | 17 | 57.01 m | 147.98% | 4 / 17 | 23.5% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L7) |
| **30 Seconds** | **Static Dilated TCN** | 16 | **54.62 m** | **65.57%** | **5 / 16** | **31.2%** | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L10) |
| | **NavSense Adaptive Fusion**| 16 | 61.83 m | 71.38% | 4 / 16 | 25.0% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L13) |
| | Pure Kinematics | 16 | 109.57 m | 118.22% | 2 / 16 | 12.5% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L11) |
| **60 Seconds** | **NavSense Adaptive Fusion**| 13 | **96.65 m** | **16.76%** | **3 / 13** | **23.1%** | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L17) |
| | Static Dilated TCN | 13 | 95.53 m | 18.26% | 3 / 13 | 23.1% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L14) |
| | Fixed Damped Momentum | 13 | 173.71 m | 40.75% | 1 / 13 | 7.7% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L16) |
| | Pure Kinematics | 13 | 324.21 m | 89.66% | 0 / 13 | 0.0% | [Verify CSV](results/adaptive_fusion/adaptive_aggregate_summary.csv#L15) |

*(Trips with duration under 70 seconds are excluded from the 60-second evaluation window as they cannot accommodate a 60s outage plus pre-outage calibration).*

> [!NOTE]
> **Definitive Status Assessment:** **RESEARCH PROTOTYPE — NOT YET UNIVERSALLY ACHIEVED.**  
> Under continuous 60-second outages, NavSense achieves an aggregate mean drift of **16.76% (96.65 m)**. While 3 trips achieve $<10\%$ drift (`Vw12` at 3.01%, `Vta21` at 7.88%, `Vw14a` at 8.54%), universal $<10\%$ compliance across all road environments remains an open research frontier.

---

## 5. Empirical Evidence & Diagnostic Analysis

Every finding in NavSense is supported by verifiable numerical data and high-resolution diagnostic artifacts:

### 1. Multi-Horizon Adaptive Fusion Trajectory Benchmark
<div align="center">

![Adaptive Fusion Benchmark](assets/experiments/adaptive_fusion_plots.png)

</div>

* **Observed Reality:** Head-to-head evaluation across 19 held-out test trips demonstrates that Adaptive Fusion achieves the lowest aggregate 60s blackout drift at **16.76% (96.65 m)**.
* **Empirical Proof:** [`results/adaptive_fusion/adaptive_aggregate_summary.csv`](results/adaptive_fusion/adaptive_aggregate_summary.csv) • [`results/adaptive_fusion/adaptive_per_trip_results.csv`](results/adaptive_fusion/adaptive_per_trip_results.csv)
* **Detailed Phase Report:** [Phase 5.6: Adaptive Regime-Aware Velocity Fusion](docs/experiments/phase5_6_adaptive_fusion.md)

---

### 2. Spectral Vibration Speed Audit (Proof of $r = -0.032$ Rejection)
<div align="center">

![Vibration Spectrum Audit](assets/experiments/vibration_spectrum_plots.png)

</div>

* **Scientific Finding:** We tested the hypothesis that vehicle engine/road chassis vibration serves as a proxy for forward speed across 64 trips (>450,000 1-second windows). The Pearson correlation was **$r = -0.032$** (virtually zero). In passenger cars, tires and suspension damp high-frequency harmonics; dominant 2.2–2.5 Hz peaks reflect chassis resonance, not speed. Vibration was rejected as a speedometer.
* **Empirical Proof:** [`results/vibration_audit/vibration_speed_audit.csv`](results/vibration_audit/vibration_speed_audit.csv) • [`results/vibration_audit/vibration_cross_trip_results.csv`](results/vibration_audit/vibration_cross_trip_results.csv)
* **Detailed Phase Report:** [Phase 5.3: Spectral Vibration Speed Information Audit](docs/experiments/phase5_3_vibration_audit.md)

---

### 3. Highway Cruise Speed Unobservability Analysis
<div align="center">

![Highway Observability Analysis](assets/experiments/highway_observability_plots.png)

</div>

* **The Physics Boundary:** On straight motorway segments (`V-Vfa02`, 163 km), constant-speed cruising ($a \approx 0, \omega \approx 0$) produces near-zero net inertial force. Feature distributions at 80 km/h and 110 km/h are statistically indistinguishable (ROC-AUC = **0.625**). Memoryless models default to the dataset mean (~85 km/h), proving that stateful momentum integration is a physical necessity.
* **Empirical Proof:** [`results/highway_observability/highway_observability_summary.json`](results/highway_observability/highway_observability_summary.json) • [`results/highway_observability/speed_band_residuals.csv`](results/highway_observability/speed_band_residuals.csv)
* **Detailed Phase Report:** [Phase 5.4: Highway Steady-State Cruise Observability Analysis](docs/experiments/phase5_4_highway_observability.md)

---

### 4. Pedestrian Out-Of-Distribution (OOD) Forensic Deconstruction
<div align="center">

![Pedestrian OOD Analysis](assets/experiments/rooftop_gait_speed_analysis.png)

</div>

* **The 71.66 m/s Spike Explained:** Campus walking trials produced an anomalous **71.66 m/s (258 km/h)** prediction. Forensic feature attribution revealed human arm swings reach angular rates of **$3.81\text{ rad/s}$**—over 9× higher than vehicle training bounds ($<0.40\text{ rad/s}$). The network's linear projection multiplied this extreme out-of-distribution input into impossible velocities. Walking tests were formally qualified as sensor stress tests, never vehicle validation.
* **Empirical Proof:** [`results/field/`](results/field/) • [`docs/experiments/phase5_1_field_forensics.md`](docs/experiments/phase5_1_field_forensics.md)
* **Detailed Post-Mortem:** [Catalog of Failed Experiments: Failure #6](docs/failed_experiments_catalog.md#6-pedestrian--out-of-distribution-ood-explosion-7166-ms-peak)

---

### 5. Urban Stop-and-Go Runaway Clamping ($574.65\% \to 47.31\%$)
<div align="center">

![Stateful Kinematics Runaway](assets/experiments/stateful_kinematics_plots.png)

</div>

* **Arresting Quadratic Divergence:** On urban route `Vta26` with multiple traffic halts, pure kinematic integration accumulated sensor bias during stops, exploding to **574.65% drift (292.8 m error over 50 m travel)**. NavSense Causal ZVD detects standstills within 0.2s, clamping drift down to **47.31%**.
* **Empirical Proof:** [`results/stateful_kinematics/per_trip_horizon_results.csv`](results/stateful_kinematics/per_trip_horizon_results.csv) • [`results/adaptive_fusion/adaptive_per_trip_results.csv`](results/adaptive_fusion/adaptive_per_trip_results.csv)
* **Detailed Phase Report:** [Phase 5.5: Stateful Kinematic Integration & Momentum](docs/experiments/phase5_5_stateful_kinematics.md)

---

## 6. Native Android Mobile Prototype

NavSense includes a standalone native Android application in [`android/`](android/README.md):
* **On-Device ONNX Inference:** Executes `models/tcn_velocity_expanded.onnx` (162 KB) at 10.0 Hz with $<8\text{ ms}$ inference latency on consumer mobile processors.
* **Java ESKF Engine:** 15-state Error-State Kalman Filter utilizing the Efficient Java Matrix Library (EJML).
* **Offline Vector Mapping:** Integrated map rendering via OSMDroid with real-time heading orientation tracking.
* **Synthetic Blackout Testing:** In-app debug toggle allowing instant manual GNSS blackout injection during live road driving.
* **Dynamic Hardware Timestamps:** Resolves sensor delta time $\Delta t$ directly from monotonic hardware nanosecond clocks ($15\text{--}22\text{ ms}$), replacing legacy 100ms hardcoded timing assumptions.

---

## 7. Quickstart & Reproducibility

```bash
# 1. Clone the NavSense repository
git clone https://github.com/24f2001869/NavSense.git
cd NavSense

# 2. Setup Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Verify mathematical causality (100% clean certification)
python tests/test_causality_and_leakage.py

# 4. Reproduce Phase 5.6 Adaptive Fusion Benchmark
python scripts/experiments/run_adaptive_fusion_benchmark.py

# 5. Reproduce Phase 5.4 Highway Speed Observability Analysis
python scripts/experiments/run_highway_speed_observability.py

# 6. Reproduce Phase 5.3 Spectral Vibration Speed Correlation Audit
python scripts/experiments/run_vibration_speed_audit.py
```

Full step-by-step instructions in [**docs/reproducibility.md**](docs/reproducibility.md).

---

## 8. Dataset Provenance & Attribution

NavSense evaluates on the open-source **IO-VNBD** benchmark dataset (*Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning*). In accordance with open science practices and repository storage limits (2.34 GB raw), raw dataset files are not hosted in Git. Instructions for acquiring the data are documented in [`data/README.md`](data/README.md):

* **Official Repository:** [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
* **Citation:**  
  U. Onyekpe, V. Palade, S. Kanarachos, A. Szkolnik, *"IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning"*, *Data in Brief*, 35, 106885, 2021.  
  DOI: [`10.1016/j.dib.2021.106885`](https://doi.org/10.1016/j.dib.2021.106885)
* **Open Access:** [PMC7907232](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/) / [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S2352340921001694)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 9. Documentation Index

| Document | Description |
| :--- | :--- |
| [**START_HERE.md**](docs/START_HERE.md) | 5-minute technical onboarding |
| [**Current Status**](docs/current_status.md) | Authoritative status synthesis: what works, what failed, open challenges |
| [**Research Evidence Map**](docs/evidence_map.md) | Question-by-question chain-of-evidence matrix |
| [**Failed Experiments Catalog**](docs/failed_experiments_catalog.md) | Comprehensive post-mortems of 10 negative results and failed hypotheses |
| [**Numerical Audit**](docs/numerical_audit.md) | Full audit table mapping every published metric to primary CSV result files |
| [**Claim / Evidence Matrix**](docs/claim_evidence_matrix.md) | Epistemological grading (demonstrated, supported, rejected) and qualified wording |
| [**Methodology**](docs/methodology.md) | Mathematical derivations (ESKF, TCN, Kinematics, ZVD) |
| [**Architecture Specification**](docs/architecture.md) | In-depth 5-layer pipeline specification |
| [**Android Documentation**](android/README.md) | Mobile architecture, sensor timing fixes, and ONNX deployment |
| [**Third-Party Data & Licenses**](docs/third_party_data_and_licenses.md) | IO-VNBD attribution, paper citations, and software license catalog |
| [**Master Results Index**](results/RESULTS_INDEX.md) | Tabular summary of all experimental benchmarks |

---

## 10. License

This project is released under the [**MIT License**](LICENSE).  
The IO-VNBD dataset is governed by its original Creative Commons Attribution 4.0 International (CC BY 4.0) license.
