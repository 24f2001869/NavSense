# SIH26168-IDR

## AI/ML-Based Intelligent Dead Reckoning for Seamless Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: SIH26168](https://img.shields.io/badge/SIH-Problem%2026168-purple.svg)](docs/problem_statement.md)
[![Platform: Android](https://img.shields.io/badge/Platform-Android%2014-green.svg)](android/README.md)
[![Runtime: ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange.svg)](https://onnxruntime.ai/)
[![Benchmark Status](https://img.shields.io/badge/Status-Research%20Prototype-red.svg)](docs/current_status.md)

An open-source scientific research and software repository investigating **smartphone-based inertial dead reckoning for ground vehicles in GNSS-denied environments** (tunnels, dense urban skyscraper canyons, multi-level underground parking, and jamming zones).

Developed for Smart India Hackathon (SIH) Problem Statement **SIH26168**, this repository documents the complete research arc: from naive mathematical strapdown baselines through deep sequence modeling, forensic failure deconstruction, and hybrid kinematic fusion.

> 🚀 **New to the project?** Read [**START_HERE.md**](docs/START_HERE.md) for a 5-minute technical overview, or inspect the [**Research Evidence Map**](docs/evidence_map.md) for direct traceability from claims to data.

---

## 🚗 Problem

When a road vehicle enters a tunnel or dense urban corridor where GNSS (GPS/NavIC) signals are lost:
1. Conventional smartphone navigation apps freeze, jump erratically, or extrapolate blindly along the last heading.
2. Production vehicle navigation systems survive blackouts by relying on **wheel-speed odometry (CAN bus / transmission ticks)** and factory-calibrated chassis IMUs.
3. In consumer smartphones mounted on a windshield or dashboard, **wheel odometry is physically absent**. The device has access only to consumer-grade MEMS inertial sensors (noisy accelerometer, gyroscope, magnetometer).

```text
The Double Integration Dilemma:
An uncompensated accelerometer bias of just b_a ≈ 0.05 m/s² causes cubic position error growth:
Δd(t) = 0.5 · b_a · t²  (assuming constant bias after velocity integration)

At t = 10 s  ──►  Δd ≈    2.5 meters  (Manageable)
At t = 30 s  ──►  Δd ≈   22.5 meters  (Lane departure)
At t = 60 s  ──►  Δd ≈   90.0 meters  (Total navigation breakdown)
```

Without an independent speed observation or physical constraint, open-loop double integration inevitably fails.

---

## 🎯 SIH26168 Requirement

The benchmark target established by SIH Problem Statement 26168 is:
- **Continuous 60-second GNSS outage survival.**
- **Position error $< 10\%$ of total distance traveled** over the blackout duration.
- **Strictly smartphone-only inputs** (no OBD-II dongles, no external wheel sensors, no infrastructure beacons).

---

## 🧠 Our Approach

We designed and implemented an end-to-end modular dead reckoning pipeline combining classical error-state filtering, deep causal sequence modeling, physical motion gating, and dynamic regime arbitration:

![System Architecture](assets/architecture/system_architecture.svg)

### Pipeline Decomposition
1. **Sensor Ingestion & Preprocessing:** Resamples raw 100 Hz accelerometer and gyroscope streams to a uniform 10 Hz rate using causal anti-aliasing filters; strips static gravity via orientation tracking.
2. **Coordinate Frame Transformation:** Transforms raw phone body axes ($X, Y, Z$) into a leveled local navigation frame (horizontal plane vs. gravity vector).
3. **Triple-Branch Velocity Engine:**
   - *Stateful Kinematics:* Propagates longitudinal velocity ($v_{t} = v_{t-1} + a_{\text{lon}} \Delta t$) initialized from the pre-outage GNSS fix.
   - *Causal Dilated TCN:* A 1D Temporal Convolutional Network with a 6.1-second receptive field predicting forward speed directly from motion dynamics.
   - *Causal Zero-Velocity Detector (ZVD):* Strict variance and energy gating to identify stationary stops and eliminate phantom drift.
4. **Adaptive Dynamic Regime Fusion:** Causally blends kinematic momentum and neural predictions based on real-time maneuver intensity, while ZVD suppresses urban stop-and-go runaway.
5. **15-State Error-State Kalman Filter (ESKF):** Propagates 3D inertial mechanics; updates attitude and velocity states using virtual forward velocity pseudo-measurements and ZUPT during blackouts.
6. **Android Mobile Prototype:** Standalone Kotlin/Java Android app running real-time ONNX Runtime inference, 15-state ESKF, OSMDroid offline mapping, and a live synthetic blackout simulator.

---

## 📊 Research Journey

The repository documents a continuous, unvarnished intellectual journey of hypotheses, discoveries, failures, and architectural pivots:

```text
Phase 0: Reference Audit (CAN speed vs. wheel ticks vs. phone GPS)
   ↓
Phase 1: Pure IMU Baseline (Cubic error divergence; strapdown fails)
   ↓
Phase 2: Classical ESKF (Accurate during GNSS, but drifts without speed updates)
   ↓
Phase 3: Attitude & NHC (Lateral drift bounded; mounting tilt induces false turning)
   ↓
Phase 4: AI Forward Speed (Expanded TCN achieves 2.14 m/s MAE across 64 trips)
   ↓
Phase 5.1: Field Telemetry & Pedestrian OOD Failure (Walking gait caused 71.66 m/s spike)
   ↓
Phase 5.1.1: Android Pipeline Audit (Fixed 100ms hardcoded loop timing bug)
   ↓
Phase 5.3: Vibration Speed Audit (450,000+ windows; r = -0.032; vibration rejected)
   ↓
Phase 5.4: Highway Observability (Steady cruise unobservable from IMU alone; ROC-AUC = 0.625)
   ↓
Phase 5.5: Stateful Kinematics (Excellent on highway, but 574.6% urban stop-and-go runaway)
   ↓
Phase 5.6: Adaptive Regime Fusion (Arrests urban runaway; bounds 60s drift to 16.76%)
```

For the complete narrative, see [**Research Story (`docs/research_story.md`)**](docs/research_story.md) and [**Research Log (`docs/research_log.md`)**](docs/research_log.md).

---

## 🔬 Key Findings

Every claim in this repository is anchored in reproducible artifacts:

1. **Dilated TCN Excels During Maneuvers:** Across 64 trips, the expanded TCN effectively learns acceleration, deceleration, and turning dynamics with a Mean Absolute Error of 2.14 m/s.
2. **Pedestrian Motion is Out-of-Distribution (OOD):** Subjecting the vehicle-trained TCN to campus walking telemetry produced extreme predictions (**71.66 m/s**) because human arm swings reach angular rates ($3.81\text{ rad/s}$) far exceeding vehicle training ranges ($<0.40\text{ rad/s}$). Walking data is strictly a sensor stress test, never vehicle validation.
3. **Chassis Vibration Lacks Speed Observability:** An exhaustive spectral analysis across 64 trips (>450,000 temporal windows) revealed a correlation of only $r = -0.032$ between dominant vibration frequency and CAN speed. Vibration is governed by vehicle suspension resonance (2.2–2.5 Hz), not forward speed.
4. **Steady Highway Cruise Exposes an Inherent Ambiguity:** In unaccelerated, straight-line cruising ($a \approx 0, \omega \approx 0$), Newtonian physics dictates zero net inertial force. The IMU cannot distinguish 80 km/h from 110 km/h (ROC-AUC = 0.625).
5. **Stateful Kinematics Exhibits Regime Polarization:** Carrying forward GNSS velocity works well on highways (`V-Vfa02` drift: 11.69%), but unconstrained integration during urban stop-and-go creates catastrophic runaway (`Vta26` drift: **574.65%**).
6. **Adaptive Fusion Arrests Catastrophic Drift:** Blending kinematics with TCN and causal ZVD clamped `Vta26` drift down to **47.31%** and achieved **3.01% drift** on winding route `Vw12`.

Review the detailed post-mortems in [**Catalog of Failed Experiments (`docs/failed_experiments_catalog.md`)**](docs/failed_experiments_catalog.md).

---

## 📈 Current Benchmark

Evaluated across rolling outages on the **19 completely held-out test trips** of the IO-VNBD dataset (zero overlap with training or validation sets):

| Outage Horizon | Evaluated Architecture | Usable Trips | Mean Drift (m) | Mean Drift (%) | Passing Trips (&lt;10%) | Empirical Pass Rate |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **10 Seconds** | **Pure Kinematics** | 18 | **23.66 m** | **83.54%** | **6 / 18** | **33.3%** |
| | Fixed Damped Momentum | 18 | 23.49 m | 111.37% | 6 / 18 | 33.3% |
| | **Adaptive Regime Fusion** | 18 | 25.06 m | 111.53% | 5 / 18 | 27.8% |
| | Static Dilated TCN | 18 | 26.83 m | 106.44% | 5 / 18 | 27.8% |
| **20 Seconds** | **Static Dilated TCN** | 17 | **45.73 m** | **123.95%** | **5 / 17** | **29.4%** |
| | **Adaptive Regime Fusion** | 17 | 49.27 m | 135.51% | 4 / 17 | 23.5% |
| | Pure Kinematics | 17 | 57.01 m | 147.98% | 4 / 17 | 23.5% |
| **30 Seconds** | **Static Dilated TCN** | 16 | **54.62 m** | **65.57%** | **5 / 16** | **31.2%** |
| | **Adaptive Regime Fusion** | 16 | 61.83 m | 71.38% | 4 / 16 | 25.0% |
| | Pure Kinematics | 16 | 109.57 m | 118.22% | 2 / 16 | 12.5% |
| **60 Seconds** | **Adaptive Regime Fusion** | 13 | **96.65 m** | **16.76%** | **3 / 13** | **23.1%** |
| | Static Dilated TCN | 13 | 95.53 m | 18.26% | 3 / 13 | 23.1% |
| | Fixed Damped Momentum | 13 | 173.71 m | 40.75% | 1 / 13 | 7.7% |
| | Pure Kinematics | 13 | 324.21 m | 89.66% | 0 / 13 | 0.0% |

*(Note: Of the 19 test trips, 6 trips are shorter than 70s and cannot support a 60s outage plus calibration window).*

> [!IMPORTANT]
> **Project Status:** **RESEARCH PROTOTYPE — SIH BENCHMARK NOT YET UNIVERSALLY ACHIEVED.**  
> Under continuous 60-second outages, Adaptive Regime Fusion achieves an aggregate drift of **16.76% (96.65 m)**. While 3 trips achieve $<10\%$ drift (`Vw12` at 3.01%, `Vta21` at 7.88%, `Vw14a` at 8.54%), universal compliance across all road environments remains an open research problem.

---

## 📱 Android Prototype

The repository includes a complete native Android application in [`android/`](android/README.md):
- **On-Device Neural Inference:** Executes `tcn_velocity_expanded.onnx` (162 KB) via ONNX Runtime Android at 10 Hz with $<8\text{ ms}$ latency.
- **Java ESKF Engine:** 15-state Error-State Kalman Filter implemented with the Efficient Java Matrix Library (EJML).
- **Mapping & Visualization:** Offline map rendering via OSMDroid with a dynamic compass rose display.
- **Blackout Simulation:** Integrated UI toggle enabling instant GNSS blackout injection for real-time testing.
- **Field Telemetry Logs:** Diagnostic CSV traces captured on campus hardware are preserved in `data/field/` (strictly qualified as sensor stress tests, not vehicle navigation validation).

---

## 🧪 Reproducibility

### Setup
```bash
# Clone the repository
git clone https://github.com/24f2001869/SIH26168-IDR.git
cd SIH26168-IDR

# Create virtual environment & install dependencies
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Instant verification test (causality & zero leakage)
python tests/test_causality_and_leakage.py
```

### Reproduce Benchmarks
```bash
# Phase 5.6: Adaptive Regime-Aware Fusion Benchmark (Primary Result)
python scripts/experiments/run_adaptive_fusion_benchmark.py

# Phase 5.4: Highway Speed Observability Analysis
python scripts/experiments/run_highway_speed_observability.py

# Phase 5.3: Spectral Vibration Correlation Audit
python scripts/experiments/run_vibration_speed_audit.py
```

See [**Reproducibility Guide (`docs/reproducibility.md`)**](docs/reproducibility.md) for full instructions.

---

## 📂 Repository Structure

```text
SIH26168-IDR/
├── README.md                      ← Master project overview
├── LICENSE                        ← MIT License
├── .gitignore                     ← Excludes raw 2.34 GB data, caches, build files
├── requirements.txt               ← Pinned scientific dependencies
│
├── docs/                          ← Complete technical documentation
│   ├── START_HERE.md              ← 5-minute technical onboarding
│   ├── current_status.md          ← Honest status: What We Know / Think / Don't Know
│   ├── evidence_map.md            ← Traceability from research questions to evidence
│   ├── failed_experiments_catalog.md ← Post-mortems of 10 negative results
│   ├── research_story.md          ← Narrative intellectual history
│   ├── research_log.md            ← Chronological lab timeline
│   ├── numerical_audit.md         ← Verification table for all published numbers
│   ├── claim_evidence_matrix.md   ← Epistemological grading & safe wording
│   ├── third_party_data_and_licenses.md ← IO-VNBD provenance & open-source licenses
│   ├── github_file_audit.md       ← Large file audit & exclusion documentation
│   ├── experiments/               ← Dedicated reports for Phases 0 through 5.6
│   └── decisions/                 ← Architecture Decision Records (ADRs)
│
├── src/                           ← Core operational Python package
│   ├── data/                      ← Dataset ingestion & iterators
│   ├── preprocessing/             ← Filtering & frame transformation
│   ├── navigation/                ← Strapdown mechanization & kinematics
│   ├── fusion/                    ← 15-state ESKF & adaptive blending
│   ├── ml/                        ← Temporal Convolutional Networks
│   ├── map/                       ← Map-matching & OSM integration
│   └── evaluation/                ← Rolling outage evaluation engine
│
├── scripts/                       ← Standalone reproduction scripts
│   ├── experiments/               ← Primary benchmark execution scripts
│   └── utilities/                 ← Path sanitization & data checking
│
├── notebooks/                     ← Interactive Jupyter exploration notebooks
├── results/                       ← Authoritative numerical results & tables
│   ├── RESULTS_INDEX.md           ← Master tabular summary of all benchmarks
│   └── adaptive_fusion/           ← Phase 5.6 benchmark results (CSV & JSON)
│
├── models/                        ← Lightweight ONNX models & scalers (162 KB)
├── configs/                       ← Filter and model configuration YAMLs
├── data/                          ← Dataset instructions & field telemetry
│   ├── README.md                  ← IO-VNBD acquisition & split guide
│   └── field/                     ← Real-world smartphone sensor logs
│
├── android/                       ← Native Android Studio project & README
├── tests/                         ← Automated unit & causality test suite
├── assets/                        ← Curated SVG architecture & diagnostic plots
└── archive/                       ← Superseded & exploratory research code
```

---

## 📚 Dataset

This project uses the publicly available **IO-VNBD** benchmark dataset. Due to size constraints (2.34 GB raw) and upstream terms of use, the raw dataset is **not redistributed in this repository**. Please obtain it directly from the original authors:

* **Official Repository:** [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
* **Dataset Publication:**  
  U. Onyekpe, V. Palade, S. Kanarachos, A. Szkolnik, *"IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning"*, *Data in Brief*, 35, 106885, 2021.  
  DOI: [`10.1016/j.dib.2021.106885`](https://doi.org/10.1016/j.dib.2021.106885)
* **Open-Access Article:** [PMC7907232](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/) / [ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S2352340921001694)

See [`data/README.md`](data/README.md) for full ingestion, preprocessing, and trip partition instructions.

---

## ⚠️ Limitations

1. **Unobservable Steady Cruising:** At constant highway speeds, zero acceleration and zero rotation mean the smartphone IMU cannot physically observe absolute velocity.
2. **Yaw Orientation Drift:** Without reliable magnetometer data (which is distorted by steel car frames), consumer gyroscope bias drift slowly rotates the trajectory vector during long straightaways.
3. **Vehicle Suspension Filtering:** Vehicle springs, dampers, and rubber phone mounts act as low-pass mechanical filters, destroying direct correlations between chassis vibration and road speed.
4. **Lack of Synchronized Road Ground Truth:** Real-world field tests on campus phones validated sensor sampling rates and UI execution, but physical road validation against RTK-GNSS remains pending.

Read the exhaustive boundary catalog in [**Limitations (`docs/limitations.md`)**](docs/limitations.md).

---

## 🚧 Current Open Problems

1. **Long-Horizon Highway Observability:** Determining whether subtle secondary signals (e.g., road bump acoustic spectral shape, barometric cabin turbulence) can weakly observe cruise speed without external hardware.
2. **Dynamic Online Frame Auto-Alignment:** Formulating continuous optimization to track phone-to-vehicle mounting angles during dynamic driving maneuvers.
3. **Multi-Rate Map-Matching Feedback:** Developing a particle filter that uses road network topology to bound heading drift without catastrophic snapping errors.

See [**Open Questions (`docs/decisions/open_questions.md`)**](docs/decisions/open_questions.md).

---

## 📄 Research Documentation

| Document | Description |
| :--- | :--- |
| [**START_HERE.md**](docs/START_HERE.md) | 5-minute technical onboarding guide |
| [**Current Status**](docs/current_status.md) | Transparent accounting of what works, what doesn't, and what remains unknown |
| [**Research Evidence Map**](docs/evidence_map.md) | Complete question-to-evidence audit matrix |
| [**Failed Experiments Catalog**](docs/failed_experiments_catalog.md) | Detailed post-mortems of 10 negative results and failed hypotheses |
| [**Numerical Audit**](docs/numerical_audit.md) | 100% verification table matching all claims to source result files |
| [**Claim / Evidence Matrix**](docs/claim_evidence_matrix.md) | Epistemological grading (demonstrated, supported, rejected) and safe wording |
| [**Methodology**](docs/methodology.md) | Complete mathematical derivations (ESKF, TCN, Kinematics, ZVD) |
| [**Architecture Specification**](docs/architecture.md) | In-depth 5-layer pipeline design |
| [**Reproducibility Guide**](docs/reproducibility.md) | Environment setup, execution commands, and verification protocols |
| [**Android Documentation**](android/README.md) | Mobile architecture, sensor timing fixes, and ONNX deployment |
| [**Third-Party Data & Licenses**](docs/third_party_data_and_licenses.md) | IO-VNBD attribution, paper citations, and software license catalog |
| [**Master Results Index**](results/RESULTS_INDEX.md) | Tabular summary of all experimental benchmarks |

---

## 📜 License

This project is licensed under the [**MIT License**](LICENSE).  
The IO-VNBD dataset is governed by its original Creative Commons Attribution 4.0 International (CC BY 4.0) terms as published by Onyekpe et al.
