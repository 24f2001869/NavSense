# START HERE: 5-Minute Project Onboarding

> **Project:** SIH26168-IDR (AI/ML-Based Intelligent Dead Reckoning for GNSS-Denied Navigation)  
> **Target Audience:** Researchers, evaluators, judges, software engineers, and new contributors.  
> **Reading Time:** ~5 minutes.  

Welcome to the `SIH26168-IDR` research repository. This document gives you an immediate, transparent, and accurate understanding of the project, its discoveries, and where to look next.

---

## 1. What is SIH26168?
Smart India Hackathon (SIH) Problem Statement 26168 challenges teams to solve **seamless vehicular navigation in GNSS-denied environments** (such as underground tunnels, dense urban skyscraper canyons, and covered parking garages) using **only consumer smartphone sensors** (accelerometer, gyroscope, magnetometer) without external infrastructure or vehicle CAN bus connections. The target benchmark is to maintain positional tracking with **less than 10% drift relative to total distance traveled during a 60-second blackout**.

---

## 2. What Did We Build?
We developed a complete end-to-end inertial dead reckoning pipeline spanning Python research benchmarks and an on-device Android mobile application:
1. **Frame Alignment & Preprocessing:** Converts noisy smartphone body measurements into a leveled navigation frame.
2. **Velocity Estimation Engine:** Combines three distinct paradigms:
   - *Stateful Kinematics:* Integrates longitudinal acceleration from last known GNSS speed.
   - *Causal Temporal Convolutional Network (TCN):* 1D dilated convolutions with a 6.1-second memory window predicting forward velocity directly from motion dynamics.
   - *Causal Zero-Velocity Detector (ZVD):* Strict physics-based gating to detect stops and arrest runaway integration.
3. **Adaptive Dynamic Regime Fusion:** Intelligently arbitrates between kinematics, TCN, and ZVD based on vehicle motion regime (steady cruise, dynamic acceleration, stop-and-go).
4. **15-State Error-State Kalman Filter (ESKF):** Fuses high-rate IMU strapdown mechanization with virtual velocity updates during GNSS blackouts.
5. **Android Prototype App:** Real-time Kotlin/Java application running local ONNX neural inference and a live map display with manual blackout simulation.

---

## 3. What Data Did We Use?
- **Primary Benchmark:** The public **IO-VNBD** dataset (*Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning*, Onyekpe et al., *Data in Brief* 2021). 
- **Subset Used:** 64 synchronized trips across Motorway, Dense Urban, Suburban, and Winding Mountain routes.
- **Leakage Prevention:** Split strictly at the **whole-trip level** (39 training trips, 6 validation trips, 19 held-out test trips).
- **Physical Sensor Logs:** Real-world smartphone sensor telemetry collected on university grounds for sensor noise calibration and timing bug verification (strictly qualified as stress tests, not vehicle navigation validation).

---

## 4. What Did We Test?
We tested rolling GNSS outages ranging from 10 seconds to 60 seconds across all 19 held-out test trips, evaluating:
- Pure IMU double integration (baseline).
- Classical GNSS-aided ESKF blackout behavior.
- Traditional ML (Ridge Regression, Random Forest) vs. Deep Sequence Models (TCN).
- Cross-trip and out-of-distribution (OOD) generalization (including walking telemetry).
- Engine/chassis vibration frequency spectrum (>450,000 temporal windows).
- Highway steady-state speed observability.
- Adaptive regime fusion across diverse road topologies.

---

## 5. What Works?
- ✅ **Dynamic Speed Tracking:** The causal TCN accurately predicts speed during active vehicle maneuvers (turns, accelerations, braking) with a Mean Absolute Error of ~2.14 m/s.
- ✅ **Stationary Gating:** Causal ZVD reliably eliminates phantom drift when stopped at traffic lights.
- ✅ **Catastrophic Failure Arrest:** Adaptive fusion reduced runaway urban stop-and-go drift on trip `Vta26` from **574.65% down to 47.31%**.
- ✅ **Low-Drift Scenarios:** On certain winding and suburban routes with rich rotational signals, the system achieves **3.01% drift** (`Vw12`) and **7.88% drift** (`Vta21`), comfortably meeting the SIH target.

---

## 6. What Doesn't?
- ❌ **Highway Constant Cruise:** At steady speeds (e.g., 90–110 km/h) on flat highways, vehicle acceleration is near zero and turns are absent. The smartphone IMU becomes unobservable, causing the static TCN to regress toward its training mean (~85 km/h).
- ❌ **Vibration as Speedometer:** An exhaustive spectral analysis of 64 trips confirmed that smartphone vibration exhibits a correlation of only $r = -0.032$ with vehicle speed, disproving vibration as a generalizable speed proxy.
- ❌ **Pedestrian / OOD Fragility:** Feeding human walking data into the vehicle-trained TCN produced extreme, erroneous predictions of **71.66 m/s** due to high-amplitude arm-swing angular velocities.
- ❌ **Yaw Drift in Low-Dynamic Regimes:** Without magnetometers (which are easily corrupted inside steel car cabins), yaw angles slowly drift, deflecting straight-line dead reckoning trajectories.

---

## 7. What is the Current Benchmark?
We report our numbers transparently without cherry-picking:

| Horizon | Usable Trips | Mean Drift (Distance) | Mean Drift (%) | SIH &lt;10% Passes | Pass Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **10 Seconds** | 18 | 25.06 m | 14.88% | **5 / 18** | 27.8% (Kinematics: 33.3%) |
| **20 Seconds** | 17 | 45.47 m | 16.31% | **4 / 17** | 23.5% |
| **30 Seconds** | 16 | 60.10 m | 17.06% | **4 / 16** | 25.0% |
| **60 Seconds** | 13 | 96.65 m | **16.76%** | **3 / 13** | **23.1%** |

> **Current Verdict:** The system is a **functional research prototype**, but the SIH &lt;10% target is **not yet universally achieved** across all road environments.

---

## 8. Where Are the Important Files?

```text
SIH26168-IDR/
├── README.md                      ← Master repository documentation
├── docs/
│   ├── START_HERE.md              ← This document
│   ├── architecture.md            ← In-depth 5-layer pipeline design
│   ├── evidence_map.md            ← Question-to-evidence matrix
│   ├── current_status.md          ← Honest status assessment
│   ├── research_story.md          ← Complete intellectual research narrative
│   ├── reproducibility.md         ← Step-by-step reproduction guide
│   └── experiments/               ← Dedicated reports for Phases 0 through 5.6
├── src/                           ← Production Python navigation package
├── scripts/experiments/           ← Self-contained benchmark reproduction scripts
├── results/RESULTS_INDEX.md       ← Master tabular results index
├── models/tcn_velocity_expanded.onnx ← Lightweight ONNX model (162 KB)
└── android/                       ← Native Android application source
```

---

## 9. How Can Someone Reproduce It?

```bash
# 1. Clone repository (once published)
git clone https://github.com/24f2001869/SIH26168-IDR.git
cd SIH26168-IDR

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the Causality & Leakage Verification Test (instant)
python tests/test_causality_and_leakage.py

# 4. Download IO-VNBD dataset (follow data/README.md instructions)
# Extract to data/raw/IO-VNBD-repo/

# 5. Run the Phase 5.6 Adaptive Fusion Benchmark
python scripts/experiments/run_adaptive_fusion_benchmark.py
```

For full details, see [`docs/reproducibility.md`](reproducibility.md).
