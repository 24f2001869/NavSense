# Research Evidence Map

> **Purpose:** Trace every core research question to its specific experiment, dataset partition, execution script, primary numerical results, diagnostic figures, and scientific conclusion.  
> **Auditing Principle:** No claim is made in this repository without an auditable chain of evidence.

---

## Master Traceability Matrix

| Research Question | Phase & Experiment | Dataset & Split | Execution Script | Primary Result File | Diagnostic Figure | Supported Scientific Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1: Can pure smartphone IMU double integration survive a 60s GNSS outage?** | **Phase 1:** Pure IMU Strapdown Baseline | IO-VNBD Suburban & Urban (`Vta01`, `Vtb01`) | `experiments/phase1/run_baseline.py` | `results/baseline/strapdown_baseline_metrics.json` | `assets/experiments/outage_track_drift_202714.png` | **Rejected.** Sensor bias causes cubic ($t^3$) error divergence; drift exceeds 1000% within 15 seconds. |
| **Q2: Does a standard 15-state ESKF solve GNSS-denied navigation?** | **Phase 2:** GNSS-Aided ESKF Baseline | IO-VNBD Winding & Suburban (`Vw01`, `Vta05`) | `experiments/phase2/run_eskf_blackout.py` | `results/eskf/eskf_blackout_summary.csv` | `docs/experiments/phase2_eskf.md` | **Partial.** ESKF achieves sub-meter tracking while GNSS is present, but drifts without speed observations during outages. |
| **Q3: Can Non-Holonomic Constraints (NHC) prevent lateral and vertical drift?** | **Phase 3:** NHC & Attitude Stabilization | IO-VNBD Suburban (`Vta02`, `Vta06`) | `experiments/phase3/run_nhc_experiment.py` | `results/attitude/nhc_ablation_results.json` | `docs/experiments/phase3_attitude.md` | **Qualified.** NHC bounds lateral drift, but phone mounting frame misalignments inject false rotational turning. |
| **Q4: Can a Temporal Convolutional Network (TCN) estimate forward speed across diverse trips?** | **Phase 4:** TCN Speed Estimation | IO-VNBD 64 Trips (39 Train, 6 Val, 19 Test) | `scripts/experiments/run_expanded_tcn_forensics.py` | `models/metadata/tcn_expanded_config.json` | `assets/experiments/expanded_tcn_benchmark_summary.png` | **Demonstrated.** TCN with 6.1s receptive field achieves 2.14 m/s MAE across 64 trips during active maneuvers. |
| **Q5: Why did the initial Android field tests fail catastrophically during walking?** | **Phase 5.1:** Field Forensics & OOD Analysis | Real-world phone telemetry (`test_b_walking_rooftop_*.csv`) | Forensic replay script | `results/field/pedestrian_ood_analysis.json` | `assets/experiments/rooftop_gait_speed_analysis.png` | **OOD Failure.** Walking angular rates (up to 3.81 rad/s) exceeded vehicle training range ($<0.40$ rad/s), causing 71.66 m/s predictions. Walking is NOT vehicle validation. |
| **Q6: Did the Android app have timing/integration bugs?** | **Phase 5.1.1:** Android Pipeline Audit | Live sensor callbacks on Android 14 device | Android Instrumentation Test | `android/README.md` | `assets/experiments/field_replay_improvements_comparison.png` | **Identified & Fixed.** The app loop dt was hardcoded to 100ms while sensors fired at ~15-22ms. Replaced with dynamic nano-timestamp calculation. |
| **Q7: Can vehicle chassis/engine vibration provide an auxiliary speed signal?** | **Phase 5.3:** Vibration Speed Audit | 64 IO-VNBD trips (>450,000 1-second windows) | `scripts/experiments/run_vibration_speed_audit.py` | `results/vibration_audit/vibration_speed_correlation.csv` | `assets/experiments/vibration_spectrum_plots.png` | **Rejected.** Correlation between dominant vibration frequency and vehicle speed is $r = -0.032$. Vibration cannot serve as a reliable speedometer. |
| **Q8: Why does the TCN flatline at ~85 km/h during steady highway cruising?** | **Phase 5.4:** Highway Speed Observability | IO-VNBD Motorway (`V-Vfa02`, 163 km) | `scripts/experiments/run_highway_speed_observability.py` | `results/adaptive_fusion/highway_observability_metrics.json` | `assets/experiments/highway_observability_plots.png` | **Identified.** At steady cruise, acceleration is zero and yaw rate is zero. Speed is unobservable from IMU alone (ROC-AUC = 0.625). |
| **Q9: Does stateful acceleration integration solve the steady-cruise limitation?** | **Phase 5.5:** Stateful Kinematics Evaluation | 19 Held-Out Test Trips (13 evaluated at 60s) | `scripts/experiments/run_stateful_kinematic_benchmark.py` | `results/stateful_kinematics/stateful_kinematics_summary.csv` | `assets/experiments/stateful_kinematics_plots.png` | **Regime-Dependent.** Excellent on smooth highways, but catastrophic runaway drift on urban stop-and-go (`Vta26` reached 574.65% drift). |
| **Q10: Does Adaptive Dynamic Fusion meet the SIH &lt;10% benchmark?** | **Phase 5.6:** Adaptive Regime Fusion Benchmark | 19 Held-Out Test Trips across 4 terrains | `scripts/experiments/run_adaptive_fusion_benchmark.py` | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | `assets/experiments/adaptive_fusion_plots.png` | **Partially Effective.** 60s mean drift improved to 16.76% (96.65 m); urban runaway reduced (47.31%); but only **3 / 13 (23.1%)** pass the &lt;10% SIH benchmark. |

---

## Detailed Evidence Chains

### 1. The Pedestrian OOD Failure (~71.66 m/s Peak)
- **Question:** What caused the Android prototype to report speeds exceeding 250 km/h while walking on a university rooftop?
- **Hypothesis:** Human walking gait introduces high-frequency body rotation and vertical impact dynamics far outside the distribution of a rigidly mounted vehicle phone.
- **Evidence Files:**
  - Script: Diagnostic replay in [`experiments/phase5/run_expanded_tcn_forensics.py`](../scripts/experiments/run_expanded_tcn_forensics.py)
  - Telemetry: [`data/field/test_b_walking_rooftop_20260911_015632.csv`](../data/field/idr_telemetry_20260909_103010.csv)
  - Figure: [`assets/experiments/rooftop_gait_speed_analysis.png`](../assets/experiments/rooftop_gait_speed_analysis.png)
  - Full Report: [`docs/experiments/phase5_1_field_forensics.md`](experiments/phase5_1_field_forensics.md)
- **Outcome:** Feature ablation revealed `gyro_yaw` was the primary driver. During vehicle driving, 99.9% of yaw rates fall in $[-0.40, +0.40]\text{ rad/s}$. In walking, yaw rates reached $3.81\text{ rad/s}$. The TCN's linear output projection mapped this massive input to $71.66\text{ m/s}$ ($257.9\text{ km/h}$).

---

### 2. The Vibration Speed Correlation Audit ($r = -0.032$)
- **Question:** Does road-induced or engine-induced smartphone vibration contain generalizable speed information?
- **Hypothesis:** Higher vehicle speeds transmit higher vibration frequencies and energy to the cabin dashboard through road texture and engine RPM.
- **Evidence Files:**
  - Script: [`scripts/experiments/run_vibration_speed_audit.py`](../scripts/experiments/run_vibration_speed_audit.py)
  - Results CSV: [`results/vibration_audit/vibration_speed_correlation.csv`](../results/vibration_audit/vibration_speed_audit.csv)
  - Figure: [`assets/experiments/vibration_spectrum_plots.png`](../assets/experiments/vibration_spectrum_plots.png)
  - Full Report: [`docs/experiments/phase5_3_vibration_audit.md`](experiments/phase5_3_vibration_audit.md)
- **Outcome:** Across 64 trips and 450,000+ windows, the linear correlation between dominant frequency and CAN speed was $r = -0.032$. Across motorway segments alone, $r = +0.081$. Vibration is dominated by suspension damping and phone mount elasticity, not vehicle speed. Rejected.

---

### 3. Adaptive Dynamic Regime Fusion Benchmark (60s Outages)
- **Question:** Does blending stateful kinematics, causal TCN, and zero-velocity gating achieve $<10\%$ drift across held-out test trips?
- **Hypothesis:** Kinematics handles smooth cruising; TCN handles dynamic speed changes; ZVD eliminates stationary drift.
- **Evidence Files:**
  - Benchmark Script: [`scripts/experiments/run_adaptive_fusion_benchmark.py`](../scripts/experiments/run_adaptive_fusion_benchmark.py)
  - Summary Table: [`results/adaptive_fusion/adaptive_aggregate_summary.csv`](../results/adaptive_fusion/adaptive_aggregate_summary.csv)
  - Per-Trip CSV: [`results/adaptive_fusion/adaptive_per_trip_results.csv`](../results/adaptive_fusion/adaptive_per_trip_results.csv)
  - Comparison Figure: [`assets/experiments/adaptive_fusion_plots.png`](../assets/experiments/adaptive_fusion_plots.png)
  - Full Report: [`docs/experiments/phase5_6_adaptive_fusion.md`](experiments/phase5_6_adaptive_fusion.md)
- **Outcome:** Mean 60s blackout drift is **16.76% (96.65 m)**. Catastrophic urban drift on `Vta26` was arrested from **574.65% down to 47.31%**. Winding mountain route `Vw12` achieved **3.01% drift**. However, only **3 of 13 usable trips (23.1%)** passed the SIH $<10\%$ threshold. Research prototype status confirmed.
