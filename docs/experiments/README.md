# Master Experiment Index

This directory provides the complete chronological record of all major research experiments conducted in the **SIH26168 Intelligent Dead Reckoning (IDR)** project. Every phase represents a distinct hypothesis, empirical evaluation, and evidence-backed decision.

---

## Chronological Experiment Matrix

| Phase ID | Experiment Title | Core Scientific Question | Primary Method / Model | Key Empirical Result | Decision | Authoritative Evidence |
|:---:|:---|:---|:---|:---|:---:|:---|
| **Phase 0** | [**Dataset & Reference Audit**](phase0_reference_audit.md) | Is CAN speed identical to physical wheel speed? | Cross-correlation of CAN, wheel ticks, and GPS speed | Verified linear scale factor ($r_{\text{eff}} = 0.312\text{ m}$); CAN speed has 0.1s filtering latency | **ACCEPTED** | `results/phase0_audit/` |
| **Phase 1** | [**Pure IMU Baseline**](phase1_baseline.md) | Can standalone double-integration of smartphone IMU survive 60 s? | Double integration: $p(t) = p_0 + \iint (R_{wb} a_b - g) dt$ | Unbounded quadratic drift: $d \propto \frac{1}{2} b_a t^2$. 60s drift exceeded **800–1,500%** | **REJECTED** as complete solution | `results/baseline/` |
| **Phase 2** | [**15-State ESKF Formulation**](phase2_eskf.md) | Can an Error-State Kalman Filter track sensor biases and bound attitude drift? | 15-state continuous-discrete ESKF with closed-loop feedback | Attitude drift bounded to $<2.5^\circ$; position drift reduced by $4.2\times$ during short outages | **ACCEPTED** as core state estimator | `results/c5_2c2_bias_correction.json` |
| **Phase 3** | [**Attitude, 3D Mechanization & NHC**](phase3_attitude.md) | Can vehicle non-holonomic constraints (zero lateral/vertical velocity) stabilize position? | Vehicle-frame projection + NHC updates ($v_y = 0, v_z = 0$) | Cross-track drift reduced by **62%**; along-track velocity remains unconstrained | **ACCEPTED** with gating | `results/c7_b1_kinematic_bounds_report.md` |
| **Phase 4** | [**AI Forward Velocity Estimation**](phase4_ai_velocity.md) | Can a Temporal Convolutional Network infer forward vehicle speed from smartphone IMU? | Dilated TCN (65k params, 100-step receptive field) trained across 39 trips | Achieved **3.41 m/s MAE** across 19 held-out test trips; zero data leakage certified | **ACCEPTED** as baseline velocity branch | `results/expanded_tcn_benchmark/` |
| **Phase 5.1** | [**Field Telemetry & Pedestrian OOD Bug**](phase5_1_field_forensics.md) | Why did phone walking tests trigger extreme 70 m/s predictions? | Forensic feature attribution & OOD distribution audit | Walking gait rotations caused gyro yaw to reach **+26.08σ** outside vehicle training envelope | **ACCEPTED FIX** (OOD gating + bounds) | `results/phase5_1_field_forensic_report.md` |
| **Phase 5.2** | [**Target-Balanced TCN Training**](phase5_2_target_balancing.md) | Does inverse-density loss weighting eliminate highway speed underestimation? | Continuous Gaussian KDE loss weighting + speed-stratified batch sampling | Mountain road pass rate improved to 4/6 trips; motorway underestimation remained (11.2% $\to$ 11.6%) | **PARETO TRADE-OFF** | `results/balanced_tcn_benchmark/` |
| **Phase 5.3** | [**Vibration Speed-Information Audit**](phase5_3_vibration_audit.md) | Does vehicle engine/road vibration contain generalizable vehicle speed information? | Spectral PSD peak analysis across 64 trips (>450k rolling windows) | Correlation with speed was essentially zero ($r = -0.032$); adding vibration did not improve speed MAE | **REJECTED** as primary speedometer | `results/vibration_audit/` |
| **Phase 5.4** | [**Highway Observability Forensics**](phase5_4_highway_observability.md) | Why does the feedforward TCN fail to distinguish 85 km/h from 105 km/h cruise? | ROC-AUC classification & mutual information on steady cruise windows | Steady cruise acceleration and angular rate are indistinguishable from sensor noise floor ($ROC = 0.625$) | **OBSERVABILITY BOUND** | `results/highway_observability/` |
| **Phase 5.5** | [**Stateful Kinematic Integration**](phase5_5_stateful_kinematics.md) | Can momentum preservation ($v_{k+1} = v_k + a_{\text{long}}\Delta t$) bridge steady highway cruise? | Kinematic velocity integration with pre-outage GNSS initialization | Passes on smooth mountain road (`Vw12` 1.78% drift), but catastrophically diverges in stop-and-go (`Vta26` 574.6% drift) | **REGIME TRADE-OFF** | `results/stateful_kinematics/` |
| **Phase 5.6** | [**Adaptive Regime-Aware Velocity Fusion**](phase5_6_adaptive_fusion.md) | Can causal ZVD + adaptive momentum resolve both urban runaway and cruise underestimation? | Causal multi-state complementary fusion: ZVD + Cruise Momentum + Disagreement Bounding | Suppressed urban explosion on `Vta26` (575% $\to$ 47%); achieved lowest aggregate drift (16.76%), but 60s pass rate capped at 3/13 (23.1%) | **CURRENT SYSTEM BOUNDARY** | `results/adaptive_fusion/` |

---

## Detailed Phase Documentation Links
* [Phase 0: Dataset & Reference Audit](phase0_reference_audit.md)
* [Phase 1: Pure IMU Baseline](phase1_baseline.md)
* [Phase 2: Error-State Kalman Filtering](phase2_eskf.md)
* [Phase 3: Attitude & Non-Holonomic Constraints](phase3_attitude.md)
* [Phase 4: AI Forward Speed Estimation](phase4_ai_velocity.md)
* [Phase 5.1: Field Telemetry & Pedestrian OOD Bug](phase5_1_field_forensics.md)
* [Phase 5.2: Target Balancing & Loss Density Weighting](phase5_2_target_balancing.md)
* [Phase 5.3: Spectral Vibration Speed Information Audit](phase5_3_vibration_audit.md)
* [Phase 5.4: Highway Steady-State Cruise Observability Analysis](phase5_4_highway_observability.md)
* [Phase 5.5: Stateful Kinematic Integration & Momentum](phase5_5_stateful_kinematics.md)
* [Phase 5.6: Adaptive Regime-Aware Velocity Fusion](phase5_6_adaptive_fusion.md)
