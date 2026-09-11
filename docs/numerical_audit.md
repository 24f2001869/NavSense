# Numerical Integrity Audit

This document formally audits and cross-references every critical numerical figure and metric published in the repository against its authoritative primary data file.

---

## 1. Primary Metrics Verification Table

| Claimed Metric Description | Documented Value | Authoritative Primary Source File | Primary Key / Column | Verification Status |
|:---|:---:|:---|:---|:---:|
| **Peak Pedestrian Speed Spike** | **71.66 m/s (258 km/h)** | `results/phase4_1_forensics/tcn_kin_forensic_report.md` | Line 18: `max_pred = 71.66` | 🟢 **VERIFIED** |
| **Peak Gyro Yaw OOD Sigma** | **+26.08σ** | `results/phase4_1_forensics/tcn_kin_forensic_report.md` | Table 1: `gyro_yaw = +2.85 rad/s (+26.08σ)` | 🟢 **VERIFIED** |
| **Expanded TCN 60s Unweighted Drift** | **93.0 m (14.2%)** | `results/expanded_tcn_benchmark/per_trip_forensic.csv` | Mean of `drift_60s_m` across 19 trips | 🟢 **VERIFIED** |
| **Expanded TCN Distance-Weighted Drift**| **148.0 m** | `results/expanded_tcn_benchmark/per_trip_forensic.csv` | Epoch-weighted distance sum | 🟢 **VERIFIED** |
| **Vibration Dominant Freq Correlation** | **r = -0.032** | `results/vibration_audit/vibration_audit_report.md` | Section 3.1: Pearson correlation across 64 trips | 🟢 **VERIFIED** |
| **Vibration Motorway Suspension Peak** | **2.86 to 2.93 Hz** | `results/vibration_audit/vibration_audit_report.md` | Section 3.2: Motorway PSD dominant frequency | 🟢 **VERIFIED** |
| **Motorway Vibration Ablation MAE** | **4.39 m/s → 4.38 m/s**| `results/vibration_audit/vibration_audit_report.md` | Table: Motorway RF with/without FFT features | 🟢 **VERIFIED** |
| **Highway Cruise ROC-AUC Separability** | **ROC-AUC = 0.625** | `results/highway_observability/highway_observability_report.md` | Section 3: Logistic Regression & RF ROC-AUC | 🟢 **VERIFIED** |
| **Pure Kinematics Vta26 60s Drift** | **574.65% (292.84 m)**| `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 105: `Vta26,Pure_Kinematics,60s,drift_pct=574.65` | 🟢 **VERIFIED** |
| **Adaptive Fusion Vta26 60s Drift** | **47.31% (66.26 m)** | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 97: `Vta26,Adaptive_Regime_Fusion,60s,drift_pct=47.31`| 🟢 **VERIFIED** |
| **Fixed Momentum Vw12 60s Drift** | **1.78% (26.52 m)** | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 181: `Vw12,Fixed_Damped_Momentum,60s,drift_pct=1.78` | 🟢 **VERIFIED** |
| **Adaptive Fusion Vw12 60s Drift** | **3.01% (45.30 m)** | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 177: `Vw12,Adaptive_Regime_Fusion,60s,drift_pct=3.01` | 🟢 **VERIFIED** |
| **Adaptive Fusion V-Vfa02 60s Drift** | **11.69% (157.21 m)**| `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 5: `V-Vfa02,Adaptive_Regime_Fusion,60s,drift_pct=11.69`| 🟢 **VERIFIED** |
| **Adaptive Fusion 60s Aggregate Drift** | **16.76% (96.65 m)** | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 17: `60s,Adaptive_Regime_Fusion,13 trips,16.76%` | 🟢 **VERIFIED** |
| **Pure Kinematics 60s Aggregate Drift**| **89.66% (324.21 m)**| `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 15: `60s,Pure_Kinematics,13 trips,89.66%` | 🟢 **VERIFIED** |
| **Fixed Momentum 60s Aggregate Drift** | **40.75% (173.71 m)**| `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 16: `60s,Fixed_Damped_Momentum,13 trips,40.75%` | 🟢 **VERIFIED** |
| **Static TCN 60s Aggregate Drift** | **18.26% (95.53 m)** | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 14: `60s,Static_TCN,13 trips,18.26%` | 🟢 **VERIFIED** |
| **60s SIH Pass Rate (Adaptive Fusion)**| **3 / 13 (23.1%)** | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 17: `trips_passed_10pct=3, pass_rate=23.1` | 🟢 **VERIFIED** |
| **10s SIH Pass Rate (Pure Kinematics)**| **6 / 18 (33.3%)** | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 3: `trips_passed_10pct=6, pass_rate=33.3` | 🟢 **VERIFIED** |

---

## 2. Test Set Trip Duration Census

Audit of why fewer trips support longer blackout horizons (`results/expanded_tcn_benchmark/per_trip_forensic.csv`):

| Test Trip Name | Road Category | Total Recorded Duration | Eligible at 10s? | Eligible at 20s? | Eligible at 30s? | Eligible at 60s? | Exclusion Reason at 60s |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Vtb10** | Dense Urban | **9.6 s** | ❌ (No) | ❌ (No) | ❌ (No) | ❌ (No) | Total duration < 10 s |
| **Vw13** | Winding Mountain | **18.5 s** | ✅ (Yes) | ❌ (No) | ❌ (No) | ❌ (No) | Total duration < 20 s |
| **Vtb11** | Dense Urban | **26.2 s** | ✅ (Yes) | ✅ (Yes) | ❌ (No) | ❌ (No) | Total duration < 30 s |
| **Vtb09** | Dense Urban | **35.3 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ❌ (No) | Total duration < 70 s (outage + pre-window) |
| **Vtb12** | Dense Urban | **34.8 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ❌ (No) | Total duration < 70 s (outage + pre-window) |
| **Vta25** | Suburban Town | **54.7 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ❌ (No) | Total duration < 70 s (outage + pre-window) |
| **Vta21** | Suburban Town | **198.0 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta22** | Suburban Town | **146.5 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta23** | Suburban Town | **101.1 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta24** | Suburban Town | **107.2 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta26** | Suburban Town | **183.6 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta27** | Suburban Town | **244.1 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vta28** | Suburban Town | **411.1 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vw12** | Winding Mountain | **81.9 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vw14a** | Winding Mountain | **303.9 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vw14b** | Winding Mountain | **1,948.9 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vw15** | Stationary Control | **128.1 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **Vw16a** | Winding Mountain | **578.0 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **V-Vfa02** | Motorway | **6,742.4 s** | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | Fully eligible (>70 s) |
| **TOTALS** | — | — | **18 Trips** | **17 Trips** | **16 Trips** | **13 Trips** | 100% Accounted For |
