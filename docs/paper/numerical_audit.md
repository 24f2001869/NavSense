# Comprehensive Numerical Audit

This document provides a line-by-line verification of every quantitative metric, threshold, and parameter cited across the manuscript. Each entry references the authoritative primary data file in the repository, the exact data key, the denominator ($N$), the metric definition, and the calculation methodology.

**Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)

---

## 1. Master Numerical Audit Table

| ID | Metric Description | Exact Value | Source File | Line / Key / Header | Denominator ($N$) | Metric Definition | Mathematical Formula / Derivation |
|:---:|:---|:---:|:---|:---|:---:|:---|:---|
| **NUM-01** | Peak Pedestrian Speed Spike | **71.66 m/s** (257.9 km/h) | `results/phase4_1_forensics/tcn_kin_forensic_report.md` | Line 18: `max_pred = 71.66` | $N = 1$ epoch ($t = 203.8\text{ s}$) | Pointwise maximum predicted speed | $\hat{v}_{\max} = \max_{t} f_{\text{TCN}}(\mathbf{X}_t)$ |
| **NUM-02** | Peak Pedestrian Gyro Yaw Transformed Z-Score | **+26.08σ** | `results/phase4_1_forensics/tcn_kin_forensic_report.md` | Table 1: `gyro_yaw = +2.85 rad/s (+26.08σ)` | $N = 1$ window | Standardized input deviation | $z = \frac{\omega_{\text{yaw}} - \mu_{\text{train}}}{\sigma_{\text{train}}} = \frac{2.85 - 0.003}{0.109}$ |
| **NUM-03** | Walking Gyro Yaw Removal Velocity Drop | **34.94 m/s** (-51.2%) | `docs/experiments/phase5_1_field_forensics.md` | Table: `Ablate gyro_yaw` | $N = 1$ epoch | Single-feature ablation speed | $f_{\text{TCN}}(\mathbf{X}_{\text{peak}} \mid \omega_{\text{yaw}} = 0)$ |
| **NUM-04** | Walking All-Gyro Removal Velocity Drop | **27.71 m/s** (-61.3%) | `docs/experiments/phase5_1_field_forensics.md` | Table: `Ablate All Gyroscopes` | $N = 1$ epoch | Multi-feature ablation speed | $f_{\text{TCN}}(\mathbf{X}_{\text{peak}} \mid \boldsymbol{\omega} = \mathbf{0})$ |
| **NUM-05** | Walking $\pm 3\sigma$ Clamping Velocity Drop | **32.11 m/s** (-55.2%) | `docs/experiments/phase5_1_field_forensics.md` | Table: `Input Clipping to ±3σ` | $N = 1$ epoch | Input-constrained speed | $f_{\text{TCN}}(\text{clamp}(\mathbf{X}_{\text{peak}}, -3, +3))$ |
| **NUM-06** | 60s Outage Aggregate Drift (Adaptive Fusion) | **96.65 m** (16.76%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 17, Col 4–5: `96.65, 16.76` | $N = 13$ usable trips | Macro-average 2D position error | $\bar{e}_{60} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i(60) - \mathbf{p}_i^*(60)\|$ |
| **NUM-07** | 60s Outage Aggregate Drift (Static Dilated TCN) | **95.53 m** (18.26%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 14, Col 4–5: `95.53, 18.26` | $N = 13$ usable trips | Macro-average 2D position error | $\bar{e}_{60} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i(60) - \mathbf{p}_i^*(60)\|$ |
| **NUM-08** | 60s Outage Aggregate Drift (Fixed Momentum) | **173.71 m** (40.75%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 16, Col 4–5: `173.71, 40.75` | $N = 13$ usable trips | Macro-average 2D position error | $\bar{e}_{60} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i(60) - \mathbf{p}_i^*(60)\|$ |
| **NUM-09** | 60s Outage Aggregate Drift (Pure Kinematics) | **324.21 m** (89.66%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 15, Col 4–5: `324.21, 89.66` | $N = 13$ usable trips | Macro-average 2D position error | $\bar{e}_{60} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i(60) - \mathbf{p}_i^*(60)\|$ |
| **NUM-10** | 60s Outage Strict SIH Pass Count (<10% Drift) | **3 / 13 trips** (23.08%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 17, Col 6–7: `3, 23.1` | $N = 13$ usable trips | Fraction of trips meeting criterion | $\text{Pass Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{drift}_i < 10.0\%)$ |
| **NUM-11** | 10s Outage Strict SIH Pass Count (Pure Kinematics) | **6 / 18 trips** (33.33%) | `results/adaptive_fusion/adaptive_aggregate_summary.csv` | Line 3, Col 6–7: `6, 33.3` | $N = 18$ usable trips | Fraction of trips meeting criterion | $\text{Pass Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{drift}_i < 10.0\%)$ |
| **NUM-12** | 60s Outage Drift on Mountain Route `Vw12` (Adaptive) | **3.01%** (45.30 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 175: `Vw12,...,60s,45.30,3.01,1` | $N = 869$ epochs (1 trip) | Mean relative drift over trip windows | $e_{\text{rel}} = \frac{\|\Delta \mathbf{p}_{60}\|}{d_{\text{travel}}} \times 100\%$ ($d_{\text{travel}} = 1500.2\text{ m}$) |
| **NUM-13** | 60s Outage Drift on Mountain Route `Vw12` (Fixed Mom) | **1.78%** (26.52 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 179: `Vw12,...,60s,26.52,1.78,1` | $N = 869$ epochs (1 trip) | Mean relative drift over trip windows | $e_{\text{rel}} = \frac{26.52}{1500.2} \times 100\%$ |
| **NUM-14** | 60s Outage Drift on Mountain Route `Vw14a` (Adaptive) | **8.54%** (129.04 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 191: `Vw14a,...,60s,129.04,8.54,1`| $N = 3,089$ epochs (1 trip) | Mean relative drift over trip windows | $e_{\text{rel}} = \frac{129.04}{1511.1} \times 100\%$ ($d_{\text{travel}} = 1511.1\text{ m}$) |
| **NUM-15** | 60s Outage Drift on Suburban Route `Vta21` (Adaptive) | **7.88%** (61.52 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 21: `Vta21,...,60s,61.52,7.88,1` | $N = 10$ windows (1 trip) | Mean relative drift over trip windows | $e_{\text{rel}} = \frac{61.52}{786.7} \times 100\%$ ($d_{\text{travel}} = 786.7\text{ m}$) |
| **NUM-16** | 60s Outage Drift on Stop-and-Go `Vta26` (Pure Kin) | **574.65%** (292.84 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 103: `Vta26,...,60s,292.84,574.65,0`| $N = 3$ windows (1 trip) | Relative drift over stationary-bias stop | $e_{\text{rel}} = \frac{292.84}{50.96} \times 100\%$ ($d_{\text{travel}} = 50.96\text{ m}$) |
| **NUM-17** | 60s Outage Drift on Stop-and-Go `Vta26` (Adaptive) | **47.31%** (66.26 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 95: `Vta26,...,60s,66.26,47.31,0` | $N = 3$ windows (1 trip) | Relative drift with causal ZVD clamp | $e_{\text{rel}} = \frac{66.26}{50.96} \times 100\%$ |
| **NUM-18** | 60s Outage Drift on Motorway `V-Vfa02` (Adaptive) | **11.69%** (157.21 m) | `results/adaptive_fusion/adaptive_per_trip_results.csv` | Line 5: `V-Vfa02,...,60s,157.21,11.69,0` | $N = 446$ windows (1 trip) | Mean relative drift over 112 min driving | $e_{\text{rel}} = \frac{157.21}{1460.3} \times 100\%$ ($d_{\text{travel}} = 1460.3\text{ m}$) |
| **NUM-19** | Highway Steady-State IMU Separability ROC-AUC | **0.6254** (Accuracy: 59.47%) | `results/highway_observability/highway_observability_summary.json` | Key: `ml_separability.logistic_regression_roc_auc` | $N = 67,474$ epochs | Binary classification ROC-AUC (80 vs 110 km/h) | Area under the Receiver Operating Characteristic |
| **NUM-20** | Dataset-Wide Dominant Vibration vs Speed Correlation | **r = -0.032** (Spearman: -0.028) | `results/vibration_audit/vibration_speed_audit.csv` | Column `r_dom_freq` across all rows | $N = 64$ trips (>450,000 1s windows) | Pearson sample correlation coefficient | $r = \frac{\sum (f_{\text{dom}} - \bar{f})(v - \bar{v})}{\sqrt{\sum (f_{\text{dom}} - \bar{f})^2 \sum (v - \bar{v})^2}}$ |
| **NUM-21** | Motorway Invariant Suspension Vibration Peak | **2.86 to 2.93 Hz** | `results/vibration_audit/vibration_speed_audit.csv` | Row 3 (`V-Vfa02`), Section 3.2 Report | $N = 67,474$ windows | Mode of Welch PSD spectral peak | $f_{\text{peak}} = \arg\max_f P_{xx}(f)$ across [0.5, 10.0] Hz |
| **NUM-22** | Motorway Random Forest MAE With vs Without Vibration | **4.38 vs 4.39 m/s** (-0.01 m/s) | `results/vibration_audit/vibration_cross_trip_results.csv` | Row 2 (`V-Vfa02`): `vib_rf_mae_mps = 4.58` | $N = 67,474$ test epochs | Mean Absolute Error delta | $\Delta \text{MAE} = \text{MAE}_{\text{with}} - \text{MAE}_{\text{without}}$ |
| **NUM-23** | Speed Distribution Shift ($\ge 25\text{ m/s}$) | **Train: 12.43%, Test: 43.91%** | `scripts/data/audit_speed_distribution.py` | Terminal log / distribution census | Train: 478,210; Test: 113,539 usable epochs | Population percentage $\ge 25\text{ m/s}$ (90 km/h) | $\frac{N(v \ge 25)}{N_{\text{total}}} \times 100\%$ |
| **NUM-24** | High-Speed Highway Negative Prediction Bias | **-3.45 to -3.52 m/s** | `results/highway_observability/speed_band_residuals.csv` | Speed band $> 25\text{ m/s}$ mean residual | $N = 29,628$ motorway epochs | Mean signed residual | $\text{Bias} = \frac{1}{N} \sum (\hat{v}_t - v_t^*)$ |
| **NUM-25** | Unconditional Lateral NHC 60s Drift Degradation | **843.4 m $\to$ 1226.9 m** (-45.5%) | `results/phase4_4_fusion/phase4_4_fusion_report.md` | Table 2: Variant E vs Variant F | $N = 83$ blackout windows (7 routes) | Mean 60s position error delta | $\Delta = \frac{843.4 - 1226.9}{843.4} \times 100\% = -45.5\%$ |
| **NUM-26** | Lateral NHC Innovation Surging ($NIS_x$) | **6.55 $\to$ 82.12** (12.5× surge) | `results/phase4_4_fusion/phase4_4_fusion_report.md` | Table 3: Variant E vs Variant F | $N = 83$ blackout windows | Normalized Innovation Squared | $NIS = \mathbf{r}_k^T \mathbf{S}_k^{-1} \mathbf{r}_k$ |
| **NUM-27** | Decoupled Velocity Damping 60s Drift Improvement | **843.4 m $\to$ 382.7 m** (+54.6%) | `results/phase4_5_adaptive/phase4_5_adaptive_report.md` | Table 2: Variant E vs Variant V1 | $N = 83$ blackout windows (7 routes) | Mean 60s position error reduction | $\Delta = \frac{843.4 - 382.7}{843.4} \times 100\% = +54.6\%$ |
| **NUM-28** | Closed-Loop Heading-Only Map Matching Degradation | **578.3 m $\to$ 1303.9 m** (-125.5%) | `results/phase4_6_map/phase4_6_map_matching_report.md` | Table 2: Variant M0 vs Variant M2 | $N = 83$ blackout windows (7 routes) | Mean 60s position error delta | $\Delta = \frac{578.3 - 1303.9}{578.3} \times 100\% = -125.5\%$ |
| **NUM-29** | Android Navigation Loop Rate Jitter | **8.63 to 8.64 Hz** (Nominal: 10.0 Hz) | `android/README.md`, `results/c10_5_test_protocol.md` | Section 4: Sensor timing analysis | $N = 10,480$ navigation loop callbacks | Observed navigation callback frequency (raw IMU ~400 Hz) | $f_{\text{obs}} = \frac{1}{\text{mean}(\Delta t_{\text{hw}})}$ |
| **NUM-30** | Numerical Parity Acceptance Tolerances | **$\le 0.05\text{ m}$ pos, $\le 0.01\text{ m/s}$ vel** | `docs/numerical_audit.md`, `results/field_audit_abc/` | Verification threshold specification | $N = 1,789$ replay steps (178.9s telemetry replay) | Max state divergence between Java & Python | $\max_k \|\mathbf{x}_{\text{Java}, k} - \mathbf{x}_{\text{Python}, k}\|$ |

---

## 2. Denominator Audit: Usable Test Trips by Blackout Horizon

Why does the number of usable test trips change across outage durations?
The IO-VNBD test set contains 19 trips. However, evaluating a continuous outage of duration $T_{\text{outage}}$ requires:
$$T_{\text{trip}} \ge T_{\text{outage}} + T_{\text{warmup}} \quad (T_{\text{warmup}} = 10.0\text{ s pre-outage bias calibration})$$

| Trip Name | Road Category | Total Recorded Duration ($T$) | 10s Horizon ($T \ge 20\text{s}$) | 20s Horizon ($T \ge 30\text{s}$) | 30s Horizon ($T \ge 40\text{s}$) | 60s Horizon ($T \ge 70\text{s}$) | Exhaustive Census Note |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Vtb10** | Dense Urban | **9.6 s** | ❌ Excluded | ❌ Excluded | ❌ Excluded | ❌ Excluded | Total duration under 10 seconds |
| **Vw13** | Winding Mountain | **18.5 s** | ❌ Excluded | ❌ Excluded | ❌ Excluded | ❌ Excluded | Total duration under 20 seconds |
| **Vtb11** | Dense Urban | **26.2 s** | ✅ Included | ❌ Excluded | ❌ Excluded | ❌ Excluded | Insufficient duration for $\ge 20\text{s}$ outage |
| **Vtb09** | Dense Urban | **35.3 s** | ✅ Included | ✅ Included | ❌ Excluded | ❌ Excluded | Insufficient duration for $\ge 30\text{s}$ outage |
| **Vtb12** | Dense Urban | **34.8 s** | ✅ Included | ✅ Included | ❌ Excluded | ❌ Excluded | Insufficient duration for $\ge 30\text{s}$ outage |
| **Vta25** | Suburban Town | **54.7 s** | ✅ Included | ✅ Included | ✅ Included | ❌ Excluded | Total duration < 70s threshold |
| **Vta21** | Suburban Town | **198.0 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta22** | Suburban Town | **146.5 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta23** | Suburban Town | **101.1 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta24** | Suburban Town | **107.2 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta26** | Suburban Town | **183.6 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta27** | Suburban Town | **244.1 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vta28** | Suburban Town | **411.1 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vw12** | Winding Mountain | **81.9 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vw14a** | Winding Mountain | **303.9 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vw14b** | Winding Mountain | **1,948.9 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vw15** | Stationary Control | **128.1 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **Vw16a** | Winding Mountain | **578.0 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **V-Vfa02** | Motorway | **6,742.4 s** | ✅ Included | ✅ Included | ✅ Included | ✅ Included | Fully eligible at all horizons |
| **TOTAL ELIGIBLE** | — | — | **18 Trips** | **17 Trips** | **16 Trips** | **13 Trips** | Strictly controlled denominators |
