# Master Results Index

This table provides a concise, authoritative lookup for every major quantitative experiment performed in this repository.

---

| Experiment ID | Scientific Focus | Dataset | Evaluated Test Set | Primary Metric | Primary Quantitative Result | Decision / Outcome |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 0** | CAN vs Wheel Speed Reference Audit | IO-VNBD | 64 trips | Phase latency & scale | CAN latency = **100 ms**; $r_{\text{eff}} = 0.312\text{ m}$ | 🟢 **Verified Reference** |
| **Phase 1** | Pure IMU Strapdown Baseline | IO-VNBD | 54 windows | 60s position drift % | Mean drift **>800% of distance** ($>650\text{ m}$) | 🔴 **Rejected as standalone** |
| **Phase 2** | 15-State ESKF Bias Calibration | IO-VNBD | 54 windows | Attitude & 10s drift | Attitude drift $<1.8^\circ$; 10s drift **6.8 m** ($4.2\times$ gain) | 🟢 **Accepted Core Filter** |
| **Phase 3** | Non-Holonomic Constraints (NHC) | IO-VNBD | 54 windows | Cross-track drift | Cross-track error reduced by **64.8%** | 🟢 **Accepted with Gating** |
| **Phase 4** | Dilated TCN (65k parameters) | IO-VNBD | 19 held-out trips | Speed MAE & 60s drift | MAE = **2.88 m/s**; 60s drift = **93.0 m (14.2%)** | 🟢 **Accepted Speed Model** |
| **Phase 5.1** | Pedestrian OOD Telemetry Forensics | Field Walk | 1 campus walk | Peak velocity spike | Gyro yaw reached **+26.08σ**; speed hit **71.66 m/s** | 🟢 **Input Gating Added** |
| **Phase 5.2** | Target-Balanced TCN Loss Weighting | IO-VNBD | 19 held-out trips | Motorway drift & MAE | Motorway drift **11.62%** (vs 11.22%); 5/19 pass | 🟡 **Pareto Trade-Off** |
| **Phase 5.3** | Spectral Vibration Speed Audit | IO-VNBD | 64 trips (>450k wins)| Pearson correlation $r$ | Correlation with speed: **r = -0.032**; MAE delta: **-0.01 m/s**| 🔴 **Rejected as Speedometer** |
| **Phase 5.4** | Steady-State Cruise Observability | IO-VNBD | Steady cruise windows | ROC-AUC separability | Separability of 85 vs 105 km/h: **ROC-AUC = 0.625** | 🔴 **Observability Bound** |
| **Phase 5.5** | Stateful Kinematics & Momentum | IO-VNBD | 14 usable trips | 60s drift (`Vw12` vs `Vta26`)| `Vw12`: **1.78% drift ✅**; `Vta26`: **574.6% drift ❌** | 🟡 **Regime Trade-Off** |
| **Phase 5.6** | Causal Adaptive Regime Fusion | IO-VNBD | 13 usable trips (60s) | 60s aggregate drift & pass rate | Mean drift = **16.76% (96.6 m)**; Pass rate = **3 / 13 (23.1%)** | 🟢 **Current Best System** |
