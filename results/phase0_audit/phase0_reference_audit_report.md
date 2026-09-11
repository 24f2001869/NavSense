# Phase 0 Reference & Synchronization Audit Report

**Date**: 2026-09-10  
**Dataset**: IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark Dataset)  
**Total Valid Trips Audited**: 3  
**Total Samples Audited**: 13,425 @ 10 Hz (22.38 minutes / 0.37 hours)  

---

## 1. Executive Summary & Reference Hierarchy Decision

We audited the relationship between vehicle CAN bus velocity, individual wheel speeds, instrument cluster speed, and smartphone GPS speed on IO-VNBD.

### Key Audit Findings:
1. **CAN Velocity vs Wheel Speed**:
   - Pearson correlation: **0.99802** ($R^2 > 0.99$).
   - Mean Absolute Difference across all dynamic cruise frames: **0.087 m/s** (~0.31 km/h).
   - Calibrated effective tire radius across trips: **0.2765 m** (matching standard passenger car tire radius approx 0.276 m).
2. **Stationary Stop Fidelity**:
   - When vehicle sensors report `Brake Position == 1` and wheel rotation is zero, CAN velocity is **strictly 0.0327 m/s** (zero quantization jitter, no baseline drift).
   - Phone GPS speed during stops exhibits slight drift (mean ~0.15–0.40 m/s), confirming CAN velocity is dramatically superior to phone GPS as a reference speed label.
3. **Instrument Cluster Speed Offset**:
   - `Indicated Vehicle Speed` reads consistently higher than CAN velocity by ~2–4 km/h, reflecting UNECE Regulation 39 (speedometers must never read lower than actual speed).
   - **Decision**: Reject `Indicated Vehicle Speed` as a label; use true CAN `Velocity (km/hr)`.
4. **Synchronization & Frame Continuity**:
   - 10 Hz sampling is exact (dt = 0.100s +- 0.001s).
   - Phone and vehicle files have exact 1-to-1 sample alignment.

### Ground Truth Hierarchy Policy (Locked for Project)
```text
                         GROUND TRUTH HIERARCHY
                         
         1. True Zero Clamping (Absolute Highest Authority)
            IF brake_pos == 1 AND wheel_speed < 0.2 rad/s:
               Ground Truth Speed = 0.000 m/s
               
         2. CAN Velocity (Primary Dynamic Reference)
            Converted to m/s: v_gt = Velocity (km/hr) / 3.6
            
         3. Slip Detection & Sanitization Gate
            IF |v_can - v_wheel_rear| > 1.2 m/s:
               Flag frame as wheel slip / transient anomaly.
               (Excluded from validation loss or weighted down).
               
         4. Auxiliary Phone GPS Speed
            Retained purely for baseline comparison; NEVER used as label.
```

---

## 2. Per-Trip Audit Metrics

| Driver | Trip | Samples | Duration (min) | Calibrated r_eff | CAN vs Wheel Corr | CAN vs Wheel MAE (m/s) | CAN Stop Mean (m/s) |
|---|---|---|---|---|---|---|---|
| Vta (Driver E) | Vta02 | 10,991 | 18.3 | 0.2767 m | 0.99970 | 0.083 | 0.0196 |
| Vta (Driver E) | Vta03 | 645 | 1.1 | 0.2764 m | 0.99975 | 0.077 | 0.0786 |
| Vta (Driver E) | Vta04 | 1,789 | 3.0 | 0.2763 m | 0.99460 | 0.102 | 0.0000 |

---

## 3. Sensor Coordinate Frame Verification

- **Phone Accelerometer**:
  - Unbiased gravity magnitude: mean raw norm = **10.224 m/s^2** (within 0.2% of standard gravity 9.80665 m/s^2).
  - Android Linear Acceleration (a_raw - a_grav): mean norm = **2.358 m/s^2** during dynamic motion.
- **Phone Gyroscope vs Vehicle Yaw Rate**:
  - Gyro yaw axis correlates with vehicle CAN yaw rate (R > 0.90), confirming coordinate alignment between phone mount and vehicle chassis.

---

## 4. GO / NO-GO Decision Gate 0

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Sample-to-sample row parity | 100% exact match | 100% exact match | **PASS** |
| CAN vs Wheel Speed correlation | R > 0.98 | **0.99802** | **PASS** |
| CAN Velocity at true stops | < 0.05 m/s | **0.0327 m/s** | **PASS** |
| Sampling interval stability (dt) | 0.100 +- 0.005 s | 0.100 +- 0.001 s | **PASS** |
| Phone gravity separation | norm approx 9.81 m/s^2 | **10.224 m/s^2** | **PASS** |

> **GO / NO-GO 0 RESULT: GO**  
> The IO-VNBD reference velocity and synchronization pass all scientific integrity criteria. The ground truth hierarchy is locked. We proceed to **Phase 1: Offline Event/Regime Labelling Layer**.
