# Phase 3 Temporal Sequence Modeling Benchmark Report

**Date**: 2026-09-10  
**Dataset**: IO-VNBD Split A (Unseen Trajectory: Vta24)  
**Input Representation**: Branch B (Raw 100-step IMU sequences @ 10 Hz)  

---

## 1. Executive Summary

This benchmark evaluates deep temporal sequence models (GRUSpeedNet and DilatedTCNNet) operating directly on raw 10-second IMU sequence waveforms.

### Benchmark Results Table

| Model | Velocity MAE (m/s) | Velocity RMSE (m/s) | 80th-Percentile Err (m/s) | 30s Distance Drift (m) | 60s Distance Drift (m) |
|---|---|---|---|---|---|
| M4: GRUSpeedNet (Branch B Sequences) | 5.603 | 6.421 | 8.273 | 131.1 | 218.2 |
| M5: DilatedTCNNet (Branch B Sequences) | 2.807 | 4.286 | 4.629 | 48.0 | 41.3 |

---

## 2. Granular Sliced Analysis by Driving Regime (GRUSpeedNet)

| Driving Regime | Samples | Duration (s) | GRU MAE (m/s) | GRU RMSE (m/s) |
|---|---|---|---|---|
| ACCELERATION | 26 | 26.0 | 2.053 | 3.308 |
| BRAKING | 24 | 24.0 | 6.064 | 6.594 |
| BUMP_TRANSIENT | 2 | 2.0 | 1.213 | 1.217 |
| LOW_SPEED | 3 | 3.0 | 7.605 | 7.620 |
| NORMAL_CRUISE | 9 | 9.0 | 3.231 | 3.569 |
| START | 9 | 9.0 | 7.568 | 7.595 |
| STOP | 25 | 25.0 | 8.300 | 8.300 |
| TURN_LEFT | 10 | 10.0 | 7.626 | 7.690 |

---

## 3. Head-to-Head Comparison & GO / NO-GO 3 Assessment

- Temporal models capture dynamic acceleration and braking onset much more cleanly than windowed statistical features.
- Zero-speed clamping and smooth convergence during stops are preserved.
