# System Architecture Specification

## 1. Architectural Principles

The SIH26168 navigation engine is designed around three core architectural tenets:
1. **Physical First, AI Assisted**: Classical physical mechanics (strapdown mechanization, coordinate frames, non-holonomic constraints, Kalman filtering) govern spatial propagation. Deep learning is applied strictly as an **auxiliary forward velocity measurement provider**.
2. **Causality & Zero Lookahead**: Every filter update, threshold check, rolling feature extraction, and neural prediction operates strictly on current and past observations ($t \le t_k$). Zero future lookahead is permitted.
3. **Graceful Degradation**: In the event of extreme sensor transients or out-of-distribution dynamics, the engine degrades gracefully to bounded kinematic momentum rather than diverging.

---

## 2. Five-Layer System Decomposition

```
[Layer 1: Sensor Ingestion]
  └── High-rate Android FIFO (100 Hz IMU, 1 Hz GNSS, Gravity, Mag)
  └── Monotonic hardware clock interpolation
       │
       ▼
[Layer 2: Preprocessing & Attitude Alignment]
  └── Dynamic gravity vector leveling (Roll/Pitch alignment)
  └── Principal Component Analysis (PCA) longitudinal acceleration alignment
  └── Input standardization (z-score normalization via tcn_scaler.json)
       │
       ▼
[Layer 3: Dual-Branch Velocity Engine]
  ├── Branch A: Dilated TCN (65k weights, 10.0 s causal context)
  ├── Branch B: Kinematic Propagator (GNSS initial velocity + integrated accel)
  └── Branch C: Multi-Condition Zero Velocity Detection (ZVD)
       │
       ▼
[Layer 4: State Estimation & Constraint Engine]
  ├── Causal Regime Fusion (Adaptive blending weight β_k)
  ├── 15-State Error-State Kalman Filter (ESKF)
  └── Non-Holonomic Constraints (NHC: zero lateral/vertical velocity)
       │
       ▼
[Layer 5: Edge Navigation UI & Telemetry]
  └── Real-time vector map rendering
  └── Heading compass rose & drift telemetry export
```

---

## 3. Detailed Data Flow

### 3.1 Pre-Outage Steady-State Operation (GNSS Available)
```text
GNSS Position (1 Hz) ────┐
GNSS Velocity (1 Hz) ────┼──► [ESKF Measurement Update]
IMU (100 Hz) ────────────┘         │
                                   ├──► Calibrate Accelerometer Bias (b_a)
                                   ├──► Calibrate Gyroscope Bias (b_g)
                                   ├──► Align Vehicle Heading (ψ)
                                   └──► Store Pre-Outage GNSS Speed (v_0)
```

### 3.2 Outage Autonomous Operation (GNSS Denied)
```text
IMU Stream (100 Hz) ──► 10 Hz Resampler ──► Rolling Buffer (10.0 s)
                                                 │
                               ┌─────────────────┴─────────────────┐
                               ▼                                   ▼
                      [Dilated TCN]                       [Kinematics + ZVD]
                     (v_TCN estimate)                    (v_kin = v_0 + ∫a dt)
                               │                                   │
                               └─────────────────┬─────────────────┘
                                                 ▼
                                     [Causal Adaptive Blending]
                                                 │
                                                 ▼
                                         Fused Forward Speed
                                                 │
                                                 ▼
                                     [15-State ESKF Engine]
                                      + Gated NHC (v_lat = 0)
                                                 │
                                                 ▼
                                      Dead-Reckoned Position
```
