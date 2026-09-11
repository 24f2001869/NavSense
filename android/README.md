# NavSense: Android Deployment Engine & Sensor Telemetry System

## 1. Overview & Architectural Role

The `android/` directory contains the production-ready edge navigation prototype of **NavSense** (developed for **SIH26168**). It is designed to run locally on commercial Android smartphones without cloud connectivity, executing real-time dead reckoning during GNSS blackouts.

```
                  ┌─────────────────────────────────────┐
                  │    Android Hardware Sensor Stream   │
                  │  (Linear Accel, Gyro, Mag, Gravity) │
                  └──────────────────┬──────────────────┘
                                     │ 100 Hz Native
                                     ▼
                  ┌─────────────────────────────────────┐
                  │   Circular Resampling Buffer (10 Hz)│
                  │       (100 Steps = 10.0 s Window)   │
                  └──────────┬───────────────────────┬──┘
                             │                       │
                             ▼                       ▼
                  ┌──────────────────────┐  ┌──────────────────────┐
                  │  ONNX Runtime Engine │  │ 15-State Java ESKF   │
                  │  (Expanded TCN Model)│  │ (Kinematic Prop.     │
                  │  Forward Speed Est.  │  │  + Non-Holonomic C.) │
                  └──────────┬───────────┘  └──────────┬───────────┘
                             │                         │
                             └───────────┬─────────────┘
                                         ▼
                  ┌─────────────────────────────────────┐
                  │   GNSS Outage Finite State Machine  │
                  │  (NORMAL ──► OUTAGE ──► REACQUIRE)  │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │     Kotlin Real-Time Display UI     │
                  │   (Position, Compass Rose, Drift)   │
                  └─────────────────────────────────────┘
```

---

## 2. Core Engine Components

### 2.1 Sensor Subsystem (`service/SensorService.kt`)
* **Sampling Rate**: High-frequency FIFO (`SENSOR_DELAY_FASTEST`), resampled via monotonic timestamp interpolation to a strict **10.0 Hz** processing clock.
* **Channels**:
  - `TYPE_LINEAR_ACCELERATION` ($a_x, a_y, a_z$ without gravity)
  - `TYPE_GYROSCOPE` ($\omega_x, \omega_y, \omega_z$ in rad/s)
  - `TYPE_GRAVITY` ($g_x, g_y, g_z$ for attitude alignment)
  - `TYPE_ROTATION_VECTOR` (Android fused quaternion reference)
  - `LocationManager` (Fused / GPS location provider for pre-outage anchoring)

### 2.2 Neural Inference Pipeline (`ml/SpeedEstimator.java`)
* **Inference Engine**: Microsoft ONNX Runtime for Android (`onnxruntime-android`).
* **Active Model**: `tcn_kin_speed.onnx` (65,409 parameters, INT8/FP32 optimized).
* **Buffer Management**: `CircularFeatureBuffer.java` maintaining a rolling 100-step $\times$ 9-feature matrix.
* **Feature Normalization**: In-engine z-score scaling using training parameters loaded from `tcn_scaler.json`.

### 2.3 Classical State Estimation (`navigation/ESKF.java`)
* **15-State Error-State Kalman Filter**:
  - Position error ($\delta \mathbf{p} \in \mathbb{R}^3$)
  - Velocity error ($\delta \mathbf{v} \in \mathbb{R}^3$)
  - Attitude error ($\delta \boldsymbol{\theta} \in \mathbb{R}^3$)
  - Accelerometer bias ($\delta \mathbf{b}_a \in \mathbb{R}^3$)
  - Gyroscope bias ($\delta \mathbf{b}_g \in \mathbb{R}^3$)
* **Constraints**:
  - Non-Holonomic Constraints (NHC: zero lateral and vertical velocity in vehicle frame).
  - Zero Velocity Detection (ZVD: stationary state clamping).

---

## 3. GNSS Outage State Machine

The navigation engine operates under a deterministic 3-state finite state machine:
1. **NORMAL (GNSS Available)**:
   - ESKF receives 1 Hz GNSS position and Doppler velocity updates.
   - Accelerometer and gyroscope biases are continuously estimated and calibrated.
   - Vehicle heading is aligned to the GNSS trajectory course.
2. **OUTAGE (GNSS Denied / Simulated Blackout)**:
   - Triggered automatically by signal loss (tunnel/canyon) or manually via UI debug button.
   - ESKF propagates position using inertial strapdown mechanization + AI forward velocity.
   - Drift metrics ($\text{drift} / \text{distance}$) are computed in real time.
3. **REACQUISITION (GNSS Restored)**:
   - GNSS fix returns. Position is gently reconciled using smooth innovation fading to prevent velocity spikes.

---

## 4. Key Engineering Fixes Implemented

During Phase 5.1 forensics, several critical Android system-level defects were discovered and corrected:
1. **Monotonic Timing ($\Delta t$) Fix**: Replaced variable system clock calls with monotonic sensor event timestamps, eliminating integration jitter during thermal throttling.
2. **CPU Wake-Lock Integration**: Acquired `PARTIAL_WAKE_LOCK` and high-performance Wi-Fi lock to prevent Android OS deep sleep from throttling the 100 Hz sensor loop when the screen dims.
3. **GNSS Reacquisition Velocity Surge Protection**: Added innovation threshold gating on the first returning GNSS epoch to prevent post-outage velocity spikes caused by stale Doppler fixes.

---

## 5. Physical Testing Scope & Critical Clarification

> [!CAUTION]
> **Strict Qualification of Field Recordings**:  
> The smartphone recordings collected on campus (e.g. `walk_hostel_mess_203838`, `rooftop_gait_speed_analysis`) are **smartphone hardware, operating system, and sensor-pipeline stress tests only**.  
> **They are NOT vehicle navigation validation.**  
> Pedestrian walking motions introduce large human gait biomechanics, phone rotations, and swinging accelerations that violate ground-vehicle non-holonomic constraints (which caused the ~71 m/s out-of-distribution artifact documented in Phase 5.1).  
> **Physical vehicle validation in real road traffic remains a required open milestone.**

---

## 6. How to Build & Install

### Prerequisites
* Android Studio Iguana / Ladybug or command-line Android SDK.
* JDK 17 (recommended).
* Connected device with USB Debugging enabled.

### Build Commands
```bash
cd android

# Build Debug APK
./gradlew assembleDebug

# Install directly to connected device
./gradlew installDebug

# Start navigation service via ADB
adb shell am start -n com.sih26168.idr/.MainActivity
```
