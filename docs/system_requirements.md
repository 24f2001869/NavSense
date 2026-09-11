# System & Software Requirements Specification

## 1. Functional Requirements

| ID | Requirement Title | Detailed Specification | Status |
|:---|:---|:---|:---:|
| **FR-01** | **Multi-Sensor Ingestion** | Ingest tri-axial linear acceleration, calibrated gyroscope, magnetometer, gravity, and location updates at native hardware rates. | 🟢 Implemented |
| **FR-02** | **Monotonic Resampling** | Resample raw sensor FIFO queues onto a strict 10.0 Hz ($\Delta t = 0.10\text{ s}$) monotonic navigation clock. | 🟢 Implemented |
| **FR-03** | **Causal Mounting Alignment** | Align arbitrary smartphone orientation to the vehicle forward-lateral-vertical frame without manual user configuration. | 🟢 Implemented |
| **FR-04** | **Neural Velocity Estimation** | Run edge neural inference (Dilated TCN via ONNX Runtime) to predict vehicle forward speed every 100 ms over a 10.0 s rolling window. | 🟢 Implemented |
| **FR-05** | **15-State ESKF Tracking** | Maintain 3D position, velocity, attitude quaternion, and online accelerometer/gyroscope bias estimates in closed loop. | 🟢 Implemented |
| **FR-06** | **Outage State Machine** | Automatically detect GNSS loss, transition to autonomous dead reckoning, and smoothly reconcile state upon signal reacquisition. | 🟢 Implemented |
| **FR-07** | **Zero Velocity Detection** | Detect vehicle standstills causally and clamp forward velocity to zero to eliminate idle bias integration. | 🟢 Implemented |
| **FR-08** | **Real-Time Display UI** | Render vehicle position, heading compass rose, speed telemetry, and drift metrics on an interactive edge UI. | 🟢 Implemented |

---

## 2. Non-Functional Requirements (Edge Performance)

| ID | Metric | Target Specification | Measured Performance on Phone |
|:---|:---|:---:|:---:|
| **NFR-01** | **Inference Latency** | $<25.0\text{ ms}$ per 10.0 Hz cycle | **$6.2\text{--}8.4\text{ ms}$** (Snapdragon 7/8 series) |
| **NFR-02** | **Memory Allocation** | $<150\text{ MB}$ total heap RAM | **$68.4\text{ MB}$** steady-state heap |
| **NFR-03** | **CPU Utilization** | $<15\%$ average continuous load | **$7.2\text{--}11.5\%$** (single efficiency core) |
| **NFR-04** | **Power Consumption** | $<500\text{ mA}$ average battery drain | **$380\text{--}440\text{ mA}$** (screen on, full logging) |
| **NFR-05** | **Network Independence** | 100% offline; zero cloud network calls | **Fully Certified (Zero network calls)** |
