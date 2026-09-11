# Literature-to-Implementation Specification

## Objective

This document formalizes the exact signals, coordinate frames, sampling rates, feature representations, architectures, loss functions, and evaluation metrics from peer-reviewed literature on smartphone-based vehicular dead reckoning. It establishes the benchmark specifications against which we evaluate models on the IO-VNBD dataset.

---

## 1. Literature Survey & Method Specifications

### Method 1: DVSE — Deep Vehicle Speed Estimation (arXiv:2505.18490, 2025)
- **Paper Title**: *Deep Vehicle Speed Estimation Using Smartphone Inertial Sensors with High Robustness*
- **Problem Formulation**: Estimates vehicle speed changes ($\Delta v_t$) to prevent error accumulation from direct acceleration double integration.
- **Input Channels**: 6 IMU axes:
  - 3-axis Accelerometer (specific force / linear acceleration)
  - 3-axis Gyroscope (angular velocity)
- **Sampling & Windowing**:
  - Original paper: 50 Hz IMU, 1.0-second sliding windows.
  - Adapted for IO-VNBD: 10 Hz IMU, 2.0-second sliding windows (20 samples) with 0.5-second stride (5 samples).
- **Network Architecture (Noise Compensation Network - NCN)**:
  - Input Linear Projection: $d_{in} \to 64$
  - 2-layer Gated Recurrent Unit (GRU): hidden dimension 64 or 128, dropout 0.1
  - Multi-Layer Perceptron (MLP) head: $128 \to 64 \to 1$
- **Loss Function**:
  $$\mathcal{L}_{DVSE} = \text{SmoothL1}(\Delta v_{pred}, \Delta v_{ref}) + \alpha \cdot \text{SmoothL1}(v_{pred}, v_{ref})$$
  where $\Delta v = v_t - v_{t-1}$ and $\alpha \in [0.1, 0.5]$.
- **Reported Benchmark Performance**:
  - Velocity MAE: $\approx 2.26\text{ m/s}$ on complex driving routes without GPS.
  - Significantly outperforms naive acceleration integration over 30s and 60s horizons.

---

### Method 2: AVNet — Learning Attitude and Velocity with InEKF (2025)
- **Paper Title**: *AVNet: Learning Attitude and Velocity for Vehicular Dead Reckoning with Smartphone Inertial Sensors*
- **Problem Formulation**: Jointly learns forward vehicle speed AND attitude error dynamics, fused via an Invariant Extended Kalman Filter (InEKF).
- **Input Channels**:
  - Raw 6-axis IMU sequences (time-series window of length $T = 100$ samples @ 10 Hz = 10.0 seconds).
- **Network Architecture**:
  - **Feature Extractor**: 3-layer 1D Temporal CNN (kernel sizes [5, 3, 3], channels [32, 64, 128], ReLU, BatchNorm).
  - **Sequence Modeling**: Bidirectional GRU (hidden dim 128).
  - **Dual Heads**:
    1. *Velocity Head*: Predicts scalar forward velocity $v_{lon} \in \mathbb{R}^1$ and measurement variance $\sigma_v^2$.
    2. *Attitude Head*: Predicts relative rotation increment / attitude correction quaternion $\Delta q \in \mathbb{H}$ and variance $\Sigma_q$.
- **Downstream Fusion**:
  - Invariant EKF on matrix Lie group $SE_2(3)$: updates velocity and position states using learned forward velocity as a virtual odometer, while enforcing Non-Holonomic Constraints (NHC: $v_{lat} \approx 0, v_{vert} \approx 0$).
- **Reported Benchmark Performance**:
  - Velocity RMSE: $0.38\text{ m/s}$.
  - Position Drift: $< 1.5\%$ of total distance travelled.

---

### Method 3: Dilated Causal 1D-CNN (TCN) for Speed Estimation (2025)
- **Paper Title**: *Vehicle Forward Velocity Estimation Using Dilated CNN and Smartphone IMU*
- **Problem Formulation**: Eliminates recurrent bottleneck and gradient vanishing by using dilated temporal convolutions to achieve large receptive fields (e.g., 10 seconds of vehicle dynamics history).
- **Architecture**:
  - Dilated residual causal blocks with dilation rates $d \in \{1, 2, 4, 8, 16\}$.
  - Kernel size $k = 3$, residual connections, weight normalization.
  - Receptive field: $> 64$ samples ($> 6.4$ seconds at 10 Hz).
- **Loss Function**:
  $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \text{Huber}(v_{pred,i} - v_{ref,i}, \delta=1.0)$$
- **Advantage**: Fast non-recurrent inference, deterministic latency, suitable for mobile on-device execution.

---

## 2. IO-VNBD Dataset Specification

### Source & Structure
- **Dataset Title**: *IO-VNBD: An Indoor-Outdoor Vehicular Navigation Benchmark Dataset*
- **Location**: `data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/`
- **Driver / Setup Categories**:
  1. `Vta (Driver E)` — Urban / suburban route, phone mounted on dashboard
  2. `Vtb (Driver E)` — Secondary route, phone mounted on dashboard
  3. `Vf (Driver E)` — Highway / rural route
  4. `Vw (Driver E)` — Mixed condition route
  5. `M (Driver B)` — Driver B trajectory
  6. `S (Driver A)` — Driver A trajectory
  7. `Y (Driver D)` — Driver D trajectory

### Signal Inventory

#### Smartphone CSV (`S-*.csv`, 10 Hz)
| Column | Units | Description | Project Role |
|--------|-------|-------------|--------------|
| `TIME SINCE START (ms)` | ms | Monotonic timer | Timestamp sync |
| `ACCELEROMETER X/Y/Z` | $\text{m/s}^2$ | Raw specific force (includes gravity) | Primary input |
| `GRAVITY X/Y/Z` | $\text{m/s}^2$ | Android gravity estimate | Coordinate alignment |
| `GYROSCOPE Yaw/Pitch/Roll` | $\text{rad/s}$ | Angular velocity | Primary input |
| `MAGNETIC FIELD X/Y/Z` | $\mu\text{T}$ | Magnetometer field | Heading check |
| `ORIENTATION Yaw/Pitch/Roll` | deg | Android orientation | Coordinate audit |
| `GPS SPEED (Kmh)` | km/h | Smartphone GPS speed | Baseline comparison |
| `GPS ACCURACY (m)` | m | Horizontal dilution / accuracy | GNSS reliability check |

#### Vehicle CSV (`V-*.csv`, 10 Hz)
| Column | Units | Description | Project Role |
|--------|-------|-------------|--------------|
| `Time Since Start of Day (seconds)` | s | Monotonic CAN timer | Timestamp sync |
| `Velocity (km/hr)` | km/h | CAN bus forward speed | Reference candidate |
| `Wheel Speed FL/FR/RL/RR` | rad/s | 4-wheel angular rates | Reference audit & slip check |
| `Indicated Vehicle Speed (km/hr)` | km/h | Dashboard cluster speed | Cluster offset audit |
| `Yaw Rate (deg/sec)` | deg/s | CAN vehicle yaw rate | Reference regime labeling |
| `Indicated Longitudinal Accel` | g | CAN vehicle forward accel | Reference regime labeling |
| `Indicated Lateral Accel` | g | CAN vehicle lateral accel | Turning regime labeling |
| `Steering Angle` | deg | Steering wheel angle | Turning regime labeling |
| `Brake Pressure (psi)`, `Brake Position` | psi, bool | Braking system sensors | Braking / Stop regime |
| `Handbrake` | bool | 0 or 1 | Stationary ground truth |
| `Engine Speed (rev/min)` | rpm | Engine RPM | Idling / Stop confirmation |

---

## 3. Data Representation Architecture

```text
                               Raw Smartphone IMU @ 10 Hz
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ↓                                               ↓
         Branch A: Engineered Features                   Branch B: Raw Sequence
         (Window = 2.0s = 20 samples)                    (Window = 10.0s = 100 samples)
                    │                                               │
         - Mean, std, RMS, min, max                      - Shape: (B, 100, 6)
         - Jerk (da/dt), energy                          - [a_lin_x, a_lin_y, a_lin_z,
         - Norm magnitudes (|a|, |w|)                       w_x, w_y, w_z]
         - Feature dim: 56 floats                        - Preserves transient waveforms
                    │                                               │
                    ↓                                               ↓
          Ridge / Random Forest                              GRU / TCN / AVNet
```

---

## 4. Ground Truth Hierarchy & Audit Rules

Before any model is trained, the reference hierarchy is verified:
1. **Stationary State**: If `Brake Position == 1` OR `Handbrake == 1` AND wheel speed $< 0.2\text{ rad/s}$, ground truth speed is clamped to **exactly $0.0\text{ m/s}$**.
2. **Dynamic State**:
   - Compare CAN `Velocity` against rear non-driven wheel speed ($v_{wheel} = \frac{\omega_{RL} + \omega_{RR}}{2} \cdot r_{eff}$).
   - If $|v_{can} - v_{wheel}| > 1.5\text{ m/s}$, flag frame as possible wheel-slip or sensor fault.
   - If GNSS speed is available with high satellite count and low DOP, verify consistency.
3. **Reference Hierarchy**:
   - Primary: Audited CAN Velocity (converted to $\text{m/s}$).
   - Secondary: Wheel speed average (for slip detection & validation).
   - Auxiliary: Vehicle GNSS position and phone GNSS speed (used strictly for baseline comparisons and outage simulation).

---

## 5. Evaluation Protocol

### Two Distinct Evaluation Splits
1. **Split A (Unseen Trips)**:
   - Within driver category `Vta`, `Vtb`, `Vf`, `Vw`:
   - 70% trajectories for training, 10% for validation, 20% untouched trajectories for testing.
   - Measures: Route and trajectory generalization under identical vehicle/phone mounting conditions.
2. **Split B (Unseen Driver / Condition)**:
   - Train on Drivers A, B, D (`S`, `M`, `Y`).
   - Evaluate on Driver E (`Vta`, `Vtb`, `Vf`, `Vw`).
   - Measures: Out-of-distribution transfer across drivers, vehicles, and sensor mountings.

### Evaluation Metrics
- **Instantaneous Speed Metrics**:
  - Mean Absolute Error (MAE): $\frac{1}{N}\sum |v_{pred} - v_{ref}|$ [m/s]
  - Root Mean Squared Error (RMSE): $\sqrt{\frac{1}{N}\sum (v_{pred} - v_{ref})^2}$ [m/s]
  - 80th-Percentile Error: $P_{80}$ of absolute residual distribution [m/s]
- **Integrated Dead-Reckoning Horizons**:
  - 30-second Cumulative Distance Drift: $|\int_0^{30} (v_{pred} - v_{ref}) dt|$ [m]
  - 60-second Cumulative Distance Drift: $|\int_0^{60} (v_{pred} - v_{ref}) dt|$ [m]
  - Relative Position Error: $\frac{\text{Drift}}{\text{Distance Traveled}} \times 100\%$
- **Regime-Segmented MAE**:
  - MAE reported separately across each regime: `CRUISE`, `ACCEL`, `BRAKING`, `TURN`, `STOP`, `BUMP`, `ROUGH_ROAD`, `LOW_SPEED`, `HIGH_SPEED`.
