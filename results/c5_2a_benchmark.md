# Stage C5.2-A: Approved Primary Controlled Experiment Benchmark Report
## Raw Phone-Frame vs. Sensor-Only Gravity-Leveled IMU Features

---

### Executive Summary
Stage C5.2-A evaluates whether resolving the smartphone IMU into a sensor-only gravity-leveled coordinate frame provides improved speed information compared to raw phone-frame features under a strict zero-leakage cross-trip evaluation:

* **Training Set**: `Vta02` (10,972 causal windows, ~18.3 min driving)
* **Testing Set**: `Vta04` (1,770 causal windows, ~3.0 min driving)
* **Window Size**: Causal $W = 20$ samples (2.0 s at 10 Hz), stride = 1 sample (0.1 s)
* **Target**: Scalar VBOX vehicle speed magnitude (`veh_speed_ms`)
* **Zero Leakage**: Strict trip separation. No VBOX heading, VBOX speed, GPS bearing, or vehicle heading was used to construct or orient the coordinate frames.
* **Controlled Scope**: Representation C (pre-calibrated vehicle-frame $R_{pv}$) was omitted from the primary benchmark because the existing $R_{pv}$ yaw estimation utilized VBOX speed-derived acceleration information.

---

### 1. Exact Transformation Equations

Representation B derives its orientation exclusively from the smartphone-derived static gravity vector $\mathbf{g}^p = [\bar{g}_x, \bar{g}_y, \bar{g}_z]^T$ obtained from the Android gravity sensor channel:

$$\phi = \arctan2(\bar{g}_y, \bar{g}_z) \quad (\text{Roll})$$
$$\theta = \arctan2(-\bar{g}_x, \sqrt{\bar{g}_y^2 + \bar{g}_z^2}) \quad (\text{Pitch})$$

$$\mathbf{R}_{\text{level}} = \mathbf{R}_y(\theta) \cdot \mathbf{R}_x(\phi) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}$$

The 3-axis accelerometer and gyroscope streams are transformed via:
$$\mathbf{a}_{\text{level}} = \mathbf{R}_{\text{level}} \cdot \mathbf{a}_{\text{phone}}, \quad \boldsymbol{\omega}_{\text{level}} = \mathbf{R}_{\text{level}} \cdot \boldsymbol{\omega}_{\text{phone}}$$

> **Physical Interpretation**: Gravity leveling removes the estimated static roll/pitch orientation component and reduces sensitivity to phone tilt. It does not guarantee complete separation of gravity from dynamic specific-force disturbances during acceleration, braking, cornering, or rough-road events.

---

### 2. Feature Construction & Mathematical Invariance

Both representations extract identical 8 signal streams with identical 9 causal window statistics ($8 \times 9 = 72$ features):

| Stream # | Representation A (Raw Phone) | Representation B (Gravity-Leveled) | Physical Description |
|:---:|---|---|---|
| 1 | `acc_x` | `accel_x_level` | Horizontal channel X |
| 2 | `acc_y` | `accel_y_level` | Horizontal channel Y |
| 3 | `acc_z` | `accel_z_level` | Vertical channel Z (upward, ~+9.8 m/s² static) |
| 4 | `acc_mag` | `accel_horizontal_magnitude` | Leveled horizontal plane norm: $\sqrt{a_{x,\text{level}}^2 + a_{y,\text{level}}^2}$ |
| 5 | `gyro_x` | `gyro_x_level` | Horizontal gyro X |
| 6 | `gyro_y` | `gyro_y_level` | Horizontal gyro Y |
| 7 | `gyro_z` | `gyro_z_level` | Vertical yaw rate Z |
| 8 | `gyro_mag` | `gyro_level_magnitude` | 3D angular rate magnitude: $\|\boldsymbol{\omega}\|$ |

#### Invariance & Structural Integrity Verification:
1. **Yaw Invariance**: In Representation B, `accel_horizontal_magnitude` ($\sqrt{a_{x,\text{level}}^2 + a_{y,\text{level}}^2}$) is mathematically invariant to horizontal phone mounting yaw:
   $$\max |\Delta| = 4.44 \times 10^{-16} \text{ (machine precision)}$$
2. **Dimension & Feature Equivalence**: Both representations produce exactly $(N, 72)$ matrices with identical feature ordering between `Vta02` and `Vta04`.
3. **Leakage Audit**: Verified that zero VBOX/GNSS information enters Representation B.

---

### 3. Primary Benchmark Results (Full Test Trip Vta04, 1,770 Windows)

| Representation | Model | MAE (m/s) | RMSE (m/s) | $R^2$ | Bias (m/s) |
|---|---|---:|---:|---:|---:|
| **Baseline Mean** | Constant Prior | 2.07 | 2.55 | -0.232 | -1.11 |
| **Phone Frame (Rep A)** | Random Forest | 2.24 | 2.84 | -0.531 | +0.92 |
| **Gravity-Leveled (Rep B)** | Random Forest | 2.22 | 2.80 | -0.488 | +0.91 |
| **Phone Frame (Rep A)** | Gradient Boosting | 2.10 | 2.61 | -0.296 | +0.78 |
| **Gravity-Leveled (Rep B)** | Gradient Boosting | 2.15 | 2.72 | -0.409 | +0.85 |

#### Performance Deltas ($\Delta = \text{Rep B} - \text{Rep A}$):
* **Random Forest**: $\Delta\text{MAE} = \mathbf{-0.03\text{ m/s}}$ ($-1.1\%$), $\Delta\text{RMSE} = -0.04\text{ m/s}$, $\Delta R^2 = +0.043$
* **Gradient Boosting**: $\Delta\text{MAE} = \mathbf{+0.05\text{ m/s}}$ ($+2.5\%$), $\Delta\text{RMSE} = +0.11\text{ m/s}$, $\Delta R^2 = -0.113$

---

### 4. Rough-Road Diagnostic Breakdown ($t = 29.6\text{--}33.2\text{ s}$, 36 Windows)

Evaluating the known rough-road / suspension excitation interval on `Vta04`:

| Model | Phone MAE (m/s) | Leveled MAE (m/s) | $\Delta\text{MAE}$ (m/s) | Phone RMSE (m/s) | Leveled RMSE (m/s) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Mean** | 1.23 | 1.23 | 0.00 | 1.24 | 1.24 |
| **Random Forest** | 3.18 | 3.54 | **+0.36** | 3.51 | 3.74 |
| **Gradient Boosting** | 2.98 | 3.39 | **+0.41** | 3.23 | 3.59 |

**Observation**: During rough-road excitation, both models exhibit significant over-prediction (+3.0 to +3.5 m/s positive bias). Gravity leveling slightly worsens the error (+0.36 to +0.41 m/s higher MAE). During rough-road excitation, dynamic suspension/attitude motion and non-kinematic specific-force components can contaminate the leveled horizontal acceleration features; static gravity leveling cannot remove these time-varying disturbances.

---

### 5. Scientific Interpretation & Locked Project Conclusion

1. **Equivalence of Phone-Frame and Gravity-Leveled Representations**:
   Gravity leveling alone yields negligible change across the test journey ($\Delta\text{MAE} \in [-0.03, +0.05]\text{ m/s}$). All models remain worse than the trivial training-set mean baseline ($2.07\text{ m/s}$ MAE, $R^2 = -0.232$).
2. **Why Leveling Does Not Solve Cross-Trip Failure**:
   - The estimated static roll/pitch offsets were small relative to the large horizontal mounting-yaw differences observed between trips; therefore static roll/pitch correction alone was unlikely to explain the dominant cross-trip discrepancy.
   - High-frequency chassis vibration and road roughness dominate the IMU acceleration channels (Kinematic $\text{SNR} = -3.7\text{ dB}$ to $-4.7\text{ dB}$, as proven in Part 1).
   - The results are consistent with the hypothesis that the current statistical features are strongly influenced by road/suspension disturbance characteristics and therefore suffer from cross-trip domain dependence.
   - Consequently, removing static tilt does not alleviate the fundamental domain shift between road surfaces.

---

### 6. Locked C5.2-A Project Conclusion
> *"Under a strict Vta02→Vta04 cross-trip evaluation, sensor-only static gravity leveling produced no meaningful improvement in IMU-based vehicle-speed estimation. Random Forest MAE changed from 2.24 to 2.22 m/s, while Gradient Boosting changed from 2.10 to 2.15 m/s; all models remained inferior to the constant training-set mean baseline. During the identified rough-road interval, leveling further increased model error. These results indicate that static phone roll/pitch orientation is not the dominant factor limiting the current cross-trip estimator. The findings are instead consistent with substantial dependence on dynamic specific-force disturbances and trip-dependent sensor/road characteristics, motivating evaluation of broader training-domain diversity."*

---

### 7. Next Controlled Experiment: Stage C5.2-B

> **Stage C5.2-B: Multi-Trip Training Evaluation**
> * **Primary Comparison**: $Vta02 \to Vta04$ (single-trip train) vs. $Vta02 + Vta03 \to Vta04$ (multi-trip train).
> * **Domain Distribution Audit**: Explicitly compute and report individual feature and kinematics distributions $P(Vta02), P(Vta03), P(Vta04)$ to determine whether $Vta03$ bridges the domain gap to $Vta04$.
> * **Strict Control**: Feature extractor, causal windowing ($W=20$, stride=1), models (RF and GBDT), and test target remain strictly frozen.
