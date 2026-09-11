# Phase 3: 3D Strapdown, Attitude & Non-Holonomic Constraints (NHC)

## 1. Context & Motivation
Even with an ESKF, unconstrained 3D integration allows the vehicle state to drift in all spatial directions (e.g. drifting sideways through guardrails or floating vertically).  
However, wheeled ground vehicles operate under strict **physical kinematic constraints**: unless skidding or rolling over, a car cannot move laterally or vertically in its own body frame.

---

## 2. Non-Holonomic Constraints (NHC) Formulation

In the vehicle body frame $v = [x_{\text{forward}}, y_{\text{lateral}}, z_{\text{vertical}}]$:

$$v_y^v \approx 0 \quad (\text{Zero Lateral Velocity})$$
$$v_z^v \approx 0 \quad (\text{Zero Vertical Velocity})$$

### 2.1 Coordinate Frame Transformation
Let $\mathbf{R}_n^b$ be the attitude matrix from navigation frame to phone body frame, and $\mathbf{R}_b^v$ be the phone-to-vehicle mounting matrix:

$$\mathbf{v}^v = \mathbf{R}_b^v \mathbf{R}_n^b \mathbf{v}^n$$

The measurement residual is:

$$\mathbf{y}_{\text{NHC}} = \mathbf{0}_{2 \times 1} - \begin{bmatrix} \mathbf{v}_y^v \\ \mathbf{v}_z^v \end{bmatrix} = \mathbf{H}_{\text{NHC}} \delta \mathbf{x} + \boldsymbol{\nu}_{\text{NHC}}$$

where the measurement covariance is set to reflect vehicle tire slip: $\mathbf{R}_{\text{NHC}} = \operatorname{diag}(\sigma_{\text{lat}}^2, \sigma_{\text{vert}}^2)$ with $\sigma_{\text{lat}} = 0.3\text{ m/s}, \sigma_{\text{vert}} = 0.2\text{ m/s}$.

---

## 3. The Discovery of Gyro-Frame Cross-Coupling

During Stage C8 diagnostics, an important failure mode was uncovered:
* **The Phone-to-Vehicle Mounting Matrix $\mathbf{R}_b^v$**:
  If a phone is tilted slightly in a dashboard suction mount (e.g., $8^\circ$ pitch and $4^\circ$ roll relative to the vehicle chassis), applying NHC in the uncalibrated phone frame directly projects **forward longitudinal acceleration into the lateral constraint**!
* **Consequence**: The filter interpreted vehicle acceleration as lateral slip, inducing an erroneous heading bias correction and causing the estimated vehicle trajectory to swerve into a circle!

### Remediation (Causal Alignment Pipeline):
1. **Vertical Alignment**: Estimated directly from the static gravity vector $\mathbf{g}^b$ ($z$-axis).
2. **Forward Alignment**: Estimated causally via Principal Component Analysis (PCA) of horizontal acceleration during straight-line braking and acceleration events ($x$-axis).
3. **Cross-Check**: Cross-product yields lateral axis ($y$-axis).

---

## 4. Empirical Evaluation

Benchmarked on suburban trajectory windows:

| Navigation Metric | ESKF Without NHC | ESKF With Gated NHC | Net Improvement |
|:---|:---:|:---:|:---:|
| **Cross-Track Position Error (30 s)** | $68.4\text{ m}$ | **$26.1\text{ m}$** | **61.8% Reduction** |
| **Cross-Track Position Error (60 s)** | $165.8\text{ m}$ | **$58.3\text{ m}$** | **64.8% Reduction** |
| **Along-Track Position Error (60 s)** | $312.5\text{ m}$ | **$298.1\text{ m}$** | **<5% Change (Unconstrained!)** |
| **Vertical Height Divergence (60 s)** | $42.1\text{ m}$ | **$1.4\text{ m}$** | **96.7% Reduction** |

---

## 5. Conclusion & Decision

> [!IMPORTANT]
> **Core Finding**:  
> Non-Holonomic Constraints effectively eliminate **lateral skidding and vertical altitude divergence** (reducing cross-track error by >60%).  
> **However, NHC provides ZERO information about along-track forward speed.**  
> If the vehicle's forward speed is underestimated or overestimated, NHC simply constrains the wrong position along the road. The system requires an independent source of forward velocity.
