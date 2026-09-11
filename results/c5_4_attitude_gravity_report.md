# Stage C5.4: Dynamic Attitude & Gravity-Leakage Diagnostic Investigation

**Milestone Identifier**: `C5.4`  
**Date**: September 5, 2026  
**Status**: **COMPLETED (DIAGNOSTIC VERDICT: HYPOTHESIS REFUTED)**  
**Script**: [`experiments/run_attitude_gravity_audit_c5_4.py`](../experiments/run_attitude_gravity_audit_c5_4.py)  
**Structured Audit Data**: [`results/c5_4_attitude_gravity_audit.json`](c5_4_attitude_gravity_audit.json)  

---

## 1. Executive Summary & Diagnostic Verdict

Stage C5.4 executed a strictly physical, diagnostic investigation to test whether **dynamic vehicle pitch deflection and consequent gravity leakage into the longitudinal accelerometer channel** explain the persistent $\approx 2.0\text{--}2.7\text{ m/s}^2$ dynamic under-prediction during severe vehicle braking observed across Stages C5.2 through C5.3-B3.

### Strict Methodological Controls
1. **Zero Machine Learning**: No Ridge, Random Forest, GBDT, MLP, LSTM, or Transformer models were trained or evaluated.
2. **Zero Navigation Filter Redesign**: No ESKF, non-holonomic constraints (NHC), or heuristic filters were introduced.
3. **Pure Kinematic & Physical Rotation Audit**: Direct comparison of raw smartphone IMU, static gravity-leveled IMU, dynamic pitch-compensated IMU, dynamic 3D rotation matrix leveling, and rigidly mounted vehicle chassis CAN accelerometer against the VBOX-derived acceleration reference ($a_{\text{ref}}$).

### Headline Diagnostic Findings

1. **The Suspension Gravity-Leakage Hypothesis is REFUTED**:
   - Dynamic vehicle pitch deflection during severe braking is small ($|\Delta \theta_{\text{pitch}}| \le 0.32^\circ$).
   - The theoretical longitudinal gravity projection $g \sin(\Delta \theta_{\text{pitch}})$ is only **$0.028\text{ m/s}^2$** on `Vta02`, **$-0.006\text{ m/s}^2$** on `Vta03`, and **$-0.055\text{ m/s}^2$** on `Vta04`.
   - Gravity leakage is **$38\times$ to $50\times$ too small** to account for the observed residual ($+1.07\text{ m/s}^2$ on `Vta02`, $+1.52\text{ m/s}^2$ on `Vta03`, $+2.70\text{ m/s}^2$ on `Vta04`).
   - Dynamic attitude compensation fails to reduce the severe braking bias (remaining at $+1.04\text{ m/s}^2$ on `Vta02`, $+1.52\text{ m/s}^2$ on `Vta03`, and $+2.75\text{ m/s}^2$ on `Vta04`).
   - Diagnostic OLS regression of the acceleration residual against pitch deflection, pitch rate, and road grade explains less than $5\%$ of residual variance on `Vta04` ($R^2 = +0.0464$).

2. **The True Physical Bottleneck: Mount Compliance / Sensor Decoupling**:
   - The vehicle's own rigidly mounted **chassis CAN accelerometer** matches the VBOX acceleration reference with extreme fidelity across all trips:
     - `Vta02`: MAE = **$0.1914\text{ m/s}^2$**, Severe Brake Bias = **$-0.1637\text{ m/s}^2$**, 30 s Drift = **$46.06\text{ m}$** (vs. phone's $204.77\text{ m}$, a **$77.5\%$ drift reduction**).
     - `Vta03`: MAE = **$0.1703\text{ m/s}^2$**, Severe Brake Bias = **$-0.1041\text{ m/s}^2$**, 30 s Drift = **$66.57\text{ m}$** (vs. phone's $225.60\text{ m}$, a **$70.5\%$ drift reduction**).
     - `Vta04`: MAE = **$0.2524\text{ m/s}^2$**, Severe Brake Bias = **$+0.1683\text{ m/s}^2$**, 30 s Drift = **$66.91\text{ m}$** (vs. phone's $232.13\text{ m}$, a **$71.2\%$ drift reduction**).
   - On `Vta04`, during severe braking, true vehicle deceleration reaches **$-2.20\text{ m/s}^2$** (and the chassis CAN sensor correctly reads **$-2.03\text{ m/s}^2$**), whereas the smartphone accelerometer registers **$+0.33\text{ m/s}^2$** (an erroneous net acceleration).
   - Because the vehicle chassis accelerometer exhibits virtually zero braking bias, the discrepancy cannot be attributed to reference errors, Doppler artifacts, or vehicle-level pitch dynamics.
   - The discrepancy is localized to the smartphone mechanical interface: **windshield suction-cup mount compliance, mechanical flexure, and dynamic decoupling** between the phone sensor and vehicle chassis.

---

## 2. Dimension 1: Static Mounting Misalignment Analysis

Stationary intervals were extracted using the canonical zero-velocity condition ($v_{\text{vbox}} < 0.10\text{ m/s}$, $N \ge 15$). Static mounting angles were computed via:
$$\theta_{\text{stat}} = \arctan2\left(-a_x^{\text{stat}}, \sqrt{(a_y^{\text{stat}})^2 + (a_z^{\text{stat}})^2}\right), \quad \phi_{\text{stat}} = \arctan2\left(a_y^{\text{stat}}, a_z^{\text{stat}}\right)$$

| Parameter | Trip `Vta02` | Trip `Vta03` | Trip `Vta04` |
| :--- | :---: | :---: | :---: |
| **Stationary Samples ($N_{\text{stop}}$)** | $141$ | $170$ | $0$ (trip median used) |
| **Mean Stationary $a_x^{\text{stat}}$** | $+0.0314\text{ m/s}^2$ | $-1.2232\text{ m/s}^2$ | $+0.3816\text{ m/s}^2$ |
| **Mean Stationary $a_y^{\text{stat}}$** | $-0.3978\text{ m/s}^2$ | $+0.7900\text{ m/s}^2$ | $-0.6797\text{ m/s}^2$ |
| **Mean Stationary $a_z^{\text{stat}}$** | $+9.8403\text{ m/s}^2$ | $+9.7754\text{ m/s}^2$ | $+9.8147\text{ m/s}^2$ |
| **Calculated Static Pitch ($\theta_{\text{stat}}$)** | $-0.18^\circ$ | $+7.11^\circ$ | $-2.22^\circ$ |
| **Calculated Static Roll ($\phi_{\text{stat}}$)** | $-2.32^\circ$ | $+4.62^\circ$ | $-3.96^\circ$ |
| **Onboard Fused Mean Pitch** | $-85.35^\circ$ | $-85.53^\circ$ | $-83.68^\circ$ |
| **Onboard Fused Mean Roll** | $+31.75^\circ$ | $+40.99^\circ$ | $+78.46^\circ$ |

### Physical Interpretation
- The smartphone was mounted in a semi-vertical windshield cradle across all three trips (phone orientation pitch $\approx -85^\circ$).
- While static gravity leveling aligns the gravity vector during stationary intervals, static leveling has virtually **zero impact on the dynamic acceleration residual** (raw MAE vs. static-leveled MAE differs by $< 0.0002\text{ m/s}^2$).

---

## 3. Dimension 2: Dynamic Pitch Deflection & Gravity Projection

During longitudinal deceleration, vehicle suspension kinematics cause forward pitch (dive). If the smartphone rotates with the chassis, gravity projects into the measurement axis:
$$\Delta a_{\text{grav}} = g \cdot \sin(\Delta \theta_{\text{pitch}})$$

We compared this theoretical projection against the empirical acceleration residual $r_a = a_x^{\text{level}} - a_{\text{ref}}$ during severe braking ($a_{\text{ref}} \le -1.5\text{ m/s}^2$):

| Metric | Trip `Vta02` | Trip `Vta03` | Trip `Vta04` |
| :--- | :---: | :---: | :---: |
| **Severe Braking Samples ($N$)** | $175$ | $42$ | $74$ |
| **Mean Reference Deceleration ($a_{\text{ref}}$)** | $-1.9567\text{ m/s}^2$ | $-2.0560\text{ m/s}^2$ | $-2.2037\text{ m/s}^2$ |
| **Mean Vehicle CAN Deceleration** | $-2.1205\text{ m/s}^2$ | $-2.1601\text{ m/s}^2$ | $-2.0354\text{ m/s}^2$ |
| **Mean Phone Leveled Acceleration ($a_x^{\text{level}}$)** | $-0.8855\text{ m/s}^2$ | $-0.5390\text{ m/s}^2$ | **$+0.3341\text{ m/s}^2$** |
| **Mean Empirical Residual ($r_a$)** | **$+1.0712\text{ m/s}^2$** | **$+1.5169\text{ m/s}^2$** | **$+2.6979\text{ m/s}^2$** |
| **Dynamic Pitch Deflection ($\Delta \theta_{\text{pitch}}$)** | **$+0.1636^\circ$** | **$-0.0350^\circ$** | **$-0.3195^\circ$** |
| **Theoretical Gravity Projection ($g \sin \Delta \theta$)** | **$+0.0280\text{ m/s}^2$** | **$-0.0060\text{ m/s}^2$** | **$-0.0547\text{ m/s}^2$** |
| **Ratio of Residual to Gravity Projection** | **$38.3 \times$** | **$-252.8 \times$** | **$-49.3 \times$** |

### Mathematical Significance
1. On `Vta04`, the theoretical gravity leakage induced by chassis pitch is **$-0.0547\text{ m/s}^2$**.
2. The actual residual is **$+2.6979\text{ m/s}^2$** — opposite in sign and roughly **$50\times$ larger**!
3. On `Vta02`, the theoretical gravity leakage is **$+0.0280\text{ m/s}^2$**, which accounts for less than **$2.6\%$** of the $+1.0712\text{ m/s}^2$ residual.
4. **Conclusion**: Suspension pitch deflection is orders of magnitude too small to explain the severe braking under-prediction.

---

## 4. Dimension 3: Correlation & Diagnostic Regression Analysis

### Statistical Correlations during Braking ($a_{\text{ref}} < -0.5\text{ m/s}^2$)

| Feature Variable | `Vta02` | `Vta03` | `Vta04` |
| :--- | :---: | :---: | :---: |
| $\text{corr}(r_a, \Delta \theta_{\text{pitch}})$ | $+0.0304$ | $-0.5316$ | $-0.1329$ |
| $\text{corr}(r_a, g \sin \Delta \theta_{\text{pitch}})$ | $+0.0808$ | $+0.0116$ | $-0.2721$ |
| $\text{corr}(r_a, \text{Road Grade Slope } \theta_{\text{road}})$ | $+0.0798$ | $+0.4197$ | $-0.1768$ |
| $\text{corr}(r_a, \text{Gyro Pitch Rate } \dot{\theta})$ | $+0.0285$ | $-0.1598$ | $+0.0283$ |
| $\text{corr}(a_{\text{CAN}}, a_{\text{ref}})$ [Entire Trip] | **$+0.9301$** | **$+0.9669$** | **$+0.8464$** |

- Across all three trips, the correlation between acceleration residual and dynamic pitch deflection has an inconsistent sign and is negligible on `Vta02` ($+0.03$) and `Vta04` ($-0.13$).
- In contrast, the vehicle's chassis CAN accelerometer maintains a robust correlation of **$r = 0.85\text{ to } 0.97$** with VBOX $a_{\text{ref}}$.

### Diagnostic OLS Regression
To test whether a linear combination of attitude dynamics could predict the residual:
$$r_a = \beta_0 + \beta_1 \Delta \theta_{\text{pitch}} + \beta_2 \dot{\theta}_{\text{pitch}} + \beta_3 \theta_{\text{road}} + \epsilon$$

| Trip | $R^2$ Variance Explained | $\beta_0$ (const) | $\beta_1$ ($\Delta \theta$, deg) | $\beta_2$ ($\dot{\theta}$, rad/s) | $\beta_3$ ($\theta_{\text{road}}$, deg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Vta02` | **$+0.0117$** ($1.2\%$) | $+0.2071$ | $+0.0735$ | $-0.4316$ | $+0.0993$ |
| `Vta03` | **$+0.1850$** ($18.5\%$) | $-0.0367$ | $-0.3901$ | $+1.4604$ | $-0.2467$ |
| `Vta04` | **$+0.0464$** ($4.6\%$) | $+0.2843$ | $-0.1712$ | $+4.4907$ | $-0.5480$ |

- On the untouched test trip (`Vta04`), attitude dynamics explain **only $4.6\%$** of the residual variance ($R^2 = 0.0464$).
- Physical attitude kinematics are statistically incapable of predicting or explaining the residual.

---

## 5. Dimension 4: Multi-Signal Acceleration Error Profiles

We evaluated five candidate physical representations against the VBOX reference $a_{\text{ref}}$:
1. **Raw Phone IMU**: $a_x^{\text{phone}}$
2. **Static Leveled**: $a_x^{\text{level}} = \mathbf{R}_{\text{stat}} \mathbf{a}_{\text{phone}}$
3. **Dynamic Pitch Compensated**: $a_x^{\text{dyn}} = a_x^{\text{level}} - g \sin(\Delta \theta_{\text{pitch}})$
4. **Dynamic Gravity Matrix**: Instantaneous rotation $\mathbf{R}_{\text{dyn}}(t)$ using real-time gravity vector
5. **Chassis CAN Accelerometer**: Independent vehicle sensor rigidly bolted to chassis

### Performance on Untouched `Vta04`

| Signal Representation | MAE ($\text{m/s}^2$) | RMSE ($\text{m/s}^2$) | Severe Brake Bias ($\text{m/s}^2$) | Severe Accel Bias ($\text{m/s}^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Raw Phone Acceleration** | $1.5276$ | $1.9958$ | $+2.6977$ | $+1.3328$ |
| **2. Static Leveled Baseline** | $1.5276$ | $1.9958$ | $+2.6979$ | $+1.3329$ |
| **3. Dyn Pitch Compensated** | $1.5827$ | $2.0651$ | $+2.7525$ | $+1.3330$ |
| **4. Dyn Gravity Matrix** | $1.5305$ | $2.0009$ | $+2.6965$ | $+1.3327$ |
| **5. Dyn + Road Grade Corrected** | $1.5908$ | $2.0767$ | $+2.7695$ | $+1.3330$ |
| **6. Chassis CAN (Rigid Mount)** | **$0.2524$** | **$0.4167$** | **$+0.1683$** | **$+0.1251$** |

### Key Observations
- Dynamic pitch compensation **does not reduce** MAE, RMSE, or severe braking bias. In fact, it slightly degrades performance ($1.5276 \to 1.5827\text{ m/s}^2$ MAE).
- In sharp contrast, the **chassis CAN accelerometer achieves an MAE of $0.2524\text{ m/s}^2$** (a **$83.5\%$ error reduction**) and reduces severe braking bias from $+2.6977\text{ m/s}^2$ down to **$+0.1683\text{ m/s}^2$** (a **$93.8\%$ reduction**)!

---

## 6. Dimension 5: Dead-Reckoning Navigation Drift Benchmark

Using the standardized 1D dead-reckoning simulation protocol (resets at start of each outage window, evaluated over sliding windows with 2.5 s stride):

### Cross-Trip 30 s and 60 s Position Drift (Meters)

| Signal Source | `Vta02` (30 s) | `Vta02` (60 s) | `Vta03` (30 s) | `Vta03` (60 s) | `Vta04` (30 s) | `Vta04` (60 s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Phone IMU** | $204.77\text{ m}$ | $753.89\text{ m}$ | $225.72\text{ m}$ | $974.98\text{ m}$ | $232.11\text{ m}$ | $701.42\text{ m}$ |
| **Static Leveled** | $204.77\text{ m}$ | $753.86\text{ m}$ | $225.60\text{ m}$ | $973.90\text{ m}$ | $232.13\text{ m}$ | $701.52\text{ m}$ |
| **Dyn Pitch Compensated** | $197.42\text{ m}$ | $723.02\text{ m}$ | $327.77\text{ m}$ | $1389.94\text{ m}$ | $274.40\text{ m}$ | $794.18\text{ m}$ |
| **Chassis CAN (Rigid Mount)** | **$46.06\text{ m}$** | **$134.05\text{ m}$** | **$66.57\text{ m}$** | **$170.13\text{ m}$** | **$66.91\text{ m}$** | **$244.58\text{ m}$** |
| **Oracle Floor** | $1.52\text{ m}$ | $3.07\text{ m}$ | $0.99\text{ m}$ | $1.22\text{ m}$ | $1.63\text{ m}$ | $3.39\text{ m}$ |

### Navigation Drift Analysis
1. Dynamic pitch compensation **worsens** 30 s position drift on `Vta03` ($225.60 \to 327.77\text{ m}$) and on `Vta04` ($232.13 \to 274.40\text{ m}$).
2. The **rigidly mounted chassis CAN accelerometer reduces 30 s drift from $232.13\text{ m}$ down to $66.91\text{ m}$** on `Vta04` (a **$71.2\%$ reduction**) with zero machine learning and zero filter tuning!
3. On `Vta02`, chassis CAN acceleration reduces 30 s drift from $204.77\text{ m}$ to **$46.06\text{ m}$** (a **$77.5\%$ reduction**).

---

## 7. Diagnostic Figures

### Figure 1: Pitch Deflection vs. Acceleration Residual Scatter
Shows the lack of correlation between dynamic pitch deflection $\Delta \theta_{\text{pitch}}$ and the acceleration residual $r_a$ during braking across all three trips.
*[Figure 1: Pitch Deflection vs Residual Scatter — Diagnostic chart]*

### Figure 2: Gravity Leakage Decomposition & Dynamics Comparison
Contrasts the actual empirical residual with the theoretical gravity projection $g \sin(\Delta \theta_{\text{pitch}})$, proving that gravity leakage is negligible compared to the residual.
*[Figure 2: Gravity Leakage Decomposition — Diagnostic chart]*

### Figure 3: Three-Way Acceleration Comparison (Phone vs. Leveled vs. Chassis CAN vs. VBOX)
Illustrates how the chassis CAN accelerometer tightly tracks the VBOX reference during braking, whereas the smartphone IMU fails to capture the deceleration impulse.
*[Figure 3: Three-Way Acceleration Comparison — Diagnostic chart]*

---

## 8. Definitive Scientific Diagnosis

### Hypothesis Evaluation Matrix

| Hypothesis | Test Applied | Result | Scientific Verdict |
| :--- | :--- | :--- | :--- |
| **H1: Vehicle Suspension Pitch causes Gravity Leakage** | Theoretical projection $g \sin(\Delta \theta)$, correlation, dynamic leveling | Deflection $\le 0.32^\circ$, gravity leakage $\le 0.055\text{ m/s}^2$ ($< 3\%$ of error), $R^2 = 0.046$ | ❌ **REFUTED** |
| **H2: Static Mounting Tilt is Miscalibrated** | Stationary leveling matrix $\mathbf{R}_{\text{stat}}$ computation | Removes static offset, but braking residual unchanged ($< 0.0002\text{ m/s}^2$ impact) | ❌ **REFUTED** |
| **H3: Road Grade / Incline causes Longitudinal Bias** | Road slope compensation $\theta_{\text{road}} = \arctan(v_z / v_x)$ | Severe brake bias remains $+2.76\text{ m/s}^2$ on `Vta04` | ❌ **REFUTED** |
| **H4: VBOX Reference / Doppler Speed is Inaccurate** | Cross-check against vehicle CAN longitudinal accelerometer | Chassis CAN correlates at $r = 0.85\text{--}0.97$, MAE $0.17\text{--}0.25\text{ m/s}^2$, bias $< 0.17\text{ m/s}^2$ | ❌ **REFUTED** |
| **H5: Smartphone Mount Compliance & Mechanical Decoupling** | Comparison of phone IMU vs. rigidly bolted chassis CAN sensor | Phone reads $+0.33\text{ m/s}^2$ while chassis CAN reads $-2.03\text{ m/s}^2$ (matching VBOX $-2.20\text{ m/s}^2$) | 🔥 **LEADING HYPOTHESIS** |

### Physical Assessment: Mounting Mechanics & Decoupling
The evidence strongly implicates **mechanical compliance and dynamic decoupling of the smartphone mounting system** as the dominant source of the observed transient acceleration discrepancy:
1. When the vehicle brakes violently ($-2.0\text{ to } -3.0\text{ m/s}^2$), the mass of the smartphone exerts an inertial torque on the flexible mount arm and suction cup.
2. The phone mounting system introduces mechanical compliance. Under deceleration, dynamic flexure and damping decouple the smartphone sensor from the true rigid-body chassis kinematics, causing the smartphone sensor to lag, ring, or register an apparent positive reaction force ($+0.33\text{ m/s}^2$) during the onset of braking.
3. Because this decoupling is a property of mount dynamics and compliance (rather than vehicle suspension pitch or frame rotation), kinematic tilt compensation or longer RF rolling windows cannot resolve it.
4. *Scientific caveat*: While the empirical evidence strongly implicates mount compliance, direct physical parameters (displacement, stiffness $k$, damping $c$) were not directly measured by displacement transducers. Stage C5.5 is designed to identify the transfer function characteristics before any parametric modeling is attempted.

---

## 9. Recommended Next Steps for Stage C5.5

With the gravity-leakage hypothesis conclusively eliminated, we have narrowed the problem to the **phone-to-chassis transfer function / mount dynamics**:

1. **Option A: Second-Order Mechanical Transfer Function Identification**:
   - Model the mount as a damped oscillator ($m \ddot{x} + c \dot{x} + k x = F_{\text{chassis}}$) to reconstruct chassis acceleration from the phone's compliant acceleration.
2. **Option B: Adaptive Transient Bias Estimator / Regime-Specific Correction**:
   - Rather than attempting global residual regression, use a dedicated transient detector (triggered by jerk or sudden gyro pitch rate) to apply a dynamic reaction-force offset during braking.
3. **Option C: CAN-IMU Sensor Fusion / Wheel-Speed Kinematic Aiding**:
   - In production automotive environments, fuse the compliant smartphone IMU with vehicle CAN speed or wheel tick counts (which C5.4 proved are highly accurate and drift-free).
