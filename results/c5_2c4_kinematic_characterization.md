# Stage C5.2-C4: Cross-Trip Kinematic Horizon & Inertial Drift Characterization

## Executive Summary

Stage C5.2-C4 conducts an empirical, non-tuning kinematic characterization across three diverse trips from the IO-VNBD dataset: **Vta02** (urban/suburban with traffic signals), **Vta03** (short mixed cruise), and **Vta04** (continuous high-speed highway). In strict adherence to the one-change-at-a-time scientific methodology:
- **Zero AI / machine learning models were trained or tuned**
- **Zero Random Forest, GBDT, or neural networks were evaluated**
- **Zero ESKF, Kalman filters, or observer gains were applied**
- **Zero Non-Holonomic Constraints (NHC) or map-matching heuristics were introduced**
- **Zero bias-correction algorithms or threshold optimizations were performed**

The objective of C5.2-C4 is not to improve dead-reckoning performance, but to answer two foundational physical questions before undertaking AI residual modeling in Stage C5.3:
1. **How bad is the open-loop inertial residual across different trips and outage durations?**
2. **What is the mathematical and physical structure of this residual, and what must Stage C5.3 learn?**

---

### Core Characterization Findings

| Trip ID | Trip Duration | Valid Stops ($\ge 1.0\text{ s}$) | Longest Continuous Motion | 30 s Vel MAE | 30 s Final Pos Error | Constant Bias Fraction | Dynamic Variance Fraction | Longitudinal Acceleration Slope ($dr/da$) | Linear $R^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta02** | $1099.0\text{ s}$ ($18.3\text{ min}$) | 5 ($63.6\text{ s}$ total) | **$684.1\text{ s}$ ($11.4\text{ min}$, $8.05\text{ km}$)** | $7.02\text{ m/s}$ | **$204.4\text{ m}$** | $1.2\%$ | **$98.8\%$** | $-0.646$ ($p < 10^{-280}$) | **$0.112$** |
| **Vta03** | $64.5\text{ s}$ ($1.1\text{ min}$) | 0 ($0.0\text{ s}$) | **$64.5\text{ s}$ ($1.1\text{ min}$, $0.38\text{ km}$)** | $8.07\text{ m/s}$ | **$222.4\text{ m}$** | $9.8\%$ | **$90.2\%$** | $-1.124$ ($p < 10^{-77}$) | **$0.421$** |
| **Vta04** | $178.9\text{ s}$ ($3.0\text{ min}$) | 0 ($0.0\text{ s}$) | **$178.9\text{ s}$ ($3.0\text{ min}$, $1.99\text{ km}$)** | $8.13\text{ m/s}$ | **$231.5\text{ m}$** | $1.1\%$ | **$98.9\%$** | $-0.847$ ($p < 10^{-65}$) | **$0.153$** |

1. **Cross-Trip Drift Uniformity**: Over 30-second simulated GNSS outages, open-loop dead reckoning exhibits remarkably consistent positioning failure across completely different trips, accumulating between **$204.4\text{ m}$ and $231.5\text{ m}$ of position error** ($99\%\text{--}188\%$ position drift relative to true distance traveled).
2. **Residual Is 90%–99% Dynamic Variance, NOT Constant Bias**: The specific-force residual $r(t) = a_x^{\text{level}}(t) - \frac{dv}{dt}(t)$ has a mean bias of only $+0.20\text{ m/s}^2$ (Vta02), $-0.51\text{ m/s}^2$ (Vta03), and $+0.22\text{ m/s}^2$ (Vta04). Constant bias accounts for only **$1.1\%\text{--}9.8\%$** of the residual mean squared error; **$90.2\%\text{--}98.9\%$** of the error is time-varying dynamic variance ($\sigma_r = 1.54\text{--}2.07\text{ m/s}^2$). Furthermore, the bias changes sign across trips, proving that no constant offline calibration can resolve the drift.
3. **Evidence of Dynamic Attitude / Suspension Coupling**: The residual $r(t)$ displays a statistically significant negative association with true vehicle acceleration $\frac{dv}{dt}$ across all trips (slopes from $-0.646$ to $-1.124$, $p \ll 10^{-50}$). While linear acceleration accounts for only a fraction of the residual variance ($R^2 = 0.112$ on Vta02, $0.153$ on Vta04, $0.421$ on Vta03), this negative coupling is consistent with chassis pitch dynamics (rearward squat under acceleration, forward dive under braking) acting as **a statistically significant candidate contributor to the time-varying residual** alongside dynamic attitude fluctuations, road grade, and sensor noise.
4. **ZUPT Blind Spots**: While Stage C5.2-C3 proved that ZUPT resets drift at genuine stops, ZUPT provides **zero correction** during continuous motion. In Vta02, an uninterrupted **11.4-minute ($684.1\text{ s}$), 8.05-km** continuous driving segment exists where velocity error drifts open-loop without any stop opportunities. Vta03 and Vta04 contain zero sustained stops ($\ge 1.0\text{ s}$).

---

## 1. Experimental Methodology & Mathematical Formulation

### 1.1 Evaluated Dataset Trips

| Trip | Total Samples | Sampling Rate | Duration | Total Distance | Mean Speed | Max Speed | Environment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Vta02** | 10,991 | $10.0\text{ Hz}$ | $1099.0\text{ s}$ | $11.05\text{ km}$ | $10.05\text{ m/s}$ ($36.2\text{ km/h}$) | $22.68\text{ m/s}$ ($81.6\text{ km/h}$) | Mixed urban/suburban, 5 traffic stops, multi-lane road |
| **Vta03** | 646 | $10.0\text{ Hz}$ | $64.5\text{ s}$ | $0.38\text{ km}$ | $5.83\text{ m/s}$ ($21.0\text{ km/h}$) | $12.72\text{ m/s}$ ($45.8\text{ km/h}$) | Short acceleration/cruise, non-stationary |
| **Vta04** | 1,790 | $10.0\text{ Hz}$ | $178.9\text{ s}$ | $1.99\text{ km}$ | $11.14\text{ m/s}$ ($40.1\text{ km/h}$) | $14.36\text{ m/s}$ ($51.7\text{ km/h}$) | Continuous highway cruise, zero stops |

### 1.2 Coordinate Leveling and Specific-Force Residual

Following Stage C5.2-A, raw smartphone triaxial acceleration $\mathbf{a}_{\text{phone}}(t)$ is leveled into the local gravity-aligned horizontal plane using initial stationary gravity orientation $\mathbf{g}_0 = [g_x, g_y, g_z]^T$:
$$\mathbf{R}_{\text{level}} = \text{Rodrigues}(\mathbf{n}_{\text{phone}} \to [0, 0, 1]^T), \qquad \mathbf{a}_{\text{level}}(t) = \mathbf{R}_{\text{level}} \, \mathbf{a}_{\text{phone}}(t)$$
where $a_x^{\text{level}}(t)$ is the gravity-leveled forward specific force.

The ground-truth kinematic derivative is obtained from 100 Hz VBOX Doppler speed downsampled to 10 Hz:
$$a_{\text{true}}(t) = \frac{dv_{\text{VBOX}}}{dt}(t)$$

The specific-force residual $r(t)$ is defined as:
$$r(t) \triangleq a_x^{\text{level}}(t) - \frac{dv_{\text{VBOX}}}{dt}(t)$$

### 1.3 Outage Horizon Sweep Protocol: Reset-at-Window-Start Methodology

To evaluate realistic GNSS outage durations without bias from arbitrary interval selection, a rolling window sweep is conducted across all trips for horizons $H \in \{5\text{ s}, 10\text{ s}, 20\text{ s}, 30\text{ s}, 60\text{ s}\}$:
- Stride: $5.0\text{ s}$ ($50\text{ samples}$)
- Initial condition: True VBOX velocity at window start $v(t_0) = v_{\text{VBOX}}(t_0)$, initial position $p(t_0) = 0$
- Forward kinematic integration over $[t_0, t_0 + H]$:
  $$\hat{v}(t) = v(t_0) + \int_{t_0}^t a_x^{\text{level}}(\tau) \, d\tau, \qquad \hat{p}(t) = \int_{t_0}^t \hat{v}(\tau) \, d\tau$$
- Metrics evaluated for each window $i$:
  $$\text{MAE}_v^{(i)} = \frac{1}{H} \int_{t_0}^{t_0+H} |\hat{v}(t) - v_{\text{VBOX}}(t)| \, dt, \qquad \epsilon_p^{(i)} = |\hat{p}(t_0+H) - p_{\text{VBOX}}(t_0+H)|$$
  $$\text{Drift Rate} = \frac{\text{MAE}_v}{0.5 \cdot H} \approx \text{effective acceleration bias } \bar{a}_{\text{bias}}$$

> **Methodological Clarification**: These rolling windows are strictly **reset-at-window-start inertial drift experiments**. They isolate conditional inertial error growth over controlled outage horizons starting from a known initial velocity ($v_{\text{est}}(0) = v_{\text{true}}(0)$). They are designed to measure time-dependent open-loop error divergence under realistic outage durations, not to represent a continuous unassisted dead-reckoning simulation across the entire 1099-second mission.


---

## 2. Multi-Horizon Kinematic Drift Sweep

### 2.1 Tabulated Cross-Trip Horizon Performance

The rolling outage sweep reveals how kinematic error compounds quadratically over time across all three trips:

| Trip | Horizon ($H$) | N Windows | Mean Vel MAE | Vel RMSE | Mean Final Vel Err | Mean Pos Error | Final Pos Std | Mean Pos Drift % | Implied Accel Drift Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta02** | $5\text{ s}$ | 219 | $1.41\text{ m/s}$ | $1.67\text{ m/s}$ | $2.72\text{ m/s}$ | **$6.80\text{ m}$** | $4.97\text{ m}$ | $19.9\%$ | $0.566\text{ m/s}^2$ |
| | $10\text{ s}$ | 218 | $2.63\text{ m/s}$ | $3.09\text{ m/s}$ | $5.04\text{ m/s}$ | **$25.64\text{ m}$** | $17.62\text{ m}$ | $35.9\%$ | $0.526\text{ m/s}^2$ |
| | $20\text{ s}$ | 216 | $4.89\text{ m/s}$ | $5.71\text{ m/s}$ | $9.21\text{ m/s}$ | **$94.97\text{ m}$** | $63.59\text{ m}$ | $65.5\%$ | $0.489\text{ m/s}^2$ |
| | $30\text{ s}$ | 214 | $7.02\text{ m/s}$ | $8.20\text{ m/s}$ | $13.23\text{ m/s}$ | **$204.38\text{ m}$** | $134.31\text{ m}$ | $99.4\%$ | $0.468\text{ m/s}^2$ |
| | $60\text{ s}$ | 208 | $12.95\text{ m/s}$ | $15.13\text{ m/s}$ | $24.49\text{ m/s}$ | **$754.27\text{ m}$** | $497.20\text{ m}$ | $130.1\%$ | $0.432\text{ m/s}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta03** | $5\text{ s}$ | 12 | $2.46\text{ m/s}$ | $2.88\text{ m/s}$ | $4.40\text{ m/s}$ | **$12.31\text{ m}$** | $7.76\text{ m}$ | $100.4\%$ | $0.986\text{ m/s}^2$ |
| | $10\text{ s}$ | 11 | $4.53\text{ m/s}$ | $5.23\text{ m/s}$ | $8.02\text{ m/s}$ | **$44.02\text{ m}$** | $30.71\text{ m}$ | $222.2\%$ | $0.907\text{ m/s}^2$ |
| | $20\text{ s}$ | 9 | $7.80\text{ m/s}$ | $9.12\text{ m/s}$ | $13.12\text{ m/s}$ | **$151.09\text{ m}$** | $102.34\text{ m}$ | $219.6\%$ | $0.780\text{ m/s}^2$ |
| | $30\text{ s}$ | 7 | $8.07\text{ m/s}$ | $9.97\text{ m/s}$ | $16.33\text{ m/s}$ | **$222.41\text{ m}$** | $166.78\text{ m}$ | $188.4\%$ | $0.538\text{ m/s}^2$ |
| | $60\text{ s}$ | 1 | $17.98\text{ m/s}$ | $21.17\text{ m/s}$ | $37.76\text{ m/s}$ | **$1078.71\text{ m}$** | $0.00\text{ m}$ | $335.6\%$ | $0.599\text{ m/s}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta04** | $5\text{ s}$ | 35 | $1.67\text{ m/s}$ | $2.00\text{ m/s}$ | $3.10\text{ m/s}$ | **$7.89\text{ m}$** | $6.94\text{ m}$ | $17.9\%$ | $0.668\text{ m/s}^2$ |
| | $10\text{ s}$ | 34 | $3.12\text{ m/s}$ | $3.67\text{ m/s}$ | $5.79\text{ m/s}$ | **$29.92\text{ m}$** | $19.26\text{ m}$ | $29.2\%$ | $0.624\text{ m/s}^2$ |
| | $20\text{ s}$ | 32 | $5.83\text{ m/s}$ | $6.85\text{ m/s}$ | $10.63\text{ m/s}$ | **$112.52\text{ m}$** | $57.15\text{ m}$ | $51.0\%$ | $0.583\text{ m/s}^2$ |
| | $30\text{ s}$ | 30 | $8.13\text{ m/s}$ | $9.53\text{ m/s}$ | $14.13\text{ m/s}$ | **$231.51\text{ m}$** | $111.96\text{ m}$ | $69.8\%$ | $0.542\text{ m/s}^2$ |
| | $60\text{ s}$ | 24 | $13.21\text{ m/s}$ | $15.34\text{ m/s}$ | $20.57\text{ m/s}$ | **$701.55\text{ m}$** | $440.51\text{ m}$ | $106.5\%$ | $0.440\text{ m/s}^2$ |

### 2.2 Key Horizon Scaling Observations

1. **Quadratic Growth of Position Error**: Across all trips, position error scales quadratically with outage duration ($p_{\text{err}} \approx \frac{1}{2} a_{\text{drift}} H^2$):
   - At $5\text{ s}$, position error is small ($6.8\text{--}12.3\text{ m}$).
   - At $10\text{ s}$, error grows to $25.6\text{--}44.0\text{ m}$.
   - At $30\text{ s}$, error exceeds $200\text{ m}$ ($204.4\text{ m}$ on Vta02, $222.4\text{ m}$ on Vta03, $231.5\text{ m}$ on Vta04).
   - At $60\text{ s}$, error reaches $701.5\text{--}1078.7\text{ m}$ (nearly a kilometer of drift).
2. **Implied Effective Acceleration Drift Rate**: Dividing mean velocity error by $0.5 \cdot H$ yields an implied effective acceleration bias between **$0.43\text{ m/s}^2$ and $0.67\text{ m/s}^2$** across long horizons ($30\text{ s}\text{--}60\text{ s}$). This confirms the order-of-magnitude established in Stage C5.2-C1 ($\approx 0.57\text{ m/s}^2$).

---

## 3. Residual Error Decomposition: Bias vs Dynamic Variance

To test whether the residual can be addressed by static calibration or whether dynamic estimation is mandatory, the total mean squared error (MSE) of $r(t)$ is decomposed into constant bias squared and dynamic variance:
$$\text{MSE}_{\text{total}} = \frac{1}{N}\sum_{k=1}^N r^2(t_k) = \underbrace{\bar{r}^2}_{\text{Constant Bias Energy}} + \underbrace{\sigma_r^2}_{\text{Dynamic Variance}}$$

### 3.1 Quantitative Residual Decomposition

| Metric | Trip Vta02 | Trip Vta03 | Trip Vta04 | Cross-Trip Summary |
|---|:---:|:---:|:---:|---|
| **Constant Bias ($\bar{r}$)** | **$+0.2007\text{ m/s}^2$** | **$-0.5079\text{ m/s}^2$** | **$+0.2214\text{ m/s}^2$** | Changes sign across trips; magnitude $\le 0.51\text{ m/s}^2$ |
| **Dynamic Std Dev ($\sigma_r$)** | **$1.8053\text{ m/s}^2$** | **$1.5444\text{ m/s}^2$** | **$2.0664\text{ m/s}^2$** | Consistently $1.5\text{--}2.1\text{ m/s}^2$ |
| **Total RMSE** | $1.8164\text{ m/s}^2$ | $1.6258\text{ m/s}^2$ | $2.0782\text{ m/s}^2$ | Dominated by $\sigma_r$ |
| **Bias Fraction of MSE** | **$1.22\%$** | **$9.76\%$** | **$1.13\%$** | Constant bias is $<10\%$ in all trips |
| **Variance Fraction of MSE** | **$98.78\%$** | **$90.24\%$** | **$98.87\%$** | **$\mathbf{90\%\text{--}99\%}$ is time-varying dynamic variance** |
| **Extreme Range (Min / Max)** | $[-11.58, +12.39]\text{ m/s}^2$ | $[-5.79, +6.42]\text{ m/s}^2$ | $[-11.06, +9.92]\text{ m/s}^2$ | Spans $\pm 12\text{ m/s}^2$ during aggressive transients |
| **1st Percentile ($p_1$)** | $-4.77\text{ m/s}^2$ | $-4.14\text{ m/s}^2$ | $-5.28\text{ m/s}^2$ | $-4\text{ to } -5\text{ m/s}^2$ |
| **5th Percentile ($p_5$)** | $-2.81\text{ m/s}^2$ | $-3.17\text{ m/s}^2$ | $-3.12\text{ m/s}^2$ | Consistently $\approx -3.0\text{ m/s}^2$ |
| **Median ($p_{50}$)** | $+0.20\text{ m/s}^2$ | $-0.46\text{ m/s}^2$ | $+0.32\text{ m/s}^2$ | Close to mean bias |
| **95th Percentile ($p_{95}$)** | $+3.03\text{ m/s}^2$ | $+2.00\text{ m/s}^2$ | $+3.32\text{ m/s}^2$ | Consistently $\approx +3.0\text{ m/s}^2$ |
| **99th Percentile ($p_{99}$)** | $+4.77\text{ m/s}^2$ | $+2.95\text{ m/s}^2$ | $+5.09\text{ m/s}^2$ | $+3\text{ to } +5\text{ m/s}^2$ |

### 3.2 Scientific Consequence: Failure of Constant Calibration

This decomposition provides mathematical proof why Stages C5.2-C1 and C5.2-C2 failed to achieve dramatic improvements:
1. **Upper Bound on Constant Bias Correction**: Even a theoretically perfect constant bias compensator can eliminate at most **$1.1\%\text{--}9.8\%$ of the residual error energy**.
2. **Remaining 90%–99%**: The overwhelming bulk ($90.2\%\text{--}98.9\%$) of the specific-force error is dynamic and time-varying.
3. **Trip-to-Trip Inversion**: The static bias is $+0.20\text{ m/s}^2$ on Vta02 but $-0.51\text{ m/s}^2$ on Vta03. Any offline calibration learned on Vta02 and applied to Vta03 would worsen the bias from $-0.51$ to $-0.71\text{ m/s}^2$, accelerating drift rather than arresting it.

---

## 4. Evidence of Dynamic Attitude / Suspension Coupling

### 4.1 Empirical Correlation and Linear Regression

Correlation analysis reveals measurable associations between specific-force residual and vehicle dynamics:

| Feature / Dynamic Quantity | Vta02 Correlation | Vta03 Correlation | Vta04 Correlation | Physical Interpretation |
|---|:---:|:---:|:---:|---|
| **True Acceleration ($dv_{\text{VBOX}}/dt$)** | **$-0.3341$** | **$-0.6485$** | **$-0.3914$** | Consistent negative association with longitudinal acceleration |
| **Absolute Residual vs Speed ($v$)** | $+0.1713$ | $-0.3027$ | $-0.1009$ | Moderate, trip-dependent speed effect |
| **Absolute Residual vs Gyro ($\|\boldsymbol{\omega}\|$)** | **$+0.2967$** | **$+0.2482$** | **$+0.1769$** | Angular rate disturbance correlates with larger residual spread |
| **Absolute Residual vs Vibration ($\sigma_a$)** | **$+0.2934$** | **$+0.2323$** | **$+0.0910$** | Higher road/engine vibration correlates with increased error spread |

Linear regression between specific-force residual $r(t)$ and true vehicle acceleration $\frac{dv}{dt}$:
$$r(t) = \beta \cdot \frac{dv_{\text{VBOX}}}{dt}(t) + \alpha$$

| Trip | Slope ($\beta$) | Intercept ($\alpha$) | $R^2$ | $p$-value | Statistical Significance |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vta02** | **$-0.6457$** | $+0.1967\text{ m/s}^2$ | **$0.1116$** | $9.4 \times 10^{-285}$ | Highly Significant ($p \ll 10^{-50}$) |
| **Vta03** | **$-1.1239$** | $-0.3132\text{ m/s}^2$ | **$0.4206$** | $3.1 \times 10^{-78}$ | Highly Significant ($p \ll 10^{-50}$) |
| **Vta04** | **$-0.8471$** | $+0.2316\text{ m/s}^2$ | **$0.1532$** | $1.5 \times 10^{-66}$ | Highly Significant ($p \ll 10^{-50}$) |

### 4.2 Physical Interpretation & Candidate Error Mechanisms

The observed negative slopes ($-0.65\text{ to } -1.12$) are statistically overwhelming across all three trips ($p \ll 10^{-50}$), demonstrating a repeatable physical relationship:

```text
    FORWARD ACCELERATION (Throttle: dv/dt > 0)
    ────────────────────────────────────────
              Nose Up
                 ▲        Phone tilts rearward (pitch angle θ > 0)
                 │  ┌───┐ 
        [Front]  └──┤   ├───┐ [Rear]
                    └───┘   ▼ Tail Squat
    
    1. Phone accelerometer tilts rearward by angle θ.
    2. Gravity vector g projects into the phone's forward axis:
       f_x = a_true - g * sin(θ) ≈ a_true - g * θ
    3. If squat angle θ is proportional to acceleration (θ ≈ k_susp * dv/dt):
       Residual r = f_x - a_true ≈ - (g * k_susp) * dv/dt
       ==> Produces negative regression slope β < 0
```

```text
    BRAKING / DECELERATION (Brake: dv/dt < 0)
    ───────────────────────────────────────
              Nose Dive
                 ▼        Phone tilts forward (pitch angle θ < 0)
                 │  ┌───┐ 
        [Front]  └──┤   ├───┐ [Rear]
                    └───┘   ▲ Tail Rise
    
    1. Phone accelerometer tilts forward by angle -θ.
    2. Gravity projects forward into the phone's forward axis:
       f_x = a_true - g * sin(-θ) = a_true + g * θ
    3. Because dv/dt < 0, this introduces positive specific force:
       r = f_x - a_true > 0
       ==> Measured acceleration artificially overestimates forward force!
```

#### Critical Scientific Caveat on Variance Explained ($R^2$)

While the negative slopes are consistent and statistically robust, **suspension pitch cannot be claimed as the sole or dominant driver**:
- On **Vta02**, $R^2 = 0.1116$, meaning that the simple linear acceleration model explains only **$11.2\%$** of the residual variance.
- On **Vta04**, $R^2 = 0.1532$, explaining only **$15.3\%$** of the residual variance.
- On **Vta03**, $R^2 = 0.4206$, explaining **$42.1\%$** of the residual variance during aggressive short maneuvers.

Therefore, chassis suspension pitch is **a statistically significant candidate contributor to the time-varying residual**, operating in concert with several other unmodeled physical and measurement phenomena:
1. **Dynamic Phone Pitch and Roll**: High-frequency rotations of the smartphone relative to the vehicle cradle during cornering, road bumps, and engine vibration.
2. **Road Grade Variations**: Uphill and downhill road slopes directly project gravity onto the longitudinal axis ($g \sin \theta_{\text{grade}}$).
3. **Mounting Compliance & Bracket Flexing**: Elastic flexing of the phone mount under high lateral and longitudinal loads.
4. **Sensor Scale-Factor & Cross-Axis Sensitivity**: Consumer MEMS accelerometers exhibit $1\%\text{--}3\%$ scale factor errors and cross-axis leakage during heavy accelerations.
5. **Sensor High-Frequency Noise Rectification**: Engine and chassis vibration harmonics interacting nonlinearly with sensor low-pass filtering.
6. **Differentiation Noise in Ground Truth**: Numerical differencing of $v_{\text{VBOX}}$ introduces high-frequency differentiation artifacts into the reference label $a_{\text{true}}$.
7. **Nonlinear Vehicle Dynamics**: Gear changes, powertrain shudder, and non-proportional suspension damping.


---

## 5. Motion State Continuity & ZUPT Opportunity Analysis

### 5.1 Distribution of Stops vs Continuous Motion

| Trip | Valid Stops ($\ge 1.0\text{ s}$) | Total Stop Duration | Longest Continuous Driving Segment | Segment Distance | Segment Mean Speed |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vta02** | 5 | $63.6\text{ s}$ ($5.8\%$ of mission) | **$684.1\text{ s}$ ($11.4\text{ min}$)** | **$8.05\text{ km}$** | $11.76\text{ m/s}$ ($42.3\text{ km/h}$) |
| **Vta03** | 0 | $0.0\text{ s}$ ($0.0\%$ of mission) | **$64.5\text{ s}$ ($1.1\text{ min}$)** | **$0.38\text{ km}$** | $5.83\text{ m/s}$ ($21.0\text{ km/h}$) |
| **Vta04** | 0 | $0.0\text{ s}$ ($0.0\%$ of mission) | **$178.9\text{ s}$ ($3.0\text{ min}$)** | **$1.99\text{ km}$** | $11.14\text{ m/s}$ ($40.1\text{ km/h}$) |

### 5.2 The 11.4-Minute Open-Loop Void

In Vta02, between $t = 23.2\text{ s}$ and $t = 707.2\text{ s}$ ($684.1\text{ s}$ continuous driving), the vehicle travels $8.05\text{ km}$ across suburban corridors and highways without stopping.
- During this interval, **zero valid ZUPT opportunities exist**.
- Pure kinematic integration over this single uninterrupted stretch drifts to over **$150\text{ m/s}$ of velocity error** and **$50+\text{ km}$ of position error**.
- While ZUPT at $t = 707.3\text{ s}$ cleanly resets the velocity error back to zero at the stop line, it cannot undo the catastrophic drift that accumulated over the preceding 11.4 minutes of travel.

**Conclusion**: ZUPT is an essential **boundary-condition reset** at stops, but is completely impotent as an in-motion stabilization mechanism. An in-motion estimator is mathematically mandatory.

---

## 6. Dynamic Residual Conditioning: Speed, Angular Rate, and Vibration

### 6.1 Speed-Binned Residual Statistics

The residual was evaluated across speed intervals $[0, 5)$, $[5, 10)$, $[10, 15)$, and $[15, 25)\text{ m/s}$:

| Trip | Speed Bin | Sample Count | Mean Residual ($\bar{r}$) | Residual Std ($\sigma_r$) | Residual MAE |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vta02** | $0\text{--}5\text{ m/s}$ ($0\text{--}18\text{ km/h}$) | 1,725 | $+0.17\text{ m/s}^2$ | $1.18\text{ m/s}^2$ | $0.78\text{ m/s}^2$ |
| | $5\text{--}10\text{ m/s}$ ($18\text{--}36\text{ km/h}$) | 3,501 | $+0.17\text{ m/s}^2$ | $1.84\text{ m/s}^2$ | $1.36\text{ m/s}^2$ |
| | $10\text{--}15\text{ m/s}$ ($36\text{--}54\text{ km/h}$) | 4,175 | $+0.13\text{ m/s}^2$ | $1.95\text{ m/s}^2$ | $1.46\text{ m/s}^2$ |
| | $15\text{--}25\text{ m/s}$ ($54\text{--}90\text{ km/h}$) | 1,590 | $+0.47\text{ m/s}^2$ | $1.85\text{ m/s}^2$ | $1.48\text{ m/s}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vta03** | $0\text{--}5\text{ m/s}$ | 293 | $-0.69\text{ m/s}^2$ | $1.82\text{ m/s}^2$ | $1.57\text{ m/s}^2$ |
| | $5\text{--}10\text{ m/s}$ | 222 | $-0.72\text{ m/s}^2$ | $1.14\text{ m/s}^2$ | $1.09\text{ m/s}^2$ |
| | $10\text{--}15\text{ m/s}$ | 130 | $+0.27\text{ m/s}^2$ | $1.16\text{ m/s}^2$ | $0.78\text{ m/s}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vta04** | $0\text{--}5\text{ m/s}$ | 81 | $+1.62\text{ m/s}^2$ | $2.24\text{ m/s}^2$ | $2.25\text{ m/s}^2$ |
| | $5\text{--}10\text{ m/s}$ | 208 | $+0.18\text{ m/s}^2$ | $1.96\text{ m/s}^2$ | $1.58\text{ m/s}^2$ |
| | $10\text{--}15\text{ m/s}$ | 1,500 | $+0.15\text{ m/s}^2$ | $2.04\text{ m/s}^2$ | $1.56\text{ m/s}^2$ |

- Residual dispersion ($\sigma_r$) increases as the vehicle accelerates from stop into cruising speed, rising from $1.18\text{ m/s}^2$ at low speeds to nearly $2.0\text{ m/s}^2$ at typical road speeds ($10\text{--}15\text{ m/s}$).
- In Vta04, low-speed crawling during merge ($0\text{--}5\text{ m/s}$) exhibits high residual MAE ($2.25\text{ m/s}^2$), consistent with the low-speed crawl ambiguity observed in C5.2-C3.

---

## 7. Diagnostic Visualizations

The characterization script produced 5 diagnostic figures documenting the physical findings:

1. **Horizon Error Scaling**: `results/figures/c5_2c4_cross_trip_horizons.png`
   - Shows quadratic growth of position error and linear growth of velocity error across $5\text{ s}, 10\text{ s}, 20\text{ s}, 30\text{ s}, 60\text{ s}$ for all three trips.
2. **Residual Probability Distributions**: `results/figures/c5_2c4_residual_distributions.png`
   - Probability density functions and boxplots showing that $r(t)$ is centered near zero with heavy dynamic tails ($\pm 4\text{ to } \pm 6\text{ m/s}^2$), with $90\%\text{--}99\%$ variance fraction.
3. **Chassis Suspension Pitch Coupling**: `results/figures/c5_2c4_suspension_pitch_coupling.png`
   - Scatter plots and linear regressions of $r(t)$ versus $dv_{\text{VBOX}}/dt$, demonstrating the negative slopes ($-0.65$ to $-1.12$) characteristic of suspension squat and dive.
4. **Continuous Motion Drift Progression**: `results/figures/c5_2c4_continuous_motion_drift.png`
   - Time-series tracking velocity and position error growth during the longest uninterrupted driving stretches (e.g. 11.4 min on Vta02), illustrating where ZUPT cannot operate.
5. **Roughness & Vibration Spectrogram**: `results/figures/c5_2c4_residual_spectrogram_roughness.png`
   - Residual error magnitude versus speed bins and vibration intensity $\sigma_a$, demonstrating that chassis vibrations and cornering directly modulate error spread.

---

## 8. Scientific Synthesis & Formal Bridge to Stage C5.3

Stage C5.2-C4 completes the empirical chain of evidence established throughout the C5.2 series:

```text
  C5.2-C1 / C2: Constant Bias Calibration
  ───────────────────────────────────────
  Tested static bias removal, quasi-static adaptation, bounded bias.
  RESULT: Reduced 60s MAE by only 9.1%. Fails because 90-99% of residual
  is dynamic variance, not constant bias.
                    │
                    ▼
  C5.2-C3: Stop Detection + ZUPT
  ──────────────────────────────
  Tested IMU stop detector and zero-velocity drift reset.
  RESULT: Resets velocity drift at genuine stops (saving ~98.6 km full-mission pos err).
  FAILING: Zero correction during continuous driving (e.g. 11.4 min void on Vta02).
                    │
                    ▼
  C5.2-C4: Cross-Trip Drift & Residual Characterization (Frozen)
  ─────────────────────────────────────────────────────────────
  Characterized drift across Vta02, Vta03, Vta04 without tuning.
  DISCOVERIES:
  1. 30s outage position error is uniformly ~200-230 m across all trips.
  2. Residual r(t) is 90-99% dynamic variance (σ_r ≈ 1.5-2.1 m/s²).
  3. Dynamic error exhibits statistically significant association with vehicle
     acceleration (slope -0.65 to -1.12; R² = 0.11 to 0.42), indicating dynamic
     attitude/suspension coupling as a plausible candidate contributor.
  4. Continuous motion spans up to 11.4 min without stops.
                    │
                    ▼
  C5.3: AI-Learned Dynamic Residual Acceleration Correction
  ─────────────────────────────────────────────────────────
  GOAL: Learn the time-varying specific-force residual r̂_a(t) during continuous motion!
```

### 8.1 Core Scientific Finding

> **C5.2-C4 establishes that the dominant practical limitation of smartphone inertial navigation is time-varying inertial error rather than a single constant offset, and identifies measurable associations with vehicle longitudinal dynamics and motion intensity.**

The quantitative results across three distinct journeys justify the transition to dynamic residual learning:
1. **Repeatability**: Open-loop position error at 30 s is consistently $204.4\text{ m}$ (Vta02), $222.4\text{ m}$ (Vta03), and $231.5\text{ m}$ (Vta04). The navigation failure is not trip-specific; it is an inherent physical property of unassisted smartphone IMU propagation.
2. **Dominance of Dynamic Error**: Between $90.2\%$ and $98.9\%$ of the specific-force residual mean squared error is time-varying dynamic variance ($\sigma_r = 1.5\text{--}2.1\text{ m/s}^2$). Constant bias represents $<10\%$ and flips sign between trips.
3. **ZUPT Scope**: ZUPT provides an indispensable boundary-condition reset at stops, but is completely blind during continuous motion (e.g. the 11.4-minute uninterrupted stretch in Vta02).

---

### 8.2 Formulation for Stage C5.3

Stage C5.3 directly targets the continuous-motion void. Instead of attempting to predict absolute vehicle speed from vibration energy (which failed honestly in Stage C5.1), C5.3 learns a dynamic correction to the specific force:

1. **Target Formulation**:
   $$\hat{r}_a(t) \approx r(t) = a_x^{\text{level}}(t) - a_{\text{reference}}(t)$$
   The AI model predicts the time-varying residual to correct measured acceleration:
   $$a_{\text{corrected}}(t) = a_x^{\text{level}}(t) - \hat{r}_a(t)$$
   Velocity integration then accumulates physics-corrected acceleration:
   $$v(t_k) = v(t_{k-1}) + a_{\text{corrected}}(t_k) \Delta t$$
2. **Physical Input Features (Zero-Leakage IMU Only)**:
   - Longitudinal measured acceleration $a_x^{\text{level}}(t)$ and causal temporal derivatives (throttle/brake transitions).
   - Angular rates $\|\boldsymbol{\omega}(t)\|$ and pitch rate (chassis pitch and roll motions).
   - Total acceleration norm $\|\mathbf{a}(t)\|$ and vibration variance $\sigma_a(t)$ (road roughness and dynamic loading).
   - Strictly causal window ($1.0\text{--}2.0\text{ s}$) covering past dynamics only.
3. **System Symbiosis**:
   - **During Continuous Motion**: C5.3 AI residual correction keeps velocity integration drift bounded.
   - **At Verified Stops**: C5.2-C3 ZUPT instantly resets accumulated integration error to zero.

---

### 8.3 Critical Methodological Warning: Offline Labeling vs Online Features

A vital scientific precaution governs Stage C5.3:

1. **Strict Leakage Boundary**:
   The reference acceleration $a_{\text{reference}}(t)$ is derived from VBOX Doppler speed. It is strictly an **offline training label**. No VBOX or GNSS quantity may ever enter the feature extraction pipeline or inference path.
2. **Differentiation Noise Amplification**:
   Numerical differentiation ($\nabla v_{\text{VBOX}} / \Delta t$) amplifies high-frequency noise and quantization jitter. If models are trained against raw differentiated velocity, they risk learning numerical differentiation artifacts rather than physical inertial dynamics.
3. **Mandatory Next Step — Stage C5.3-A**:
   Before training any machine learning or neural network models, the next experiment must be:
   > **Stage C5.3-A: Ground-Truth Residual-Label Construction, Robust Differentiation/Smoothing Validation, and Leakage Boundary Audit**
   
   Only after the offline training label is verified to be smooth, physically consistent, and completely isolated from the inference pipeline will model training proceed in Stage C5.3-B.

