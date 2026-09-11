# Stage C5.3-A: Offline Acceleration Reference Construction, Robust Differentiation Validation, CAN Cross-Audit & Leakage Certification

## Executive Summary

Stage C5.3-A establishes the prerequisite data-engineering and scientific foundation for Stage C5.3 (Learned Dynamic Residual Acceleration Correction). In strict adherence to experimental discipline:
- **Zero AI / machine learning models were trained or evaluated** (No RF, GBDT, MLP, or neural nets)
- **Zero ESKF, Kalman filters, Non-Holonomic Constraints (NHC), or map matching were applied**
- **The differentiation method was NOT chosen to minimize residual error against the smartphone IMU**
- Selection was governed strictly by **temporal fidelity, stationary flat-line consistency, jerk noise suppression, velocity reconstruction consistency, and physical plausibility**.

The core research question resolved by Stage C5.3-A is:
> **“What is the most defensible acceleration reference we can derive from VBOX Doppler speed, how does it compare against independent vehicle instrumentation, and what does the smartphone specific-force residual actually look like when that reference is used?”**

---

### Core Benchmark Findings

| Differentiation Candidate | Stationary Noise Std ($\sigma_{\text{stop}}$) | RMS Jerk Energy ($\text{m/s}^3$) | Jerk Reduction vs Raw Central | Peak Braking Accel ($a_{\text{min}}$) | Velocity Recon MAE ($\text{m/s}$)* | HF Power ($>2.0\text{ Hz}$) | Assessment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1. Raw Central Difference** | $0.1016\text{ m/s}^2$ | $5.07\text{ m/s}^3$ | Baseline ($0.0\%$) | $-7.41\text{ m/s}^2$** | $0.0909\text{ m/s}$ | $26.66\%$ | **REJECTED**: Severe numerical jitter & artificial spikes |
| **2a. Butterworth $1.0\text{ Hz}$** | $0.0671\text{ m/s}^2$ | $1.07\text{ m/s}^3$ | $-95.6\%$ | $-5.15\text{ m/s}^2$ | $0.1115\text{ m/s}$ | $0.01\%$ | Slight over-attenuation of sharp deceleration |
| **2b. Butterworth $1.5\text{ Hz}$** | $0.0723\text{ m/s}^2$ | $1.80\text{ m/s}^3$ | $-87.5\%$ | $-5.28\text{ m/s}^2$ | $0.1023\text{ m/s}$ | $0.18\%$ | Strong alternative with excellent spectral roll-off |
| **2c. Butterworth $2.0\text{ Hz}$** | $0.0780\text{ m/s}^2$ | $2.55\text{ m/s}^3$ | $-74.7\%$ | $-5.43\text{ m/s}^2$ | $0.1011\text{ m/s}$ | $1.42\%$ | Passes some high-frequency quantization noise |
| **3a. Savitzky-Golay ($W=5, p=2$)** | $0.0769\text{ m/s}^2$ | $2.32\text{ m/s}^3$ | $-79.1\%$ | $-5.43\text{ m/s}^2$ | $0.1380\text{ m/s}$ | $3.14\%$ | Short window retains substantial differencing ripple |
| **3b. Savitzky-Golay ($W=9, p=2$)** | **$0.0700\text{ m/s}^2$** | **$1.00\text{ m/s}^3$** | **$-96.1\%$** | **$-4.98\text{ m/s}^2$** | **$0.1022\text{ m/s}$** | **$0.40\%$** | **SELECTED**: Optimal fidelity-noise Pareto point |
| **3c. Savitzky-Golay ($W=15, p=2$)** | $0.0811\text{ m/s}^2$ | $0.60\text{ m/s}^3$ | $-98.6\%$ | $-4.65\text{ m/s}^2$ | $0.4230\text{ m/s}$ | $0.05\%$ | **REJECTED**: Oversmoothed, clips braking by $10\%$ |
| **4a. Gaussian ($\sigma=0.1\text{ s}$)** | $0.0787\text{ m/s}^2$ | $2.83\text{ m/s}^3$ | $-68.9\%$ | $-5.49\text{ m/s}^2$ | $0.0693\text{ m/s}$ | $5.35\%$ | Insufficient noise rejection above $2\text{ Hz}$ |
| **4b. Gaussian ($\sigma=0.2\text{ s}$)** | $0.0695\text{ m/s}^2$ | $1.06\text{ m/s}^3$ | $-95.7\%$ | $-4.98\text{ m/s}^2$ | $0.2133\text{ m/s}$ | $0.01\%$ | Clean, but higher reconstruction drift |
| **5. Cubic Spline ($s=0.1$)** | $0.1973\text{ m/s}^2$ | $0.36\text{ m/s}^3$ | $-99.5\%$ | $-3.92\text{ m/s}^2$ | $0.1822\text{ m/s}$ | $0.00\%$ | **REJECTED**: Severe peak clipping ($-3.92\text{ m/s}^2$) |

*\*Velocity reconstruction MAE measures mathematical consistency between the derivative and source speed, not an independent physical proof.*
*\*\*Raw difference $-7.41\text{ m/s}^2$ is an isolated single-sample quantization artifact; true surrounding deceleration is $-5.0\text{ to } -5.3\text{ m/s}^2$.*

---

## 1. Candidate Differentiation Methods Evaluated

Ten differentiation configurations spanning five mathematical families were evaluated on the 10 Hz VBOX Doppler speed stream $v_{\text{VBOX}}(t)$:

1. **Raw Central Differencing**:
   $$a_{\text{raw}}(k) = \frac{v(k+1) - v(k-1)}{2 \Delta t}, \qquad \Delta t = 0.1\text{ s}$$
2. **Zero-Phase Butterworth Filter + Central Difference**:
   Forward-backward 2nd-order Butterworth filtering with cutoff frequencies $f_c \in \{1.0\text{ Hz}, 1.5\text{ Hz}, 2.0\text{ Hz}\}$.
3. **Savitzky-Golay Polynomial Derivative Filters**:
   Local least-squares parabolic fit ($p=2$) over centered windows $W \in \{5, 9, 15\text{ samples}\}$ ($0.5\text{ s}, 0.9\text{ s}, 1.5\text{ s}$).
4. **Gaussian Analytical Derivative Filters**:
   Convolution with analytical derivative of Gaussian kernel $\frac{d}{dt} \mathcal{N}(0, \sigma^2)$ for $\sigma \in \{0.1\text{ s}, 0.2\text{ s}\}$.
5. **Regularized Smoothing Splines**:
   Cubic smoothing spline with roughness penalty parameterized by smoothing factor $s$.

---

## 2. Independent Cross-Check: SG9 Reference vs. Factory CAN-Bus Accelerometer

To address whether the selected **`3b_savgol_w9_p2`** reference reflects genuine physical vehicle acceleration rather than an arbitrary mathematical construct, SG9 was cross-checked against the Ford Fiesta's factory chassis accelerometer channel (`veh_accel_long_ms2` from the CAN bus / ESC cluster).

| Trip ID | Correlation ($\text{corr}(a_{\text{SG9}}, a_{\text{CAN}})$) | Temporal Lag ($\tau_{\text{peak}}$) | Amplitude Ratio ($\sigma_{\text{CAN}} / \sigma_{\text{SG9}}$) | Mean Diff ($a_{\text{CAN}} - a_{\text{SG9}}$) | Std Diff ($\sigma_{\Delta}$) | Peak Braking Timing Match |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta02** (Urban, 1099 s) | **$0.9301$** | **$0.00\text{ s}$ ($0\text{ samples}$)** | **$1.0510$** | $-0.0174\text{ m/s}^2$ | $0.2763\text{ m/s}^2$ | Exact ($t = 1069.1\text{ s}$ for both) |
| **Vta03** (Mixed, 65 s) | **$0.9669$** | **$0.00\text{ s}$ ($0\text{ samples}$)** | **$1.0848$** | $-0.0895\text{ m/s}^2$ | $0.2272\text{ m/s}^2$ | Exact ($t = 17.7\text{ s}$ for both) |
| **Vta04** (Highway, 179 s) | **$0.8464$** | **$0.00\text{ s}$ ($0\text{ samples}$)** | **$0.9716$** | $+0.0894\text{ m/s}^2$ | $0.4070\text{ m/s}^2$ | Consistent ($0.0\text{ s}$ lag peak) |

### Key Physical Findings from the CAN Cross-Audit:
1. **Zero Temporal Delay**: Cross-correlation peaks at precisely **$\tau = 0.00\text{ s}$ (zero samples delay)** across all three journeys, confirming that Savitzky-Golay parabolic differentiation introduces zero phase distortion or time lag.
2. **Matched Dynamic Scale**: The standard deviation ratio $\sigma_{\text{CAN}} / \sigma_{\text{SG9}}$ is between **$0.97$ and $1.08$** across all trips, proving that SG9 preserves genuine vehicle dynamic scale without amplitude attenuation.
3. **Physical Distinction (Kinematic $\frac{dv}{dt}$ vs Chassis Specific Force)**:
   - During extreme braking in Vta02, CAN accelerometer reads $-5.78\text{ m/s}^2$, whereas SG9 reads $-4.98\text{ m/s}^2$.
   - **Why?** The CAN sensor is a physical accelerometer bolted to the vehicle floorboard. During hard braking, the chassis dives forward (nose down, pitch angle $\theta < 0$), tilting Earth's gravity vector into the forward axis as $+g \sin(-\theta) \approx -g \theta$, which increases the measured negative specific force.
   - Conversely, SG9 is the derivative of true VBOX Doppler ground speed ($dv_{\text{VBOX}}/dt$) and is immune to gravitational tilt.
   - This physical difference provides compelling independent confirmation of chassis pitch dynamics.

---

## 3. Methodological Clarifications: Causality & Reconstruction

### 3.1 Non-Causal Offline Reference vs. Strictly Causal Online Deployment

The Savitzky-Golay filter ($W=9\text{ samples} = 0.8\text{ s}, p=2$) is a **centered, non-causal operator** that utilizes $4\text{ samples}$ ($0.4\text{ s}$) of future context to fit a smooth local parabola.
- **Offline Supervision (Permissible)**: For generating ground-truth training labels, accessing the complete mission trajectory is standard, valid practice to eliminate noise and phase distortion.
- **Online Navigation (Strictly Forbidden)**: In real-time vehicle deployment, future data cannot exist. The smartphone navigation engine uses strictly causal trailing windows ($j \le k$) of smartphone IMU data only.

```text
    OFFLINE LABEL GENERATION (C5.3-A)
    ────────────────────────────────
    VBOX Doppler Speed v_VBOX(t)
                │
                ▼
    Non-Causal SG9 Derivative Filter (±0.4 s context)
                │
                ▼
    VBOX-Derived Acceleration Reference a_reference(t)
                │
                ▼
    Supervision Target: r_train(t) = a_x^level(t) - a_reference(t)
    
    
    ONLINE DEPLOYMENT (C5.3-B)
    ──────────────────────────
    Raw Smartphone IMU [ax, ay, az, gx, gy, gz]
                │
                ▼
    Strictly Causal Trailing Window (past & current samples j <= k)
                │
                ▼
    Trained AI Residual Predictor
                │
                ▼
    Estimated Residual r_hat(t)
                │
                ▼
    Corrected Specific Force: a_corrected(t) = a_x^level(t) - r_hat(t)
    (Zero VBOX, zero future samples, zero external aiding)
```

### 3.2 Velocity Reconstruction Consistency

Integrating $a_{\text{reference}}(t)$ over the entire 18.3-minute ($1099.0\text{ s}$, $11.05\text{ km}$) journey reconstructs VBOX speed with an MAE of **$0.1022\text{ m/s}$** ($0.37\text{ km/h}$).
- **Correct Interpretation**: This metric confirms that the derivative/integrator pair is mathematically conservative and does not introduce numerical energy drift over long durations.
- **Caveat**: Because $a_{\text{reference}}$ was differentiated from $v_{\text{VBOX}}$, recovering $v_{\text{VBOX}}$ via numerical integration is a consistency check of the filtering operator, not an independent proof of absolute physical truth. Independent cross-validation is instead provided by the CAN bus accelerometer audit in Section 2.

---

## 4. Canonical Training Residual Characterization

With $a_{\text{reference}}(t) = a_{\text{SG9}}(t)$, the target residual is formulated as:
$$r_{\text{train}}(t) \triangleq a_x^{\text{level}}(t) - a_{\text{reference}}(t)$$

### 4.1 Quantitative Residual Distribution Across Trips

| Dataset | Total Samples | Mean Bias ($\bar{r}$) | Dynamic Std ($\sigma_r$) | Dynamic Variance % | 1st Percentile ($p_1$) | 5th Percentile ($p_5$) | Median ($p_{50}$) | 95th Percentile ($p_{95}$) | 99th Percentile ($p_{99}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vta02** (Train/Val) | 10,991 | **$+0.2009\text{ m/s}^2$** | **$1.7314\text{ m/s}^2$** | **$98.67\%$** | $-4.35\text{ m/s}^2$ | $-2.58\text{ m/s}^2$ | $+0.20\text{ m/s}^2$ | $+2.88\text{ m/s}^2$ | $+4.46\text{ m/s}^2$ |
| **Vta03** (Train/Val) | 645 | **$-0.5056\text{ m/s}^2$** | **$1.4393\text{ m/s}^2$** | **$89.02\%$** | $-3.67\text{ m/s}^2$ | $-2.85\text{ m/s}^2$ | $-0.46\text{ m/s}^2$ | $+1.75\text{ m/s}^2$ | $+2.75\text{ m/s}^2$ |
| **Vta04** (Test) | 1,789 | **$+0.2200\text{ m/s}^2$** | **$1.9837\text{ m/s}^2$** | **$98.78\%$** | $-5.36\text{ m/s}^2$ | $-2.98\text{ m/s}^2$ | $+0.32\text{ m/s}^2$ | $+3.15\text{ m/s}^2$ | $+5.08\text{ m/s}^2$ |

### 4.2 Physical Meaning of the Residual

The residual $r_{\text{train}}(t)$ is defined as **the remaining time-varying error between the smartphone-derived acceleration representation and the VBOX-derived acceleration reference**. It is an aggregate of several underlying physical and instrumentation components:
$$r_a(t) = \underbrace{b_a(t)}_{\text{MEMS bias}} + \underbrace{g \sin \theta_{\text{pitch}}(t)}_{\text{dynamic chassis tilt}} + \underbrace{g \sin \theta_{\text{grade}}(t)}_{\text{road slope}} + \underbrace{\epsilon_{\text{mount}}(t)}_{\text{cradle compliance}} + \underbrace{\delta_{\text{scale}} a_{\text{true}}(t)}_{\text{scale factor}} + \underbrace{w_a(t)}_{\text{sensor noise}} + \underbrace{\epsilon_{\text{ref}}(t)}_{\text{reference approx}}$$

- **Dynamic Tilt Dominance**: Because $a_x^{\text{level}}$ is leveled using static initial gravity orientation, dynamic chassis pitch $\theta(t)$ projects gravity into the forward axis as $g \sin \theta \approx g \theta$. A modest $3^\circ$ pitch rotation injects $\approx 0.51\text{ m/s}^2$ of false acceleration.
- **The AI's Role**: The machine learning model in Stage C5.3 does not need to explicitly disentangle these individual physical mechanisms. Its task is to predict the **effective time-varying correction $\hat{r}_a(t)$ required by the inertial navigation system** using causal smartphone IMU features.

---

## 5. Formal Programmatic Leakage Boundary Audit

Automated programmatic assertions were executed on all generated canonical datasets:

| Audit Check | Vta02 Status | Vta03 Status | Vta04 Status | Verification Details |
|---|:---:|:---:|:---:|---|
| **VBOX Speed in Features** | **PASSED** (False) | **PASSED** (False) | **PASSED** (False) | No column containing `'vbox'` exists in the feature partition. |
| **Reference Accel in Features** | **PASSED** (False) | **PASSED** (False) | **PASSED** (False) | `a_reference` is strictly prefixed with `LABEL_`. |
| **Target Residual in Features** | **PASSED** (False) | **PASSED** (False) | **PASSED** (False) | `target_residual` exists strictly in the supervision partition. |
| **GNSS Speed in Features** | **PASSED** (False) | **PASSED** (False) | **PASSED** (False) | Zero GPS or CAN speed variables are available to input space. |
| **Strict Input Partition** | **VERIFIED** | **VERIFIED** | **VERIFIED** | Features are strictly: `ax_phone, ay_phone, az_phone, gx_phone, gy_phone, gz_phone, ax_level`. |

### Canonical Datasets Generated
- `data/processed/c5_3_labels_vta02.csv` (10,991 samples, urban training)
- `data/processed/c5_3_labels_vta03.csv` (645 samples, mixed validation)
- `data/processed/c5_3_labels_vta04.csv` (1,789 samples, highway test)

---

## 6. Diagnostic Figures

Six diagnostic figures are available in the artifact directory:

1. **CAN vs. SG9 Cross-Audit**: `c5_3a_can_vs_sg9_audit.png`
   - Braking transient overlay, scatter regression ($r=0.930$), cross-correlation lag ($\tau=0.0\text{ s}$), and difference distribution.
2. **Differentiation Transient Overlay**: `c5_3a_differentiation_comparison.png`
   - Shows VBOX speed, raw differencing, Butterworth, Savitzky-Golay, and Gaussian candidates during the Stop 3 deceleration transient in Vta02.
3. **Stationary Noise Analysis**: `c5_3a_stationary_noise_analysis.png`
   - Bar charts comparing standard deviation and worst-case acceleration spikes during verified vehicle stops.
4. **Power Spectral Density & Jerk Energy**: `c5_3a_spectral_density_jerk.png`
   - Frequency roll-off curves and mean squared jerk comparison confirming $96.1\%$ jitter reduction.
5. **Velocity Reconstruction Consistency**: `c5_3a_velocity_reconstruction.png`
   - Cumulative velocity reconstruction over 18.3 minutes, confirming mathematical consistency of the filter pair.
6. **Canonical Residual Distributions**: `c5_3a_canonical_residual_distribution.png`
   - Probability density functions and cross-trip boxplots for the target residual.

---

## 7. Staged Roadmap for Stage C5.3-B

In accordance with the one-change-at-a-time experimental protocol, Stage C5.3-B will proceed through incremental, controlled sub-stages rather than training multiple complex models simultaneously:

1. **C5.3-B0 — Causal Feature Pipeline & Timing Sanity**:
   - Establish strictly causal windowing ($W \in \{10, 20\text{ samples}\} = 1.0\text{--}2.0\text{ s}$, past and current samples only).
   - Validate timestamp alignment, window boundaries, and zero-leakage isolation.
   - Fixed split: **Train on Vta02**, **Validate on Vta03**, **Final Test on Vta04 (untouched)**.
2. **C5.3-B1 — Linear Baseline (Ridge Regression)**:
   - Fit a simple regularized linear model: $\hat{r}_a = \mathbf{w}^T \mathbf{x} + b$.
   - Determine whether the residual has a predictable linear relationship with causal IMU statistics.
3. **C5.3-B2 — Random Forest Regressor**:
   - Non-linear tree ensemble baseline.
4. **C5.3-B3 — Gradient Boosted Decision Trees (GBDT)**.
5. **C5.3-B4 — Compact Multi-Layer Perceptron (MLP)**.

### Primary Evaluation Metric: Integrated Navigation Drift
Performance will not be judged solely on ML loss (MAE of residual). The decisive metric is **integrated navigation drift over outage horizons**:

$$a_{\text{corrected}}(t) = a_x^{\text{level}}(t) - \hat{r}_a(t)$$
$$v(t_k) = v(t_{k-1}) + a_{\text{corrected}}(t_k) \Delta t, \qquad p(t_k) = p(t_{k-1}) + v(t_k) \Delta t$$

Evaluated across $H \in \{5, 10, 20, 30, 60\}\text{ s}$ outages against the uncorrected kinematic baseline:
$$\text{Drift Reduction} = \frac{\epsilon_{p,\text{baseline}} - \epsilon_{p,\text{AI}}}{\epsilon_{p,\text{baseline}}} \times 100\%$$
