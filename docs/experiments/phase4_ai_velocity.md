# Phase 4: AI Forward Velocity Estimation & The Expanded TCN

## 1. Context & Motivation
Having proven that:
1. Pure IMU integration diverges quadratically ($t^2$),
2. ESKF bounds attitude and calibrates pre-outage bias, and
3. Non-Holonomic Constraints (NHC) bound lateral and vertical drift but leave **along-track speed unobservable**,

we turned to **machine learning** to estimate forward vehicle speed directly from smartphone IMU dynamics.

---

## 2. Model Evolution: From Trees to Temporal Convolutions

### 2.1 The Failure of Tree-Based Baselines (Stage C5)
* Initial experiments tested **Ridge Regression**, **Random Forest (RF)**, and **Gradient Boosted Decision Trees (GBDT)**.
* **Failure Mode**: When evaluated on untouched trips from different drivers or vehicles, Random Forests exhibited severe cross-trip generalization collapse (MAE exploded from $1.8\text{ m/s} \to 5.5\text{ m/s}$). Tree models memorized vehicle-specific vibration and pitch distributions rather than true temporal kinematics.

### 2.2 The Dilated Temporal Convolutional Network (Expanded TCN)
To capture temporal dynamics without recurrence bottlenecks, we implemented a **Dilated TCN** (`src/ml/models/temporal_speed_net.py`):
* **Parameters**: 65,409 weights (FP32 size: 262 KB, edge-deployable via ONNX Runtime).
* **Receptive Field**: 100 timesteps at 10.0 Hz = **10.0 seconds** of temporal context.
* **Dilation Schedule**: Exponential causal dilation rates $d \in \{1, 2, 4, 8, 16, 32\}$ with kernel size $k = 3$.
* **Input Features (9 Channels)**:
  1. $a_x$: Lateral linear acceleration
  2. $a_y$: Longitudinal linear acceleration
  3. $a_z$: Vertical linear acceleration
  4. $\omega_{\text{yaw}}$: Angular yaw rate
  5. $\omega_{\text{pitch}}$: Angular pitch rate
  6. $\omega_{\text{roll}}$: Angular roll rate
  7. $a_{\text{horiz}} = \sqrt{a_x^2 + a_y^2}$: Total horizontal acceleration
  8. $a_{\text{vert}}$: Gravity-aligned vertical acceleration
  9. $\kappa = \|\boldsymbol{\omega}\| \cdot a_{\text{horiz}}$: Dynamic curvature proxy

---

## 3. Leakage-Proof Training Protocol

* **Training Set**: 39 whole trips from IO-VNBD (over 450,000 timesteps).
* **Test Set**: 19 completely held-out trips (zero overlap in time, driver, or vehicle trip).
* **Standardization**: Feature means and standard deviations computed **strictly on the 39 training trips** and frozen into `models/tcn_scaler.json`.

---

## 4. Empirical Evaluation on 19 Held-Out Test Trips

The Expanded TCN was benchmarked across all 19 held-out test trips without retraining:

| Road Category | Test Trips | Speed MAE (m/s) | Speed RMSE (m/s) | Bias (m/s) | 60s Drift (m) | 60s Drift (%) | SIH <10% Passes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Winding Mountain (Vw)** | 6 | **2.85 m/s** | 3.59 m/s | -1.15 m/s | **87.7 m** | **7.42%** | **3 / 6 (50.0%)** |
| **Suburban Town (Vta)** | 8 | **2.59 m/s** | 3.65 m/s | +0.96 m/s | **87.2 m** | **23.51%** | **1 / 8 (12.5%)** |
| **Dense Urban (Vtb)** | 4 | **3.31 m/s** | 3.77 m/s | -1.53 m/s | **108.7 m** | **20.25%** | **0 / 4 (0.0%)** |
| **Motorway (Vf)** | 1 (`V-Vfa02`) | **3.41 m/s** | 4.47 m/s | -2.22 m/s | **163.9 m** | **11.22%** | **0 / 1 (0.0%)** |
| **Aggregate (All 19 Trips)** | **19** | **2.88 m/s** | **3.81 m/s** | **-0.54 m/s** | **93.0 m** | **14.20%** | **4 / 19 (21.1%)** |

---

## 5. Critical Discoveries & Unresolved Bottlenecks

1. **The Mountain Driving Success**:
   - On mountain roads (`Vw12`, `Vw14a`), lateral acceleration and yaw rates are rich ($a_y = v^2/R$). The TCN achieved **5.87% and 6.94% drift at 60 s**, easily beating the SIH <10% target.
2. **The Motorway Cruising Ceiling (~85 km/h)**:
   - On the straight motorway (`V-Vfa02`), actual speeds averaged 90–115 km/h.
   - However, the TCN predictions asymptotically plateaued at **23.5–24.5 m/s (~85–88 km/h)**, resulting in a persistent $-3.45\text{ m/s}$ negative bias and **11.22% drift** (missing the 10% threshold).
3. **The 93-Meter Unweighted Caveat**:
   - The headline **93.0 m mean drift** was unweighted across trips. Distance-weighted drift was **148 meters**, showing that on high-speed roads, drift remains significant.
