# Models Directory Guide

This directory stores deployable ONNX models, feature normalizers, and model specifications used across the SIH26168 navigation engine.

---

## 1. Production Edge Models

| File Name | Architecture | Input Shape | Parameter Count | File Size | Primary Role |
|:---|:---|:---:|:---:|:---:|:---|
| **`tcn_kin_speed.onnx`** | Dilated 1D TCN | `(batch, 100, 9)` | **65,409 weights** | **257.6 KB** | Active production forward-speed estimator for Android engine. |
| **`tcn_scaler.json`** | StandardScaler | 9 channels | — | **743 B** | Feature means and standard deviations fitted on 39 training trips. |
| **`balanced_tcn_speed.onnx`** | Dilated 1D TCN | `(batch, 100, 9)` | **65,409 weights** | **43.1 KB (+ .data)** | Target-balanced model trained with inverse-density loss weighting. |
| **`rf_speed_model.json`** | Random Forest | Static features | — | **281.8 KB** | Baseline Random Forest model specification. |

---

## 2. Input Feature Order (`tcn_scaler.json`)

The Expanded TCN expects a 9-channel standardized input sequence:
1. `lin_acc_x`: Lateral linear acceleration ($\text{m/s}^2$)
2. `lin_acc_y`: Longitudinal linear acceleration ($\text{m/s}^2$)
3. `lin_acc_z`: Vertical linear acceleration ($\text{m/s}^2$)
4. `gyro_yaw`: Calibrated yaw rate ($\text{rad/s}$)
5. `gyro_pitch`: Calibrated pitch rate ($\text{rad/s}$)
6. `gyro_roll`: Calibrated roll rate ($\text{rad/s}$)
7. `a_horiz`: Total horizontal acceleration ($\text{m/s}^2$)
8. `a_vert`: Gravity-aligned vertical acceleration ($\text{m/s}^2$)
9. `kappa`: Dynamic curvature proxy ($\|\boldsymbol{\omega}\| \cdot a_{\text{horiz}}$)

---

## 3. Heavy Checkpoint Archival Notice

Large historical Scikit-Learn `.joblib` binary checkpoints (>150 MB total from exploratory Stage C5) are excluded from git tracking via `.gitignore` to keep the repository lightweight. Their metadata and configuration parameters are preserved.
