# Stage C5.1: Standalone AI Forward-Speed Estimator Benchmark

## Executive Summary
This document summarizes the results of the strictly isolated Stage C5.1 benchmark evaluating whether smartphone MEMS IMU features can predict vehicle forward speed across different trips without data leakage.

* **Training Set**: `Vta02` (10,972 causal windows, ~18.3 min driving)
* **Testing Set**: `Vta04` (1,770 causal windows, ~3.0 min driving)
* **Feature Window**: Strictly causal $W = 20$ samples (2.0 s at 10 Hz), stride = 1 sample (0.1 s).
* **Sensor Channels**: 8 channels (`acc_x, acc_y, acc_z, acc_mag, gyro_x, gyro_y, gyro_z, gyro_mag`) $\times$ 9 statistics = 72 features.

---

## 1. Benchmark Results Table

| Model | Train | Test | MAE m/s | RMSE m/s | MAE km/h | RMSE km/h | Max Err (m/s) | Bias m/s | R² |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Mean** | Vta02 | Vta04 | **2.07** | **2.55** | **7.44** | **9.17** | 8.41 | -1.11 | -0.232 |
| **Model A (Random Forest)** | Vta02 | Vta04 | 2.24 | 2.84 | 8.08 | 10.22 | 7.83 | +0.92 | -0.531 |
| **Model B (Gradient Boosting)** | Vta02 | Vta04 | **2.07** | **2.62** | **7.45** | **9.44** | 8.97 | +0.65 | -0.307 |
| **Model C (Small MLP)** | Vta02 | Vta04 | 2.73 | 3.44 | 9.82 | 12.39 | 11.74 | +1.37 | -1.250 |
| **Previous Model (W=10 baseline)** | Vta02 | Vta04 | 2.21 | 2.82 | 7.95 | 10.14 | 9.50 | -0.69 | -0.502 |

---

## 2. Speed-Regime Performance Breakdown (Model B: Gradient Boosting)

| Speed Bin | Sample Count | MAE (m/s) | RMSE (m/s) | MAE (km/h) | Bias (m/s) | Performance Assessment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **0 - 5 m/s (0 - 18 km/h)** | 81 | 2.17 | 2.45 | 7.80 | +1.61 | Over-predicts at low speeds |
| **5 - 10 m/s (18 - 36 km/h)** | 189 | 3.22 | 3.87 | 11.60 | +2.88 | Strong positive bias during deceleration |
| **10 - 15 m/s (36 - 54 km/h)** | 1500 | 1.92 | 2.43 | 6.91 | +0.32 | Cruising speed tracking (~2 m/s MAE) |
| **> 15 m/s (> 54 km/h)** | 0 | N/A | N/A | N/A | N/A | No high-speed samples on Vta04 |

---

## 3. Key Findings & Diagnostic Observations

1. **Failure to Outperform Trivial Baseline**: The tested IMU-only feature representations and model configurations did not achieve reliable cross-trip speed prediction on Vta04 when trained exclusively on Vta02 (RMSE $2.55\text{ m/s}$ vs $2.62 - 3.44\text{ m/s}$).
2. **Negative R² Score ($R^2 = -0.307 \text{ to } -1.250$)**: Because `Vta04` has a narrow speed variance around $11.17\text{ m/s}$ ($\text{Var}(y) = 5.26\text{ m}^2/\text{s}^2$), the cross-trip MSE ($\approx 6.8\text{ m}^2/\text{s}^2$) exceeds the true trajectory variance. The model learned an implicit cruising speed prior rather than instantaneous velocity dynamics.
3. **Deceleration / Stop Pathology**: When the vehicle slows down, road vibration decays slowly or engine idle vibration maintains a spurious floor, causing the model to overpredict speed by $+1.6\text{ m/s}$ to $+2.9\text{ m/s}$. Pure vibration features lack unambiguous directional velocity information during braking vs cruising.
4. **Kinematic Link vs Feature Representation**: The direct kinematic link exists ($\dot{v}_x \approx a_x$), but reliable long-term integration requires accurate frame alignment, bias estimation, gravity compensation, and disturbance handling. The current statistical window features fail to exploit this.
5. **Data Leakage Confirmation**: Zero leakage was confirmed across all 10 checklist tests. The previous report's claim of $1.64\text{ m/s}$ MAE was achieved by applying a post-hoc EMA filter, whereas raw cross-trip model MAE is $2.07 - 2.24\text{ m/s}$.

---

## 4. Locked Project Conclusion
> *"Under a strict Vta02→Vta04 cross-trip evaluation, the tested smartphone-IMU-only statistical and neural speed estimators did not outperform a trivial training-set mean predictor. This indicates substantial cross-trip/domain dependence and insufficient extraction of instantaneous longitudinal speed information from the current feature representation. The result does not establish that smartphone IMU speed estimation is impossible; it establishes that the present representation and training configuration are inadequate for reliable cross-trip longitudinal velocity estimation."*
