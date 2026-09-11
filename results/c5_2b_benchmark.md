# Stage C5.2-B: Multi-Trip Training Evaluation Benchmark Report
## Single-Trip ($Vta02 \to Vta04$) vs. Multi-Trip ($Vta02 + Vta03 \to Vta04$)

---

### Executive Summary
Stage C5.2-B tests whether expanding the training distribution from single-trip (`Vta02`) to multi-trip (`Vta02 + Vta03`) improves cross-trip forward velocity prediction on `Vta04`.

* **Training Set A (Single-Trip)**: `Vta02` (10,972 causal windows, ~18.3 min driving)
* **Training Set B (Multi-Trip)**: `Vta02 + Vta03` ($10,972 + 626 = 11,598$ causal windows, ~19.4 min driving)
* **Testing Set**: `Vta04` (1,770 causal windows, ~3.0 min driving)
* **Feature Representation**: Frozen 72 causal features ($W=20$ samples / 2.0 s, stride = 1 sample / 0.1 s).
* **Target**: Scalar VBOX vehicle speed magnitude (`veh_speed_ms`).
* **Zero Leakage**: Strict trip separation. `Vta04` is never seen during training or scaling.

---

### 1. Domain Distribution Diagnostic Audit: $P(Vta02), P(Vta03), P(Vta04)$

Before benchmarking, individual kinematic distributions were audited across all three journeys to determine whether `Vta03` bridges the domain gap between `Vta02` and `Vta04`:

| Kinematic Metric | Trip Vta02 (Train 1) | Trip Vta03 (Train 2) | Trip Vta04 (Frozen Test) | Domain Comparison Assessment |
|---|:---:|:---:|:---:|---|
| **Sample Count / Duration** | 10,991 pts (1,099.0 s) | 645 pts (64.4 s) | 1,789 pts (178.8 s) | Vta03 is a short trip (~1.1 min) |
| **Mean Speed ($v$)** | **$10.05\text{ m/s}$** ($36.2\text{ km/h}$) | **$5.83\text{ m/s}$** ($21.0\text{ km/h}$) | **$11.14\text{ m/s}$** ($40.1\text{ km/h}$) | Vta03 has substantially lower speed |
| **Speed Variance ($\text{Var}(v)$)** | $26.26\text{ m}^2/\text{s}^2$ | $18.82\text{ m}^2/\text{s}^2$ | $5.28\text{ m}^2/\text{s}^2$ | Vta04 has narrow cruising variance |
| **Speed Range $[v_{\min}, v_{\max}]$** | $[0.00, 22.68]\text{ m/s}$ | $[0.01, 12.71]\text{ m/s}$ | $[1.65, 14.36]\text{ m/s}$ | Vta03 tops out at 12.7 m/s |
| **Speed Median** | $10.23\text{ m/s}$ | $6.64\text{ m/s}$ | $11.60\text{ m/s}$ | Vta04 cruising speed is double Vta03 |
| **Mean Horiz Accel ($\|\mathbf{a}_h\|$)** | $2.43\text{ m/s}^2$ | **$1.30\text{ m/s}^2$** | **$3.11\text{ m/s}^2$** | Vta03 has lowest vibration energy |
| **Mean Gyro Norm ($\|\boldsymbol{\omega}\|$)** | $0.275\text{ rad/s}$ | **$0.156\text{ rad/s}$** | **$0.357\text{ rad/s}$** | Vta04 has highest angular motion |

#### Diagnostic Distribution Finding:
`Vta03` does **not** bridge the domain gap between `Vta02` and `Vta04`. Instead, it occupies a distinct lower-speed, lower-vibration regime (mean speed $5.83\text{ m/s}$, horizontal acceleration $1.30\text{ m/s}^2$) compared to `Vta04` (mean speed $11.14\text{ m/s}$, horizontal acceleration $3.11\text{ m/s}^2$).
Combining `Vta03` with `Vta02` pulls the pooled training speed mean **downward** from $10.06\text{ m/s} \to 9.84\text{ m/s}$, shifting the training prior *further away* from `Vta04`'s test distribution.

---

### 2. Primary Benchmark Results (Full Trip Vta04, 1,770 Windows)

| Training Regime | Model | MAE (m/s) | RMSE (m/s) | $R^2$ | Bias (m/s) |
|---|---|---:|---:|---:|---:|
| **Single-Trip (Vta02)** | Baseline Mean | 2.07 | 2.55 | -0.232 | -1.11 |
| **Multi-Trip (Vta02+03)** | Baseline Mean | 2.22 | 2.65 | -0.335 | -1.33 |
| *Delta (Multi - Single)* | | *+0.16* | *+0.10* | *-0.103* | *-0.22* |
| **Single-Trip (Vta02)** | Random Forest | 2.24 | 2.84 | -0.531 | +0.92 |
| **Multi-Trip (Vta02+03)** | Random Forest | 2.28 | 2.83 | -0.521 | +0.60 |
| *Delta (Multi - Single)* | | *+0.04* | *-0.01* | *+0.010* | *-0.32* |
| **Single-Trip (Vta02)** | Gradient Boosting | 2.10 | 2.61 | -0.296 | +0.78 |
| **Multi-Trip (Vta02+03)** | Gradient Boosting | 2.22 | 2.71 | -0.394 | +0.31 |
| *Delta (Multi - Single)* | | *+0.12* | *+0.10* | *-0.098* | *-0.47* |

---

### 3. Rough-Road Diagnostic Results ($t = 29.6\text{--}33.2\text{ s}$, 36 Windows)

| Model | Single-Trip MAE (m/s) | Multi-Trip MAE (m/s) | $\Delta\text{MAE}$ (m/s) | Single-Trip RMSE (m/s) | Multi-Trip RMSE (m/s) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Mean** | 1.23 | 1.45 | +0.22 | 1.24 | 1.46 |
| **Random Forest** | 3.18 | 3.17 | -0.01 | 3.51 | 3.42 |
| **Gradient Boosting** | 2.98 | 2.96 | -0.02 | 3.23 | 3.28 |

---

### 4. Scientific Interpretation & Locked Project Conclusion

1. **Multi-Trip Training Outcome**:
   Adding `Vta03` to the training set did not improve cross-trip speed prediction on `Vta04`; MAE changed by $+0.04\text{ m/s}$ (RF) and $+0.12\text{ m/s}$ (GBDT). All models continue to exhibit negative $R^2$ scores ($-0.394$ to $-0.521$) and remain inferior to the single-trip mean prior.
2. **Mechanism of the Performance Shift**:
   - As revealed by the domain distribution audit, `Vta03` is a short urban trip ($5.83\text{ m/s}$ mean) with low vibration energy ($1.30\text{ m/s}^2$ horizontal norm), whereas `Vta04` is a higher-speed cruise ($11.14\text{ m/s}$ mean) with higher vibration energy ($3.11\text{ m/s}^2$).
   - The reduction in positive prediction bias (from $+0.78\text{ m/s} \to +0.31\text{ m/s}$ on GBDT and $+0.92\text{ m/s} \to +0.60\text{ m/s}$ on RF) is consistent with the lower-speed Vta03 samples shifting the learned prediction distribution, without producing improved instantaneous velocity estimation.
3. **Rough-Road Segment Observation**:
   During the rough-road disturbance ($29.6\text{--}33.2\text{ s}$), multi-trip training produced essentially zero change in MAE ($3.18 \to 3.17\text{ m/s}$ on RF, $2.98 \to 2.96\text{ m/s}$ on GBDT). This particular additional training trip did not provide sufficient information to improve robustness to this disturbance.

---

### 5. Locked C5.2-B Project Conclusion

> *"C5.2-B demonstrates that naive multi-trip pooling of Vta02 and Vta03 does not improve cross-trip speed prediction on Vta04. Although Vta03 reduces the positive prediction bias, its distinct lower-speed and lower-vibration distribution does not provide sufficient complementary coverage of the Vta04 operating regime to improve overall accuracy. Therefore, the failure cannot be resolved by simply adding this additional trip under the current windowed statistical feature representation.*
>
> *This result does not establish that additional training data are universally insufficient; rather, it shows that adding the specific Vta03 distribution is insufficient."*

---

### 6. Next Controlled Step: Stage C5.2-C0 (Longitudinal-Frame / Kinematic Integrity Audit)

Before integrating acceleration into velocity, we must rigorously audit the physical relationship between smartphone IMU longitudinal acceleration and actual vehicle acceleration:

> **Stage C5.2-C0: Longitudinal-Frame Kinematic Integrity Audit**
> * Verify $\mathbf{a}^p \to \mathbf{a}^v$ across all trips (`Vta02`, `Vta03`, `Vta04`).
> * Specifically determine whether the resulting $a_x^v$ exhibits the expected physical correlation, sign consistency, and scale with $\frac{dv_{\text{VBOX}}}{dt}$ across different candidate mounting transforms (e.g. Identity, leveled, $R_{pv}$).
> * Use VBOX acceleration **solely as an offline diagnostic reference** (never as an estimator feature).
> * Proceed to pure kinematic integration (Stage C5.2-C1) only after the longitudinal coordinate alignment is physically verified.
