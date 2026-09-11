# Architectural Decision Record: Accepted Approaches

This document details the validated core algorithms, architectural decisions, and safeguards that are **formally accepted** into the SIH26168 Intelligent Dead Reckoning engine.

---

## 1. 15-State Error-State Kalman Filter (ESKF) Formulation

* **Decision**: Separate navigation propagation into a high-rate non-linear nominal state ($\mathbf{p}, \mathbf{v}, \mathbf{q}$) and a linearized 15-state error state ($\delta \mathbf{p}, \delta \mathbf{v}, \delta \boldsymbol{\theta}, \delta \mathbf{b}_a, \delta \mathbf{b}_g$).
* **Rationale**: Avoids quaternion singularity issues, guarantees numerical stability, and allows continuous online tracking and subtraction of accelerometer and gyroscope biases.
* **Empirical Validation**: Bounded attitude drift to $<2.5^\circ$ and reduced short-outage drift by $4.2\times$ over raw integration.
* **Primary Implementation**: `src/navigation/eskf.py` and `android/app/src/main/java/com/sih26168/idr/navigation/ESKF.java`.

---

## 2. Causal Phone-to-Vehicle Frame Alignment (PCA + Gravity)

* **Decision**: Align the arbitrary phone mounting orientation to the vehicle body frame using static gravity decomposition for pitch/roll and longitudinal acceleration PCA for yaw.
* **Rationale**: Prevents forward vehicle acceleration from leaking into lateral Non-Holonomic Constraints (NHC), eliminating artificial circular swerves.
* **Empirical Validation**: Reduced cross-track drift by **64.8%** during blackout benchmarks.
* **Primary Implementation**: `src/preprocessing/alignment.py`.

---

## 3. Dilated Temporal Convolutional Network (Expanded TCN)

* **Decision**: Implement a 65,409-parameter 1D Dilated TCN with causal exponential dilation rates ($d \in \{1, 2, 4, 8, 16, 32\}$) over a 10.0-second rolling window.
* **Rationale**: Outperforms tree-based models across unseen drivers and vehicles; captures temporal dynamics without the vanishing gradient or latency bottlenecks of recurrent architectures (RNN/LSTM).
* **Empirical Validation**: Achieved **2.88 m/s mean MAE** across 19 completely held-out test trips, with mountain road drift reaching **3.51%** at 60 seconds.
* **Primary Implementation**: `src/ml/models/temporal_speed_net.py`, exported to `models/tcn_kin_speed.onnx`.

---

## 4. Multi-Condition Zero Velocity Detection (ZVD)

* **Decision**: Clamp estimated forward velocity to $0.0\text{ m/s}$ when and only when all three stationary conditions are met causally:
  1. Gyroscope magnitude: $\|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$
  2. Acceleration variance: $\sigma_a < 0.16\text{ m/s}^2$
  3. Total specific force magnitude: $|\|\mathbf{a}\| - 9.81| < 0.28\text{ m/s}^2$
* **Rationale**: Completely prevents the quadratic integration of residual accelerometer bias during stoplights and traffic congestion.
* **Empirical Validation**: Suppressed catastrophic urban divergence on `Vta26` from **574.6% down to 47.3%**.
* **Primary Implementation**: `src/fusion/zero_velocity.py`.

---

## 5. Input $\pm 3.5\sigma$ Gating & Kinematic OOD Bounds

* **Decision**: Clamp all sensor feature inputs to $\pm 3.5$ standard deviations of the vehicle training set and enforce a maximum acceleration delta constraint ($\Delta v \le 6.0\text{ m/s}^2 \cdot \Delta t$).
* **Rationale**: Protects neural inference from extreme out-of-distribution rotational spikes caused by handling the phone or human movement.
* **Empirical Validation**: Prevented the ~71 m/s velocity spike observed during campus pedestrian tests.
* **Primary Implementation**: `src/ml/inference.py` and `android/app/src/main/java/com/sih26168/idr/ml/SpeedEstimator.java`.

---

## 6. Whole-Trip Leakage-Proof Splitting Protocol

* **Decision**: Split the IO-VNBD dataset strictly at the whole-trip level (39 trips training, 19 trips testing), ensuring that no trip, driver, or time segment from the test set is ever seen during feature normalization or model training.
* **Rationale**: Eliminates false benchmark inflation caused by temporal window correlation.
* **Primary Implementation**: `tests/test_causality_and_leakage.py` and `src/data/dataset_builder.py`.
