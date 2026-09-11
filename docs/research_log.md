# Project Research Log: Chronological Timeline

A complete chronological record of the engineering investigations, hypotheses, experimental results, and architectural decisions conducted throughout the SIH26168 project.

---

### [2026-09-02] — Phase 0: Ground Truth & Label Calibration
* **Question**: Is vehicle CAN bus speed authoritative ground truth, and does it match wheel encoder ticks?
* **What We Tried**: Cross-correlation of raw transmission wheel encoder ticks, CAN bus speed, and GPS Doppler velocity.
* **Result**: Dynamic tire radius calibrated to $r_{\text{eff}} = 0.312\text{ m}$. CAN speed exhibited a constant $100\text{ ms}$ digital filtering latency relative to raw wheel ticks.
* **What We Learned**: CAN speed is a filtered proxy, not instantaneous truth.
* **Decision**: Applied $-100\text{ ms}$ time shift to all training labels to maintain phase alignment with 100 Hz smartphone IMU.
* **Next Question**: Can raw smartphone IMU integration maintain position during a GNSS blackout?

---

### [2026-09-03] — Phase 1: Pure Inertial Strapdown Baseline
* **Question**: How far does a vehicle drift if we integrate smartphone accelerometer and gyroscope measurements directly?
* **What We Tried**: 3D strapdown mechanization in the East-North-Up frame using raw smartphone linear acceleration and angular velocity.
* **Result**: Accelerometer bias produced quadratic position error ($\frac{1}{2} b_a t^2$). 60-second drift exceeded $800\text{--}1,500\%$ of distance ($>650\text{ m}$ drift).
* **What We Learned**: Standalone double integration on consumer smartphone sensors is mathematically unviable for intervals $>15\text{ s}$.
* **Decision**: REJECTED pure IMU integration as a complete solution.
* **Next Question**: Can an Error-State Kalman Filter track sensor biases and bound attitude drift?

---

### [2026-09-04] — Phase 2: 15-State Error-State Kalman Filtering
* **Question**: Can closed-loop Kalman estimation of accelerometer and gyroscope biases bound drift?
* **What We Tried**: Implemented a 15-state ESKF ($\delta \mathbf{p}, \delta \mathbf{v}, \delta \boldsymbol{\theta}, \delta \mathbf{b}_a, \delta \mathbf{b}_g$) receiving 1 Hz GNSS fixes during open-sky driving.
* **Result**: Pre-outage bias calibration reduced 10s drift from $28.4\text{ m} \to 6.8\text{ m}$ ($4.2\times$ improvement) and constrained attitude drift to $<1.8^\circ$. However, during 60s blackouts without measurement updates, along-track position error continued to drift by $200–380\text{ m}$.
* **What We Learned**: Filtering bounds attitude and calibrates bias, but along-track velocity remains unobserved during outages.
* **Decision**: ACCEPTED ESKF as core estimator; required an independent forward velocity estimator.
* **Next Question**: Can vehicle Non-Holonomic Constraints (NHC) eliminate drift?

---

### [2026-09-05] — Phase 3: Non-Holonomic Constraints & Mounting Dynamics
* **Question**: Does enforcing zero lateral and vertical velocity in the vehicle chassis frame prevent position divergence?
* **What We Tried**: Projected velocities into vehicle frame and applied measurement updates $v_y^v = 0, v_z^v = 0$.
* **Result**: Cross-track drift was reduced by **64.8%** ($165.8\text{ m} \to 58.3\text{ m}$). However, along-track forward velocity drift was completely unaffected (<5% change).
* **What We Learned**: NHC eliminates skidding and vertical flight, but provides zero longitudinal speed observability. Furthermore, uncalibrated phone mounting angles cause forward acceleration to leak into the lateral constraint, causing false turning circles.
* **Decision**: ACCEPTED gated NHC with causal PCA frame alignment.
* **Next Question**: Can machine learning predict vehicle forward velocity directly from IMU dynamics?

---

### [2026-09-06] — Phase 4: AI Forward Speed Estimation (Dilated TCN)
* **Question**: Can a deep neural network infer vehicle speed from 10 seconds of smartphone inertial dynamics?
* **What We Tried**: Evaluated Ridge, Random Forest, and a 65,409-parameter Dilated Temporal Convolutional Network (Expanded TCN) trained across 39 trips.
* **Result**: Expanded TCN achieved **2.88 m/s mean MAE** across 19 held-out test trips (zero leakage). Mountain road drift dropped to **5.87% and 6.94%** at 60s. However, on straight motorways, predictions hit an asymptotic ceiling at ~85 km/h, underestimating 110 km/h driving.
* **What We Learned**: The TCN captures dynamic maneuvers and curvature well, but under-predicts high-speed straight cruise.
* **Decision**: ACCEPTED Expanded TCN as baseline forward velocity branch.
* **Next Question**: Why did preliminary campus walking tests trigger extreme 70 m/s predictions?

---

### [2026-09-08] — Phase 5.1: Field Telemetry & Pedestrian OOD Forensics
* **Question**: What caused the Android navigation app to report 71.66 m/s (258 km/h) on a campus walking test?
* **What We Tried**: Feature attribution audit, gradient decomposition, and input ablation on phone telemetry.
* **Result**: Human walking gait and arm swing generated gyroscope yaw rates of **$+26.08\sigma$** outside vehicle training bounds. Ablating `gyro_yaw` reduced speed predictions from $71.66 \to 34.94\text{ m/s}$.
* **What We Learned**: Pedestrian movement is severely out-of-distribution for automotive neural networks. Walking data is NOT vehicle validation.
* **Decision**: Implemented input $\pm 3.5\sigma$ clamping and OOD motion gating.
* **Next Question**: Is the highway ~85 km/h ceiling caused by training target distribution imbalance?

---

### [2026-09-09] — Phase 5.2: Target-Balanced TCN Training
* **Question**: Does continuous inverse-density loss weighting eliminate the motorway speed underestimation?
* **What We Tried**: Gaussian KDE continuous sample weighting ($1.48\times$ on motorways) + 4-quartile stratified batch sampling.
* **Result**: Mountain pass rate improved to 4/6 trips (`Vw12` reached 3.51% drift). However, motorway drift remained at **11.62%** (vs 11.22%) and bias stayed at $-3.45\text{ m/s}$. Suburban MAE degraded from $2.59 \to 2.94\text{ m/s}$.
* **What We Learned**: The highway plateau is not a sample density issue; loss rebalancing represents a Pareto trade-off.
* **Decision**: Retained Expanded TCN as primary; rejected hypothesis that loss weighting solves highway cruising.
* **Next Question**: Does high-frequency chassis vibration contain generalizable vehicle speed information?

---

### [2026-09-10] — Phase 5.3: Spectral Vibration Speed-Information Audit
* **Question**: Can tire rotation harmonics or chassis vibration power predict vehicle speed?
* **What We Tried**: Spectral analysis (Welch PSD, FFT peak tracking) across all 64 vehicle trips (>450,000 windows).
* **Result**: Dominant vibration frequency correlation with speed was **$r = -0.032$**. Motorway vibration peak stayed locked at $\sim 2.9\text{ Hz}$ (suspension bounce) from 85 to 104 km/h. Adding vibration changed speed MAE by an imperceptible $-0.01\text{ m/s}$ ($-0.2\%$).
* **What We Learned**: Commercial smartphones in dashboard mounts are mechanically damped by the vehicle suspension; tire harmonics are unobservable.
* **Decision**: REJECTED vibration as a primary vehicle speedometer.
* **Next Question**: What happens if we preserve velocity kinematically from the pre-outage GNSS fix?

---

### [2026-09-11 Morning] — Phase 5.5: Stateful Kinematics & Momentum Preservation
* **Question**: Does initializing velocity at the last GNSS fix and integrating longitudinal acceleration solve highway cruising?
* **What We Tried**: Compared Pure Kinematics, Fixed Damped Momentum ($\beta = 0.985$), and Static TCN across held-out trips.
* **Result**: On smooth mountain road `Vw12`, momentum achieved **1.78% drift at 60s**. But on urban stop-and-go trip `Vta26`, pure kinematics created phantom motion, exploding to **574.6% drift (292.8 m error on a 50 m trip)**!
* **What We Learned**: Momentum is one half of the solution. Transition cannot be based on elapsed time; it must be regime-aware.
* **Decision**: REJECTED static time-based switching; formulated condition-based fusion.
* **Next Question**: Can causal adaptive regime fusion resolve both failure modes?

---

### [2026-09-11 Afternoon] — Phase 5.6: Adaptive Regime-Aware Velocity Fusion
* **Question**: Does combining causal ZVD, cruise momentum, and dynamic blending achieve universal <10% drift?
* **What We Tried**: Implemented 4-state causal fusion (Stationary ZVD, Highway Cruise, Nominal Dynamic, Disagreement Gating).
* **Result**: Causal ZVD suppressed `Vta26` divergence from **574.6% down to 47.3%**. Adaptive fusion achieved the lowest 60s aggregate drift (**16.76%**). However, cross-trip pass rate at 60s was capped at **3 / 13 usable trips (23.1%)**.
* **What We Learned**: Standalone smartphone IMU dead reckoning faces an empirical ceiling around 23–33% pass rate due to the steady-state cruise observability bound ($ROC = 0.625$) and urban micro-crawling integration error.
* **Decision**: FROZE algorithmic experimentation; shifted to comprehensive repository documentation, reproducibility audit, and open-source release preparation.
