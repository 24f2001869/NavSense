# Literature Novelty and Prior Art Audit

This document audits relevant published literature in inertial dead reckoning, deep learning-assisted navigation, and smartphone sensor fusion to formally delineate what prior works have established versus what this research legitimately contributes.

**Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)

---

## 1. Prior Art Taxonomy & Comparative Analysis

| Study / Reference | Primary Focus & Platform | Model Architecture | Evaluated Sensors | Blackout Horizon | Reported Metrics | Cross-Trip Isolation | Negative Results Reported? | Key Limitation / Gap Addressed by NavSense |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|
| **Onyekpe et al. (2021)**<br>*Data in Brief*<br>DOI: 10.1016/j.dib.2021.106885 | Released public IO-VNBD dataset; baseline speed regression. | Basic CNN & LSTM | Smartphone IMU + GPS + CAN bus | 10–60s synthetic | MAE $\approx 1.2\text{--}2.5\text{ m/s}$ on random slices | ❌ Mixed / Shuffled slices | ❌ No | Evaluated on randomly shuffled time slices, obscuring cross-trip generalization failure; did not investigate vibration spectra or highway observability. |
| **Wang et al. (2025)**<br>*Applied Sciences*<br>DOI: 10.3390/app15168824 | Vehicle forward speed estimation from smartphone IMU. | LSTM with Attention mechanism | Smartphone accelerometer + gyroscope | Pointwise speed estimation | RMSE $\approx 1.8\text{--}2.4\text{ m/s}$ | ⚠️ Partial (Few trips) | ❌ No | Focuses solely on pointwise speed regression error; does not evaluate end-to-end multi-horizon dead reckoning drift inside an ESKF. |
| **AVNet (2025)**<br>*Satellite Navigation*<br>DOI: 10.1186/s43020-025-00162-4 | Adapting invariant EKF (RIEKF) with learned attitude and velocity. | Deep neural network + Invariant EKF | Windshield-mounted smartphone IMU | 30–60s outages | Positional drift $\approx 2.5\text{--}5.0\%$ on selected runs | ⚠️ Curated test set | ❌ No | Evaluates on smooth trajectories with active turns; does not analyze constant-speed straight highway unobservability or map-matching feedback instability. |
| **Wang et al. (2025)**<br>*arXiv:2505.18490* | Inertial sequence learning for vehicle speed via smartphone IMU. | Temporal Dilated Conv / Transformer | Smartphone IMU + coordinate alignment | Pointwise speed estimation | Speed MAE $\approx 2.1\text{ m/s}$ | ✅ Disjoint trips | ❌ No | Evaluates neural velocity prediction; does not investigate vehicle chassis vibration correlation or why high-speed highway prediction exhibits systematic negative bias. |
| **AI-IMU (Brossard et al., 2020)**<br>*IEEE Trans. Robotics*<br>DOI: 10.1109/TRO.2020.3014902 | Dynamically learns measurement covariance matrices in an EKF. | 1D Dilated CNN for covariance estimation | Calibrated automotive IMU rig | Full driving runs (KITTI) | Translation drift $< 1.10\%$ of distance | ✅ Whole trajectories | ❌ No | Evaluated on rigidly bolted industrial IMUs on vehicle roof; does not evaluate uncalibrated consumer smartphones with loose windshield mounts or severe distribution shift. |
| **RoNIN (Herath et al., 2020)**<br>*IEEE/CVF CVPR*<br>DOI: 10.1109/CVPR.2020.00331 | Neural inertial navigation for handheld and pocket devices. | ResNet / LSTM / TCN for pedestrian velocity | Smartphone IMU | 1–5 minute pedestrian tracks | ATE $\approx 2.5\text{--}4.0\text{ m}$ | ✅ Disjoint buildings | ❌ No | Designed strictly for pedestrian biomechanics (human walking gait). As proved in Phase 5.1, pedestrian models cannot navigate vehicles and vehicle models fail catastrophically on pedestrians. |
| **Farrell (2008) / Groves (2013)**<br>*Textbook Classical INS* | Classical strapdown inertial navigation + EKF + NHC. | 15-state linear / error-state Kalman filter | Industrial / tactical grade IMU + wheel ticks | Analytical error models | Cubic divergence $\mathcal{O}(t^3)$ without aiding | N/A (Analytical) | ✅ Known limits | Mathematically derived why standalone MEMS IMU double integration fails; requires physical wheel speed encoders (CAN bus), which do not exist on smartphones. |

---

## 2. What Is NOT Novel (Prohibited Claims)

To maintain strict scientific integrity, the manuscript explicitly disclaims the following:

1. **❌ We did NOT invent the Dilated TCN architecture**:
   * Dilated causal convolutions with exponential receptive fields were formalized by Lea et al. (2017) and Bai et al. (2018).
   * Applying TCNs or LSTMs to smartphone IMU velocity regression is an established research direction already published in 2024–2025.
2. **❌ We did NOT invent the Error-State Kalman Filter (ESKF)**:
   * The 15-state error-state formulation (position, velocity, attitude errors, accelerometer/gyro biases) is a classical standard derived by Madyastha et al. (2011) and Sola (2017).
3. **❌ We did NOT solve the SIH <10% navigation benchmark universally**:
   * The empirical data clearly show that across 13 usable 60s test trips, only 3 trips (23.1%) pass the strict <10% threshold. Claiming that the problem is solved is a false claim.
4. **❌ We did NOT perform physical vehicle road validation**:
   * The physical Android 14 tests on the OnePlus smartphone were handheld and pedestrian stress tests. Real-car telemetry in this paper originates from the IO-VNBD dataset.

---

## 3. What IS Genuinely Novel and Publishable

The genuine contribution of this paper lies in its **systematic, failure-aware empirical evaluation across the 64-trip IO-VNBD census and held-out test routes**, providing large-scale empirical findings under tested sampling and feature conditions:

### Contribution 1: Empirical Evaluation of the "Vibration Speedometer" Hypothesis
* *Literature Context*: Several recent studies have hypothesized that high-rate smartphone accelerometers can detect engine RPM harmonics or tire-road roughness to infer vehicle speed without GNSS.
* *Our Evidence*: Across 64 trips and >450,000 one-second FFT windows, we performed an extensive spectral audit. The Pearson correlation between dominant vibration frequency and speed was **$r = -0.032$** (virtually zero). Modern automotive suspensions mechanically damp out high-frequency road excitations; the dominant 2.2–2.5 Hz peak is consistent with low-frequency vehicle body/chassis dynamics, showing no statistically significant correlation with forward vehicle speed.

### Contribution 2: Empirical Identification of Highway Cruise Observability Limits
* *Literature Context*: Prior machine-learning speed estimators have consistently under-predicted high-speed highway driving (~85 km/h plateau), typically dismissing the issue as "training data imbalance."
* *Our Evidence*: On 112 minutes of motorway driving (`V-Vfa02`, 163 km), we demonstrated through ROC-AUC analysis (0.625) that constant-speed cruising ($a_x \approx 0, \omega_z \approx 0$) produces specific forces with weak statistical separability from the smartphone MEMS sensor noise floor. Because the instantaneous speed state is weakly observable, neural models regress toward the training set mean.

### Contribution 3: The Danger of Unconditional Sensor Fusion & Map Feedback
* *Literature Context*: Prior works often assume that adding Non-Holonomic Constraints (NHC) or closed-loop OpenStreetMap (OSM) matching will monotonically reduce dead reckoning drift.
* *Our Evidence*:
  * Adding unconditional lateral NHC updates degraded 60s drift by **45.5%** (843.4 m to 1226.9 m) in the tested setup, consistent with cornering tire sideslip and mounting misalignment corrupting heading.
  * Closed-loop map-heading feedback during 60s outages more than doubled drift (**578.3 m to 1303.9 m**) due to incorrect link latching.
  * We demonstrated that **decoupled velocity damping** ($K_y[6:15] = 0$) substantially improves stability, reducing drift by 54.6% to 382.7 m.

### Contribution 4: Out-of-Distribution (OOD) Dynamics in Edge Deployment
* *Literature Context*: Papers evaluate models offline on in-vehicle datasets without documenting the failure modes encountered during real mobile deployment.
* *Our Evidence*: We documented how pedestrian arm-swings ($3.81\text{ rad/s}$, $+26.08\sigma$) trigger 71.66 m/s (258 km/h) velocity spikes in vehicle-trained models, and established the necessity of input $\sigma$-clipping, physical acceleration bounds, and dynamic hardware timestamping ($\Delta t$) on Android edge devices.
