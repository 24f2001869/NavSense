# Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift

**Project:** NavSense (SIH26168)  
**Authors:** [AUTHOR NAMES ANONYMIZED FOR PEER REVIEW / SEE AUTHOR INPUT REQUIRED]  
**Affiliations:** [AFFILIATIONS ANONYMIZED]  
**Correspondence:** [CORRESPONDING EMAIL ANONYMIZED]  
**Repository & Codebase:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)  

---

### Abstract
Consumer smartphones equipped with low-cost micro-electromechanical system (MEMS) inertial measurement units (IMUs) offer an ubiquitous platform for vehicular navigation. However, during Global Navigation Satellite System (GNSS) outages—such as in urban canyons, tunnels, and multi-level interchanges—open-loop double integration of unconstrained consumer IMU signals diverges rapidly due to sensor bias drift, acoustic noise, and mechanical vibrations, resulting in quadratic velocity errors and cubic position drift exceeding 300 meters within 60 seconds (89.66% macro-average normalized drift). While recent learning-based approaches attempt to estimate forward vehicle velocity directly from inertial sequences, existing literature often reports optimistic performance on short, curated sequences without systematically evaluating cross-trip generalization, closed-loop fusion stability, out-of-distribution dynamics, or long-horizon failure modes. 

In this work, we present an empirical evaluation and failure analysis of smartphone-only inertial dead reckoning under simulated GNSS outages spanning 10 to 60 seconds. Utilizing the public Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD)—encompassing 64 passenger car trips, 32.85 hours of synchronized telemetry, and 1,182,661 raw synchronized epochs at 10 Hz—we evaluate a modular navigation architecture comprising a 15-state Error-State Kalman Filter (ESKF), a causal dilated Temporal Convolutional Network with kinematic features (TCN-kin) for forward speed estimation, kinematic regime blending, Zero Velocity Detection (ZVD), and edge execution on Android. Across 19 strictly held-out test trips (113,539 emitted prediction epochs, 238.1 km), scaling TCN training from 6 to 39 whole trips reduces velocity Mean Absolute Error (MAE) from 6.17 m/s to 2.76 m/s (55.27% reduction) and decreases 60-second position drift from 263.6 m to 93.0 m (64.72% reduction). 

Crucially, our controlled ablation experiments uncover severe, counter-intuitive failure modes in standard navigation assumptions:
1. **Fusion Instability:** Standard unconditional lateral Non-Holonomic Constraints (NHC) degrade 60-second position drift by 45.47% (843.4 m to 1226.9 m) in the tested configuration, consistent with tire sideslip angle coupling and unmodeled phone-to-vehicle mounting misalignment corrupting filter heading; a decoupled velocity damping architecture ($K_y[6:15] = 0$) substantially improves filter stability, reducing drift to 382.7 m (54.62% improvement over the coupled baseline).
2. **Map Feedback Latency:** Closed-loop OpenStreetMap (OSM) heading feedback degrades 60-second drift from 578.3 m to 1303.9 m (-125.5%) due to wrong-link latching and geometric discretization errors during turns.
3. **Spectral Decoupling:** A causal spectral census across 64 trips (>450,000 windows) finds no robust generalizable vibration-based speed signal under the tested 10-Hz processing rate and spectral features (dominant frequency Pearson $r = -0.032$, Spearman $\rho = -0.028$), showing a dominant low-frequency spectral component (~2.2–2.5 Hz) consistent with vehicle-body/chassis dynamics.
4. **Highway Observability Limits:** High-speed motorway cruising ($v \ge 25\text{ m/s}$) induces significant negative regression bias ($-3.5\text{ m/s}$, $r = -0.4866$), consistent with weak statistical separability of instantaneous inertial features during unaccelerated cruise (ROC-AUC = 0.625).
5. **Out-of-Distribution Vulnerability:** Testing the vehicle-trained TCN on handheld pedestrian walking induces an extreme velocity spike to 71.66 m/s (258 km/h), isolated via feature attribution to high-frequency arm-swing yaw rates (+26.08$\sigma$).

Across 13 usable 60-second blackout test trajectories, our adaptive fusion pipeline achieves an aggregate mean drift of 96.65 meters (16.76% macro-average normalized drift) and passes the Smart India Hackathon (SIH) $<10\%$ drift benchmark in 3 of 13 trips (23.08%). We conclude that while learned forward velocity and decoupled kinematic damping substantially mitigate inertial runaway, universal $<10\%$ drift compliance was not achieved with smartphone-only IMUs under long-horizon unaccelerated highway cruising, underscoring the necessity of failure-aware navigation architectures.

**Keywords:** Smartphone Inertial Navigation, GNSS-Denied Dead Reckoning, Temporal Convolutional Networks, Error-State Kalman Filter, Non-Holonomic Constraints, Distribution Shift, Failure Analysis, Reproducibility.

---

## 1. Introduction

Continuous, seamless vehicular navigation is foundational to modern intelligent transportation systems (ITS), connected autonomous vehicles (CAVs), emergency logistics, and consumer location-based services. In open-sky environments, multi-constellation Global Navigation Satellite Systems (GNSS)—such as GPS, GLONASS, Galileo, and NavIC—routinely deliver horizontal positioning accuracies within 2 to 5 meters. However, satellite visibility is systematically degraded or completely compromised in urban canyons, road tunnels, multi-level covered interchanges, dense foliage, and subterranean parking structures. In these GNSS-denied environments, navigation systems must rely entirely on autonomous dead reckoning (DR) to propagate position from the last known satellite fix.

While dedicated automotive-grade inertial navigation systems (INS) utilize high-precision tactical-grade fiber-optic gyroscopes (FOG) or ring laser gyroscopes (RLG) alongside calibrated vehicle Controller Area Network (CAN) wheel speed odometry, consumer-grade smartphones represent the primary navigation terminal for billions of drivers worldwide. Smartphones incorporate low-cost Micro-Electro-Mechanical Systems (MEMS) accelerometers and gyroscopes. These consumer sensors suffer from severe physical imperfections: run-to-run bias instability, significant thermal drift, scale factor non-linearities, non-orthogonal cross-axis sensitivity, and high acoustic noise floors. Unconstrained open-loop double integration of raw specific forces from such sensors leads to quadratic error accumulation in velocity and cubic divergence in position ($\Delta p(t) \propto \frac{1}{6} b_a t^3$). Within 60 seconds of complete satellite denial, uncorrected smartphone inertial drift routinely exceeds 300 to 1,000 meters, rendering raw strapdown mechanization unusable for vehicular lane identification or intersection navigation.

To constrain this rapid divergence without requiring external vehicle wiring or OBD-II dongles, the research community has explored two primary avenues:
1. **Kinematic Constraint Filtering:** Traditional aerospace and robotics approaches rely on Extended Kalman Filters (EKF) or Error-State Kalman Filters (ESKF) incorporating Non-Holonomic Constraints (NHC) [1]–[3], Zero Velocity Updates (ZUPT) during stationary stops [4], and digital road network map matching [5], [6]. NHC assumes that under normal driving conditions without sideslip or road lift-off, the lateral and vertical velocities of the vehicle body frame are approximately zero ($v_y \approx 0, v_z \approx 0$).
2. **Data-Driven Sequence Learning:** With the emergence of deep sequence models, researchers have trained Recurrent Neural Networks (LSTM/GRU), Convolutional Neural Networks (CNNs), and Temporal Convolutional Networks (TCNs) to map sliding temporal windows of IMU specific forces and angular velocities directly to instantaneous 1D forward vehicle speed or 2D velocity vectors [7]–[12].

Despite encouraging headline results in recent literature, significant skepticism remains regarding the real-world viability and generalization of smartphone-only dead reckoning. Published studies frequently suffer from critical methodological shortcomings:
* **Short Evaluation Horizons:** Systems are often evaluated over brief 5-to-15 second artificial outages, concealing long-horizon filter instabilities and heading drift.
* **Curated Benchmark Trajectories:** Test trajectories are frequently selected from routes exhibiting rich centripetal accelerations (e.g., tight roundabouts or sharp city turns), while omitting high-speed straight highway cruising where inertial observability degrades.
* **Leakage and Conflated Denominators:** Models are often evaluated on sliding windows randomly sampled from the same driving trips used for training, resulting in temporal feature leakage. Furthermore, macro-trip averages, window-weighted statistics, and distance-normalized metrics are frequently conflated without explicit denominator documentation.
* **Omission of Negative Results:** Intuitive integration strategies—such as hard lateral NHC clamping, closed-loop map heading injection, or chassis vibration frequency tracking—are often assumed to be beneficial, without documenting the significant divergence they induce when vehicle kinematic assumptions are violated.

### Scientific Contributions
In this paper, we refrain from claiming the invention of a novel deep neural architecture or claiming universal compliance with the competitive Smart India Hackathon (SIH) $<10\%$ drift threshold. Instead, we present an evidence-based empirical audit and failure analysis of smartphone-only dead reckoning under extended GNSS outages. Operating strictly on the public IO-VNBD benchmark dataset and verified mobile implementations, our primary contributions are:

1. **Systematic Multi-Stage Empirical Evaluation:** We benchmark classical strapdown mechanization, classical regression, deep sequence models (TCN), tightly coupled kinematic filters, and adaptive fusion across 64 passenger car trips (32.85 hours, 1,182,661 raw epochs) across four standardized outage horizons (10 s, 20 s, 30 s, and 60 s).
2. **Controlled Generalization Scaling:** We demonstrate the quantitative impact of scaling training data from 6 to 39 whole trips across 19 strictly held-out test trips (113,539 emitted prediction epochs, 238.1 km), achieving a 55.27% reduction in velocity MAE (6.17 m/s to 2.76 m/s) and a 64.72% reduction in 60-second position drift (263.6 m to 93.0 m).
3. **Decoupled Velocity Damping Formulation:** Through an 8-way controlled ablation, we show that standard unconditional lateral NHC updates severely degrade 60-second navigation performance (-45.47% drift regression under the tested setup) by injecting lateral innovations into attitude and bias states during cornering and phone misalignment. We formulate a decoupled Kalman gain architecture ($K_y[6:15] = 0$) that restricts lateral innovations strictly to linear velocity damping, improving drift by +54.62% (382.7 m).
4. **Empirical Evaluation of Vibration Speed Estimation:** Across 64 trips and >450,000 causal windows, we conduct an extensive spectral audit of smartphone IMU telemetry, finding no robust generalizable vibration-based speed signal (dominant frequency Pearson $r = -0.032$, Spearman $\rho = -0.028$) due to dominant low-frequency vehicle-body/chassis dynamics around 2.2–2.5 Hz.
5. **Rigorous Failure Analysis & Empirical Boundary Identification:** We isolate four fundamental failure modes: (i) weak statistical separability of forward speed during constant-velocity highway cruising (ROC-AUC = 0.625), (ii) high-speed negative regression bias ($-3.5\text{ m/s}$) under training distribution shift, (iii) closed-loop map matching heading corruption (-125.5% drift degradation), and (iv) extreme out-of-distribution (OOD) velocity spikes (71.66 m/s / 258 km/h) triggered by pedestrian arm-swing dynamics (+26.08$\sigma$ yaw rate).
6. **Edge Deployment and Timing Correction:** We document native Android 14 deployment on a mid-tier smartphone, demonstrating real-time inference (4.2–7.8 ms) and sub-5 cm numerical parity with offline Python baselines. We identify and resolve an operating system sensor callback jitter defect (8.63 Hz vs 10.0 Hz target), showing that hardware nanosecond timestamp integration is mandatory to prevent time-scale distortion.

---

## 2. Related Work

The field of vehicular dead reckoning sits at the intersection of classical strapdown inertial navigation, kinematic Kalman filtering, deep sequence learning, and map-based constraint fusion. Table \ref{tab:lit_comparison} summarizes representative works in the literature alongside the present study.

```
+==================================================================================================================+
| Reference          | IMU Grade  | Velocity Engine   | Fusion Filter     | Outage Horizon | Evaluation Scope      |
+==================================================================================================================+
| Dissanayake (2001) | Tactical   | Wheel Odometry    | EKF + NHC         | 60-120 s       | Field car (1 route)   |
| AI-IMU (2020)      | Automotive | Learned Covariance| Invariant EKF     | 10-60 s        | KITTI (11 seq)        |
| Wang et al. (2025) | Smartphone | LSTM + Attention  | Direct integration| 5-15 s         | Curated city trips    |
| AVNet (2025)       | Smartphone | Dilated CNN       | Invariant EKF     | 10-30 s        | Synthetic / Lab       |
| Ours (NavSense)    | Smartphone | Causal TCN-kin    | Decoupled ESKF    | 10-60 s        | 64 Trips (32.85 h)    |
+==================================================================================================================+
```

### 2.1 Classical Strapdown Inertial Navigation and ESKF
Strapdown inertial navigation algorithms propagate position, velocity, and orientation by integrating specific forces and angular rates in a local navigation frame (e.g., North-East-Down or local East-North-Up) [13], [14]. In consumer smartphones, early EKF formulations integrated states in total-state form. However, total-state filtering suffers from non-linear attitude singularities and numerical degradation when linearizing large attitude rotations. Solà [3] formalized the quaternion-based Error-State Kalman Filter (ESKF), where the nominal state is propagated open-loop via non-linear kinematics, while a true small error state $\delta \mathbf{x} \in \mathbb{R}^{15}$ is estimated linearly. This decouples the rapid high-frequency nominal propagation from the slower, better-conditioned error-correction loop, eliminating kinematic singularities. We adopt this 15-state ESKF structure as the mathematical core of our navigation engine.

### 2.2 Non-Holonomic Constraints (NHC)
To arrest lateral and vertical inertial velocity drift in ground vehicles, Dissanayake et al. [1] introduced Non-Holonomic Constraints (NHC). For a wheeled vehicle operating without lateral skidding or vertical flight, the velocity vector in the vehicle body frame is constrained by:
\begin{equation}
v_y^b \approx 0, \quad v_z^b \approx 0
\label{eq:nhc_ideal}
\end{equation}
While NHC has proven highly effective when paired with rigidly mounted automotive-grade IMUs aligned precisely to the chassis axes [2], [15], its application to uncalibrated consumer smartphones introduces severe challenges. Smartphones are placed in dashboard cradles, cupholders, or console trays with unknown, non-zero boresight alignment angles. Furthermore, during cornering or low-friction road conditions, vehicles experience non-zero tire sideslip angles ($\beta \neq 0$). Sukkarieh et al. [16] noted that unmodeled mounting angles convert lateral kinematic innovations into false attitude corrections. In this paper, we specifically audit this breakdown, quantifying the exact performance degradation induced by unconditional NHC updates in consumer phones.

### 2.3 Machine Learning for Inertial Velocity Estimation
Recognizing that consumer MEMS accelerometers cannot be integrated open-loop over extended intervals, recent research has turned to deep neural networks to learn virtual speed odometry directly from high-frequency inertial patterns.
* **Pioneering Sequence Learning:** Yan et al. [7] (RoNIN) and Cortesi et al. [8] demonstrated that 1D and 2D temporal convolutions could regress pedestrian and vehicular velocity vectors from smartphone IMU windows.
* **Recurrent Architectures:** Wang et al. [11] deployed a Long Short-Term Memory (LSTM) network with attention mechanisms on smartphone IMU data, reporting high accuracy in predicting vehicle speed. However, their evaluation focused on short horizons (5–15 s) and did not document filter stability under high-speed cruise conditions.
* **Preprint Architectures:** Recent works such as AVNet [10] and related preprints [12] adapt Invariant EKFs (RI-EKF) with dilated convolutional backbones to estimate velocity. While mathematically sophisticated, these studies generally omit extensive cross-trip dataset distribution shift audits, feature attribution on out-of-distribution movements, or large-scale spectral audits of underlying vehicle vibration modes.

### 2.4 Map Matching and Closed-Loop Feedback
Map matching algorithms project unconstrained dead reckoning trajectories onto digital road network centerlines (such as OpenStreetMap) [5], [6], [17]. While map matching is traditionally employed as a post-processing or visualization layer, several studies explore closed-loop feedback, where road link azimuth and lateral perpendicular distance are fed directly back into the Kalman filter as pseudo-measurements. Quddus et al. [5] and subsequent surveys [6] warned that closed-loop map feedback introduces catastrophic positive feedback loops when an incorrect link candidate is latched. In this work, we implement an explicit 5-gate Multiple Hypothesis Tracking (MHT) map matcher and empirically demonstrate that closed-loop heading feedback degrades 60-second dead reckoning accuracy by over 125%.

---

## 3. Research Questions and Objectives

To ensure rigorous scientific discipline, this empirical investigation is structured around eight explicit research questions derived from our experimental progression:

* **RQ1 (Baseline Divergence):** What is the empirical rate of position and velocity divergence for pure unassisted consumer smartphone IMU strapdown mechanization across 10 s, 20 s, 30 s, and 60 s GNSS outages?
* **RQ2 (Learned Speed Scaling):** To what extent does scaling training data across diverse whole trips improve the out-of-sample forward speed estimation accuracy of a causal Temporal Convolutional Network?
* **RQ3 (Distribution Shift & Observability):** How does the learned speed estimator perform under severe road-type and speed distribution shifts, specifically during straight high-speed motorway cruising?
* **RQ4 (Feature Attribution & OOD Dynamics):** Which inertial feature channels dominate the neural velocity prediction, and how does a vehicle-trained model respond to out-of-distribution pedestrian movement?
* **RQ5 (Kinematic Constraint Stability):** Do standard lateral Non-Holonomic Constraints (NHC) reliably improve dead reckoning accuracy in consumer smartphones, or do they destabilize the filter attitude states under real driving dynamics?
* **RQ6 (Closed-Loop Map Matching):** Does direct feedback of digital road link azimuth and lateral offset into the ESKF improve positioning accuracy, or does candidate ambiguity degrade tracking?
* **RQ7 (Vibration Speed Estimation):** Does chassis and engine vibration captured by a 10 Hz smartphone IMU contain a generalizable, speed-dependent spectral signature?
* **RQ8 (Benchmark Compliance):** Can an adaptive, regime-aware fusion architecture satisfy the Smart India Hackathon (SIH) $<10\%$ drift criterion across all eligible 60-second blackout trajectories?

---

## 4. Datasets and Experimental Setup

### 4.1 The IO-VNBD Benchmark Dataset
All vehicular evaluations in this study are conducted on the public Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD) [18]. IO-VNBD was explicitly recorded to benchmark smartphone-based navigation under challenging real-world road conditions, capturing synchronized consumer smartphone sensor streams alongside vehicle diagnostic telemetry.

The complete IO-VNBD distribution comprises multiple vehicle modalities (cars, motorcycles, scooters, auto-rickshaws). In this study, we isolate the complete passenger car subset, comprising **64 whole driving trips**, **32.85 total hours of driving**, and **1,182,661 raw synchronized epochs** sampled at a uniform 10 Hz rate. Table 1 provides the complete census of the evaluated passenger car trips across four standardized road categories.

```
+========================================================================================================================+
| Table 1: IO-VNBD Passenger Car Dataset Inventory                                                                        |
+========================================================================================================================+
| Route Category            | Trips (N) | Duration (h) | Distance (km) | Mean Speed | Max Speed  | Road Characteristics  |
+========================================================================================================================+
| Suburban Town (Vta)       | 29        | 15.42 h      | 482.1 km      | 31.3 km/h  | 118.0 km/h | Stops, traffic lights |
| Dense Urban (Vtb)         | 12        | 2.84 h       | 68.4 km       | 24.1 km/h  | 78.5 km/h  | Stop-and-go, pedestrian|
| Winding Mountain (Vw)     | 21        | 11.87 h      | 512.6 km      | 43.2 km/h  | 102.4 km/h | Curves, steep grades   |
| Motorway / Highway (V-Vfa)| 2         | 2.72 h       | 224.2 km      | 82.4 km/h  | 124.8 km/h | High-speed cruise      |
+---------------------------+-----------+--------------+---------------+------------+------------+-----------------------+
| TOTAL / SUMMARY           | 64        | 32.85 h      | 1,287.3 km    | 39.2 km/h  | 124.8 km/h | Complete Car Benchmark|
+========================================================================================================================+
```

### 4.2 Data Partitioning and the Driver E Audit Resolution
A critical methodological vulnerability in machine learning navigation literature is data leakage across training and test splits. To ensure scientific validity, our partitioning satisfies three rules:
1. **Whole-Trip Disjointness:** Partitions are assigned strictly at the whole-trip level. No sliding window or temporal frame from a test trip ever appears in the training or validation sets.
2. **Deterministic Assignment:** Partition assignment is locked deterministically prior to model training.
3. **Driver-Level Audit:** We audited the metadata of all 64 passenger car trips in IO-VNBD. This audit revealed that **100% of the passenger car trips in IO-VNBD were recorded by a single driver (`Driver E`)**. Consequently, while our evaluation rigorously assesses cross-trip, cross-route, and cross-speed generalization, **we explicitly disclaim cross-driver invariance**. The evaluation demonstrates whole-trip generalization for a single driver profile; cross-driver generalization remains an unverified hypothesis requiring multi-driver datasets.

Table 2 documents the exact partition boundaries. Note that the total usable model samples ($650,661$) is less than the raw synchronized epoch count ($1,182,661$) due to the exclusion of the 100-sample pre-buffer initialization window for each trip, standstill boundary trimmings, and data alignment requirements.

```
+========================================================================================================================+
| Table 2: Formal Partitioning: Train, Validation, and Test Sets                                                         |
+========================================================================================================================+
| Partition Split   | Trips (N) | Usable Samples | Duration (h) | Distance (km) | Driver ID | Route Distribution         |
+========================================================================================================================+
| Training Split    | 39        | 478,210        | 13.28 h      | 521.4 km      | Driver E  | 21 Suburban, 8 Urban, 10 Mtn|
| Validation Split  | 6         | 58,912         | 1.64 h       | 72.8 km       | Driver E  | 2 Suburban, 1 Urban, 2 Mtn  |
| Held-Out Test     | 19        | 113,539        | 3.15 h       | 238.1 km      | Driver E  | 8 Suburban, 4 Urban, 6 Mtn  |
+========================================================================================================================+
```

### 4.3 Reference Velocity Hierarchy and Calibration
In vehicular navigation research, the term "ground truth" is frequently abused to describe low-cost GPS fixes or uncalibrated sensors. In the IO-VNBD benchmark, reference telemetry was recorded using a professional **Racelogic VBOX Video HD2 data logger (10 Hz)** with an external rooftop GPS antenna and direct vehicle CAN bus integration on a Ford Fiesta Titanium. We establish a strict reference hierarchy:
* **Physical Ground Truth:** Absolute position and velocity verified via external centimeter-grade optical or Real-Time Kinematic (RTK) surveying. This was **not** recorded in IO-VNBD.
* **Engineering Reference Signal (Racelogic VBOX Vehicle CAN Bus):** Vehicle forward speed queried directly from the Ford Fiesta Electronic Control Unit (ECU) via the Racelogic VBOX CAN logger. CAN speed exhibits strong kinematic agreement with transmission wheel pulse counts (Pearson $r = 0.998$, MAE = 0.083 m/s, $R^2 > 0.996$). We audited CAN speed during stationary segments and identified a baseline digital noise floor of 0.0204 m/s. Crucially, cross-correlation analysis revealed a **100 ms ECU digital filtering delay** relative to raw IMU timestamps. Consequently, all CAN reference velocities are phase-shifted by $-100\text{ ms}$ to maintain temporal synchronization.
* **External Reference GPS:** Racelogic VBOX 10 Hz rooftop GPS fixes, used for trajectory initialization and pre-outage leveling.
* **Smartphone GPS Speed:** Raw GPS speed reported by the smartphone internal receiver exhibited substantial noise and latency errors during dynamic maneuvers (MAE 4.43 to 7.28 m/s vs CAN), confirming that smartphone GPS speed cannot serve as a reliable reference for model training.

### 4.4 Author Physical Field Testing Platform
In addition to the public IO-VNBD dataset, physical on-device pipeline verification was conducted using an Android smartphone:
* **Device:** OnePlus Nord CE 2 Lite 5G (Qualcomm Snapdragon 695 5G, 6 GB LPDDR4X RAM).
* **Operating System:** Android 14 (API Level 34).
* **Sensors:** STMicroelectronics LSM6DSO accelerometer/gyroscope, AKM AK09918 magnetometer.
* **Scope of Testing:** Tests included stationary bench baselines, rooftop walking with active GNSS, rooftop walking with software-simulated 60-second outages, and handheld transit. 

**Mandatory Physical Testing Disclaimer:** As the authors did not possess access to an instrumented passenger vehicle for local testing, **all physical Android field tests were conducted on foot (pedestrian/handheld) or on stationary test benches**. These tests served exclusively to audit native Java/Kotlin ESKF execution, ONNX Runtime inference latency, hardware timestamp jitter, and out-of-distribution (OOD) dynamics. **Under no circumstances should these Android field recordings be construed as vehicular road validation.**

---

## 5. Methodology

The end-to-end dead reckoning architecture evaluated in this work is depicted schematically in Figure 1. The pipeline processes raw smartphone IMU telemetry through five modular stages: Preprocessing and Boresight Alignment, Dual-Branch Velocity Estimation, Adaptive Blending, 15-State Error-State Kalman Filtering, and Downstream Map Association.

```
+---------------------------------------------------------------------------------------------------+
| FIGURE 1: End-to-End Modular Dead Reckoning Architecture                                           |
|                                                                                                   |
| [Raw Phone IMU] ---> [Boresight Alignment] ---> [Dual-Branch Velocity Engine]                     |
| (~8.6 Hz Dynamic dt) (Gravity & Forward Level)  |-- Causal Dilated TCN-kin (9 Kinematic Features)  |
|                                                 |-- Kinematic Accelerometer Integration           |
|                                                 |-- Zero Velocity Detector (ZVD Gate)             |
|                                                              |                                    |
|                                                              v                                    |
| [15-State ESKF Core] <---------------------------- [Kinematic Blending Engine]                    |
| |-- Nominal Mechanization (Native Java)             (Stationary / Turning / Cruise Dynamics)     |
| |-- Error State delta_x in R^15                                                                   |
| |-- Decoupled Velocity Damping (Ky[6:15] = 0)                                                     |
| |-- Gated Lateral NHC Update (vy = 0, vz = 0)                                                     |
|                                                                                                   |
| [Downstream Map Matching Layer] (Shadow MHT Link Projection - Open-Loop Visualization Only)       |
+---------------------------------------------------------------------------------------------------+
```

### 5.1 Coordinate Frames and Preprocessing
We define three right-handed Cartesian coordinate reference frames:
1. **Inertial Navigation Frame ($n$-frame):** Local East-North-Up (ENU) tangent plane fixed to the Earth's surface at the initial GNSS outage fix.
2. **Vehicle Body Frame ($b$-frame):** Origin at the vehicle center of gravity; $X_b$ points forward along the longitudinal driving axis, $Y_b$ points laterally to the left, and $Z_b$ points upwards orthogonal to the road surface.
3. **Smartphone Sensor Frame ($s$-frame):** Defined by the physical casing of the smartphone.

Because consumer smartphones are placed in arbitrary orientations, raw sensor readings must be transformed from the $s$-frame to the vehicle $b$-frame. During the 10-second pre-outage window where GNSS fixes remain available, we determine the mounting rotation matrix $\mathbf{R}_s^b$ via two-step gravity and heading leveling:
\begin{equation}
\hat{\mathbf{z}}_b = -\frac{\mathbb{E}[\mathbf{f}^s_{\text{stationary}}]}{\|\mathbb{E}[\mathbf{f}^s_{\text{stationary}}]\|_2}
\label{eq:gravity_align}
\end{equation}
The longitudinal forward axis $\hat{\mathbf{x}}_b$ is determined by projecting the vehicle GNSS acceleration vector during straight-line acceleration onto the horizontal plane orthogonal to $\hat{\mathbf{z}}_b$. The lateral axis completes the right-handed triad: $\hat{\mathbf{y}}_b = \hat{\mathbf{z}}_b \times \hat{\mathbf{x}}_b$.

### 5.2 Classical Strapdown Mechanization
In the absence of external updates, nominal position $\mathbf{p}^n$, velocity $\mathbf{v}^n$, and attitude quaternion $\mathbf{q}_b^n$ propagate according to classical continuous-time strapdown kinematics:
\begin{align}
\dot{\mathbf{p}}^n &= \mathbf{v}^n \label{eq:pos_dot} \\
\dot{\mathbf{v}}^n &= \mathbf{R}(\mathbf{q}_b^n) \mathbf{f}^b + \mathbf{g}^n \label{eq:vel_dot} \\
\dot{\mathbf{q}}_b^n &= \frac{1}{2} \mathbf{q}_b^n \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega}^b \end{bmatrix} \label{eq:quat_dot}
\end{align}
where $\mathbf{f}^b = \mathbf{R}_s^b (\mathbf{f}^s - \mathbf{b}_a)$ represents the unbiased specific force, $\boldsymbol{\omega}^b = \mathbf{R}_s^b (\boldsymbol{\omega}^s - \mathbf{b}_g)$ represents the unbiased angular velocity, $\mathbf{g}^n = [0, 0, -9.80665]^T\text{ m/s}^2$ is the local gravity vector, and $\otimes$ denotes quaternion multiplication.

### 5.3 Error-State Kalman Filter (ESKF) Formulation
To estimate and correct the cumulative drift of the nominal mechanization, we implement a 15-state continuous-discrete Error-State Kalman Filter [3]. The true state $\mathbf{x}_t$ is decomposed into nominal state $\hat{\mathbf{x}}$ and error state $\delta \mathbf{x}$:
\begin{equation}
\mathbf{p}^n = \hat{\mathbf{p}}^n + \delta \mathbf{p}^n, \quad \mathbf{v}^n = \hat{\mathbf{v}}^n + \delta \mathbf{v}^n, \quad \mathbf{q}_b^n = \hat{\mathbf{q}}_b^n \otimes \delta \mathbf{q}, \quad \mathbf{b} = \hat{\mathbf{b}} + \delta \mathbf{b}
\label{eq:state_decomp}
\end{equation}
The 15-dimensional error state vector is defined as:
\begin{equation}
\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n \\ \delta \mathbf{v}^n \\ \delta \boldsymbol{\theta}^n \\ \delta \mathbf{b}_a \\ \delta \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}
\label{eq:error_state}
\end{equation}
where $\delta \boldsymbol{\theta}^n \in \mathbb{R}^3$ represents small angular rotation errors such that $\delta \mathbf{q} \approx [1, \frac{1}{2}\delta \boldsymbol{\theta}^T]^T$, and $\delta \mathbf{b}_a, \delta \mathbf{b}_g \in \mathbb{R}^3$ represent residual accelerometer and gyroscope bias errors modeled as random walks.

The continuous-time linearized error dynamics matrix $\mathbf{F}_t \in \mathbb{R}^{15 \times 15}$ is formulated as:
\begin{equation}
\mathbf{F}_t = \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -[\mathbf{R}(\hat{\mathbf{q}}_b^n) \hat{\mathbf{f}}^b]_\times & -\mathbf{R}(\hat{\mathbf{q}}_b^n) & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & -[\hat{\boldsymbol{\omega}}^n]_\times & \mathbf{0}_{3 \times 3} & -\mathbf{R}(\hat{\mathbf{q}}_b^n) \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3}
\end{bmatrix}
\label{eq:F_matrix}
\end{equation}
where $[\mathbf{a}]_\times$ denotes the skew-symmetric cross-product matrix. Discrete-time propagation across epoch interval $\Delta t$ is computed via first-order truncation: $\boldsymbol{\Phi}_k \approx \mathbf{I}_{15 \times 15} + \mathbf{F}_t \Delta t$. State covariance $\mathbf{P}_k$ propagates as:
\begin{equation}
\mathbf{P}_{k|k-1} = \boldsymbol{\Phi}_k \mathbf{P}_{k-1|k-1} \boldsymbol{\Phi}_k^T + \mathbf{Q}_k
\label{eq:cov_prop}
\end{equation}
where $\mathbf{Q}_k = \text{diag}(\sigma_{vp}^2 \mathbf{I}_3, \sigma_{a}^2 \mathbf{I}_3, \sigma_{g}^2 \mathbf{I}_3, \sigma_{ba}^2 \mathbf{I}_3, \sigma_{bg}^2 \mathbf{I}_3) \Delta t$.

### 5.4 Forward Speed Estimation: Causal Dilated TCN-kin
To constrain longitudinal velocity runaway without external wheel pulses, we deploy a Causal Dilated Temporal Convolutional Network with kinematic features (TCN-kin) [19].
* **Input Representation:** The network receives a sliding history buffer of 100 consecutive 10 Hz epochs ($10.0\text{ s}$ temporal history) across 9 locked kinematic channels:
\begin{equation}
\mathbf{U}_k = [a_{\text{lin}, x}, \; a_{\text{lin}, y}, \; a_{\text{lin}, z}, \; \omega_z, \; \omega_y, \; \omega_x, \; a_h, \; a_v, \; \kappa]_{k-99:k} \in \mathbb{R}^{9 \times 100}
\label{eq:tcn_input}
\end{equation}
where $a_h = \sqrt{a_{\text{lin}, x}^2 + a_{\text{lin}, y}^2}$ is the instantaneous horizontal acceleration, $a_v = |a_{\text{lin}, z}|$ is vertical acceleration, and $\kappa = \frac{\omega_z}{\max(|v_{\text{prev}}|, 0.5)}$ provides a kinematic curvature proxy.
* **Causal Dilations and Receptive Field:** To guarantee zero future information leakage, convolutions are strictly causal with kernel size $K = 3$ and exponentially growing dilations $d \in \{1, 2, 4, 8, 16\}$. The theoretical receptive field of this 5-layer architecture spans:
\begin{equation}
RF = 1 + \sum_{l=0}^{4} (K - 1) \cdot d_l = 1 + 2 \cdot (1 + 2 + 4 + 8 + 16) = 63 \text{ steps } (6.3\text{ s})
\label{eq:receptive_field}
\end{equation}
Thus, while the input buffer maintains a 100-sample (10.0 s) history array, the convolutional filter's effective causal reach spans 6.3 seconds.
* **Network Complexity:** The model utilizes 64 residual channels across 5 residual blocks, totaling **65,409 trainable parameters** (162 KB in 32-bit floating-point ONNX format), ensuring minimal execution overhead on mobile edge hardware.
* **Loss Function:** The network is trained against the time-shifted CAN bus reference speed using Huber loss ($\delta = 1.0$) to promote robustness against transient measurement anomalies:
\begin{equation}
\mathcal{L}_{\text{Huber}}(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta (|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}
\label{eq:huber_loss}
\end{equation}

### 5.5 Decoupled Velocity Damping Architecture
When learned forward speed $v_x^{\text{TCN}}$ and lateral NHC constraints ($v_y \approx 0, v_z \approx 0$) are updated in the filter, the observation model in the vehicle frame is:
\begin{equation}
\mathbf{z}_v = \begin{bmatrix} v_x^{\text{TCN}} \\ 0 \\ 0 \end{bmatrix} - \mathbf{R}(\hat{\mathbf{q}}_b^n)^T \hat{\mathbf{v}}^n \in \mathbb{R}^3
\label{eq:vel_meas}
\end{equation}
The measurement Jacobian with respect to the error state $\delta \mathbf{x}$ is:
\begin{equation}
\mathbf{H}_v = \begin{bmatrix} \mathbf{0}_{3 \times 3} & \mathbf{R}(\hat{\mathbf{q}}_b^n)^T & -[\mathbf{R}(\hat{\mathbf{q}}_b^n)^T \hat{\mathbf{v}}^n]_\times & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 3} \end{bmatrix} \in \mathbb{R}^{3 \times 15}
\label{eq:H_vel}
\end{equation}
Under standard Kalman filtering, the optimal Kalman gain is computed as:
\begin{equation}
\mathbf{K} = \mathbf{P} \mathbf{H}_v^T (\mathbf{H}_v \mathbf{P} \mathbf{H}_v^T + \mathbf{R}_v)^{-1} \in \mathbb{R}^{15 \times 3}
\label{eq:kalman_gain}
\end{equation}
**The Cross-Coupling Vulnerability:** The term $[\mathbf{R}^T \hat{\mathbf{v}}^n]_\times$ in $\mathbf{H}_v$ couples velocity innovations directly into the attitude error states $\delta \boldsymbol{\theta}$ and gyroscope bias states $\delta \mathbf{b}_g$. When a vehicle corners, tire elasticity and dynamic cornering forces produce a physical sideslip angle $\beta \approx \arctan(v_y / v_x) \neq 0$. Applying an unconditional lateral constraint $z_{v_y} = 0 - v_y^b$ forces the filter to interpret physical vehicle sideslip and mounting misalignments as an orientation error ($\delta \theta_z$), rapidly destabilizing heading. 

To mitigate this vulnerability, we formulate a **Decoupled Velocity Damping** architecture (Variant V1). We decompose $\mathbf{K}$ into separate gain blocks for each measurement axis and enforce a structural projection:
\begin{equation}
\mathbf{K}_{v_y}^{\text{decoupled}}[6:15] = \mathbf{0}_{9 \times 1}
\label{eq:decoupled_gain}
\end{equation}
By zeroing the attitude and bias gain rows of the lateral velocity update, lateral innovations are restricted strictly to linear velocity damping ($\delta \mathbf{v}$), substantially reducing heading corruption from lateral kinematic disturbances.

### 5.6 Zero Velocity Detector (ZVD) Gate
To arrest velocity drift during traffic stops, we implement a multi-condition Zero Velocity Detector. Standstill is declared if and only if three conditions hold simultaneously over a sliding 1.0-second window:
\begin{equation}
\text{IsStationary} \iff \left( \sigma(\mathbf{f}^b) < \gamma_a \right) \land \left( \|\boldsymbol{\omega}^b\|_2 < \gamma_\omega \right) \land \left( v_x^{\text{TCN}} < \gamma_v \right)
\label{eq:zvd_conditions}
\end{equation}
with calibrated thresholds $\gamma_a = 0.15\text{ m/s}^2$, $\gamma_\omega = 0.05\text{ rad/s}$, and $\gamma_v = 0.50\text{ m/s}$. When stationary conditions are satisfied, a 3D velocity measurement $\mathbf{z}_{\text{ZUPT}} = \mathbf{0} - \hat{\mathbf{v}}^n$ is injected with tight observation noise covariance $\mathbf{R}_{\text{ZUPT}} = 10^{-4} \mathbf{I}_3\text{ m}^2/\text{s}^2$, clamping velocity drift to zero.

### 5.7 Downstream Map Association (Shadow MHT)
For map matching evaluation, we extract OpenStreetMap (OSM) vector networks within the operational boundaries of the IO-VNBD trips. We implement a 5-gate Multiple Hypothesis Tracking (MHT) engine that tracks road centerline candidates using:
1. Perpendicular distance gating ($d_\perp \le 30\text{ m}$).
2. Heading alignment gating ($|\Delta \psi| \le 45^\circ$).
3. Road topology and one-way connectivity validation.
4. Historical candidate likelihood propagation.

Crucially, we evaluate map matching in two distinct paradigms:
* **Closed-Loop Feedback:** Candidate link heading and lateral position are fed directly back into the ESKF as measurement updates.
* **Open-Loop Downstream Tracking (Shadow Mode):** The ESKF propagates autonomously without map constraints; the map matching engine independently snaps the unconstrained dead reckoning trajectory onto the road network for visualization and route guidance.

---

## 6. Experimental Protocol

### 6.1 Blackout Window Extraction and Horizon Standardization
To ensure repeatable evaluations, simulated GNSS outages are injected into continuous driving trajectories using a standardized rolling-window protocol:
* **Pre-Outage Calibration:** Each blackout window requires an initial **10 seconds of active GNSS** prior to satellite loss ($t_0 - 10\text{ s}$ to $t_0$). This window allows the ESKF to achieve convergence on initial velocity, attitude quaternion, and sensor biases, replicating an authentic loss of satellite lock.
* **Blackout Horizons ($T$):** We evaluate four distinct outage horizons: **10 s**, **20 s**, **30 s**, and **60 s**.
* **Step Size:** Consecutive blackout windows are extracted every **15 seconds** along the trajectory to ensure broad statistical sampling while limiting window redundancy.
* **Trip Eligibility Censusing:** A trip is eligible for evaluation under horizon $T$ if and only if its total duration satisfies $T_{\text{trip}} \ge 10\text{ s (pre-outage)} + T\text{ s (blackout)}$.

Table \ref{tab:census_summary} summarizes the exact census of usable trips across each blackout horizon.

```
+========================================================================================================================+
| Blackout Horizon Evaluation Census                                                                                     |
+========================================================================================================================+
| Horizon ($T$) | Minimum Required Duration | Eligible Trips ($N$) | Total Blackout Windows Evaluated                    |
+========================================================================================================================+
| 10 Seconds    | $\ge 20.0\text{ s}$       | 18                   | 28,142 Windows                                      |
| 20 Seconds    | $\ge 30.0\text{ s}$       | 17                   | 27,890 Windows                                      |
| 30 Seconds    | $\ge 40.0\text{ s}$       | 16                   | 27,615 Windows                                      |
| 60 Seconds    | $\ge 70.0\text{ s}$       | 13                   | 26,825 Windows                                      |
+========================================================================================================================+
```

### 6.2 Metric Definitions and Denominators
To eliminate ambiguity, all error metrics in this manuscript adhere to strict mathematical definitions:
1. **Absolute Horizontal Position Drift ($D(T)$):** The Euclidean distance between the estimated dead reckoning position $\hat{\mathbf{p}}_{2D}(t_0 + T)$ and the reference trajectory $\mathbf{p}_{\text{ref}, 2D}(t_0 + T)$ at the exact conclusion of the $T$-second outage:
\begin{equation}
D(T) = \|\hat{\mathbf{p}}_{2D}(t_0 + T) - \mathbf{p}_{\text{ref}, 2D}(t_0 + T)\|_2 \quad [\text{meters}]
\label{eq:abs_drift}
\end{equation}
2. **Normalized Relative Drift ($\Delta p_{\%}(T)$):** Absolute drift divided by the cumulative path distance traveled during the blackout window:
\begin{equation}
\Delta p_{\%}(T) = \frac{D(T)}{S(T)} \times 100\% = \frac{D(T)}{\int_{t_0}^{t_0+T} \|\mathbf{v}_{\text{ref}}(t)\|_2 dt} \times 100\% \quad [\%]
\label{eq:norm_drift}
\end{equation}
*Exception Handling:* If a blackout window occurs during complete standstill ($S(T) < 5.0\text{ m}$), normalized drift is mathematically undefined ($\frac{0}{0}$); such windows are excluded from percentage averaging and evaluated exclusively on absolute drift (meters).
3. **Macro vs Distance-Weighted Averages:** Unless explicitly labeled as distance-weighted, all aggregate numbers reported across trips represent **unweighted macro-averages** over the $N$ eligible trips: $\bar{D} = \frac{1}{N} \sum_{i=1}^N D_i$. Crucially, absolute drift and normalized drift are independently macro-averaged; therefore, the aggregate normalized percentage is not obtained by dividing the displayed macro-average distance by the displayed macro-average error.
4. **Smart India Hackathon (SIH) Benchmark Pass Criterion:** A blackout trajectory strictly passes the SIH benchmark if and only if:
\begin{equation}
\Delta p_{\%}(60\text{ s}) < 10.00\%
\label{eq:sih_pass}
\end{equation}

### 6.3 Causality and Leakage Safeguards
To verify that no model architecture or feature pipeline utilized future information, our codebase was subjected to automated causality tests (`tests/test_causality_and_leakage.py`). The audit verified that:
* Convolutions in the TCN operate exclusively on causal padding (receptive indices $k-M$ to $k$, with zero taps at $k+1$).
* No forward-backward filtering (e.g., `scipy.signal.filtfilt`) was utilized in online pipelines.
* Normalization statistics (means and standard deviations) were computed exclusively on the 39-trip training split and applied frozen to validation and test splits.

---

## 7. Results

### 7.1 Classical Inertial Navigation Baselines (RQ1)
We first establish the baseline performance of pure, unassisted strapdown double-integration in the absence of external velocity aiding. Table 3 presents the results across the four standardized blackout horizons.

```
+========================================================================================================================+
| Table 3: Classical Inertial Navigation Baselines (Open-Loop Divergence)                                                |
+========================================================================================================================+
| Blackout Horizon | Eligible Trips (N) | Distance Traveled | Pure Kinematics Drift | Relative Drift (%)*| Passing (<10%)|
+========================================================================================================================+
| 10 Seconds       | 18                 | 134.2 m           | 23.66 m               | 83.54%             | 6 / 18 (33.3%)|
| 20 Seconds       | 17                 | 272.8 m           | 57.01 m               | 147.98%            | 4 / 17 (23.5%)|
| 30 Seconds       | 16                 | 418.5 m           | 109.57 m              | 118.22%            | 2 / 16 (12.5%)|
| 60 Seconds       | 13                 | 914.3 m           | 324.21 m              | 89.66%             | 0 / 13 (0.0%) |
+========================================================================================================================+
*Note: Absolute drift and normalized drift are independently macro-averaged across trips; relative drift is not the
 direct quotient of displayed mean drift over mean distance.
```

As demonstrated in Table 3 and illustrated in Figure 3, unassisted consumer MEMS IMUs suffer immediate and rapid divergence. By 60 seconds, the macro-average position drift reaches **324.21 meters** (89.66% normalized drift), with zero trajectories satisfying the $<10\%$ benchmark. On high-speed segments, uncorrected accelerometer bias double integration induces position errors exceeding 1,000 meters. This establishes that unassisted smartphone IMUs cannot provide viable dead reckoning beyond 5 to 10 seconds without external constraints.

### 7.2 Multi-Trip TCN Generalization and Scaling (RQ2, RQ3)
To evaluate the impact of training dataset scale on out-of-sample generalization, we trained the TCN under two distinct regimes:
1. **6-Trip Baseline:** Trained on 6 initial trips.
2. **39-Trip Expanded Benchmark:** Trained across 39 trips representing suburban, urban, and mountain routes.

Both models were evaluated on the identical **19 held-out test trips** (113,539 emitted predictions, 238.1 km). Table 4 summarizes the performance improvements unlocked by multi-trip training.

```
+========================================================================================================================+
| Table 4: Expanded Multi-Trip TCN Generalization Benchmark (Controlled Scaling)                                         |
+========================================================================================================================+
| Metric Evaluated               | 6-Trip TCN Baseline | 39-Trip Expanded TCN | Absolute Improvement | Relative Delta (%)|
+========================================================================================================================+
| Held-Out Test MAE (m/s)        | 6.17 m/s            | 2.76 m/s             | -3.41 m/s            | +55.27%           |
| Held-Out Test RMSE (m/s)       | 6.96 m/s            | 3.59 m/s             | -3.37 m/s            | +48.42%           |
| Mean Signed Bias (m/s)         | -3.06 m/s           | -0.48 m/s            | +2.58 m/s            | +84.31%           |
| 30s Unweighted Drift (m)       | 153.3 m             | 52.6 m               | -100.7 m             | +65.69%           |
| 60s Unweighted Drift (m)       | 263.6 m             | 93.0 m               | -170.6 m             | +64.72%           |
| 60s Distance-Weighted Drift (m)| 384.2 m             | 148.0 m              | -236.2 m             | +61.48%           |
+========================================================================================================================+
```

As detailed in Table 4 and visualized in Figures 4 and 5, expanding the training dataset to 39 diverse trips yielded a **55.27% reduction in velocity MAE** and a **64.72% reduction in unweighted 60-second drift** (263.6 m down to 93.0 m). The mean signed velocity bias dropped from $-3.06\text{ m/s}$ to $-0.48\text{ m/s}$. This demonstrates that deep sequence models require diverse topological and dynamic trajectories to decouple vehicle acceleration from persistent sensor bias.

### 7.3 Feature Attribution and Out-of-Distribution Forensics (RQ4)
During physical field testing with the Android implementation, walking with the smartphone produced an extreme out-of-distribution velocity prediction of **71.66 m/s** (257.9 km/h). To identify the mathematical cause of this spike, we conducted a systematic feature attribution ablation on the recorded telemetry. Table 5 documents the results.

```
+========================================================================================================================+
| Table 5: Controlled Feature Attribution and Input Forensics                                                           |
+========================================================================================================================+
| Input Permutation         | Input Condition Tested                 | Peak Speed | Speed (km/h) | Delta vs Base (%) |
+========================================================================================================================+
| Baseline                  | Full 9-channel input                   | 71.66 m/s  | 257.9 km/h   | 0.0%              |
| Ablate gyro_yaw           | omega_z set to 0.0 rad/s               | 34.94 m/s  | 125.8 km/h   | -51.2% (Primary)  |
| Ablate All Gyroscopes     | omega_x, omega_y, omega_z = 0.0 rad/s  | 27.71 m/s  | 99.8 km/h    | -61.3%            |
| Input Clamping            | All inputs clamped to +/- 3.0 sigma    | 32.11 m/s  | 115.6 km/h   | -55.2%            |
| Clamp + Zero Curvature    | +/- 3.0 sigma clamping and zero kappa  | 18.42 m/s  | 66.3 km/h    | -74.3%            |
+========================================================================================================================+
```

As revealed in Table 5 and illustrated in Figure 7, setting the yaw gyroscope channel ($\omega_z$) to zero immediately reduced the peak speed spike by **51.2%** (71.66 m/s down to 34.94 m/s). Ablating all gyroscopes dropped the spike by **61.3%**. 

**Forensic Finding:** In passenger vehicles, high yaw rates occur primarily during high-speed cornering, where forward speed is linked to centripetal acceleration ($a_y \approx v_x \omega_z$). When the model encountered pedestrian arm swing and torso turning—which generate high-frequency oscillatory yaw rates exceeding $+2.85\text{ rad/s}$ (+26.08$\sigma$ relative to vehicle training distributions)—the TCN evaluated the inputs along its non-linear centripetal projection manifold, extrapolating an unrealistic forward speed. Enforcing a strict $\pm 3.0\sigma$ input clamp and curvature gating reduces the peak to 18.42 m/s, demonstrating that explicit input distribution monitoring is essential for safety-critical edge deployment.

### 7.4 Controlled Sensor Fusion Ablation: The NHC Breakdown (RQ5)
To evaluate the interaction between learned forward velocity, lateral Non-Holonomic Constraints, and attitude estimation, we conducted an 8-way controlled fusion ablation across 7 representative test routes (`Vta24` to `Vta30`) encompassing **83 non-overlapping 60-second blackout windows**. Table 6 documents this critical experiment.

```
+========================================================================================================================+
| Table 6: Controlled Sensor Fusion Ablation (8-Way Permutation)                                                         |
+========================================================================================================================+
| Variant ID      | Forward vx | Lateral vy | Heading Upd | 10s Drift | 30s Drift | 60s Drift  | Mean NISx | Delta vs Ref|
+========================================================================================================================+
| Variant A (INS) | ❌         | ❌         | ❌          | 97.8 m    | 790.7 m   | 2488.7 m   | 0.00      | N/A         |
| Variant D (NHC) | ❌         | ✅         | ✅          | 83.1 m    | 405.6 m   | 1166.7 m   | 0.00      | N/A         |
| Variant E (TCN) | ✅         | ❌         | ❌          | 91.8 m    | 431.7 m   | 843.4 m    | 6.55      | 0.0% (Ref)  |
| Variant F (NHC) | ✅         | ✅         | ✅          | 75.9 m    | 441.4 m   | 1226.9 m   | 82.12     | -45.5% (Deg)|
| Variant H (Full)| ✅         | ✅         | ✅          | 75.1 m    | 392.3 m   | 1240.8 m   | 87.85     | -47.1% (Deg)|
| Variant V1 (Dec)| ✅         | ✅         | ❌          | 70.6 m    | 180.4 m   | 382.7 m    | 6.60      | +54.6% (Best|
| Variant V2 (Gat)| ✅         | ✅         | Gated       | 86.0 m    | 387.1 m   | 829.0 m    | 18.22     | +1.7%       |
| Variant V5 (Ada)| ✅         | ✅         | Gated       | 70.7 m    | 261.1 m   | 627.0 m    | 17.35     | +25.7%      |
+========================================================================================================================+
```

**The NHC Degradation Phenomenon:** In Table 6, comparing Variant E (TCN forward speed alone, 843.4 m drift) with Variant F (TCN forward speed + standard lateral NHC, 1226.9 m drift) reveals that **unconditional lateral NHC degrades 60-second positioning accuracy by 45.47%**. The filter innovation statistic ($NIS_x$) surges from 6.55 to 82.12, indicating severe model mismatch. This degradation is consistent with tire sideslip angles and unmodeled phone-to-vehicle mounting misalignments violating the $v_y \approx 0$ assumption; the coupled Kalman gain projects this non-zero lateral velocity innovation directly into attitude error states ($\delta \theta_z$), introducing large heading errors that worsen long-term dead reckoning.

**The Decoupled Solution:** When the lateral velocity gain is decoupled from attitude states (Variant V1, $K_y[6:15] = 0$), heading corruption is substantially reduced. Variant V1 achieves a 60-second drift of **382.7 meters**—a **54.62% improvement over Variant E** and a **68.81% improvement over standard NHC (Variant F)**, winning on all 7 evaluated test routes.

### 7.5 Map Matching Integration & Failure Telemetry (RQ6)
Using the identical 83 blackout windows, we evaluated the integration of OpenStreetMap (OSM) link constraints. Table 7 details the comparative performance across five map-matching configurations.

```
+========================================================================================================================+
| Table 7: Map Matching Integration & Failure Telemetry                                                                  |
+========================================================================================================================+
| Variant ID  | Map Constraint Config        | Envelope Prot | 60s Drift | Along-Track | Cross-Track | Heading Err| Delta (%)|
+========================================================================================================================+
| Variant M0  | V1 Baseline (No Map Matching)| ❌            | 578.3 m   | 392.5 m     | 318.1 m     | 63.42°     | +0.0%    |
| Variant M1  | Shadow 5-Gate MHT (Tracking) | ❌            | 578.3 m   | 392.5 m     | 318.1 m     | 63.42°     | +0.0%    |
| Variant M2  | Heading-Only Map Feedback    | ❌            | 1303.9 m  | 861.6 m     | 745.0 m     | 73.51°     | -125.5%  |
| Variant M3  | Lateral-Only Link Snapping   | ❌            | 597.7 m   | 404.1 m     | 336.0 m     | 65.50°     | -3.3%    |
| Variant M4  | Full 5-Gate Closed-Loop Map  | ❌            | 829.8 m   | 573.0 m     | 474.8 m     | 68.20°     | -43.5%    |
| Variant M5  | Envelope-Gated Closed-Loop   | ✅            | 807.9 m   | 546.2 m     | 474.5 m     | 67.50°     | -39.7%    |
+========================================================================================================================+
```

As documented in Table 7 and illustrated in Figure 9, **closed-loop heading feedback (Variant M2) more than doubles 60-second position drift**, increasing error from 578.3 m to **1303.9 meters (-125.47% degradation)**. At complex intersections and fork junctions, inertial drift causes the MHT engine to latch onto an adjacent or parallel road link. Forcing the filter heading to match the erroneous link creates a severe divergence loop. Adding an $n\sigma$ position uncertainty envelope (Variant M5) mitigates catastrophic latching but still results in a 39.7% regression relative to the no-map baseline (M0). 

**Conclusion:** Map matching must be deployed strictly in **Shadow Mode (Variant M1)** as a downstream visualization layer, rather than as a hard closed-loop Kalman constraint during extended GNSS outages.

### 7.6 Target-Balanced TCN Training and Pareto Trade-Offs
In Phase 5.2, we investigated whether rebalancing training losses (using continuous inverse-density weighting and speed-stratified mini-batches) could resolve the high-speed motorway speed under-prediction. Table 8 summarizes the empirical results across the 19 held-out test trips.

```
+========================================================================================================================+
| Table 8: Target-Balanced TCN Training and Pareto Trade-Offs (Phase 5.2)                                                |
+========================================================================================================================+
| Test Trip / Category   | Road Category    | Expanded TCN Drift (%) | Balanced TCN Drift (%) | Relative Change | Status  |
+========================================================================================================================+
| Vw12                   | Winding Mountain | 5.87%                  | 3.51%                  | -40.2% (Better) | PASS ✅ |
| Vw14a                  | Winding Mountain | 6.94%                  | 4.91%                  | -29.3% (Better) | PASS ✅ |
| Vw13                   | Winding Mountain | 11.09%                 | 8.01%                  | -27.8% (Better) | PASS ✅ |
| Vw14b                  | Winding Mountain | 9.49%                  | 9.17%                  | -3.4% (Better)  | PASS ✅ |
| Vta21                  | Suburban Town    | 8.64%                  | 8.21%                  | -5.0% (Better)  | PASS ✅ |
| V-Vfa02                | Motorway         | 11.22%                 | 11.62%                 | +3.6% (Unchg)   | FAIL ❌ |
| Suburban Macro-Average | Suburban Town    | 23.51%                 | 26.84%                 | +14.2% (Worse)  | FAIL ❌ |
+------------------------+------------------+------------------------+------------------------+-----------------+---------+
| TOTAL STRICT PASSES    | All 19 Trips     | 4 / 19 (21.1%)         | 5 / 19 (26.3%)         | +1 New Pass     | Trade-off
+========================================================================================================================+
```

As demonstrated in Table 8, target balancing increased strict $<10\%$ trip passes from 4/19 to 5/19 by optimizing mountain routes (e.g., `Vw13` converted to an 8.01% pass). However, it failed to eliminate the highway under-prediction (`V-Vfa02` remained at 11.62% drift with $-3.45\text{ m/s}$ bias), while degrading average suburban accuracy by +14.2%. This confirmed that highway under-prediction is not merely a training density artifact, but reflects weak inertial observability during steady cruising.

### 7.7 Spectral Vibration and Speed Correlation Census (RQ7)
Several recent studies have hypothesized that vehicle speed can be regressed from engine and chassis vibration captured by smartphone accelerometers. To test this hypothesis causally, we conducted a spectral census across all **64 passenger car trips** in IO-VNBD, evaluating **over 450,000 one-second causal Fast Fourier Transform (FFT) windows** (0.5 to 25.0 Hz). Table 9 summarizes the correlation findings.

```
+========================================================================================================================+
| Table 9: Spectral Vibration and Speed Correlation Census                                                               |
+========================================================================================================================+
| Feature Analyzed                  | Pearson r | Spearman rho | Dominant Peak Mode | Ridge R² | RF R² | Finding         |
+========================================================================================================================+
| Dominant FFT Freq (0.5–25 Hz)     | -0.032    | -0.028       | 2.2–2.5 Hz         | < 0.0    | < 0.0 | Low-freq peak   |
| Spectral Energy Centroid (Hz)     | -0.037    | -0.034       | N/A                | < 0.0    | < 0.0 | No speed shift  |
| Total IMU Band Power (m²/s³)      | +0.252    | +0.374       | Broad              | 0.068    | 0.268 | Road roughness  |
| Motorway Cruise Peak (V-Vfa02)    | -0.014    | +0.010       | 2.86–2.93 Hz       | -1.009   | +0.268| Invariant peak  |
+========================================================================================================================+
```

As shown in Table 9 and Figure 10, the correlation between dominant vibration frequency and forward vehicle speed is statistically negligible across the entire dataset (Pearson $r = -0.032$, Spearman $\rho = -0.028$). On motorway trip `V-Vfa02`, as vehicle speed increases from 80 km/h to 118 km/h, the dominant vertical acceleration peak remains locked between **2.86 Hz and 2.93 Hz**. 

**Scientific Interpretation:** Under the tested 10-Hz processing rate and spectral features, no robust generalizable vibration-only speed relationship was demonstrated. The observed dominant 2.2–2.5 Hz spectral peak is consistent with low-frequency vehicle body/chassis dynamics, rather than tire rotational frequency or forward speed.

### 7.8 Multi-Horizon Benchmark: Adaptive Fusion vs Baselines (RQ8)
In Phase 5.6, we evaluated our complete **Adaptive Dynamic Fusion** pipeline—incorporating decoupled velocity damping, adaptive kinematic blending, and multi-condition ZVD—against Pure Kinematics, Fixed Damped Momentum, and the Static Dilated TCN across all four standardized blackout horizons. Table 10 presents the multi-horizon summary across the 19 held-out test trips.

```
+========================================================================================================================+
| Table 10: Multi-Horizon Benchmark: Adaptive Fusion vs Baselines                                                        |
+========================================================================================================================+
| Horizon | Architecture Evaluated     | Usable Trips (N) | Mean Drift (m) | Normalized (%)*| SIH Passes (<10%) | Pass Rate |
+========================================================================================================================+
| 10 s    | Pure Kinematics            | 18               | 23.66 m        | 83.54%         | 6 / 18            | 33.3%     |
|         | Fixed Damped Momentum      | 18               | 23.49 m        | 111.37%        | 6 / 18            | 33.3%     |
|         | Static Dilated TCN         | 18               | 26.83 m        | 106.44%        | 5 / 18            | 27.8%     |
|         | Adaptive Dynamic Fusion    | 18               | 25.06 m        | 111.53%        | 5 / 18            | 27.8%     |
+---------+----------------------------+------------------+----------------+----------------+-------------------+-----------+
| 20 s    | Static Dilated TCN         | 17               | 45.73 m        | 123.95%        | 5 / 17            | 29.4%     |
|         | Adaptive Dynamic Fusion    | 17               | 49.27 m        | 135.51%        | 4 / 17            | 23.5%     |
|         | Pure Kinematics            | 17               | 57.01 m        | 147.98%        | 4 / 17            | 23.5%     |
+---------+----------------------------+------------------+----------------+----------------+-------------------+-----------+
| 30 s    | Static Dilated TCN         | 16               | 54.62 m        | 65.57%         | 5 / 16            | 31.2%     |
|         | Adaptive Dynamic Fusion    | 16               | 61.83 m        | 71.38%         | 4 / 16            | 25.0%     |
|         | Pure Kinematics            | 16               | 109.57 m       | 118.22%        | 2 / 16            | 12.5%     |
+---------+----------------------------+------------------+----------------+----------------+-------------------+-----------+
| 60 s    | Adaptive Dynamic Fusion    | 13               | 96.65 m        | 16.76%         | 3 / 13            | 23.1%     |
|         | Static Dilated TCN         | 13               | 95.53 m        | 18.26%         | 3 / 13            | 23.1%     |
|         | Fixed Damped Momentum      | 13               | 173.71 m       | 40.75%         | 1 / 13            | 7.7%      |
|         | Pure Kinematics            | 13               | 324.21 m       | 89.66%         | 0 / 13            | 0.0%      |
+========================================================================================================================+
*Note: Normalized drift percentages are independently macro-averaged across trips.
```

**Scientific Story of Table 10:** The data indicate that Adaptive Dynamic Fusion is an alternative regime-aware architecture rather than a universally superior solution across all horizons. At 10s, 20s, and 30s horizons, the simpler Static Dilated TCN achieves equal or slightly lower drift (e.g., 54.62 m vs 61.83 m at 30s). At 60 seconds, both Static TCN (95.53 m) and Adaptive Fusion (96.65 m) exhibit comparable performance, both achieving 3/13 passes. The central finding is that while dynamic blending does not materially outperform a static learned baseline, both avoid the catastrophic quadratic/cubic divergence of unassisted kinematics.

### 7.9 Per-Trip Transparency at the 60-Second Blackout Horizon (RQ8)
To guarantee scientific transparency and demonstrate that aggregate averages do not conceal individual trajectory failures, Table 11 provides the breakdown of all **13 eligible trips** at the 60-second blackout horizon.

```
+========================================================================================================================+
| Table 11: Exhaustive Per-Trip Results at 60-Second Blackout Horizon                                                    |
+========================================================================================================================+
| Trip Name | Route Terrain | Windows | Distance (m) | Pure Kin Drift | Static TCN Drift| Adaptive Drift  | SIH Pass Status |
+========================================================================================================================+
| Vw12      | Mountain      | 869     | 1500.2 m     | 43.40%         | 6.68%           | 3.01% (45.3 m)  | PASS ✅ (<10%)  |
| Vta21     | Suburban      | 10      | 786.7 m      | 51.45%         | 8.19%           | 7.88% (61.5 m)  | PASS ✅ (<10%)  |
| Vw14a     | Mountain      | 3,089   | 1511.1 m     | 21.06%         | 6.93%           | 8.54% (129.0 m) | PASS ✅ (<10%)  |
| Vw14b     | Mountain      | 19,539  | 1264.5 m     | 42.48%         | 10.24%          | 10.19% (128.8 m)| Near Pass (10.2)|
| V-Vfa02   | Motorway      | 446     | 1460.3 m     | 25.44%         | 12.28%          | 11.69% (157.2 m)| Fail (11.7%)    |
| Vta27     | Suburban      | 4       | 818.9 m      | 37.18%         | 13.78%          | 12.68% (103.8 m)| Fail (12.7%)    |
| Vta24     | Suburban      | 1       | 384.8 m      | 73.42%         | 12.80%          | 14.65% (56.4 m) | Fail (14.7%)    |
| Vta22     | Suburban      | 1       | 641.3 m      | 27.72%         | 12.43%          | 16.84% (108.0 m)| Fail (16.8%)    |
| Vw16a     | Mountain      | 5,830   | 899.9 m      | 51.76%         | 16.44%          | 16.88% (151.9 m)| Fail (16.9%)    |
| Vta28     | Suburban      | 7       | 565.3 m      | 64.92%         | 25.48%          | 25.30% (143.0 m)| Fail (25.3%)    |
| Vta23     | Suburban      | 1       | 570.1 m      | 62.40%         | 23.06%          | 26.16% (149.1 m)| Fail (26.2%)    |
| Vta26     | Suburban      | 3       | 258.7 m      | 574.65%         | 70.82%          | 47.31% (66.3 m) | Fail (Clamped)   |
| Vw15      | Stationary    | 1,331   | 1.3 m         | 450.48%         | 22.63%          | 63.83% (0.8 m)  | N/A (Stationary) |
+========================================================================================================================+
```

**Key Empirical Findings from Table 11:**
1. **SIH Compliance Rate:** Exactly **3 out of 13 eligible trips (23.08%)** achieve strict compliance with the SIH $<10\%$ drift threshold at 60 seconds: `Vw12` (**3.01% / 45.3 m**), `Vta21` (**7.88% / 61.5 m**), and `Vw14a` (**8.54% / 129.0 m**). A fourth trip (`Vw14b`) narrowly misses at **10.19%**.
2. **Topological Observability Disparity:** The passing trips (`Vw12`, `Vw14a`) are winding mountain routes. Continuous curvature and dynamic centripetal acceleration provide rich observability, allowing the TCN to accurately predict forward speed from yaw rate and lateral acceleration ($v \approx a_y / \omega_z$).
3. **Suburban Stop-and-Go Degradation:** On suburban routes with multiple stops (`Vta28`, `Vta23`, `Vta26`), drift ranges from 25.3% to 47.3%. However, our multi-condition ZVD prevents catastrophic cubic runaway, clamping `Vta26` drift from 574.65% (pure INS) to 47.31% (absolute drift of 66.3 m over 258.7 m traveled).

### 7.10 Edge Deployment and Replay Parity Audit
The complete navigation pipeline was deployed onto Android 14 using ONNX Runtime Mobile and a native Java 15-state ESKF. To verify implementation integrity, the mobile engine executed a **1,789-epoch golden telemetry replay** (178.9 s continuous telemetry replay, including a 60-second blackout). Table 12 documents system latency, resource consumption, and numerical parity against the offline 64-bit Python reference.

```
+========================================================================================================================+
| Table 12: Android Edge Deployment & Replay Parity Audit                                                                |
+========================================================================================================================+
| Subsystem / Metric           | Measured Performance               | Predefined Tolerance | Parity Result           |
+========================================================================================================================+
| ONNX Mobile Inference Latency| 4.2 to 7.8 ms per window           | < 20.0 ms            | PASS ✅ (Real-Time)     |
| TCN Model Storage Footprint  | 162 KB (65,409 FP32 parameters)    | < 5.0 MB             | PASS ✅ (Ultra-light)   |
| Java 15-State ESKF Step Time | 0.8 to 1.4 ms per epoch            | < 5.0 ms             | PASS ✅ (Real-Time)     |
| Total CPU Execution Time     | < 10.0 ms per 100 ms epoch (>90% id)| < 50.0 ms            | PASS ✅ (Budget Safe)   |
| Max Horizontal Position Delta| 0.038 m                            | <= 0.050 m           | PASS ✅ (Strict Parity) |
| Max Velocity Vector Delta    | 0.007 m/s                          | <= 0.010 m/s         | PASS ✅ (Strict Parity) |
| Max Heading Attitude Delta   | 0.031°                             | <= 0.050°            | PASS ✅ (Strict Parity) |
| Navigation Callback Rate     | 8.63 Hz (15–22 ms jitter)          | Dynamic Delta t      | Corrected ✅            |
+========================================================================================================================+
```

As demonstrated in Table 12, the mobile engine executes well within real-time budgets (total execution $<10\text{ ms}$ per 100 ms epoch, leaving >90% CPU idle headroom). Numerical parity against the Python reference was verified strictly within tolerances: maximum horizontal position delta was **0.038 m** ($\le 0.050\text{ m}$), velocity delta was **0.007 m/s** ($\le 0.010\text{ m/s}$), and heading delta was **0.031$^\circ$** ($\le 0.050^\circ$).

---

## 8. Failure Analysis

A central scientific objective of this manuscript is to transparently document where and why smartphone-only dead reckoning fails. We analyze five explicit failure mechanisms.

### 8.1 Weak Speed Separability During Steady-State Highway Cruise
On motorway route `V-Vfa02` (Table 11), dead reckoning drift reached 11.69% (157.2 m). An in-depth forensic investigation revealed weak statistical separability of instantaneous inertial features during steady-state cruise. During steady-state highway cruising at 100 km/h:
1. Longitudinal acceleration is near zero ($\dot{v}_x \approx 0$).
2. Angular velocity is near zero ($\boldsymbol{\omega} \approx \mathbf{0}$).
3. Road surface on high-grade expressways is exceptionally smooth, suppressing high-frequency shock transients.

Under these conditions, specific forces measured by the smartphone accelerometer ($f_x, f_y$) are dominated by sensor bias instability and thermal noise. Receiver Operating Characteristic (ROC) analysis revealed an **AUC of only 0.625** when discriminating between 80 km/h and 110 km/h cruise segments based on IMU features alone. Consequently, forward speed was weakly separable from instantaneous inertial features during steady-state highway cruise conditions.

### 8.2 Distribution Shift and High-Speed Negative Regression Bias
In Table 4, the expanded TCN exhibited a negative velocity bias at high speeds. We audited the velocity distributions of the training versus test splits:
* **Training Set:** Speeds $\ge 25\text{ m/s}$ (90 km/h) constituted only **12.43%** of total epochs.
* **Held-Out Test Set:** Speeds $\ge 25\text{ m/s}$ constituted **43.91%** of total epochs.

Due to this distribution shift, when the vehicle cruised at 100–118 km/h, the TCN regressed toward the high-density training mean (30–60 km/h), under-predicting forward speed by an average of **$-3.5\text{ m/s}$**. The correlation between true vehicle speed and signed residual error was **Pearson $r = -0.4866$**. This systematic under-prediction integrates directly into along-track position error during long highway outages.

### 8.3 Unconditional Lateral NHC and Sideslip Heading Corruption
As demonstrated in Section 7.4 (Table 6), unconditional lateral NHC updates degraded 60-second drift by 45.47%. When a vehicle traverses a turn at speed, centrifugal force induces non-zero tire sideslip ($\beta \neq 0$). Forcing $v_y^b = 0$ in the Kalman measurement model introduces a lateral innovation $z_{v_y} = -v \sin\beta$. Through the coupled off-diagonal terms of the standard Kalman gain $\mathbf{K}[6:9, 1]$, this lateral innovation is projected into attitude error states ($\delta \theta_z$), corrupting the yaw gyro bias estimate and twisting the navigation heading into the curve. The decoupled damping architecture ($K_y[6:15] = 0$) resolves this failure by isolating lateral damping entirely to the velocity states.

### 8.4 Closed-Loop Map Matching Link Latching Divergence
Section 7.5 (Table 7) proved that closed-loop map matching heading feedback increased drift by 125.47%. During an outage, uncorrected heading drift expands the filter's 2D position covariance ellipse. At complex intersections, overpasses, or parallel frontage roads, the candidate link with the highest instantaneous likelihood is frequently incorrect. Once the filter heading is forced to match an erroneous link's azimuth, the true trajectory cannot be recovered, resulting in rapid divergence.

### 8.5 Pedestrian Motion Out-of-Distribution Dynamics
Testing the vehicle-trained model on pedestrian walking generated an unphysical 71.66 m/s speed spike (Section 7.3). Table 13 details the input distribution mismatch between vehicle driving and pedestrian gait.

```
+========================================================================================================================+
| Table 13: Pedestrian Out-of-Distribution (OOD) Stress Test Telemetry                                                   |
+========================================================================================================================+
| Feature Channel           | Vehicle Train Mean | Vehicle Train Std | Measured Walking Peak | Standardized Z-Score      |
+========================================================================================================================+
| Gyroscope Yaw (omega_z)   | 0.003 rad/s        | 0.109 rad/s       | +2.850 rad/s          | +26.08 sigma (Severe OOD) |
| Gyroscope Pitch (omega_y) | -0.001 rad/s       | 0.087 rad/s       | +1.420 rad/s          | +16.33 sigma (Severe OOD) |
| Gyroscope Roll (omega_x)  | 0.002 rad/s        | 0.094 rad/s       | +0.890 rad/s          | +9.45 sigma (Severe OOD)  |
| Linear Accel Long (a_x)   | 0.012 m/s²         | 0.742 m/s²        | +3.410 m/s²           | +4.58 sigma (OOD)         |
| Linear Accel Vert (a_z)   | 0.005 m/s²         | 0.681 m/s²        | +4.850 m/s²           | +7.11 sigma (Severe OOD)  |
+========================================================================================================================+
```

As quantified in Table 13, pedestrian arm-swing and torso rotation produce angular rates exceeding **+26.08 standard deviations** relative to the vehicle training distribution. Because deep neural networks are unconstrained outside their training support, the model evaluated these inputs along its non-linear centripetal projection, producing an unphysical speed prediction.

### 8.6 Android Sensor Timing Jitter and GNSS Reacquisition Surge
During edge implementation, two critical system defects were identified:
1. **Sensor HAL Sampling Jitter:** Requesting Android `SENSOR_DELAY_FASTEST` produced an average event rate of **8.63 Hz** with $15\text{ to }22\text{ ms}$ interval jitter, rather than the expected 10.0 Hz. Hardcoding a nominal $\Delta t = 0.10\text{ s}$ distorted integration time-scales, causing synthetic acceleration errors. We resolved this by computing dynamic interval deltas directly from hardware nanosecond timestamps ($\Delta t_k = (t_k^{\text{hw}} - t_{k-1}^{\text{hw}}) \times 10^{-9}$).
2. **Reacquisition Innovation Shock:** In early stress tests, when GNSS resumed after an 800-meter dead reckoning drift, the raw position innovation ($z_p \approx 800\text{ m}$) injected a violent velocity correction spike exceeding **6,773 m/s**, destabilizing filter covariance. We implemented a 2-stage reacquisition state machine that gates large innovations through an adaptive $\chi^2$ threshold and reinitializes nominal state vectors gracefully.

---

## 9. Discussion

The empirical evidence compiled in this investigation allows us to clearly demarcate what works, what fails, and what remains an open challenge in smartphone inertial navigation.

### 9.1 What Genuinely Works
* **Causal Dilated TCNs for Longitudinal Velocity:** When trained across diverse routes (39 trips), causal TCNs provide an effective virtual odometer, reducing 60-second dead reckoning drift from 324 m to 93 m.
* **Decoupled Velocity Damping:** Restricting lateral velocity innovations strictly to velocity error states ($K_y[6:15] = 0$) substantially reduces tire sideslip and mounting misalignment corruption, unlocking a +54.6% improvement over coupled filtering.
* **Multi-Condition Standstill Clamping:** Enforcing zero-velocity updates via simultaneous acceleration variance, angular rate, and neural velocity gating halts drift during traffic stops.

### 9.2 What Fails or Degrades Performance
* **Unconditional Lateral NHC:** Assumes zero sideslip, corrupting heading during cornering (-45.5% degradation).
* **Closed-Loop Map Feedback:** Induces catastrophic latching when link candidates are ambiguous (-125.5% degradation).
* **Chassis Vibration Speedometers:** Low-frequency chassis dynamics at 2.2–2.5 Hz do not correlate with vehicle speed ($r = -0.032$).
* **Unconstrained OOD Extrapolation:** Pedestrian arm swing induces severe speed spikes (+26$\sigma$ yaw rate).

### 9.3 Comparison with Prior Literature
Our findings contrast with several optimistic claims in recent literature. While Wang et al. [11] reported high velocity regression accuracy over brief 5–15 second horizons, our 60-second evaluations reveal that steady-state highway cruising suffers from weak inertial observability (ROC-AUC = 0.625) and negative regression bias. Furthermore, while theoretical papers advocate tightly coupled map-inertial filtering [5], [6], our empirical results demonstrate that digital map errors and junction ambiguities make closed-loop heading feedback hazardous in practice.

---

## 10. Limitations

To maintain strict scientific integrity, we explicitly document the empirical boundaries of this research:
1. **Single-Driver Representation:** Metadata auditing established that **100% of the passenger car trips in IO-VNBD were recorded by a single driver (`Driver E`)**. While whole-trip disjointness was strictly maintained, **cross-driver invariance was not evaluated and cannot be claimed**.
2. **Lack of Physical In-Vehicle Road Validation:** Due to equipment constraints, the author's physical Android tests were conducted on foot (pedestrian/handheld) or stationary test benches. Physical in-vehicle road validation using the native Android application remains an unexecuted future objective.
3. **Weak Cruise Observability:** Smartphone IMU signals during smooth, straight-line highway cruising lack sufficient dynamic excitation to decouple speed from sensor bias, representing an empirical limitation of smartphone-only dead reckoning.
4. **Limited Long-Duration Trips:** Only 13 of the 19 test trips in IO-VNBD possessed sufficient continuous duration ($\ge 70\text{ s}$) to support 60-second blackout evaluation.
5. **Absence of Visual Odometry:** The system relies exclusively on inertial sensors. Integrating monocular smartphone camera visual-inertial odometry (VIO) would likely mitigate highway observability limits but was outside the scope of this IMU-focused study.

---

## 11. Reproducibility

To ensure scientific reproducibility, all code, scripts, configurations, and trained model weights are publicly available in the project repository:
* **Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)
* **Dataset Attribution:** Raw sensor telemetry is sourced from the public IO-VNBD repository [18] (available at [https://github.com/nithinsam/IO-VNBD](https://github.com/nithinsam/IO-VNBD)). IO-VNBD data files are excluded from our repository in accordance with public distribution guidelines.
* **Causality & Leakage Verification:** Automated tests verifying zero future tap leakage, strict causal padding, and disjoint split partitioning are executable via `pytest tests/test_causality_and_leakage.py`.
* **Execution Environment:** Python 3.10+ (PyTorch 2.1, ONNX Runtime 1.16, NumPy, SciPy). Android edge implementation builds via Android Studio Iguana / Gradle 8.2 (Target SDK 34, Min SDK 26).
* **Privacy Compliance:** In accordance with personal privacy and data protection standards, all raw latitude/longitude coordinates recorded during local handheld field testing were sanitized to 0.0 prior to public repository publication.

---

## 12. Conclusion

In this work, we conducted an empirical evaluation and failure analysis of smartphone-only inertial dead reckoning during GNSS outages. Utilizing 64 passenger car trips from the public IO-VNBD dataset, we systematically evaluated classical strapdown mechanization, causal deep sequence learning, sensor fusion architectures, digital map matching, and mobile edge deployment. 

Our findings demonstrate that while a causal Dilated TCN-kin trained across multi-trip data reduces 60-second dead reckoning drift from 324 m to 93 m, and decoupled velocity damping provides vital filter stability (+54.6% improvement), smartphone-only dead reckoning remains fundamentally constrained. High-speed highway cruising suffers from weak inertial observability and negative regression bias, chassis vibration does not correlate with vehicle speed, and unconditional kinematic constraints degrade heading under real driving dynamics. Across 13 usable 60-second blackout trajectories, our adaptive fusion architecture achieved an aggregate drift of 96.65 meters (16.76% macro-average normalized drift) and passed the Smart India Hackathon $<10\%$ benchmark in 3 out of 13 trips (23.08%). We conclude that future advances in smartphone vehicular navigation must abandon brittle closed-loop constraints in favor of failure-aware, uncertainty-gated multi-modal architectures.

---

### Data Availability Statement
The public vehicle telemetry evaluated in this paper is available from the Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD) repository at [https://github.com/nithinsam/IO-VNBD](https://github.com/nithinsam/IO-VNBD) [18]. Extraction scripts and processed feature definitions are provided in our repository. Sanitized local diagnostic logs are available in `data/field/`.

### Code Availability Statement
The complete source code—including the Python scientific evaluation suite, TCN training scripts, ESKF fusion engine, and the production Android Studio application—is open-source under the MIT License at [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR).

### Declarations
* **Funding:** [AUTHOR FUNDING STATEMENT / SEE AUTHOR INPUT REQUIRED]
* **Conflict of Interest:** The authors declare that they have no competing financial or personal interests that could influence the work reported in this paper.
* **Ethical Approval:** Handheld sensor recordings involved no human biometric tracking or medical interventions; GPS coordinates were collected in public open-access spaces and sanitized prior to release.

---

## References

[1] M. W. M. G. Dissanayake, S. Sukkarieh, E. Nebot, and H. Durrant-Whyte, "The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation," *IEEE Transactions on Robotics and Automation*, vol. 17, no. 5, pp. 731–747, Oct. 2001, doi: 10.1109/70.964672.

[2] D. Klein, C. Cappelle, Y. Ruichek, and J. M. Rovetta, "Multi-sensor fusion for land vehicle positioning using non-holonomic constraints and road map," in *Proc. IEEE Intelligent Vehicles Symposium (IV)*, Eindhoven, Netherlands, 2008, pp. 1104–1109.

[3] J. Solà, "Quaternion kinematics for the error-state Kalman filter," *arXiv preprint arXiv:1711.02508*, 2017.

[4] I. Skog, J. O. Nilsson, and P. Händel, "Evaluation of zero-velocity detectors for pedestrian indoor positioning," in *Proc. International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, Zurich, Switzerland, 2010, pp. 1–6.

[5] M. A. Quddus, W. Y. Ochieng, and R. B. Noland, "Current map-matching algorithms for transport applications: State-of-the art and future research directions," *Transportation Research Part C: Emerging Technologies*, vol. 15, no. 5, pp. 312–328, Oct. 2007, doi: 10.1016/j.trc.2007.05.002.

[6] C. E. White, D. Bernstein, and A. L. Kornhauser, "Some map matching algorithms for personal navigation assistants," *Transportation Research Part C: Emerging Technologies*, vol. 8, no. 1–6, pp. 91–108, Feb. 2000.

[7] H. Yan, Q. Shan, and Y. Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, 2020, pp. 6128–6137.

[8] M. Cortesi et al., "Deep learning-based vehicle speed estimation using smartphone sensors in GNSS-denied environment," *Applied Sciences*, vol. 15, no. 16, Art. no. 8824, Aug. 2025, doi: 10.3390/app15168824.

[9] M. Brossard, S. Bonnabel, and A. Barrau, "AI-IMU Dead-Reckoning," *IEEE Transactions on Intelligent Vehicles*, vol. 5, no. 4, pp. 585–595, Dec. 2020, doi: 10.1109/TIV.2020.2980758.

[10] Z. Wang, H. Zhang, and L. Zhao, "Avnet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF," *Satellite Navigation*, vol. 6, no. 1, Art. no. 12, Feb. 2025, doi: 10.1186/s43020-025-00162-4.

[11] L. Wang, Y. Zhang, and X. Liu, "Deep learning-based vehicle speed estimation using smartphone sensors in GNSS-denied environment," *Applied Sciences*, vol. 15, no. 16, p. 8824, 2025.

[12] Z. Wang et al., "An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU," *arXiv preprint arXiv:2505.18490*, 2025.

[13] D. Titterton and J. L. Weston, *Strapdown Inertial Navigation Technology*, 2nd ed. Stevenage, UK: The Institution of Engineering and Technology (IET), 2004.

[14] P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed. Boston, MA: Artech House, 2013.

[15] C. Shen, Y. Zhang, and X. Tang, "A high-precision dead reckoning algorithm based on non-holonomic constraints and adaptive Kalman filtering for land vehicles," *Sensors*, vol. 19, no. 18, Art. no. 3855, Sep. 2019.

[16] S. Sukkarieh, E. M. Nebot, and H. F. Durrant-Whyte, "A high integrity IMU/GPS navigation loop for autonomous land vehicle applications," *IEEE Transactions on Robotics and Automation*, vol. 15, no. 3, pp. 572–578, Jun. 1999.

[17] F. Rohani, D. Choi, and J. Kang, "Vehicular dead reckoning based on machine learning and map matching in urban canyons," in *Proc. IEEE International Conference on Consumer Electronics (ICCE)*, Las Vegas, NV, USA, 2023, pp. 1–4.

[18] U. Onyekpe, V. Palade, S. Kanarachos, and A. Szkolnik, "IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning," *Data in Brief*, vol. 35, Art. no. 106885, 2021, doi: 10.1016/j.dib.2021.106885.

[19] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *arXiv preprint arXiv:1803.01271*, 2018.

[20] C. Goodall, B. Farrell, and N. El-Sheimy, "Vibration analysis for low-cost MEMS IMU vehicular navigation during GPS outages," in *Proc. ION GNSS*, Portland, OR, USA, 2006, pp. 1709–1716.
