# Adversarial Peer Reviewer Attack & Defense Audit

This document simulates an exhaustive, hostile peer review of the research paper. Thirty rigorous adversarial challenges are presented and answered strictly using verified empirical evidence from the repository, followed by a formal **Reviewer Objection Table** prioritizing remaining gaps by severity.

**Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)

---

## 1. Thirty Hostile Reviewer Questions & Evidence-Based Answers

### Q1: What exactly is novel about this work? Deep learning for smartphone IMUs and Kalman filtering are already well known.
* **Reviewer Challenge**: TCNs, LSTMs, and ESKFs are standard tools. Where is the fundamental algorithmic contribution?
* **Evidence-Based Answer**: The manuscript explicitly disclaims inventing new neural network blocks or the Kalman filter. The contribution is a **systematic, failure-aware empirical study across 64 vehicle trips** that exposes three fundamental phenomena overlooked in prior literature:
  1. Empirical finding that chassis vibration does not provide a generalizable speed signal ($r = -0.032$ across 450,000 windows; low-frequency chassis dynamics remain centered around 2.2–2.5 Hz).
  2. Evidence of weak statistical separability of instantaneous inertial features during steady highway cruise (ROC-AUC = 0.625; constant-speed IMU signals are dominated by sensor noise floor, causing neural models to regress toward the training mean).
  3. Evidence that unconditional lateral NHC updates and closed-loop map-heading feedback severely degrade navigation (consistent with sideslip Jacobian errors and link latching), and demonstration that decoupled velocity damping ($K_y[6:15] = 0$) substantially improves filter stability (+54.6% drift reduction).
* **Evidence Source**: `results/vibration_audit/`, `results/highway_observability/`, `results/phase4_4_fusion/`, `results/phase4_5_adaptive/`.

### Q2: Isn't this merely an engineering system integration rather than scientific research?
* **Reviewer Challenge**: You combined an off-the-shelf TCN with an off-the-shelf ESKF on Android. Isn't this just software engineering?
* **Evidence-Based Answer**: Engineering integration asks "does the system run?" Scientific research asks "what physical mechanisms govern the boundary conditions of the system?" We conducted controlled ablations isolating the failure modes:
  * Why does NHC degrade drift by 45.5%? (Cornering sideslip and mounting misalignment inject false lateral innovations into attitude error states).
  * Why does map feedback double drift from 578m to 1304m? (Small heading errors latch onto adjacent links, creating positive-feedback error loops).
  * Why did walking trigger a 71.66 m/s speed spike? (Pedestrian arm swing introduces $+26.08\sigma$ out-of-distribution yaw rates).
  These are experimentally isolated failure mechanisms, with physically motivated interpretations where the underlying disturbance was not directly instrumented.
* **Evidence Source**: `docs/experiments/phase5_1_field_forensics.md`, `results/phase4_4_fusion/phase4_4_fusion_report.md`.

### Q3: Is the IO-VNBD dataset too narrow to draw generalizable conclusions?
* **Reviewer Challenge**: You evaluated on trips from a single car model in the UK. How do you know these findings hold for SUVs, electric vehicles, or different road surfaces?
* **Evidence-Based Answer**: This is an acknowledged limitation. The 64 trips cover 4 distinct road topologies (motorway, suburban town, winding mountain, dense urban) over 32.8 hours. While suspension damping frequencies may shift slightly across vehicle classes, the fundamental physical constraints—lack of longitudinal acceleration during constant cruising, tire sideslip during cornering, and the absence of step impacts—are universal to all wheeled ground vehicles. Cross-vehicle generalization is explicitly scoped as an open research challenge.
* **Evidence Source**: `docs/limitations.md`, `docs/paper/methodology_audit.md`.

### Q4: Is the test split truly independent?
* **Reviewer Challenge**: Many ML navigation papers accidentally leak data by splitting overlapping time windows. Did your test set contain windows from the training trips?
* **Evidence-Based Answer**: The split is strictly trip-disjoint. Zero temporal or window overlap exists between the 39 training trips and the 19 test trips. Training, validation, and test trips were collected on different dates and different routes. Trajectory isolation was verified via automated assertions in `tests/test_causality_and_leakage.py`.
* **Evidence Source**: `tests/test_causality_and_leakage.py#L9-L10`.

### Q5: Is there driver leakage? Did the same driver appear in both train and test sets?
* **Reviewer Challenge**: If Driver E drove the training trips and Driver E also drove the test trips, your model may have simply memorized one person's throttle habits.
* **Evidence-Based Answer**: **CRITICAL AUDIT ACKNOWLEDGMENT**: 100% of the 64 passenger car trips in the evaluated IO-VNBD subset were operated by Driver E. While earlier research notes informally discussed "unseen drivers," the manuscript explicitly disclaims cross-driver generalization. Trip-level generalization across different routes and speeds was achieved, but cross-driver invariance was not evaluated.
* **Evidence Source**: `scripts/experiments/run_expanded_tcn_forensics.py#L108`, `docs/paper/methodology_audit.md`.

### Q6: Why do you treat vehicle CAN bus speed as "ground truth"? CAN speed has tire slip and quantization errors.
* **Reviewer Challenge**: CAN speed is derived from transmission encoders and varies with tire pressure and wear. Calling it ground truth is scientifically flawed.
* **Evidence-Based Answer**: The manuscript does not call CAN speed "absolute ground truth." Phase 0 formally evaluated the reference hierarchy: CAN bus speed was found to have a calibrated rolling radius error of $\pm 0.003\text{ m}$ and an ECU digital filtering delay of exactly $100\text{ ms}$. All CAN timestamps were shifted by $-100\text{ ms}$ to maintain phase alignment. CAN speed is strictly designated as the **engineering reference velocity**, and its limitations are documented.
* **Evidence Source**: `docs/experiments/phase0_reference_audit.md`.

### Q7: Why do you criticize smartphone GPS speed when developing a smartphone navigation system?
* **Reviewer Challenge**: If you have smartphone GPS, why not use it to calibrate during outages?
* **Evidence-Based Answer**: Smartphone GNSS is single-frequency and suffers from multipath reflections, clock drift, and constellation dilution (GDOP) in urban canyons, producing speed jitter of $0.4\text{--}1.2\text{ m/s}$ even when stationary. Using raw smartphone GPS Doppler during degraded reception corrupts the Kalman filter's bias states. More importantly, during a continuous tunnel blackout, GPS is completely absent ($0\text{ Hz}$).
* **Evidence Source**: `docs/experiments/phase0_reference_audit.md`, `android/README.md`.

### Q8: Why did you exclude wheel speed sensors if you want <10% drift?
* **Reviewer Challenge**: Real car dead reckoning uses wheel speed sensors. Why handicap your system by excluding them?
* **Evidence-Based Answer**: Smart India Hackathon Problem Statement 26168 explicitly requires a **standalone consumer smartphone solution** without OBD-II dongles, CAN bus wiring, or vehicle hardware modifications. The entire scientific challenge is to determine how far smartphone-only inertial navigation can be pushed when physical wheel odometry is completely absent.
* **Evidence Source**: `docs/problem_statement.md`.

### Q9: Why is the model vehicle-trained? Can it navigate bicycles, motorcycles, or pedestrians?
* **Reviewer Challenge**: A practical navigation app must handle multi-modal travel. Why restrict training to cars?
* **Evidence-Based Answer**: Ground vehicles satisfy strict Non-Holonomic Constraints ($v_y \approx 0, v_z \approx 0$) that allow 2D forward speed to bound 3D spatial drift. Pedestrians, bicycles, and hand-held motion violate these kinematic constraints (e.g., side-stepping, arm swinging, banking). Training a single model across all modalities without dynamic mode recognition causes severe negative transfer. The model was intentionally bounded to automotive physics.
* **Evidence Source**: `docs/experiments/phase5_1_field_forensics.md`.

### Q10: Why does the model persistently fail at high speed on highways?
* **Reviewer Challenge**: Why does your model plateau at ~85 km/h on motorways when actual speed is 110–120 km/h? Is it just poor training?
* **Evidence-Based Answer**: It is not poor training. We verified this by testing inverse-density loss weighting, speed-stratified mini-batches, and piecewise calibration (Phase 5.2). None resolved the highway plateau. Phase 5.4 investigated the underlying reason: on a smooth, flat highway during constant cruise, forward acceleration is near zero ($a_x \approx 0$) and yaw rate is near zero ($\omega_z \approx 0$). The inertial readings at 80 km/h and 110 km/h are statistically weakly separable from sensor noise (ROC-AUC = 0.625). Without acceleration or turns, forward speed is weakly observable from instantaneous inertial features.
* **Evidence Source**: `results/highway_observability/highway_observability_summary.json`.

### Q11: Why did your model output 71.66 m/s (258 km/h) when walking?
* **Reviewer Challenge**: That is an absurd prediction. Doesn't that prove the model is fragile and unusable?
* **Evidence-Based Answer**: It proves the fundamental vulnerability of deep neural networks under out-of-distribution (OOD) dynamic shift. The TCN learned that high yaw rates coupled with horizontal acceleration represent high-speed turns ($\kappa \propto v^2/R$). Human arm swing reached $+2.85\text{ rad/s}$ ($+26.08\sigma$ outside the vehicle training distribution). The linear projection layer multiplied this extreme input into 71.66 m/s. This failure is documented in the paper to highlight the necessity of input $\sigma$-clipping and kinematic bounds.
* **Evidence Source**: `results/phase4_1_forensics/tcn_kin_forensic_report.md`.

### Q12: Why did map matching make dead reckoning worse? Every commercial GPS uses map matching.
* **Reviewer Challenge**: Commercial apps like Google Maps use map matching successfully. Why did yours degrade drift from 578m to 1304m?
* **Evidence-Based Answer**: Commercial apps use map matching as a **downstream visualization snapping tool** while GNSS fixes are actively bounding the position uncertainty ellipse. In our experiment (Phase 4.6), we tested **closed-loop feedback into the Kalman filter's attitude and velocity states** during extended 60s blackouts. When open-loop gyro integration develops a $10^\circ$ heading error, the filter snaps to an adjacent parallel road, injecting false heading corrections that steer the vehicle further off-course. Map matching during outages is safe for visualization, but dangerous for closed-loop state feedback.
* **Evidence Source**: `results/phase4_6_map/phase4_6_map_matching_report.md`.

### Q13: Why did Non-Holonomic Constraints (NHC) increase error during turns?
* **Reviewer Challenge**: Textbook navigation theory states NHC ($v_y = 0$) prevents lateral drift. Why did your filter degrade by 45.5%?
* **Evidence-Based Answer**: The standard NHC observation Jacobian assumes zero sideslip ($v_y^v = 0$). In reality, pneumatic tires generate dynamic slip angles during cornering, and smartphone mounts have non-zero alignment errors. During sustained cornering, true lateral velocity is non-zero ($v_y \ne 0$). The Kalman filter misinterpreted this lateral innovation as an attitude yaw error $\delta \theta_z$, applying a fictitious heading correction that corrupted the vehicle's heading.
* **Evidence Source**: `results/phase4_4_fusion/phase4_4_fusion_report.md`.

### Q14: Why is there no real-car field validation with the physical Android app?
* **Reviewer Challenge**: You built an Android app but only tested it on walking students. Without an in-car road test, your mobile implementation claims are unverified.
* **Evidence-Based Answer**: **CRITICAL AUDIT ACKNOWLEDGMENT**: The author did not have access to an instrumented motor vehicle for physical road validation. The Android app was verified via:
  1. Offline golden replay of 1,789 real vehicle epochs achieving numerical parity with Python ($\le 0.05\text{ m}$ position divergence).
  2. Live physical hardware tests verifying sensor HAL callback timing, execution budget headroom, and dynamic timestamp calculation.
  The absence of physical road driving validation is explicitly highlighted as the primary limitation of the manuscript.
* **Evidence Source**: `docs/limitations.md`, `android/README.md`.

### Q15: Does the system actually meet the SIH <10% drift target?
* **Reviewer Challenge**: Your hackathon objective was <10% drift at 60 seconds. Did you achieve it?
* **Evidence-Based Answer**: **NO, NOT UNIVERSALLY.** Exactly 3 out of 13 usable trips (23.08%) passed the <10% threshold at 60 seconds. Macro-average normalized drift across all 13 trips was 16.76% (96.65 m). The manuscript explicitly states that universal compliance was not achieved.
* **Evidence Source**: `results/adaptive_fusion/adaptive_aggregate_summary.csv#L17`.

### Q16: How many trajectories actually pass the 10% benchmark?
* **Reviewer Challenge**: Which trips passed, and why did they pass while others failed?
* **Evidence-Based Answer**: Exactly three trips passed at 60 seconds:
  1. `Vw12` (Mountain): **3.01% drift** (45.30 m error over 1500 m travel).
  2. `Vta21` (Suburban): **7.88% drift** (61.52 m error over 787 m travel).
  3. `Vw14a` (Mountain): **8.54% drift** (129.04 m error over 1511 m travel).
  These trips passed because they featured dynamic turns and continuous motion where lateral centripetal acceleration ($a_y = v \cdot \omega$) provides velocity observability. Straight motorway trips (`V-Vfa02` at 11.69%) and urban stop-and-go trips (`Vta26` at 47.31%) failed.
* **Evidence Source**: `results/adaptive_fusion/adaptive_per_trip_results.csv`.

### Q17: Are your aggregate averages hiding catastrophic failures?
* **Reviewer Challenge**: An aggregate drift of 16.76% sounds moderate. Does that conceal trips that drifted thousands of percent?
* **Evidence-Based Answer**: The per-trip breakdown is fully reported. Under pure kinematics, `Vta26` suffered rapid runaway (**574.65% drift**). Under Adaptive Fusion, causal ZVD clamped this down to **47.31%**. Trip `Vw15` (stationary control) showed 63.83% normalized drift because the vehicle moved only 1.3 meters, making relative percentage mathematically inflated (absolute error was 0.82 m). All per-trip metrics are published transparently.
* **Evidence Source**: `results/adaptive_fusion/adaptive_per_trip_results.csv#L95-L107`.

### Q18: Are your comparisons statistically meaningful?
* **Reviewer Challenge**: You report mean drift reductions (e.g. 54.6%), but did you test for statistical significance?
* **Evidence-Based Answer**: The 83 non-overlapping 60s blackout windows across 7 test routes in Phase 4.4 and 4.5 provide consistent pairwise improvements: Variant V1 beat Variant E on all 7 evaluated test routes. The manuscript reports absolute differences and percentage changes directly from empirical data without claiming unverified p-values.
* **Evidence Source**: `results/phase4_5_adaptive/phase4_5_adaptive_report.md`.

### Q19: Are your baseline comparisons fair?
* **Reviewer Challenge**: Did you tune the classical baselines poorly to make your ML model look better?
* **Evidence-Based Answer**: All baselines used identical pre-outage calibration protocols (10.0 seconds of GNSS availability for accelerometer and gyro bias initialization), identical causal coordinate transformations, identical 10.0 Hz sampling rates, and identical error-state Kalman filter covariance matrices ($Q, R$). No baseline was artificially degraded.
* **Evidence Source**: `scripts/experiments/run_adaptive_fusion_benchmark.py`.

### Q20: Is the Android implementation genuinely real-time?
* **Reviewer Challenge**: Running a deep neural network and an ESKF on a smartphone at 10 Hz can cause thermal throttling and frame drops. Did you measure inference latency?
* **Evidence-Based Answer**: Yes. The ONNX Runtime model (`tcn_velocity_expanded.onnx`) is lightweight (65k parameters, 162 KB) and evaluates in $<8\text{ ms}$ on mobile CPU threads. The 15-state ESKF takes $<1.5\text{ ms}$ per step. The total execution time per 100 ms epoch is $<10\text{ ms}$, leaving over 90% CPU idle time and operating well within the device execution budget.
* **Evidence Source**: `android/README.md`.

### Q21: The Android sensor loop ran at ~8.6 Hz instead of 10.0 Hz. Doesn't that invalidate the TCN's 10 Hz training?
* **Reviewer Challenge**: If the model was trained on 10.0 Hz data and your phone samples at 8.6 Hz, temporal scale distortion will degrade accuracy.
* **Evidence-Based Answer**: This is an identified engineering finding. In early Android tests, assuming fixed $dt = 0.1\text{ s}$ while the sensor loop operated at 8.6 Hz caused a 14% integration scale distortion. We corrected this by implementing hardware monotonic timestamp delta interpolation ($\Delta t_{\text{hw}}$), ensuring that kinematic integration uses the true elapsed nanoseconds. The slight temporal mismatch in the TCN receptive field (6.1s vs 7.0s) had minor impact compared to the kinematic timing fix.
* **Evidence Source**: `android/README.md`, `results/c10_5_test_protocol.md`.

### Q22: Is the TCN strictly causal?
* **Reviewer Challenge**: Many dilated convolutional models inadvertently use centered padding, creating future lookahead. Can you prove zero leakage?
* **Evidence-Based Answer**: Mathematical causality was formally verified via automated gradient perturbation in `tests/test_causality_and_leakage.py`. For intermediate evaluation timesteps, gradients with respect to all future inputs ($t > t_{\text{eval}}$) were confirmed to be strictly zero ($\nabla_{\text{future}} = 0.0000000000$). The model uses left-side padding exclusively.
* **Evidence Source**: `tests/test_causality_and_leakage.py#L29-L76`.

### Q23: Was future GNSS or map data used anywhere in the evaluation?
* **Reviewer Challenge**: Did the blackout evaluation know the post-outage trajectory or destination?
* **Evidence-Based Answer**: No. The blackout simulator strictly cuts off GNSS fixes at $t = t_{\text{blackout\_start}}$. During the 60-second window, the system receives zero position, velocity, or DOP measurements. Pre-outage GNSS data is used strictly during the 10-second calibration window preceding the outage.
* **Evidence Source**: `src/simulation/blackout_simulator.py`.

### Q24: Is the vibration conclusion overclaimed? Just because your 10 Hz phone didn't detect vibration doesn't mean vibration doesn't correlate with speed.
* **Reviewer Challenge**: High-frequency tire noise occurs at 50–500 Hz. A 10 Hz or 100 Hz smartphone IMU cannot observe high-frequency harmonics. Claiming "vibration contains no speed information" is unscientific.
* **Evidence-Based Answer**: The manuscript explicitly qualifies this claim:
  * Safe Wording: *"Under the tested smartphone IMU sampling rate (100 Hz native, 10 Hz processing) and passenger car platforms, spectral features derived from chassis vibration showed no generalizable correlation with vehicle speed ($r = -0.032$)."*
  * It does NOT claim that 1 kHz wheel accelerometers cannot measure vibration.
* **Evidence Source**: `docs/claim_evidence_matrix.md#L20`, `results/vibration_audit/`.

### Q25: Is the "highway observability" discussion mathematically justified, or just an empirical excuse?
* **Reviewer Challenge**: Is speed unobservable in a formal Kalman observability sense, or did your ML model simply underfit?
* **Evidence-Based Answer**: It is justified by classical Newtonian mechanics. In the body frame of a vehicle traveling at constant velocity $\mathbf{v} = [v, 0, 0]^T$ on a straight, level road:
  $$\mathbf{a}_{\text{true}} = \dot{\mathbf{v}} + \boldsymbol{\omega} \times \mathbf{v} = \mathbf{0} + \mathbf{0} = \mathbf{0}$$
  The accelerometer measures specific force $\mathbf{f} = \mathbf{a} - \mathbf{g} = -\mathbf{g}$. The measured specific force is identical whether $v = 10\text{ m/s}$ or $v = 30\text{ m/s}$. Speed appears primarily in centripetal acceleration ($a_y = v \omega_z$) and longitudinal transients ($a_x = \dot{v}$). When both $\dot{v} \approx 0$ and $\omega_z \approx 0$, forward velocity has minimal projection onto the IMU measurement vector.
* **Evidence Source**: `docs/experiments/phase5_4_highway_observability.md`.

### Q26: Are the results device-specific? Would an iPhone or Samsung produce different drift?
* **Reviewer Challenge**: Sensor noise density and bias stability vary widely across MEMS manufacturers (Bosch, STMicroelectronics, InvenSense). How generalizable are these numbers?
* **Evidence-Based Answer**: The IO-VNBD dataset utilized a Samsung Galaxy smartphone, while our edge deployment testing used a OnePlus Nord smartphone. While the absolute drift magnitude will scale with the sensor's Allan variance noise parameters, the structural failure mechanisms (cubic bias integration, highway cruise unobservability, sideslip NHC corruption, pedestrian OOD yaw rates) are fundamental physical properties independent of the specific silicon vendor.
* **Evidence Source**: `docs/limitations.md`.

### Q27: Could a loose phone mount invalidate the coordinate alignment?
* **Reviewer Challenge**: What happens if the phone slips in its mount during a hard brake?
* **Evidence-Based Answer**: The coordinate alignment layer estimates the gravity vector dynamically via low-pass filtering and aligns the forward longitudinal axis via Principal Component Analysis (PCA) during pre-outage acceleration transients. However, sudden mount slipping during an active GNSS blackout will misalign the forward velocity vector and inject lateral position drift. Dynamic continuous re-leveling during outages is an open challenge.
* **Evidence Source**: `docs/architecture.md`.

### Q28: Is the proposed architecture fully reproducible?
* **Reviewer Challenge**: Can another researcher clone your repository and reproduce every number in your tables?
* **Evidence-Based Answer**: Yes. The repository provides self-contained reproduction scripts for every experiment:
  * `python scripts/experiments/run_adaptive_fusion_benchmark.py` (Table 10 & 11)
  * `python scripts/experiments/run_vibration_speed_audit.py` (Vibration correlations)
  * `python scripts/experiments/run_highway_speed_observability.py` (ROC-AUC 0.625)
  * `python tests/test_causality_and_leakage.py` (Causality verification)
  The raw IO-VNBD dataset must be downloaded from its official source due to size (2.34 GB), but all preprocessing and split scripts are fully automated.
* **Evidence Source**: `docs/reproducibility.md`.

### Q29: Are map errors and wrong-road errors adequately analyzed?
* **Reviewer Challenge**: You tested map matching on 7 routes. Did you analyze road network density, parallel link spacing, or junction complexity?
* **Evidence-Based Answer**: Yes. Phase 4.6 recorded telemetry across all 83 blackout windows, tracking gate rejection reasons: 55.9% of rejections were due to candidate ambiguity (`gate1_ambiguous` at junctions), and 20.9% were due to covariance envelope dilution (`envelope_diluted`). On suburban parallel roads with $<15\text{ m}$ lateral separation, heading errors of $<5^\circ$ caused incorrect link latching, leading to the -125.5% drift degradation in Variant M2.
* **Evidence Source**: `results/phase4_6_map/phase4_6_map_matching_report.md#L37-L45`.

### Q30: What single additional experiment would most strengthen this paper?
* **Reviewer Challenge**: If you had one more month of funding, what would you do to make this paper bulletproof?
* **Evidence-Based Answer**: An instrumented in-vehicle road drive using the OnePlus smartphone mounted in a passenger car, with synchronized RTK-GNSS ground truth, driving through actual road tunnels and urban underpasses. This would convert the current hardware stress testing into end-to-end physical vehicle validation.
* **Evidence Source**: `docs/paper/author_input_required.md`.

---

## 2. Reviewer Objection Table

| Reviewer Concern | Evidence Available | Current Answer | Severity | How Manuscript Addresses It | Remaining Gap |
|:---|:---|:---|:---:|:---|:---|
| **No In-Car Physical Road Validation** | Handheld walking & stationary phone tests (`test_b_walking_*.csv`) | The author lacked access to an instrumented test car; Android tests were hardware/timing stress tests. | **CRITICAL** | Explicitly stated in Abstract, Introduction, Methodology, and Limitations. Never claimed as "road test." | Requires future instrumented road driving trials with RTK reference. |
| **All Car Trips from Driver E** | Dataset metadata inspection in `scripts/experiments/run_expanded_tcn_forensics.py` | 100% of the 64 passenger car trips in IO-VNBD belong to Driver E. | **HIGH** | Disclaims cross-driver generalization. Clearly frames evaluation as trip-level disjoint cross-route generalization. | Evaluation across multiple independent drivers on identical routes. |
| **Universal <10% SIH Target Not Met** | Exact counts in `adaptive_aggregate_summary.csv` | Only 3 of 13 usable trips (23.1%) pass <10% at 60 seconds; mean drift is 16.76%. | **HIGH** | Explicitly reported in Abstract and Results. Framed as a failure-aware empirical study, not a solved problem. | Universal compliance across straight motorways and urban halts. |
| **Lack of Algorithmic Novelty in TCN/ESKF** | Literature audit (Wang 2025, AVNet 2025, AI-IMU 2020) | TCNs and ESKFs are established; novelty lies in empirical failure discovery and fusion characterization. | **HIGH** | Manuscript disclaims architectural novelty; positions contribution as an empirical benchmark & failure analysis. | Proposing a theoretically novel invariant observer. |
| **Highway Cruise Negative Speed Bias** | Observability audit (`ROC-AUC = 0.625`) in `highway_observability_summary.json` | Constant speed produces zero specific force; speed is instantaneous-state weakly observable. | **MEDIUM** | Formulated using Newtonian mechanics; explains why memoryless neural models regress to mean. | External scale constraint (e.g. visual odometry or transmission wheel speed). |
| **Lateral NHC Turn Instability** | 8-way ablation in `phase4_4_fusion_report.md` | Tire sideslip and phone misalignment violate $v_y=0$, creating spurious yaw corrections; decoupled damping ($K_y[6:15]=0$) substantially improves stability. | **LOW** | Fully demonstrated across 83 outage windows; validated with innovation statistics ($NIS_x$). | Dynamic tire slip estimation. |
| **Map Matching Divergence** | Closed-loop ablation in `phase4_6_map_matching_report.md` | Minor heading errors latch to wrong parallel links, doubling drift from 578m to 1304m. | **LOW** | Demonstrated across 83 windows; paper cautions against closed-loop heading injection during outages. | Multi-hypothesis particle filter map tracker. |
| **Vibration Speed Estimation Failure** | Spectral audit over 450,000 windows in `vibration_speed_audit.csv` | Correlation is $r = -0.032$; dominant ~2.4 Hz peak is consistent with chassis dynamics, showing no speed correlation. | **LOW** | Large-scale empirical evaluation across 64 trips; framed carefully under smartphone sampling conditions. | High-frequency acoustic sensors (>1 kHz). |
