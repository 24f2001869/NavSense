Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift

Short running title: Failure-Aware Smartphone Dead Reckoning

Rahul Kumar
Integrated M.Tech. (Materials Engineering), School of Engineering Sciences & Technology, University of Hyderabad, Hyderabad, India
Corresponding author email: 24etim23@uohyd.ac.in

Abstract
During Global Navigation Satellite System (GNSS) outages, smartphone inertial dead reckoning suffers from rapid nonlinear open-loop position drift. We present an empirical study evaluating learned forward velocity estimation, sensor fusion, kinematic constraints, map feedback, and distribution shifts using the public IO-VNBD benchmark dataset (64 passenger-car trips, 16.27 hours). On 19 trip-disjoint test routes, scaling a causal temporal convolutional network from 6 to 39 training trips reduces velocity error by 55.27% (6.17 to 2.76 m/s) and 60-second drift by 64.72% (263.6 to 93.0 m). Unconditional lateral kinematic constraints degrade 60-second drift by 45.47%, and closed-loop map heading feedback degrades drift by 125.5%. Across 13 usable 60-second blackout trajectories, adaptive fusion achieves 96.65 m mean absolute drift and 16.76% macro-average normalized drift across 12 dynamic routes (excluding stationary control Vw15), satisfying a sub-10% drift benchmark on 3 of 13 routes (23.08%). Physical road validation and multi-driver generalisation remain open challenges.

1. Introduction

Continuous vehicular positioning represents an essential requirement for intelligent transportation systems, connected vehicles, and automated navigation services. In open sky conditions, Global Navigation Satellite Systems (GNSS) deliver consistent horizontal positioning with errors typically below five metres. However, satellite reception is regularly degraded or lost in urban street canyons, road tunnels, covered multi-level interchanges, and subterranean parking facilities. In these GNSS-denied environments, vehicles must rely on autonomous dead reckoning to propagate position estimates from the last known satellite fix (Groves, 2013; Titterton and Weston, 2004).

Automotive dead reckoning traditionally employs factory-installed wheel speed encoders connected via Controller Area Network (CAN) diagnostic buses, combined with tactical-grade inertial measurement units (IMUs) rigidly mounted to the vehicle chassis (Dissanayake et al., 2001; Sukkarieh et al., 1999). Conversely, consumer smartphones serve as the primary navigation terminal for millions of motorists worldwide. Consumer smartphones incorporate low-cost micro-electromechanical system (MEMS) accelerometers and gyroscopes. These sensors exhibit severe run-to-run bias instability, thermal sensitivity, scale-factor non-linearities, and high acoustic noise floors.

Unconstrained open-loop integration of raw specific forces and angular velocities leads to rapid nonlinear open-loop position drift. For a constant residual accelerometer bias $b_a$ along the travel axis, open-loop double integration produces quadratic position error, $\Delta p(t) = \frac{1}{2} b_a t^2$. Simultaneously, uncompensated gyroscope bias $b_g$ induces linear orientation error $\delta \theta(t) \approx b_g t$, which misprojects the gravitational acceleration vector $\mathbf{g}$ into the horizontal plane and produces a cubic position error $\frac{1}{6} g b_g t^3$. The combined horizontal position divergence follows:

$$\Delta p(t) = \frac{1}{2} b_a t^2 + \frac{1}{6} g b_g t^3$$

where $b_a$ denotes residual accelerometer bias, $b_g$ denotes gyroscope bias, and $g \approx 9.81\text{ m/s}^2$ is gravitational acceleration. With a residual acceleration bias of $0.05\text{ m/s}^2$, unconstrained integration produces approximately 2.5 metres of quadratic position error after 10 seconds, 22.5 metres after 30 seconds, and 90 metres after 60 seconds; when coupled with gyro-induced gravity misprojection, position drift rapidly exceeds 300 metres within 60 seconds. This mathematical divergence renders raw strapdown inertial integration incapable of sustaining vehicular lane identification during sustained satellite blackouts.

To mitigate this divergence without requiring vehicle wiring or external hardware, two principal methodologies have been investigated:
First, classical kinematic filtering incorporates non-holonomic constraints (NHC) under the assumption that a wheeled vehicle cannot slide laterally or lift off vertically during normal road driving (Dissanayake et al., 2001; Klein et al., 2008; Shen et al., 2019), zero-velocity updates (ZUPT) during stationary halts (Skog et al., 2010), and digital road map matching (Quddus et al., 2007; White et al., 2000). Second, data-driven sequence learning applies temporal convolutional networks (TCN), convolutional backbones, and recurrent networks to estimate forward vehicle speed directly from sliding windows of inertial measurements (Bai et al., 2018; Brossard et al., 2020; Qian et al., 2025; Shin et al., 2025; Xiao et al., 2025; Yan et al., 2020).

Despite encouraging published metrics, practical questions persist regarding the limits of smartphone-only dead reckoning. Existing studies frequently evaluate performance over brief artificial outages (5 to 15 seconds), test on windows sampled from the same driving trips used for training (inducing temporal feature leakage), or assume that standard kinematic constraints and map feedback operate stably on consumer devices without examining failure modes.

In this study, we present an evidence-based empirical study examining the behaviour, scaling properties, failure modes, and deployment realities of smartphone-based dead reckoning under extended GNSS blackouts up to 60 seconds. Operating on the public IO-VNBD benchmark dataset (Onyekpe et al., 2021) and on-device Android execution, the contributions of this study are:

1. Multi-trip training scaling: We quantify the empirical effect of scaling training data from 6 to 39 whole trips across 19 strictly held-out test routes (240.6 km, 115,420 synchronized epochs, 113,539 prediction windows), demonstrating a 55.27% reduction in velocity error and a 64.72% reduction in 60-second position drift.

2. Kinematic constraint failure analysis: We evaluate standard unconditional lateral NHC updates on smartphone telemetry, demonstrating that unmodelled mounting misalignment and vehicle sideslip degrade 60-second drift by 45.47%. We evaluate a decoupled lateral velocity damping formulation that isolates linear velocity damping from attitude states, recovering performance by 54.62%.

3. Map-feedback failure characterisation: In the tested implementation, closed-loop road network heading feedback degrades 60-second drift by 125.47% (578.3 to 1303.9 m) due to candidate link ambiguity and heading latching during turns.

4. Operational boundary assessment: We evaluate the system under high-speed highway cruising and spectral vibration features across 64 trips (>450,000 windows), establishing that spectral vibration features at 10 Hz exhibit near-zero correlation with vehicle speed ($r = -0.032, \rho = -0.028$), consistent with low-frequency vehicle-body/chassis dynamics, while constant-speed cruising exhibits weak statistical separability (ROC-AUC = 0.625) and negative regression bias (-3.5 m/s). We further document an out-of-distribution velocity spike (71.66 m/s) induced by pedestrian arm swing, mitigated to 27.71 m/s via channel ablation.

5. Android mobile edge deployment: We evaluate on-device execution on a consumer smartphone, measuring inference latency (4.2–7.8 ms), Kalman filter execution (0.8–1.4 ms), cross-platform numerical parity, and operating system sensor dispatch (mean interval $\Delta t = 115.8\text{ ms}$, effective callback rate $8.63\text{ Hz}$ versus 10.0 Hz nominal).

6. Benchmark reality check: We evaluate the combined navigation engine against the Smart India Hackathon (SIH) Problem Statement 26168 criterion (<10% distance-normalized drift over 60 seconds), demonstrating that while the macro-average normalized drift across dynamic routes is reduced to 16.76% (96.65 m mean absolute drift across all 13 eligible trajectories), the sub-10% criterion is satisfied on 3 of 13 eligible test routes (23.08%).

2. Dataset and Methods

2.1. Dataset and reference signals

All vehicular evaluations are conducted on the public Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD) published by Onyekpe et al. (2021). The dataset was collected across real-world road networks in the United Kingdom using a test passenger vehicle (Ford Fiesta) instrumented with consumer smartphones and professional reference instrumentation.

The complete IO-VNBD repository contains 1,182,661 raw sensor epochs across multiple transportation modes (motorcycle, bicycle, pedestrian, and passenger car). We isolate the complete passenger-car corpus (Driver E, Ford Fiesta), comprising 64 driving trips and 585,643 synchronized 10.0 Hz epochs (16.27 hours of synchronized driving, 902.4 km). For causal sequence learning, the synchronized time epochs are segmented into causal sliding windows of 100 samples (10.0 s). Excluding the initial 99 samples per trip (forming the 100-sample initialization buffer), the corpus yields exactly 579,307 usable sequence samples across the 64 trips ($585,643 - 64 \times 99 = 579,307$), partitioned across training, validation, and held-out evaluation splits. The trips encompass four road environments: suburban town routes (Vta, 30 trips), suburban mixed routes (Vtb, 12 trips), winding mountain routes (Vw, 20 trips), and high-speed motorways (V-Vfa, 2 trips). Table 1 provides the dataset census.

Table 1. IO-VNBD passenger-car dataset census across route categories.
Road Category | Trips (N) | Duration (h) | Total Epochs | Distance (km) | Mean Speed (km/h) | Notes & Operational Profile
Suburban Towns (Vta) | 30 | 3.57 | 128,618 | 147.9 | 41.4 | Frequent intersections, traffic signals, stop-and-go
Suburban Mixed (Vtb) | 12 | 3.18 | 114,437 | 163.6 | 51.5 | Multi-lane arterials, urban street canyons, variable speeds
Mountain Routes (Vw) | 20 | 7.32 | 263,579 | 409.0 | 55.9 | Continuous curvature, elevation changes, unbanked turns
High-Speed Motorways (V-Vfa) | 2 | 2.19 | 79,009 | 181.9 | 82.7 | Sustained high-speed cruising (80–118 km/h), minimal turns
Complete Passenger-Car Corpus | 64 | 16.27 | 585,643 | 902.4 | 55.5 | Synchronized 10 Hz telemetry; Driver E; Ford Fiesta test vehicle

Reference instrumentation consists of a Racelogic VBOX Video HD2 system equipped with a 10 Hz rooftop GNSS antenna and a 10 Hz vehicle CAN bus diagnostic stream extracting wheel speed from the electronic control unit (ECU). CAN forward velocity serves as the engineering reference signal for model training and error evaluation. Because IO-VNBD does not provide dual-frequency carrier-phase Real-Time Kinematic (RTK) GNSS positioning, CAN bus velocity represents an engineering reference subject to normal tyre radius variation and slip, rather than absolute millimetre-level ground truth.

2.2. Trip-disjoint experimental split

To prevent data contamination, all model training, validation, and held-out evaluation procedures enforce strict whole-trip disjointness. Under this protocol, each individual driving trip is allocated in its entirety to exactly one partition; no sliding window, feature frame, or temporal sequence from an evaluation trip ever enters the training or validation sets.

Table 2. Whole-trip-disjoint dataset partitioning.
Data Split | Trip Count (N) | Synchronized Epochs | Duration (h) | Distance (km) | Driver Identity | Partitioning Protocol
Training Split | 39 | 448,993 | 12.47 | 634.5 | Driver E | Whole-trip disjoint; zero temporal or route overlap
Validation Split | 6 | 21,230 | 0.59 | 27.2 | Driver E | Model selection, checkpoint early stopping, loss tuning
Held-Out Test Split | 19 | 115,420 | 3.21 | 240.6 | Driver E | Strictly unseen test routes; multi-horizon evaluation
Total Synchronized Corpus | 64 | 585,643 | 16.27 | 902.4 | Driver E | Usable sliding sequence samples: 579,307 (100-sample buffer)

As detailed in Table 2, usable model sequence samples (579,307 samples across 585,643 synchronized epochs) reflect sliding temporal windows extracted across all 64 passenger-car trips. Metadata inspection reveals that all 64 passenger-car trips in IO-VNBD were driven by a single individual (Driver E). Consequently, all evaluations in this study reflect whole-trip generalisation for a single driver; cross-driver generalisation cannot be claimed from this dataset.

2.3. Coordinate frames and inertial processing

The navigation mechanization defines three primary orthogonal coordinate systems:

1. Device Body Frame ($b$): Defined by smartphone IMU axes ($x_b$ right, $y_b$ forward, $z_b$ upward relative to the screen).

2. Leveled Navigation Frame ($l$): An intermediate frame where vertical axis $z_l$ is aligned opposite to local gravity $\mathbf{g}$, and the $x_l\text{--}y_l$ plane forms the horizontal plane.

3. Local East-North-Up Frame ($n$): The geographic tangent navigation frame with axes directed East, North, and Up.

Smartphone attitude is parameterized by unit quaternion $\mathbf{q}_b^n$, relating vectors in the body frame to the navigation frame:

$$\mathbf{v}^n = \mathbf{q}_b^n \otimes \mathbf{v}^b \otimes (\mathbf{q}_b^n)^*$$

Gravity leveling is performed by estimating the static gravity vector during pre-outage stationary epochs:

$$\mathbf{g}^b = \frac{1}{N} \sum_{k=1}^N \mathbf{f}_k^b, \quad \hat{\mathbf{z}}_l = -\frac{\mathbf{g}^b}{\|\mathbf{g}^b\|}$$

Horizontal vehicle forward axis alignment is resolved through dynamic horizontal acceleration Principal Component Analysis (PCA) during pre-outage driving. Specific forces and angular rates are filtered using a second-order causal low-pass Butterworth filter with a cut-off frequency of 4.5 Hz to suppress high-frequency structural vibration before 10.0 Hz decimation.

2.4. Error-state Kalman filter

The core state estimation engine is a 15-state continuous-discrete Error-State Kalman Filter (ESKF) formulated according to Solà (2017). The true kinematic state $\mathbf{x} \in \mathbb{R}^{16}$ comprises position $\mathbf{p}^n \in \mathbb{R}^3$, velocity $\mathbf{v}^n \in \mathbb{R}^3$, unit orientation quaternion $\mathbf{q}_b^n \in \mathbb{H}$, accelerometer bias $\mathbf{b}_a \in \mathbb{R}^3$, and gyroscope bias $\mathbf{b}_g \in \mathbb{R}^3$. The filter tracks small error states $\delta \mathbf{x} \in \mathbb{R}^{15}$:

$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}^n & \delta \mathbf{v}^n & \delta \boldsymbol{\theta} & \delta \mathbf{b}_a & \delta \mathbf{b}_g \end{bmatrix}^T$$

where $\delta \boldsymbol{\theta} \in \mathbb{R}^3$ denotes the attitude error vector defined via quaternion perturbation:

$$\mathbf{q} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \boldsymbol{\theta} \end{bmatrix}$$

Nominal state kinematics propagate at each IMU epoch $\Delta t$:

$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2} (\mathbf{R}(\hat{\mathbf{q}}_k) \mathbf{f}_k^b + \mathbf{g}^n) \Delta t^2$$

$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\mathbf{R}(\hat{\mathbf{q}}_k) \mathbf{f}_k^b + \mathbf{g}^n) \Delta t, \quad \hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}(\boldsymbol{\omega}_k^b \Delta t)$$

Error state covariance propagates as $\mathbf{P}_{k+1|k} = \mathbf{F}_k \mathbf{P}_{k|k} \mathbf{F}_k^T + \mathbf{Q}_k$, where $\mathbf{F}_k \approx \mathbf{I}_{15} + \mathbf{F}_c \Delta t$, with continuous transition matrix $\mathbf{F}_c$:

$$\mathbf{F}_c = \begin{bmatrix} \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & -[\mathbf{R}\hat{\mathbf{a}}^b]_\times & -\mathbf{R} & \mathbf{0}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & -[\hat{\boldsymbol{\omega}}^b]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\ \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \end{bmatrix}$$

When a linear velocity measurement $\mathbf{z}_v$ becomes available, the innovation is computed as $\tilde{\mathbf{y}}_v = \mathbf{z}_v - \hat{\mathbf{v}}^n$, with measurement matrix $\mathbf{H}_v = \begin{bmatrix} \mathbf{0}_{3 \times 3} & \mathbf{I}_3 & \mathbf{0}_{3 \times 9} \end{bmatrix}$.

Figure 1 illustrates the overall NavSense pipeline, showing the relationship between inertial preprocessing, learned velocity estimation, kinematic integration, adaptive fusion, ESKF propagation, and downstream map matching.

[Figure 1 about here: publication/journal_of_navigation_submission/figures/Figure_01.png]

2.5. Learned velocity estimation

To provide an independent velocity pseudo-measurement during satellite blackouts, we utilize a causal Temporal Convolutional Network with kinematic features (TCN-kin). The network processes a rolling temporal history of 100 samples (10.0 s buffer), but enforces causal receptive field constraints such that the effective mathematical receptive field span is 61 samples (corresponding to a 6.0-s causal look-back interval at 10 Hz, $[t-60, t]$), preventing dependence on future epochs.

The input tensor $\mathbf{X}_k \in \mathbb{R}^{100 \times 9}$ comprises nine causal kinematic channels:
1–3. Tri-axial specific force in the leveled frame: $a_{\text{lin},x}, a_{\text{lin},y}, a_{\text{lin},z}$
4–6. Tri-axial angular rates in the device body frame: $\omega_z, \omega_y, \omega_x$

7. Planar horizontal acceleration magnitude: $a_h = \sqrt{a_{\text{lin},x}^2 + a_{\text{lin},y}^2}$

8. Vertical acceleration magnitude: $a_v = |a_{\text{lin},z}|$
9. Turning-acceleration coupling product: $\kappa = \omega_{\text{mag}} \cdot a_h$, where $\omega_{\text{mag}} = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$, capturing instantaneous centripetal dynamics directly from IMU channels without velocity feedback or target leakage.

TCN-kin consists of four residual blocks with dilated 1D causal convolutions (dilations $d \in \{1, 2, 4, 8\}$, kernel size $k = 3$, channel dimensions 32 to 64), batch normalization, ReLU activation, and a temporal dropout rate of 0.15. The final temporal feature projects through a linear regression head producing scalar forward speed $\hat{v}_{\text{TCN}} \in \mathbb{R}^+$. The total parameter count is exactly 60,225 weights (~60k parameters). The network is trained using AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$) minimizing Huber loss with parameter $\delta = 1.0\text{ m/s}$.

2.6. Constraint and fusion mechanisms

The navigation engine incorporates three distinct velocity estimation branches:

1. Neural velocity branch: The causal TCN-kin speed estimate $\hat{v}_{\text{TCN}}$.

2. Stateful kinematic branch: Pure kinematic forward acceleration integration initialized at blackout onset $t_0$:

$$v_{\text{kin}}(t) = v(t_0) + \int_{t_0}^t a_{\text{lon}}(\tau) d\tau$$

with velocity damping applied during steady-state cruise: $\dot{v}_{\text{kin}} = a_{\text{lon}} - \gamma v_{\text{kin}}$, where $\gamma = 0.015\text{ s}^{-1}$.

3. Causal zero-velocity detector (ZVD): Standstill conditions are evaluated over a 5-epoch window (0.5 s) using acceleration variance and angular velocity norm thresholds:

$$\sigma_a^2 = \frac{1}{W} \sum_{i=0}^{W-1} \|\mathbf{f}_{k-i}^b - \bar{\mathbf{f}}\|^2 < \gamma_a, \quad \|\boldsymbol{\omega}_k^b\| < \gamma_\omega$$

When triggered, zero velocity updates (ZUPT, $\mathbf{z} = \mathbf{0}$) and zero angular rate updates (ZARU) clamp filter velocity and estimate IMU bias.

Adaptive regime blending modulates forward velocity $\hat{v}_{\text{blend}}$ between neural and kinematic branches based on real-time dynamics:

$$\hat{v}_{\text{blend}} = \alpha_k \hat{v}_{\text{TCN}} + (1 - \alpha_k) v_{\text{kin}}$$

where weight $\alpha_k \in [0.1, 0.9]$ is dynamically scheduled: during dynamic manoeuvres ($|a_{\text{lon}}| > 0.5\text{ m/s}^2$ or $|\omega_z| > 0.15\text{ rad/s}$), $\alpha_k \to 0.85$ favoring neural speed estimation; during unaccelerated highway cruise ($|a_{\text{lon}}| < 0.1\text{ m/s}^2$ at high speeds), $\alpha_k \to 0.20$ prioritizing kinematic momentum.

Lateral velocity handling: Standard non-holonomic constraints (NHC) assume zero lateral velocity in the vehicle body frame: $v_{\text{lat}}^b = 0$. In our decoupled formulation, to prevent unmodelled mounting misalignment and vehicle sideslip angles from corrupting attitude states, the Kalman gain for lateral velocity innovations (along body axis $y$) is restricted strictly to velocity states:

$$\mathbf{K}_y = \begin{bmatrix} \mathbf{0}_{3 \times 1} \\ K_v \\ \mathbf{0}_{9 \times 1} \end{bmatrix}$$

setting the attitude, position, and bias cross-gain blocks to zero ($K_y[6:15] = 0$).

2.7. Map matching

To evaluate road network aiding, an OpenStreetMap (OSM) vector engine extracts roadway centerlines within a 250-metre bounding radius. Vehicle position is projected onto candidate polyline links using a multiple-hypothesis tracker (MHT) evaluating link proximity, heading agreement, and connectivity topology. In the closed-loop evaluation mode, candidate road link azimuth $\psi_{\text{map}}$ is fed into the filter as a heading observation: $\tilde{y}_\psi = \psi_{\text{map}} - \hat{\psi}_{\text{filter}}$.

2.8. Android implementation

To assess operational edge viability, the navigation pipeline was ported to a native Android application (Kotlin/Java) executing on an off-the-shelf consumer smartphone (Google Pixel 7a, Android 14). Deep learning inference is executed locally using ONNX Runtime Mobile (model size 162 KB) with a single-thread CPU execution provider. The 15-state ESKF is implemented using the Efficient Java Matrix Library (EJML). Real-time telemetry callbacks from the Android SensorManager run asynchronously, tracking hardware nanosecond timestamps to handle operating system scheduling jitter.

3. Experimental Protocol

3.1. GNSS-outage evaluation

The system is evaluated by simulating GNSS outages across four standardized durations: 10, 20, 30, and 60 seconds. Each blackout is injected into held-out test routes after a 10.0-second pre-outage GNSS calibration phase during which satellite positioning is available to level the filter, align heading, and estimate initial sensor biases. Once the blackout begins, all GNSS updates terminate, forcing the filter to operate strictly on smartphone IMU inputs, virtual velocity updates, and internal kinematic constraints.

Trips with durations shorter than 70.0 seconds are excluded from the 60-second evaluation set because they cannot accommodate the 10-second calibration plus the 60-second blackout duration. Across the 19 held-out test trips, 18 trips are usable at 10 s, 17 trips at 20 s, 16 trips at 30 s, and 13 trips at 60 s.

3.2. Baselines

The system is evaluated against three baseline architectures:

1. Pure strapdown kinematics: Open-loop double integration of linear acceleration and gyroscope rates without external velocity aiding or ZVD clamping.

2. Static TCN-kin: Causal TCN velocity estimation with static ESKF integration without dynamic momentum blending.

3. Fixed damped momentum: Forward kinematic acceleration integration with fixed exponential velocity damping ($\gamma = 0.015\text{ s}^{-1}$).

3.3. Metrics

Dead reckoning performance is quantified using four primary metrics:

1. Absolute horizontal position drift ($E_p$): Euclidean distance between estimated horizontal position $\hat{\mathbf{p}}_{2\text{D}}^n$ and reference position $\mathbf{p}_{\text{ref},2\text{D}}^n$ at outage termination $T$: $E_p(T) = \|\hat{\mathbf{p}}_{2\text{D}}^n(T) - \mathbf{p}_{\text{ref},2\text{D}}^n(T)\|_2$.

2. Macro-average normalized drift ($\bar{D}_{\%}$): Position drift expressed as a percentage of total reference distance traveled $S(T)$ during the blackout, averaged with equal weight across dynamic driving routes:

$$\bar{D}_{\%} = \frac{1}{M_{\text{dyn}}} \sum_{m=1}^{M_{\text{dyn}}} \left( \frac{E_{p,m}(T)}{S_m(T)} \times 100\% \right)$$

where $M_{\text{dyn}} = 12$ denotes the set of dynamic driving routes, with the stationary control route (Vw15) excluded from normalized percentage calculation to avoid distortion from the near-zero travel-distance denominator ($S < 2\text{ m}$). Normalised drift percentages are calculated using the reference distance travelled during the corresponding 60-s blackout window, $S_m(T)$, which can differ from total trip distance. Absolute drift $E_p$ is reported across all 13 eligible trajectories.

3. Velocity estimation accuracy: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and signed bias ($\bar{e}_v$) relative to CAN bus reference velocity over all held-out prediction epochs.

4. Benchmark pass rate: The proportion of usable test trips achieving $D_{\%} < 10.0\%$ at 60 seconds, defined by Smart India Hackathon (SIH) Problem Statement 26168.

3.4. Ablation and failure analysis

To identify underlying mechanisms, controlled ablations examine:
- Unconditional lateral NHC ($v_y^b \approx 0$) versus decoupled lateral velocity damping.
- Open-loop map visualization versus closed-loop road heading injection.
- Zero-velocity detector gating thresholds.

3.5. Distribution-shift tests

Three operational stress tests evaluate boundary conditions:

1. Highway cruise observability and feature separability: Evaluating learned velocity behaviour on motorway route V-Vfa02 (163 km). The investigation addresses two specific questions: First, signed velocity prediction residuals ($\hat{v} - v^*$) are evaluated during steady-state highway cruising in the high-speed regime ($v \ge 25\text{ m/s}$, $\ge 90\text{ km/h}$, 90–120 km/h). Second, to quantify within-route statistical feature separability under low acceleration excitation, matched cruising observations at 80 km/h and 110 km/h are classified using a logistic regression model over the nine causal kinematic channels with five-fold stratified cross-validation, quantifying within-route statistical separability rather than cross-route predictive generalisation.

2. Spectral vibration speed audit: Causal Welch periodogram and Fast Fourier Transform (FFT) analysis across 64 trips (>450,000 1-s windows) correlating vibration frequency against forward speed.

3. Pedestrian gait out-of-distribution test: Evaluating the vehicle-trained TCN on handheld pedestrian walking data recorded across outdoor walkways.

3.6. Deployment replay

On-device Android feasibility is assessed via continuous golden telemetry replay, feeding synchronized sensor streams into the mobile application to measure computational latency and quantify cross-platform floating-point parity against the Python reference engine.

3.7. Automated reproducibility and numerical verification

To ensure reproducible reporting, automated audit scripts executed within the Google Antigravity environment verified mathematical causality, confirmed zero future-gradient leakage, cross-checked all reported tabular metrics against primary raw result logs, and audited reference DOIs against the CrossRef registry before manuscript finalisation.

4. Results

4.1. Baseline inertial divergence

Table 3 summarizes trajectory drift and failure metrics of classical strapdown mechanization alongside intermediate baselines across the four evaluation horizons.

Table 3. Baseline inertial navigation performance across evaluation horizons.
Outage Horizon | Navigation Architecture | Evaluated Trips | Mean Drift (m) | Normalized Drift (%) | Passing Trips (<10%) | Pass Rate (%)
10 s | Pure Kinematics | 18 | 23.66 m | 83.54% | 6 / 18 | 33.3%
10 s | Fixed Damped Momentum | 18 | 23.49 m | 111.37% | 6 / 18 | 33.3%
10 s | Adaptive Regime Fusion | 18 | 25.06 m | 111.53% | 5 / 18 | 27.8%
10 s | Static Causal TCN | 18 | 26.83 m | 106.44% | 5 / 18 | 27.8%
20 s | Static Causal TCN | 17 | 45.73 m | 123.95% | 5 / 17 | 29.4%
20 s | Adaptive Regime Fusion | 17 | 49.27 m | 135.51% | 4 / 17 | 23.5%
20 s | Pure Kinematics | 17 | 57.01 m | 147.98% | 4 / 17 | 23.5%
20 s | Fixed Damped Momentum | 17 | 65.81 m | 193.51% | 4 / 17 | 23.5%
30 s | Static Causal TCN | 16 | 54.62 m | 65.57% | 5 / 16 | 31.2%
30 s | Adaptive Regime Fusion | 16 | 61.83 m | 71.38% | 4 / 16 | 25.0%
30 s | Fixed Damped Momentum | 16 | 98.07 m | 127.13% | 2 / 16 | 12.5%
30 s | Pure Kinematics | 16 | 109.57 m | 118.22% | 2 / 16 | 12.5%
60 s | Adaptive Regime Fusion | 13 | 96.65 m | 16.76% | 3 / 13 | 23.1%
60 s | Static Causal TCN | 13 | 95.53 m | 18.26% | 3 / 13 | 23.1%
60 s | Fixed Damped Momentum | 13 | 173.71 m | 40.75% | 1 / 13 | 7.7%
60 s | Pure Kinematics | 13 | 324.21 m | 89.66% | 0 / 13 | 0.0%

Under pure strapdown mechanization, position drift exhibits rapid nonlinear growth, expanding from 23.66 metres at 10 seconds to 324.21 metres at 60 seconds. At 60 seconds, zero of the 13 usable trips achieve <10% normalized drift (Table 3), with unconstrained accelerometer bias and gyro tilt error generating large lane-scale position error (exceeding lane-scale tolerances within 20–30 seconds).

Figure 2 illustrates the divergence of open-loop strapdown mechanization compared to the vehicle reference trajectory during an extended satellite blackout.

[Figure 2 about here: publication/journal_of_navigation_submission/figures/Figure_02.png]

4.2. Training-trip scaling

To examine the empirical effect of training set diversity, the TCN-kin model was trained across two distinct whole-trip partitions: an initial 6-trip baseline (representing approximately 1.1 hours of driving) and the full 39-trip training corpus (12.5 hours, 448,993 epochs). Both models were evaluated on the identical 19 held-out test routes (115,420 synchronized epochs, 113,539 prediction windows, 240.6 km).

Table 4. Empirical scaling of TCN-kin forward speed estimation across 19 held-out test trips.
Evaluation Metric | 6-Trip TCN Model | 39-Trip TCN Model | Relative Performance Delta (%)
Training Scope | 6 trips (1.1 h, 40,890 epochs) | 39 trips (12.5 h, 448,993 epochs) | +550% training route exposure
Forward Velocity MAE (m/s) | 6.17 m/s | 2.76 m/s | -55.27% error reduction
Forward Velocity RMSE (m/s) | 6.96 m/s | 3.59 m/s | -48.42% error reduction
Velocity Prediction Bias (m/s) | -3.06 m/s | -0.41 m/s | -86.60% bias reduction
30-Second Rolling Position Drift | 153.31 m | 52.64 m | -65.67% drift reduction
60-Second Rolling Position Drift | 263.55 m | 92.96 m | -64.72% drift reduction

As shown in Table 4, scaling training data from 6 to 39 whole trips reduces out-of-sample velocity MAE from 6.17 m/s to 2.76 m/s (55.27% improvement) and eliminates substantial negative bias (-3.06 m/s to -0.41 m/s). When integrated into the dead reckoning filter, the 39-trip model reduces 60-second position drift from 263.55 metres to 92.96 metres, representing a 64.72% error reduction.

Figure 3 illustrates comparative scaling performance and residual error distributions between the 6-trip and 39-trip models.

[Figure 3 about here: publication/journal_of_navigation_submission/figures/Figure_03.png]

4.3. Learned velocity and adaptive fusion

At the 60-second horizon, the NavSense adaptive fusion engine achieves an aggregate mean horizontal position drift of 96.65 metres across all 13 usable trips, and a macro-average normalized drift of 16.76% across the 12 dynamic driving trips per Equation (13). While the static TCN achieves a slightly lower raw absolute drift (95.53 m versus 96.65 m), the adaptive blending of kinematic momentum provides superior stability across urban stop-and-go trips, reducing normalized macro-average drift from 18.26% to 16.76%.

4.4. Constraint failure and decoupled recovery

To examine the interaction between kinematic vehicle constraints and consumer smartphone IMUs, a controlled ablation evaluating six representative configurations from an eight-configuration study was conducted on the filter architecture across 83 evaluation windows.

Table 5. Controlled ablation of kinematic constraints and fusion configurations at 60-second blackout horizon.
Configuration Variant | Forward Speed | Lateral Constraint | Attitude Feedback | Mean 60-s Drift (m) | Relative Drift to Variant E (%) | Mean NIS_x | Filter State
Variant A (Pure-INS Baseline) | None | None | None | 2488.7 m | 295.1% | 0.00 | Divergent
Variant D (Coupled 3D NHC alone) | None | Coupled (v_y, v_z) | Full | 1166.7 m | 138.3% | 0.00 | Marginal
Variant E (Learned TCN Alone) | TCN v_x | None | None | 843.4 m | 100.0% (Ref) | 6.55 | Stable
Variant F (TCN + Unconditional Lateral NHC) | TCN v_x | Coupled (v_y = 0) | Full | 1226.9 m | 145.5% | 82.12 | Degraded
Variant H (TCN + Full Coupled 3D NHC) | TCN v_x | Coupled (v_y, v_z = 0) | Full | 1240.8 m | 147.1% | 87.85 | Degraded
Variant V1 (Proposed Decoupled Damping) | TCN v_x | Decoupled Damping | None (K_y[6:15]=0) | 382.7 m | 45.4% | 6.60 | Stable

Protocol Note: The controlled ablation results in Table 5 evaluate six representative filter configurations across an 83-window multi-trip configuration protocol, whereas the final held-out benchmark in Table 7 evaluates continuous 60-second blackout trajectories across the 13 eligible held-out test routes. Relative drift is expressed as a percentage of the learned TCN reference baseline (Variant E = 100.0%); values above 100% indicate performance degradation relative to Variant E, while values below 100% indicate lower drift (improved accuracy).

Applying standard unconditional lateral Non-Holonomic Constraints (Variant F) degraded 60-second position drift from 843.40 metres to 1226.90 metres, representing a 45.47% performance penalty in the tested configuration, accompanied by an innovation surge in lateral normalized innovation squared ($NIS_x$) from an empirical baseline reference level of 6.55 to 82.12. The observed failure is consistent with non-zero tyre sideslip during cornering together with unmodelled smartphone mounting misalignment. The standard Kalman update interprets lateral velocity innovations as attitude errors, corrupting the filter heading state.

In contrast, the decoupled lateral damping formulation (Variant V1) isolates lateral velocity innovations strictly to velocity damping ($K_y[6:15] = 0$), preventing false attitude adjustments and maintaining bounded innovation ($NIS_x = 6.60$, near the baseline reference level of 6.55). This architectural decoupling reduces 60-second drift to 382.74 metres, a 54.62% improvement over the baseline unconstrained filter. Combining decoupled damping with the 39-trip scaled TCN in the full system further reduces 60-second drift to 95.53 metres.

Figure 4 illustrates trajectory instability induced by coupled lateral constraints and the stabilization achieved through decoupled velocity damping.

[Figure 4 about here: publication/journal_of_navigation_submission/figures/Figure_04.png]

4.5. Map-feedback failure

Table 6 documents the evaluation of digital road map matching operating in open-loop visualization mode versus closed-loop Kalman filter feedback.

Table 6. Performance of open-loop versus closed-loop map matching under 60-second GNSS blackouts.
Variant ID | Map Matching Mode | Heading Feedback | Envelope Gate | Mean 60-s Drift (m) | Heading Error (deg) | Delta vs. M0 (%) | Regression Rate (%)
M0 (Baseline) | Open-Loop / None | Disabled | No | 578.3 m | 63.4 deg | +0.0% | 0.0%
M1 (Shadow Map) | Shadow 5-Gate MHT | Disabled | No | 578.3 m | 63.4 deg | +0.0% | 0.0%
M2 (Closed-Loop Heading) | Heading Constraint | Enabled (Direct) | No | 1303.9 m | 73.5 deg | -125.5% | 47.0%
M3 (Lateral Map) | Lateral Constraint | Disabled | No | 597.7 m | 65.5 deg | -3.3% | 28.9%
M4 (Joint Map) | Joint Heading & Lateral | Enabled | No | 829.8 m | 68.2 deg | -43.5% | 33.7%
M5 (Envelope-Gated) | Joint Heading & Lateral | Enabled | Yes | 807.9 m | 67.5 deg | -39.7% | 22.9%

Protocol Note: Delta vs. M0 is defined as $(M_0 - M_i) / M_0$; therefore, negative percentage values indicate performance degradation (increased position drift) relative to open-loop baseline M0. A regression was counted when the closed-loop variant produced greater 60-s position error than the open-loop M0 baseline by more than 0.5 m ($E_{p,\text{map}} > E_{p,\text{M0}} + 0.5\text{ m}$) for the corresponding evaluation window. Over the 83 evaluation windows, M2 experienced regressions in 39 windows (47.0%), M3 in 24 windows (28.9%), M4 in 28 windows (33.7%), and M5 in 19 windows (22.9%).

Feeding candidate road link azimuth directly into the Kalman filter heading state increased mean 60-second drift from 578.30 metres to 1303.90 metres, representing a 125.47% error increase. This failure occurred because during turns at complex intersections or curved interchanges, dead reckoning drift caused the map matcher to latch onto parallel frontage roads or incorrect branch links. The filter accepted the incorrect link azimuth as a high-confidence heading observation, locking the dead reckoning trajectory into divergence.

Figure 5 shows the breakdown of closed-loop map matching during intersection manoeuvres.

[Figure 5 about here: publication/journal_of_navigation_submission/figures/Figure_05.png]

4.6. Highway cruise boundary

Evaluating the learned TCN on high-speed motorway route V-Vfa02 revealed a distinct performance boundary under steady-state highway cruising ($v \ge 25\text{ m/s}$, $\ge 90\text{ km/h}$, 90–120 km/h). During unaccelerated cruising, longitudinal specific force and yaw rates remain near zero ($a_{\text{lon}} \approx 0, \omega_z \approx 0$).

Under these conditions, instantaneous inertial feature distributions at 80 km/h and 110 km/h exhibit weak statistical separability, yielding a receiver operating characteristic area under the curve (ROC-AUC) of 0.625. Consequently, the learned temporal estimator regresses toward the training set mean speed, producing an average negative bias of approximately -3.5 m/s ($r = -0.4866$) on high-speed highway segments.

Figure 6 illustrates statistical feature overlap and speed regression residuals during steady-state motorway cruising.

[Figure 6 about here: publication/journal_of_navigation_submission/figures/Figure_06.png]

4.7. Vibration analysis

To test whether vehicle engine and chassis vibrations contain a speed-dependent spectral signature that could aid forward velocity estimation, a spectral audit was performed across all 64 passenger-car trips, analyzing over 450,000 one-second windows.

Across the complete corpus, the correlation between dominant inertial spectral frequency and reference vehicle speed was $r = -0.032$ (Pearson) and $\rho = -0.028$ (Spearman). While a consistent spectral peak was observed between 2.2 and 2.5 Hz, this peak remained invariant across speeds from 0 to 120 km/h. Under the evaluated 10 Hz feature construction, no reliable speed dependence was identified in the dominant vibration frequency, with the ~2.2–2.5 Hz component being consistent with low-frequency vehicle-body/chassis dynamics; the physical mechanism was not directly identified.

Figure 7 shows the acceleration power spectral density across vehicle speed bands and the cross-trip correlation between dominant frequency and vehicle speed.

[Figure 7 about here: publication/journal_of_navigation_submission/figures/Figure_07.png]

4.8. Pedestrian distribution shift

To assess the vulnerability of the vehicle-trained TCN to out-of-distribution dynamic movement, handheld pedestrian walking trials were conducted with a consumer smartphone.

When the vehicle-trained model was evaluated on pedestrian walking data, predicted speed escalated to an anomalous peak of 71.66 m/s (257.98 km/h). Input feature channel ablation revealed that human arm swing produces high-frequency rotational yaw rates reaching 3.81 rad/s (+26.08 standard deviations above the vehicle training corpus maximum of 0.40 rad/s). The learned regression mapping transformed this out-of-distribution input into extreme forward-velocity predictions. When all gyroscope channels were ablated from the model input, the peak prediction dropped to 27.71 m/s (with individual axis ablation yielding 34.9 m/s for yaw, 52.2 m/s for pitch, and 29.9 m/s for roll), indicating that rotational arm-swing dynamics were the primary contributor to the observed out-of-distribution failure.

Figure 8 depicts the pedestrian out-of-distribution velocity response alongside Android on-device execution benchmarks.

[Figure 8 about here: publication/journal_of_navigation_submission/figures/Figure_08.png]

4.9. Android deployment

The native Android deployment was evaluated on a Google Pixel 7a via golden telemetry replay across 10,000 synchronized epochs. On-device ONNX inference required 4.2 to 7.8 ms per 100-sample window, while the 15-state Java ESKF required 0.8 to 1.4 ms per epoch. Total measured computational latency averaged 9.15 ms per epoch (with a 99th-percentile latency of 13.89 ms), remaining well within the nominal 100 ms budget for 10 Hz real-time processing.

Cross-platform parity analysis between the Android Java/EJML implementation and the offline Python 64-bit reference engine demonstrated close numerical agreement over a 300-second continuous replay: maximum position discrepancy was 0.24177 metres, velocity discrepancy was 0.02117 m/s, and heading discrepancy was 0.01829 degrees.

The deployment audit also revealed that although the phone IMU hardware sampled internally at ~400 Hz, Android operating system inter-process communication dispatch yielded an average navigation callback rate of 8.63 Hz with inter-epoch intervals averaging 115.8 ms ($\Delta t \in [104, 127]\text{ ms}$). Integrating dynamic hardware nanosecond timestamps was found to be necessary to prevent time-scale distortion.

4.10. Final benchmark

Table 7 summarizes the final 60-second blackout benchmark results across the 13 usable held-out test routes, grouped by road category. Comprehensive route-by-route metrics for each individual trip are provided in Supplementary Table S1.

Table 7. Final 60-second GNSS blackout evaluation across 13 usable held-out test trips by road category.
Road Category | Trips (N) | Mean Ref Distance (m) | Pure Strapdown Drift (m) | Pure Macro Drift (%) | Adaptive Fusion Drift (m) | Adaptive Macro Drift (%) | SIH Pass Count (<10%)
Suburban Towns (Vta) | 7 | 575.1 m | 301.6 m | 127.4% | 95.1 m | 21.6% | 1 / 7 (14.3%)
Mountain Routes (Vw) | 4 | 1293.9 m | 446.1 m | 39.7% | 108.2 m | 9.7% | 2 / 4 (50.0%)
High-Speed Motorway (V-Vfa) | 1 | 1460.3 m | 313.9 m | 25.4% | 157.2 m | 11.7% | 0 / 1 (0.0%)
Stationary Control (Vw15) | 1 | 1.3 m | 5.6 m | 450.5% | 0.82 m | 63.8% | 0 / 1 (0.0%)
Overall Benchmark | 13 | 820.2 m | 324.21 m | 89.66% | 96.65 m | 16.76% | 3 / 13 (23.08%)

Across the 13 evaluated trips, NavSense achieves an aggregate mean absolute drift of 96.65 metres across all 13 routes and a macro-average normalized drift of 16.76% across the 12 dynamic routes per Equation (13). Three trips achieve normalized drift below the 10% threshold: route Vw12 (3.01%), route Vta21 (7.88%), and route Vw14a (8.54%), resulting in an overall pass rate of 23.08% (3/13). Mean reference distance across all 13 routes is 820.2 metres (575.1 m for Suburban, 1293.9 m for Mountain, 1460.3 m for Motorway, and 1.3 m for Stationary Control), reproducing the route-by-route records in Supplementary Table S1 with exact arithmetic parity.

5. Discussion

5.1. What improves with additional trips

The scaling results demonstrate that expanding the training corpus from 6 to 39 whole trips produces substantial performance gains: velocity MAE decreased by 55.27% and 60-second position drift decreased by 64.72%. These improvements stem from exposure to a broader range of road grade variations, throttle transitions, braking dynamics, and vehicle cornering maneuvers. Single-trip or few-trip training sets overfit to specific route dynamics, leaving models vulnerable to distribution shift on unseen roads.

5.2. Why constraints are configuration-sensitive

The degradation observed with unconditional lateral non-holonomic constraints highlights a critical gap between theoretical robotics assumptions and consumer smartphone realities. In robotics, IMUs are rigidly bolted to the vehicle chassis with known lever arms and boresight angles. On consumer smartphones, mounting angles in phone cradles vary, and vehicles experience non-zero tire sideslip during cornering. When the filter forces lateral velocity to zero through coupled Kalman gain blocks, lateral innovations corrupt attitude error states, turning minor lateral slip into heading divergence. Restricting lateral updates to linear velocity damping eliminates this destabilizing cross-coupling.

5.3. What map feedback can and cannot provide

The empirical failure of closed-loop map heading feedback underscores the hazards of measurement ambiguity in dead reckoning. When a dead reckoning filter experiences heading drift during an outage, map matching cannot reliably differentiate between the true roadway and an adjacent frontage road, exit ramp, or parallel urban street. Feeding candidate road link azimuths directly back into the filter creates a positive feedback loop that accelerates divergence. In the tested implementation, downstream/open-loop map visualisation was safer than direct closed-loop heading injection.

5.4. Distribution-shift implications

The highway cruise and pedestrian walking evaluations identify two distinct operational boundaries. On straight highways, the absence of longitudinal acceleration removes the dynamic cues required for learned speed estimation, resulting in negative regression bias toward the training mean. During handheld pedestrian movement, high-frequency arm-swing yaw rates fall far outside the vehicular training distribution, triggering velocity spikes. These findings demonstrate that learned inertial models require operational domain gating to verify vehicle context before activating dead reckoning updates.

5.5. Practical implications for smartphone navigation

For real-world automotive dead reckoning on consumer smartphones, these empirical findings suggest that learning-based velocity estimation can substantially outperform raw strapdown integration, but cannot operate reliably as an unconstrained black box. Robust dead reckoning requires hybrid architectures combining learned velocity estimation, decoupled kinematic damping, standstill detection, and conservative failure-aware filtering.

6. Limitations

Several scientific limitations characterize this study:

1. Single driver evaluation: All 64 passenger-car trips in the public IO-VNBD dataset were recorded by a single individual (Driver E). Consequently, all findings demonstrate whole-trip generalisation for a single driver profile; cross-driver generalisation was not evaluated and cannot be claimed.

2. Absence of physical road-vehicle testing: On-device mobile evaluations were restricted to golden telemetry replay, computational latency profiling, and handheld pedestrian stress tests. No physical road-vehicle driving tests were conducted with the Android application.

3. Engineering reference signals: Vehicle reference velocity was obtained from the factory CAN bus diagnostic stream via vehicle wheel encoders rather than dual-frequency carrier-phase RTK GNSS. Reference velocity is therefore subject to tyre slip and radius variations.

4. Sampling rate constraints: Spectral vibration analyses were conducted at the dataset's 10 Hz feature rate; high-frequency acoustic vibrations above 5 Hz were attenuated by anti-aliasing filtering.

5. Specific map matching formulation: The map matching evaluation reflects one specific multiple-hypothesis implementation; alternative topological map matchers may exhibit different feedback characteristics.

6. Competitive benchmark compliance: Under continuous 60-second satellite blackouts, the system achieved a 16.76% macro-average normalized drift across dynamic routes (96.65 m mean absolute drift across all 13 eligible routes) and met the sub-10% threshold on only 3 of 13 eligible routes (23.08%), indicating that universal compliance remains an open challenge.

7. Conclusions

This study has presented an empirical evaluation of smartphone-only inertial dead reckoning under extended GNSS blackouts. Using the public IO-VNBD benchmark dataset, we quantified the benefits and failure modes of learned velocity estimation, sensor fusion, kinematic constraints, map feedback, and operational distribution shifts.

Our findings show that scaling causal TCN training across diverse whole trips reduces velocity estimation error by 55.27% and decreases 60-second position drift by 64.72%. However, our ablation experiments also reveal that standard navigation assumptions can break down on consumer devices: unconditional lateral non-holonomic constraints degraded 60-second drift by 45.47%, while closed-loop map heading feedback degraded drift by 125.47%. Decoupling lateral velocity damping from attitude states restored filter stability, reducing drift by 54.62%. Across 13 held-out 60-second blackout routes, adaptive fusion achieved a mean absolute position drift of 96.65 metres (16.76% macro-average normalized error across the 12 dynamic routes), satisfying a competitive sub-10% drift benchmark on 23.08% of evaluated trips. These results emphasize the necessity of failure-aware design in smartphone dead reckoning architectures.

Data Availability Statement

The empirical evaluations in this study were conducted on the publicly available IO-VNBD benchmark dataset (Onyekpe et al., 2021), accessible under the Creative Commons Attribution 4.0 International licence at https://doi.org/10.1016/j.dib.2021.106885. Derived summary metrics and evaluation logs generated during this research are available within the project repository.

Code Availability Statement

The software code supporting the analyses, models, filtering pipelines, and Android deployment in this study is available under an open-source MIT licence at https://github.com/24f2001869/NavSense. All reproducibility materials, frozen configuration scripts, and evaluation logs for this manuscript are anchored to frozen release tag `paper-jon-v1.0`.

Acknowledgements

The author acknowledges the use of the Google Antigravity Agentic Assistant, powered by Gemini 2.5 Flash, during manuscript preparation and publication-engineering activities, including formatting, language harmonisation, consistency checking, and automated verification. The author independently verified the resulting manuscript, analyses, and scientific content, and retains full responsibility for the work.

Funding

No specific external funding was received for this research.

Competing Interests

The author declares no competing financial or non-financial interests.

AI Disclosure Statement

During manuscript preparation and auditing, the author utilized the Google Antigravity Agentic Assistant (powered by Gemini 2.5 Flash, Google DeepMind, September 2026) for publication engineering. Automated verification scripts cross-checked numerical tables, causality, and future-gradient leakage; underlying experimental data, models, filtering algorithms, and analyses were generated by the author. Assistance included formatting alignment with Cambridge guidelines, language harmonization, and CrossRef DOI verification. All visual figures were generated deterministically using Python/Matplotlib scripts; AI assistance was limited to script-editing support without generating visual content directly. The author reviewed all text and takes full responsibility for the manuscript.

References

Bai, S., Kolter, J. Z. and Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

Brossard, M., Bonnabel, S. and Barrau, A. (2020). AI-IMU Dead-Reckoning. IEEE Transactions on Intelligent Vehicles, 5(4), pp. 585–595. doi: 10.1109/TIV.2020.2980758.

Dissanayake, M. W. M. G., Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. (2001). The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation. IEEE Transactions on Robotics and Automation, 17(5), pp. 731–747. doi: 10.1109/70.964672.

Groves, P. D. (2013). Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems. 2nd ed. Artech House, Boston, MA, USA.

Klein, D., Cappelle, C., Ruichek, Y. and Rovetta, J. M. (2008). Multi-sensor fusion for land vehicle positioning using non-holonomic constraints and road map. In Proceedings of the 2008 IEEE Intelligent Vehicles Symposium (IV), Eindhoven, Netherlands, pp. 1104–1109. doi: 10.1109/IVS.2008.4621287.

Onyekpe, U., Palade, V., Kanarachos, S. and Szkolnik, A. (2021). IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning. Data in Brief, 35, 106885. doi: 10.1016/j.dib.2021.106885.

Qian, L., Lin, X., Niu, X., Huang, Q., Li, L., Guo, G., Wang, Z. and Chen, R. (2025). Avnet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF. Satellite Navigation, 6, 15. doi: 10.1186/s43020-025-00168-7.

Quddus, M. A., Ochieng, W. Y. and Noland, R. B. (2007). Current map-matching algorithms for transport applications: State-of-the art and future research directions. Transportation Research Part C: Emerging Technologies, 15(5), pp. 312–328. doi: 10.1016/j.trc.2007.05.002.

Shen, C., Zhang, Y. and Tang, X. (2019). A high-precision dead reckoning algorithm based on non-holonomic constraints and adaptive Kalman filtering for land vehicles. Sensors, 19(18), 3855. doi: 10.3390/s19183855.

Shin, B., Li, S. and Kim, B. (2025). Deep Learning-Based Vehicle Speed Estimation Using Smartphone Sensors in GNSS-Denied Environment. Applied Sciences, 15(16), 8824. doi: 10.3390/app15168824.

Skog, I., Nilsson, J. O. and Händel, P. (2010). Evaluation of zero-velocity detectors for pedestrian indoor positioning. In Proceedings of the 2010 International Conference on Indoor Positioning and Indoor Navigation (IPIN), Zurich, Switzerland, pp. 1–6. doi: 10.1109/IPIN.2010.5646936.

Solà, J. (2017). Quaternion kinematics for the error-state Kalman filter. arXiv preprint arXiv:1711.02508.

Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. F. (1999). A high integrity IMU/GPS navigation loop for autonomous land vehicle applications. IEEE Transactions on Robotics and Automation, 15(3), pp. 572–578. doi: 10.1109/70.768181.

Titterton, D. H. and Weston, J. L. (2004). Strapdown Inertial Navigation Technology. 2nd ed. Institution of Engineering and Technology (IET), Stevenage, UK.

White, C. E., Bernstein, D. and Kornhauser, A. L. (2000). Some map matching algorithms for personal navigation assistants. Transportation Research Part C: Emerging Technologies, 8(1–6), pp. 91–108. doi: 10.1016/S0968-090X(00)00026-4.

Xiao, X., Ren, X. and Li, H. (2025). An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU. arXiv preprint arXiv:2505.18490.

Yan, H., Shan, Q. and Furukawa, Y. (2020). RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, pp. 6128–6137. doi: 10.1109/CVPR42600.2020.00616.
