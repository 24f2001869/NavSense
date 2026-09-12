# Accessibility Descriptions (WCAG 2.1 Level AA): Figures & Tables

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Standard:** Web Content Accessibility Guidelines (WCAG) 2.1 Level AA / Cambridge Journals Artwork Guide  
**Author:** Rahul Kumar  
**Audit Date:** 2026-09-12  

Cambridge University Press requires that all figures, tables, and multimedia content be accessible to readers with visual or cognitive impairments. This document provides short alternative text ("alt-text"), extended structural image descriptions, and essential numerical takeaways for every figure and table, accurately reflecting the final rendered artwork.

---

## 1. Figure Accessibility Descriptions

### Figure 1: System Architecture and Data Flow of NavSense
* **Short Alt-Text**: Block diagram of the NavSense smartphone dead reckoning architecture showing IMU preprocessing, 4-block causal dilated TCN velocity estimation, zero-velocity detection, 15-state ESKF with decoupled lateral velocity damping, and downstream shadow map tracking.
* **Long Image Description**: A multi-layer architecture diagram organized from top to bottom into five sequential functional layers:
  1. *Layer 1 (Raw Sensors)*: Two input blocks representing a consumer smartphone IMU (accelerometer and gyroscope sampled at ~400 Hz hardware rate and ~8.63 Hz Android navigation callback rate; CAN forward velocity as engineering reference) and intermittent smartphone GNSS (1 Hz PVT fix used for pre-outage initialization).
  2. *Layer 2 (Preprocessing)*: A single box showing gravity compensation via orientation quaternion subtraction and formation of a rolling 100-sample sliding buffer yielding a 9-channel kinematic tensor.
  3. *Layer 3 (Parallel Estimators)*: Two parallel branches. Branch A contains the causal Dilated TCN network (4 residual blocks, dilations {1, 2, 4, 8}, kernel size 3, 60,225 parameters, 61-sample receptive field corresponding to 6.0-second causal look-back at 10 Hz) emitting forward velocity estimates. Branch B contains the Zero-Velocity Detector (ZVD) clamping velocity during detected stationary intervals.
  4. *Layer 4 (State Estimation)*: A central 15-state Error-State Kalman Filter (ESKF) tracking position error, velocity error, attitude error, accelerometer bias, and gyroscope bias. A dedicated callout highlights Decoupled Velocity Damping ($K_y[6:15] = 0$) restricting lateral velocity updates strictly to velocity states.
  5. *Layer 5 (Downstream Services)*: Output dead-reckoning trajectory display and open-loop shadow map matching with closed-loop heading feedback disabled.
* **Essential Reader Takeaway**: Shows how raw sensor signals are processed through causal deep sequence estimation and filtered through an ESKF with decoupled lateral damping to prevent attitude corruption during satellite outages.

---

### Figure 2: Open-Loop Inertial Divergence During 60-Second GNSS Outage
* **Short Alt-Text**: Trajectory map and error drift curve demonstrating rapid nonlinear position divergence of unassisted strapdown inertial double-integration during a 60-second satellite blackout on test route Vta20.
* **Long Image Description**: Two-panel plot:
  - *Panel (a) (Planar Trajectory)*: Compares the reference vehicle route (solid green line, 448.2 m traveled) against the unassisted strapdown INS dead reckoning trajectory (dashed red line) over a 60-second window on trip `Vta20`. The unassisted trace rapidly deviates from the road geometry, drifting across road boundaries and accumulating 324.2 metres of cumulative horizontal drift at the 60-second endpoint.
  - *Panel (b) (Position Drift vs. Outage Time)*: A line graph showing horizontal position error in metres on the vertical axis (0 to 360 m) against elapsed outage time from 0 to 65 seconds on the horizontal axis for route `Vta20`. Data points highlight route-specific divergence: 10 s: 23.7 m, 20 s: 57.0 m, 30 s: 109.6 m, and 60 s: 324.2 m. An orange dotted horizontal line marks the nominal lane-scale tolerance (3.75 m), annotated with 'Nominal lane-scale tolerance exceeded at t ≈ 2.1 s'.
* **Essential Reader Takeaway**: Illustrates why pure strapdown double-integration of consumer smartphone MEMS sensors fails rapidly without external velocity aiding, accumulating 324.2 metres of drift on route `Vta20` (multi-route mean values across all eligible held-out routes are reported separately in Table 3).

---

### Figure 3: Controlled Scaling: 6-Trip vs. 39-Trip TCN Generalization Benchmark
* **Short Alt-Text**: Four-panel comparative visualization contrasting held-out forward velocity estimation accuracy, integrated position drift at 30 s and 60 s, cross-route generalization across four road environments, and per-trip error variance across 19 test routes between 6-trip (1.1 h) and 39-trip (12.5 h) training datasets.
* **Long Image Description**: Four-panel comparative visualization across 19 held-out test routes (115,420 prediction epochs, 240.6 km):
  - *Panel (a) (Forward Velocity MAE and RMSE)*: Bar chart showing Mean Absolute Error dropping from 6.17 m/s with 6 training trips down to 2.76 m/s with 39 training trips (a 55.27% reduction). RMSE drops from 6.96 m/s to 3.59 m/s (a 48.42% reduction).
  - *Panel (b) (Integrated Position Drift at 30 s & 60 s Blackout Horizons)*: Grouped bar chart comparing integrated dead reckoning position drift. At 30 seconds of GNSS blackout, mean drift decreases from 153.3 metres (6-trip model) to 52.6 metres (39-trip model), a 65.67% reduction. At 60 seconds of blackout, mean drift decreases from 263.6 metres to 93.0 metres, representing a 64.72% reduction.
  - *Panel (c) (Cross-Route Generalisation Across Road Environments)*: Grouped bar chart depicting forward velocity MAE across four distinct driving environments: Suburban Towns (dropping from 3.11 to 1.74 m/s, a 44.05% reduction), Dense Urban (4.65 to 2.18 m/s, a 53.12% reduction), Mountain Winding Routes (6.84 to 3.25 m/s, a 52.49% reduction), and High-Speed Motorways (8.42 to 3.42 m/s, a 59.38% reduction).
  - *Panel (d) (Per-Trip Error Variance Across 19 Held-Out Test Routes)*: Box-and-whisker plot displaying the distribution of forward velocity prediction errors across all 19 individual held-out test routes. Expanding the training dataset from 6 to 39 trips substantially compresses inter-quartile ranges, reduces median MAE, and tightens error dispersion across all routes.
* **Essential Reader Takeaway**: Quantifies the systematic multi-trip scaling benefit on forward speed accuracy, demonstrates that 60-second drift decreases by 64.72%, and confirms consistent generalization across diverse road environments and test routes.

---

### Figure 4: Kinematic Constraint Instability and Decoupled Recovery
* **Short Alt-Text**: Trajectory plots, normalized innovation squared ($NIS_x$) time-series across 83 windows, and heading error curves contrasting standard coupled Non-Holonomic Constraints against decoupled velocity damping.
* **Long Image Description**: Three-panel plot:
  - *Panel (a) (Trajectory Comparison)*: Compares the ground reference route against coupled NHC (Variant F, terminating with 1226.9 m drift), TCN-only (Variant E, terminating with 843.4 m drift), and the proposed decoupled damping architecture (Variant V1, terminating with 382.7 m drift). Variant V1 closely follows the true road trajectory.
  - *Panel (b) (Innovation Instability Across 83 Windows)*: Forward normalized innovation squared ($NIS_x$) across 83 evaluation windows. Coupled NHC experiences severe spikes during turns peaking at 82.12 (mean 82.12), whereas TCN alone (mean 6.55) and Decoupled Damping (mean 6.60) remain bounded near the empirical baseline reference level of 6.55.
  - *Panel (c) (Heading Attitude Error)*: Shows heading error escalating past 16 degrees under coupled NHC (annotated with 'Lateral Innovation → Attitude-State Corruption'), while decoupled damping ($K_y[6:15] = 0$) keeps heading error below 3.5 degrees (margin 5.0 degrees).
* **Essential Reader Takeaway**: Demonstrates that unconditional lateral NHC updates inject spurious turn innovations into attitude states due to cornering sideslip, whereas decoupled damping isolates innovations to velocity decay, reducing drift by 54.62%.

---

### Figure 5: Closed-Loop Road Network Map Matching Breakdown Under GNSS Outages
* **Short Alt-Text**: Map view comparing trajectory propagation under decoupled open-loop shadow map matching versus closed-loop heading feedback across OpenStreetMap corridors.
* **Long Image Description**: Two-panel geographical map display:
  - *Panel (a) (Shadow Map - Open-Loop M0/M1)*: Vehicle trajectory during a 60-second blackout where road matching is used strictly for downstream display. The filter dead-reckons smoothly along the arterial road with a mean 60-second drift across 83 evaluation windows of 578.3 metres, 0.0% wrong-road association, and heading feedback disabled.
  - *Panel (b) (Closed-Loop Heading Injection M2)*: The same evaluation protocol with closed-loop map matching enabled. Minor initial heading jitter causes the algorithm to latch onto an adjacent, non-parallel side street at an ambiguous junction. The forced heading correction drives the Kalman filter into an unrecoverable diverging branch, resulting in a mean 60-second drift across 83 evaluation windows of 1,303.9 metres (+125.47% degradation), heading error surge to 73.5°, and a regression rate of 47.0%.
* **Essential Reader Takeaway**: Demonstrates that, in the tested implementation, closed-loop map heading injection can cause severe divergence when the matcher latches onto an incorrect road link, confirming that digital road geometry is safer when retained for downstream open-loop visualization.

---

### Figure 6: Highway Cruise Speed Separability and Negative Prediction Bias
* **Short Alt-Text**: Receiver Operating Characteristic (ROC) curve and speed-dependent residual scatter plot demonstrating weak statistical separability during steady-state highway cruising.
* **Long Image Description**: Two-panel analysis on 112 minutes of motorway driving (`V-Vfa02`, 163 km):
  - *Panel (a) (ROC Curve)*: Binary classification ROC curve discriminating between 80 km/h and 110 km/h cruising based on 9 causal kinematic channels (6 IMU + 3 derived) using 5-fold stratified cross-validation. The curve lies close to the diagonal chance line, yielding an area under the curve ($\text{ROC-AUC}$) of 0.625 (classification accuracy 59.47%, sensitivity 63.2%, specificity 62.0% at operating point).
  - *Panel (b) (Residual vs. Vehicle Speed)*: Scatter plot of signed velocity residual ($\hat{v} - v^*$) against true vehicle speed. At lower speeds (0–50 km/h), residuals are centered near zero. In the high-speed regime ($\ge 90\text{ km/h}$, $v \ge 25\text{ m/s}$), residuals become strongly negative, averaging $-3.5\text{ m/s}$ ($r = -0.4866$), illustrating systematic regression toward the training set mean.
* **Essential Reader Takeaway**: Empirically confirms that constant-speed highway cruising produces near-zero longitudinal specific force, leaving 80 km/h vs 110 km/h cruising weakly separable from 10-Hz smartphone IMU signals and causing neural networks to regress toward the training set mean.

---

### Figure 7: Spectral Vibration Speed Audit Across 64 Vehicle Trips
* **Short Alt-Text**: Power spectral density line plots across speed bands and correlation scatter plot showing that smartphone accelerometer vibration frequency is invariant to vehicle speed across >450,000 one-second windows.
* **Long Image Description**: Two-panel spectral audit:
  - *Panel (a) (Acceleration PSD on Motorway `V-Vfa02`)*: Power spectral density (PSD) of vertical acceleration plotted against frequency from 0.5 to 8.0 Hz for three steady-state speed bands: 80–90 km/h (blue curve), 90–100 km/h (orange curve), and 100–110 km/h (green curve). All three curves exhibit a prominent, persistent peak between 2.86 and 2.93 Hz (highlighted by a yellow persistent peak band and vertical dashed line at 2.90 Hz), with no monotonic upward frequency shift across speed bands.
  - *Panel (b) (Dominant Frequency vs. Speed Scatter)*: Population scatter plot across 64 passenger-car trips (>450,000 one-second windows) showing within-trip dominant frequency correlation $r(f_{\text{dom}}, v)$ against average speed (15 to 115 km/h) across four driving categories (Suburban N=30, Dense Urban N=12, Mountain N=20, Motorway N=2). The population mean Pearson correlation is $r = -0.032$ (horizontal dashed red line) and mean Spearman correlation is $\rho = -0.028$, situated entirely within the $\pm 0.10$ null correlation band.
* **Essential Reader Takeaway**: Provides no evidence that the tested 10-Hz smartphone vibration features provide a reliable forward speed signal; the persistent low-frequency peak is consistent with vehicle-body/chassis dynamics, but its physical mechanism was not directly identified.

---

### Figure 8: Pedestrian Out-of-Distribution Dynamics and Android Edge Timing Verification
* **Short Alt-Text**: Four-panel edge deployment diagnosis illustrating pedestrian arm-swing induced velocity spikes, input channel ablation waterfall, Android sensor callback timing distribution, and on-device execution latency breakdown.
* **Long Image Description**: Four-panel edge deployment verification:
  - *Panel (a) (Pedestrian Walk Telemetry)*: Time-series of forward speed during handheld walking over 120 seconds, comparing reference walking speed (~1.1 m/s) against the vehicle-trained TCN predicted forward speed. The TCN oscillates around 10–15 m/s and exhibits an anomalous out-of-distribution spike reaching 71.66 m/s at t = 65 s (annotated with '+26.08$\sigma$ arm-swing yaw rate').
  - *Panel (b) (Peak Prediction Under Feature Ablation)*: Horizontal bar chart showing predicted peak speed across systematic channel zeroing. Ablating all gyroscope channels drops peak speed from 71.66 m/s down to 27.71 m/s (a 61.3% reduction), with individual-axis ablations yielding 34.9 m/s for yaw, 52.2 m/s for pitch, and 29.9 m/s for roll, indicating that rotational channels make a substantial contribution to the observed out-of-distribution failure.
  - *Panel (c) (Mobile Android Sensor Dispatch Interval)*: Histogram of measured inter-epoch Android sensor callback intervals ($\Delta t$) on a Google Pixel 7a smartphone (5,000 samples, range 104 to 127 ms). Vertical dashed markers indicate the 115.8 ms empirical mean interval (effective callback rate 8.63 Hz) and the nominal 100.0 ms (10.0 Hz) interval.
  - *Panel (d) (On-Device Execution Latency Breakdown)*: Grouped bar chart showing mean (9.15 ms) and 99th-percentile (13.89 ms) execution latency across navigation pipeline components on Google Pixel 7a, comfortably within the 100 ms epoch budget (marked by a dashed red line at 100 ms).
* **Essential Reader Takeaway**: Reveals the vulnerability of vehicle-trained sequence models to out-of-distribution pedestrian arm-swing rotations, pinpoints rotational coupling via channel ablation, and confirms that while the computational pipeline easily meets the 100-ms epoch budget, Android OS sensor dispatch averages 8.63 Hz, necessitating hardware nanosecond timestamp handling.

---

## 2. Table Accessibility Guidelines

1. **Table 1 (Dataset Census)**: Clear column headers with units in parentheses (`Duration (h)`, `Distance (km)`, `Mean Speed (km/h)`). No merged cells across data rows.
2. **Table 2 (Partition Split)**: Explicitly delineates trip count ($N$), usable model samples, and duration. A footnote defines the single-driver constraint.
3. **Table 3 (Classical Baselines and Multi-Horizon Divergence)**: Outage horizons clearly grouped (10 s, 20 s, 30 s, 60 s) with drift reported in both absolute metres and relative percentages.
4. **Table 4 (TCN Scaling)**: Direct before-and-after comparison with absolute and relative delta columns.
5. **Table 5 (Kinematic Ablation Across 83 Controlled Windows)**: Clearly contrasts coupled NHC degradation (+45.47%) against decoupled damping improvement (+54.62%) with accompanying innovation statistics ($NIS$).
6. **Table 6 (Map Matching Ablation)**: Categorizes map integration modes (M0, M1, M2, M3, M4, M5) and explicitly identifies closed-loop failure and regression rates across 83 windows.
7. **Table 7 (Held-Out Route Benchmark Across 13 Test Trips)**: Category-level benchmark summary across 13 eligible 60-s blackout trajectories (Suburban, Mountain, Motorway, Stationary Control, and Overall Benchmark), with complete route-by-route forensic metrics provided in Supplementary Table S1.
