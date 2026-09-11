# Architectural Decision Record: Rejected Approaches

This document details hypotheses, algorithms, and architectural variants that were evaluated empirically and **definitively rejected**. In research engineering, documenting what failed—and why—is just as valuable as documenting what succeeded.

---

## 1. Standalone Pure IMU Double Integration

* **What We Tried**: Direct double-integration of smartphone accelerometer and gyroscope measurements using classical strapdown mechanization without external updates.
* **Why We Tried It**: Standard baseline to evaluate how long consumer MEMS inertial sensors can navigate autonomously.
* **What Happened**: Position error grew quadratically with accelerometer bias ($\frac{1}{2} b_a t^2$) and cubically with gyroscope tilt errors ($\frac{1}{6} b_g g t^3$). By 30 seconds, drift exceeded $280\text{ m}$; by 60 seconds, drift exceeded $800\text{--}1,500\%$ of distance.
* **Why It Was Rejected**: Consumer smartphone IMUs have uncalibrated run-to-run biases ($\sim 0.1\text{--}0.2\text{ m/s}^2$) that mathematically guarantee catastrophic divergence within 15–20 seconds.
* **Supporting Evidence**: [`docs/experiments/phase1_baseline.md`](../experiments/phase1_baseline.md), [`results/baseline/`](../../results/baseline/).

---

## 2. Unconditional / Ungated Non-Holonomic Constraints (NHC)

* **What We Tried**: Applying continuous vehicle zero-lateral ($v_y^v = 0$) and zero-vertical ($v_z^v = 0$) velocity updates directly in the raw phone coordinate frame.
* **Why We Tried It**: Wheeled ground vehicles rarely slip sideways under normal driving.
* **What Happened**: When a smartphone was mounted with a slight pitch or roll angle in a dashboard mount, forward longitudinal acceleration leaked into the lateral axis. The filter interpreted forward vehicle acceleration as lateral slip, corrupting heading and forcing the estimated trajectory into circular swerves.
* **Why It Was Rejected**: NHC is only mathematically valid when projected through a dynamically calibrated phone-to-vehicle mounting matrix ($\mathbf{R}_b^v$).
* **Supporting Evidence**: [`docs/experiments/phase3_attitude.md`](../experiments/phase3_attitude.md), Stage C8-10b audit report.

---

## 3. Naive Map Heading Replacement / Snapping

* **What We Tried**: Forcing the vehicle's heading to match the nearest OpenStreetMap polyline segment azimuth ($\psi_{\text{meas}} = \psi_{\text{road}}$) during outages (Phase 4.6 Variant M2).
* **Why We Tried It**: To eliminate gyroscope yaw integration drift by assuming the vehicle stays centered on the road.
* **What Happened**: Produced a **125.5% regression in 60s position drift** (increasing drift from $578\text{ m} \to 1,304\text{ m}$). Road polylines have coarse chords and vehicles change lanes; forcing the heading caused the ESKF to fight the gyroscope, inducing massive artificial turn errors.
* **Why It Was Rejected**: Direct heading snapping destroys physical state continuity and causes catastrophic divergence during road curves.
* **Supporting Evidence**: [`results/phase4_6_map/phase4_6_map_matching_report.md`](../../results/phase4_6_map/phase4_6_map_matching_report.md).

---

## 4. Spectral Vibration as a Primary Vehicle Speedometer

* **What We Tried**: Extracting rolling FFT peak frequencies and multi-band Power Spectral Density (PSD) from smartphone accelerometer channels to predict forward vehicle speed (Phase 5.3).
* **Why We Tried It**: Hypothesized that tire rotation harmonics ($f = v / 2\pi r$) or chassis vibration power would scale with vehicle road speed.
* **What Happened**: Across 64 vehicle trips (>450,000 windows), the correlation between dominant vibration frequency and forward speed was essentially zero ($r = -0.032$). On motorways, the dominant peak remained locked at $\sim 2.9\text{ Hz}$ (chassis suspension bounce) regardless of whether the car did 85 or 105 km/h. Adding vibration features reduced speed error by an imperceptible $0.01\text{ m/s}$ ($0.2\%$).
* **Why It Was Rejected**: Commercial smartphones in dashboard mounts are isolated by vehicle suspension and rubber mount damping, filtering out high-frequency tire harmonics.
* **Supporting Evidence**: [`docs/experiments/phase5_3_vibration_audit.md`](../experiments/phase5_3_vibration_audit.md), [`results/vibration_audit/`](../../results/vibration_audit/).

---

## 5. Pedestrian Walking Tests as Vehicle Navigation Validation

* **What We Tried**: Walking with the phone in hand across campus (hostel to mess hall, rooftop) and treating the trajectory as an early system validation test.
* **Why We Tried It**: Rapid initial physical sanity check without arranging a motor vehicle.
* **What Happened**: Pedestrian arm swing and body rotations generated yaw rates of $+26.08\sigma$ outside vehicle training bounds, causing the Expanded TCN to predict **71.66 m/s (258 km/h)**.
* **Why It Was Rejected**: Pedestrian biomechanics radically violate ground vehicle kinematics and non-holonomic constraints.
* **Supporting Evidence**: [`docs/experiments/phase5_1_field_forensics.md`](../experiments/phase5_1_field_forensics.md), [`results/phase5_1_field_forensic_report.md`](../experiments/phase5_1_field_forensics.md).

---

## 6. Blind "More Neural Networks" as the Solution to Highway Underestimation

* **What We Tried**: Exploring whether training deeper neural networks, LSTMs, or Transformers on the same smartphone IMU streams would solve the highway speed plateau.
* **Why We Tried It**: Common machine learning assumption that underfitting is solved by greater model capacity.
* **What Happened**: Phase 5.4 proved mathematically that during steady-state cruising on a smooth highway, acceleration and angular rate are identical to the sensor noise floor ($ROC\text{-}AUC = 0.625$). The information is physically missing from the instantaneous IMU stream.
* **Why It Was Rejected**: An algorithm cannot extract information that does not physically exist in the input sensor signals.
* **Supporting Evidence**: [`results/highway_observability/highway_observability_report.md`](../../results/highway_observability/highway_observability_report.md).
