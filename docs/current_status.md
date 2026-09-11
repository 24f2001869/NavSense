# Current Project Status: The Definitive Assessment

**Last Updated**: 2026-09-11  
**Authoritative Scope**: Post-Phase 5.6 Comprehensive Evaluation  

This document provides a single, brutally honest answer to the question:  
**"Where does the SIH26168 Intelligent Dead Reckoning project stand today?"**

---

## 1. Executive Status: Has the SIH Benchmark Been Achieved?

### **The Honest Answer: NO.**
Under the current standalone smartphone-IMU-only approach, the system has **NOT** achieved the SIH requirement of universal $<10\%$ distance-drift across arbitrary ground vehicle trips during a 60-second GNSS blackout.

* **60-Second Outage Pass Rate**: Exactly **3 out of 13 usable trips passed <10% (23.1% pass rate)**.
* **10-Second Outage Pass Rate**: Exactly **6 out of 18 usable trips passed <10% (33.3% pass rate)**.
* **Aggregate Mean Drift at 60 s**: **16.76%** (Adaptive Regime Fusion) vs **18.26%** (Static TCN) vs **89.66%** (Pure Kinematics).

While causal regime-aware fusion achieved major physical victories—specifically eliminating the 500% urban quadratic divergence on `Vta26` (cutting drift from 575% to 47%) and passing high-speed mountain trips like `Vw12` at 3.01%—**47.3% is still a failure, and 23.1% pass rate is not universal compliance.**

---

## 2. Component Health Matrix

| Subsystem | Component | Status | Operational Maturity |
|:---|:---|:---:|:---|
| **Sensor Ingestion** | 100 Hz Android IMU FIFO + Monotonic Clock | 🟢 **Working** | Production-ready on Android |
| **State Estimation** | 15-State Java / Python ESKF | 🟢 **Working** | Formally verified, numerically stable |
| **Kinematic Bounds** | Causal Non-Holonomic Constraints (NHC) | 🟢 **Working** | Eliminates lateral skidding ($>60\%$ drift reduction) |
| **Neural Inference** | 65k Dilated TCN via ONNX Runtime | 🟢 **Working** | Real-time edge inference ($<8\text{ ms}$ per window) |
| **Stationary Detection** | Multi-Condition Zero Velocity Detection (ZVD) | 🟢 **Working** | Completely eliminates standstill quadratic runaway |
| **OOD Protection** | Input $\sigma$-clipping + Rotational Gating | 🟢 **Working** | Solves the pedestrian 70 m/s velocity spike |
| **Highway Cruise** | Straight-Line Cruise Speed Separation | 🔴 **Unsolved** | IMU acceleration & yaw identical to sensor noise floor |
| **Urban Stop-and-Go** | Micro-crawling ($0.5–2\text{ m/s}$) Integration | 🔴 **Unsolved** | Small velocity errors cause high % drift on short trips |
| **Cross-Trip 60s <10%** | Universal SIH Benchmark Compliance | 🔴 **Unachieved**| Bounded between 23% and 33% cross-trip pass rate |
| **In-Vehicle Testing** | Physical Automotive Road Validation | 🔴 **Missing**  | Real vehicle testing with synchronized reference pending |

---

## 3. Scientific Categorization

### 🟢 WHAT WE KNOW (Empirically Proven)
1. **Pure IMU integration diverges quadratically ($t^2$)**: Accelerometer bias $b_a \approx 0.1\text{ m/s}^2$ guarantees $\sim 180\text{ m}$ of drift in 60 seconds without aiding.
2. **Pedestrian motions are severely out-of-distribution**: Human walking gait generates yaw rates up to $+26.08\sigma$ outside automotive training priors, triggering the 70 m/s prediction artifact.
3. **Vibration contains zero generalizable speed information**: Across 64 trips and >450,000 windows, correlation between dominant vibration frequency and road speed is $r = -0.032$. Adding vibration features does not improve speed MAE ($4.39 \to 4.38\text{ m/s}$).
4. **Steady-state cruise speeds are weakly observable from IMU alone**: On straight highways ($a \approx 0, \omega \approx 0$), 85 km/h and 105 km/h driving are statistically indistinguishable from sensor noise ($ROC\text{-}AUC = 0.625$).
5. **Causal ZVD suppresses urban runaway**: On severe congestion trip `Vta26`, ZVD and adaptive damping reduced 60s drift from **574.6% (292.8 m) down to 47.3% (66.2 m)**.
6. **Kinematic momentum works on open mountain roads**: On `Vw12`, preserving pre-outage speed achieved **1.78% to 3.01% drift at 60 s**.

### 🟡 WHAT WE THINK (Strong Engineering Hypotheses)
1. **Long outages (>30 s) require external scale aiding**: A standalone smartphone sitting in a dashboard mount simply lacks the longitudinal metric scale that wheel-speed encoders or dual-frequency carrier-phase GNSS Doppler provide.
2. **Road geometry can observe along-track speed on curves**: Known digital road curvature ($\kappa = 1/R$) allows the gyroscope yaw rate to directly observe speed ($v = \omega / \kappa$) without wheel speed sensors.
3. **Discrete topological features can anchor along-track position**: Road bends, turns, and junctions provide discrete events that can reset longitudinal integration error.

### 🔴 WHAT WE DON'T KNOW (Unresolved / Unobservable)
1. Whether commercial digital maps (OSM) have sufficiently accurate curvature metadata to constrain speed without introducing false lane-change errors.
2. Whether dual-frequency (L1/L5) GNSS Doppler carrier-phase observables immediately prior to an outage can calibrate accelerometer bias tightly enough ($<0.01\text{ m/s}^2$) to allow pure integration for 60 seconds.
3. How real-world vehicle engine mounting stiffness, chassis dynamics, and suspension wear affect cross-vehicle generalization during live road testing.

---

## 4. Current Best System Architecture

The most capable and defensible architecture developed in this repository is:

$$\boxed{\textbf{Causal Adaptive Regime-Aware Velocity Fusion (Phase 5.6)}}$$

* **Stationary ($v < 2.5\text{ m/s}, \sigma_a < 0.16$ m/s²)**: Clamped to $0.0\text{ m/s}$ via ZVD.
* **Stable Cruise ($v > 15\text{ m/s}, \|\boldsymbol{\omega}\| < 0.04$ rad/s)**: $\beta = 0.993$ (preserves pre-outage GNSS momentum).
* **Dynamic Driving**: $\beta = 0.970$ (complementary blending of kinematics and Dilated TCN).
* **Disagreement Gating ($|v_{\text{kin}} - v_{\text{TCN}}| > 4.5\text{ m/s}$)**: Acceleration clipped to $\pm 1.5\text{ m/s}^2$ with $\beta = 0.950$.

This system achieves **16.76% aggregate drift at 60 seconds**, but is transparently acknowledged to be an experimental research prototype rather than a completed commercial product.
