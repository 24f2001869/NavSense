# Catalog of Failed Experiments & Negative Results

> **Scientific Philosophy:** Negative results, model collapses, and unobservability boundaries are essential research discoveries. In this repository, failed experiments are preserved with full diagnostic evidence.  
> **Rule:** Every failure documents: What We Expected, What Happened, Why It Mattered, What We Learned, and What We Changed Afterward.

---

## 1. Pure IMU Strapdown Double Integration
* **Phase:** Phase 1 (Pure IMU Baseline)
* **WHAT WE EXPECTED:** We expected numerical integration of linear acceleration ($v = \int a \, dt, p = \int v \, dt$) to provide usable dead reckoning position for at least 10–30 seconds of GNSS outage.
* **WHAT HAPPENED:** Position error diverged cubically ($\sim t^3$). Even with calibrated resting biases, uncompensated smartphone sensor noise produced hundreds of meters of drift within 15 seconds ($>1000\%$ error).
* **WHY IT MATTERED:** It definitively proved that consumer-grade smartphone MEMS accelerometers cannot sustain open-loop double integration without continuous velocity bounding.
* **WHAT WE LEARNED:** Dead reckoning requires an independent speed observation or constraint to bound velocity error growth before integrating position.
* **WHAT WE CHANGED AFTERWARD:** We moved immediately to an Error-State Kalman Filter (ESKF) architecture and began investigating AI-based forward speed estimation.

---

## 2. GNSS-Aided ESKF Without Speed Observability
* **Phase:** Phase 2 (Classical ESKF)
* **WHAT WE EXPECTED:** A 15-state ESKF (position, velocity, attitude, gyro bias, accel bias) would retain accurate bias estimates during GNSS availability and coast smoothly through 60-second blackouts.
* **WHAT HAPPENED:** Sub-meter accuracy was maintained while GNSS was active, but immediately upon GNSS loss, velocity errors began accumulating linearly and position errors quadratically ($\sim t^2$). Without speed measurements, covariance inflated and trajectory drifted off-course.
* **WHY IT MATTERED:** Confirmed that Kalman filtering without speed updates during an outage is still subject to random walk drift.
* **WHAT WE LEARNED:** An ESKF needs synthetic velocity updates (virtual pseudo-measurements) during outages to constrain the velocity error state $\delta v$.
* **WHAT WE CHANGED AFTERWARD:** Formulated virtual velocity measurement updates ($z = [v_{\text{est}}, 0, 0]^T$) injected into the filter measurement matrix $H$.

---

## 3. Gyroscope Frame Mismatch with Non-Holonomic Constraints (NHC)
* **Phase:** Phase 3 (Attitude & NHC)
* **WHAT WE EXPECTED:** Non-Holonomic Constraints (assuming ground vehicles cannot slip sideways or fly: $v_y \approx 0, v_z \approx 0$) would clamp lateral position drift.
* **WHAT HAPPENED:** While lateral drift was bounded, any slight physical tilt or misalignment between the phone mounting axis and the vehicle forward axis caused the filter to interpret forward speed as lateral slip, injecting artificial turning moments and warping straight highway trajectories into spirals.
* **WHY IT MATTERED:** Highlighted the acute sensitivity of classical mechanical constraints to unmodeled frame misalignment.
* **WHAT WE LEARNED:** Rigidly enforcing NHC in a consumer smartphone context without dynamic auto-alignment causes more harm than unconstrained dead reckoning.
* **WHAT WE CHANGED AFTERWARD:** Softened NHC measurement covariances and restricted constraints to motion regimes where phone leveling and heading alignment were statistically confident.

---

## 4. Cross-Trip Generalization Failure of Shallow ML (Ridge & Random Forest)
* **Phase:** Phase 4 (AI Speed Estimation)
* **WHAT WE EXPECTED:** Feature engineering (rolling mean/std of accel, jerk, spectral energy) fed into Random Forest or Ridge regression would accurately predict vehicle forward speed.
* **WHAT HAPPENED:** Models achieved high $R^2 > 0.92$ when evaluated on random train-test splits of the *same* trip, but collapsed to near-zero $R^2$ ($<0.15$) and massive errors ($>8\text{ m/s}$) when tested on held-out trips driven by different drivers or on different vehicles.
* **WHY IT MATTERED:** Exposed massive time-series data leakage in intra-trip random splits; proved hand-crafted summary features fail to capture driving dynamics across different routes.
* **WHAT WE LEARNED:** Machine learning for inertial navigation must be evaluated strictly using **disjoint whole-trip splits**, and models must ingest sequential temporal dependencies rather than static feature windows.
* **WHAT WE CHANGED AFTERWARD:** Enforced strict trip-level partitioning and migrated to causal 1D Temporal Convolutional Networks (TCN).

---

## 5. TCN High-Speed Highway Flatline (~85 km/h Collapse)
* **Phase:** Phase 5.2 / Phase 5.4 (Highway Speed Observability)
* **WHAT WE EXPECTED:** An expanded TCN trained across 64 trips would track vehicle speed from 0 to 120 km/h across all environments.
* **WHAT HAPPENED:** On straight motorways (`V-Vfa02`), once the vehicle cruised at constant speed for longer than the TCN's 6.1-second receptive field, the model output collapsed to $\sim 85\text{--}89\text{ km/h}$, completely unable to distinguish 95 km/h from 115 km/h.
* **WHY IT MATTERED:** Revealed an intrinsic physics limit: when acceleration is zero ($\ddot{x} = 0$) and turn rate is zero ($\dot{\psi} = 0$), Newtonian mechanics states that constant velocity produces zero inertial force. The IMU input is identical at 60 km/h and 120 km/h.
* **WHAT WE LEARNED:** A memory-bounded neural network receiving only IMU inputs cannot observe absolute velocity during unaccelerated, un-turned motion. It defaults to the empirical conditional mean of its training set.
* **WHAT WE CHANGED AFTERWARD:** Introduced stateful kinematic tracking ($v(t) = v_{\text{prior}} + \int a \, dt$) to carry forward the last known GNSS velocity into smooth cruise regimes.

---

## 6. Pedestrian / Out-Of-Distribution (OOD) Explosion (~71.66 m/s Peak)
* **Phase:** Phase 5.1 (Field Forensics)
* **WHAT WE EXPECTED:** Testing the Android prototype during a walking test across a campus rooftop would demonstrate smooth zero-to-low-speed tracking.
* **WHAT HAPPENED:** The TCN predicted insane speeds exceeding $71.66\text{ m/s}$ ($257.9\text{ km/h}$) while the user was walking at $\sim 1.4\text{ m/s}$.
* **WHY IT MATTERED:** Showed that deep sequence models deployed on smartphones without safety gating can produce catastrophic predictions when subjected to out-of-distribution human motion.
* **WHAT WE LEARNED:** Human walking involves cyclic arm swings with angular velocities up to $3.81\text{ rad/s}$—nearly 10 times higher than the vehicle dataset maximum ($0.40\text{ rad/s}$). The network's linear projection multiplied this extreme feature into impossible speed predictions.
* **WHAT WE CHANGED AFTERWARD:** Implemented strict input distribution clamping, OOD novelty detection, and verified that walking sensor traces must never be conflated with vehicle validation.

---

## 7. Vibration Speed Correlation Failure ($r = -0.032$)
* **Phase:** Phase 5.3 (Vibration Audit)
* **WHAT WE EXPECTED:** We hypothesized that acoustic/mechanical chassis vibration captured by high-frequency smartphone accelerometers would shift upward in frequency with wheel speed and engine RPM, providing an unaccelerated speed proxy.
* **WHAT HAPPENED:** Analysis of 64 trips (>450,000 temporal windows) revealed that dominant vibration frequency had a Pearson correlation of $r = -0.032$ with CAN forward speed. On motorways, correlation was $r = +0.081$.
* **WHY IT MATTERED:** Conclusively eliminated vibration frequency analysis as a viable primary speedometer on consumer smartphones.
* **WHAT WE LEARNED:** In modern passenger vehicles, cabin suspension, tires, and rubber phone mounts dampen high-frequency wheel harmonics. The dominant 2.2–2.5 Hz peaks reflect chassis natural resonant frequency, not vehicle speed.
* **WHAT WE CHANGED AFTERWARD:** Rejected vibration spectral features from the model feature dictionary and avoided introducing false complexity.

---

## 8. Closed-Loop Map-Matching Trajectory Snapping Failure
* **Phase:** Phase 4 Exploration (Map-Matching)
* **WHAT WE EXPECTED:** Snapping dead-reckoned positions to OpenStreetMap road vectors would fix lateral drift in urban canyons.
* **WHAT HAPPENED:** When dead reckoning accumulated lateral heading errors exceeding road half-width, the nearest-neighbor map-matcher snapped the position onto parallel service alleys, opposite lanes, or perpendicular cross-streets, creating irreversible topological trajectory tears.
* **WHY IT MATTERED:** Demonstrated that closed-loop map feedback amplifies orientation errors if the underlying inertial heading is unconstrained.
* **WHAT WE LEARNED:** Map-matching must operate as a loose downstream visualization layer or probabilistic particle filter, never as a hard constraint feeding back into the ESKF error state.
* **WHAT WE CHANGED AFTERWARD:** Decoupled map display from core inertial dead reckoning; the core filter propagates in metric Cartesian space without premature map warping.

---

## 9. Stateful Kinematic Stop-and-Go Runaway (574.65% Urban Drift)
* **Phase:** Phase 5.5 (Stateful Kinematics)
* **WHAT WE EXPECTED:** Integrating longitudinal acceleration from the initial GNSS speed would solve the TCN steady-cruise flatline universally.
* **WHAT HAPPENED:** On smooth highway trips (`V-Vfa02`), drift dropped to a remarkable 11.69%. But on urban trip `Vta26` with multiple traffic lights and stops, tiny accelerometer biases ($\sim 0.05\text{ m/s}^2$) integrated into phantom speeds during stops, causing position drift to explode to **574.65%** (562.9 m).
* **WHY IT MATTERED:** Proved that pure kinematic integration is dangerous in urban environments with frequent stops.
* **WHAT WE LEARNED:** Kinematics requires continuous physical clamping by a Zero-Velocity Detector (ZVD) whenever the vehicle is stationary.
* **WHAT WE CHANGED AFTERWARD:** Engineered a causal Zero-Velocity Detector that overrides kinematic momentum and forces velocity to zero when energy and variance drop below stationary thresholds.

---

## 10. Adaptive Dynamic Fusion Remaining Limitations (3 / 13 Passes at 60s)
* **Phase:** Phase 5.6 (Adaptive Regime Fusion Benchmark)
* **WHAT WE EXPECTED:** Blending stateful kinematics, causal TCN, and ZVD would achieve $<10\%$ drift across all 19 held-out test trips.
* **WHAT HAPPENED:** Adaptive fusion successfully prevented urban runaway (`Vta26` drift was cut from 574.65% down to 47.31%) and maintained excellent tracking on winding routes (`Vw12` reached 3.01%), bringing overall 60s drift to **16.76% (96.65 m)**. However, only **3 of 13 usable trips (23.1%)** passed the strict $<10\%$ threshold.
* **WHY IT MATTERED:** Conclusively defined the empirical capability boundary of smartphone-only inertial dead reckoning without external reference signals.
* **WHAT WE LEARNED:** Without magnetometer heading reference or wheel speed sensors, residual gyroscope bias drift over 60 seconds causes angular trajectory deflection even when speed magnitude is accurate.
* **WHAT WE CHANGED AFTERWARD:** We frozen technical experimentation, published the exact empirical boundaries honestly, and documented the open problems in [`docs/decisions/open_questions.md`](decisions/open_questions.md).
