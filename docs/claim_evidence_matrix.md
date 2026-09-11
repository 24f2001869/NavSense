# Claim / Evidence Matrix

To guarantee scientific credibility and prevent accidental overclaiming in papers, presentations, and competitions, this matrix defines the authoritative evidentiary status and recommended safe wording for every technical claim in this project.

---

## 1. Evidentiary Ratings Legend
* 🟢 **Demonstrated**: Formally verified through controlled, reproducible experiments on held-out test data.
* 🟡 **Supported Hypothesis**: Strongly indicated by empirical data, but subject to specific boundary conditions or partial observability.
* 🔴 **Unsupported / Rejected**: Disproven empirically, mathematically unobservable, or unsubstantiated by evidence.

---

## 2. Core Technical Claims Matrix

| Technical Claim | Primary Evidence | Evidentiary Rating | Recommended Safe Wording | Forbidden / Overclaiming Wording |
|:---|:---|:---:|:---|:---|
| **Adaptive regime fusion reduces urban quadratic divergence** | On `Vta26` (congestion), 60s drift dropped from **574.6% (292.8 m) to 47.3% (66.2 m)** via causal ZVD | 🟢 **Demonstrated** | *"Causal Zero Velocity Detection and adaptive damping substantially mitigate quadratic integration error during urban stops."* | *"ZVD completely eliminates urban dead-reckoning drift."* |
| **Dilated TCN achieves high accuracy on winding mountain roads** | `Vw12` achieved **3.01% to 3.51% drift**, `Vw14a` achieved **4.91% to 6.93% drift** at 60 s | 🟢 **Demonstrated** | *"On winding roads where lateral centripetal dynamics are rich, the Dilated TCN provides accurate forward speed estimation."* | *"The TCN has solved vehicle speed estimation."* |
| **Vibration spectrum contains speed information** | Audit of 64 trips (>450k windows): $r = -0.032$; dominant peak locked at 2.9 Hz on motorway; MAE changed $-0.01\text{ m/s}$ | 🔴 **Rejected** | *"No reliable, generalizable speed-specific vibration signal was demonstrated under the tested smartphone dataset/sampling conditions."* | *"Vibration contains zero speed information in the universe."* / *"Vibration is a reliable speedometer."* |
| **Steady-state highway speed is observable from instantaneous IMU** | Classifier ROC-AUC = 0.625; $a_{\text{long}} \approx 0, \omega_{\text{yaw}} \approx 0$ on smooth highway cruise | 🔴 **Rejected** | *"Within this dataset and tested features, steady-cruise speed classes show weak statistical separability from the sensor noise floor."* | *"It is mathematically impossible to navigate on a highway."* |
| **Universal <10% SIH benchmark achieved at 60 seconds** | Held-out 60s benchmark: only **3 out of 13 usable trips passed <10% (23.1% pass rate)** | 🔴 **Unsupported** | *"While specific mountain and highway segments achieve <10% drift, cross-trip pass rate at 60 seconds remains capped at 23.1%."* | *"Our system achieves the SIH <10% requirement."* |
| **Non-Holonomic Constraints eliminate lateral skidding** | Cross-track position error reduced by **64.8%** across blackout windows when projected via causal PCA | 🟢 **Demonstrated** | *"Gated Non-Holonomic Constraints substantially bound lateral and vertical position divergence."* | *"NHC prevents position drift along the road."* |
| **Pedestrian walking test validates vehicle navigation** | Peak error window generated $+26.08\sigma$ gyro yaw rate, triggering 71.66 m/s speed predictions | 🔴 **Rejected** | *"Pedestrian walking tests are hardware and timing stress tests only; they violate vehicle non-holonomic kinematics."* | *"The algorithm was tested and validated on campus."* |
| **15-State ESKF bounds attitude drift during GNSS outages** | Attitude error held to $<1.8^\circ$ over 30 seconds; short-outage drift reduced by $4.2\times$ | 🟢 **Demonstrated** | *"Pre-outage bias calibration in the 15-state ESKF successfully stabilizes vehicle attitude and bounds short-term drift."* | *"The Kalman filter prevents long-term dead-reckoning drift."* |
| **Long outages (>30s) require external scale anchoring** | Pure kinematics diverges quadratically; TCN plateaus at training mean; pass rate drops to 23% | 🟡 **Supported Hypothesis** | *"Extended dead reckoning on commercial smartphones requires external scale constraints, such as road curvature geometry or dual-frequency Doppler."* | *"Smartphone dead reckoning is hopeless without wheel sensors."* |
