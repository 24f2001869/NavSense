# Architectural Decision Record: Open Research Questions

This document articulates the genuine unresolved technical challenges and open research questions in smartphone vehicular dead reckoning.

---

## 1. The Steady-State Cruise Observability Deficit

### The Problem
During unaccelerated cruising on a straight, smooth highway:
$$a_{\text{long}} \approx 0, \quad \omega_{\text{yaw}} \approx 0$$
The instantaneous inertial signals measured by a smartphone IMU sitting on a dashboard mount are statistically indistinguishable between **85 km/h** and **115 km/h** ($ROC\text{-}AUC = 0.625$).

### Open Research Questions
1. Can external digital road network attributes (such as OpenStreetMap `maxspeed` tags or road functional classifications) be used to set the initial prior speed distribution during long outages without causing false assumptions during traffic jams?
2. Does air turbulence or high-speed acoustic aerodynamic noise captured by the smartphone microphone correlate with vehicle airspeed, and can it be safely decoupled from in-cabin passenger speech and ventilation?

---

## 2. Robust Along-Track Road Geometry Coupling

### The Problem
Classical map matching snaps position to the nearest road centerline, which only constrains **cross-track error** ($d_\perp$). It provides zero information about **along-track position** ($s$) on a straight road. Furthermore, in dense suburban networks, prior covariance inflation causes statistical gates to collapse towards zero, leading to false parallel street latching.

### Open Research Questions
1. **Curvature-Speed Coupling**: On curved road segments ($R < 500\text{ m}$), can the gyroscope yaw rate directly observe vehicle speed ($v = \omega / \kappa$) via known digital road curvature $\kappa(s)$ without numerical instability?
2. **Topological Apex Anchoring**: Can discrete geometric features (such as 90-degree turns, roundabouts, or highway exit ramps) be matched topologically to reset along-track integration drift?
3. How can multi-hypothesis tracking (MHT) prevent parallel street attraction in dense suburban grids when prior uncertainty $\sigma_p > 30\text{ m}$?

---

## 3. Pre-Outage GNSS Doppler Bias Anchoring

### The Problem
Consumer smartphone GPS fixes typically provide 1 Hz position and coarse speed with $0.2\text{--}0.8\text{ m/s}$ jitter. A residual accelerometer bias of just $0.05\text{ m/s}^2$ causes $90\text{ meters}$ of drift in 60 seconds.

### Open Research Questions
1. Can modern dual-frequency (L1/L5) Android raw GNSS pseudorange and carrier-phase Doppler observables (`GnssMeasurement`) constrain the accelerometer bias to $<0.01\text{ m/s}^2$ immediately prior to entering a tunnel?
2. How long does a thermal bias calibration remain valid before phone processor heating causes the MEMS accelerometer bias to shift?

---

## 4. Physical In-Vehicle Road Validation

### The Problem
All offline research in this repository has been rigorously validated on synchronized benchmark datasets (IO-VNBD). However, the physical Android prototype has only been subjected to sensor logging, UI stress tests, and campus pedestrian movement.

### Open Research Questions
1. Does the production Android engine maintain its 10.0 Hz execution clock without frame drops during a live 30-minute vehicle trip?
2. How does the causal PCA mounting alignment perform across different real-world phone mounts (air vent clip vs windshield suction cup vs wireless charging pad)?
