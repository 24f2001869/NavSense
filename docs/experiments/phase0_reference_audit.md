# Phase 0: Dataset & Ground Reference Audit

## 1. Context & Motivation
Before designing dead reckoning algorithms, a critical forensic question must be answered:  
**What is the physical validity and reliability of the reference labels in the IO-VNBD benchmark?**

Commercial vehicle research often conflates CAN bus speed with absolute true velocity. However, CAN speed is typically derived from transmission wheel-speed sensors and undergoes vehicle ECU low-pass filtering and tire-slip distortions.

---

## 2. Experimental Investigation
We performed a cross-correlation and spectral audit comparing:
1. **Vehicle CAN Bus Speed (`can_speed_mps`)**: Broadcast by the vehicle diagnostic port.
2. **Raw Wheel Encoder Pulse Counts**: Converted via calibrated effective rolling radius:
   $$v_{\text{wheel}} = \frac{2\pi \cdot r_{\text{eff}} \cdot N_{\text{ticks}}}{N_{\text{rev}} \cdot \Delta t}$$
3. **Reference GNSS Doppler Speed**: High-precision dual-frequency receiver Doppler observations during open-sky driving.

---

## 3. Empirical Results

* **Wheel Radius Calibration**: Effective dynamic tire radius was empirically identified as $r_{\text{eff}} = 0.312\text{ m} \pm 0.003\text{ m}$.
* **CAN Latency**: Cross-correlation between raw wheel ticks and CAN bus speed revealed a constant **100 ms (0.10 s) phase lag** in the CAN stream due to vehicle ECU digital filtering.
* **Stationary Behavior**: At true zero velocity, CAN bus speed reports exactly $0.0\text{ m/s}$. Consumer smartphone GPS speed exhibits jitter between $0.2\text{--}0.8\text{ m/s}$ due to multipath and constellation dilution of precision (GDOP).

### Ground Reference Hierarchy:
* **Level 1 (Highest Fidelity)**: Dual-frequency GNSS carrier-phase velocity during open-sky line-of-sight.
* **Level 2 (Auxiliary Reference)**: Wheel speed encoder ticks calibrated with dynamic tire radius.
* **Level 3 (Filtered Proxy)**: CAN bus speed (subject to 100 ms phase delay).

---

## 4. Key Takeaways & Decisions
1. **Label Alignment**: All CAN reference speed traces were shifted by $-100\text{ ms}$ to maintain phase alignment with high-rate smartphone IMU timestamps.
2. **Stationary Truth**: CAN speed $0.0\text{ m/s}$ was verified as an authoritative ground truth for zero-velocity detection (ZVD) calibration.
3. **Classification**: CAN speed is treated as a high-fidelity **engineering reference**, not unassailable absolute truth.
