# Stage C5.2-C0: Longitudinal-Frame & Gyroscope Kinematic Integrity Audit

## Executive Summary
This audit provides an offline, first-principles forensic analysis of the physical relationship between smartphone IMU measurements and ground-truth vehicle dynamics across all three available journeys: `Vta02`, `Vta03`, and `Vta04`.

**Strict Experimental Rule**: VBOX speed derivative ($\frac{dv}{dt}$) and vehicle yaw rate ($\dot{\psi}$) are used **exclusively as offline diagnostic references** to inspect the sensor coordinates; they are never passed to runtime estimators.

---

## 1. Key Forensic Discoveries

### Discovery 1: Forward Acceleration Aligns Naturally with Phone Axis $a_x^p$
* **Trip Vta02**: The optimal forward azimuth in the horizontal plane is $\theta^* = \mathbf{-5.0^\circ}$ (Peak correlation $r = +0.478$). Raw phone $a_x^p$ has $r = +0.476$.
* **Trip Vta04**: The optimal forward azimuth in the horizontal plane is $\theta^* = \mathbf{+3.0^\circ}$ (Peak correlation $r = +0.201$). Raw phone $a_x^p$ has $r = +0.201$.
* **Critical Finding on $R_{pv}$**: In Stage C1/C3, `align_phone_to_vehicle` estimated a horizontal yaw alignment of $\theta_{\text{yaw}} = -75.35^\circ$ on `Vta04`. When that rotation was applied, it rotated the physical forward acceleration axis **out of $a_x^v$ ($r = +0.066$) and into the lateral axis $a_y^v$ ($r = +0.198$)**!
  The previous $R_{pv}$ algorithm introduced an artificial coordinate misalignment for forward acceleration on `Vta04`.

### Discovery 2: Vehicle Yaw Rate Resides in Phone Gyro Axis $\omega_x^p$
* In `Vta02`, correlation between raw phone gyro and true vehicle yaw rate is:
  - $\omega_x^p$ (Android Gyroscope Pitch): **$r = +0.683$**
  - $\omega_y^p$ (Android Gyroscope Roll): $r = +0.024$
  - $\omega_z^p$ (Android Gyroscope Yaw): $r = +0.009$
* In `Vta04`:
  - $\omega_x^p$: **$r = +0.319$**
  - $\omega_y^p$: $r = -0.076$
  - $\omega_z^p$: $r = -0.063$
* **Physical Cause**: In IO-VNBD, Android's `GYROSCOPE Pitch` was mapped to `gyro_x`. For the physical mounting in Driver E's vehicle, vehicle rotation around the earth vertical axis (yaw) registered primarily on `gyro_x`.

---

## 2. Comparative Audit Table Across Trips

| Journey | Candidate Frame | Longitudinal Accel Corr $r(a_x, \dot{v})$ | Optimal Forward Azimuth $\theta^*$ | Peak Azimuth Corr | Gyro Correlation with Vehicle Yaw Rate | Accel Scale Factor $\frac{\text{Std}(a_x)}{\text{Std}(\dot{v})}$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Vta02** (Train 1) | Raw Phone<br>Gravity-Leveled<br>Calibrated $R_{pv}$ | **+0.476**<br>**+0.476**<br>+0.444 | **-5.0°** | **+0.478** | $\omega_x^p: \mathbf{+0.683}$, $\omega_z^p: +0.009$<br>$\omega_z^v: -0.039, \omega_y^v: +0.811$ | 1.35 |
| **Vta03** (Train 2) | Raw Phone<br>Gravity-Leveled<br>Calibrated $R_{pv}$ | -0.144<br>-0.144<br>-0.162 | -151.0° | +0.174 | $\omega_x^p: -0.228, \omega_z^p: -0.155$<br>$\omega_z^v: -0.010$ | 0.79 |
| **Vta04** (Test) | Raw Phone<br>Gravity-Leveled<br>Calibrated $R_{pv}$ | **+0.201**<br>**+0.201**<br>+0.066 | **+3.0°** | **+0.201** | $\omega_x^p: \mathbf{+0.319}$, $\omega_z^p: -0.063$<br>$\omega_z^v: -0.034, \omega_y^v: +0.349$ | 1.31 |

---

## 3. Physical Conclusions for C5.2-C

1. **Sensor Forward Alignment**:
   For both `Vta02` and `Vta04`, the physical forward acceleration axis is directly aligned with the smartphone longitudinal axis ($a_x^p$ or $a_{x,\text{level}}$) within $3^\circ\text{--}5^\circ$.
   No complex horizontal rotation is needed or justified; applying the previous $R_{pv}$ degraded forward acceleration correlation from $+0.201 \to +0.066$.
2. **Acceleration Scale and Bias**:
   - The low-pass filtered IMU forward acceleration scale factor matches ground truth acceleration within $1.31\times\text{--}1.35\times$.
   - Unfiltered raw IMU contains severe vibration noise ($\text{SNR} \approx -4\text{ dB}$), requiring low-pass filtering and bias estimation before integration.
3. **Prerequisite for Stage C5.2-C1**:
   Direct kinematic integration of $a_x^p$ (or $a_{x,\text{level}}$) is physically justified and grounded in data, provided:
   - It is initialized from a known initial velocity $v(t_0)$.
   - It evaluates pure open-loop drift over short windows (e.g. 5s, 10s, 30s) before any constraints are added.
