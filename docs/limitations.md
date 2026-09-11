# Known Technical Limitations & Failure Modes

To ensure engineering honesty and transparency, this document cataloging all known failure modes and boundary conditions where the current system does not perform reliably.

---

## 1. The Steady-State Cruise Plateau (Straight Motorways)
* **Failure Mode**: When driving on long, straight highways at speeds above 95 km/h, estimated speed plateaus at 85–88 km/h.
* **Root Cause**: During unaccelerated straight driving ($a_{\text{long}} \approx 0, \omega_{\text{yaw}} \approx 0$), the instantaneous IMU signal is identical to sensor noise floor ($ROC\text{-}AUC = 0.625$). Pure kinematics drifts due to bias; the memoryless TCN falls back to the training mean.
* **Impact**: On motorway trip `V-Vfa02`, 60-second drift is **11.69%**, narrowly missing the 10% target.

---

## 2. Low-Speed Urban Crawling ($0.5–2.0\text{ m/s}$)
* **Failure Mode**: In stop-and-go congestion where a vehicle creeps forward at walking pace, position drift percentage can be high ($25\text{--}45\%$).
* **Root Cause**:
  1. Micro-movements ($<2.0\text{ m/s}$) produce acceleration signals that are close to the stationary threshold, causing intermittent ZVD false-triggering.
  2. Because the total distance traveled during a 60-second jam is small (e.g. 50 meters), even a 15-meter integration error results in a **30% drift percentage**.

---

## 3. Road Grade & Slope Ingestion (Gravity Leakage)
* **Failure Mode**: Driving up or down steep hills ($>6\%$ grade) can induce an along-track velocity bias.
* **Root Cause**: Smartphone MEMS accelerometers measure the sum of linear vehicle acceleration and the projection of gravity. If the vehicle pitches up on a hill:
  $$a_{\text{meas}} = a_{\text{vehicle}} + g \sin(\theta_{\text{grade}})$$
  Without barometer or digital terrain elevation aiding, road grade leaks directly into forward velocity estimation.

---

## 4. In-Vehicle Thermal Drift
* **Failure Mode**: Smartphone processor heating under intensive screen and ONNX execution can cause the MEMS accelerometer bias to shift during prolonged runs.
* **Mitigation Needed**: Online thermal compensation curves or periodic re-anchoring when GNSS is available.

---

## 5. Lack of Synchronized Real-Car Validation
* **Status**: All quantitative validation in this repository was performed on pristine offline benchmark datasets (IO-VNBD) and smartphone campus stress recordings.
* **Limitation**: Real in-vehicle driving with live GNSS denial testing has not yet been conducted.
