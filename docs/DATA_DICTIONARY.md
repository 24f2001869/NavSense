# SIH26168 — Data Dictionary (IO-VNBD Benchmark Verified)

This document tracks exact field definitions, units, coordinate axes, and verified nominal sampling frequencies from the IO-VNBD benchmark dataset.

---

## 1. Smartphone Sensory Stream (`S-*.csv`)
- **Source**: Android Smartphone (Huawei P20 Pro / Motorola Moto G7 Power / Blackberry Priv) running **AndroSensor**.
- **Nominal Sampling Frequency**: **10.00 Hz** ($\Delta t = 100\text{ ms}$, range: 90–108 ms).
- **Update Frequency of GNSS**: 1 Hz (updated once every 10 samples; held constant between updates).
- **Total Columns**: 24

| No | Column Name | Physical Unit | Frame / Coordinate System | Nominal Freq | Description & SIH Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `GPS LATITUDE (degrees)` | Decimal degrees | WGS-84 Geodetic | 1 Hz | Ground truth reference & EKF measurement update |
| 2 | `GPS LONGITUDE (degrees)` | Decimal degrees | WGS-84 Geodetic | 1 Hz | Ground truth reference & EKF measurement update |
| 3 | `GPS ALTITUDE (m)` | Meters ($m$) | WGS-84 Ellipsoid | 1 Hz | Altitude above sea level / ellipsoid |
| 4 | `GPS SPEED (Kmh)` | $km/h$ | Navigation Frame ($v_h$) | 1 Hz | Smartphone GNSS ground speed (Doppler-derived) |
| 5 | `GPS ACCURACY (m)` | Meters ($m$) | Horizontal 1-$\sigma$ | 1 Hz | Uncertainty radius; sets EKF measurement noise $R_k$ |
| 6 | `GPS ORIENTATION (°)` | Degrees ($0^\circ - 360^\circ$) | Geographic North | 1 Hz | Course over ground heading |
| 7 | `GPS SATELLITES IN RANGE` | Ratio String (e.g. `20 / 23`) | N/A | 1 Hz | Satellites used vs tracked; GNSS outage indicator |
| 8 | `TIME SINCE START (ms)` | Milliseconds ($ms$) | System monotonic clock | 10 Hz | Primary clock for smartphone strapdown integration |
| 9 | `DATE (YYYY-MO-DD HH-MI-SS_SSS)` | Timestamp string | UTC Clock | 10 Hz | Wall-clock time for multi-device cross-referencing |
| 10 | `ACCELEROMETER X (m/s²)` | $m/s^2$ | Phone Body Frame ($X_b$) | 10 Hz | Lateral acceleration in screen plane |
| 11 | `ACCELEROMETER Y (m/s²)` | $m/s^2$ | Phone Body Frame ($Y_b$) | 10 Hz | Longitudinal acceleration in screen plane |
| 12 | `ACCELEROMETER Z (m/s²)` | $m/s^2$ | Phone Body Frame ($Z_b$) | 10 Hz | Normal acceleration perpendicular to screen |
| 13 | `GRAVITY X (m/s²)` | $m/s^2$ | Phone Body Frame ($X_b$) | 10 Hz | Filtered gravity vector component |
| 14 | `GRAVITY Y (m/s²)` | $m/s^2$ | Phone Body Frame ($Y_b$) | 10 Hz | Filtered gravity vector component |
| 15 | `GRAVITY Z (m/s²)` | $m/s^2$ | Phone Body Frame ($Z_b$) | 10 Hz | Filtered gravity vector component (baseline calibration) |
| 16 | `GYROSCOPE Yaw (rad/s)` | $rad/s$ | Phone Body Frame ($Z_b$) | 10 Hz | Angular rate around device normal axis |
| 17 | `GYROSCOPE Pitch (rad/s)` | $rad/s$ | Phone Body Frame ($X_b$) | 10 Hz | Angular rate around device horizontal axis |
| 18 | `GYROSCOPE Roll (rad/s)` | $rad/s$ | Phone Body Frame ($Y_b$) | 10 Hz | Angular rate around device vertical axis |
| 19 | `MAGNETIC FIELD X (μT)` | Microteslas ($\mu T$) | Phone Body Frame ($X_b$) | 10 Hz | Geomagnetic field (contains soft/hard iron vehicle noise) |
| 20 | `MAGNETIC FIELD Y (μT)` | Microteslas ($\mu T$) | Phone Body Frame ($Y_b$) | 10 Hz | Geomagnetic field |
| 21 | `MAGNETIC FIELD Z (μT)` | Microteslas ($\mu T$) | Phone Body Frame ($Z_b$) | 10 Hz | Geomagnetic field |
| 22 | `ORIENTATION (Yaw) (°)` | Degrees ($0^\circ - 360^\circ$) | Navigation Frame ($n$-frame) | 10 Hz | Android fused azimuth/heading |
| 23 | `ORIENTATION (Pitch) (°)` | Degrees ($-90^\circ$ to $+90^\circ$) | Navigation Frame ($n$-frame) | 10 Hz | Android fused pitch angle |
| 24 | `ORIENTATION (Roll ) (°)` | Degrees ($-180^\circ$ to $+180^\circ$) | Navigation Frame ($n$-frame) | 10 Hz | Android fused roll angle |

---

## 2. Vehicle Reference / Odometry Stream (`V-*.csv`)
- **Source**: Racelogic VBOX Video HD2 + Ford Fiesta Titanium ECU / CAN Bus logger.
- **Nominal Sampling Frequency**: **10.00 Hz** ($\Delta t = 0.1\text{ s}$).
- **Total Columns**: 29

| No | Column Name | Physical Unit | Description & SIH Relevance |
| :--- | :--- | :--- | :--- |
| 1 | `No of GPS Satellites Available` | Count | Reference VBOX satellite tracking status |
| 2 | `Time Since Start of Day (seconds)` | Seconds ($s$) | Reference high-precision epoch clock (0.1s step) |
| 3 | `Latitude (degrees)` | Decimal degrees | Ground truth vehicle geodetic latitude |
| 4 | `Longitude (degrees)` | Decimal degrees | Ground truth vehicle geodetic longitude |
| 5 | `Velocity (km/hr)` | $km/h$ | True reference vehicle ground speed |
| 6 | `Heading (degrees)` | Degrees | True vehicle heading from VBOX GPS antenna |
| 7 | `Height (km)` | Kilometers | Reference altitude |
| 8 | `Vertical velocity (km/hr)` | $km/h$ | Vertical motion velocity (useful for grade/slope) |
| 9 | `Sample period (seconds)` | Seconds | Constant $0.1\text{ s}$ |
| 10 | `Steering Angle (degrees)` | Degrees | Steering wheel displacement |
| 11 | `Wheel Speed Front Left (rad/sec)` | $rad/s$ | High-rate wheel angular speed |
| 12 | `Wheel Speed Front Right (rad/sec)` | $rad/s$ | High-rate wheel angular speed |
| 13 | `Wheel Speed Rear Left (rad/sec)` | $rad/s$ | Non-driven wheel speed (ideal odometry baseline) |
| 14 | `Wheel Speed Rear Right (rad/sec)` | $rad/s$ | Non-driven wheel speed (ideal odometry baseline) |
| 15 | `Yaw Rate (deg/sec)` | $^\circ/s$ | Chassis yaw rate from vehicle stability sensors |
| 16 | `Indicated Vehicle Speed (km/hr)` | $km/h$ | Dashboard/CAN speedometer reading |
| 17 | `Indicated Longitudinal Acceleration (g)` | $g$ ($9.80665\text{ m/s}^2$) | True vehicle forward acceleration |
| 18 | `Indicated Lateral Acceleration (g)` | $g$ ($9.80665\text{ m/s}^2$) | True vehicle cornering lateral acceleration |
| 19 | `Handbrake (0 or 1)` | Binary | Handbrake state |
| 20 | `Gear Requested` | Integer (1-5) | Transmission requested gear |
| 21 | `Gear` | Integer (1-5) | Transmission active gear |
| 22 | `Engine Speed (rev/min)` | $RPM$ | Crankshaft rotational speed |
| 23 | `Coolant Temperature (degrees)` | $^\circ C$ | Engine coolant temp |
| 24 | `Clutch Position (0 or 1)` | Binary | Clutch pedal state |
| 25 | `Brake Pressure (psi)` | $PSI$ | Hydraulic braking pressure |
| 26 | `Brake Position (0 or 1)` | Binary | Brake pedal switch (critical for zero-velocity updates / ZUPT) |
| 27 | `Battery Voltage (volts)` | $V$ | Electrical bus voltage |
| 28 | `Air Temperature (degrees)` | $^\circ C$ | Ambient temperature |
| 29 | `Accelerator Pedal Position (0 or 1)`| Float / % | Throttle input |

---

## 3. Key Observations & Design Implications for SIH26168
1. **Synchronized Frequency**: Both smartphone and ECU streams are strictly sampled at **10.00 Hz**, perfectly fulfilling the SIH problem statement's target for 10 Hz real-time output.
2. **True Ground Truth Velocity Available**: We have true `Velocity (km/hr)` and `Wheel Speed Rear Left/Right (rad/sec)` to supervise and evaluate the AI forward-velocity estimator.
3. **Brake Position for ZUPT**: `Brake Position` and zero wheel speeds in the reference data allow automated labeling of Zero Velocity Updates (ZUPT) to prevent drift during stops.
4. **Encoding & Headers**: Smartphone files use `latin1` encoding due to special characters (`m/s²`, `°`, `μT`) and have leading whitespace in column headers that must always be trimmed (`.strip()`).
