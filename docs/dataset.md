# IO-VNBD Dataset Technical Reference

## 1. Dataset Origin & Metadata
* **Dataset**: IO-VNBD (Indoor-Outdoor Vehicle Navigation Benchmark Dataset)
* **Authors / Origin**: Multimodal vehicular dataset collected across Spain and Central Europe under varied driving conditions.
* **Recording Vehicles**: Instrumented passenger vehicles equipped with high-precision reference navigation systems, CAN bus loggers, transmission wheel-pulse counters, and commercial Android smartphones.
* **Total Vehicle Trips**: 64 standardized trips across 4 road domains.
* **Benchmark Split in this Project**:
  - **Training Set**: 39 whole trips (over 450,000 timesteps at 10 Hz)
  - **Held-Out Test Set**: 19 completely untouched trips (zero temporal or driver leakage)

---

## 2. Sensor Channel Mapping & Coordinates

### 2.1 Coordinate Conventions
* **Navigation Frame ($n$)**: Local East-North-Up (ENU) Cartesian frame tangent to WGS84 ellipsoid.
* **Phone Body Frame ($b$)**: Standard Android sensor frame ($x$ right, $y$ top, $z$ screen outward).
* **Vehicle Frame ($v$)**: Standard SAE automotive frame ($x$ forward, $y$ right, $z$ downward) or ISO ($x$ forward, $y$ left, $z$ upward). In this repository, ISO is adopted: $x_{\text{forward}}, y_{\text{lateral}}, z_{\text{vertical}}$.

### 2.2 Channel Dictionary
| Column Name | Physical Quantity | Raw Range | Units |
|:---|:---|:---:|:---:|
| `acc_x, acc_y, acc_z` | Total specific force (with gravity) | $[-20, +20]$ | $\text{m/s}^2$ |
| `lin_acc_x, y, z` | Linear acceleration (gravity removed) | $[-10, +10]$ | $\text{m/s}^2$ |
| `gyro_yaw, pitch, roll`| Calibrated angular velocity | $[-2.5, +2.5]$ | $\text{rad/s}$ |
| `grav_x, grav_y, grav_z`| Gravity vector direction | $[-9.81, +9.81]$| $\text{m/s}^2$ |
| `can_speed_mps` | Ground reference speed | $[0, 36.0]$ | $\text{m/s}$ |
| `wheel_speed_mps` | Uncalibrated wheel tick speed | $[0, 38.0]$ | $\text{m/s}$ |
| `gnss_lat, gnss_lon` | Consumer smartphone GPS fix | Coordinates | Degrees |
