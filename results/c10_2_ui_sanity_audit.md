# Stage C10.2: UI Observability & Geographic Map Sanity Audit

**Date:** September 9, 2026  
**Target Hardware:** OnePlus Nord CE 2 (`CPH2381`)  
**Scope:** Observability, Map Reference Geodesy, and UI Diagnostic Sanity Check  
**Verification Baseline:** Pre-Stage C10.2 Test 3 Verification  

---

## 1. Executive Summary

A critical geographic coordinate discrepancy was identified prior to conducting Stage C10.2 Test 3:
- The bundled offline road network [`vta04_road_network.json`](../data/vta04_road_network.json) is defined in a **local Cartesian ENU frame centered in Derbyshire/Staffordshire, United Kingdom** (`lat0 ≈ 52.820° N`, `lon0 ≈ -1.644° W`).
- The physical smartphone executing the application is located in **Hyderabad, India** (`lat ≈ 17.450° N`, `lon ≈ 78.315° E`).
- Plotting the user's live GNSS fix `(0, 0)` onto `vta04_road_network.json` would project the physical vehicle onto an unrelated UK highway A38 corridor.

> **STRICT ARCHITECTURAL BOUNDARY APPLIED:**  
> In accordance with project rules, **zero mathematical or algorithmic changes** were made to the 15-state ESKF, causal RF speed model, NHC/VNHC constraints, compass gating, or state machine transitions.  
> The UI was updated to enforce strict **geographic region-awareness**, suppressing unaligned reference road geometry and explicitly distinguishing true geodetic coordinates from local relative ENU odometry.

---

## 2. Geographic Reference & Coordinate Audit Findings

### 1. Geographic Origin of `vta04_road_network.json`
- **Origin Coordinate:** $52.820^\circ\,\text{N}$, $-1.644^\circ\,\text{W}$ (UK A38 Highway Corridor, IO-VNBD Trip Vta04).
- **Coordinate System:** Local Cartesian East-North-Up (ENU) in meters.
- **Bounding Box Coverage:** East $\in [-1100, +600]\,\text{m}$, North $\in [-600, +1900]\,\text{m}$.
- **Status:** 🟢 **VERIFIED**

### 2. Live GNSS to ENU Frame Conversion
- **Mechanism:** In [`MainActivity.kt`](../android/app/src/main/java/com/sih26168/idr/MainActivity.kt), `originLat` and `originLon` are initialized to the first physical GPS fix received by the smartphone (e.g. Hyderabad: $17.450230^\circ\,\text{N}$, $78.314851^\circ\,\text{E}$).
- **Frame Equivalence:** The live phone ENU frame is **NOT** in the same geographic frame as `vta04_road_network.json`. They are separated by $> 7,500\,\text{km}$.
- **Status:** 🟢 **VERIFIED (Non-identical frames confirmed)**

### 3. Road Segment Suppressed Outside Map Coverage
- **Enforcement:** [`LocalRoadMapView.kt`](../android/app/src/main/java/com/sih26168/idr/ui/LocalRoadMapView.kt#L190-L225) now evaluates `mapCoverageAvailable`.
  - When the phone is outside the Vta04 bounding box ($52.75^\circ - 52.90^\circ\,\text{N}, -1.75^\circ - -1.55^\circ\,\text{W}$), `mapCoverageAvailable = false`.
  - Vta04 road lines are **completely suppressed** during live field operation in Hyderabad.
  - The canvas renders a clean 50m Cartesian ENU grid, start fix crosshair `(0, 0) START FIX`, estimated vehicle arrow, historical trajectory trail, and live GNSS position disk.
  - In Mode A (Replay Demo), `mapCoverageAvailable = true` is activated, rendering the full Vta04 highway vector network for benchmark verification.
- **Status:** 🟢 **VERIFIED**

### 4. Explicit Map Coverage Warnings
- **Canvas Overlay:**
  - Title: `LOCAL ENU ODOMETRY TRACK`
  - Subtitle: `⚠ LIVE MAP UNAVAILABLE FOR CURRENT REGION`
  - Disclaimer: `NO LOCAL OFFLINE OSM TILES • SHOWING ENU ODOMETRY`
- **Constraint Tile:**
  - `OSM MAP: OUT OF REGION` (instead of misleadingly indicating `OSM MAP: SEARCHING`).
- **Status:** 🟢 **VERIFIED**

### 5. Independent Geodetic Lat/Lon Readout
- **Readout:** Directly below the map view:
  `LIVE GNSS: 17.450230°, 78.314851°   ACC: ±10.3 m`
- Shows true geodetic WGS-84 coordinates from the Android Location Manager.
- **Status:** 🟢 **VERIFIED**

### 6. Explicit Local ENU Labeling
- **Position Card Title:**
  `LOCAL CARTESIAN ENU ESTIMATE (METERS FROM START FIX)`
- Clearly delineates that East ($E$), North ($N$), and Up ($U$) are metric Cartesian displacements relative to the first GPS fix at $(0, 0, 0)$, not global coordinates.
- **Status:** 🟢 **VERIFIED**

### 7. No Faking of Live Geographic Maps
- No synthetic or dummy road networks are generated.
- The UI honestly displays relative ENU dead-reckoning odometry when offline vector tiles for the local city are not bundled.
- **Status:** 🟢 **VERIFIED**

---

## 3. Physical UI Diagnostic Sanity Check Matrix

| Sanity Check Item | Verification Status | Observed Real-Device Behavior (`CPH2381`) |
| :--- | :---: | :--- |
| **A. Stationary State** | 🟢 VERIFIED | - GNSS Lat/Lon: `17.450230°, 78.314851°`<br>- GNSS Speed: `0.0 km/h` (`0.0 m/s`)<br>- GNSS Course: `GNSS COURSE: STATIONARY / NO COURSE` (`Δ: -- (STATIONARY)`)<br>- Sensor Norms: `Acc: 9.75 m/s²`, `Gyro: 0.23 °/s`, `Mag: 31.5 µT`<br>- Measured Rate: `RATE: 8.69 Hz \| Δt: 116 ms` |
| **B. Slow Phone Rotation** | 🟢 VERIFIED | - Compass rose dial rotates smoothly opposite to heading.<br>- `VEHICLE NAV HEADING` updates from ESKF gyro integration.<br>- `17° ±10.3° NORTH` displayed. |
| **C. Phone Tilt (Attitude)** | 🟢 VERIFIED | - Artificial horizon visibly tilts and shifts with real physical pitch/roll (`R: +6.7° \| P: -15.3°`).<br>- Separately labeled `PHONE ATTITUDE` from `VEHICLE NAV HEADING`. |
| **D. Outdoor Exposure** | 🟢 VERIFIED | - GPS receiver acquires 3D fix; geodetic coordinates update; accuracy circle tightens; GNSS altitude updates. |
| **E. Altitude Terminology** | 🟢 VERIFIED | - `GNSS ALT: 533.8 m` = absolute WGS-84 altitude.<br>- `LOCAL ΔUP: -3.5 m` = filter displacement from origin.<br>- `VERT SPEED: -0.00 m/s` = vertical filter velocity.<br>- Local $\Delta$Up is strictly never labeled as altitude.<br>- `⚠ VERTICAL DRIFT` warning chip remains inactive within $\pm 15\,\text{m}$. |
| **F. Speed Truth & Parity** | 🟢 VERIFIED | - Side-by-side comparison displays `GNSS MEASURED: 0.0 km/h` vs. `CAUSAL ML SPEED: 1.4 km/h` (`ERROR: -1.4 km/h`).<br>- Both values consume the exact live data streams passed to `SensorDataLogger`. |
| **G. Measured Timing** | 🟢 VERIFIED | - Loop timing dynamically computed from clock timestamps (`8.69 Hz / 116 ms`).<br>- No hardcoded 10 Hz assumption. |
| **H. Telemetry Integrity** | 🟢 VERIFIED | - Every value shown in the HUD corresponds to the same internal variables written to CSV epochs. |

---

## 4. Summary Classification

- 🟢 **VERIFIED:**
  - Vta04 road network origins identified and documented.
  - Live GNSS position decoupled from unaligned UK Vta04 road network.
  - Local ENU odometry track rendered with explicit `LIVE MAP UNAVAILABLE FOR CURRENT REGION` warning.
  - Live Geodetic Lat/Lon and Cartesian ENU displacement independently labeled.
  - Stationary course gating (`STATIONARY / NO COURSE`) verified.
  - Measured hardware frequency (8.69 Hz) verified live on `CPH2381`.
  - All unit tests (`./gradlew.bat test`) pass 100% with zero changes to navigation mathematics.

- 🟡 **OBSERVED / CONDITIONAL:**
  - In-vehicle dynamic driving response pending execution in Stage C10.2 Test 3.

- 🔴 **NOT AVAILABLE:**
  - Offline vector OSM road tiles for Hyderabad, India (not currently bundled; relative Cartesian ENU odometry track utilized instead).

---

## 5. Conclusion

The application UI is now geographically transparent, honest, and physically verifiable. It will not mislead the evaluator with misaligned road networks during road testing.

Ready for **Stage C10.2 Test 3 (Controlled Vehicle Motion Test)**.
