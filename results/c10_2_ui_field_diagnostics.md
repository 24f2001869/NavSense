# C10.2 UI Field Diagnostics & Transparency Audit Report

**Date:** September 9, 2026  
**Target Hardware:** OnePlus Nord CE 2 (`CPH2381` / `68afd407`)  
**Scope:** Observability, UI, Sensor Diagnostics, and Real-Device Transparency  
**Verification Baseline:** Stage C10.2 Pre-Test 3 Verification  

---

## 1. Executive Summary & Objective

Prior to conducting **Stage C10.2 Test 3 (Controlled Vehicle Motion Test)**, the Android application was upgraded from an opaque navigation HUD into a **high-transparency laboratory-grade field diagnostic dashboard**.

> **STRICT ARCHITECTURAL BOUNDARY:**  
> This task is **strictly an observability, UI, and diagnostic enhancement**.  
> **ZERO algorithmic or mathematical modifications** were made to:
> - The frozen 15-state Error-State Kalman Filter (ESKF) equations
> - Causal Random Forest speed model or its feature extraction
> - Non-Holonomic Constraints (NHC) and Vertical NHC (VNHC) mathematics
> - Magnetometer gating thresholds or map-matching search radii
> - Phone-to-vehicle rotation matrix ($R_{vp}$) or state-machine transition logic.

The goal is to allow the driver/evaluator during field tests to look at the smartphone in its vehicle mount and independently, transparently cross-examine the phone's physical motion against what the navigation engine estimates.

---

## 2. Feature & Diagnostic Verification Matrix

| Diagnostic Feature | Implementation Layer | Status | Verification Notes |
| :--- | :--- | :---: | :--- |
| **Local OSM Road Map (Vector ENU)** | [`LocalRoadMapView.kt`](../android/app/src/main/java/com/sih26168/idr/ui/LocalRoadMapView.kt) | 🟢 Verified | 2D vector canvas rendering local highway segments from `vta04_road_network.json`. Displays Layer A (cyan GNSS point + accuracy disk) and Layer B (amber ESKF vehicle marker + heading arrow + trail buffer). Does not fake online tiles. |
| **Side-by-Side Speed Comparison** | [`MainActivity.kt`](../android/app/src/main/java/com/sih26168/idr/MainActivity.kt#L597-L612) | 🟢 Verified | Prominently contrasts `GNSS MEASURED` (km/h & m/s) against `CAUSAL ML SPEED` (km/h & m/s). Computes dynamic `SPEED ERROR (GNSS - ML)`. When GNSS is unavailable, cleanly displays `--`. |
| **Heading & Compass with Course $\Delta$** | [`CompassRoseView.kt`](../android/app/src/main/java/com/sih26168/idr/ui/CompassRoseView.kt) | 🟢 Verified | Rotating dial with 8 cardinal/intercardinal markers (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`), $\pm 1\sigma$ uncertainty wedge, numeric readout, and explicit source tag (`SOURCE: 15-STATE ESKF`). Computes GNSS Course vs Filter Heading $\Delta$ **only** when moving ($v \ge 1.5$ m/s); stationary state correctly suppresses course comparison. |
| **Phone Attitude (Artificial Horizon)** | [`AttitudeHorizonView.kt`](../android/app/src/main/java/com/sih26168/idr/ui/AttitudeHorizonView.kt) | 🟢 Verified | Sky/ground canvas indicator with pitch ladder and central reticle. Driven directly by raw accelerometer attitude (`curAccel`), visibly tilting with phone handling. Separately labeled `PHONE ATTITUDE` from `VEHICLE NAV HEADING`. |
| **Altitude & Vertical Motion** | [`MainActivity.kt`](../android/app/src/main/java/com/sih26168/idr/MainActivity.kt#L636-L650) | 🟢 Verified | Explicitly delineates `GNSS ALT` (absolute WGS84) from `LOCAL ΔUP` (filter displacement relative to origin) and `VERT SPEED` (m/s). Includes diagnostic `⚠ VERTICAL DRIFT` warning chip when $\|LOCAL\ \Delta UP\| > 15.0$ m without modifying filter math. |
| **Sensor Health & Measured Timing** | [`MainActivity.kt`](../android/app/src/main/java/com/sih26168/idr/MainActivity.kt#L652-L666) | 🟢 Verified | Computes live execution rate and period from system timestamps (`RATE: X.XX Hz | Δt: XXX ms`). Does not assume 10 Hz. Displays norms for Accel ($m/s^2$), Gyro ($^\circ/s$), and Mag ($\mu T$) with 4-chip health indicators (`ACCEL`, `GYRO`, `MAG`, `GNSS`). |
| **Inertial Constraints & Rejection Reasons** | [`MainActivity.kt`](../android/app/src/main/java/com/sih26168/idr/MainActivity.kt#L668-L697) | 🟢 Verified | Explicitly displays constraint status with reasons: Compass rejection displays `(TURN RATE)` ($\omega_z > 3^\circ/s$) or `(DISTURBANCE)` ($dB/dt > 5\,\mu T/s$); Map guidance displays `(ACCEPTED)` or `(SEARCHING)`. |
| **Cartesian ENU Position & Outage Stopwatch** | [`activity_main.xml`](../android/app/src/main/res/layout/activity_main.xml#L487-L567) | 🟢 Verified | Displays local Cartesian displacement ($E$, $N$, $Up$) in meters with position uncertainty ($\pm 1\sigma$), plus geodetic Lat/Lon beneath map. Displays outage timer during dead-reckoning and resets cleanly during locked state. |
| **Golden Reference Verification** | [`DeadReckoningEngineTest.java`](../android/app/src/test/java/com/sih26168/idr/engine/DeadReckoningEngineTest.java) | 🟢 Verified | 1,789-epoch end-to-end replay verified via `./gradlew.bat test`. Position error RMSE during 60s blackout = 0.171 m, Max position error = 0.242 m, 100% constraint decision parity. 0 failures. |
| **Physical Vehicle Motion Validation** | Real-world Driving | 🟡 Observed (Pre-Test 3) | Software state-machine transitions and sensor streaming verified on physical `CPH2381` hardware in Test 2. Controlled in-car road driving validation to be executed in Test 3. |
| **Live Online Map Tile Database** | Map Rendering Layer | 🔴 Not Implemented | Offline local vector road network is used intentionally to maintain 100% offline autonomy and prevent cloud reliance during tunnel/blackout testing. |

---

## 3. Screen Organization & UX Architecture

The layout in [`activity_main.xml`](../android/app/src/main/res/layout/activity_main.xml) is structured into a dark-mode automotive diagnostic hierarchy:

```text
┌─────────────────────────────────────────────────────────┐
│ LIVE FIELD DIAGNOSTICS + NAV            GNSS LOCKED /   │
│ Real-Device Field Observer              DEAD RECKONING  │
├─────────────────────────────────────────────────────────┤
│ [ LOCAL OSM ROAD MAP (VECTOR ENU) ]                     │
│ Road Network: vta04 Segments (50m Grid)                 │
│ Layer A: Cyan GNSS ●   Layer B: Amber Vehicle Arrow →   │
│ GNSS: 17.445210°, 78.349120°   ACC: ±3.0 m             │
├─────────────────────────────────────────────────────────┤
│ SPEED COMPARISON (GROUND TRUTH VS CAUSAL ML)            │
│ GNSS MEASURED: 42.3 km/h    CAUSAL ML SPEED: 41.7 km/h  │
│ SPEED ERROR (GNSS - ML): -0.6 km/h                      │
├─────────────────────────────────────────────────────────┤
│ [ COMPASS ROSE ]        VEHICLE NAV HEADING: 338.4° NW   │
│ Rotating Dial           GNSS COURSE: 341.2°             │
│ ±1σ Uncertainty Wedge   FILTER HEADING: 338.4°          │
│ Needle forward          Δ: -2.8°  |  MAGNETIC: 335.1°   │
├──────────────────────────┬──────────────────────────────┤
│ [ ARTIFICIAL HORIZON ]   │ ALTITUDE / VERTICAL          │
│ PHONE ATTITUDE           │ GNSS ALT: 531.3 m            │
│ ROLL: +2.4° PITCH: -1.7° │ LOCAL ΔUP: +0.4 m            │
│ (Separated from vehicle) │ VERT SPEED: +0.02 m/s        │
│                          │ [⚠ VERTICAL DRIFT] (if >15m) │
├──────────────────────────┴──────────────────────────────┤
│ SENSOR HEALTH & MEASURED RATE                           │
│ RATE: 8.53 Hz | Δt: 117 ms                              │
│ [ACCEL | OK] [GYRO | OK] [MAG | OK] [GNSS | OK]         │
│ Acc: 9.81 m/s²   Gyro: 0.12 °/s   Mag: 44.5 µT          │
├─────────────────────────────────────────────────────────┤
│ INERTIAL CONSTRAINTS & QUALITY GATES                    │
│ LATERAL NHC: ACTIVE         VERTICAL NHC: ACTIVE        │
│ COMPASS: ACCEPTED           OSM MAP: ACCEPTED           │
│ (Shows reasons like REJ TURN RATE or REJ DISTURBANCE)   │
│ Accel Bias: +0.012 m/s²     ZARU Gyro: -0.045 °/s       │
├─────────────────────────────────────────────────────────┤
│ LOCAL CARTESIAN ENU ESTIMATE                            │
│ EAST: +120.4 m   NORTH: +450.2 m   UP: +0.4 m   ±1.50m  │
│ OUTAGE TIMER: 00:00.0 (or 0.0 s when locked)            │
├─────────────────────────────────────────────────────────┤
│ [ ▶ START 60s BLACKOUT REPLAY DEMO (MODE A) ]           │
│ [ 🔴 START LOGGING ]        [ ⚡ SIMULATE OUTAGE ]       │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Hardware Deployment & Verification Steps

1. **APK Packaging:**
   Debug APK assembled at:
   `android/app/build/outputs/apk/debug/app-debug.apk` (Size: 6.69 MB)

2. **Deploy to OnePlus Hardware (`CPH2381`):**
   Connect device via USB and run:
   ```bash
   adb -d install -r android/app/build/outputs/apk/debug/app-debug.apk
   adb -d shell am start -n com.sih26168.idr/.MainActivity
   ```

3. **Field Observer Checklist (Pre-Test 3 Verification):**
   - **Stationary Phone:** Compass rose points accurately; Artificial horizon responds smoothly when phone is physically tilted in hand; Update rate measures hardware frequency (e.g. ~8.53 Hz, not fixed 10 Hz).
   - **Outdoor Exposure:** Live geodetic Lat/Lon and GNSS accuracy appear; cyan marker snaps onto local road network; speed comparison reads 0.0 / 0.0 km/h; course says `STATIONARY`.
   - **Driving / Moving:** As vehicle accelerates, GNSS speed and Causal ML speed track side-by-side with dynamic error readout; GNSS course activates and displays direct angular difference $\Delta$; local road map tracks estimated trajectory with amber heading arrow.
   - **Outage Simulation:** Tapping `⚡ SIMULATE OUTAGE` turns badge red (`DEAD RECKONING`); GNSS speed changes to `--`; road map maintains estimated dead-reckoning trajectory; outage stopwatch counts elapsed blackout duration.

---

## 5. Conclusion

The application is now transparent and physically verifiable. It allows the driver and test observer to inspect real sensor norms, measured loop latency, phone tilt, causal speed error, course differences, and constraint rejection reasons without altering the underlying frozen navigation algorithms.

Ready for **Stage C10.2 Test 3 (Controlled Vehicle Motion Test)**.
