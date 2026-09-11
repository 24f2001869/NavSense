# Stage C10.5 Controlled Vehicle-Mounted Test Protocol

**Objective**: Establish a clean physical and kinematic baseline on real hardware under rigid mounting **before** modifying any navigation mathematics.  
**Platform**: OnePlus Nord CE 2 (`CPH2381` / Device `68afd407`), Android 13  
**Mounting**: Rigid vehicle phone holder, aligned with vehicle forward axis.  
**Constraint**: **NO CODE CHANGES. NO ESKF TUNING. CONTROL CONDITION.**

---

## 1. Physical Setup (C10.5-A)

```text
               VEHICLE FORWARD
                      ↑
                      │
             ┌─────────────────┐
             │     PHONE       │
             │   [Screen]      │
             │      📱         │
             └─────────────────┘
             Rigid Car/Bike Mount
```

### Pre-Drive Checklist:
- [ ] Phone locked securely in rigid mount.
- [ ] Phone screen facing driver/passenger; oriented along vehicle forward direction.
- [ ] **NO HANDHELD OPERATION**: Do not touch, rotate, tilt, or handle the phone during logging.
- [ ] Verify GNSS signal acquisition before starting recording.

---

## 2. Pre-Drive Stationary Calibration (C10.5-B)

To directly validate Stage C10.3's hypothesis regarding engine idle vibration vs. handheld noise, execute the following 3-minute sequence at the start of logging:

1. **Step 1 — Engine OFF (~60 s)**:
   - Phone mounted rigidly. Vehicle parked. Engine completely OFF.
   - Record ~60 seconds of baseline sensor noise floor.
2. **Step 2 — Engine ON, Stationary (~60 s)**:
   - Turn engine ON. Vehicle remains parked/stationary ($v=0$).
   - Record ~60 seconds of pure structural engine idle vibration.
3. **Step 3 — Vehicle Starts Moving (~60 s)**:
   - Shift into gear and begin moving gently ($v > 2\,\text{m/s}$).

---

## 3. Controlled Driving Route (C10.5-C)

**Total Duration**: ~5 to 10 minutes.  
**Critical Instruction**: **KEEP GNSS ON. DO NOT PRESS "SIMULATE OUTAGE".**

### Required Maneuvers Along Route:
- [ ] **Segment 1**: Straight cruising at normal road speed (30–50 km/h).
- [ ] **Segment 2**: Smooth acceleration from low to high speed.
- [ ] **Segment 3**: Smooth braking down to a halt.
- [ ] **Segment 4**: Left turn (intersection or roundabout).
- [ ] **Segment 5**: Right turn (intersection or roundabout).
- [ ] **Segment 6**: Normal road bumps / rough patch if encountered.
- [ ] **Segment 7**: Complete traffic stop (stand still at red light / stop for at least 30–60 seconds).

---

## 4. Post-Drive Automated Evaluation Checklist

Once logging is stopped and the CSV file is pulled via ADB:
```powershell
adb -s 68afd407 pull /sdcard/Android/data/com.sih26168.idr/files/ files/
python scratch/evaluate_c10_5_test1.py files/<latest_telemetry_file>.csv
```

### Key Decision Metrics:

1. **ML Speed Stationary Response**:
   - *If ML speed drops to 0–3 km/h when stopped*: The previous ~21 km/h floor was an artifact of handheld fidgeting.
   - *If ML speed remains ~20 km/h when stopped*: The problem is structural engine vibration coupling into RF feature energy.
2. **Moving Speed Concordance**:
   - Measure MAE, RMSE, and bias during moving cruising ($v > 2.0\,\text{m/s}$).
3. **Heading Observability**:
   - Compare filter heading against GNSS course during straight cruising. Check azimuth tracking during turns.
4. **Attitude Physical Sanity**:
   - Check roll and pitch angles during acceleration, braking, and turns. (Ensure smooth, bounded angles $< 10^\circ$).
5. **Vertical Channel Stability**:
   - Verify filter $pos\_u$ tracks GNSS altitude without unconstrained runaway.
6. **Sampling Timing**:
   - Verify effective logging rate (~8 Hz) and $\Delta t$ distribution.

---

## 5. Decision Gate (Next Step)

```text
               C10.5 Test 1 (GNSS-ON Baseline)
                            │
               ┌────────────┴────────────┐
               ↓                         ↓
           [HEALTHY]                 [UNHEALTHY]
               │                         │
     Proceed to C10.5-D          Diagnose Specific
   Controlled Blackouts          Channel Failure
   (10s, 30s, 60s Outages)        (No random tuning!)
```
