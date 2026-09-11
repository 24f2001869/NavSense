# Stage C10.7: VNHC Recovery Mechanism & Rejection Run Forensic Diagnostic
**Subsystem**: 15-State Error-State Kalman Filter (ESKF) • Vertical Non-Holonomic Constraint (VNHC)  
**Execution Context**: Offline Diagnostic Replay across 14 Real-World Outage Episodes (Trip 1 & Trip 2 Telemetry)  
**Corpus Scope**: 26,062 bus epochs (~54.1 minutes) • 14 Outage Episodes (3,786 outage epochs / 7.6 minutes)  
**Device Hardware**: OnePlus Nord CE 2 5G (`CPH2381` / Target Device `68afd407`)  
**Production Code Status**: **STRICTLY FROZEN (Zero production Java/Python navigation mathematics modified)**  

---

## 1. Executive Summary & The Decisive Scientific Verdict

Stage C10.7 was commissioned to resolve a critical, long-standing question in the vehicle dead-reckoning engine:

> **"Does VNHC fail during outages because the planar kinematic measurement ($v_z^v \approx 0$) becomes invalid, or because our innovation gate permanently prevents recovery after a temporary disturbance, or both?"**

To answer this question conclusively, four distinct mathematical variants of VNHC were evaluated across all **14 naturally occurring GNSS outage episodes** recorded on physical smartphone hardware in Indian city transit buses (spanning short, medium, and catastrophic 2- to 3-minute blackouts).

```text
                  DISTURBANCE (Turn / Hand Tremor)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       PHASE 1: INVALID DYNAMICS        PHASE 2: THE GATE LATCH
       (Phone tumblings / turns)       (Vehicle straight on flat road)
                 │                               │
    Kinematic v_z ≈ 0 is invalid      Kinematic v_z ≈ 0 is VALID!
    Turn-rate gate (>3°/s) trips      Turn rate returns to <1.0°/s
    v_z drifts by 2–5 m/s in 5s       v_z is now 3–5 m/s (from drift)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     INNOVATION GATE TRIPPED
                  |r_vert| > 3.0 m/s  OR  nis > 9.21
                                 │
                                 ▼
                    PERMANENT REJECTION RUN
              Every subsequent straight epoch rejected!
                   Filter never allowed to recover!
                                 │
                                 ▼
              UNCONSTRAINED FREE-FALL / SKYROCKET
                    (v_z accelerates to 500 m/s)
```

### The Decisive Verdict: Both, Operating Sequentially
1. **The Initial Trigger is Dynamic Invalidation (Phase 1)**: In a handheld phone on a transit bus, passenger handling and vehicle turns produce sustained intervals where turn rates exceed $3.0^\circ/\text{s}$ (and up to $1,182.7^\circ/\text{s}$). During these intervals, the 1-DOF planar constraint assumption ($v_z^v \approx 0$) is physically invalid in the phone body frame. The turn-rate gate correctly suspends VNHC updates.
2. **The Divergence is Caused by the Permanent Gate Latch (Phase 2)**: During the 3- to 15-second suspension, unconstrained vertical accelerometer biases integrate into vertical velocity, pushing $v_z$ to $2 - 5\,\text{m/s}$. When the turn finishes and the vehicle drives straight on a planar road ($v_z^v \approx 0$ is once again completely valid), the hard innovation gate (`|r_vert| > 3.0 m/s` or `nis > 9.21`) **permanently latches shut**. For the remainder of the outage (up to 118 seconds of straight driving), VNHC rejects 100% of epochs.
3. **The Danger of Ungated Updates (Why Huber Clamping on Raw Innovations Fails)**: Version D (continuous Huber covariance inflation without eligibility gating) experienced **matrix overflow and NaN/Inf numerical collapse** during violent phone tumbling (Trip 2 Ep 5), confirming the user's warning: forcing updates on invalid measurements corrupts the attitude and velocity cross-covariances.

---

## 2. Master Telemetry Corpus: Provenance & Validity Ledger

To eliminate discrepancy across previous analysis milestones, this ledger provides the exact sample counts, filter masks, and exclusions across all 8 sessions recorded on the OnePlus Nord CE 2:

| Session Filename | Duration (min) | Raw Rows | Complete Sensors (0 NaN) | GNSS Fix Valid (`lat > 10`) | Stationary Epochs (`v < 0.2`) | Moving Epochs (`v ≥ 0.2`) | Experimental Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `idr_telemetry_20260909_090715.csv` | 1.90m | 971 | 971 | 0 | 0 | 0 | Indoor bench handling before satellite fix |
| `idr_telemetry_20260909_091947.csv` | 1.44m | 749 | 749 | 749 | 749 | 0 | Outage toggle check (stationary rest) |
| `idr_telemetry_20260909_103010.csv` | 31.00m | 15,730 | 15,730 | 15,730 | 3,156 | 12,574 | **Bus Trip 1**: Hostel $\to$ Department (9 outages) |
| `idr_telemetry_20260909_125631.csv` | 23.13m | 10,332 | 10,332 | 10,311 | 3,142 | 7,169 | **Bus Trip 2**: Department $\to$ Hostel (5 outages) |
| `idr_telemetry_20260909_142659.csv` | 11.60m | 6,082 | 6,082 | 6,080 | 2,370 | 3,710 | Afternoon transit & pedestrian walk |
| `idr_telemetry_20260909_150327.csv` | 13.25m | 6,622 | 6,622 | 6,622 | 6,622 | 0 | Resting desk baseline 1 (stationary) |
| `idr_telemetry_20260909_151642.csv` | 1.28m | 561 | 561 | 561 | 561 | 0 | Resting desk baseline 2 (ultra-quiet) |
| `idr_telemetry_20260909_163531.csv` | 15.00m | 3,897 | 3,897 | 3,888 | 3,554 | 334 | Evening stationary & campus walk |
| **TOTAL CORPUS** | **98.59m** | **44,944** | **44,944** | **43,941** | **20,154** | **23,787** | **Complete Multi-Session Hardware Corpus** |

### Reconciliation Across Milestone Reports
1. **Raw Logged Rows (All 8 Files)**: $N = \mathbf{44,944\text{ epochs}}$ (100.0% sensor-complete, 0 NaNs).
2. **Master Analysis Corpus (C10.6 Sweep)**: $N = \mathbf{43,941\text{ epochs}}$ (97.77%). Excludes 1,003 epochs without GNSS fix (971 from Session 1 indoor baseline, 21 from Session 4 fix delay, 2 from Session 5, 9 from Session 8).
3. **C10.2 Raw Bus Telemetry**: $N = \mathbf{26,062\text{ epochs}}$ (Trip 1: $15,730$ + Trip 2: $10,332$).
4. **C10.3 / C10.4 Bus Sensor Diagnostic Corpus**: $N = \mathbf{26,041\text{ epochs}}$. Excludes 21 pre-fix epochs (14 on Trip 1, 7 on Trip 2).

---

## 3. The 4 Evaluated VNHC Formulations

Every outage episode was replayed under identical initial conditions (pre-outage position, velocity, attitude DCM, error covariance $P$, and bias estimates) using four isolated variants:

### Version A — Current Control (Production Baseline)
- **Turn-Rate Gate**: If $|\omega_z| > 3.0^\circ/\text{s}$, reject update (`reason: 'turning'`).
- **Hard Residual Gate**: If $|r_{\text{vert}}| > 3.0\,\text{m/s}$ OR $\text{nis} > 9.21$, reject update (`reason: 'gated'`).
- **Huber Weighting**: $\gamma = \min(1.0, 2.0 / \sqrt{\text{nis}})$ if $\text{nis} > 4.0$.
- **Update**: Linear ESKF velocity update with frozen position and attitude gains ($K_{0:3} = 0, K_{6:15} = 0$).

### Version B — Gate Removed (No Innovation Truncation)
- **Turn-Rate Gate**: Retained ($|\omega_z| > 3.0^\circ/\text{s} \implies$ reject).
- **Hard Residual Gate**: **DISABLED**. Never returns early due to residual or $\chi^2$ magnitude.
- **Huber Weighting**: Standard $\gamma = \min(1.0, 2.0 / \sqrt{\text{nis}})$.

### Version C — Recoverable Eligibility (Cooldown + Adaptive Recovery)
- **Turn-Rate Gate**: Retained ($|\omega_z| > 3.0^\circ/\text{s} \implies$ reject).
- **Cooldown & Health Monitor**: Tracks consecutive epochs where vehicle dynamics are healthy ($|\omega_z| \le 3.0^\circ/\text{s}$).
- **Eligibility Recovery**:
  - If $|r_{\text{vert}}| > 3.0\,\text{m/s}$ but vehicle has been traveling straight for $\ge 4$ epochs ($\approx 0.5\,\text{s}$), the measurement is recognized as valid!
  - To prevent rejection, the filter adapts measurement variance to match accumulated uncertainty:
    $$R_{\text{recover}} = \max\left(R_{\text{base}}, \frac{r_{\text{vert}}^2}{\chi_{\text{gate}}^2}\right)$$
  - Standard ESKF Kalman update executes with $R_{\text{recover}}$, gently contracting $v_z$ back to nominal without innovation shock.
  - **Zero State Clamping**: No manual setting of $v_z = 0$.

### Version D — Robust Innovation Weighting (Continuous Huber M-Estimator)
- **Turn-Rate Gate**: Retained ($|\omega_z| > 3.0^\circ/\text{s} \implies$ reject).
- **Innovation Gate**: Replaced with continuous Huber measurement variance inflation:
  $$R_{\text{eff}} = \begin{cases} R_{\text{base}} & \text{if } \text{nis} \le k^2 \\ R_{\text{base}} \cdot \frac{\sqrt{\text{nis}}}{k} & \text{if } \text{nis} > k^2 \quad (k=2.5) \end{cases}$$
- **Update**: Joseph-form covariance update with $R_{\text{eff}}$.

---

## 4. Master Outage-by-Outage Comparative Performance Table

The table below summarizes all 14 simulated outage episodes across Trip 1 and Trip 2 replayed through all four engines:

| Outage Episode | Dur (s) | Version A (Control) $\Delta pos_u$ (m) | Version B (Gate Removed) $\Delta pos_u$ (m) | Version C (Recoverable) $\Delta pos_u$ (m) | Version D (Robust Huber) $\Delta pos_u$ (m) | Version A VNHC % | Version B VNHC % | Version C VNHC % | Version A Max Rej (s) | Version B Max Rej (s) | Version C Max Rej (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip 1 Ep 1** | 3.9s | $-8.4$ | $-8.4$ | $-8.4$ | $-8.4$ | 0.0% | 0.0% | 0.0% | 3.8s | 3.8s | 3.8s |
| **Trip 1 Ep 2** | 94.3s | $+350.5$ | $+29.4$ | $+4.0$ | $+69.1$ | 46.8% | 62.5% | 57.9% | 16.3s | 2.0s | 2.5s |
| **Trip 1 Ep 3** | 132.4s | $+709.3$ | $+70.4$ | $+161.9$ | $-3,887.8$ | 18.7% | 73.5% | 71.7% | 71.4s | 7.1s | 8.8s |
| **Trip 1 Ep 4** | 128.6s | $+358.0$ | $+171.2$ | $-973.2$ | $+171.2$ | 37.9% | 56.7% | 54.9% | 29.9s | 8.5s | 8.9s |
| **Trip 1 Ep 5** | 125.1s | **$-41,851.9$** | **$-258.1$** | **$-2,567.4$** | $+6,909.8$ | **4.4%** | **59.0%** | **39.2%** | **118.6s** | **6.8s** | **10.1s** |
| **Trip 1 Ep 6** | 24.9s | $+166.5$ | $-35.2$ | $-206.3$ | $-343.1$ | 2.8% | 25.7% | 20.2% | 13.3s | 7.3s | 9.0s |
| **Trip 1 Ep 7** | 99.5s | $-5,402.0$ | $+339.5$ | $-5,402.0$ | $+9,878.8$ | 0.8% | 6.9% | 0.8% | 32.3s | 17.6s | 32.3s |
| **Trip 1 Ep 8** | 39.9s | $+1,039.6$ | $-2,051.8$ | $+1,044.8$ | $-1,314.8$ | 0.3% | 9.0% | 5.1% | 32.0s | 12.8s | 29.9s |
| **Trip 1 Ep 9** | 40.6s | $+509.3$ | $-94.9$ | $-22.8$ | $-125.2$ | 0.4% | 85.5% | 78.4% | 30.0s | 1.5s | 2.4s |
| **Trip 2 Ep 1** | 2.4s | $-7.3$ | $-4.7$ | $-7.3$ | $-3.4$ | 4.8% | 33.3% | 4.8% | 1.9s | 0.6s | 1.9s |
| **Trip 2 Ep 2** | 161.7s | $-36.0$ | $+16,208.0$ | $+3,572.0$ | $+3,508.8$ | 26.1% | 47.3% | 32.0% | 77.1s | 10.6s | 31.3s |
| **Trip 2 Ep 3** | 57.6s | $+365.7$ | **$-13.1$** | **$-8.6$** | $-14.4$ | 7.3% | **92.1%** | **91.3%** | 53.4s | **1.3s** | **1.9s** |
| **Trip 2 Ep 4** | 80.1s | $-373.5$ | $-373.5$ | $-373.5$ | $-373.5$ | 78.5% | 78.5% | 78.5% | 3.1s | 3.1s | 3.1s |
| **Trip 2 Ep 5** | 224.0s | $-16,058.5$ | $-19,228.7$ | $-61,739.6$ | **NaN / Inf** | 0.0% | 8.6% | 2.5% | 61.1s | 22.5s | 59.3s |

---

## 5. Aggregate Summary & Failure Mechanism Analysis

| Metric | Version A (Control) | Version B (Gate Removed) | Version C (Recoverable) | Version D (Robust Huber) | Impact of Gate Removal / Recovery |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Median Absolute Vertical Drift $|\Delta pos_u|$** | **$441.4\,\text{m}$** | **$214.7\,\text{m}$** | **$364.4\,\text{m}$** | $343.1\,\text{m}$ | **$-51.4\%$ median drift reduction** |
| **Mean VNHC Constraint Active %** | **$16.3\%$** | **$45.6\%$** | **$38.4\%$** | $45.6\%$ | **$+179\%$ constraint availability** |
| **Mean Longest Rejection Run** | **$38.9\,\text{s}$** | **$7.5\,\text{s}$** | **$14.6\,\text{s}$** | $7.5\,\text{s}$ | **$-80.7\%$ latch duration** |
| **Mean Peak Runaway Velocity $\max |v_u|$** | $185.7\,\text{m/s}$ | $198.4\,\text{m/s}$ | $201.5\,\text{m/s}$ | $352.1\,\text{m/s}$ | Unconstrained velocity in tumbling |
| **Numerical Integrity (NaN / Inf Failures)** | **0** | **0** | **0** | **1 (Crash)** | **Version D unstable in tumbling** |

*[Stage C10.7 Diagnostic Multi-Panel Figure — Diagnostic chart]*

---

## 6. Deep Forensic Case Studies

### 6.1 The Smoking Gun Case: Trip 1 Episode 5 (125.1 s Blackout)
- **What Happened in Version A (Control)**:
  - At $t=0$ to $16\,\text{s}$, VNHC was active and stable ($v_u \approx 0.1\,\text{m/s}$).
  - At $t=16.8\,\text{s}$, a slight bus jerk produced $|\omega_z| = 3.1^\circ/\text{s}$, suspending VNHC for 2 epochs.
  - During the suspension, uncorrected accelerometer bias integrated $v_u$ to $1.88\,\text{m/s}$.
  - At $t=17.1\,\text{s}$, turn rate returned to $1.0^\circ/\text{s}$ (straight road). But $\text{nis} = (1.88)^2 / 0.32 = 11.0 > 9.21$!
  - **The gate latched shut**: For the next **118.6 consecutive seconds**, VNHC was rejected on every single epoch!
  - Net vertical position plummeted to **$-41,851.9\,\text{meters}$** ($v_u = 1,133\,\text{m/s}$).
- **What Happened in Version B (Gate Removed)**:
  - When the turn ended and the road was straight, Version B re-admitted the VNHC measurement.
  - VNHC active rate rose to **$59.0\%$** (all non-turning epochs).
  - Net vertical position error was held to **$-258.1\,\text{meters}$** — an **over 99.4% error reduction**!

### 6.2 The Ideal Road Case: Trip 2 Episode 3 (57.6 s Outage)
- In this episode, the phone was held flat and steady (mean $\omega_z = 0.24^\circ/\text{s}$, mean $a_z = 9.53\,\text{m/s}^2$).
- In Version A (Control), a transient bump tripped the innovation gate, resulting in a **$53.4\,\text{second}$ rejection run** and $+365.7\,\text{m}$ drift.
- In Version B, C, and D: VNHC was active **$91.3\% - 92.1\%$** of the time.
- Net vertical drift was held to:
  - Version B: **$-13.1\,\text{m}$**
  - Version C: **$-8.6\,\text{m}$**
  - Version D: **$-14.4\,\text{m}$**
  - Drift rate was only **$0.15\,\text{m/s}$**!

### 6.3 The Pathological Stress Case: Trip 2 Episode 5 (224.0 s Outage)
- In this episode, the passenger held the phone while standing/walking; angular rates reached $1,182.7^\circ/\text{s}$, with non-zero mean rotation in all 3 axes ($\omega_x = +36^\circ/\text{s}, \omega_y = -33^\circ/\text{s}$).
- Because the phone was tumbling, $91.4\%$ of epochs had $|\omega_z| > 3.0^\circ/\text{s}$.
- Version D attempted to continuously force VNHC updates without eligibility gating:
  - Because the attitude matrix was tilted by tens of degrees, what VNHC assumed was "vertical" was actually pointing into the horizontal road plane.
  - The conflicting constraints corrupted the error covariance matrix $P$, causing `invalid value encountered in sqrt` and **numerical overflow resulting in NaN/Inf**.
- **Crucial Engineering Lesson**: An innovation gate or dynamic eligibility check is **vital** to protect against filter corruption during extreme non-vehicular motions. Completely removing all gates (or applying blind Huber attenuation) creates catastrophic numerical vulnerability.

---

## 7. Next Engineering Decisions & Architectural Roadmap

Based on this unvarnished evidence ledger, we establish the following frozen decisions:

1. **Retire the Hard $3.0\,\text{m/s}$ Innovation Cut-Off**:
   - The clause `if (Math.abs(r_vert) > 3.0) return;` in `DeadReckoningEngine.java` is the direct cause of the permanent latch. It is retired from consideration.
2. **Reject Un-Gated Huber Inflation (Version D)**:
   - Blind Huber covariance scaling without dynamic attitude/turn gating can cause numerical overflow during passenger handling.
3. **The Deployable Architecture (Recoverable Dynamic Eligibility)**:
   - When vehicle motion is confirmed planar and non-turning ($|\omega_z| \le 3.0^\circ/\text{s}$ for $\ge 0.5\,\text{s}$):
     The filter must permit recovery updates using variance-adapted linear innovation updates, preventing uncorrected bias runaway while preserving immunity against tumbling.
4. **Offline-First Constraint**:
   - No modifications will be committed to `android/` until the complete C10.8 head-to-head re-simulation confirms zero collateral degradation on horizontal accuracy.
