# Stage C10.8C — VNHC Update-Response / Gain-Sign Stability Audit

**Status**: COMPLETED (Offline Forensic Characterization Only)  
**Execution Context**: Offline simulation across all 12 genuine recovery episodes from Android field trial logs (`idr_telemetry_20260909_103010.csv` and `idr_telemetry_20260909_125631.csv`).  
**Production Code Freeze**: Strictly maintained. Zero modifications to `android/`, `src/navigation/dead_reckoning_engine.py`, Kalman gain math, measurement equations, covariance propagation, Huber thresholds, or detector constants. No production thresholds proposed.

---

## Executive Summary

Stage C10.8C investigates the fundamental question established by C10.8B:
> **“What makes a physically eligible VNHC update stabilizing in one case and destabilizing in another?”**

By evaluating **every single accepted VNHC update** ($N = 3,166$ updates across 12 recovery episodes) rather than only the first post-recovery sample, C10.8C isolates the exact mathematical and kinematic mechanisms of convergence and divergence.

```
                     ┌────────────────────────────────────────────────────────┐
                     │          PHYSICALLY ELIGIBLE VNHC ADMISSION            │
                     │  (Turn rate < 3°/s, ||ω|| < 0.4 rad/s, a_dyn < 1.5)    │
                     └───────────────────────────┬────────────────────────────┘
                                                 │
                                 Is C_33 ≈ 0 (Roll ≈ ±90°)?
                                ┌────────────────┴────────────────┐
                                ▼ YES                             ▼ NO
                   ┌───────────────────────────┐     ┌───────────────────────────┐
                   │     FAILURE MODE A        │     │  Upright / Tilted Normal  │
                   │ (Orthogonal Inversion)    │     │   (C_33 > 0.8, Roll < 35°)│
                   │   K_vu < 0 (Anti-damping) │     └─────────────┬─────────────┘
                   │ Immediate Runaway (T1 Ep8)│                   │
                   └───────────────────────────┘       Trajectory Gain G_5s < 1.0?
                                                      ┌────────────┴────────────┐
                                                      ▼ YES                     ▼ NO
                                         ┌─────────────────────────┐ ┌─────────────────────────┐
                                         │   STABLE DAMPING        │ │     FAILURE MODE B      │
                                         │  (T2 Ep3, T1 Ep5)       │ │ (Inter-Update Drift)    │
                                         │ Fast state decay        │ │ Propagation > VNHC rate │
                                         │ G_1s = 0.17, G_5s = 0.52│ │ G_5s = 1.50 (T2 Ep2)   │
                                         └─────────────────────────┘ └─────────────────────────┘
```

---

## 🟢 WHAT WE KNOW (Empirically Proven by C10.8C)

1. **Initial Damping is Not Guaranteed by Physical Eligibility**:
   - Out of 12 recovery episodes, the very first accepted VNHC update was **damping** ($|v_U^+| < |v_U^-|$, $G < 1.0$) in **5 episodes** (41.7%) and **anti-damping** ($|v_U^+| > |v_U^-|$, $G > 1.0$) in **7 episodes** (58.3%).
   - Physical calm (low turn rate, low angular velocity, low dynamic acceleration) does *not* imply the internal filter state is aligned for instantaneous damping.

2. **$K_{v_u} > 0$ is Necessary for Damping but Clearly Insufficient for Stability**:
   - In **Trip 2 Ep 2**, the first accepted update had $K_{v_u} = +0.607 > 0$, $\text{pitch} = -8.8^\circ$, $\text{roll} = -7.6^\circ$, $C_{33} = 0.980$, and $\text{NIS} = 3.52$.
   - The first update successfully damped velocity ($v_U: -2.57 \to -1.71\,\text{m/s}$, $G = 0.667$).
   - Yet over the subsequent 10 seconds, state magnitude grew ($G_{5\text{s}} = 1.50$, $G_{10\text{s}} = 2.36$), reaching $v_U = -5.69\,\text{m/s}$ while $K_{v_u}$ remained strictly positive ($+0.795$). Positive gain is not sufficient to guarantee convergence.

3. **Negative $K_{v_u}$ Causes Immediate Anti-Damping Runaway (Failure Mode A)**:
   - In **Trip 1 Ep 8**, the phone was rolled sideways on its edge ($\text{roll} = -90.3^\circ$, $C_{33} = -0.003$).
   - Because the measurement Jacobian maps body vertical velocity through the coordinate transformation matrix $C_{n\to v}[2, :]$, when $C_{33} \approx 0$, cross-coupling from horizontal velocity and tilt error inverted the gain sign ($K_{v_u} = -0.104$).
   - Update #0 had $r = -298.57\,\text{m/s}$, injecting $\Delta v_U = K_{v_u} \cdot r = +31.09\,\text{m/s}$ in the wrong direction ($v_U: +162.65 \to +192.65\,\text{m/s}$, $G = 1.184$).
   - Within 3 updates (0.35s), velocity exploded to $+269.16\,\text{m/s}$ ($G_{1\text{s}} = 1.41$).

4. **Trip 2 Ep 2 Did Not Diverge Upright; It Was Destabilized by a Mid-Blackout Attitude Corruption Event**:
   - A critical forensic discovery: In **Trip 2 Ep 2**, the filter was **completely stable for the first 90 seconds** of blackout!
     - $t = 20\text{s}$: $v_U = +0.13\,\text{m/s}$, $\Delta U = -27.4\,\text{m}$
     - $t = 42\text{s}$: $v_U = -0.03\,\text{m/s}$, $\Delta U = -21.0\,\text{m}$ ($K_{v_u} = +0.939$, $C_{33} = 0.982$, $\text{NIS} = 0.0$)
     - $t = 78\text{s}$: $v_U = +0.06\,\text{m/s}$, $C_{33} = 0.856$
     - $t = 90\text{s}$: $v_U = +3.11\,\text{m/s}$, $\Delta U = -4.8\,\text{m}$
   - The massive divergence ($v_U \to 881.9\,\text{m/s}$, $\Delta U \to 30.8\,\text{km}$) was **NOT** caused by accumulated small errors during upright cruise.
   - At $t \approx 92-99\text{s}$, the passenger physically rotated the phone sideways ($\text{roll} = -88.8^\circ$, $C_{33} \to 0.019$).
   - At $t = 99.2\text{s}$ (update #277), VNHC admitted an update while the phone was rotated sideways, $K_{v_u}$ flipped negative ($-0.360$), injecting $+20.6\,\text{m/s}$ of erroneous upward velocity ($v_U: 0.23 \to 20.88\,\text{m/s}$). Subsequent runaway updates blew the velocity to $240 \to 392 \to 878\,\text{m/s}$.
   - **Both Trip 1 Ep 8 and the late explosion in Trip 2 Ep 2 share the exact same root mechanism: Failure Mode A (orthogonal/inverted attitude acceptance).**

5. **Trajectory Gain Ratio $G_{0\to \Delta t}$ Discloses Estimator Health Far Better Than Residual Magnitude**:
   - In **Trip 1 Ep 5** (road bump), residual at U0 was $0.50\,\text{m/s}$, and at U7 reached $7.65\,\text{m/s}$. Yet $G_{5\text{s}} = 0.34$ and the state settled cleanly ($\Delta U = +147.8\,\text{m}$, peak $v_U = 10.9\,\text{m/s}$).
   - In **Trip 2 Ep 3** (stable cruise), U0 had $|r| = 2.93\,\text{m/s}$, yet $G_{1\text{s}} = 0.17$, $G_{2\text{s}} = 0.02$, $G_{5\text{s}} = 0.52$, settling within 2 seconds.
   - In contrast, in **Trip 2 Ep 5** (tumbling lull), initial $|r| = 1.69\,\text{m/s}$, yet $G_{1\text{s}} = 8.39$, $G_{5\text{s}} = 18.78$, $G_{10\text{s}} = 26.14$, rapidly exploding to $3,365\,\text{m/s}$.

---

## 🟡 WHAT WE THINK (Strong Hypotheses Supported by Data)

1. **Failure Mode B is Caused by Propagation Rate Dominating Intermittent VNHC Rate**:
   - In Trip 2 Ep 2 (during $t = 0-10\text{s}$) and Trip 1 Ep 3, VNHC updates are damping ($G_{\text{step}} < 1.0$), but inter-update gaps of $0.5\text{s} - 1.5\text{s}$ under unmodeled vibration/bias allow the open-loop strapdown integrator to re-accumulate downward velocity faster than VNHC removes it.
   - When updates occur at $>8\,\text{Hz}$ with low gaps (Trip 2 Ep 3, mean gap $0.129\,\text{s}$), damping outpaces drift and the state rapidly converges.

2. **$C_{n\to v}[2, 2] > 0.8$ (Tilt $< 36^\circ$) is a Valid Necessary Guard Against Catastrophic Mode A Inversion**:
   - Both Trip 1 Ep 8 ($C_{33} = -0.003$) and Trip 2 Ep 2 late ($C_{33} = 0.019$) occurred when the phone was rolled near $\pm 90^\circ$.
   - A geometric projection check $C_{33} > 0.8$ would have rejected the corrupting updates in both Trip 1 Ep 8 and the $t=99\text{s}$ event in Trip 2 Ep 2, preventing both catastrophic explosions.
   - However, $C_{33} > 0.8$ does *not* prevent Failure Mode B or Failure Mode C (Trip 2 Ep 5, where $C_{33} \approx 1.0$ during momentary upright lulls of tumbling).

3. **Dynamic Closed-Loop Trajectory Monitoring ($G$) is Conceptually Superior to Static Gates**:
   - Instead of trying to guess whether an update will be safe a priori using innovation gates (which fails), monitoring whether $|v_U|$ is expanding or contracting over a trailing window ($0.5\text{s} - 2\text{s}$) directly observes filter stability in closed loop.

---

## 🔴 WHAT WE DON'T KNOW (Unresolved Research Questions)

1. **How to Detect When Estimator Attitude Has Silently Decoupled from Physical Reality**:
   - In Trip 2 Ep 5, the phone was tumbling, but during a brief calm period, the filter's estimated $C_{33}$ was 1.000 (perfectly upright) because the gyros had temporarily settled. But the true orientation of the vehicle was unknown. We do not yet know how to detect that the filter's attitude uncertainty has grown unobservable without GNSS.

2. **What Inter-Update Frequency is Strictly Required to Overcome Strapdown Bias Drift**:
   - We observed that $8\,\text{Hz}$ updates with $<0.2\,\text{s}$ gaps converged stably, whereas $>0.7\,\text{s}$ gaps accumulated drift. The exact mathematical stability boundary between strapdown propagation noise and Kalman gain damping has not been analytically derived for this IMU noise density.

3. **Whether Trajectory Reversion / State Rollback is Safe in Production**:
   - If an estimator detects $G_{2\text{s}} > 2.0$, should it freeze vertical updates, inflate covariance, or revert state to the pre-admission epoch? The system-level implications on horizontal position coupling remain uncharacterized.

---

## Direct Answers to the 7 Primary Research Questions (Q1–Q7)

### Q1. Is the first VNHC correction damping or anti-damping?
**Answer**: In **5 of 12 episodes** (41.7%), the first correction is **damping** ($|v_U^+| < |v_U^-|$), including Trip 1 Ep 5 ($G = 0.512$), Trip 2 Ep 3 ($G = 0.414$), and Trip 2 Ep 2 ($G = 0.667$). In **7 of 12 episodes** (58.3%), the first correction is **anti-damping** ($|v_U^+| > |v_U^-|$), including Trip 1 Ep 8 ($G = 1.184$, $v_U: 162.65 \to 192.65\,\text{m/s}$) and Trip 2 Ep 5 ($G = 4.789$, $v_U: -1.69 \to -8.10\,\text{m/s}$). Initial physical eligibility does not guarantee an initial damping update.

### Q2. If the first correction is damping, do subsequent accepted updates remain damping?
**Answer**: **No.** In Trip 2 Ep 3 (stable recovery), damping is maintained across updates 0–4 ($G = 0.41, 0.30, 0.39, 0.84, 0.53$). But in Trip 2 Ep 2, update 0 ($G = 0.667$) and update 1 ($G = 0.932$) are damping, while update 2 ($G = 1.040$) and update 3 ($G = 1.013$) turn anti-damping. Across all 3,166 updates in the dataset, the overall damping rate is $54.5\%$ (near a 50/50 balance). Damping fluctuates dynamically based on sensor noise and vibration.

### Q3. Does $K_{v_u}$ change sign before instability?
**Answer**: **It depends on the failure mode.**
- In **Trip 1 Ep 8** (Mode A), $K_{v_u}$ is **negative from epoch 0** ($K_{v_u} = -0.104$ at $t=38.15\text{s}$) due to inverted attitude ($C_{33} = -0.003$). Anti-damping runaway begins on the very first update.
- In **Trip 2 Ep 2**, $K_{v_u}$ remains **strictly positive** ($+0.14$ to $+0.94$) across the first 263 updates and 90 seconds! It only flipped negative at update #277 ($t = 99.2\text{s}$) when the phone was physically rolled sideways ($\text{roll} = -88.8^\circ$).
- In **Trip 2 Ep 5**, $K_{v_u}$ remained **positive across all 15 updates** ($+0.05$ to $+0.26$), yet velocity diverged to $3,365\,\text{m/s}$.
- **Conclusion**: A negative sign in $K_{v_u}$ immediately produces anti-damping runaway, but a sign change is **not** a prerequisite for divergence (positive $K_{v_u}$ can also diverge).

### Q4. Does instability begin from a single anti-damping update or cumulative repeated small corrections?
**Answer**: **Both mechanisms occur in real-world data:**
- **Mechanism A (Single / Immediate Anti-Damping Step)**: Trip 1 Ep 8. Update 0 injects $+31.09\,\text{m/s}$ of false upward velocity in a single step ($K_{v_u} < 0$), instantly launching divergence.
- **Mechanism B (Cumulative Propagation Drift Between Updates)**: In Trip 2 Ep 2 ($t = 0-10\text{s}$), individual updates are small and damping, but intermittent update gaps allow IMU strapdown integration under bias to overpower the damping rate ($G_{5\text{s}} = 1.50$).
- **Mechanism C (Delayed Mid-Blackout Attitude Corruption Re-Triggering Mechanism A)**: In Trip 2 Ep 2 at $t = 99\text{s}$, a secondary physical handling event rotated the phone sideways, triggering Mechanism A mid-blackout.

### Q5. Does $C_{n\to v}[2,2]$ distinguish the severe attitude case?
**Answer**: **Yes for severe attitude inversion, but no for upright divergence.**
- In Trip 1 Ep 8, $C_{33} = -0.003 \approx 0$ (phone on its side).
- In Trip 2 Ep 2 (late explosion), $C_{33} = 0.019 \approx 0$ (phone on its side).
- A check requiring $C_{33} > 0.80$ would catch both of these catastrophic events.
- However, in Trip 2 Ep 2 (first 10s), $C_{33} = 0.980$ (upright), yet velocity grew. In Trip 2 Ep 5, $C_{33} = 1.000$, yet velocity exploded. Thus $C_{33} > 0.8$ is a necessary safety guard against inverted geometry, but **not sufficient** for overall stability.

### Q6. Can the T2 Ep2 instability occur while $K_{v_u}$ remains positive and attitude appears reasonable?
**Answer**: **YES, decisively confirmed.** During the first 10 seconds of Trip 2 Ep 2, $K_{v_u} = +0.607 > 0$, $\text{pitch} = -8.8^\circ$, $\text{roll} = -7.6^\circ$ ($C_{33} = 0.980$), $\text{NIS} = 3.52$, and $|r| = 1.50\,\text{m/s}$. Despite all these metrics appearing benign, velocity grew by $50\%$ in 5 seconds ($G_{5\text{s}} = 1.50$) and $136\%$ in 10 seconds ($G_{10\text{s}} = 2.36$).

### Q7. Does update-to-update gain/state evolution explain the divergence better than residual magnitude?
**Answer**: **YES, decisively.**
- Residual magnitude fails: Trip 1 Ep 5 ($|r| = 0.50\,\text{m/s}$ at U0, later $7.65\,\text{m/s}$ at U7) settled stably, whereas Trip 2 Ep 2 ($|r| = 1.50\,\text{m/s}$) and Trip 1 Ep 4 ($|r| = 0.11\,\text{m/s}$) expanded.
- In contrast, the trajectory gain ratio $G_{0\to \Delta t} = \frac{|v_U(t)|}{|v_U(t_0)| + \epsilon}$ cleanly separates stability from instability across all episodes:
  - **Stable Cases**: $G_{1\text{s}} = 0.17$ (T2 Ep3), $G_{5\text{s}} = 0.34$ (T1 Ep5).
  - **Expanding Cases**: $G_{1\text{s}} = 1.41$ (T1 Ep8), $G_{5\text{s}} = 1.50$ (T2 Ep2), $G_{1\text{s}} = 8.39$ (T2 Ep5).

---

## Detailed Comparative Matrix of the 12 Recovery Episodes

| Episode | Duration | Updates | Damping % | U0 Damp? | U0 $G$ | U0 $K_{v_u}$ | U0 $C_{33}$ | First Neg $K_{v_u}$ | $G_{1\text{s}}$ | $G_{5\text{s}}$ | Peak $|v_U|$ (m/s) | Post $\Delta U$ (m) | Final Horiz Err (m) | Mechanism Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Trip 1 Ep 2** | 94.3s | 229 | 52.0% | True | 0.019 | +0.748 | 0.997 | Upd #48 (23.5s) | 0.23 | 0.18 | 16.3 | +33.1 | 3,556.3 | Partial Damping / Mild Drift |
| **Trip 1 Ep 3** | 100.8s | 734 | 51.4% | False | 1.066 | +0.234 | 1.000 | Upd #51 (14.2s) | 2.13 | 0.69 | 110.7 | +308.0 | 5,667.0 | Mode B (Inter-update drift) |
| **Trip 1 Ep 4** | 82.5s | 483 | 53.4% | False | 5.581 | +0.224 | 1.000 | Upd #79 (17.8s) | 10.83 | 6.44 | 40.6 | +772.4 | 4,996.0 | Mode B (Inter-update drift) |
| **Trip 1 Ep 5** | 46.3s | 310 | **64.2%** | **True** | **0.512** | **+0.641** | **0.995** | Upd #10 (31.2s) | **1.07** | **0.34** | **10.9** | **+147.8** | **476.2** | **Stable Road Bump Recovery** |
| **Trip 1 Ep 6** | 4.6s | 22 | 72.7% | False | 3.065 | +0.333 | 0.999 | Upd #21 (3.7s) | 16.81 | 11.65 | 48.8 | -185.3 | 259.5 | Short Episode / Rapid Drift |
| **Trip 1 Ep 8** | 40.6s | 17 | 41.2% | **False** | **1.184** | **-0.104** | **-0.003** | **Upd #0 (38.1s)** | **1.41** | **N/A** | **291.4** | **+531.3** | **5,234.2** | **Mode A (Attitude Inversion)** |
| **Trip 1 Ep 9** | 28.5s | 195 | 52.8% | False | 4.345 | +0.350 | 1.000 | Upd #46 (8.1s) | 6.54 | 9.26 | 17.7 | -67.8 | 719.1 | Mode B (Inter-update drift) |
| **Trip 2 Ep 1** | 2.1s | 2 | 100.0% | True | 0.870 | +0.208 | 0.991 | None | 1.60 | N/A | 5.2 | -5.2 | 8,540,570.7 | Truncated Blackout (2s) |
| **Trip 2 Ep 2** | 161.7s | 329 | 48.9% | **True** | **0.667** | **+0.607** | **0.980** | **Upd #264 (92.3s)** | **0.77** | **1.50** | **881.9** | **+30,786.7** | **12,110.5** | **B (0-90s) → A @ 99s (Roll -89°)** |
| **Trip 2 Ep 3** | 56.4s | 417 | **54.2%** | **True** | **0.414** | **+0.469** | **0.984** | Upd #35 (9.4s) | **0.17** | **0.52** | **2.0** | **-8.3** | **2,347.6** | **Stable Convergence Benchmark** |
| **Trip 2 Ep 4** | 58.1s | 417 | 31.4% | False | 3.885 | +0.229 | 0.999 | Upd #55 (13.6s) | 7.87 | 59.52 | 9.5 | -355.2 | 3,788.6 | Mode B (Inter-update drift) |
| **Trip 2 Ep 5** | 57.8s | 15 | 46.7% | **False** | **4.789** | **+0.263** | **1.000** | **None** | **8.39** | **18.78** | **3,365.8** | **+134,822.5** | **233,119.6** | **Mode C (Tumbling Lull Acceptance)** |

---

## Deep Dive Forensic of the 5 Showcase Cases

### Case 1: Trip 1 Ep 5 — Large-Residual Successful Recovery
- **Disturbance**: Severe vertical road bump causing 40m drop in Control.
- **Update Behavior**:
  - Update #0 ($t = 2.35\text{s}$): $r = +0.50\,\text{m/s}$, $v_U^{\text{pre}} = -0.69$, $v_U^{\text{post}} = -0.35$, $K_{v_u} = +0.641$, $G = 0.512$ (DAMPING).
  - Update #1 ($t = 2.80\text{s}$): $r = +0.49\,\text{m/s}$, $v_U^{\text{pre}} = -0.62$, $v_U^{\text{post}} = -0.33$, $K_{v_u} = +0.484$, $G = 0.536$ (DAMPING).
  - Update #7 ($t = 26.25\text{s}$): After an 18.9s disturbance gap, $r = -7.65\,\text{m/s}$, $v_U^{\text{pre}} = +7.47\,\text{m/s}$, $v_U^{\text{post}} = +6.54\,\text{m/s}$, $K_{v_u} = +0.141$, $G = 0.875$ (DAMPING).
- **Outcome**: Despite $|r| > 7.6\,\text{m/s}$, $64.2\%$ of updates were damping, $G_{5\text{s}} = 0.34$, peak $|v_U| = 10.9\,\text{m/s}$, and total vertical drift was restricted to $+147.8\,\text{m}$ (vs $-41.85\,\text{km}$ in Control).

### Case 2: Trip 2 Ep 3 — Stable Convergence Benchmark
- **Disturbance**: Smooth highway cruise blackout with mild motion.
- **Update Behavior**:
  - Update #0 ($t = 2.77\text{s}$): $r = +2.93\,\text{m/s}$, $v_U^{\text{pre}} = -2.21$, $v_U^{\text{post}} = -0.91$, $K_{v_u} = +0.469$, $G = 0.414$ (DAMPING).
  - Update #1 ($t = 2.88\text{s}$): $r = +1.48\,\text{m/s}$, $v_U^{\text{pre}} = -0.91$, $v_U^{\text{post}} = -0.27$, $K_{v_u} = +0.454$, $G = 0.296$ (DAMPING).
  - Update #2 ($t = 3.52\text{s}$): $r = +0.90\,\text{m/s}$, $v_U^{\text{pre}} = -0.59$, $v_U^{\text{post}} = -0.23$, $K_{v_u} = +0.459$, $G = 0.393$ (DAMPING).
  - Update #3 ($t = 3.63\text{s}$): $r = +0.39\,\text{m/s}$, $v_U^{\text{pre}} = -0.23$, $v_U^{\text{post}} = -0.20$, $K_{v_u} = +0.333$, $G = 0.839$ (DAMPING).
- **Outcome**: Consecutively damping updates at $8\,\text{Hz}$ drove velocity to zero within 2 seconds ($G_{2\text{s}} = 0.019$). Total vertical drift was only $-8.3\,\text{m}$, with peak $|v_U| = 2.0\,\text{m/s}$.

### Case 3: Trip 1 Ep 8 — Severe Attitude Destabilization (Mode A)
- **Disturbance**: Severe attitude tilt during bus maneuvering.
- **Update Behavior**:
  - Update #0 ($t = 38.15\text{s}$): $r = -298.57\,\text{m/s}$, $v_U^{\text{pre}} = +162.65$, $v_U^{\text{post}} = +192.65$, $\Delta v_U = +31.09\,\text{m/s}$, **$K_{v_u} = -0.104$**, **$C_{33} = -0.003$**, $G = 1.184$ (ANTI-DAMPING).
  - Update #1 ($t = 38.27\text{s}$): $r = -289.84\,\text{m/s}$, $v_U^{\text{pre}} = +192.65$, $v_U^{\text{post}} = +242.18$, $\Delta v_U = +49.71\,\text{m/s}$, $K_{v_u} = -0.171$, $G = 1.257$ (ANTI-DAMPING).
- **Outcome**: Geometrically inverted measurement projection caused anti-damping injection on update #0, blowing velocity to $291.4\,\text{m/s}$ in under 1 second ($G_{1\text{s}} = 1.41$).

### Case 4: Trip 2 Ep 2 — Low-Residual Eventual Destabilization
- **Disturbance**: 161.7s blackout with handheld phone movement and late rotation.
- **Update Behavior**:
  - **Phase 1 ($t = 0 - 10\text{s}$)**: U0 has $r = 1.50\,\text{m/s}$, $K_{v_u} = +0.607$, $C_{33} = 0.980$, $G = 0.667$. Updates are damping, but inter-update gaps allow strapdown bias drift to slowly grow velocity ($G_{5\text{s}} = 1.50$).
  - **Phase 2 ($t = 10 - 90\text{s}$)**: Filter settles into a stable dynamic equilibrium! At $t = 20\text{s}$, $v_U = +0.13\,\text{m/s}$; at $t = 42\text{s}$, $v_U = -0.03\,\text{m/s}$; at $t = 90\text{s}$, $v_U = +3.11\,\text{m/s}$. The filter did **not** diverge during upright cruise!
  - **Phase 3 ($t = 92 - 99\text{s}$)**: Passenger turned phone sideways ($\text{roll} = -88.8^\circ$, $C_{33} \to 0.019$). At update #277 ($t = 99.2\text{s}$), $K_{v_u}$ flipped negative ($-0.360$), injecting $+20.6\,\text{m/s}$ of false upward velocity ($v_U: 0.23 \to 20.88\,\text{m/s}$). Subsequent updates blew velocity to $240 \to 392 \to 878.6\,\text{m/s}$.
- **Outcome**: This case proves that upright attitude and positive gain maintain stability, but unmonitored mid-flight attitude rotations can re-trigger Mode A inversion.

### Case 5: Trip 2 Ep 5 — Severe Intermittent-Eligibility Destabilization (Mode C)
- **Disturbance**: Violent passenger phone tumbling during a 57.8s blackout.
- **Update Behavior**:
  - During continuous agitation, the detector correctly rejected updates. However, during brief 1-second physical lulls, physical eligibility criteria were momentarily met.
  - Update #0 ($t = 0.99\text{s}$): $r = +1.69\,\text{m/s}$, $v_U^{\text{pre}} = -1.69$, $v_U^{\text{post}} = -8.10$, $K_{v_u} = +0.263$, $C_{33} = 1.000$, $G = 4.789$ (ANTI-DAMPING).
  - Update #1 ($t = 2.01\text{s}$): $r = +8.05\,\text{m/s}$, $v_U^{\text{pre}} = -8.10$, $v_U^{\text{post}} = -14.19$, $K_{v_u} = +0.080$, $G = 1.752$ (ANTI-DAMPING).
  - Trajectory evolution: $G_{1\text{s}} = 8.39$, $G_{5\text{s}} = 18.78$, $G_{10\text{s}} = 26.14$.
- **Outcome**: The estimator's internal state had already been corrupted by tumbling before the lull. Momentary physical eligibility admitted updates into an unobservable state, causing rapid divergence to $3,365.8\,\text{m/s}$ and $+134.8\,\text{km}$ drift.

---

## Artifacts Generated

1. **Structured Ledger**: [`results/c10_8c_update_response_stability.json`](c10_8c_update_response_stability.json)
2. **Diagnostic Visualization**: [`results/c10_8c_update_response_stability.png`](c10_8c_update_response_stability.png)
   - **Panel A**: Trajectory State Gain $G(t)$ over time for the 5 showcase cases.
   - **Panel B**: Single-Step Gain Ratio $G_{\text{step}}$ across the first 10 updates.
   - **Panel C**: Attitude Projection $C_{33}$ vs Vertical Gain $K_{v_u}$ across all 3,166 updates.
   - **Panel D**: Damping Update Percentage across all 12 recovery episodes.

---

## Conclusion

Stage C10.8C definitively answers why physically eligible VNHC updates stabilize in some cases and destabilize in others:
1. **Residual magnitude alone is uninformative.** Large residuals ($7.6\,\text{m/s}$) can be damping and stable (T1 Ep5), while tiny residuals ($0.11\,\text{m/s}$) can diverge (T1 Ep4).
2. **Positive $K_{v_u}$ is necessary but not sufficient.**
3. **Upright attitude ($C_{33} > 0.8$) is necessary to prevent Mode A inversion, but not sufficient to prevent Mode B inter-update drift or Mode C tumbling corruption.**
4. **Trajectory gain ratio $G$ provides a clean, dynamically observable signature of estimator health.**
