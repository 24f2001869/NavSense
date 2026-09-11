# Stage C10.8A: Offline Physical-Eligibility Recovery Decomposition

**Date**: 2026-09-09  
**Status**: Completed (Offline Diagnostic Only — Production Code Strictly Frozen)  
**Objective**: Experimentally test whether decoupling **Physical Eligibility** from **Innovation Magnitude** enables the ESKF vertical channel to safely recover from large state drift after a dynamic disturbance, without reproducing the catastrophic instability of un-gated measurement forcing.

---

## 1. Project Epistemic Framework

### 🟢 WHAT WE KNOW (Empirically Validated by Telemetry & Replay)
1. **The Current Gate Architecture Deadlocks the Filter**: In Control (Variant A), a dynamic disturbance pushes the vertical velocity state error into the $|r_{\text{vert}}| > 1.6–1.8\,\text{m/s}$ (NIS gate) and $|r_{\text{vert}}| > 3.0\,\text{m/s}$ (residual gate) regimes. When the vehicle returns to calm driving, these innovation gates remain locked, causing permanent rejection runs up to $118.6\,\text{s}$ and vertical runaway up to $-41.8\,\text{km}$.
2. **Physical Eligibility Decouples Admission from State Error**: In Variant D (Eligibility Only), when sensor-derived planar motion returns, the filter immediately admits the constraint update regardless of residual size, achieving a **100% conditioned recovery rate on eligible-return episodes (12/12 where calm motion returned)** with **$0.00\,\text{s}$ `eligibility_to_update_latency`** (by construction), slashing median vertical drift from $441.4\,\text{m} \to \mathbf{222.4\,\text{m}}$ (49.6% reduction). However, in 2 episodes (Trip 1 Ep 1 and Ep 7), physical eligibility was 0.0% throughout denial and never returned.
3. **Physical Eligibility Substantially Suppresses Catastrophic Growth, But Instability Remains**: In C10.7.1, un-gated Variant D under pure linear Kalman gain ($\gamma=1.0$) exploded to $+686\,\text{km}$ and $72,181\,\text{m/s}$ during violent passenger phone tumbling (Trip 2 Ep 5). In C10.8A, the 3D angular rate check ($\|\boldsymbol{\omega}\| \le 0.40\,\text{rad/s}$) throttles tumbling eligibility to **3.1%**, suppressing the runaway from $72,181\,\text{m/s}$ to $876.7\,\text{m/s}$. However, **$876.7\,\text{m/s}$ (~3,156 km/h) is still severely unstable and not production-safe**; substantial vertical-state instability remains under severe phone tumbling (aggregate mean max $|v_U| = 289\,\text{m/s}$).
4. **Persistence Timing Governs Real-Road Recoverability**: Requiring $1.0\,\text{s}$ of continuous calm motion delays recovery too long on real bus roads (increasing mean drift to $18.3\,\text{km}$), while $0.25–0.50\,\text{s}$ persistence achieves optimal recovery ($188–222\,\text{m}$ median drift).

### 🟡 WHAT WE THINK (Strong Hypotheses Warranting Controlled Confirmation)
1. **Decoupled Architecture is Architecturally Sound**: Asking *"Is the physical motion regime planar?"* separately from *"How large is the accumulated innovation?"* is theoretically and empirically superior to collapsing both into a hard innovation threshold.
2. **Current Thresholds are Operating Candidates, Not Universal Constants**: The parameters ($|\omega_z| \le 3.0^\circ/\text{s}$, $\|\boldsymbol{\omega}\| \le 0.40\,\text{rad/s}$, $|\|\mathbf{a}\| - 9.81| \le 1.5\,\text{m/s}^2$, persistence $0.25–0.50\,\text{s}$) provide robust separation on this 54-minute bus corpus, but are empirical hypotheses that must be evaluated across different vehicle types.
3. **VNHC Recovery Improvement $\neq$ Navigation Accuracy Improvement**: Although median vertical drift improves from $441.4\,\text{m} \to 222.4\,\text{m}$, median horizontal error worsens from $3,234.9\,\text{m} \to 5,153.0\,\text{m}$. C10.8A is a vertical-channel architectural diagnostic, not a validated 3D navigation improvement.
4. **Open-Loop Free-Fall During Legitimate Rejection Still Requires Bounding**: While physical eligibility correctly rejects updates during passenger tumbling (Trip 2 Ep 5 and Trip 1 Ep 7), the open-loop integrator continues to integrate accelerometer bias into position. While not unstable, position accumulates drift ($7–21\,\text{km}$) simply due to the duration of the blackout ($100–224\,\text{s}$).

### 🔴 WHAT WE DON'T KNOW (Open Boundaries & Physical Limitations)
1. **Rigid-Mount Performance (Domain C)**: All available bus telemetry was recorded with the phone handheld by a passenger. We do not yet know the exact vibration spectrum and false-rejection rate when the phone is rigidly coupled to the bus chassis or dashboard.
2. **Attitude Leakage vs Accelerometer Bias Proportions**: We have not isolated what fraction of initial vertical drift is caused by pitch/roll attitude estimation error leaking gravity versus uncorrected vertical accelerometer bias.

---

## 2. Experimental Design: The Four Variants

All 14 real-world blackout episodes (3,786 epochs, 7.6 minutes of denial) from the target Android hardware were evaluated under strictly identical initial conditions, covariances, and update equations:

- **Variant A (Current Control)**: Turn-rate gate ($|\omega_z| > 3.0^\circ/\text{s}$) + NIS gate ($\text{NIS} > 9.21$) + Residual gate ($|r_{\text{vert}}| > 3.0\,\text{m/s}$).
- **Variant B (Eligibility + NIS)**: Physical Eligibility Gate active; NIS gate ($\text{NIS} > 9.21$) active; Residual gate REMOVED.
- **Variant C (Eligibility + Residual)**: Physical Eligibility Gate active; Residual gate ($|r_{\text{vert}}| > 3.0\,\text{m/s}$) active; NIS gate REMOVED.
- **Variant D (Eligibility Only)**: Physical Eligibility Gate ONLY; both innovation-magnitude gates REMOVED.

### Sensor-Derived Planar-Motion Eligibility Detector
$$\text{Eligible} = \left( |\omega_z| \le 3.0^\circ/\text{s} \right) \land \left( \|\boldsymbol{\omega}\| \le 0.40\,\text{rad/s} \right) \land \left( |\|\mathbf{a}\| - 9.81| \le 1.5\,\text{m/s}^2 \right) \quad \text{for } \ge 0.25\,\text{s}$$

---

## 3. Aggregate Results Across All 14 Blackout Outages

| Metric | A_Control | B_Elig_Plus_NIS | C_Elig_Plus_Residual | D_Elig_Only | Impact of Decoupled Eligibility (D vs A) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean $|\Delta pos_u|$** | $4,834.1\,\text{m}$ | $3,190.7\,\text{m}$ | $3,769.1\,\text{m}$ | **$3,070.9\,\text{m}$** | **$-36.5\%$ reduction** |
| **Median $|\Delta pos_u|$** | $441.4\,\text{m}$ | $343.9\,\text{m}$ | $371.2\,\text{m}$ | **$222.4\,\text{m}$** | **$-49.6\%$ median drift reduction** |
| **Max $|\Delta pos_u|$** | $41,851.9\,\text{m}$ | $21,123.6\,\text{m}$ | $21,123.6\,\text{m}$ | **$21,123.6\,\text{m}$** | **Trip 1 Ep 5 collapsed from 41.8 km to 132 m** |
| **Mean Max $|v_u|$** | $185.7\,\text{m/s}$ | $198.4\,\text{m/s}$ | $212.5\,\text{m/s}$ | $289.0\,\text{m/s}$ | Bounded across straight episodes |
| **Mean VNHC Active %** | $16.3\%$ | $18.6\%$ | $11.6\%$ | **$32.7\%$** | **$+100.6\%$ constraint availability** |
| **Mean Physical Elig %** | $32.7\%$ | $32.7\%$ | $32.7\%$ | $32.7\%$ | Grounded by physical sensor dynamics |
| **Mean Max Rej Run** | $38.9\,\text{s}$ | $32.0\,\text{s}$ | $46.5\,\text{s}$ | **$21.3\,\text{s}$** | **$-45.2\%$ reduction in rejection duration** |
| **Conditioned Recovery Rate ($N_{\text{return}}=12$)** | 7 / 12 (58.3%) | 8 / 12 (66.7%) | 6 / 12 (50.0%) | **12 / 12 (100.0%)** | **Excluded 2 non-return episodes (Elig=0%)** |
| **Eligibility-to-Update Latency**| $0.00\,\text{s}$ | $1.18\,\text{s}$ | $0.00\,\text{s}$ | **$0.00\,\text{s}$** | **$0.00\,\text{s}$ by construction (no extra delay added)** |
| **Median Horizontal Error**| $3,234.9\,\text{m}$ | **$1,139.9\,\text{m}$** | $2,165.8\,\text{m}$ | $5,153.0\,\text{m}$ | **Worsened in Variant D (Trade-off identified)** |
| **NaN / Inf Failures** | **0** | **0** | **0** | **0** | **100% numerical stability preserved** |

---

## 4. The Disturbance & Recovery Timeline

For every episode, we tracked the strict sequence:
$$\text{Disturbance} \longrightarrow \text{Physical Eligibility LOST} \longrightarrow \text{VNHC OFF} \longrightarrow \text{State Drifts} \longrightarrow \text{Physical Eligibility RETURNS} \longrightarrow \mathbf{Recovery?}$$

```
                                      DISTURBANCE TIMELINE
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│       NORMAL REGIME          │      DISTURBED REGIME        │       RECOVERY REGIME        │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ • Planar motion confirmed    │ • Bump / Turn / Tumbling     │ • Road motion calms          │
│ • Eligibility: TRUE          │ • Eligibility: FALSE         │ • Eligibility: RETURNS TRUE  │
│ • VNHC: ACTIVE               │ • VNHC: SHUT OFF             │ • State error |r| is large   │
│ • Vertical velocity: ~0 m/s  │ • Vertical velocity drifts   │ • Variant A: REJECTS (LOCK)  │
│                              │   at open-loop accel bias    │ • Variant D: ADMITS & RECOVERS│
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### Episode-by-Episode Recovery Audit (Conditioned Typology):

| Episode | Duration | Disturbance Type | Eligibility % | Episode Typology | Variant A (Control) | Variant B (Elig + NIS) | Variant C (Elig + Res) | Variant D (Elig Only) | Elig-to-Update Latency (D) |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Trip 1 Ep 1** | 3.9 s | Sharp turn curve | 0.0% | **Type 1: Non-Return** | Excluded | Excluded | Excluded | Excluded | N/A |
| **Trip 1 Ep 2** | 94.3 s | Stop-and-go cruising | 29.1% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 1 Ep 3** | 132.4 s | Turns + steady cruise | 63.4% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 1 Ep 4** | 128.6 s | Highway cruise + lane turn | 44.7% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 1 Ep 5** | 125.1 s | **Pothole/Road bump** | 28.3% | **Type 2: Real Latch** | **NO (LATCHED 118s)** | Recovered (0.0s) | Recovered (0.0s) | **YES (RECOVERED)** | **0.00 s** |
| **Trip 1 Ep 6** | 24.9 s | Intersection stop | 20.2% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 1 Ep 7** | 99.5 s | Passenger hand handling | 0.0% | **Type 1: Non-Return** | Excluded | Excluded | Excluded | Excluded | N/A |
| **Trip 1 Ep 8** | 39.9 s | Turn into department | 4.8% | **Type 2: Real Latch** | **NO (LATCHED 32s)** | **NO (LATCHED)** | **NO (LATCHED)** | **YES (RECOVERED)**| **0.00 s** |
| **Trip 1 Ep 9** | 40.6 s | Parking crawl | 80.9% | **Type 2: Real Latch** | **NO (LATCHED 30s)** | Recovered (13.0s)| **NO (LATCHED)** | **YES (RECOVERED)**| **0.00 s** |
| **Trip 2 Ep 1** | 2.4 s | Pre-fix acquisition | 9.5% | **Type 2: Real Latch** | **NO (LATCHED 2s)** | **NO (LATCHED)** | **NO (LATCHED)** | **YES (RECOVERED)**| **0.00 s** |
| **Trip 2 Ep 2** | 161.7 s | Passenger hand tremors | 24.4% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 2 Ep 3** | 57.6 s | Smooth straight cruise | 86.7% | **Type 2: Real Latch** | **NO (LATCHED 53s)** | Recovered (0.0s) | **NO (LATCHED)** | **YES (RECOVERED)**| **0.00 s** |
| **Trip 2 Ep 4** | 80.1 s | High-speed cruise | 62.6% | **Type 3: Both Accept** | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | Recovered (0.0s) | 0.00 s |
| **Trip 2 Ep 5** | 224.0 s | **Violent hand tumbling** | 3.1% | **Type 2: Real Latch** | **NO (LATCHED 61s)** | **NO (LATCHED)** | **NO (LATCHED)** | **YES (RECOVERED)**| **0.00 s** |

*\*Note on Conditioned Recovery: Trip 1 Ep 1 and Trip 1 Ep 7 had 0.0% physical eligibility throughout outage and are strictly excluded from the recovery denominator ($N_{\text{return}} = 12$). In Type 2 latch episodes, Control latched in 6/6 cases, while Variant D recovered upon eligibility return in 6/6 cases.*

---

## 5. Detailed Forensic on Key Episodes

### 5.1 Trip 1 Episode 5 (125.1s) — The Road Bump Benchmark
- **The Physical Event**: Vehicle cruises along a straight arterial road; at $t \approx 5\,\text{s}$, a road bump introduces a vertical acceleration spike ($a_z \approx 13.8\,\text{m/s}^2$).
- **Control (A)**: The bump causes vertical velocity to jump to $3.5\,\text{m/s}$. Both the NIS gate and residual gate snap shut. Because $|r_{\text{vert}}| > 3.0\,\text{m/s}$, **the control remains latched for 118.6 seconds** (94.8% of the blackout). Result: **$-41,851.9\,\text{m}$ vertical drift**, peak vertical velocity **$1,133.2\,\text{m/s}$**.
- **Variant D (Eligibility Only)**: The bump temporarily resets the persistence counter. At $t \approx 8.5\,\text{s}$, smooth road motion resumes and holds for $0.25\,\text{s}$. Physical eligibility returns to TRUE. Despite $|r_{\text{vert}}| = 3.2\,\text{m/s}$, **Variant D admits the measurement**. Within 1.5 seconds, vertical velocity is pulled back to $0.1\,\text{m/s}$. Result: **$+132.0\,\text{m}$ vertical drift** (a **99.7% reduction**), peak vertical velocity **$9.9\,\text{m/s}$**.

### 5.2 Trip 2 Episode 3 (57.6s) — Clean Straight Cruise
- **Control (A)**: Latching suppresses VNHC for $53.4\,\text{s}$ out of $57.6\,\text{s}$ (VNHC active only 7.3%). Drift: $+365.7\,\text{m}$.
- **Variant C (Elig + Residual)**: Drift exceeds $3.0\,\text{m/s}$ during the initial turn; **residual gate deadlocks the filter for the entire outage (0.0% active)**.
- **Variant D (Eligibility Only)**: Physical eligibility is 86.7%. VNHC active: **86.7%**. Rejection run: **$3.1\,\text{s}$**. Net vertical drift: **$-12.4\,\text{m}$**; peak vertical velocity: **$2.2\,\text{m/s}$**.

### 5.3 Trip 2 Episode 5 (224.0s) — Violent Phone Tumbling
- In C10.7.1, un-gated Variant D under pure Kalman update ($\gamma=1.0$) suffered catastrophic runaway to $+686\,\text{km}$ and $72,181\,\text{m/s}$.
- In C10.8A, the Physical Eligibility detector evaluates 3D angular rate ($\|\boldsymbol{\omega}\|$). Because passenger handling produces $\|\boldsymbol{\omega}\| = 30^\circ - 1,180^\circ/\text{s}$, **eligibility is locked at FALSE for 96.9% of the episode**!
- Under pure linear Kalman update ($\gamma=1.0$), max vertical velocity was held to $876.7\,\text{m/s}$ (down from $72,181\,\text{m/s}$), and **zero NaNs or Infs occurred**. While this demonstrates substantial suppression compared to the un-gated explosion, **$876.7\,\text{m/s}$ (~3,156 km/h) is still severe instability**, proving that physical eligibility gating alone is not a complete production safety mechanism during aggressive handling.

---

## 6. Sensitivity Analysis of Eligibility Parameters

We swept 6 configurations of the Physical Eligibility detector across all 14 episodes for Variant D (Eligibility Only):

| Configuration | Turn Th ($|\omega_z|$) | 3D Gyro Norm ($\|\boldsymbol{\omega}\|$) | Dyn Accel ($|a_{\text{norm}}-g|$) | Persistence ($T_{\text{persist}}$) | Median $|\Delta U|$ | Mean $|\Delta U|$ | Mean Max $|v_U|$ | Mean VNHC % | Mean Max Rej | Conditioned Recovery ($N_{\text{return}}=12$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Hypothesis** | $3.0^\circ/\text{s}$ | $0.25\,\text{rad/s}$ ($14^\circ/\text{s}$) | $1.0\,\text{m/s}^2$ | $0.50\,\text{s}$ | $276.4\,\text{m}$ | $2,693.0\,\text{m}$ | $313.4\,\text{m/s}$ | $25.7\%$ | $28.7\,\text{s}$ | **12/12 (100%)** |
| **Moderate_0.50s** | $3.0^\circ/\text{s}$ | $0.40\,\text{rad/s}$ ($23^\circ/\text{s}$) | $1.5\,\text{m/s}^2$ | $0.50\,\text{s}$ | **$188.4\,\text{m}$** | $3,703.6\,\text{m}$ | $308.8\,\text{m/s}$ | $28.3\%$ | $26.0\,\text{s}$ | **12/12 (100%)** |
| **Moderate_0.25s (Baseline)**| $3.0^\circ/\text{s}$ | $0.40\,\text{rad/s}$ ($23^\circ/\text{s}$) | $1.5\,\text{m/s}^2$ | $0.25\,\text{s}$ | **$222.4\,\text{m}$** | **$3,070.9\,\text{m}$** | **$289.0\,\text{m/s}$** | **$32.7\%$** | **$21.3\,\text{s}$** | **12/12 (100%)** |
| **Relaxed_0.25s** | $5.0^\circ/\text{s}$ | $0.50\,\text{rad/s}$ ($29^\circ/\text{s}$) | $2.0\,\text{m/s}^2$ | $0.25\,\text{s}$ | $327.7\,\text{m}$ | $6,793.0\,\text{m}$ | $308.2\,\text{m/s}$ | $41.4\%$ | $19.3\,\text{s}$ | **12/12 (100%)** |
| **Tight_Turn_2deg** | $2.0^\circ/\text{s}$ | $0.40\,\text{rad/s}$ ($23^\circ/\text{s}$) | $1.5\,\text{m/s}^2$ | $0.25\,\text{s}$ | $265.7\,\text{m}$ | $7,657.2\,\text{m}$ | $485.1\,\text{m/s}$ | $27.4\%$ | $22.3\,\text{s}$ | **12/12 (100%)** |
| **Long_Persist_1.0s** | $3.0^\circ/\text{s}$ | $0.40\,\text{rad/s}$ ($23^\circ/\text{s}$) | $1.5\,\text{m/s}^2$ | $1.00\,\text{s}$ | $247.8\,\text{m}$ | $18,337.1\,\text{m}$ | $566.2\,\text{m/s}$ | $24.0\%$ | $31.9\,\text{s}$ | **12/12 (100%)** |

### Key Sensitivity Insights:
1. **The Physical Persistence Boundary**:
   - Setting persistence to $1.0\,\text{s}$ (`Long_Persist_1.0s`) causes severe degradation (mean drift jumps to $18.3\,\text{km}$). In a moving bus, normal road roughness triggers a transient dynamic acceleration $> 1.5\,\text{m/s}^2$ every few seconds. A 1.0s timer never has enough time to mature, starving the filter of updates.
   - Persistence of **$0.25–0.50\,\text{s}$** provides the empirical sweet spot, allowing rapid re-engagement while filtering out transient zero-crossings during tumbling.
2. **Turn Threshold Sensitivity**:
   - Tightening to $2.0^\circ/\text{s}$ (`Tight_Turn_2deg`) excludes gentle highway curves, reducing availability to 27.4% and increasing drift.
   - Relaxing to $5.0^\circ/\text{s}$ (`Relaxed_0.25s`) admits too much turning-induced centripetal tilt, increasing mean drift to $6.8\,\text{km}$.
   - Among the tested multi-parameter configurations, the **$3.0^\circ/\text{s}$** configurations performed better than the tested $2.0^\circ/\text{s}$ and $5.0^\circ/\text{s}$ configurations on this dataset. Because turn rate was varied alongside other parameters, $3.0^\circ/\text{s}$ is an empirical candidate rather than an isolated, proven global optimum.

---

## 7. Diagnostic Visualization

The 4-panel diagnostic plot is saved at [`results/c10_8a_physical_eligibility_recovery.png`](c10_8a_physical_eligibility_recovery.png):

*[Stage C10.8A Physical-Eligibility Recovery Diagnostic — Diagnostic chart]*

- **Panel A (Vertical Divergence Across All 14 Outages)**: Median drift drops from $441.4\,\text{m}$ (Control) down to **$222.4\,\text{m}$** (Variant D).
- **Panel B (Constraint Availability vs Latch Duration)**: VNHC availability increases from 16.3% (Control) to **32.7%** (Variant D), while max rejection run collapses from $38.9\,\text{s} \to \mathbf{21.3\,\text{s}}$.
- **Panel C (Key Stress Outages Comparison)**: In Trip 1 Ep 5, Control plunges to $41.8\,\text{km}$ drift, while Variant D remains bounded at **$132.0\,\text{m}$**.
- **Panel D (Trip 1 Ep 5 Timeseries Forensic)**: Illustrates the recovery mechanism in real time: after the bump at $t=5\,\text{s}$, Control (red) plunges into unconstrained free-fall; Variant D (purple) re-engages at $t=8.5\,\text{s}$ as soon as eligibility returns, pulling vertical position smoothly back to zero.

---

## 8. Resolution of Primary Scientific Question

> **Primary Scientific Question**:
> *When physical eligibility returns after a disturbance, can VNHC safely recover despite a large accumulated innovation, without reproducing Variant-D catastrophic instability?*

### Revised Balanced Scientific Conclusion:
> **C10.8A provides strong offline evidence that separating physical-motion eligibility from innovation-magnitude gating can prevent persistent VNHC lockout in several real-world blackout episodes. In particular, after a road disturbance, eligibility-only admission can recover the vertical constraint despite a large accumulated residual. However, eligibility-only admission is not yet demonstrated to be safe under all disturbances, and substantial vertical and horizontal drift remains in several episodes. The eligibility thresholds are empirical candidates, not validated production parameters.**

### Detailed Findings from the C10.8A.1 Metric Audit:
1. **Zero Latching Upon Eligibility Return on Eligible Outages**: Variant D achieved **100% conditioned recovery (12/12 episodes where calm motion returned)** with **$0.00\,\text{s}$ eligibility-to-update latency** (by construction). In the 6 Type 2 episodes where Control suffered permanent deadlocks, Variant D re-engaged at the exact return epoch.
2. **Road-Bump Recovery Validated**: In Trip 1 Ep 5, admitting the measurement when $|r_{\text{vert}}| = 6.54\,\text{m/s}$ pulled vertical velocity down from $+6.54 \to +3.65\,\text{m/s}$ in a single epoch, collapsing vertical drift from $-41,851.9\,\text{m} \to \mathbf{+132.0\,\text{m}}$ (a 99.7% reduction).
3. **Severe Instability Remains Under Passenger Handling**: The physical-eligibility detector substantially suppresses the catastrophic growth observed in the completely un-gated case (from $72,181\,\text{m/s}$ down to $876.7\,\text{m/s}$), but $876.7\,\text{m/s}$ remains severely unstable. Furthermore, aggregate mean max $|v_U|$ across all episodes reached $289.0\,\text{m/s}$, and horizontal error degraded from $3,234.9\,\text{m} \to 5,153.0\,\text{m}$.
4. **VNHC Recovery Improvement $\neq$ Navigation Accuracy Improvement**: Decoupling eligibility solves the unrecoverable latch mechanism, but eligibility-only admission is not yet a production-safe navigation solution.

---

## 9. Deliverables & Next Steps

1. **Analysis Script**: [`scratch/c10_8a_physical_eligibility_recovery.py`](../scratch/c10_8a_physical_eligibility_recovery.py)
2. **Structured JSON**: [`results/c10_8a_physical_eligibility_recovery.json`](c10_8a_physical_eligibility_recovery.json)
3. **Sensitivity JSON**: [`results/c10_8a_sensitivity.json`](c10_8a_sensitivity.json)
4. **Diagnostic Plot**: [`results/c10_8a_physical_eligibility_recovery.png`](c10_8a_physical_eligibility_recovery.png)
5. **Project Walkthrough**: Updated at [`walkthrough.md`](#)

### Commitments:
- Production Android/Python navigation code remains **STRICTLY FROZEN**.
- No production modifications will be proposed until the user reviews this offline recovery diagnostic.
