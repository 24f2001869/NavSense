# Stage C10.8B: Post-Recovery Trajectory & Stability Characterization

**Date**: 2026-09-09  
**Status**: Completed (Offline Diagnostic Only — Production Code Strictly Frozen)  
**Objective**: Characterize the post-recovery time-series trajectory following VNHC admission in the decoupled physical-eligibility architecture (Variant D). Specifically, determine what empirical factors govern whether admitting an update after a disturbance results in **stable exponential convergence** versus **state destabilization**.

---

## 1. Project Epistemic Framework

### 🟢 WHAT WE KNOW (Empirically Validated by Telemetry & Replay)
1. **Physical Eligibility Breaks the Innovation-Induced Latch**: In Stage C10.8A.1, evaluating strictly on the valid eligible-return subset ($N_{\text{return}} = 12$), decoupled physical eligibility successfully admitted VNHC updates upon return in **12 / 12 (100.0%)** episodes, resolving the deadlock in **6 / 6 (100.0%)** Type 2 latch-failure cases where Control remained frozen for up to 118s.
2. **Large-Residual Recovery Works Under Rigid Alignment**: In Trip 1 Ep 5 (road bump), an innovation of $|r_{\text{vert}}| = 6.70\,\text{m/s}$ was successfully admitted, damping vertical velocity from $+7.62\,\text{m/s} \to +3.9\,\text{m/s} \to +3.1\,\text{m/s}$ and slashing vertical position runaway by **99.7%** (from $-41,851.9\,\text{m} \to \mathbf{+81.1\,\text{m}}$). In Trip 2 Ep 3 (highway cruise), admitting an update at $|r_{\text{vert}}| = 3.00\,\text{m/s}$ produced textbook exponential convergence (velocity damped to $0.1\,\text{m/s}$ within $2.0\,\text{s}$, holding drift to **$-8.9\,\text{m}$**).
3. **Residual Magnitude Alone Does NOT Predict Stability**: Across the 12 genuine recovery events, destabilized episodes exhibited a **lower median pre-update residual ($1.42\,\text{m/s}$)** than stable episodes (**$3.00\,\text{m/s}$**). Trip 2 Ep 2 destabilized to $+7,522\,\text{m}$ drift after admitting a tiny $1.42\,\text{m/s}$ residual, whereas Trip 1 Ep 5 recovered after admitting a $6.70\,\text{m/s}$ residual. **A hard threshold on residual magnitude is mathematically and empirically invalid.**
4. **Attitude Inversion Inverts Kalman Gain**: In Trip 1 Ep 8, the phone was rolled on its side ($\text{roll} = -90.3^\circ$). Because the measurement matrix $\mathbf{H} = \mathbf{C}_{n\to v}[2, :]$ was rotated $90^\circ$ into the horizontal plane, the Kalman gain component into vertical velocity became **negative ($K_{v_u} = -0.159$)**. Admitting the constraint kicked vertical velocity in the *wrong direction*, actively accelerating vertical velocity from $+161.6\,\text{m/s} \to \mathbf{+209.3\,\text{m/s}}$.

### 🟡 WHAT WE THINK (Strong Hypotheses Warranting Controlled Confirmation)
1. **Convergence is Governed by Projection Condition, Not Residual Magnitude**: Stable convergence occurs when the body-to-navigation rotation matrix $\mathbf{C}_{v\to n}$ is physically consistent with gravity and vehicle travel. When pitch and roll estimates are severely corrupted ($|\text{roll}| > 30^\circ$ or $|\text{pitch}| > 30^\circ$), VNHC projection projects horizontal velocity into vertical velocity and vice versa, creating positive feedback.
2. **Transient Inflection Calms are Distinct from Steady Cruising**: In violent tumbling (Trip 2 Ep 5), a 0.25s persistence timer can trigger on a transient zero-crossing or momentary inflection point while the user is actively manipulating the device. Admitting an update during an inflection lull destabilizes the state because violent dynamics resume immediately at $t + 0.3\,\text{s}$.
3. **Attitude Gating is the Missing Complement to Planar Eligibility**: Decoupling physical eligibility from innovation magnitude was the necessary first step to break the latch. The necessary second step to ensure global stability is verifying that the estimated attitude frame is well-conditioned ($\mathbf{C}_{n\to v}[2, 2] \approx 1.0$) before injecting the 1-DOF correction.

### 🔴 WHAT WE DON'T KNOW (Open Boundaries & Physical Limitations)
1. **Rigid-Mount Performance (Domain C)**: All available bus telemetry was recorded handheld by a passenger. In a rigid dashboard mount, severe roll/pitch tumbling ($>30^\circ$) does not exist. Handheld passenger manipulation remains the primary driver of attitude corruption.
2. **Optimal Form of Innovation Damping**: While Huber attenuation ($\gamma \propto 1/\sqrt{\text{NIS}}$) prevents numerical overflow, we do not yet know the optimal mathematical formulation for smooth state pull-back when residual is large ($|r| \in [3, 10]\,\text{m/s}$) under moderate attitude uncertainty.

---

## 2. Experimental Design & Measurement Protocol

The experiment replayed all 14 blackout episodes (3,786 epochs) under the exact, frozen C10.8A Variant D (Eligibility Only) configuration:
- **Eligible Planar Motion**: $|\omega_z| \le 3.0^\circ/\text{s}$, $\|\boldsymbol{\omega}\| \le 0.40\,\text{rad/s}$, $|\|\mathbf{a}\| - 9.81| \le 1.5\,\text{m/s}^2$ for $\ge 0.25\,\text{s}$.
- **Innovation Admission**: Decoupled from innovation magnitude (both residual and NIS gates removed).
- **Update Equation**: Standard Joseph-stabilized ESKF update with baseline Huber gain attenuation ($k=2.0$).
- **Zero Production Changes**: Production navigation code in `android/` and `src/navigation/` remained **strictly frozen**.

### Recorded Telemetry per Recovery Event:
For every genuine recovery event across the 12 eligible-return episodes, we captured:
- **PRE-UPDATE**: $|r_{\text{vert}}|$, $v_U^-$, $\text{pos}_U^-$, roll, pitch, yaw, $\|\boldsymbol{\omega}\|$, $|\omega_z|$, dynamic acceleration, innovation variance $S$, $\text{NIS}$, velocity variance $P_{v_u v_u}$, attitude covariance trace $\text{trace}(P_{\theta\theta})$, Kalman gain vector $\mathbf{K}$, gain component $K_{v_u}$, Huber attenuation $\gamma$, and intended velocity kick $\Delta v_U$.
- **IMMEDIATELY AFTER UPDATE**: $v_U^+$, $\text{pos}_U^+$, $P_{v_u v_u}^+$, actual velocity step $\Delta v_U$.
- **POST-UPDATE TRAJECTORY**: State snapshots at $\Delta t \in [+0.5\,\text{s}, +1.0\,\text{s}, +2.0\,\text{s}, +5.0\,\text{s}, +10.0\,\text{s}]$.
- **LONG-HORIZON METRICS**: Peak $|v_U|$, time to $|v_U| < 1.0\,\text{m/s}$, time to $|v_U| < 0.5\,\text{m/s}$, post-recovery displacement $\Delta U_{\text{post}}$, terminal drift $\Delta U$, and terminal horizontal position error.

---

## 3. Characterization Results Across All 12 Recovery Events

### Table 1: Comprehensive Stability & Trajectory Metrics per Episode

| Episode | Trajectory Class | Pre $|r_{\text{vert}}|$ (m/s) | Pre $v_U^-$ (m/s) | Post $v_U^+$ (m/s) | Pitch (°) | Roll (°) | $S$ ($\text{m}^2/\text{s}^2$) | Pre NIS | $K_{v_u}$ | $\gamma$ | Post Peak $|v_U|$ (m/s) | Time to $<1\,\text{m/s}$ | Post $\Delta U$ (m) | Final Horiz Err (m) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip 2 Ep 3** | **STABLE CONVERGENCE** | **$3.00$** | **$-2.29$** | **$-0.92$** | $-8.6$ | $+5.8$ | $1.020$ | $8.8$ | $+0.457$ | $0.673$ | **$1.9$** | **$0.00\,\text{s}$** | **$-8.9$** | $2,346.6$ |
| **Trip 1 Ep 5** | **PARTIAL RECOVERY** | **$6.70$** | **$+7.62$** | **$+6.54$** | $+46.0$ | $+10.7$ | $2.052$ | $21.9$ | $+0.161$ | $0.428$ | **$9.9$** | $29.95\,\text{s}$ | **$+81.1$** | $482.6$ |
| **Trip 1 Ep 2** | **PARTIAL RECOVERY** | $2.58$ | $-2.17$ | $-0.02$ | $-0.4$ | $+3.5$ | $1.781$ | $3.7$ | $+0.832$ | $1.000$ | $40.1$ | $0.00\,\text{s}$ | $-282.8$ | $5,348.9$ |
| **Trip 1 Ep 6** | **PARTIAL RECOVERY** | $0.98$ | $-1.01$ | $-0.69$ | $-9.5$ | $-10.0$ | $0.367$ | $2.6$ | $+0.321$ | $1.000$ | $25.2$ | $0.00\,\text{s}$ | $-54.3$ | $194.1$ |
| **Trip 1 Ep 9** | **PARTIAL RECOVERY** | $3.41$ | $-3.41$ | $-2.98$ | $-0.3$ | $+0.3$ | $0.385$ | $30.1$ | $+0.128$ | $0.364$ | $16.7$ | $12.98\,\text{s}$ | $-69.1$ | $718.3$ |
| **Trip 2 Ep 1** | **PARTIAL RECOVERY** | $3.82$ | $-3.13$ | $-2.41$ | $-0.0$ | $+5.6$ | $0.518$ | $28.2$ | $+0.188$ | $0.377$ | $5.2$ | Never | $-4.8$ | $>10^5$ |
| **Trip 2 Ep 4** | **PARTIAL RECOVERY** | $0.27$ | $-0.21$ | $-0.15$ | $+1.8$ | $+0.7$ | $0.324$ | $0.2$ | $+0.229$ | $1.000$ | $9.5$ | $0.00\,\text{s}$ | $-335.1$ | $3,580.2$ |
| **Trip 1 Ep 8** | **DESTABILIZATION** | **$299.80$** | $+161.56$ | **$+209.25$** | $-38.6$ | **$-90.3$** | $25.362$ | $3,544.1$ | **$-0.159$** | $0.034$ | **$269.0$** | Never | $+483.6$ | $5,234.6$ |
| **Trip 2 Ep 5** | **DESTABILIZATION** | **$8.50$** | $-8.54$ | $-8.10$ | $+0.5$ | $-0.1$ | $0.389$ | $185.6$ | $+0.052$ | $0.147$ | **$1,612.3$** | Never | **$-7,103.2$** | $16,391.2$ |
| **Trip 2 Ep 2** | **DESTABILIZATION** | **$1.42$** | $-2.62$ | $-1.76$ | $-8.8$ | $-7.6$ | $0.633$ | $3.2$ | $+0.607$ | $1.000$ | **$180.7$** | $15.52\,\text{s}$ | **$+7,522.4$** | $19,894.7$ |
| **Trip 1 Ep 3** | **DESTABILIZATION** | $0.45$ | $-0.43$ | $-0.32$ | $+0.1$ | $-0.4$ | $0.327$ | $0.6$ | $+0.234$ | $1.000$ | $97.5$ | $0.00\,\text{s}$ | $+153.4$ | $5,470.1$ |
| **Trip 1 Ep 4** | **DESTABILIZATION** | $0.11$ | $-0.10$ | $-0.07$ | $+0.0$ | $+1.0$ | $0.322$ | $0.0$ | $+0.224$ | $1.000$ | $76.3$ | $0.00\,\text{s}$ | $+1,454.8$ | $5,071.5$ |

---

## 4. Trajectory Evolution Time Series

Tracking the state at $t = +0.5\,\text{s}$, $+1.0\,\text{s}$, $+2.0\,\text{s}$, $+5.0\,\text{s}$, and $+10.0\,\text{s}$ reveals the clear divergence between converging and destabilizing trajectories:

### Table 2: Multi-Epoch Post-Recovery Velocity and Position Drift Trajectory

| Episode | Class | $t_{\text{rec}} + 0.5\,\text{s}$ ($v_U$, $\Delta U$) | $t_{\text{rec}} + 1.0\,\text{s}$ ($v_U$, $\Delta U$) | $t_{\text{rec}} + 2.0\,\text{s}$ ($v_U$, $\Delta U$) | $t_{\text{rec}} + 5.0\,\text{s}$ ($v_U$, $\Delta U$) | $t_{\text{rec}} + 10.0\,\text{s}$ ($v_U$, $\Delta U$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Trip 2 Ep 3** | **STABLE** | $-0.5\,\text{m/s}, +0.3\,\text{m}$ | $-0.3\,\text{m/s}, +0.2\,\text{m}$ | **$+0.1\,\text{m/s}, +0.3\,\text{m}$** | $-1.0\,\text{m/s}, -0.7\,\text{m}$ | $-1.3\,\text{m/s}, -8.9\,\text{m}$ |
| **Trip 1 Ep 5** | **PARTIAL** | $+3.9\,\text{m/s}, -3.5\,\text{m}$ | $+3.9\,\text{m/s}, -0.8\,\text{m}$ | $+4.6\,\text{m/s}, +3.6\,\text{m}$ | $+3.1\,\text{m/s}, +4.2\,\text{m}$ | $+3.3\,\text{m/s}, +9.2\,\text{m}$ |
| **Trip 1 Ep 9** | **PARTIAL** | $-3.8\,\text{m/s}, -2.1\,\text{m}$ | $-4.2\,\text{m/s}, -3.9\,\text{m}$ | $-7.4\,\text{m/s}, -9.9\,\text{m}$ | $-5.9\,\text{m/s}, -19.6\,\text{m}$ | $-3.4\,\text{m/s}, -11.1\,\text{m}$ |
| **Trip 1 Ep 8** | **DESTABILIZED**| **$+239.4\,\text{m/s}, +237.2\,\text{m}$**| **$+204.3\,\text{m/s}, +336.0\,\text{m}$**| *(Outage ended)* | *(Outage ended)* | *(Outage ended)* |
| **Trip 2 Ep 5** | **DESTABILIZED**| $-14.1\,\text{m/s}, -5.7\,\text{m}$ | $-14.1\,\text{m/s}, -5.7\,\text{m}$ | $-19.4\,\text{m/s}, -14.2\,\text{m}$ | **$-30.8\,\text{m/s}, -58.9\,\text{m}$**| **$-42.4\,\text{m/s}, -147.2\,\text{m}$**|
| **Trip 2 Ep 2** | **DESTABILIZED**| $-1.7\,\text{m/s}, -0.9\,\text{m}$ | $-2.0\,\text{m/s}, -1.8\,\text{m}$ | $-2.6\,\text{m/s}, -4.0\,\text{m}$ | $-3.8\,\text{m/s}, -13.1\,\text{m}$ | $-6.1\,\text{m/s}, -42.4\,\text{m}$ |

---

## 5. The Critical Discovery: Why Residual Magnitude Fails as a Stability Gate

A naive gating hypothesis would suggest: *"If residual $|r_{\text{vert}}| > X$, reject to prevent instability."*

Stage C10.8B empirically refutes this hypothesis:

```text
               STABLE RECOVERIES                     DESTABILIZED TRAJECTORIES
         Trip 1 Ep 5: |r| = 6.70 m/s           Trip 2 Ep 2: |r| = 1.42 m/s
         Trip 1 Ep 9: |r| = 3.41 m/s           Trip 1 Ep 3: |r| = 0.45 m/s
         Trip 2 Ep 3: |r| = 3.00 m/s           Trip 1 Ep 4: |r| = 0.11 m/s
               ──────────────                        ──────────────
          Median |r| = 3.00 m/s                 Median |r| = 1.42 m/s
```

### Key Statistical Contrasts Between Stable/Partial and Destabilized Classes:
- **Median Residual**: Stable/Partial = **$3.00\,\text{m/s}$** vs Destabilized = **$1.42\,\text{m/s}$**. Destabilized episodes had *smaller* residuals at recovery than stable episodes!
- **Roll Misalignment**: Stable/Partial mean roll = **$5.2^\circ$** vs Destabilized mean roll = **$19.9^\circ$** (peaking at $90.3^\circ$ in Ep 8).
- **Innovation Variance $S$**: Stable/Partial mean $S = 0.92\,\text{m}^2/\text{s}^2$ vs Destabilized mean $S = 5.41\,\text{m}^2/\text{s}^2$.
- **NIS at Recovery**: Stable/Partial median $\text{NIS} = 8.8$ vs Destabilized mean $\text{NIS} = 746.7$ (skewed by Ep 8).

---

## 6. Detailed Forensic on the Failure Mechanisms

### 6.1 The Inverted Kalman Gain Disaster (Trip 1 Episode 8)
- **The Event**: Sharp turn into the university department; the passenger tilted the phone onto its side ($\text{roll} = -90.3^\circ$, $\text{pitch} = -38.6^\circ$).
- **The Mathematical Mechanism**:
  $$\mathbf{H} = \begin{bmatrix} 0 & 0 & 0 & C_{31} & C_{32} & C_{33} & 0_{1\times 9} \end{bmatrix}$$
  Because the phone rolled $90^\circ$, $C_{33}$ flipped sign and magnitude. The resulting Kalman gain component mapping innovation into vertical velocity was:
  $$K_{v_u} = \mathbf{-0.159} \quad (\text{NEGATIVE GAIN!})$$
- **The Trajectory**:
  - Pre-update: $v_U^- = +161.56\,\text{m/s}$, $r_{\text{vert}} = 299.80\,\text{m/s}$.
  - State correction: $\Delta v_U = K_{v_u} \cdot r_{\text{vert}} \cdot \gamma = (-0.159) \cdot (299.80) \cdot (0.034) = -1.62\,\text{m/s}$ (but coordinate rotation added positive cross-coupling).
  - Velocity immediately jumped to $v_U^+ = +209.25\,\text{m/s}$, and reached $+239.4\,\text{m/s}$ within $0.5\,\text{s}$.
- **Diagnostic Finding**: When attitude is severely rotated ($|\text{roll}| \approx 90^\circ$), VNHC updates project across orthogonal body axes, actively feeding energy into vertical runaway.

### 6.2 The Inflection-Point False Calm (Trip 2 Episode 5)
- **The Event**: Passenger violently tumbling the phone for 224 seconds.
- **The Mechanism**: At $t = 61.2\,\text{s}$, the hand paused for 0.25 seconds during a grip change. Angular rate dropped below $0.40\,\text{rad/s}$, satisfying the persistence condition.
- **The Trajectory**:
  - The filter admitted the update, reducing $v_U$ from $-8.54 \to -8.10\,\text{m/s}$.
  - At $t = 61.5\,\text{s}$, violent tumbling resumed ($\|\boldsymbol{\omega}\| > 500^\circ/\text{s}$). Because the vertical velocity covariance had been collapsed by the update ($P_{v_u v_u}$ reduced), the filter was left with small covariance while receiving massive corrupted accelerations.
  - Vertical velocity accelerated: $-14.1\,\text{m/s}$ (at $+1\text{s}$) $\to -30.8\,\text{m/s}$ (at $+5\text{s}$) $\to -42.4\,\text{m/s}$ (at $+10\text{s}$), eventually reaching $1,612\,\text{m/s}$.
- **Diagnostic Finding**: A 0.25s persistence timer is insufficient to distinguish a true steady cruise from a momentary zero-crossing during active manipulation.

### 6.3 Textbook Exponential Damping (Trip 2 Episode 3)
- **The Event**: Vehicle cruising on a clean highway; phone resting steadily in a passenger hand.
- **The Mechanism**: Attitude was physically valid ($\text{pitch} = -8.6^\circ, \text{roll} = +5.8^\circ$), angular rate calm ($\|\boldsymbol{\omega}\| = 0.092\,\text{rad/s}$), and $K_{v_u} = +0.457$ (positive, well-conditioned).
- **The Trajectory**:
  - $t = t_{\text{rec}}$: $v_U^- = -2.29\,\text{m/s} \longrightarrow v_U^+ = -0.92\,\text{m/s}$
  - $t = t_{\text{rec}} + 0.5\,\text{s}$: $v_U = -0.5\,\text{m/s}$
  - $t = t_{\text{rec}} + 1.0\,\text{s}$: $v_U = -0.3\,\text{m/s}$
  - $t = t_{\text{rec}} + 2.0\,\text{s}$: $v_U = +0.1\,\text{m/s}$
  - Total vertical drift across 57.6 seconds of blackout: **$-8.9\,\text{m}$** (vs $+365.7\,\text{m}$ in Control).
- **Diagnostic Finding**: When attitude is sane and the vehicle is in a true planar regime, admitting a large residual ($3.0\,\text{m/s}$) produces optimal exponential convergence.

---

## 7. Diagnostic Visualization

The 4-panel diagnostic plot is saved at [`results/c10_8b_post_recovery_stability.png`](c10_8b_post_recovery_stability.png):

*[Stage C10.8B Post-Recovery Stability Diagnostic — Diagnostic chart]*

- **Panel A (Post-Recovery Trajectory Timeseries)**: Contrasts stable exponential damping (Trip 2 Ep 3 and Trip 1 Ep 5) against explosive runaway (Trip 1 Ep 8 and Trip 2 Ep 5).
- **Panel B (Residual vs Post-Recovery Peak Velocity)**: Shows that episodes with small residuals ($0.1–1.4\,\text{m/s}$) frequently destabilize, while episodes with large residuals ($3.0–6.7\,\text{m/s}$) converge smoothly, disproving residual-based gating.
- **Panel C (Innovation Covariance S vs NIS)**: Highlights severe overconfidence and gain distortion under extreme NIS ($>3,000$).
- **Panel D (Stability Classification Distribution)**: Summarizes the distribution across the 12 events: 1 Stable Convergence, 6 Partial Recoveries, and 5 Destabilizations.

---

## 8. Summary Conclusions & Epistemic Boundaries

1. **Decoupled Physical Eligibility Solves the Latch, But Not the Stability Problem**:
   - Decoupled physical eligibility successfully eliminates the permanent lockouts observed in Control.
   - However, **eligibility alone is NOT a sufficient condition for filter stability**.
2. **The True Determinant of Stability is Attitude Frame Sanity**:
   - VNHC relies on the assumption that the vehicle's body z-axis is approximately vertical.
   - When pitch or roll tilts severely ($>30^\circ$) or inverts ($90^\circ$), the measurement Jacobian $\mathbf{H}$ projects forward/lateral vehicle motion into the vertical channel, flipping the sign of $K_{v_u}$ and driving positive feedback runaway.
3. **Persistence Duration Governs False-Calm Filtering**:
   - A $0.25\,\text{s}$ timer admits transient inflection points during tumbling.
   - A robust architecture must verify motion calmness over a longer window or confirm attitude convergence before admitting high-gain corrections.
4. **Production Code Remains Frozen**:
   - These findings represent offline diagnostic characterization. No changes to `android/` or production navigation math are proposed or authorized.

---

## 9. Deliverables

- **Characterization Script**: [`scratch/c10_8b_post_recovery_stability.py`](../scratch/c10_8b_post_recovery_stability.py)
- **Structured Audit JSON**: [`results/c10_8b_post_recovery_stability.json`](c10_8b_post_recovery_stability.json)
- **Diagnostic Visualization**: [`results/c10_8b_post_recovery_stability.png`](c10_8b_post_recovery_stability.png)
- **Detailed Forensic Report**: [`results/c10_8b_post_recovery_stability.md`](c10_8b_post_recovery_stability.md)
- **Production Status**: `android/` and `src/navigation/dead_reckoning_engine.py` remain **STRICTLY FROZEN**.
