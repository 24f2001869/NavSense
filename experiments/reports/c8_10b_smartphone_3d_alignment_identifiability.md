# Stage C8-10B: Smartphone-Only 3D Alignment Identifiability Audit

**Module**: `experiments/audit_3d_alignment_identifiability_c8_10b.py`  
**Results JSON**: `results/c8_10b_smartphone_3d_alignment_identifiability.json`  
**Status**: Strict Diagnostic Completed — Offline Verification  
**Classification**: **B — PARTIALLY IDENTIFIABLE**

---

## Executive Summary

> **Central Question**: *"Is the full 3D phone-to-vehicle rotation matrix $R_p^v$ identifiable from smartphone-only signals?"*

### Answer: **Classification B — PARTIALLY IDENTIFIABLE**

| DOF | Status | Observable From | Impact |
|:----|:------:|:----------------|:-------|
| Phone roll relative to vertical | ✅ OBSERVABLE | Gravity (accelerometer) | Constrains 1 of 3 DOFs of $R_p^n$ |
| Phone pitch relative to vertical | ✅ OBSERVABLE | Gravity (accelerometer) | Constrains 2 of 3 DOFs of $R_p^n$ |
| Phone-to-vehicle horizontal heading | ✅ OBSERVABLE | Magnetometer + pre-outage GNSS | Constrains $R_n^v$ yaw DOF |
| Cradle mechanical pitch to chassis | ❌ **UNIDENTIFIABLE** | Conflated with road grade | Controls `gyro_x` → vehicle yaw weight |
| Cradle mechanical roll to chassis | ❌ **UNIDENTIFIABLE** | Conflated with road bank | Controls `gyro_y` → vehicle yaw weight |

> **C8-10B Frozen Conclusion (Decision B — Partial Identifiability)**:
> Smartphone sensors can establish phone attitude information and provide useful heading information, but the complete phone-to-vehicle 3D rotation is not uniquely identifiable under the tested assumptions. The ~20° dynamic pitch result is an empirical consistency finding, not a validated physical mounting angle. Therefore the system should not rely on a gravity-only 3D phone-to-vehicle alignment or assume phone gyro-Z is vehicle yaw.

The tested smartphone-only signals do not uniquely identify the complete phone-to-vehicle 3D rotation under the available assumptions, because static accelerometer specific force conflates phone cradle tilt, sustained road grade/bank, and chassis leveling offsets.

---

## TASK 1 — Formal Observability Matrix Construction

### Observation Matrix $H$ for Phone-to-Navigation Rotation $R_p^n$

Two physical vector observations are available:
1. **Gravity** $\mathbf{g}_n = [0, 0, g]^T$ — provides Jacobian $J_g \in \mathbb{R}^{3 \times 3}$
2. **Magnetic field** $\mathbf{B}_n$ — provides Jacobian $J_B \in \mathbb{R}^{3 \times 3}$

Stacked observation matrix: $H = \begin{bmatrix} J_g \\ J_B \end{bmatrix} \in \mathbb{R}^{6 \times 3}$

| Trip | Gravity Rank | Gravity Nullspace | Gravity + Mag Rank | Nullspace Dim |
|:-----|:---:|:---|:---:|:---:|
| **Vta04** | **2** | Yaw axis $[0, 0, 1]^T$ | **3** (full rank) | **0** |
| **Vta02** | **2** | Yaw axis $[0, 0, 1]^T$ | **3** (full rank) | **0** |

### Key Result
- Gravity alone: rank 2/3 → roll and pitch observable, **yaw in nullspace** (nullspace vector = $[0, 0, 1]^T$, confirming the mathematical proof from C8-10A)
- Gravity + Magnetometer: **rank 3/3** → $R_p^n$ (phone-to-navigation) is **fully observable**
- The magnetometer fills the yaw nullspace left by gravity

---

## TASK 2 — Distinguish $R_p^n$ from $R_p^v$

### The Critical Decomposition

$$R_p^v = R_n^v \cdot R_p^n$$

| Matrix | Meaning | Observable? |
|:-------|:--------|:------------|
| $R_p^n$ | Phone-to-navigation frame | **YES** — gravity (2 DOFs) + magnetometer (1 DOF) |
| $R_n^v$ | Navigation-to-vehicle frame | **PARTIALLY** — yaw (heading offset) YES, pitch/roll NO |
| $R_p^v$ | Phone-to-vehicle (what we need) | **PARTIALLY** — only 4 of 6 constraint DOFs identifiable |

### $R_n^v$ Decomposition Results

| Trip | $R_n^v$ Yaw | $R_n^v$ Pitch | $R_n^v$ Roll | Off-Vertical Magnitude | Pure Yaw? |
|:-----|:---:|:---:|:---:|:---:|:---:|
| **Vta04** | Identifiable | $5.55°$ | $1.01°$ | **$5.64°$** | **NO** — significant pitch |
| **Vta02** | Identifiable | $1.16°$ | $-0.03°$ | **$1.16°$** | Approximately yes |

### Fundamental Ambiguity Proof

The phone accelerometer measures:
$$\mathbf{a}_p = R_v^p \cdot (\mathbf{a}_\text{vehicle} + \mathbf{g}_n)$$

A constant lateral tilt in $R_n^v$ would project gravity into the lateral axis **identically** to:
1. A constant road bank angle
2. Sustained cornering force
3. Cradle mechanical roll in the mount

Without an independent measurement of vehicle body attitude, $R_n^v$ pitch/roll **remain unidentifiable**.

---

## TASK 3 — Centripetal Acceleration Constraint Analysis

During turns, centripetal acceleration $a_c = v \cdot \dot\psi$ provides a horizontal direction constraint.

| Trip | Turn Episodes Detected | Episodes with $|a_c| > 0.3$ m/s² | Mean $|a_c|$ (m/s²) |
|:-----|:---:|:---:|:---:|
| **Vta04** | 60 | **56** | Sufficient excitation |
| **Vta02** | 318 | **299** | Strong excitation |

### Finding
- Centripetal acceleration IS detectable in the phone frame during turns
- The direction of centripetal acceleration in the **leveled** frame constrains the vehicle lateral axis
- **However**: centripetal direction is measured relative to the navigation frame (via leveling), NOT relative to the vehicle chassis
- The centripetal constraint helps identify $R_p^n$ yaw (which is already identified by magnetometer), but does NOT resolve the $R_n^v$ pitch/roll ambiguity

---

## TASK 4 — Gravity-Subtracted Dynamic Acceleration

| Trip | Acceleration Episodes (straight) | Braking Episodes (straight) | Forward Axis ID |
|:-----|:---:|:---:|:---:|
| **Vta04** | Insufficient | Insufficient | **NOT viable** |
| **Vta02** | Insufficient | Insufficient | **NOT viable** |

### Finding
With the strict constraint of straight-line driving (gyro $< 2°/s$) AND significant acceleration/braking ($> 0.5$ m/s²), there are insufficient episodes in these trips for reliable forward axis identification from dynamic acceleration alone. The phone GNSS speed derivative is too noisy to reliably detect pure longitudinal acceleration episodes.

---

## TASK 5 — Fisher Information Matrix Analysis

### FIM Eigenanalysis for $R_p^n$ Parameters

| Trip | Static FIM Condition # | Total FIM Condition # | CRLB Roll | CRLB Pitch | CRLB Yaw |
|:-----|:---:|:---:|:---:|:---:|:---:|
| **Vta04** | 177.5 | 176.2 | Well-constrained | Well-constrained | **$7.73°$** |
| **Vta02** | 147.6 | 146.7 | Well-constrained | Well-constrained | **$7.05°$** |

### Critical Distinction
- These CRLB bounds apply to $R_p^n$ (phone-to-**navigation**), NOT $R_p^v$ (phone-to-**vehicle**)
- The FIM correctly shows that yaw is the weakest DOF of $R_p^n$ (highest CRLB)
- The FIM **does not capture** the $R_n^v$ pitch/roll ambiguity because the accelerometer physically cannot distinguish cradle tilt from road grade — this is not a noise-limited problem but a structural observability gap

---

## TASK 6 — Phone Gyro Excitation During Vehicle Turns

| Trip | Dominant Gyro Axis | Energy Fraction (x/y/z) | Ref Yaw Correlation (x/y/z) | Per-Turn Consistency $\sigma$ |
|:-----|:---:|:---:|:---:|:---:|
| **Vta04** | **gyro_x** | 55% / 32% / 14% | +0.267 / -0.138 / -0.079 | Rigid |
| **Vta02** | **gyro_x** | 59% / 30% / 12% | +0.261 / -0.073 / -0.027 | Rigid |

### Key Findings
1. **`gyro_x` dominates turn response** on both trips (55–59% of total gyro energy during turns)
2. **`gyro_x` is the most yaw-correlated axis** ($r = +0.267$ on Vta04 during turns)
3. **`gyro_z` contains only 12–14% of turn energy** despite being the "Yaw" channel in the Android sensor axis convention
4. This confirms the phone is mounted approximately landscape with a significant cradle tilt that places vehicle yaw primarily into phone `gyro_x`
5. **The axis ratios are consistent across turns** → rigid mounting confirmed

---

## TASK 7 — Dynamic Acceleration Direction Consistency

| Trip | Accel Episodes | Brake Episodes | Anti-Parallel Check |
|:-----|:---:|:---:|:---:|
| **Vta04** | Insufficient | Insufficient | N/A |
| **Vta02** | Insufficient | Insufficient | N/A |

The strict thresholds (straight-line + significant acceleration/braking from phone GNSS) yield insufficient episodes. Forward axis identification from dynamic acceleration alone is NOT viable in these datasets.

---

## TASK 8 — Cross-Axis Gyro-Acceleration Consistency Test

During turns: yaw rate (vertical axis) should be perpendicular to centripetal acceleration (horizontal).

| Trip | Dominant Gyro | Dominant Centripetal | Cross-Correlation | Perpendicular? |
|:-----|:---:|:---:|:---:|:---:|
| **Vta04** | gyro_X | leveled_Y | $-0.043$ | ✅ YES |
| **Vta02** | gyro_X | leveled_Y | $+0.038$ | ✅ YES |

### Finding
The dominant gyro axis (X) and dominant centripetal axis (leveled Y) are approximately uncorrelated ($|r| < 0.05$), confirming geometric consistency of the mounting: phone `gyro_x` captures vertical rotation and leveled `accel_y` captures horizontal centripetal acceleration, as expected for perpendicular physical axes.

---

## TASK 9 — Identifiability Classification

### Classification: **B — PARTIALLY IDENTIFIABLE**

| Component | DOFs | Identifiable? | Method |
|:----------|:---:|:---:|:------|
| $R_p^n$ (phone-to-nav) | 3 | ✅ ALL | Gravity + Magnetometer |
| $R_n^v$ yaw (heading offset) | 1 | ✅ | Pre-outage GNSS bearing |
| $R_n^v$ pitch (cradle pitch/road grade) | 1 | ❌ | **Fundamentally ambiguous** |
| $R_n^v$ roll (cradle roll/road bank) | 1 | ❌ | **Fundamentally ambiguous** |

### Fundamental Physical Limitation

> The smartphone accelerometer measures $\mathbf{a}_p = R_v^p (\mathbf{a}_\text{veh} + \mathbf{g}_n)$. Any constant tilt of the phone cradle relative to the vehicle chassis is **indistinguishable** from a constant road gradient or vehicle body pitch/roll. This is a **PHYSICAL OBSERVABILITY CONSTRAINT** — no algorithm operating on smartphone accelerometer data alone can resolve this ambiguity.

---

## TASK 10 — Residual Ambiguity Quantification

| Trip | $R_\text{error}$ Total Angle | True Row 2 | Smartphone Row 2 | True Yaw $r$ (turns) | Phone Yaw $r$ (turns) |
|:-----|:---:|:---:|:---:|:---:|:---:|
| **Vta04** | **$5.65°$** | $[0.016, 0.004, 1.000]$ | $[0.024, -0.094, 0.995]$ | $-0.051$ | $-0.016$ |
| **Vta02** | **$1.16°$** | $[-0.013, -0.032, 0.999]$ | $[0.005, -0.041, 0.999]$ | $-0.028$ | $+0.019$ |

### Critical Finding
- **Both** the true $R_{pv}$ and the smartphone-only $R_{pv}$ yield row 2 $\approx [0, 0, 1]$
- Both produce near-zero correlation with vehicle yaw rate during turns ($|r| < 0.06$)
- **The true alignment is ALSO degenerate** — this is not a calibration failure but a physical property of the near-vertical phone mounting
- The $5.65°$ alignment error on Vta04 (dominated by $R_n^v$ pitch = $5.55°$) does not significantly change the yaw projection because both the true and estimated row 2 are dominated by the Z component

---

## TASK 11 — Yaw Projection Sensitivity to Mounting Tilt

### Sensitivity Analysis: $\partial r / \partial (\text{tilt angle})$

| Trip | Cradle Pitch Sensitivity | Cradle Roll Sensitivity | Dominant DOF |
|:-----|:---:|:---:|:---:|
| **Vta04** | $0.0230$ $dr/°$ | $0.0046$ $dr/°$ | **Pitch** (5× more sensitive) |
| **Vta02** | $0.0478$ $dr/°$ | $0.0002$ $dr/°$ | **Pitch** (200× more sensitive) |

### Optimal Cradle Tilt (Reference Validation Only)

| Trip | Optimal Pitch | Optimal Roll | Achieved $r$ with Vehicle Yaw |
|:-----|:---:|:---:|:---:|
| **Vta04** | $-20°$ | $+20°$ | **$0.322$** |
| **Vta02** | $-20°$ | $+20°$ | **$0.684$** |

### Critical Finding
- At $0°$ cradle tilt (gravity-level): gyro yaw correlation is essentially zero ($r \approx 0$).
- At $-15°$ cradle pitch: correlation rises to $r = 0.23$ (Vta04) and $r = 0.50$ (Vta02).
- At the empirical consistency peak of $-20°$ pitch: correlation reaches $r = 0.32$ (Vta04) and $r = 0.68$ (Vta02).
- **Physical Interpretation Caution**: A pitch rotation of approximately $-20°$ produced substantially stronger turn-related cross-axis consistency (especially in Vta02), but the physical origin of this rotation is **not uniquely identifiable** from smartphone accelerometer data alone. Road grade, longitudinal acceleration, suspension/mount dynamics, coordinate-frame effects, and filtering can all influence that correlation. It is an empirical consistency finding, not a validated physical mounting angle.

### Yaw Projection Correlation vs Cradle Pitch Perturbation

| Perturbation | Vta04 $r$ | Vta02 $r$ |
|:---:|:---:|:---:|
| $-15°$ | 0.230 | 0.501 |
| $-10°$ | 0.179 | 0.427 |
| $-5°$ | 0.096 | 0.261 |
| $-2°$ | 0.031 | 0.118 |
| $0°$ (current) | **-0.016** | **0.019** |
| $+2°$ | -0.061 | -0.071 |
| $+5°$ | -0.121 | -0.181 |
| $+10°$ | -0.194 | -0.301 |

The sensitivity curve confirms that gyro yaw observability is **strongly** dependent on cradle pitch, and the current gravity-level alignment places the phone at the **worst possible operating point** ($r \approx 0$) for yaw recovery.

---

## TASK 12 — Final Engineering Decision

### Classification: **B — PARTIALLY IDENTIFIABLE**

### What IS Identifiable (Smartphone-Only)

1. **$R_p^n$ (phone-to-navigation)**: Fully identifiable from gravity + magnetometer → 3 DOFs
2. **$R_n^v$ yaw (heading offset)**: Identifiable from pre-outage GNSS bearing → $3.7°$ std (Vta04, 60s window)
3. **Compass-aided heading**: Achievable with $8–17°$ MAE, providing **84–96% drift reduction** at 10–60s horizons

### What Is NOT Identifiable (Under Tested Assumptions)

1. **$R_n^v$ pitch/roll**: The tested smartphone-only signals do not uniquely identify the complete phone-to-vehicle 3D rotation under the available assumptions, because static accelerometer specific force conflates phone cradle tilt, sustained road grade/bank, and chassis leveling offsets.
2. **Physical Cradle Angle**: The ~20° dynamic pitch result is an empirical consistency finding, not a validated physical mounting angle. Road grade, longitudinal acceleration, suspension/mount dynamics, coordinate-frame effects, and filtering can all influence that correlation.
3. **Gyro Yaw Assumption**: The system must NOT rely on a gravity-only 3D phone-to-vehicle alignment or assume phone gyro-Z is vehicle yaw simply because the phone reports a Z-axis gyro.

### Recommendation & Target Architecture

> **C8-10B Frozen Conclusion (Decision B — Partial Identifiability)**:
> Smartphone sensors can establish phone attitude information and provide useful heading information, but the complete phone-to-vehicle 3D rotation is not uniquely identifiable under the tested assumptions. The ~20° dynamic pitch result is an empirical consistency finding, not a validated physical mounting angle. Therefore the system should not rely on a gravity-only 3D phone-to-vehicle alignment or assume phone gyro-Z is vehicle yaw.
> 
> **Next Stage Focus**:
> Extract the maximum *legitimate* vehicle-heading information available from smartphone GNSS course + magnetometer + gyro, rather than trying to force a full 3D alignment from gravity.
> Specifically audit: *Can we exploit GNSS course during pre-outage motion to estimate the vehicle-forward direction relative to the phone, and does that estimate remain stable across straight driving, acceleration, braking, and turns?*

### Circularity Compliance

**PASS** — All identifiable calibration uses smartphone-only signals. Zero CAN/VBOX/vehicle reference data in the proposed calibration protocol. Validation against vehicle reference is performed post-hoc only.

---

## Summary of Diagnostic Chain (C8-9 → C8-10B)

| Stage | Finding |
|:------|:--------|
| **C8-9** | Heading divergence is the primary error mechanism |
| **C8-9.1** | Gravity-only alignment produces $R_{pv}[2,:] \approx [0,0,1]$ → gyro yaw unobservable |
| **C8-10A** | Pre-outage heading offset IS observable → 84–96% drift reduction via compass aiding |
| **C8-10A** | Horizontal calibration does NOT fix gyro projection (R_z preserves row 2) |
| **C8-10B** | $R_p^n$ is fully observable; $R_n^v$ yaw is observable; **full 3D rotation not uniquely identifiable under tested assumptions** |
| **C8-10B** | Empirical ~20° pitch correlation is not a validated physical mount angle; do NOT force gyro-Z as vehicle yaw |
| **C8-10B** | **Decision B Frozen** — audit pre-outage GNSS-course vehicle-forward axis identifiability before implementation |

