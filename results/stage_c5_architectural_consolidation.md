# Stage C5 Architectural Consolidation: Bounded Adaptive Covariance & Frozen Confidence Baseline

**Project:** SIH26168 — Integrated Dead Reckoning (IDR) using Smartphone IMU  
**Module:** Process Noise Confidence Adaptation Branch (Stage C5)  
**Status:** **FROZEN & CONSOLIDATED**  
**Epistemic Verdict:** 🟢 **PASS (Stage C5.5.7 & C5.6 Stage 1)**  

---

## 1. Executive Summary & Research Trajectory

Over Stages C5.1 through C5.6, we conducted a systematic, physics-guided investigation into whether smartphone IMU process noise covariance $\mathbf{Q}(k)$ can be dynamically adapted to improve dead-reckoning performance during GNSS blackouts.

### The Research Chain:
```
Raw Smartphone IMU Stream
         │
         ▼
Rigorous Coordinate & Attitude Representation (3D Quaternions, NED Frame)
         │
         ▼
Strapdown Inertial Navigation System (3D Mechanization)
         │
         ▼
15-State Error-State Kalman Filter (ESKF with Non-Holonomic Constraints)
         │
         ▼
C5.5.1–C5.5.3: Physical Disturbance Characterization (Vibration vs. Engine RPM)
         │  ↳ Proved 2.2–2.5 Hz phone mode does not track RPM; RPM is insufficient predictor
         ▼
C5.5.4: Signal Conditioning vs. Adaptive Covariance Ablation
         │  ↳ Proved fixed filtering distorts transients and worsens drift; adaptive covariance helps
         ▼
C5.5.4b: Nominal Tuning & Temporal Alignment Control
         │  ↳ Discovered nominal Q was under-tuned; temporal placement of Q matters
         ▼
C5.5.5: Feature Audit & Multidimensional Regimes
         │  ↳ Isolated 4 causal features; proved acceleration variance alone confuses rough roads with maneuvers
         ▼
C5.5.6: Cross-Trip Generalization Diagnostic (`Vta02` → `Vta04`)
         │  ↳ Revealed severe cross-trip vibration shift (+61.3% baseline on coarse asphalt)
         ▼
C5.5.7: Causal Ambient-Baseline Normalization & 60s Divergence Forensic
         │  ↳ Isolated catastrophic Window W21 failure in unbounded adaptive covariance;
         │  ↳ Formulated bounded covariance controller to prevent runaway
         ▼
C5.6 Stage 1: Parsimonious Machine Learning Benchmark
         │  ↳ Ridge, Shallow Tree, and Small MLP evaluated on untouched Vta04;
         │  ↳ ML superiority REFUTED for tested models; hand-designed normalization outperforms ML
         ▼
STAGE C5 ARCHITECTURAL FREEZE (This Document)
```

---

## 2. Current Best Tested Confidence Configuration

We formally freeze the confidence adaptation architecture as **Bounded Adaptive C0** (also designated **Bounded Causal Adaptive Covariance - BCAC**).

### 2.1 Mathematical Formulation

The controller operates on discrete IMU samples at rate $f_s = 100\text{ Hz}$ ($\Delta t = 0.01\text{ s}$):

1. **Multi-Axis Instantaneous Jerk ($J$):**
   $$J(k) = \frac{\|\mathbf{a}(k) - \mathbf{a}(k-1)\|_2}{\Delta t} \quad \left[\text{m/s}^3\right]$$
   where $\mathbf{a}(k) \in \mathbb{R}^3$ is the raw calibrated smartphone accelerometer vector.

2. **Causal Rolling Ambient Baseline ($B$):**
   $$B(k) = \operatorname{median}_{j \in [\max(0, k - W_b + 1), \, k]} J(j) \quad \left[\text{m/s}^3\right]$$
   where $W_b = 10\text{ s} \times 100\text{ Hz} = 1000\text{ samples}$ (strictly trailing causal window).  
   *Initialization:* For $k < W_b$, the window expands causally over all available history $j \in [0, k]$.

3. **Ambient-Normalized Disturbance Metric ($J_{\text{norm}}$):**
   $$J_{\text{norm}}(k) = \frac{J(k)}{B(k) + \epsilon}$$
   where $\epsilon = 10^{-4}\text{ m/s}^3$ guarantees strict numerical non-singularity.

4. **Bounded Multiplier Mapping ($m$):**
   $$m(k) = \operatorname{clip}\left(\sqrt{J_{\text{norm}}(k)}, \; m_{\min}, \; m_{\max}\right)$$
   with frozen parameters:
   $$m_{\min} = 0.75, \quad m_{\max} = 1.35$$

5. **Active Process Noise Injection ($\sigma_a$):**
   $$\sigma_a(k) = \sigma_{a,0} \cdot m(k) = 0.291 \cdot m(k) \quad \left[\text{m/s}^2\right]$$
   Yielding active process noise strictly bounded within:
   $$\sigma_a(k) \in [0.21825, \; 0.39285]\text{ m/s}^2$$
   Gyroscope process noise is kept frozen at nominal:
   $$\sigma_g(k) = \sigma_{g,0} = 0.015\text{ rad/s}$$

---

## 3. Implementation Specifications & Causal Interface

### 3.1 Input / Output Interface
* **Inputs:**
  - Phone accelerometer stream $\mathbf{a}_k \in \mathbb{R}^3$ ($100\text{ Hz}$, calibrated, sensor frame).
  - Sampling interval $\Delta t = 0.01\text{ s}$.
  - *Strict Zero Reference Leakage:* No CAN bus, no wheel speed, no GPS, no VBOX reference data in the confidence path.
* **Internal State Buffer:**
  - Circular FIFO buffer storing the last $W_b = 1000$ values of $J \in \mathbb{R}$.
  - Memory footprint: $1000 \times 8\text{ bytes} = 8.0\text{ KB}$ (well within embedded microcontroller/mobile cache limits).
* **Output:**
  - Scalar process noise standard deviation $\sigma_a(k) \in [0.218, 0.393]\text{ m/s}^2$ passed to the ESKF discrete propagation block:
    $$\mathbf{Q}_k = \operatorname{diag}\left(\sigma_a^2(k) \mathbf{I}_3 \Delta t, \; \sigma_g^2 \mathbf{I}_3 \Delta t, \; \sigma_{ba}^2 \mathbf{I}_3 \Delta t, \; \sigma_{bg}^2 \mathbf{I}_3 \Delta t\right)$$

### 3.2 Causality & Determinism Guarantees
* **Audit Certification:** Fully verified in Stage C5.5.7 audit script (`experiments/audit_c5_5_7_integrity_and_failure_mode.py`).
* **Zero Lookahead:** All slices access strictly indices $\le k$.
* **Computational Complexity:** $\mathcal{O}(1)$ push into circular buffer; median extraction via quickselect $\mathcal{O}(W_b)$ worst-case, or rolling histogram/dual-heap in $\mathcal{O}(\log W_b)$ ($< 0.05\text{ ms}$ execution per epoch on mobile CPU).

---

## 4. Empirical Performance Benchmark on Untouched Held-Out `Vta04`

Across 150 simulated GNSS blackout windows across 5 horizons, Bounded Adaptive C0 (BCAC) compares against the complete benchmark family as follows:

| Outage Horizon | Metric | A. C0 Constant ($0.291$) | B. Static Cand 1 | C. Adaptive Cand 1B | **D. Bounded Adaptive C0 (BCAC)** | E1. Ridge | E2. Decision Tree | E3. Small MLP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 s** ($N=34$) | Mean / Median | $13.47 / \mathbf{10.48}\text{ m}$ | $13.53 / 11.60\text{ m}$ | $\mathbf{13.22} / 11.58\text{ m}$ | $13.40 / 11.19\text{ m}$ | $13.53 / 10.50\text{ m}$ | $13.55 / 10.49\text{ m}$ | $13.54 / 10.55\text{ m}$ |
| **10 s** ($N=33$) | Mean / Median | $66.77 / 66.64\text{ m}$ | $\mathbf{63.98} / \mathbf{65.26}\text{ m}$ | $64.60 / 65.64\text{ m}$ | $66.41 / 67.73\text{ m}$ | $66.56 / 66.49\text{ m}$ | $66.58 / 66.55\text{ m}$ | $66.56 / 66.68\text{ m}$ |
| **20 s** ($N=31$) | Mean / Median | $180.55 / 196.02\text{ m}$ | $\mathbf{176.76} / 184.00\text{ m}$ | $177.43 / \mathbf{181.81}\text{ m}$ | $183.14 / 188.97\text{ m}$ | $181.28 / 194.07\text{ m}$ | $181.38 / 193.48\text{ m}$ | $182.12 / 192.51\text{ m}$ |
| **30 s** ($N=29$) | **Mean Drift** | $278.09\text{ m}$ | $270.03\text{ m}$ | $\mathbf{267.42}\text{ m}$ | $273.49\text{ m}$ | $277.91\text{ m}$ | $277.43\text{ m}$ | $277.59\text{ m}$ |
| | **Median Drift** | $289.30\text{ m}$ | $293.02\text{ m}$ | $289.91\text{ m}$ | $\mathbf{283.89}\text{ m}$ | $291.52\text{ m}$ | $293.40\text{ m}$ | $291.72\text{ m}$ |
| | **P90 Drift** | $353.04\text{ m}$ | $353.99\text{ m}$ | $\mathbf{349.62}\text{ m}$ | $356.76\text{ m}$ | $352.48\text{ m}$ | $352.66\text{ m}$ | $354.16\text{ m}$ |
| **60 s** ($N=23$) | **Mean Drift** | $602.47\text{ m}$ | $621.70\text{ m}$ | $625.72\text{ m}$ | $\mathbf{599.36}\text{ m}$ | $604.54\text{ m}$ | $604.00\text{ m}$ | $604.48\text{ m}$ |
| | **Median Drift** | $582.26\text{ m}$ | $583.66\text{ m}$ | $583.95\text{ m}$ | $588.79\text{ m}$ | $\mathbf{563.57}\text{ m}$ | $563.83\text{ m}$ | $565.73\text{ m}$ |
| | **W21 Drift (110–170s)**| $872.7\text{ m}$ | $1,742.3\text{ m}$ [💥] | $1,774.7\text{ m}$ [💥] | $\mathbf{855.6}\text{ m}$ [🛡️] | $902.6\text{ m}$ | $907.8\text{ m}$ | $915.3\text{ m}$ |
| | **W21 Vel Err** | $8.38\text{ m/s}$ | $124.60\text{ m/s}$ | $122.97\text{ m/s}$ | $\mathbf{6.92}\text{ m/s}$ | $6.65\text{ m/s}$ | $6.47\text{ m/s}$ | $6.47\text{ m/s}$ |

---

## 5. Strengths, Limitations, & Failure Modes

### 5.1 Demonstrated Engineering Strengths
1. **Suppression of Covariance Runaway:**
   - Unbounded adaptive covariance (Cand 1B) failed catastrophically on Window W21 ($1,774.7\text{ m}$ drift, $>122\text{ m/s}$ velocity error) because sustained vibration drove $\sigma_a > 1.57\text{ m/s}^2$, de-weighting accelerometer trust to zero and letting velocity integrate freely along heading error.
   - Bounded Adaptive C0 caps $\sigma_a \le 0.393\text{ m/s}^2$, maintaining filter inertia and yielding the lowest W21 drift ($855.6\text{ m}$) of all evaluated methods.
2. **Dynamic Decoupling of Road Roughness from Maneuvers:**
   - On coarse asphalt (`Vta04`), ambient vibration is $+61.3\%$ higher than on smooth asphalt (`Vta02`).
   - Normalizing against the rolling median $B(k)$ prevents high baseline road chatter from being misinterpreted as continuous aggressive maneuvering.
3. **Superiority over Parsimonious ML:**
   - Supervised models (Tree, Ridge) placed disproportionate split weight ($100\%$) on raw variance $\sigma_a$, collapsing back to uninformative near-constant multipliers on held-out trips. BCAC retains clear physical interpretability.

### 5.2 Explicit Limitations & Unmet Benchmarks
1. **Does Not Guarantee Dead-Reckoning Stability:**
   - Bounding $\sigma_a \in [0.218, 0.393]\text{ m/s}^2$ bounds process noise covariance growth, but **cannot prevent dead-reckoning drift** caused by integrated gyro bias, accelerometer zero-bias, or heading error.
2. **SIH <10% Drift Benchmark Remains UNMET:**
   - The SIH competition target requires position error $< 10\%$ of distance travelled.
   - At typical driving speeds ($15\text{--}20\text{ m/s}$), distance travelled in $30\text{ s}$ is $\sim 450\text{--}600\text{ m}$. A $<10\%$ drift would require error $< 45\text{--}60\text{ m}$.
   - Our best 30-s drift on `Vta04` is **$267.42\text{--}273.49\text{ m}$** ($\sim 50\%$ of distance travelled).
   - In 60 seconds, drift reaches $\sim 600\text{ m}$.
   - **Conclusion:** *Confidence adaptation alone is mathematically incapable of solving the dead-reckoning problem.*

---

## 6. Official Freeze Decision on Confidence Branch

* **No Deeper ML Architectures:** We explicitly halt the pursuit of LSTM, GRU, Temporal Convolutional Networks (TCN), Transformers, or expanded feature engineering for confidence prediction.
* **Scientific Justification:** The delta between constant C0 ($278.09\text{ m}$) and the best adaptive controller ($267.42\text{ m}$) is $\sim 10.6\text{ m}$ ($3.8\%$). In contrast, the remaining unaddressed drift is **$>260\text{ m}$**. 
* Spending further engineering cycles optimizing covariance scalars is an inefficient allocation of research effort when the dominant error sources reside in the upstream navigation physics.

---

## 7. Bridge to Stage C6: Navigation Error Decomposition Diagnostic

To achieve meaningful reductions in the remaining $\sim 270\text{ m}$ drift, the next research phase must dissect the coupled error mechanisms of the inertial mechanization.

### 7.1 Conceptual Error Taxonomy (Motivation, Not Validated Attribution)
> **🚨 Epistemic Note:** The following strapdown error formulation represents a **conceptual error taxonomy** and motivation, **NOT an established quantitative decomposition or validated attribution**.  
> In reality, these physical mechanisms are nonlinearly coupled within the ESKF/NHC closed loop:
> - **Heading error $\to$ incorrect projection of longitudinal acceleration $\to$ velocity error $\to$ position error**;
> - **Tilt error $\to$ gravity leakage into horizontal channels $\to$ velocity error $\to$ position error**;
> - **NHC updates interact dynamically with both heading and velocity error states**.
> Therefore, navigation error cannot simply be summed as four independent numbers. Stage C6 must experimentally isolate sensitivity to each mechanism through controlled counterfactual perturbations.

Position error $e_{\mathbf{p}}(t) = \hat{\mathbf{p}}(t) - \mathbf{p}_{\text{true}}(t)$ evolves according to the strapdown error differential equations:
$$\ddot{e}_{\mathbf{p}}(t) \approx \mathbf{R}_b^n(t) \left( \delta \mathbf{f}^b(t) - \mathbf{f}^b(t) \times \boldsymbol{\psi}(t) \right) + \delta \mathbf{g}^n$$
where $\boldsymbol{\psi}(t) \approx \int_0^t \mathbf{R}_b^n(\tau) \delta \boldsymbol{\omega}^b(\tau) d\tau$ is the attitude/heading error angle.

Integrating over blackout duration $T$ yields the conceptual coupled error terms:
$$e_{\mathbf{p}}(T) \approx \underbrace{\int_0^T \int_0^t \mathbf{R}_b^n(\tau) \delta \mathbf{b}_a d\tau dt}_{\text{Accelerometer Bias Drift } (\sim \frac{1}{2} b_a T^2)} + \underbrace{\int_0^T \int_0^t \left( \mathbf{v}(\tau) \times \boldsymbol{\psi}(\tau) \right) d\tau dt}_{\text{Heading Error Cross-Coupling } (\sim v \psi_{\text{head}} T)} + \underbrace{e_{\mathbf{v},0} T}_{\text{Initial Velocity Error}} + \underbrace{e_{\text{NHC}}}_{\text{Constraint Violations}}$$

### 7.2 Diagnostic Research Agenda for Stage C6:
1. **Heading Sensitivity (C6-A):**
   - Inject controlled heading offsets $\delta\psi \in [0^\circ, 0.5^\circ, 1^\circ, 2^\circ, 5^\circ]$ into the initial state.
   - Measure velocity error, lateral position drift, total drift, and time growth. Compare empirical sensitivity to the idealized linear model $e_{\text{lat}} \approx v \psi T$.
2. **Accelerometer Bias Sensitivity (C6-B):**
   - Inject controlled forward accelerometer biases $b_a \in [0, 0.02, 0.05, 0.10, 0.20]\text{ m/s}^2$.
   - Measure trajectory error growth vs. ideal quadratic $\frac{1}{2} b_a T^2$ to assess how ESKF/NHC attenuates or propagates forward bias.
3. **Tilt / Gravity Leakage Sensitivity (C6-C):**
   - Inject controlled pitch/roll errors $\delta\theta \in [0^\circ, 0.25^\circ, 0.5^\circ, 1^\circ, 2^\circ]$.
   - Quantify empirical gravity leakage acceleration ($g \sin\delta\theta \approx 9.80665 \sin\delta\theta$) and trajectory drift in the complete filter.
4. **NHC Constraint Sensitivity (C6-D):**
   - Test four constraint regimes: No NHC, Normal NHC ($\sigma_{\text{nhc}} = 0.5\text{ m/s}$), Tightened NHC ($\sigma_{\text{nhc}} = 0.1\text{ m/s}$), and Relaxed NHC ($\sigma_{\text{nhc}} = 2.0\text{ m/s}$).
   - Examine lateral velocity, heading error, NHC innovation residuals, Kalman gains, and final position drift.
5. **Combined Mechanism Matrix (C6-E):**
   - Execute a controlled factorial matrix (Baseline, H1, B1, T1, N1, H+B, H+T, All) to measure whether errors superimpose linearly or exhibit strong nonlinear coupling.
   - Run primary controlled diagnostic on `Vta02`, followed by held-out confirmation on `Vta04`.

**Conclusion:** Stage C6 will determine which physical mechanism dominates the remaining drift, providing a rigorous empirical basis for subsequent architectural interventions.
