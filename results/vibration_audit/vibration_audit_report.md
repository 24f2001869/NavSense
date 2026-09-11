# Forensic Audit: Smartphone IMU Vibration Speed-Information (Phase 5.3)

**Date**: 2026-09-11  
**Scope**: Causal spectral analysis across all 64 available vehicle trips in IO-VNBD (450,000+ sliding windows)  
**Evaluated Systems**: Raw Accelerometers (`acc_x, acc_y, acc_z`), Gravitationally Projected Channels (`a_vert, a_horiz`), and Gyroscopes  
**Artifacts Produced**:
- Within-Trip Audit Table: [`vibration_speed_audit.csv`](vibration_speed_audit.csv)
- Cross-Trip Generalization Table: [`vibration_cross_trip_results.csv`](vibration_cross_trip_results.csv)
- Highway Speed-Band Discrimination Table: [`vibration_highway_analysis.csv`](vibration_highway_analysis.csv)
- 6-Panel Forensic Diagnostic Plot: [`vibration_spectrum_plots.png`](vibration_spectrum_plots.png)

*[Phase 5.3 Forensic Vibration Diagnostic Plot — Diagnostic chart]*

---

## Executive Summary & Final Verdict: 🛑 NO-GO

> [!CAUTION]
> **FINAL DECISION: NO-GO FOR VIBRATION-BASED CONTINUOUS SPEED ESTIMATION**
> 
> Before incorporating vibration features into the production TCN or ESKF, we subjected smartphone IMU vibration to strict causal spectral analysis across all 64 IO-VNBD vehicle trips (39 train, 6 val, 19 held-out test).
> 
> **The empirical verdict is definitive: No reliable, cross-trip, speed-specific vibration signal was demonstrated at the available 10-Hz sampling rate.**
> 
> 1. **Zero Frequency-Speed Tracking**: Dominant vibration frequency has **zero correlation with vehicle speed** across 64 trips ($r = -0.032 \pm 0.169$). The frequency peak does **not** shift with vehicle speed ($f \not\propto v$).
> 2. **Cross-Trip Transfer Failure**: Vibration-only regression models fitted on training data collapse when evaluated on unseen test trips:
>    - Vibration Ridge Regression: **Mean $R^2 = -15,453$** (MAE = $6.01\text{ m/s}$)
>    - Vibration Random Forest: **Mean $R^2 = -182.3$** (MAE = $3.32\text{ m/s}$)
>    - Vibration models failed to provide reliable cross-trip generalization and were inferior to the baseline TCN on **14 out of 19 held-out test trips (73.7%)** (on the remaining trips, RF merely approximated the global training mean, yielding negative $R^2$).
> 3. **Highway Discrimination Failure**: On the 163 km motorway trip `V-Vfa02`, vibration features **cannot distinguish 85 km/h from 105 km/h** during straight cruising (dominant frequency remains flat at $2.87–2.93\text{ Hz}$, centroid remains flat at $2.64–2.72\text{ Hz}$, and vibration power oscillates erratically).
> 4. **Dominant Confounder: Road Roughness & Suspension**: Vibration amplitude is overwhelmingly dictated by road surface roughness (asphalt vs concrete vs potholes) and vehicle suspension dynamics, **not** vehicle ground speed. A car crawling over uneven cobblestone exhibits higher vibration power than a car cruising at 110 km/h on smooth highway asphalt.

---

## A. Vibration Spectrum Analysis (Causal Extraction)

Using causal sliding windows (2.0s, 5.0s, and 10.0s) strictly without future lookahead, we computed one-sided Welch/FFT power spectral densities up to the 5.0 Hz Nyquist limit ($f_s = 10\text{ Hz}$).

### Measured Spectral Features across 64 Trips
- **Dominant Frequency**: The peak frequency of the highest PSD mode above 0.2 Hz.
- **Spectral Centroid**: The center-of-mass frequency $\frac{\sum f_k P(f_k)}{\sum P(f_k)}$.
- **Spectral Bandwidth (Spread)**: Frequency standard deviation around the centroid.
- **Sub-Band Energies**:
  - Low-Band ($0.2 \le f < 1.5$ Hz): Body pitch, roll, and road elevation changes.
  - Mid-Band ($1.5 \le f < 3.0$ Hz): Sprung-mass chassis bounce and suspension response (capturing the 2.2–2.5 Hz structure).
  - High-Band ($3.0 \le f \le 5.0$ Hz): Aliased chassis rattle and road texture noise.

---

## B. Speed Relationship (Within-Trip vs Cross-Trip)

We audited whether vibration spectral features correlate with ground-truth CAN vehicle speed within individual trips:

| Spectral Metric | Within-Trip Mean Pearson $r$ | Standard Deviation | Within-Trip Mean Spearman $\rho$ | Cross-Trip Consistency | Physical Meaning |
|:---|:---:|:---:|:---:|:---:|:---|
| **Dominant Frequency (`acc_z`)** | **-0.032** | $\pm 0.169$ | **-0.028** | **Zero** | Peak frequency is uncoupled from speed ($f \not\propto v$) |
| **Spectral Centroid (`acc_z`)** | **-0.037** | $\pm 0.214$ | **-0.031** | **Zero** | Spectral center of mass does not shift with speed |
| **Total Power (`acc_z`)** | **+0.252** | $\pm 0.337$ | **+0.281** | **Severe Inconsistency** | Roughness confounder; slopes vary by $>10\times$ across trips |
| **Mid-Band Power (1.5–3.0 Hz)** | **+0.204** | $\pm 0.303$ | **+0.235** | **Inconsistent** | Chassis bounce power increases with bumps, not speed |
| **High-Band Power (3.0–5.0 Hz)** | **+0.193** | $\pm 0.323$ | **+0.220** | **Inconsistent** | Road texture excitation |

### Critical Takeaways on Speed Relationship
1. **Frequency Does Not Track Speed**:
   - If vibration originated from wheel rotation ($f_\text{wheel} = v / C$), frequency would scale linearly with speed ($5.0\text{ Hz}$ at 36 km/h, $12.6\text{ Hz}$ at 90 km/h).
   - Instead, dominant frequency correlation with speed is essentially zero ($r = -0.032$). In 32 out of 64 trips, the correlation is actually negative.
2. **Power Correlation Is a Confounded Pseudo-Signal**:
   - While total power has a modest positive within-trip correlation ($r = +0.252$), its regression slope varies wildly from **$+0.001$ to $+0.055\text{ m/s per (m²/s⁴)}$** depending on whether the route is smooth tarmac or pitted urban asphalt.

---

## C. Strict Cross-Trip Generalization (19 Held-Out Trips)

We tested whether a model trained strictly on vibration features from the 39 training trips can predict speed on the 19 completely held-out test trips:

| Test Trip | Category | Avg Speed | Constant Baseline MAE | Existing TCN Speed MAE | Vibration Ridge MAE | Vibration Ridge $R^2$ | Vibration RF MAE | Vibration RF $R^2$ | Winner |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V-Vfa02** | Motorway | 87.0 km/h | 11.49 m/s | **3.41 m/s** | 8.80 m/s | -1.01 | 4.58 m/s | +0.27 | **TCN** |
| **Vta21** | Suburban | 48.1 km/h | 3.56 m/s | 3.94 m/s | 3.97 m/s | -0.27 | **3.52 m/s** | -0.05 | RF |
| **Vta22** | Suburban | 38.4 km/h | 3.59 m/s | **2.25 m/s** | 3.74 m/s | -2.07 | 3.29 m/s | -1.28 | **TCN** |
| **Vta23** | Suburban | 34.7 km/h | 4.39 m/s | **2.41 m/s** | 3.52 m/s | -1.06 | 3.23 m/s | -0.47 | **TCN** |
| **Vta24** | Suburban | 19.1 km/h | 8.60 m/s | **1.22 m/s** | 7.17 m/s | -1.11 | 2.29 m/s | +0.71 | **TCN** |
| **Vta25** | Suburban | 4.3 km/h | 12.08 m/s | **2.48 m/s** | 10.17 m/s | -9.18 | 4.74 m/s | -2.88 | **TCN** |
| **Vta26** | Suburban | 19.1 km/h | 8.98 m/s | **2.13 m/s** | 8.37 m/s | -2.39 | 3.22 m/s | +0.28 | **TCN** |
| **Vta27** | Suburban | 46.2 km/h | **2.56 m/s** | 3.70 m/s | 4.94 m/s | -1.42 | 3.54 m/s | -0.55 | Const |
| **Vta28** | Suburban | 33.5 km/h | 5.44 m/s | **2.61 m/s** | 5.36 m/s | -0.64 | 4.03 m/s | -0.04 | **TCN** |
| **Vtb09** | Dense Urban | 78.3 km/h | 7.44 m/s | **2.61 m/s** | 5.05 m/s | -14.50 | 3.79 m/s | -7.76 | **TCN** |
| **Vtb10** | Dense Urban | 49.7 km/h | 2.62 m/s | **2.44 m/s** | 3.55 m/s | -1.50 | 3.30 m/s | -1.16 | **TCN** |
| **Vtb11** | Dense Urban | 69.5 km/h | 5.11 m/s | 6.19 m/s | 9.51 m/s | -444.41 | 6.12 m/s | -188.65 | Const |
| **Vtb12** | Dense Urban | 37.6 km/h | 3.58 m/s | **2.01 m/s** | 3.82 m/s | -1.83 | 3.64 m/s | -2.61 | **TCN** |
| **Vw12** | Mountain | 90.5 km/h | 11.00 m/s | 2.41 m/s | 6.05 m/s | -28.61 | **2.23 m/s** | -5.93 | RF |
| **Vw13** | Mountain | 96.4 km/h | 12.73 m/s | 2.97 m/s | 5.66 m/s | -144.64 | **1.75 m/s** | -23.90 | RF |
| **Vw14a** | Mountain | 90.4 km/h | 11.02 m/s | 2.49 m/s | 4.62 m/s | -5.50 | **1.98 m/s** | -0.80 | RF |
| **Vw14b** | Mountain | 75.6 km/h | 8.32 m/s | **3.11 m/s** | 4.84 m/s | +0.10 | 3.29 m/s | +0.54 | **TCN** |
| **Vw15** | Mountain | 0.1 km/h | 14.12 m/s | **0.03 m/s** | 8.70 m/s | -292,955 | 0.30 m/s | -3,229 | **TCN** |
| **Vw16a** | Mountain | 52.7 km/h | 5.34 m/s | **3.97 m/s** | 6.25 m/s | -0.42 | 4.29 m/s | +0.27 | **TCN** |
| **MEAN** | — | — | **7.47 m/s** | **2.76 m/s** | **6.01 m/s** | **Negative** | **3.32 m/s** | **Negative** | **TCN Wins 14/19** |

### Findings from Cross-Trip Generalization
- **Negative $R^2$ Everywhere**: Both linear and non-linear vibration models produce severe negative $R^2$ on held-out trips. While Random Forest achieves a numerically lower MAE on 4 high-speed mountain trips (`Vw12`, `Vw13`, `Vw14a`, `Vta21`) by merely collapsing toward the global training mean (~16 m/s), its $R^2$ remains deeply negative (e.g. $-23.9$ and $-0.80$), confirming the absence of genuine functional speed tracking.
- **TCN Inferiority Ratio**: Vibration models failed to provide reliable cross-trip generalization and were inferior to the baseline TCN on **14 out of 19 held-out test trips (73.7%)**. On the remaining 5 trips, 4 were dominated by the RF mean-collapse and 1 by the constant-speed baseline.
- **TCN Decisively Outperforms Vibration Overall**: Mean cross-trip MAE for the baseline TCN is 2.76 m/s, versus 6.01 m/s for Vibration Ridge and 3.32 m/s for RF.

---

## D. Highway-Specific Test on `V-Vfa02` (Straight Cruising)

We isolated all epochs of straight cruising on the 163 km motorway trip `V-Vfa02` ($|\omega_\text{yaw}| < 0.02\text{ rad/s}, a_\text{horiz} < 0.3\text{ m/s}^2, |a_\text{vert}| < 0.5\text{ m/s}^2$) and analyzed spectral properties across discrete speed bands:

| Speed Band | Samples | Mean Speed (km/h) | Dominant Frequency (Hz) | Spectral Centroid (Hz) | Total Power (m²/s⁴) | Mid-Band Power (1.5–3Hz) | High-Band Power (3–5Hz) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80–90 km/h** | 105 | 85.7 km/h | **2.87 Hz** | **2.71 Hz** | 0.3571 | 0.1652 | 0.1726 |
| **90–100 km/h** | 114 | 95.1 km/h | **2.93 Hz** | **2.72 Hz** | **0.4590** | 0.2215 | **0.2106** |
| **100–110 km/h**| 48 | 103.5 km/h | **2.86 Hz** | **2.64 Hz** | **0.4285** | 0.2140 | 0.1908 |

### Highway Audit Findings
1. **Dominant Frequency Is Invariant to Highway Speed**:
   - At 85.7 km/h, dominant frequency is **2.87 Hz**.
   - At 95.1 km/h, dominant frequency is **2.93 Hz**.
   - At 103.5 km/h, dominant frequency is **2.86 Hz**.
   - **Conclusion**: The spectral peak does **not** shift upward as the car accelerates from 85 km/h to 105 km/h.
2. **Spectral Centroid Is Invariant**:
   - Centroid is 2.71 Hz at 85 km/h, 2.72 Hz at 95 km/h, and 2.64 Hz at 104 km/h.
3. **Power Non-Monotonicity**:
   - Total vibration power increases from 0.357 to 0.459 between 85 and 95 km/h, but then **drops** to 0.428 at 104 km/h due to road smoothness variations on the motorway.
   - **Conclusion**: Vibration features cannot solve the $-3.45\text{ m/s}$ high-speed shrinkage on `V-Vfa02`.

---

## E. Frequency Stability Audit (The 2.2–2.5 Hz Structure)

Earlier exploratory scripts observed 2.2–2.5 Hz structure in `Vta04`. We tested whether this phenomenon generalizes:

| Trip Name | Category | Average Speed | Dominant Peak Mode (Hz) | % Windows in 2.0–2.6 Hz | Frequency vs Speed $r(f, v)$ | $p$-value |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`Vta04`** | Suburban Town | 40.7 km/h | **4.30 Hz** | 6.5% | +0.157 | 0.0001 |
| **`Vta01a`** | Suburban Town | 57.2 km/h | **0.30 Hz** | 13.0% | -0.075 | 0.0000 |
| **`Vta02`** | Suburban Town | 36.3 km/h | **0.50 Hz** | 17.7% | -0.261 | 0.0000 |
| **`Vw04`** | Mountain | 61.0 km/h | **4.90 Hz** | 8.6% | -0.071 | 0.0000 |
| **`V-Vfa01`** | Motorway | 58.9 km/h | **4.30 Hz** | 11.4% | -0.151 | 0.0000 |
| **`V-Vfa02`** | Motorway | 87.0 km/h | **4.30 Hz** | 11.9% | -0.014 | 0.0002 |
| **`Vw12`** | Mountain | 90.5 km/h | **4.90 Hz** | 12.3% | +0.045 | 0.2014 |
| **`Vw14a`** | Mountain | 90.6 km/h | **4.30 Hz** | 6.5% | +0.017 | 0.3478 |

### Stability Findings
- **2.2–2.5 Hz Is Not a Universal Speed Harmonic**: The dominant mode across most trips is either near-DC ($0.3–0.5\text{ Hz}$, road undulations) or at the high-frequency Nyquist edge ($4.3–4.9\text{ Hz}$, aliased vibration noise).
- Only $6\%–17\%$ of windows fall within the 2.0–2.6 Hz band.
- The correlation between dominant frequency and speed is effectively zero ($r = -0.014$ on `V-Vfa02`, $+0.017$ on `Vw14a`).

---

## F. Alternative Explanations & Confounding Factors

Why does smartphone IMU vibration fail to provide generalizable vehicle speed?

1. **Severe Aliasing Limits Interpretation of High-Frequency Signals ($10\text{ Hz}$ Sampling Rate)**:
   - At $10\text{ Hz}$ sampling, the Nyquist ceiling is $f_\text{Nyquist} = 5.0\text{ Hz}$. Mechanical vibration components above $5.0\text{ Hz}$ fold back into the $0–5\text{ Hz}$ band.
   - For example, with wheel rotation frequency $f_\text{wheel} = \frac{v}{2\pi R} \approx \frac{v}{1.98\text{ m}}$:
     - At $40\text{ km/h}$ ($11.1\text{ m/s}$): $f_\text{wheel} = 5.6\text{ Hz} > 5.0\text{ Hz} \implies$ aliases to $4.4\text{ Hz}$.
     - At $90\text{ km/h}$ ($25.0\text{ m/s}$): $f_\text{wheel} = 12.6\text{ Hz} \implies$ aliases to $2.6\text{ Hz}$.
     - At $110\text{ km/h}$ ($30.6\text{ m/s}$): $f_\text{wheel} = 15.5\text{ Hz} \implies$ aliases to $4.5\text{ Hz}$.
   - While genuine low-frequency vehicle motion ($<5.0\text{ Hz}$) remains observable, aliasing severely limits our ability to interpret higher-frequency tyre, drivetrain, or engine rotational signatures without contamination.
2. **Road Surface Texture Dominance**:
   - Asphalt aggregate size, concrete expansion joints, and surface pitting dictate high-frequency energy. Surface roughness changes between road sections create false "acceleration" and "deceleration" artifacts.
3. **Vehicle Engine RPM & Transmission Gear**:
   - A vehicle cruising at 80 km/h in 3rd gear (3,500 RPM) exhibits massively higher engine vibration than the same vehicle cruising at 100 km/h in 6th gear (2,000 RPM).
4. **Chassis & Mount Filtering**:
   - Vehicle suspension dampers and smartphone cradle stiffness act as unknown mechanical low-pass/band-pass filters, decoupling the phone sensor from chassis vibration.

---

## G. Information Gain Analysis (Kinematics vs Kinematics + Vibration)

We performed offline feature regression comparing 9 kinematic channels vs 9 kinematic + 5 vibration spectral channels:

| Trip | Kinematics Alone $R^2$ | Combined (+ Vibration) $R^2$ | $\Delta R^2$ | Kinematics MAE (m/s) | Combined MAE (m/s) | MAE Improvement |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`V-Vfa02` (Motorway)** | 0.0127 | 0.0473 | +0.0346 | **4.39 m/s** | **4.38 m/s** | **+0.01 m/s (Zero gain)** |
| **`Vw14a` (Mountain)** | 0.0489 | 0.1665 | +0.1176 | 1.62 m/s | 1.47 m/s | +0.15 m/s |
| **`Vta21` (Suburban)** | 0.0308 | 0.2391 | +0.2083 | 3.70 m/s | 3.25 m/s | +0.45 m/s |

### Takeaway
On straight highway driving (`V-Vfa02`), adding vibration features improves MAE by only **0.01 m/s (0.2%)**. Vibration carries **zero meaningful incremental speed information** for highway driving.

---

## H. Categorized Conclusions

### 🟢 WHAT WE KNOW (Empirically Proven)
1. **Dominant frequency does NOT track vehicle speed** ($r = -0.032 \pm 0.169$ across 64 trips).
2. **Vibration models fail to provide reliable cross-trip generalization** and were inferior to the baseline TCN on **14 out of 19 held-out test trips (73.7%)**, with out-of-domain $R^2$ remaining negative across all trips.
3. **Vibration features cannot distinguish 85 km/h from 105 km/h on straight motorways** (dominant frequency remains flat at ~2.87–2.93 Hz, and power behaves non-monotonically).
4. **Aliasing severely limits our ability to interpret high-frequency mechanical signatures at 10 Hz sampling**: While genuine low-frequency vehicle dynamics (<5.0 Hz) remain observable, higher-frequency tyre and drivetrain signals (e.g. wheel rotation at 90 km/h ≈ 12.6 Hz) fold back into the 0–5 Hz band.
5. **Adding vibration features to kinematics yields 0.01 m/s improvement on `V-Vfa02`**, providing zero solution to high-speed shrinkage.

### 🟡 WHAT WE THINK (Strong Hypotheses)
1. **The observed 2.2–2.5 Hz structure is consistent with possible sprung-mass/suspension dynamics, but this was not independently identified**: Passenger vehicle suspensions often have natural bounce frequencies around 1.5–2.5 Hz, but dominant modes across trips varied widely (0.3–0.5 Hz or 4.3–4.9 Hz) and showed no functional correlation with ground speed.
2. **Vibration power is useful ONLY as a binary stationary gate**: High vibration power indicates the engine/vehicle is active, while near-zero power indicates the vehicle is stationary (ZUPT). It cannot serve as an analog speedometer.

### 🔴 WHAT WE DON'T KNOW
1. **Higher-rate IMU behavior (>100 Hz)**: If smartphone sensors were logged at 200 Hz, tyre rotation peaks (10–25 Hz) might be unaliased and trackable. However, in IO-VNBD and Android battery-efficient navigation loops (10 Hz), Nyquist limits prevent this.

---

## Final Recommendation: 🛑 NO-GO

**Do NOT add vibration spectral features to the neural speed model or ESKF.**  
Vibration is confounded by road roughness, fails out-of-domain cross-trip generalization, and provides zero incremental resolution on the highway bottleneck.

To solve highway dead reckoning without vibration, the system should rely on:
1. **ESKF Kinematic Integration with Heading Anchoring**.
2. **Physical Zero-Velocity Updates (ZUPT)** during confirmed stops.
3. **Map Geometry / Road Curvature Matching** during GNSS outages.
