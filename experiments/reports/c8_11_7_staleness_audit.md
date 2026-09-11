# Stage C8-11.7: Calibration-State Integration & Staleness Stress Test Report

**Status:** Diagnostic Audit Complete — **Decision Hierarchy Exit: Reject Uncertainty Model / Diagnose Only**  
**Constraints Enforced:** Diagnostic Only (Zero production modifications; no ESKF redesign; no RF retraining; no threshold tuning)  
**Pre-Execution Manifest:** [`experiments/freeze_manifest_c8_11_6.json`](../freeze_manifest_c8_11_6.json)  
**Diagnostic Script:** [`experiments/audit_staleness_c8_11_7.py`](../audit_staleness_c8_11_7.py)  
**Machine-Readable Results:** [`results/c8_11_7_staleness_audit.json`](../../results/c8_11_7_staleness_audit.json)

---

## 1. Executive Summary & Objective

Stage **C8-11.7** tested the operational integration question:
> **"Once calibration is accepted, rejected, or retained, how should its age and uncertainty affect the navigation filter?"**

Following the disciplined three-part separation:
1. **Actual Heading-Offset Degradation:** Does calibration error genuinely grow as time/distance passes?
2. **Uncertainty Representation:** Can a parametric covariance-aging hypothesis construct an honest uncertainty envelope covering that degradation?
3. **Navigation-Filter Behavior:** Does that uncertainty envelope prevent the ESKF from becoming falsely overconfident?

```text
                                C8-11.7 Decision Hierarchy Result
                                                │
                                                ▼
                         [Q1: Actual Heading Degradation?]
                                        │
                                       YES (rho = +0.401, p = 0.0016)
                                        │
                                        ▼
                         [Q2: Candidate Aging Hypothesis Covers Error?]
                                        │
                                       NO (1-sigma coverage <= 40.7% vs 68.3% target)
                                        │
                                        ▼
                         [Decision: REJECT UNCERTAINTY MODEL]
                                        │
                                        ▼
                         [Q3: Filter Integration Status]
                                  DIAGNOSE ONLY
                        Do NOT manufacture covariance aging
                           for production ESKF pipeline.
```

---

## 2. Test 0 — Provenance & Baseline Reconstruction

Every candidate outage was reconstructed with its exact operational provenance, strictly separating **estimate age** ($\Delta t_{\text{est}} = t_{\text{outage}} - t_{\text{calib}}$) from **underlying data age** ($\Delta t_{\text{data}} = t_{\text{outage}} - (t_{\text{calib}} - W)$):

- **Vta02 (Highway/Arterial, $N=59$ Outages):**
  * Fresh Calibrations ($\Delta t_{\text{est}} = 0$ s, accepted immediately prior): **39 outages (66.1%)**.
  * Stale Calibrations ($\Delta t_{\text{est}} > 0$ s, retained across rejections): **20 outages (33.9%)**.
  * Maximum estimate staleness: **105.0 s** ($\Delta d = 294.7$ m).
- **Vta04 (Urban Loop, $N=9$ Outages):**
  * Fresh Calibrations ($\Delta t_{\text{est}} = 0$ s): **4 outages (44.4%)**.
  * Stale Calibrations ($\Delta t_{\text{est}} > 0$ s): **5 outages (55.6%)**.
  * Maximum estimate staleness: **30.0 s** ($\Delta d = 90.1$ m).

---

## 3. Tests 1, 2, & 3: Actual Heading Degradation vs. Age, Distance, & Environment

### Empirical Correlation Analysis

| Predictor | Physical Meaning | Vta02 Pearson $r$ ($p$) | Vta02 Spearman $\rho$ ($p$) | Vta04 Spearman $\rho$ ($p$) | Statistical Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Estimate Age ($\Delta t_{\text{est}}$)** | Elapsed seconds since last accepted update | $+0.239$ ($0.0688$) | **$+0.401$ ($0.0016$)** | $+0.378$ ($0.3162$) | **Statistically Significant** ($p < 0.01$) |
| **Distance Traveled ($\Delta d$)** | Cumulative path distance since calibration | $+0.231$ ($0.0781$) | **$+0.401$ ($0.0017$)** | $+0.339$ ($0.3715$) | **Statistically Significant** ($p < 0.01$) |
| **Cumulative Field Dev ($\Delta B_{\text{cum}}$)** | Environmental magnetic anomaly accumulation | **$+0.490$ ($0.0001$)** | **$+0.498$ ($0.0001$)** | $-0.533$ ($0.1392$) | **Highly Significant on Highway** |

### The Real Impact of Staleness: Fresh vs. Stale Outages

| Condition on Vta02 | Outage Count | Mean Outage MAE | Median Outage MAE | Minimum MAE | Maximum MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fresh Calibration** ($\Delta t = 0$ s) | 39 | **$10.97^\circ$** | **$6.80^\circ$** | $1.74^\circ$ | $35.15^\circ$ |
| **Stale Calibration** ($\Delta t > 0$ s) | 20 | **$17.11^\circ$** | **$16.38^\circ$** | $4.21^\circ$ | $30.28^\circ$ |
| **Staleness Penalty ($\Delta$)** | — | **$+6.14^\circ$ (+56%)** | **$+9.58^\circ$ (+141%)** | — | — |

> **Finding (Question 1 Confirmed):** Calibration error systematically increases as staleness accumulates. When updates are rejected, the retained calibration's median error more than doubles ($6.80^\circ \to 16.38^\circ$).

---

## 4. Tests 4 & 5: Diagnostic Covariance-Aging Hypotheses & Coverage Testing

We tested two distinct categories of diagnostic hypotheses against empirical Gaussian coverage targets ($1\sigma \approx 68.3\%$, $2\sigma \approx 95.4\%$):

### Category A: Sample Standard Error Based ($\sigma_0 = \text{cstd}/\sqrt{N} \approx 1.5^\circ$)
Using the circular standard error of the mean over the calibration window:

| Hypothesis | Parameterization | Mean $\sigma(t)$ | Max $\sigma(t)$ | $1\sigma$ Coverage (Target 68.3%) | $2\sigma$ Coverage (Target 95.4%) | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A0 (Static Sample)** | $\sigma(t) = \sigma_0$ | $1.51^\circ$ | $2.00^\circ$ | **0.0%** | **8.5%** | 🔴 **Massively Overconfident** |
| **A1 (Time-Linear)** | $\sigma^2(t) = \sigma_0^2 + 0.5 \Delta t$ | $2.73^\circ$ | $7.40^\circ$ | **1.7%** | **11.9%** | 🔴 **Massively Overconfident** |
| **A2 (Distance-Linear)** | $\sigma^2(d) = \sigma_0^2 + 0.2 \Delta d$ | $2.75^\circ$ | $7.82^\circ$ | **1.7%** | **13.6%** | 🔴 **Massively Overconfident** |

### Category B: Physical Sensor Floor Based ($\sigma_{\text{base}} = 8.0^\circ$)
Using a realistic empirical baseline uncertainty reflecting unmodeled in-vehicle magnetic disturbances:

| Hypothesis | Parameterization | Mean $\sigma(t)$ | Max $\sigma(t)$ | $1\sigma$ Coverage (Target 68.3%) | $2\sigma$ Coverage (Target 95.4%) | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **B0 (Static Physical)** | $\sigma(t) = 8.0^\circ$ | $8.00^\circ$ | $8.00^\circ$ | **37.3%** | 67.8% | 🔴 **Undercovers at $1\sigma$** |
| **B1 (Physical Time Aging)** | $\sigma^2(t) = 8.0^2 + (0.15 \Delta t)^2$ | $9.20^\circ$ | $17.67^\circ$ | **40.7%** | 79.7% | 🔴 **Undercovers at $1\sigma$** |
| **B2 (Physical Dist Aging)** | $\sigma^2(d) = 8.0^2 + (0.05 \Delta d)^2$ | $8.94^\circ$ | $16.77^\circ$ | **40.7%** | 79.7% | 🔴 **Undercovers at $1\sigma$** |
| **B3 (Physical Bimodal)** | Fresh: $8^\circ$, Stale: $\sqrt{8^2 + 10^2}$ | $9.63^\circ$ | $12.81^\circ$ | **42.4%** | 81.4% | 🔴 **Undercovers at $1\sigma$** |

### Why Gaussian Covariance Aging Fails (The Heavy-Tail Mechanism)
1. **Sample Noise $\neq$ Calibration Accuracy:** Sample standard error ($\approx 1.5^\circ$) only reflects white sensor noise over a clean straight line. It ignores the spatial field gradients along the route, vehicle chassis magnetization, and gyro dead-reckoning drift during the outage.
2. **Heavy Non-Gaussian Tails:** Outage heading errors are distinctly non-Gaussian: $50\%$ of fresh errors are under $6.8^\circ$, but the 95th percentile extends beyond $28^\circ–35^\circ$. A linear or quadratic Gaussian variance growth cannot simultaneously represent the tight mode without severely undercovering the tail.

> **Decision Hierarchy Outcome (Question 2):** **All candidate Gaussian aging hypotheses are REJECTED.** No simple parametric covariance growth law achieves honest $1\sigma$ coverage.

---

## 5. Test 6: Fresh Update Contraction Dynamics

When a fresh calibration is accepted after a period of rejection, how does the state transition behave?

- **Vta02 Update Events ($N=39$):**
  * Mean innovation jump $|\delta\psi_{\text{new}} - \delta\psi_{\text{old}}|$: **$5.22^\circ$** (Median = $1.94^\circ$).
  * Maximum innovation jump: **$32.68^\circ$** (occurred after a 105 s rejection run across a sharp highway interchange curve).
  * Maximum jump-to-prior-uncertainty ratio: **$21.8\times$** (if prior uncertainty had not been inflated).
- **Vta04 Update Events ($N=4$):**
  * Mean innovation jump: **$12.31^\circ$** (Median = $10.15^\circ$, Max = $25.73^\circ$).
  * Maximum jump-to-prior-uncertainty ratio: **$17.2\times$**.

*Implication:* If the navigation filter maintains a tight static uncertainty ($\sigma \approx 2^\circ$) during rejection, the arrival of a fresh valid calibration produces an extreme measurement residual ($>15\sigma$), causing innovation rejection or state shocks in a standard Kalman filter.

---

## 6. Test 7: Repeated Rejection & Degradation Stress

Tracking consecutive rejection run lengths $k$ on Vta02:

| Consecutive Rejections ($k$) | Outage Count | Mean Outage MAE | Maximum Outage MAE | Observed Operational Status |
| :---: | :---: | :---: | :---: | :--- |
| **$k = 1$** (15 s stale) | 6 | $16.96^\circ$ | $30.28^\circ$ | Initial divergence after turn |
| **$k = 2$** (30 s stale) | 3 | $20.97^\circ$ | $26.08^\circ$ | Peak degradation |
| **$k = 3$** (45 s stale) | 3 | $16.82^\circ$ | $18.50^\circ$ | Bounded by previous straight heading |
| **$k = 4$** (60 s stale) | 3 | $13.47^\circ$ | $16.91^\circ$ | Vehicle re-aligns with arterial corridor |
| **$k = 5$** (75 s stale) | 3 | $15.32^\circ$ | $16.49^\circ$ | Corridor stability holds |
| **$k = 6$** (90–105 s stale) | 2 | $20.34^\circ$ | $22.28^\circ$ | Spatial drift accumulation |

*Key Insight:* Error does not grow without bound to $90^\circ$ or $180^\circ$. On real roads, retained calibration error hovers in the **$15^\circ–22^\circ$ range** during sustained rejection sequences because vehicles travel along structured corridors.

---

## 7. Evidence Tiers & Scientific Assessment

### 🟢 WHAT WE KNOW (Empirically Confirmed)
1. **Calibration Staleness Systematically Degrades Heading Accuracy:** Estimate age ($\Delta t_{\text{est}}$) and distance traveled ($\Delta d$) correlate significantly with subsequent heading error ($\rho = +0.401$, $p = 0.0016$). Median error increases from $6.80^\circ$ when fresh to $16.38^\circ$ when stale ($+141\%$).
2. **Sample Standard Error is Massively Overconfident:** Using calibration window dispersion divided by $\sqrt{N}$ yields $\sigma_0 \approx 1.5^\circ$, which achieves **$0.0\%$ to $1.7\%$ empirical coverage at $1\sigma$**. Sample estimation variance must never be used as a standalone measurement covariance in an ESKF.
3. **Parametric Gaussian Aging Hypotheses Fail Coverage Testing:** Even assuming an empirical baseline floor ($\sigma_{\text{base}} = 8^\circ$), time-linear and distance-linear aging laws achieve only $40.7\%$ coverage at $1\sigma$ (far below the $68.3\%$ target) due to heavy non-Gaussian tails.
4. **Fresh Updates Produce Substantial Innovation Jumps:** Fresh calibrations arriving after extended rejections jump by an average of $5.2^\circ$ (and up to $32.7^\circ$), indicating that hard resetting without soft covariance transitions would destabilize a naive linear filter.

### 🟡 WHAT WE THINK (Empirically Motivated Findings)
1. **Hard Gating Outweighs Dynamic Covariance Inflation:** Rather than attempting to tune an artificial continuous covariance growth function $f(t)$ that fails coverage testing, the navigation filter should rely on **stateful quality gating**: using the heading measurement when valid, and decoupling/suspending magnetometer updates when invalid.
2. **Bounded Regional Variance Floor:** If a measurement variance must be supplied to an ESKF, it should be set to a conservative, empirical, heavy-tailed regional bound ($\sigma_\psi \approx 12^\circ–15^\circ$) rather than a dynamic formula that pretends to know the instantaneous uncertainty.

### 🔴 WHAT WE DON'T KNOW (Unresolved Limits)
1. **Long-Term Multi-Kilometer Drift Growth:** In trips exceeding 10–20 km without fresh calibration, spatial magnetic declination and regional gradient variations will eventually cause retained estimates to drift beyond $30^\circ$.

---

## 8. Final Decision Hierarchy Outcome

Following the pre-registered decision tree:

- **Question 1 (Actual Degradation):** **YES (🟢 Confirmed).** Calibration error reliably grows with age, distance, and magnetic anomalies.
- **Question 2 (Uncertainty Coverage):** **NO (🔴 All Hypotheses Rejected).** Parametric Gaussian covariance aging cannot honestly cover the heavy-tailed error distribution.
- **Question 3 (Filter Integration):** **DIAGNOSE ONLY / DO NOT INTEGRATE YET 🔴.**
  We do **not** manufacture covariance growth formulas for the ESKF. The filter should rely on the frozen C8-11.6 quality gating architecture, using binary acceptance/rejection rather than unvalidated continuous covariance scaling.
