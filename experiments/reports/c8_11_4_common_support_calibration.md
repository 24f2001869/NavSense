# Stage C8-11.4: Common-Support Rolling Calibration & Uncertainty Audit
## Diagnostic Report

---

## Executive Summary

This audit tests whether the apparent superiority of 5–10 s rolling magnetic calibration reported in C8-11.3 is genuine or an artifact of different outage subsets and overlapping simulated outages. The answer is unambiguous:

> **The 5–10 s advantage reported in C8-11.3 was an artifact of subset selection bias.**

When evaluated on a **common-support outage set** (where ALL six window lengths have valid calibration data), the performance differences between 5 s, 10 s, 20 s, and 30 s windows shrink to **< 0.5°** on Vta02. The apparent "optimal" window shifts from 5–10 s to **20–30 s** on common support. The proposed dynamic $R_\psi$ model **fails cross-validation** and the uncertainty model is **overconfident**.

---

## Task 1 — Common-Support Window Comparison

### 1.1 Common-Support Outage Selection

For each of the 59 (Vta02) / 9 (Vta04) dense candidate outages, we checked whether ALL six window lengths ($W \in \{5, 10, 20, 30, 60, 120\}$ s) had $\ge 10$ valid straight calibration samples. Only outages satisfying this criterion for ALL windows were included.

| Trip | Dense Candidates | Common-Support Outages | Excluded | Exclusion Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Vta02** | 59 | **13** | 46 | 78.0% |
| **Vta04** | 9 | **4** | 5 | 55.6% |

> [!WARNING]
> **78% of Vta02 outages were excluded** from common-support because the 5 s window lacked sufficient valid straight samples. The C8-11.3 "5 s is best" result was evaluated on a **biased subset** of outages where the 5 s window happened to have enough data — typically locations with sustained straight driving, which are intrinsically easier to calibrate.

### 1.2 Common-Support Window Performance

#### Table 1.2a: Vta02 Common-Support (N = 13)
| Window | Mean MAE | Median MAE | RMSE | P95 | Max |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 s | $9.98°$ | $10.41°$ | $11.43°$ | $19.59°$ | $20.01°$ |
| 10 s | $9.50°$ | $9.49°$ | $10.95°$ | $20.13°$ | $20.32°$ |
| **20 s** | **$9.49°$** | $9.95°$ | $11.04°$ | $20.34°$ | $22.07°$ |
| **30 s** | **$9.46°$** | **$9.70°$** | **$11.01°$** | $20.49°$ | $22.07°$ |
| 60 s | $10.45°$ | $9.85°$ | $11.68°$ | $20.51°$ | $21.86°$ |
| 120 s | $11.63°$ | $9.65°$ | $13.24°$ | $22.41°$ | $26.26°$ |

#### Table 1.2b: Vta04 Common-Support (N = 4)
| Window | Mean MAE | Median MAE | RMSE | P95 | Max |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 s | $19.59°$ | $18.45°$ | $20.03°$ | $25.21°$ | $26.04°$ |
| 10 s | $22.08°$ | $18.55°$ | $23.53°$ | $33.50°$ | $35.79°$ |
| 20 s | $20.15°$ | $16.99°$ | $21.33°$ | $29.86°$ | $32.12°$ |
| **30 s** | **$17.89°$** | **$17.12°$** | **$18.15°$** | **$22.01°$** | **$22.84°$** |
| 60 s | $18.65°$ | $17.41°$ | $19.33°$ | $25.29°$ | $26.14°$ |
| 120 s | $19.39°$ | $17.96°$ | $20.58°$ | $28.24°$ | $29.29°$ |

### 1.3 Temporal Integrity Verification
- **max($t_{\text{calib}}$) < $t_{\text{outage\_start}}$** verified for every outage and every window.
- Zero leakage violations.

---

## Task 2 — Non-Overlapping Outage Subset

Dense outages placed every 15 s are heavily overlapping (30 s outage on a 15 s grid). We applied a deterministic greedy selection requiring $\ge 30$ s separation between successive outage starts.

### Table 2.1: Non-Overlapping Performance

#### Vta02 Non-Overlapping (N = 11)
| Window | Mean MAE | Median MAE | RMSE | P95 | Max |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 s | $9.01°$ | $8.91°$ | $10.37°$ | $17.65°$ | $19.30°$ |
| **10 s** | **$8.50°$** | $9.45°$ | **$9.81°$** | $15.66°$ | $20.32°$ |
| 20 s | $8.56°$ | $8.13°$ | $10.07°$ | $16.49°$ | $22.07°$ |
| **30 s** | **$8.50°$** | **$7.81°$** | $9.98°$ | $16.49°$ | $22.07°$ |
| 60 s | $9.66°$ | $9.67°$ | $10.83°$ | $17.45°$ | $21.86°$ |
| 120 s | $11.07°$ | $9.37°$ | $12.76°$ | $22.82°$ | $26.26°$ |

#### Vta04 Non-Overlapping (N = 2)
| Window | Mean MAE | Median MAE |
| :---: | :---: | :---: |
| 5 s | $15.91°$ | $15.91°$ |
| 10 s | $16.01°$ | $16.01°$ |
| **20 s** | **$15.71°$** | **$15.71°$** |
| 30 s | $15.71°$ | $15.71°$ |
| 60 s | $19.89°$ | $19.89°$ |
| 120 s | $21.45°$ | $21.45°$ |

> [!IMPORTANT]
> **On non-overlapping outages, 10 s and 30 s are tied at $8.50°$ mean MAE on Vta02.** The difference across the 5–30 s range is only $0.51°$. On Vta04, only 2 non-overlapping outages survived — this is statistically meaningless.

---

## Task 3 — Short-Window Estimator Comparison

Testing whether estimator choice matters at 5 s and 10 s on the common-support outage set:

### Table 3.1: Estimator Comparison (Vta02 Common-Support, N = 13)
| Window | Circ. Mean | Circ. Median | Trimmed Mean (15%) | Huber M |
| :---: | :---: | :---: | :---: | :---: |
| **5 s** | $9.98°$ | $10.19°$ | $10.29°$ | $10.02°$ |
| **10 s** | $9.50°$ | $9.40°$ | $9.39°$ | $9.47°$ |

### Table 3.2: Estimator Comparison (Vta04 Common-Support, N = 4)
| Window | Circ. Mean | Circ. Median | Trimmed Mean (15%) | Huber M |
| :---: | :---: | :---: | :---: | :---: |
| **5 s** | $19.59°$ | $19.16°$ | $19.21°$ | $19.68°$ |
| **10 s** | $22.08°$ | $20.19°$ | $21.25°$ | $22.18°$ |

🟢 **WHAT WE KNOW**: Estimator choice does not materially affect short-window performance. Maximum spread across estimators is $0.31°$ at 5 s and $0.11°$ at 10 s on Vta02. The circular mean is adequate.

---

## Task 4 — Calibration Sample Count

### Table 4.1: Sample Counts (Vta02 Common-Support, N = 13)
| Window | Raw Samples | Valid Straight [Min, Median, Max] | Fraction Accepted [Min, Mean, Max] |
| :---: | :---: | :---: | :---: |
| 5 s | 50 | [10, 21, 42] | [0.20, 0.46, 0.84] |
| 10 s | 100 | [11, 39, 75] | [0.11, 0.37, 0.75] |
| 20 s | 200 | [17, 49, 114] | [0.09, 0.27, 0.57] |
| 30 s | 300 | [17, 57, 121] | [0.06, 0.21, 0.40] |
| 60 s | 600 | [17, 87, 225] | [0.03, 0.15, 0.38] |
| 120 s | 1200 | [25, 132, 264] | [0.02, 0.11, 0.22] |

> [!IMPORTANT]
> **At 5 s, the minimum valid straight sample count is only 10** (the threshold itself). This means some outages are calibrated on barely 1 second of truly straight driving. At 10 s, the minimum is 11 — only marginally better. However, the median sample count at 10 s (39 samples = 3.9 s of data) is adequate for circular averaging.

🟡 **WHAT WE THINK**: The 5 s window operates at the margin of sample sufficiency. It passes on common-support outages by definition (since we require $\ge 10$ samples), but may fail in conditions with fewer straight segments.

---

## Task 5 — Window Age / Spatial Lag

### Table 5.1: Age and Spatial Lag vs. Outage Error (Vta02)
| Window | Mean Age (s) | Mean Spatial Lag (m) | Mean MAE | $r$(age, MAE) | $r$(dist, MAE) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 s | 2.2 | 14 | $9.98°$ | +0.282 | +0.219 |
| 10 s | 3.6 | 26 | $9.50°$ | +0.022 | −0.199 |
| 20 s | 6.5 | 53 | $9.49°$ | +0.554 | +0.443 |
| 30 s | 8.3 | 76 | $9.46°$ | +0.447 | +0.025 |
| 60 s | 17.8 | 157 | $10.45°$ | +0.510 | +0.337 |
| 120 s | 35.3 | 268 | $11.63°$ | +0.726 | +0.268 |

### Pooled Correlations (All Windows, Vta02):
- **MAE vs. mean calibration age**: $r = +0.373$ ($p = 0.0008$), $\rho = +0.290$
- **MAE vs. spatial lag to oldest sample**: $r = +0.215$ ($p = 0.059$), $\rho = +0.204$

🟢 **WHAT WE KNOW**: Calibration age is a statistically significant positive predictor of outage error ($p < 0.001$ pooled). Older calibration data does produce worse outage headings. However, within any single window length, the age variable is relatively constant and the intra-window correlation is weaker.

🟡 **WHAT WE THINK**: The age correlation is dominated by the **between-window** contrast (short windows = young data = lower MAE), not within-window variation. This is consistent with spatial non-stationarity of the magnetic environment, but does not prove that temporal freshness per se is the mechanism — it could equally be spatial proximity.

🔴 **WHAT WE DON'T KNOW**: Whether the age effect is temporal (sensor drift) or spatial (magnetic environment change). The two are confounded in a moving vehicle.

---

## Task 6 — Outage Error Distribution

### Table 6.1: Error Distribution (Vta02 Common-Support, N = 13)
| Window | Median | IQR | MAD | P90 | P95 | Max | $> 20°$ count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 s | $10.41°$ | $4.21°$ | $3.62°$ | $18.64°$ | $19.59°$ | $20.01°$ | 1/13 (8%) |
| 10 s | $9.49°$ | $4.41°$ | $2.21°$ | $18.21°$ | $20.13°$ | $20.32°$ | 2/13 (15%) |
| 20 s | $9.95°$ | $4.36°$ | $2.68°$ | $17.53°$ | $20.34°$ | $22.07°$ | 1/13 (8%) |
| 30 s | $9.70°$ | $4.36°$ | $2.42°$ | $17.73°$ | $20.49°$ | $22.07°$ | 1/13 (8%) |
| 60 s | $9.85°$ | $4.49°$ | $2.57°$ | $18.29°$ | $20.51°$ | $21.86°$ | 1/13 (8%) |
| 120 s | $9.65°$ | $4.87°$ | $2.40°$ | $19.74°$ | $22.41°$ | $26.26°$ | 1/13 (8%) |

🟢 **WHAT WE KNOW**: The error distributions are remarkably similar across windows 5–60 s. The IQR is approximately $4.2°$–$4.9°$ for all windows. No single catastrophic outage dominates the mean. The 120 s window is the only one with clearly elevated tail risk (max $26.26°$).

---

## Task 7 — Dynamic $R_\psi$ Audit

### 7.1 Univariate Predictive Performance (Vta02, N = 13)

| Predictor | Pearson $r$ | $p$-value | Spearman $\rho$ | Significant? |
| :--- | :---: | :---: | :---: | :---: |
| $\sigma_\psi$ (window residual) | +0.172 | 0.573 | +0.291 | No |
| $\sigma_B$ (field noise) | +0.317 | 0.291 | +0.346 | No |
| $|\mu_B - B_0| / B_0$ (field shift) | +0.139 | 0.650 | +0.049 | No |
| Mean speed | +0.105 | 0.734 | +0.148 | No |
| $N_{\text{valid}}$ (sample count) | −0.001 | 0.998 | +0.237 | No |
| Oldest sample age | +0.492 | 0.087 | +0.497 | Marginal |

🔴 **No proposed predictor achieves $p < 0.05$ significance.** The strongest univariate predictor is calibration sample age ($r = +0.492$, $p = 0.087$), which is marginally significant.

### 7.2 Multivariate LOO Cross-Validation

| Metric | Value |
| :--- | :---: |
| LOO $r$(predicted, actual) | +0.172 |
| LOO RMSE | $6.64°$ |
| Baseline RMSE (predict mean) | $5.22°$ |
| In-sample $R^2$ | 0.577 |

> [!CAUTION]
> **The multivariate model is worse than predicting the mean.** LOO RMSE ($6.64°$) **exceeds** baseline RMSE ($5.22°$). The in-sample $R^2 = 0.577$ is entirely overfit — it does not generalize. This is classic overfitting on 13 data points with 6 predictors.

🔴 **WHAT WE DON'T KNOW**: Whether the proposed $R_\psi$ predictors have any genuine predictive power. With $N = 13$ outages and 6 predictors, we cannot reliably learn a multivariate model. The evidence for adaptive $R_\psi$ is **insufficient**.

---

## Task 8 — Uncertainty Calibration

Testing whether the C8-11.3 proposed covariance model produces calibrated uncertainty intervals, evaluated on non-overlapping outages:

### Coverage Results (Vta02, N = 11 non-overlapping outages)
| Metric | Observed | Expected |
| :--- | :---: | :---: |
| Within $\pm 1\sigma$ | 45.5% (5/11) | ~68.3% |
| Within $\pm 1.96\sigma$ | 81.8% (9/11) | ~95.0% |
| Median predicted $1\sigma$ | $6.88°$ | — |
| Median actual $|$error$|$ | $9.67°$ | — |

**Verdict: OVERCONFIDENT**

The model predicts a median $1\sigma$ of $6.88°$, but the median actual absolute error is $9.67°$. Only 45.5% of errors fall within the predicted $\pm 1\sigma$ band instead of the expected ~68%.

🔴 **WHAT WE DON'T KNOW**: Whether any simple parametric model of $R_\psi$ can be calibrated on this dataset. With only 11 non-overlapping outages, the coverage test itself has wide confidence intervals. Vta04 had only 2 non-overlapping outages — completely insufficient.

---

## Task 9 — Cross-Trip Generalization

| Criterion | Vta02 | Vta04 | Agreement? |
| :--- | :---: | :---: | :---: |
| **Best CS Window (mean MAE)** | 30 s | 30 s | Yes |
| **Best NO Window (mean MAE)** | 10 s / 30 s (tied) | 20 s / 30 s (tied) | Partial |
| **Common-Support Outages** | 13 | 4 | — |
| **Non-Overlapping Outages** | 11 | 2 | — |
| **Statistical Power** | Moderate | **Weak** | — |

> [!WARNING]
> **Vta04 has only 4 common-support and 2 non-overlapping outages.** Any conclusion drawn from Vta04 alone is statistically unreliable. Cross-trip validation cannot be considered robust with this sample size.

🟡 **WHAT WE THINK**: Both trips agree that 30 s is among the top-performing windows on common-support, and that 120 s is consistently worst. But the small sample sizes prevent confident cross-trip ranking.

---

## Task 10 — Final Window Decision

### Evidence Summary

| Evidence Source | Supports Short (5–10 s)? | Supports Medium (20–30 s)? | Supports Long (60–120 s)? |
| :--- | :---: | :---: | :---: |
| **C8-11.3 Dense (biased subsets)** | Yes | — | No |
| **C8-11.4 Common-Support** | No | **Yes** | No |
| **C8-11.4 Non-Overlapping** | Partially (10 s tied with 30 s) | **Yes** | No |
| **Cross-Trip Agreement** | No | Partial (30 s) | No |
| **Sample Count Adequacy** | Marginal (min 10 at 5 s) | Good | Good |
| **Spatial Lag Concern** | Low lag | Moderate lag | High lag |

### Quantitative Rankings (Vta02)

| Ranking | Common-Support (mean MAE) | Non-Overlapping (mean MAE) |
| :---: | :--- | :--- |
| 1st | 30 s ($9.46°$) | 10 s / 30 s (tie, $8.50°$) |
| 2nd | 20 s ($9.49°$) | 20 s ($8.56°$) |
| 3rd | 10 s ($9.50°$) | 5 s ($9.01°$) |
| 4th | 5 s ($9.98°$) | 60 s ($9.66°$) |
| 5th | 60 s ($10.45°$) | 120 s ($11.07°$) |
| 6th | 120 s ($11.63°$) | — |

- **Top-1 vs Top-2 gap on common-support**: $0.03°$ (essentially zero)
- **Top-1 vs Top-4 gap**: $0.52°$ (negligible)
- **Only 120 s is clearly worse** (by $\sim 2°$)

### Decision

> **DECISION: D — No fixed window is justified yet.**

The top-1 to top-4 spread on common-support is **$0.52°$**, which is statistically indistinguishable given $N = 13$ outages. Any window between 5 s and 60 s produces **$9.5°$–$10.5°$ mean MAE** on Vta02. The only robust conclusion is:

1. **120 s is too long** (elevated tail risk, spatial lag).
2. **5–60 s all perform comparably** on common-support outages.
3. **The C8-11.3 claim that "10 s is optimal" was an artifact of subset selection bias.**
4. **The proposed $R_\psi$ model is not validated** (LOO RMSE exceeds baseline, uncertainty is overconfident).

---

## Definitive Claims Table

| Claim | Status | Evidence |
| :--- | :---: | :--- |
| **5–10 s rolling window is superior** | 🔴 **ARTIFACT** | Common-support comparison shows 5–30 s windows differ by < $0.5°$; the C8-11.3 advantage was driven by biased subset selection. |
| **Rolling calibration outperforms initial frozen** | 🟢 **CONFIRMED** | Even on common-support, rolling (any window) outperforms initial frozen ($19.22°$) by ~$10°$. |
| **30 s is optimal** | 🟡 **TENTATIVE** | Lowest mean MAE on common-support ($9.46°$) but only $0.03°$ better than 20 s and $0.04°$ better than 10 s. |
| **120 s is too long** | 🟢 **CONFIRMED** | Consistently worst (mean $11.63°$, max $26.26°$) across all evaluation modes. |
| **Estimator choice matters for short windows** | 🔴 **REFUTED** | Maximum spread across 4 estimators is $0.31°$ at 5 s. |
| **$\sigma_B$ predicts outage error** | 🔴 **NOT CONFIRMED** | $r = +0.317$ ($p = 0.291$), not significant. |
| **$\sigma_\psi$ predicts outage error** | 🔴 **NOT CONFIRMED** | $r = +0.172$ ($p = 0.573$), not significant. |
| **Calibration age predicts outage error** | 🟡 **MARGINAL** | $r = +0.492$ ($p = 0.087$); marginally significant in pooled analysis ($p = 0.0008$) but confounded with between-window variation. |
| **Proposed $R_\psi$ model is validated** | 🔴 **FAILED** | LOO RMSE ($6.64°$) exceeds baseline ($5.22°$). Overfit on 13 points with 6 predictors. |
| **Uncertainty model is calibrated** | 🔴 **OVERCONFIDENT** | Only 45.5% of errors within $\pm 1\sigma$ (expected ~68%). Median predicted $\sigma = 6.88°$ vs actual $|$error$| = 9.67°$. |
| **Cross-trip validation is robust** | 🔴 **INSUFFICIENT** | Vta04 has 4 common-support and 2 non-overlapping outages. Cannot draw reliable cross-trip conclusions. |
