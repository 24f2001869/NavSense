# Stage C10.8A.1: Recovery Metric Integrity Audit & Conditioned Typology

**Date**: 2026-09-09  
**Status**: Completed (Offline Audit Only — Production Code Strictly Frozen)  
**Objective**: Re-analyze the C10.8A replay logs with strict metric integrity, classifying the 14 blackout episodes into a mutually exclusive typology, calculating conditioned recovery statistics on the true eligible-return subset ($N_{\text{return}}$), and establishing post-recovery stability boundaries.

---

## 1. Executive Summary & Epistemic Corrections

In response to rigorous methodological review, Stage C10.8A.1 audits and corrects the metrics and claims made in Stage C10.8A:

1. **Correction on "14/14 Recovery"**:
   - In the initial C10.8A report, all 14 episodes were reported as "recovered".
   - **This was methodologically invalid**: In Trip 1 Ep 1 (3.9s) and Trip 1 Ep 7 (99.5s), physical eligibility was **0.0% throughout the outage**. The vehicle/phone was continuously turning or tumbling in hand; physical eligibility never returned.
   - An episode where physical eligibility never returned cannot be counted as a "recovery test".
   - **Corrected Statistic**: The true eligible-return subset is **$N_{\text{return}} = 12\text{ episodes}$**.
2. **Correction on "0.00 s Recovery Latency"**:
   - The metric has been renamed to **`eligibility_to_update_latency`**.
   - By construction, Variant D updates at the exact epoch physical eligibility transitions to True. Reporting 0.00 s latency reflects the implementation definition (zero artificial delay added after eligibility), not an empirical estimation of filter convergence time.
3. **Correction on "876.7 m/s Protected"**:
   - The phrase "prevented runaway / bounded safely" at $876.7\,\text{m/s}$ is retracted.
   - $876.7\,\text{m/s}$ (~3,156 km/h) is still severe vertical-state instability.
   - **Corrected Finding**: The physical eligibility detector substantially suppresses the catastrophic $72,181\,\text{m/s}$ explosion observed in the un-gated case, but substantial vertical instability remains under severe passenger phone tumbling.
4. **Correction on "3.0°/s Mathematically Confirmed Optimal"**:
   - The phrase "mathematically confirmed optimal" is retracted.
   - In the sensitivity sweep, turn threshold was varied simultaneously with persistence, gyro norm, and dynamic acceleration.
   - **Corrected Finding**: Among the tested multi-parameter configurations, the $3.0^\circ/\text{s}$ configurations performed better than the tested $2.0^\circ/\text{s}$ and $5.0^\circ/\text{s}$ configurations on this dataset.
5. **Horizontal Error Trade-Off**:
   - Median horizontal position error increased from **$3,234.9\,\text{m}$ (Control) $\to 5,153.0\,\text{m}$ (Variant D)**.
   - **Core Scientific Distinction**: $\mathbf{\text{VNHC recovery improvement} \neq \text{navigation accuracy improvement}}$. C10.8A is a vertical-channel architectural diagnostic, not a validated navigation improvement.

---

## 2. Mutually Exclusive Episode Typology

Every episode was categorized into four mutually exclusive behavioral types:

```
                                    BLACKOUT EPISODE TYPOLOGY (N = 14)
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
     [Type 1: Non-Return] (N = 2)                                          [Eligible-Return Subset] (N = 12)
   Eligibility remained 0% throughout;                                                 │
   phone continuously disturbed.                                    ┌──────────────────┴──────────────────┐
   EXCLUDED from recovery test.                                     ▼                                     ▼
                                                       [Type 2: Real Latch Test] (N = 6)       [Type 3: Both Accept] (N = 6)
                                                     Eligibility returned; Control latched;    Eligibility returned; residual was
                                                     Variant D admitted update.                small; both accepted update.
```

### Table 1: Complete Episode Classification & Conditioned Metrics

| Episode | Duration | Physical Elig % | Episode Typology | Residual at Return $|r_{\text{vert}}|$ | Control Status | Variant D Status | Control Final $\Delta U$ | Variant D Final $\Delta U$ | Post-Recovery Max $|v_u|$ (D) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip 1 Ep 1** | 3.9 s | 0.0% | **Type 1: Eligibility Never Returns** | N/A | Excluded | Excluded | $-142.4\,\text{m}$ | $-142.4\,\text{m}$ | N/A |
| **Trip 1 Ep 7** | 99.5 s | 0.0% | **Type 1: Eligibility Never Returns** | N/A | Excluded | Excluded | $-5,402.0\,\text{m}$ | $-21,123.6\,\text{m}$ | N/A |
| **Trip 1 Ep 5** | 125.1 s | 28.3% | **Type 2: Real Latch-Recovery Test** | **$6.54\,\text{m/s}$** | **Deadlocked (118s)** | **Accepted (0.0s)** | **$-41,851.9\,\text{m}$** | **$+132.0\,\text{m}$** | **$9.9\,\text{m/s}$** |
| **Trip 1 Ep 8** | 39.9 s | 4.8% | **Type 2: Real Latch-Recovery Test** | **$209.25\,\text{m/s}$** | **Deadlocked (32s)** | **Accepted (0.0s)** | $+1,039.6\,\text{m}$ | $+4,591.5\,\text{m}$ | $389.3\,\text{m/s}$ |
| **Trip 1 Ep 9** | 40.6 s | 80.9% | **Type 2: Real Latch-Recovery Test** | **$2.98\,\text{m/s}$** | **Deadlocked (30s)** | **Accepted (0.0s)** | $+509.3\,\text{m}$ | **$-70.1\,\text{m}$** | **$16.7\,\text{m/s}$** |
| **Trip 2 Ep 1** | 2.4 s | 9.5% | **Type 2: Real Latch-Recovery Test** | **$2.41\,\text{m/s}$** | **Deadlocked (2s)** | **Accepted (0.0s)** | $-7.3\,\text{m}$ | **$-6.5\,\text{m}$** | **$5.2\,\text{m/s}$** |
| **Trip 2 Ep 3** | 57.6 s | 86.7% | **Type 2: Real Latch-Recovery Test** | **$0.92\,\text{m/s}$** | **Deadlocked (53s)** | **Accepted (0.0s)** | $+365.7\,\text{m}$ | **$-12.4\,\text{m}$** | **$2.2\,\text{m/s}$** |
| **Trip 2 Ep 5** | 224.0 s | 3.1% | **Type 2: Real Latch-Recovery Test** | **$8.10\,\text{m/s}$** | **Deadlocked (61s)** | **Accepted (0.0s)** | $-16,058.5\,\text{m}$ | $-7,105.7\,\text{m}$ | $1,612.3\,\text{m/s}$ |
| **Trip 1 Ep 2** | 94.3 s | 29.1% | **Type 3: Both Accept** | $0.02\,\text{m/s}$ | Accepted | Accepted | $-657.6\,\text{m}$ | $-291.6\,\text{m}$ | $40.1\,\text{m/s}$ |
| **Trip 1 Ep 3** | 132.4 s | 63.4% | **Type 3: Both Accept** | $0.32\,\text{m/s}$ | Accepted | Accepted | $+709.3\,\text{m}$ | $+153.3\,\text{m}$ | $97.5\,\text{m/s}$ |
| **Trip 1 Ep 4** | 128.6 s | 44.7% | **Type 3: Both Accept** | $0.07\,\text{m/s}$ | Accepted | Accepted | $+358.0\,\text{m}$ | $+1,454.8\,\text{m}$ | $76.3\,\text{m/s}$ |
| **Trip 1 Ep 6** | 24.9 s | 20.2% | **Type 3: Both Accept** | $0.69\,\text{m/s}$ | Accepted | Accepted | $+166.5\,\text{m}$ | $-54.7\,\text{m}$ | $25.2\,\text{m/s}$ |
| **Trip 2 Ep 2** | 161.7 s | 24.4% | **Type 3: Both Accept** | $1.76\,\text{m/s}$ | Accepted | Accepted | $-36.0\,\text{m}$ | $+7,519.3\,\text{m}$ | $180.7\,\text{m/s}$ |
| **Trip 2 Ep 4** | 80.1 s | 62.6% | **Type 3: Both Accept** | $0.15\,\text{m/s}$ | Accepted | Accepted | $-373.5\,\text{m}$ | $-335.2\,\text{m}$ | $9.5\,\text{m/s}$ |

---

## 3. Conditioned Recovery Statistics

Excluding Type 1 non-return episodes, the true eligible-return subset comprises **$N_{\text{return}} = 12\text{ episodes}$**:

### Table 2: True Conditioned Recovery Performance

| Statistic | Definition / Condition | Variant A (Current Control) | Variant D (Eligibility Only) | Empirical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Eligible-Return Count ($N_{\text{return}}$)** | Outages where calm motion returned | 12 | 12 | True denominator for recovery |
| **Conditioned Recovery Rate** | Accepted update upon return | **7 / 12 (58.3%)** | **12 / 12 (100.0%)** | Control locked out in 41.7% of return episodes |
| **Type 2 Latch-Recovery Rate** | Solved the deadlocked cases | **0 / 6 (0.0%)** | **6 / 6 (100.0%)** | Variant D breaks the latch in 100% of deadlock tests |
| **Recovery with $|r_{\text{vert}}| > 3.0\,\text{m/s}$** | Successfully updated when $|r| > 3.0$ | **0 / 3 (0.0%)** | **3 / 3 (100.0%)** | Proves recovery works despite large accumulated drift |
| **Mean Eligibility-to-Update Latency** | Epochs from return to update | $0.00\,\text{s}$ (when accepted) | **$0.00\,\text{s}$** | Update admitted at return epoch by definition |

---

## 4. Post-Recovery Trajectory Forensic: Convergence vs Instability

Does admitting the measurement when physical eligibility returns smoothly pull the state back, or does it destabilize the filter?

### 4.1 Smooth Convergence (Trip 1 Episode 5)
- **The Event**: At $t \approx 17.6\,\text{s}$, a road bump caused vertical velocity to exceed $3.0\,\text{m/s}$.
- **Control**: Residual gate locked the filter. Vertical velocity climbed unchecked to $1,133\,\text{m/s}$, driving a $-41.8\,\text{km}$ runaway.
- **Variant D**: At $t = 28.20\,\text{s}$, physical eligibility returned. State residual was $|r_{\text{vert}}| = \mathbf{6.54\,\text{m/s}}$ ($> 3.0\,\text{m/s}$).
- **The Trajectory**:
  - $t = 28.20\,\text{s}$: $v_u = +6.54\,\text{m/s}$ $\to$ VNHC admitted with $r_{\text{vert}} = -6.54\,\text{m/s}$.
  - $t = 28.65\,\text{s}$ (next epoch): $v_u$ plummeted from $+6.54\,\text{m/s} \to \mathbf{+3.65\,\text{m/s}}$!
  - $t = 29.50\,\text{s}$: $v_u$ stabilized near zero ($\approx 0.1\,\text{m/s}$).
  - Position error remained pinned at $\approx 46–53\,\text{m}$, ending at **$+132.0\,\text{m}$** (vs $-41.8\,\text{km}$ in Control).

### 4.2 Destabilizing Edge Case (Trip 2 Episode 2 & Trip 1 Episode 8)
- In **Trip 2 Ep 2** (161.7s outage), passenger hand tremors caused repeated intermittent eligibility drops. While Control remained rejected and stayed near $-36.0\,\text{m}$, Variant D admitted multiple noisy updates, accumulating **$+7,519.3\,\text{m}$** drift.
- In **Trip 1 Ep 8** (39.9s outage), during a sharp vehicle turn into the university department, vertical velocity drifted prior to a brief 0.25s straight lull where $|r_{\text{vert}}| = 209.25\,\text{m/s}$. Admitting this update with linear Kalman gain produced a post-recovery velocity spike of $389.3\,\text{m/s}$ and $+4,591.5\,\text{m}$ drift.
- **Verdict**: Decoupling physical eligibility successfully eliminates the permanent latch, but **eligibility alone without innovation damping or attitude confidence checks is not sufficient to guarantee global stability**.

---

## 5. Formal Revised Scientific Verdict

> **Stage C10.8A.1 Formal Verdict**:  
> C10.8A.1 provides strong offline evidence that separating physical-motion eligibility from innovation-magnitude gating can prevent persistent VNHC lockout in several real-world blackout episodes. In particular, after a road disturbance, eligibility-only admission can recover the vertical constraint despite a large accumulated residual ($|r_{\text{vert}}| > 3.0\,\text{m/s}$). However, eligibility-only admission is not yet demonstrated to be safe under all disturbances, and substantial vertical and horizontal drift remains in several episodes. The eligibility thresholds are empirical candidates, not validated production parameters.

---

## 6. Artifacts & Ledger

- **Structured Audit JSON**: [`results/c10_8a_1_recovery_metric_audit.json`](c10_8a_1_recovery_metric_audit.json)
- **Script**: [`scratch/c10_8a_1_audit.py`](../scratch/c10_8a_1_audit.py)
- **Report**: [`results/c10_8a_1_recovery_metric_audit.md`](c10_8a_1_recovery_metric_audit.md)
- **Updated C10.8A Report**: [`results/c10_8a_physical_eligibility_recovery.md`](c10_8a_physical_eligibility_recovery.md)
- **Production Code Status**: `android/` and `src/navigation/dead_reckoning_engine.py` remain **STRICTLY FROZEN**.
