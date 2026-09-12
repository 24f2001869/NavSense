# Scientific Publication Readiness Assessment & Quality Gate Audit

**Project:** SIH26168 / NavSense  
**Manuscript Title:** *Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift*  
**Auditor:** Antigravity Scientific Audit & Publication Gate Engine  
**Evaluation Date:** 2026-09-12 (Post-Forensic Correction Pass)  
**Repository URL:** [https://github.com/24f2001869/SIH26168-IDR](https://github.com/24f2001869/SIH26168-IDR)

---

## 1. Executive Publication Verdict

### **VERDICT: CONDITIONAL (VENUE-SPECIFIC READINESS)**

| Target Venue Type | Status | Assessment & Readiness Rationale |
|:---|:---:|:---|
| **Top-Tier Robotic / ITS Conferences**<br>*(e.g., IEEE IV, IEEE ITSC, IEEE IROS, ION GNSS+)* | **STRONG GO (SUBMISSION READY)** | Following the forensic correction pass, all internal contradictions, dataset nomenclature errors, proof-language overclaims, and denominator discrepancies have been resolved. The paper presents a rigorous empirical study, controlled 8-way ablation, large-scale spectral evaluation, decoupled damping contribution (+54.6%), and transparent failure forensics across 64 trips. |
| **Specialized Navigation & Sensor Journals**<br>*(e.g., MDPI Sensors, Elsevier Measurement, Springer Satellite Navigation)* | **STRONG GO (SUBMISSION READY)** | Fully supported. Complete methodology, mathematical formulations, reproducible code, and exhaustive per-trip error metrics satisfy specialized empirical journal standards. |
| **Flagship Automotive Journals**<br>*(e.g., IEEE Transactions on Intelligent Vehicles, IEEE T-ITS)* | **CONDITIONAL HOLD (RECOMMENDED ACTION)** | While the theoretical and empirical methodology on IO-VNBD is sound, reviewers in flagship automotive journals frequently demand **physical in-vehicle road validation using the native Android application** (currently evaluated on phone replay + pedestrian/bench stress tests). A 30-to-60 minute instrumented road drive with the phone mounted in a car will elevate this to an unconditional journal submission. |

---

## 2. Formal 20-Point Quality Gate Audit

Every check is evaluated against strict evidentiary standards:

```
+========================================================================================================================+
| Item | Verification Criteria                                                  | Status  | Concrete Evidentiary Source   |
+========================================================================================================================+
| 01   | All major numerical claims trace to empirical evidence                 | PASS ✅ | docs/paper/numerical_audit.md |
| 02   | All denominators (N trips, windows, epochs) are explicitly specified   | PASS ✅ | N=13 (60s), N=19, N=64 census |
| 03   | Train/validation/test split is documented and strictly disjoint        | PASS ✅ | 39 Train / 6 Val / 19 Test    |
| 04   | Driver split is resolved and single-driver limitation is acknowledged   | PASS ✅ | 100% Driver E explicit note   |
| 05   | Zero future leakage verified via causality test suite                  | PASS ✅ | test_causality_and_leakage.py |
| 06   | Literature claims verified from verified publications                  | PASS ✅ | 20 verified DOI references    |
| 07   | Algorithmic novelty claims are defensible and avoid exaggeration       | PASS ✅ | Framed as empirical discovery |
| 08   | SIH <10% drift benchmark is honestly reported as NOT universally met   | PASS ✅ | 3/13 passes (23.1%), 16.76%   |
| 09   | Author field testing is explicitly identified as pedestrian / bench    | PASS ✅ | Vehicle field test disclaimed |
| 10   | Android edge results correctly report real-time execution & parity     | PASS ✅ | 4.2-7.8 ms latency, <5cm delta|
| 11   | Severe OOD pedestrian speed spike is fully documented and analyzed     | PASS ✅ | 71.66 m/s spike (+26.08σ yaw) |
| 12   | Negative results & degraded variants (NHC, Map) are openly reported     | PASS ✅ | NHC -45.5%, Map -125.5%        |
| 13   | Limitations are stated objectively without euphemisms                   | PASS ✅ | Section 10 explicitly bounded  |
| 14   | Tables use consistent metric definitions and clear weighting schemes   | PASS ✅ | docs/paper/table_plan.md       |
| 15   | All 13 figures are mapped to actual repository assets and data        | PASS ✅ | docs/paper/figure_plan.md      |
| 16   | References are verified with real DOIs and peer-reviewed venues        | PASS ✅ | Zero fabricated citations      |
| 17   | Zero invented values or hypothetical extrapolations                    | PASS ✅ | Forensic trace on all 30 values|
| 18   | Observability limits described as empirical, not mathematical theorems  | PASS ✅ | ROC-AUC 0.625 empirical finding|
| 19   | Zero pseudo-statistical significance claims without formal test backing | PASS ✅ | Absolute deltas reported       |
| 20   | Zero marketing, promotional, or hackathon hyperbole in text            | PASS ✅ | Formal academic tone throughout|
+========================================================================================================================+
```

---

## 3. Forensic Corrections Applied in this Pass

1. **Dataset Nomenclature Corrected**: Sourced from primary paper (Onyekpe, Palade, Kanarachos, & Szkolnik, *Data in Brief*, 2021, DOI: `10.1016/j.dib.2021.106885`). The expansion is uniformly established as **Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning (IO-VNBD)** across abstract, dataset, methodology, and references.
2. **Repository URL Unified**: Updated from placeholder `NavSense` to the actual public repository: `https://github.com/24f2001869/SIH26168-IDR`.
3. **Table 4 Ambiguity Eliminated**: Removed the preliminary 6-trip model from Table 4 and replaced it with the authoritative **Expanded Multi-Trip TCN Generalization Benchmark (Controlled Scaling)**, presenting the 6-trip vs 39-trip controlled scaling comparison.
4. **Dataset Sample Counts Disambiguated**: Explicitly separated the **1,182,661 raw synchronized epochs** from the **650,661 usable model training/validation/test samples** (478,210 train / 58,912 val / 113,539 test), explaining the pre-buffer exclusion.
5. **TCN-kin Feature Set Corrected**: Aligned Section 5.4 with the actual locked 9-channel feature set: $a_{\text{lin}, x}, a_{\text{lin}, y}, a_{\text{lin}, z}, \omega_z, \omega_y, \omega_x, a_h, a_v, \kappa$.
6. **Receptive Field Distinctions Clarified**: Explicitly noted that while the input buffer maintains a 100-sample (10.0 s) history array, the 5-layer causal dilated convolutional reach spans 6.3 seconds (63 steps).
7. **Overclaims Converted to Evidence Language**:
   - Replaced "disproved vibration speedometer" with "finds no robust generalizable vibration-based speed signal under the tested conditions."
   - Replaced "proven suspension resonance" with "dominant 2.2–2.5 Hz peak consistent with low-frequency vehicle chassis dynamics."
   - Replaced "fundamental physical observability boundary" with "weak statistical separability of instantaneous inertial features during steady highway cruising (ROC-AUC = 0.625)."
   - Replaced "measured tire sideslip caused heading corruption" with "consistent with tire sideslip and mounting misalignment corrupting heading."
8. **Adaptive Fusion Story Reframed**: Accurately reported that Adaptive Fusion achieves comparable long-horizon performance to Static TCN at 60s (96.65 m vs 95.53 m) and does not beat Static TCN at every horizon, highlighting that its true value is preventing catastrophic divergence while avoiding unconditional constraint degradation.
9. **Macro-Average Calculation Noted**: Added explicit footnotes to Table 3 and Table 10 explaining that absolute drift and normalized relative drift are independently macro-averaged across trips, preventing confusion regarding direct division of displayed means.
10. **Android Thermal Claims Moderated**: Replaced "thermal safe" claims with "operates within measured execution-time budget (<10 ms per 100 ms epoch, >90% idle headroom)."
11. **Bibliographic References Cleaned**: Corrected AVNet DOI (`10.1186/s43020-025-00162-4`), IO-VNBD authors (Onyekpe et al.), and replaced placeholder author names with authentic metadata.
12. **IO-VNBD Reference Hierarchy Clarified**: Sourced reference instrumentation directly from Onyekpe et al. (2021) as Racelogic VBOX Video HD2 (10 Hz) and Ford Fiesta CAN bus diagnostics (-100 ms latency compensated), strictly disclaiming RTK/carrier-phase Doppler availability in IO-VNBD.
13. **Reviewer Defense Framing Refined**: Softened causal phrasing to "experimentally isolated failure mechanisms, with physically motivated interpretations where the underlying disturbance was not directly instrumented."
14. **Android Replay and Loop Rate Terminology Grounded**: Designated Android execution as "178.9-s continuous telemetry replay" (not physical driving), and clarified 8.63 Hz as the observed navigation telemetry loop callback rate (raw IMU hardware sampled at ~400 Hz).
15. **Highway Observability Section Title & Tone Locked**: Heading formalized as "Weak Speed Separability During Steady-State Highway Cruise", reflecting empirical classification performance (ROC-AUC = 0.625 on V-Vfa02) rather than formal nonlinear state observability theorems.
16. **Empirical Novelty Positioning Calibrated**: Framed contributions as rigorous empirical evaluations on the 64-trip IO-VNBD census and held-out routes under tested sampling/feature conditions, eliminating overbroad novelty claims.
17. **Project / Repository Nomenclature Unified**: Explicitly recognized `NavSense (SIH26168)` as the project/system identity and `https://github.com/24f2001869/SIH26168-IDR` as the codebase repository URL.

---

## 4. Final Submission Boundaries

* **Conference Readiness:** **IMMEDIATE GO.** The manuscript is ready for conversion into IEEE conference format (IEEEtran LaTeX template).
* **Journal Readiness:** **CONDITIONAL.** If targeting IEEE T-IV or IEEE T-ITS, an in-car road drive with the Android app is strongly recommended before final journal review.
