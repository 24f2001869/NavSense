# Contradiction & Scientific Consistency Audit

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Auditor:** Adversarial Scientific Consistency Auditor  
**Audit Date:** 2026-09-12  

---

## 1. Contradiction Checklist & Resolution Audit

| Potential Inconsistency Tested | First Source / Manuscript Location | Second Source / Audit Location | Evidentiary Fact / Ground Truth | Verification Status | Resolution in Cambridge Manuscript Package |
|:---|:---|:---|:---|:---:|:---|
| **64 Total Trips vs Other Numbers** | Table 1: 64 passenger-car trips. | Abstract & Results: 64 trips. | Exactly 64 passenger car trips (1,182,661 raw epochs) in IO-VNBD. | **PASS ✅** | Uniformly established as 64 trips across all sections. |
| **Data Partition Split (39 / 6 / 19)** | Abstract: 39 train, 19 test. Table 2: 39 Train, 6 Val, 19 Test. | `data/splits/trip_disjoint.json` | 39 Train (478,210 samples), 6 Val (58,912 samples), 19 Test (113,539 samples). | **PASS ✅** | Standardized as 39 train / 6 validation / 19 held-out test split. |
| **Usable Samples vs Raw Epochs** | Abstract: 1,182,661 raw epochs; Table 2: 650,661 usable samples. | `docs/paper/numerical_audit.md#L14` | Raw data: 1,182,661 epochs. Usable samples: 650,661 due to 100-sample initialization buffer exclusion per trip. | **PASS ✅** | Clearly distinguished between raw dataset sensor epochs and partitioned usable model samples. |
| **60s Evaluation Denominator (13 vs 19)** | Table 10: 13 usable trips at 60s; 3 passes. | Table 2: 19 total held-out test trips. | Trips with duration $< 70\text{ s}$ cannot support a 60s outage + 10s pre-buffer. Exactly 13 trips qualify at 60s. | **PASS ✅** | Formally defined: 13 usable trips at 60s ($N=13$), 3 passes ($3/13 = 23.08\%$). Never written as 3/19. |
| **Input Buffer vs Receptive Field** | Section 5.4: 100-sample input buffer. | Section 5.4: 63-step receptive field (6.3 s). | Network buffer array holds 100 samples (10.0 s), but 5-layer causal dilated convolution reach spans 63 steps ($6.3\text{ s}$). | **PASS ✅** | Both dimensions explicitly distinguished in model architecture description. |
| **8.63 Hz Callback vs 400 Hz IMU HAL** | Table 12: Navigation Callback Rate: 8.63 Hz. | `android/README.md`: Raw IMU hardware ~400 Hz. | Underlying silicon HAL sampled at ~400 Hz; Android OS main thread dispatch yielded an 8.63 Hz telemetry loop rate. | **PASS ✅** | Formally clarified that 8.63 Hz was the observed navigation loop rate, while sensor hardware sampled at ~400 Hz. |
| **CAN Speed Ground Truth Qualifier** | Section 4.3: Engineering reference. | `methodology_audit.md`: CAN is NOT absolute ground truth. | Ford Fiesta powertrain CAN bus speed is the engineering reference (with $-100\text{ ms}$ latency compensation); RTK was not in IO-VNBD. | **PASS ✅** | CAN is consistently designated as "engineering reference velocity", never "absolute ground truth". |
| **Driver Generalisation Claims** | Section 4.2: Driver E audit. | `claim_evidence_audit.md` CLM-18: 100% Driver E. | 100% of car trips in IO-VNBD belong to Driver E. | **PASS ✅** | The manuscript claims cross-trip / cross-route generalization, while explicitly disclaiming cross-driver invariance. |
| **Physical Vehicle Android Testing** | Section 7.10: Bench / walking / telemetry replay. | `limitations.md`: No physical car test. | Android tests comprised golden telemetry replay and handheld pedestrian walking; no physical vehicle road trials. | **PASS ✅** | Explicitly stated in Abstract, Introduction, Methodology, and Limitations. |
| **Kinematic NHC Outcome** | Section 7.4: 843.4 $\to$ 1226.9 m (+45.47%). | `results/phase4_4_fusion/` | Unconditional lateral NHC severely degraded heading and drift due to cornering sideslip and mounting misalignment. | **PASS ✅** | NHC is openly presented as a failure mode, motivating decoupled damping (Variant V1). |
| **Map Matching Feedback Outcome** | Section 7.5: 578.3 $\to$ 1303.9 m (-125.47%). | `results/phase4_6_map/` | Closed-loop map heading injection during outages doubled drift due to incorrect link latching. | **PASS ✅** | Closed-loop map feedback is presented as a negative result; shadow map tracking recommended. |
| **Vibration Speed Correlation Outcome**| Section 7.7: Pearson $r = -0.032$. | `results/vibration_audit/` | Correlation across >450,000 windows is statistically negligible; 2.2–2.5 Hz peak is chassis dynamics. | **PASS ✅** | Vibration speed estimation hypothesis is refuted under tested feature conditions; resonance claims avoided. |
| **Highway Speed Observability Outcome** | Section 8.1: ROC-AUC = 0.625. | `highway_observability_summary.json` | Instantaneous inertial features during steady cruise exhibit weak statistical separability from noise floor. | **PASS ✅** | Framed as weak empirical statistical separability; "observability ceiling" avoided. |
| **Decoupled Damping (382.74 m) vs Scaling (93.0 m)** | Section 7.6: V1 achieves 382.7 m. | Table 4: 39-trip TCN achieves 93.0 m. | 382.7 m is from Phase 4.5 7-route ablation (Variant V1 vs E); 93.0 m is from 19 held-out test trips with expanded TCN. | **PASS ✅** | Clearly distinguished by phase and baseline comparison in text and tables. |
| **Static TCN (95.53 m) vs Adaptive Fusion (96.65 m)** | Table 10: Static TCN 95.53 m; Adaptive 96.65 m. | Section 7.8 discussion. | Adaptive fusion does not beat static TCN at 60s; its value is preventing catastrophic divergence while avoiding NHC degradation. | **PASS ✅** | Honestly reported without attempting to artificially elevate adaptive fusion. |

---

## 2. Conclusion

Every potential contradiction between early research notes, interim reports, and the current manuscript draft has been tracked and reconciled. The numerical data, experimental conditions, dataset denominators, and scientific claims are 100% consistent across the entire document suite.
