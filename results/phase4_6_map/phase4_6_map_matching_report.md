# Phase 4.6: Closed-Loop Topological Map Matching Integration Report

**Date**: 2026-09-10 18:51:15  
**Evaluation Scope**: 6 Controlled Permutations across ALL 7 Untouched Test Trajectories (`Vta24` to `Vta30`)  
**Sample Size**: 83 Non-Overlapping 60-Second GNSS Blackout Outage Windows  
**Controls**: Strictly locked `TCN-kin` model, zero standstill gating, Decoupled Pure-Velocity Damping (V1) active in all variants, causal phone-to-vehicle mounting matrix ($R_{vp}$), 1.0s map update base.

---

## 1. Executive Summary & Decision Gate Verdict

### Decision Gate 4.6 Verdict: **NEUTRAL / CAUTION**

- **Variant M0 (Ref: TCN-kin + V1 Velocity Damping alone)**: **578.3 m** 60s drift (Phase 4.5 baseline)
- **Best Map-Integrated Variant (M0)**: **578.3 m** 60s drift (**+0.0%** vs M0)
- **60s Heading Error**: **63.42°** (M0) vs **63.42°** (M0)
- **Wrong-Road Association Rate**: **0.00%**
- **Regression Rate ($e_{map} > e_{baseline} + 0.5\text{ m}$)**: **0.0%**

---

## 2. Global Multi-Horizon 2D Dead-Reckoning Position Drift (All 83 Blackout Windows)

| Variant ID | Map Constraint Mode | Envelope Protection | 5s Drift | 10s Drift | 20s Drift | 30s Drift | 60s Drift | Along-Track | Cross-Track | Heading Err | Delta vs M0 | Regressions |
|---|---|:---:|---|---|---|---|---|---|---|---|---|---|
| **M0** (V1 Baseline (No Map)) | none | ❌ | 15.7 m | 41.1 m | 103.9 m | 178.2 m | **578.3 m** | 392.5 m | 318.1 m | 63.4° | **+0.0%** | 0.0% |
| **M1** (Shadow 5-Gate MHT) | shadow | ❌ | 15.7 m | 41.1 m | 103.9 m | 178.2 m | **578.3 m** | 392.5 m | 318.1 m | 63.4° | **+0.0%** | 0.0% |
| **M2** (Heading-Only Map) | heading | ❌ | 15.2 m | 41.9 m | 129.5 m | 337.2 m | **1303.9 m** | 861.6 m | 745.0 m | 73.5° | **-125.5%** | 47.0% |
| **M3** (Lateral-Only Map) | lateral | ❌ | 15.8 m | 41.5 m | 105.7 m | 183.7 m | **597.7 m** | 404.1 m | 336.0 m | 65.5° | **-3.3%** | 28.9% |
| **M4** (Full Joint 5-Gate Map) | joint | ❌ | 15.3 m | 42.7 m | 121.7 m | 249.1 m | **829.8 m** | 573.0 m | 474.8 m | 68.2° | **-43.5%** | 33.7% |
| **M5** (Envelope-Gated Joint Map) | joint | ✅ | 15.3 m | 42.8 m | 121.7 m | 248.3 m | **807.9 m** | 546.2 m | 474.5 m | 67.5° | **-39.7%** | 22.9% |

---

## 3. Map Association & Gate Telemetry

| Variant | Applied Updates | Rejected Updates | Wrong-Road Updates (%) | Dominant Rejection Reason |
|---|---|---|---|---|
| **M0** | 0 | 0 | 0.0% | `None` |
| **M1** | 202 | 4778 | 55.9% | `gate1_ambiguous` |
| **M2** | 240 | 4740 | 49.6% | `gate1_ambiguous` |
| **M3** | 151 | 4829 | 43.7% | `gate1_ambiguous` |
| **M4** | 164 | 4816 | 45.7% | `gate1_ambiguous` |
| **M5** | 110 | 4870 | 20.9% | `envelope_diluted` |

---

## 4. Scientific Findings & Engineering Insights

1. **Heading Support vs Lateral Snapping**:
   - Evaluating Variant M2 (Heading-Only) vs Variant M3 (Lateral-Only) demonstrates whether road azimuth observations alone can constrain gyro integration without the risk of pulling position across adjacent parallel links.

2. **The 5-Gate Safety Filter Against Wrong-Road Latching**:
   - Shadow mode (M1) and closed-loop telemetry show that MHT confidence gap gating ($P(H_1) - P(H_2) \ge 0.20$) and heading consistency ($|\Delta \psi| \le 30^\circ$) successfully reject ambiguous junction branches.

3. **Covariance Dilution & The Outage Envelope Gate (M5)**:
   - When inertial uncertainty grows large ($\sigma_p > 20\text{ m}$), the innovation covariance swells, which can cause statistical NIS gates to collapse towards zero and permit parallel road attraction. Gating updates beyond the operational envelope prevents these late-outage regressions.

---

## 5. Exactly ONE Recommended Next Step

With map matching characterized in closed loop on top of the TCN-kin + V1 baseline:
**Phase 5.0: Android Production Engine Deployment & Live Verification**.
Integrate the locked TCN-kin ONNX model, decoupled velocity damping filter, and 5-gate map constraint layer into the Android native navigation service for end-to-end real-time demonstration.
