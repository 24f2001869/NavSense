# Reproducibility Gap & Action Report: NavSense / SIH26168

**Document Purpose:** Systematic audit of potential reproducibility barriers, technical dependencies, documentation gaps, and execution risks for independent verification.  
**Target Journal:** The Journal of Navigation (Cambridge University Press)  
**Date:** September 2026  
**Auditor:** Reproducibility & Publication Quality Engineering  

---

## 1. Executive Gap Classification

Each identified aspect of the research pipeline is classified into one of four severity categories:
* **BLOCKER:** Missing component that prevents verification of claims or invalidates experimental repeatability. *(Must be resolved before submission).*
* **MAJOR:** Substantial dependency or structural gap requiring explicit documentation, supplementary instructions, or qualified manuscript scoping.
* **MINOR:** Minor numerical, timing, or platform variation that does not alter scientific conclusions.
* **OPTIONAL:** Quality-of-life enhancements for downstream practitioners (e.g. containerisation).

---

## 2. Itemized Gap Ledger

| ID | Domain | Description & Analysis | Severity | Resolution / Handling in Submission Package |
|:---:|:---|:---|:---:|:---|
| **GAP-01** | **Dataset Hosting** | Raw IO-VNBD dataset (2.34 GB, 64 trips) is not hosted directly inside the Git repository due to GitHub file-size constraints and upstream CC BY 4.0 licensing attribution. | **MAJOR** | Fully documented in `data/README.md` and `docs/reproducibility.md`. Manuscript explicitly cites the primary *Data in Brief* paper (Onyekpe et al., 2021) and provides DOI (`10.1016/j.dib.2021.106885`). |
| **GAP-02** | **Physical Vehicle Road Testing** | On-device Android evaluations are based on golden telemetry replay, computational latency profiling, and handheld pedestrian walking stress tests; no physical road-vehicle tests were conducted. | **MAJOR** | Preserved as an immutable, prominent limitation in the abstract, methodology, results, discussion, and limitations sections. Terminology strictly enforced: *"continuous telemetry replay"* rather than *"physical driving validation"*. |
| **GAP-03** | **Hardware Clock Jitter** | Android IMU HAL operates at ~400 Hz, but Android OS IPC callback dispatch produces an observed 8.63 Hz mean callback rate in the navigation loop, contrasting with IO-VNBD's synchronized 10.0 Hz clock. | **MINOR** | Documented in Section 2.8 and Section 4.9. The Android implementation utilizes dynamic monotonic hardware nanosecond timestamps ($\Delta t \in [15, 22]\text{ ms}$) to ensure correct numerical integration despite OS scheduling jitter. |
| **GAP-04** | **Cross-Platform Numerical Parity** | Replay of golden vehicle telemetry through Java/EJML ESKF vs. Python 64-bit reference engine exhibits minor numerical drift over extended trajectories (maximum position difference: 0.24177 m; velocity: 0.02117 m/s; heading: 0.01829°). | **MINOR** | Documented in Section 4.9 and Supplementary Material Table S4. Parity is confirmed within acceptable engineering tolerances for single-precision mobile matrix libraries. |
| **GAP-05** | **Deterministic Seeds in PyTorch** | Deep learning models rely on deterministic CUDA/CPU seeds for exact weight reproduction during re-training from scratch. | **MINOR** | Pre-trained model artifacts (`tcn_velocity_expanded.onnx`, 162 KB, 65,409 parameters) are bundled directly in `models/`, allowing instantaneous, zero-retraining verification of all inference results. |
| **GAP-06** | **Containerization** | The repository currently relies on standard Python `venv` and `requirements.txt` rather than a pre-built Docker image. | **OPTIONAL** | Virtual environment setup is fully detailed in `docs/reproducibility.md`. Python dependencies are strictly pinned to standard PyPI releases without proprietary system libraries. |
| **GAP-07** | **Map Ingestion Pipeline** | Closed-loop map feedback experiment relied on a specific local segment-projection implementation; large-scale OSM vector tiling requires network tile caching on Android. | **OPTIONAL** | Manuscript explicitly concludes that closed-loop map heading feedback is *counterproductive* under heading drift (-125.5% degradation) and recommends retaining map data purely for downstream visual routing. |

---

## 3. Summary Assessment

* **Total BLOCKERS:** **0**
* **Total MAJOR Gaps:** **2** (Both fully addressed through transparent limitation disclosures and provenance citations).
* **Total MINOR Gaps:** **3** (All documented with exact numerical tolerances and hardware explanations).
* **Total OPTIONAL Gaps:** **2** (Documented in reproducibility documentation).

**Conclusion:** The codebase and empirical evidence meet academic standards for a submission to *The Journal of Navigation*. No unresolvable reproducibility barriers exist.
