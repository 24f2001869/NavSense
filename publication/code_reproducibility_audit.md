# Code Reproducibility Audit: NavSense / SIH26168

**Document Purpose:** Verification of repository readiness, structure, dependencies, script alignment, dataset instructions, and security hygiene for journal submission.  
**Target Journal:** The Journal of Navigation (Cambridge University Press)  
**Date:** September 2026  
**Auditor:** Scientific Manuscript & Reproducibility Auditor  

---

## 1. Repository Identification & Metadata

* **Repository Names:** `NavSense` / `SIH26168-IDR`
* **Canonical Repository URL:** `https://github.com/24f2001869/NavSense`
* **Alternate / SIH Tracking URL:** `https://github.com/24f2001869/SIH26168-IDR`
* **License:** MIT License (`LICENSE` verified present at repository root).
* **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0) (Onyekpe et al., 2021).

---

## 2. Directory Structure Audit

| Directory | Purpose | Reproducibility Status | Audit Finding |
|:---|:---|:---:|:---|
| `src/` | Python core library (`data`, `filters`, `models`, `fusion`, `evaluation`) | Verified | Clean modular package, standard imports. |
| `scripts/` | Execution scripts for training, evaluation, diagnostics | Verified | Standalone entry-points corresponding to paper sections. |
| `models/` | Trained PyTorch and ONNX model artifacts | Verified | `tcn_velocity_expanded.onnx` (162 KB) present and executable. |
| `configs/` | YAML configuration files for models and filters | Verified | Hyperparameters explicitly tracked in code and config. |
| `tests/` | Unit, causality, and mathematical leakage test suite | Verified | `test_causality_and_leakage.py` passes 100% clean. |
| `android/` | Android mobile edge prototype (Kotlin/Java) | Verified | Gradle build tree, EJML ESKF, ONNX Runtime Android integration. |
| `data/` | Dataset loading specifications and directory stubs | Verified | Raw data excluded from Git (per license/size); setup guide present. |
| `docs/` | Engineering documentation, lab notebooks, audit files | Verified | Extensive forensic reports and audit trails available. |
| `results/` | CSV, JSON, and artifact archives of all evaluated runs | Verified | Raw evaluation records matching paper numbers line-by-line. |
| `publication/` | Submission package, journal requirements, figure sources | Verified | JoN-tailored submission package. |

---

## 3. Dependency & Environment Specification

* **Environment Specification:** `requirements.txt` is present at repository root.
* **Python Compatibility:** Tested on Python 3.10 and 3.11.
* **Core Libraries Pinned:**
  - `numpy>=1.24.0,<=2.3.0`
  - `scipy>=1.11.0,<=1.17.0`
  - `pandas>=2.0.0,<=2.4.0`
  - `torch>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `onnx>=1.14.0`
  - `onnxruntime>=1.16.0`
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`
  - `pyyaml>=6.0`
* **Assessment:** Sufficiently bounded to prevent dependency conflicts while allowing standard execution on clean environments.

---

## 4. Script Alignment with Manuscript Claims

Every major experimental section in the restructured manuscript maps directly to a standalone reproduction script:

| Manuscript Section | Analysis / Experiment | Reproduction Script | Result Artifact(s) |
|:---|:---|:---|:---|
| **Section 2.2 / 3.1** | Causality & Temporal Leakage Suite | `tests/test_causality_and_leakage.py` | Zero future-gradient leakage logs |
| **Section 4.1** | Baseline Inertial Divergence | `scripts/experiments/run_stateful_kinematic_benchmark.py` | `results/stateful_kinematics/` |
| **Section 4.2** | 6-trip vs 39-trip TCN Scaling | `scripts/experiments/run_expanded_tcn_forensics.py` | `results/expanded_tcn_benchmark/` |
| **Section 4.3** | Adaptive Velocity Fusion Multi-Horizon | `scripts/experiments/run_adaptive_fusion_benchmark.py` | `results/adaptive_fusion/` |
| **Section 4.4** | Constraint Failure & Decoupled Damping | `scripts/experiments/run_stateful_kinematic_benchmark.py` | `results/stateful_kinematics/` |
| **Section 4.5** | Map-Matching Closed-Loop Heading Drift | `scripts/experiments/run_map_matching_evaluation.py` | `results/map_matching/` |
| **Section 4.6** | Highway Cruise Speed Separability | `scripts/experiments/run_highway_speed_observability.py` | `results/highway_observability/` |
| **Section 4.7** | Spectral Vibration Speed Correlation | `scripts/experiments/run_vibration_speed_audit.py` | `results/vibration_audit/` |
| **Section 4.8** | Pedestrian Out-Of-Distribution Analysis | `scripts/experiments/run_field_forensic_replay.py` | `results/field/` |
| **Section 4.9** | Android Telemetry Replay & Benchmarks | `android/app/src/androidTest/` | `results/android_benchmarks/` |

---

## 5. Dataset Acquisition Instructions

* **Dataset:** IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)
* **Authors:** Onyekpe, Palade, Kanarachos, Szkolnik (2021)
* **DOI:** `10.1016/j.dib.2021.106885`
* **License:** CC BY 4.0
* **Storage Reality:** 2.34 GB raw data across 64 synchronized trips. Not redistributed in Git repository to comply with standard open-source repository size limits and upstream attribution.
* **Instructions in Repo:** `data/README.md` and `docs/reproducibility.md` provide clear, step-by-step acquisition and directory layout instructions.

---

## 6. Security, Privacy & Integrity Audit

* **API Keys / Credentials:** Zero active API keys, private tokens, or hardcoded passwords discovered.
* **Personal Data:** No personal identifiable information (PII) beyond the author's public academic email.
* **Binary Artifacts:** Only required pre-trained model weights (`.onnx`, `.pth`) are tracked. No unneeded temporary compiler caches.
* **Git Status:** Git history is clean; standard `.gitignore` prevents inadvertent tracking of `.venv`, `__pycache__`, build outputs, and raw `.csv` dumps.

---

## 7. Recommended Wording for Manuscript Declarations

### Code Availability Statement
> "The software code supporting the analyses, models, filtering pipelines, and Android deployment in this study is available under an open-source MIT licence at https://github.com/24f2001869/NavSense (mirror: https://github.com/24f2001869/SIH26168-IDR)."

### Data Availability Statement
> "The empirical evaluations in this study were conducted on the publicly available IO-VNBD benchmark dataset (Onyekpe et al., 2021), accessible under the Creative Commons Attribution 4.0 International licence at https://doi.org/10.1016/j.dib.2021.106885. Derived summary metrics and evaluation logs generated during this research are available within the repository."
