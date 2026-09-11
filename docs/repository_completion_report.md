# Repository Reorganization, Documentation & GitHub Preparation — Completion Report

> **Project Title:** AI/ML-Based Intelligent Dead Reckoning for GNSS-Denied Navigation (SIH26168)  
> **Phase:** Phase 6 — Repository Architecture, Audit, Reproducibility & Documentation Pass  
> **Date:** September 11, 2026  
> **Status:** ✅ Complete — Fully Prepared for Inspection Prior to GitHub Initialization  

---

## 1. Executive Summary

This report documents the completion of **Phase 6: Complete Project Reorganization, Reproducibility, Documentation & GitHub Preparation**. 

In accordance with strict project constraints:
1. **Zero New Research / Technical Work Frozen:** No models were trained, no navigation mathematics were modified, and no Android application logic was altered.
2. **Scientific Honesty Maintained:** No metrics were altered, no negative results or failed experiments were deleted, and no premature success was claimed. The empirical pass rate of the dead reckoning system under 60-second GNSS outages remains accurately documented as **3 / 13 (23.1%)**, and at 10 seconds as **6 / 18 (33.3%)**.
3. **Strict Qualification of Pedestrian Testing:** All rooftop and walking tests recorded on campus are documented strictly as smartphone hardware, sensor sampling, and OOD stress tests—never as vehicle navigation validation.
4. **Zero Git Modification:** No `git init`, `git commit`, or remote repository creation was performed. The local workspace has been structured, sanitized, documented, and validated so that it can be inspected before initialization.

---

## 2. Structural Evolution: Before vs. After

### Original Repository Structure (Audit State)
Prior to Phase 6, the repository had 3,405 files totaling 2.65 GB. Experimental scripts, scratch exploratory tests, diagnostic figures, Android build outputs, and raw multi-gigabyte HDF5 datasets resided in unstratified directories:

```text
SIH26168-IDR/ (Before)
├── .gitignore (basic, incomplete)
├── README.md (brief overview)
├── android/ (Android app mixed with local build caches and gradle output)
├── configs/
├── data/
│   ├── raw/ (IO-VNBD HDF5 / CSV data: 2.34 GB)
│   ├── processed/
│   └── test_a_stationary_*.csv, rooftop_walk_*.csv (unstratified field recordings)
├── docs/ (minimal, missing experiment logs and architectural specs)
├── experiments/ (mixed standalone scripts, forensic scripts, and scratch files)
├── models/ (joblib, onnx, pt files mixed across root and subfolders)
├── notebooks/ (scratch exploration)
├── results/ (scattered CSV and JSON metrics)
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── fusion/
│   ├── map/
│   ├── mapping/ (redundant empty directory)
│   ├── ml/
│   ├── navigation/
│   └── preprocessing/
└── tests/
```

### Final Standardized Target Structure
The workspace has been restructured into a modular, self-documenting research repository:

```text
SIH26168-IDR/ (Target & Current State)
├── README.md                    ← Comprehensive master overview, architecture, & multi-horizon benchmarks
├── LICENSE                      ← MIT License
├── .gitignore                   ← Comprehensive filtering (excludes raw data, caches, build files)
├── requirements.txt             ← Pinned scientific dependencies
│
├── docs/                        ← Complete engineering & scientific documentation
│   ├── README.md                ← Documentation index & reading guide
│   ├── problem_statement.md     ← SIH26168 scope & challenges
│   ├── system_requirements.md   ← Functional, non-functional, & physical constraints
│   ├── architecture.md          ← 5-layer pipeline specification & dataflow
│   ├── dataset.md               ← IO-VNBD schema, sensor channels, coordinate frames
│   ├── methodology.md           ← Complete mathematical derivations (ESKF, TCN, ZVD, Kinematics)
│   ├── evaluation_protocol.md   ← Rolling outage evaluation protocol & metric formulas
│   ├── limitations.md           ← Failure catalog, physical boundaries, & unresolved problems
│   ├── current_status.md        ← High-level summary (What We Know / Think / Don't Know)
│   ├── reproducibility.md       ← Step-by-step reproduction guide from clean environment
│   ├── research_log.md          ← Chronological lab timeline of all phases
│   ├── research_story.md        ← Intellectual research narrative & lessons learned
│   ├── repository_audit.md      ← Catalog of all 3,405 initial files & triage plan
│   ├── numerical_audit.md       ← 100% verification table matching all claims to source files
│   ├── claim_evidence_matrix.md ← Epistemological strength ratings & safe wording rules
│   ├── final_repository_validation.md ← Structural integrity, link, and test report
│   ├── repository_completion_report.md ← This document
│   │
│   ├── experiments/             ← Per-phase detailed investigation records
│   │   ├── README.md            ← Master experiment index table
│   │   ├── phase0_reference_audit.md
│   │   ├── phase1_baseline.md
│   │   ├── phase2_eskf.md
│   │   ├── phase3_attitude.md
│   │   ├── phase4_ai_velocity.md
│   │   ├── phase5_1_field_forensics.md
│   │   ├── phase5_2_target_balancing.md
│   │   ├── phase5_3_vibration_audit.md
│   │   ├── phase5_5_stateful_kinematics.md
│   │   └── phase5_6_adaptive_fusion.md
│   │
│   └── decisions/               ← Architecture Decision Records (ADRs)
│       ├── accepted_approaches.md
│       ├── rejected_approaches.md
│       └── open_questions.md
│
├── src/                         ← Core operational Python source package
│   ├── data/                    ← Dataset loaders & HDF5 iterators
│   ├── preprocessing/           ← Filtering, frame transformation, sync
│   ├── navigation/              ← Strapdown integration, kinematics, ZVD
│   ├── fusion/                  ← Error-State Kalman Filter (ESKF) & adaptive blending
│   ├── ml/                      ← Temporal Convolutional Networks (PyTorch / ONNX)
│   ├── map/                     ← Map-matching & heading projection
│   └── evaluation/              ← Rolling outage evaluation engine & metric computation
│
├── scripts/                     ← Executable scripts organized by role
│   ├── README.md                ← Script catalog & execution manual
│   ├── data/                    ← Data downloading, inspection, & sync helpers
│   ├── experiments/             ← Standalone, self-contained experiment runners
│   │   ├── run_adaptive_fusion_benchmark.py
│   │   ├── run_stateful_kinematic_benchmark.py
│   │   ├── run_highway_speed_observability.py
│   │   ├── run_vibration_speed_audit.py
│   │   └── run_expanded_tcn_forensics.py
│   ├── evaluation/              ← Evaluation & verification scripts
│   └── utilities/               ← Path sanitization & data checking utilities
│       └── sanitize_paths.py
│
├── notebooks/                   ← Interactive Jupyter exploration notebooks
│   ├── README.md                ← Notebook directory guide
│   ├── exploration/             ← Raw signal inspection & sensor characterization
│   ├── visualization/           ← Diagnostic plotting
│   ├── navigation/              ← Strapdown & ESKF prototyping
│   ├── machine_learning/        ← TCN training & loss curves
│   └── evaluation/              ← Outage drift analysis & CDF plots
│
├── experiments/                 ← Original research scripts & configs
│   ├── README.md                ← Research script guide
│   ├── phase0/                  ← CAN vs wheel speed reference checks
│   ├── phase1/                  ← Strapdown baseline evaluation
│   ├── phase2/                  ← ESKF tuning & blackout simulation
│   ├── phase3/                  ← NHC & attitude stabilization experiments
│   ├── phase4/                  ← Ridge, RF, & initial TCN speed estimators
│   └── phase5/                  ← Forensic runners, ablation scripts, & balancing
│
├── results/                     ← Authoritative numerical artifacts & tables
│   ├── README.md                ← Results guide & folder hierarchy
│   ├── RESULTS_INDEX.md         ← Master tabular summary of all evaluated benchmarks
│   ├── baseline/                ← Phase 1 pure IMU results
│   ├── eskf/                    ← Phase 2 GNSS-aided ESKF metrics
│   ├── ai_velocity/             ← Phase 4 ML velocity models
│   ├── field/                   ← Real-world smartphone sensor logs & replays
│   ├── vibration_audit/         ← Phase 5.3 FFT & power spectral density metrics
│   ├── stateful_kinematics/     ← Phase 5.5 kinematic integration logs
│   ├── adaptive_fusion/         ← Phase 5.6 benchmark results (CSV & JSON)
│   └── figures/                 ← Master visual charts generated during evaluation
│
├── models/                      ← Model weights, ONNX exports, and metadata
│   ├── README.md                ← Model catalog, training protocols, & sizes
│   ├── metadata/                ← Feature scalers (joblib) & architecture JSONs
│   │   ├── tcn_scaler_expanded.joblib
│   │   └── tcn_expanded_config.json
│   ├── tcn_velocity_best.pt     ← PyTorch checkpoint
│   └── tcn_velocity_expanded.onnx ← Android production ONNX artifact (162 KB)
│
├── configs/                     ← Configuration files for models and filters
│   ├── eskf_default.yaml        ← Standard sensor noise & covariance settings
│   └── tcn_features.yaml        ← Sensor window & feature definitions
│
├── data/                        ← Dataset instructions & field telemetry
│   ├── README.md                ← Dataset retrieval guide (IO-VNBD split definition)
│   ├── raw/                     ← (Local only, ignored by git)
│   ├── processed/               ← Preprocessed cache directory
│   ├── samples/                 ← Lightweight mock / sample format files
│   └── field/                   ← Real-world smartphone telemetry logs
│       ├── test_a_stationary_20260911_014727.csv
│       ├── test_b_walking_rooftop_20260911_015632.csv
│       ├── test_c_walking_ground_20260911_020614.csv
│       └── walk_hostel_mess_203838.csv
│
├── android/                     ← Complete native Android application
│   ├── README.md                ← Android architecture, deployment, & validation guide
│   ├── app/                     ← Kotlin / Java application code
│   └── ...                      ← Gradle configuration files
│
├── tests/                       ← Automated verification & regression test suite
│   ├── test_causality_and_leakage.py ← Verification of zero future leakage & disjoint splits
│   └── ...                      ← Unit tests for filters & kinematics
│
├── assets/                      ← Curated presentation figures & diagrams
│   ├── README.md                ← Asset index & diagram guide
│   ├── architecture/            ← Mermaid diagrams & system schematics
│   ├── experiments/             ← Curated diagnostic figures from key phases
│   └── screenshots/             ← Android application screenshots
│
└── archive/                     ← Non-operational, superseded, or exploratory code
    ├── README.md                ← Archive catalog & rationale
    ├── obsolete/                ← Dead ends (e.g., unconditional NHC, raw Ridge)
    ├── superseded/              ← Intermediate model iterations replaced by TCN-Expanded
    └── exploratory/             ← Scratch scripts & one-off diagnostic notebooks
```

---

## 3. Inventory & File Action Summary

| Action Category | Quantity | Description & Rationale |
| :--- | :---: | :--- |
| **Files Cataloged (Audit)** | 3,405 | Total files present in workspace across all directories. |
| **Files Local-Only (Excluded)** | ~2,580 | Ignored by `.gitignore`: raw IO-VNBD dataset (2.34 GB), intermediate `.npz` feature caches, `.venv`, Android `.gradle` build artifacts, and Antigravity internal files. |
| **New Documentation Files Created** | 25 | Comprehensive documentation across `docs/`, `docs/experiments/`, `docs/decisions/`, and section READMEs. |
| **Scripts Standardized & Curated** | 6 | Clean, self-contained reproduction scripts created in `scripts/experiments/` and `scripts/utilities/`. |
| **Figures Curated into Assets** | 12 | Authoritative diagnostic plots copied from workspace artifacts to `assets/experiments/`. |
| **Telemetry Consolidated** | 4 | Real-world smartphone CSV files moved from root `data/` into structured `data/field/`. |
| **Redundant Directories Removed** | 1 | Removed empty `src/mapping` directory (consolidated into existing `src/map`). |
| **Files Sanitized of Absolute Paths** | 79 | Converted hardcoded `C:\Users\<user>\...` paths into relative paths via automated utility. |

---

## 4. Complete Documentation Catalog

The following authoritative documents have been authored and placed under `docs/`:

1. [`docs/README.md`](../README.md) — Documentation index, reading paths for different audiences (judges, engineers, researchers).
2. [`docs/problem_statement.md`](problem_statement.md) — Formal statement of SIH Problem Statement 26168.
3. [`docs/system_requirements.md`](system_requirements.md) — Functional and non-functional engineering requirements.
4. [`docs/architecture.md`](architecture.md) — Complete 5-layer pipeline specification (Sensors, Preprocessing, ML, ESKF, UI).
5. [`docs/dataset.md`](dataset.md) — IO-VNBD schema, sensor coordinate systems, sampling frequencies, and synchronization.
6. [`docs/methodology.md`](methodology.md) — Mathematical derivations of strapdown integration, ESKF, TCN receptive fields, and adaptive regime fusion.
7. [`docs/evaluation_protocol.md`](evaluation_protocol.md) — Rolling blackout extraction protocol, metric definitions, and SIH pass criterion.
8. [`docs/limitations.md`](limitations.md) — Comprehensive, honest catalog of failure modes, unobservability regimes, and hardware limitations.
9. [`docs/current_status.md`](current_status.md) — Structured summary categorized into 🟢 What We Know, 🟡 What We Think, and 🔴 What We Don't Know.
10. [`docs/reproducibility.md`](reproducibility.md) — Step-by-step reproduction instructions for all benchmark scripts from a fresh environment.
11. [`docs/research_log.md`](research_log.md) — Chronological laboratory log tracking hypotheses, experiments, results, and learnings.
12. [`docs/research_story.md`](research_story.md) — Intellectual narrative outlining the scientific progression from naive integration to adaptive fusion.
13. [`docs/repository_audit.md`](repository_audit.md) — Initial comprehensive repository audit and triage categorization.
14. [`docs/numerical_audit.md`](numerical_audit.md) — Numerical integrity audit verifying every published metric against primary artifact files.
15. [`docs/claim_evidence_matrix.md`](claim_evidence_matrix.md) — Epistemological matrix providing evidence strength and safe wording rules.
16. [`docs/final_repository_validation.md`](final_repository_validation.md) — Validation report verifying link integrity, path sanitization, and automated unit test results.
17. [`docs/repository_completion_report.md`](repository_completion_report.md) — This completion document.

### Phase Experiment Reports (`docs/experiments/`)
- [`phase0_reference_audit.md`](experiments/phase0_reference_audit.md) — Reference sensor audit (CAN vs. wheel speed vs. phone GPS).
- [`phase1_baseline.md`](experiments/phase1_baseline.md) — Pure IMU strapdown double integration baseline (cubic error divergence).
- [`phase2_eskf.md`](experiments/phase2_eskf.md) — Error-State Kalman Filter formulation and blackout behavior.
- [`phase3_attitude.md`](experiments/phase3_attitude.md) — Attitude stabilization and Non-Holonomic Constraints (NHC) failure analysis.
- [`phase4_ai_velocity.md`](experiments/phase4_ai_velocity.md) — Development of TCN causal speed estimation and expanded cross-trip training.
- [`phase5_1_field_forensics.md`](experiments/phase5_1_field_forensics.md) — Investigation of the ~71.66 m/s pedestrian OOD failure & Android loop timing bug.
- [`phase5_2_target_balancing.md`](experiments/phase5_2_target_balancing.md) — Target balancing experiments and steady-state cruise collapse.
- [`phase5_3_vibration_audit.md`](experiments/phase5_3_vibration_audit.md) — Spectral audit of 64 trips (>450,000 windows) rejecting vibration as a speed proxy ($r = -0.032$).
- [`phase5_5_stateful_kinematics.md`](experiments/phase5_5_stateful_kinematics.md) — Evaluation of stateful acceleration integration vs. static ML predictions.
- [`phase5_6_adaptive_fusion.md`](experiments/phase5_6_adaptive_fusion.md) — Causal adaptive regime fusion benchmark across 19 held-out test trips.

### Architecture Decision Records (`docs/decisions/`)
- [`accepted_approaches.md`](decisions/accepted_approaches.md) — Documentation of validated architectural elements (TCN-Expanded, Causal ZVD, Adaptive Regime Weighting).
- [`rejected_approaches.md`](decisions/rejected_approaches.md) — Forensic analysis of rejected methods (naive pure IMU, unconditional NHC, raw vibration speedometer, walking-as-vehicle validation).
- [`open_questions.md`](decisions/open_questions.md) — Critical open problems (long-duration cruise observability, vehicle suspension transfer functions, dynamic yaw alignment).

---

## 5. Summary of Experiments & Results Preserved

The primary results across all experimental phases have been preserved in `results/`, verified against raw data in `docs/numerical_audit.md`, and indexed in [`results/RESULTS_INDEX.md`](../results/RESULTS_INDEX.md):

| Phase | Description | Key Metric / Observation | Source Artifact |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Pure IMU Strapdown | Catastrophic divergence: error grows with $t^3$; 100% failure within seconds | `results/baseline/` |
| **Phase 2** | GNSS-Aided ESKF | Maintains sub-meter accuracy during GNSS; drifts linearly upon blackout | `results/eskf/` |
| **Phase 3** | NHC Constraint Feedback | Lateral drift constrained, but gyro frame misalignment causes artificial turning | `docs/experiments/phase3_attitude.md` |
| **Phase 4** | Expanded TCN (64 trips) | Mean Speed MAE: 2.14 m/s; 60s blackout drift: 24.32% (140.2 m) | `models/metadata/tcn_expanded_config.json` |
| **Phase 5.1** | Pedestrian OOD Failure | TCN predicted **71.66 m/s** during walking due to severe angular velocity OOD ($3.81 > 0.40$ rad/s) | `results/field/` |
| **Phase 5.1.1**| Android Loop Bug | Loop dt was hardcoded to 100ms instead of actual 15-22ms; fixed & verified | `android/README.md` |
| **Phase 5.3** | Vibration Speed Audit | Evaluated 64 trips (>450,000 windows); correlation $r = -0.032$; rejected as speedometer | `results/vibration_audit/` |
| **Phase 5.4** | Highway Observability | Steady cruise classification ROC-AUC = 0.625; unobservable from IMU alone | `results/figures/` |
| **Phase 5.5** | Stateful Kinematics | Excellent on highway, but runaway drift on urban stop-and-go (`Vta26` drift: 574.65%) | `results/stateful_kinematics/` |
| **Phase 5.6** | Adaptive Regime Fusion | **60s aggregate drift: 16.76% (96.65 m)**; `Vta26` drift reduced to **47.31%**; 60s pass rate: **3/13 (23.1%)** | `results/adaptive_fusion/` |

---

## 6. Reproducibility Status

The repository is structured for clean, deterministic reproduction:
- **Environment:** Tested on Python 3.10+ / 3.12 with dependencies recorded in [`requirements.txt`](../requirements.txt).
- **Causality & Leakage Verification:** Automated test [`tests/test_causality_and_leakage.py`](../tests/test_causality_and_leakage.py) passes with 100% success, confirming:
  1. Zero future lookahead across all feature windows (strictly trailing edge: $[t - W, t]$).
  2. Complete trip-level separation between training, validation, and held-out test sets.
- **Reproduction Scripts:** Self-contained runners in `scripts/experiments/` allow independent re-execution of the vibration audit, highway observability analysis, stateful kinematics, and adaptive fusion benchmarks.
- **Dataset Instructions:** Detailed manual provided in [`data/README.md`](../data/README.md) allowing any researcher to download IO-VNBD and reproduce the splits.

---

## 7. Files That Must NOT Be Committed to GitHub

Before committing, ensure that the rules in [`.gitignore`](../.gitignore) prevent the following from being staged:

1. **Large Raw Datasets:**
   - `data/raw/**` (2.34 GB of HDF5 and raw CSV files from IO-VNBD).
2. **Intermediate Feature Caches:**
   - `data/processed/*.npz` and large extracted feature matrices.
3. **Local Virtual Environments & Caches:**
   - `.venv/`, `venv/`, `__pycache__/`, `*.pyc`.
4. **Android Build Artifacts:**
   - `android/.gradle/`, `android/app/build/`, `*.apk`.
5. **Machine-Specific & Antigravity Internal Files:**
   - Antigravity brain / execution logs, IDE workspace storage, OS `.DS_Store` or `Thumbs.db`.

---

## 8. Recommended Git Initialization Commands (For User Inspection)

As specified in Phase 6 instructions, **no Git commands have been executed by the agent**. The repository is fully prepared locally for your inspection.

When you are satisfied with the audit and file organization, run the following commands in PowerShell from the repository root (`.`):

```powershell
# 1. Initialize git repository
git init

# 2. Verify git status and ensure large data files / caches are ignored
git status

# 3. Check that data/raw is NOT staged
git status --ignored

# 4. Stage the clean repository files
git add README.md LICENSE .gitignore requirements.txt
git add docs/
git add src/
git add scripts/
git add notebooks/
git add experiments/
git add results/
git add models/
git add configs/
git add data/README.md data/samples/ data/field/
git add android/
git add tests/
git add assets/
git add archive/

# 5. Review the staged files list
git status

# 6. Make the initial clean commit
git commit -m "feat: complete research repository structure, documentation, and reproducibility benchmark for SIH26168"
```

---

## 9. Conclusion

Phase 6 is complete. The `SIH26168-IDR` repository has been transformed from an unorganized workspace into a rigorous, professional, and transparent scientific research repository. It honestly conveys the empirical reality of smartphone dead reckoning, honors the research journey, preserves all negative findings, and stands ready for review.
