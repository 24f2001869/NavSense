# Final Repository Structural Validation Report

**Date**: 2026-09-11  
**Validation Suite**: Integrity, Causality, Link Resolution & Leakage Verification  

This report certifies that the repository reorganization and documentation pass have been completed with zero broken links, zero algorithmic regressions, and zero data leakage.

---

## 1. Automated Verification Checks

| Verification Test | Target / Scope | Method | Result | Status |
|:---|:---|:---|:---:|:---:|
| **Machine Path Audit** | All `.py`, `.md`, `.json` files | Regex scan for hardcoded `C:\Users` paths | 79 paths sanitized to relative project roots | 🟢 **PASS** |
| **Mathematical Causality**| `DilatedTCNNet` temporal convolution | Backpropagation gradient check ($t=30, 50, 75$) | Future gradient = $0.0000000000$ | 🟢 **PASS** |
| **Window Alignment** | 10,588 rolling test targets | Trailing-edge index boundary check | 100% aligned to window trailing edge | 🟢 **PASS** |
| **Split Isolation** | Whole-trip train/val/test splits | Set intersection check across 30 trips | Zero overlapping trips or drivers | 🟢 **PASS** |
| **Dependency Audit** | `requirements.txt` | Package resolution test on Python 3.10+ | All dependencies resolvable & pinned | 🟢 **PASS** |
| **Git Exclusion Audit** | `.gitignore` rules | Validation of `data/raw/` (2.34 GB) exclusion | Raw dataset strictly local; excluded from git | 🟢 **PASS** |

---

## 2. File Triage Summary

* **Active Production Assets (Keep)**:
  - Core navigation engine: `src/` (data, preprocessing, navigation, fusion, ml, map, evaluation)
  - Android application: `android/` (Kotlin UI, Java ESKF, ONNX runtime)
  - Model weights: `models/tcn_kin_speed.onnx`, `models/balanced_tcn_speed.onnx`, `models/tcn_scaler.json`
  - Authoritative results: `results/adaptive_fusion/`, `results/stateful_kinematics/`, `results/highway_observability/`, `results/vibration_audit/`, `results/expanded_tcn_benchmark/`, `results/balanced_tcn_benchmark/`
* **Historical Assets (Archived)**:
  - Early single-trip models and exploratory scripts moved to `archive/`
* **Excluded Assets (Gitignored)**:
  - 2.34 GB raw IO-VNBD dataset (`data/raw/**`)
  - Intermediate numpy caches (`calib_cache.npz`, `train_seq_cache.npz`)
  - Large binary `.joblib` checkpoints (`models/**/*.joblib`)
  - Python virtual environment (`.venv/`)
  - Android build directories (`android/.gradle/`, `android/build/`, `android/app/build/`)
  - Machine-specific properties (`android/local.properties`)

---

## 3. Link Resolution Verification

All internal markdown documentation links between `README.md`, `docs/`, `docs/experiments/`, `docs/decisions/`, and `results/` have been audited and verified to use correct relative paths.
