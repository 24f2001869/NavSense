# Repository Audit & Inventory Report

**Date**: 2026-09-11  
**Project**: SIH26168 Intelligent Dead Reckoning (IDR) Navigation Engine  
**Audit Purpose**: Complete forensic inventory of repository assets, classification into **Keep**, **Archive**, and **Exclude**, and identification of dependencies before GitHub preparation.

---

## 1. High-Level Inventory Statistics

* **Total Workspace Files**: 3,405 files (~2,798 MB / 2.79 GB)
* **Raw Dataset (`data/raw/`)**: 1,473 files (~2,342 MB / 2.34 GB) — **Must remain strictly local (in `.gitignore`)**
* **Clean Code & Research Assets (excl. raw data, build, caches)**: 846 files (~311 MB)
* **Core Documentation & Code (excl. large binary `.joblib` checkpoints & `.npz` caches)**: ~128 MB

### Breakdown by File Extension (Top Categories):
| Extension | File Count | Size (MB) | Category Role |
|:---|:---:|:---:|:---|
| `.csv` | 1,170 | 1,734.7 MB | Raw IO-VNBD trips (local), benchmark outputs, telemetry logs |
| `.json` | 391 | 25.0 MB | Experiment metrics, gate decompositions, scalers, manifests |
| `.py` | 270 | 3.4 MB | Engine library (`src/`), stage experiments, analysis scripts |
| `.png` | 208 | 86.8 MB | Empirical diagnostic dashboards, spectrograms, waterfalls |
| `.md` | 101 | 1.4 MB | Stage reports, phase audits, documentation |
| `.joblib` | 25 | 158.2 MB | Trained Scikit-Learn baseline models (Random Forest / GBDT) |
| `.onnx` | 2 | 0.3 MB | Edge-deployable neural speed models (`tcn_kin_speed.onnx`, `balanced_tcn_speed.onnx`) |
| `.npz` | 2 | 25.0 MB | Intermediate preprocessing caches (`scratch/calib_cache.npz`, `train_seq_cache.npz`) |

---

## 2. Directory Structure & Asset Roles

```
SIH26168-IDR/
├── android/          # Native Android deployment prototype (Kotlin UI + Java ESKF + ONNX Runtime)
├── configs/          # Configuration manifests (currently empty, to be populated)
├── data/             # Datasets:
│   ├── raw/          # [EXCLUDE from git] Raw IO-VNBD synchronized dataset (2.34 GB)
│   ├── processed/    # Cleaned trip samples (1.75 MB)
│   ├── field/        # Captured phone sensor telemetry logs (5.8 MB)
│   └── samples/      # Lightweight test snippets for continuous integration
├── docs/             # Technical specifications, lab notebooks, forensic reports
├── experiments/      # Systematic stage benchmarks (C5, C6, C7, C8, C9, C10 series + Phases 0-5)
├── models/           # ONNX models, JSON feature scalers, and ML checkpoints
├── notebooks/        # Reproducible Jupyter analysis notebooks (01 through 07)
├── results/          # Authoritative experimental output CSVs, JSONs, and markdown reports
├── scratch/          # Development scripts, one-off investigative code, and intermediate caches
├── src/              # Core production Python navigation engine modules
└── tests/            # Test suite (causality, zero-leakage, numerical integrity)
```

---

## 3. Detailed Component Inventory & Triage

### 3.1 Source Code (`src/`) — 🟢 KEEP
* `src/data/`: IO-VNBD dataset builder, synchronization parsers, coordinate transforms.
* `src/preprocessing/`: Attitude initialization, gravity removal, PCA phone-to-vehicle alignment.
* `src/navigation/`: 15-state Error-State Kalman Filter (ESKF), strapdown mechanization.
* `src/fusion/`: Zero Velocity Detection (ZVD), Non-Holonomic Constraints (NHC), adaptive blending.
* `src/ml/`: Dilated TCN architecture (`temporal_speed_net.py`), loss density weighting (`loss_weighting.py`), stratified batch sampling (`speed_stratified_sampler.py`).
* `src/map/`: OpenStreetMap parser (`osm_parser.py`), Frenet geometry (`geometry.py`), Multi-Hypothesis Beam Search (`beam_search.py`).
* `src/mapping/`: Redundant directory with empty `__init__.py` $\implies$ **Consolidate into `src/map/`**.

### 3.2 Core Production Models (`models/`) — 🟢 KEEP
* `models/tcn_kin_speed.onnx` (257 KB): Production Expanded TCN forward-speed model.
* `models/tcn_scaler.json` (743 B): Input normalization parameters (means and standard deviations across 9 features).
* `models/balanced_tcn_speed.onnx` (43 KB + 238 KB weights): Target-balanced variant.
* `models/c5_1/`, `c5_2a/`, `c5_2b/`: Heavy `.joblib` checkpoints (158 MB total) $\implies$ **Move to `models/metadata/` with git tracking exclusions, preserving model config JSONs**.

### 3.3 Authoritative Experimental Results (`results/`) — 🟢 KEEP
* `results/adaptive_fusion/`: Phase 5.6 benchmark results (`adaptive_per_trip_results.csv`, `adaptive_aggregate_summary.csv`, `adaptive_fusion_report.md`).
* `results/stateful_kinematics/`: Phase 5.5 momentum and kinematic integration benchmark.
* `results/highway_observability/`: Phase 5.4 zero-acceleration cruise observability audit.
* `results/vibration_audit/`: Phase 5.3 spectral speed-correlation forensic audit.
* `results/balanced_tcn_benchmark/`: Phase 5.2 target-balancing benchmark.
* `results/expanded_tcn_benchmark/`: Phase 4 & Phase 5 per-trip forensic evaluation across 19 held-out trips.
* `results/phase4_6_map/`: Phase 4.6 closed-loop map matching evaluation.
* `results/c8_4_closed_loop_map_matching.json`: Stage C8-4 5-gate MHT map matching benchmark.

### 3.4 Root Artifacts & Loose Files Triage
* `IO-VNBD-master.zip` (1.3 MB) in root $\implies$ **EXCLUDE from GitHub (redundant zip archive)**.
* `phone_screenshot.png` (236 KB) in root $\implies$ **Move to `assets/screenshots/`**.
* `files/` (8 phone telemetry CSV files from Sept 9) $\implies$ **Move into `data/field/`**.
* `field_logs/` $\implies$ **Consolidate into `data/field/`**.
* `build/` (Java classes) $\implies$ **EXCLUDE via `.gitignore`**.

### 3.5 Scratch Directory Triage (`scratch/`)
`scratch/` contains ~100 scripts developed during rapid diagnosis. They are triaged as follows:
* **Promote to `scripts/experiments/`**:
  - `run_adaptive_fusion_benchmark.py` (Phase 5.6 reproduction)
  - `run_stateful_kinematic_benchmark.py` (Phase 5.5 reproduction)
  - `run_highway_speed_observability.py` (Phase 5.4 reproduction)
  - `run_vibration_speed_audit.py` (Phase 5.3 reproduction)
  - `run_expanded_tcn_forensics.py` (Phase 5.1/5.2 reproduction)
* **Promote to `scripts/evaluation/`**:
  - `analyze_60_70_spike.py` (pedestrian OOD forensic)
  - `audit_speed_distribution.py` (dataset density audit)
  - `field_forensics.py` (field test telemetry analysis)
* **Move to `archive/superseded/`**:
  - Historical one-off debugging scripts (`print_*.py`, `check_*.py`, `test_step.py`, etc.)
* **EXCLUDE from git**:
  - `scratch/calib_cache.npz` (8.5 MB) & `train_seq_cache.npz` (16.5 MB) — generated binary cache files.
  - `scratch/test_mock_raw.csv` (395 KB).

---

## 4. GitHub Exclusion Plan (`.gitignore`)

The following assets are classified as **EXCLUDE** and will never be committed to git:
1. **Raw Large Datasets**: `data/raw/**` (2.34 GB IO-VNBD dataset).
2. **Intermediate Numpy Caches**: `*.npz`, `calib_cache.npz`, `train_seq_cache.npz`.
3. **Python Virtual Environments & Caches**: `.venv/`, `venv/`, `__pycache__/`, `*.pyc`.
4. **Android Build Artifacts & IDE Metadata**:
   - `android/.gradle/`, `android/build/`, `android/app/build/`
   - `android/local.properties` (contains local Android SDK paths)
   - `.idea/`, `*.iml`, `.vscode/`
5. **Temporary Root Artifacts**: `IO-VNBD-master.zip`, root `build/`, OS files (`Thumbs.db`, `.DS_Store`).
6. **Heavy Obsolete Model Checkpoints**: `models/**/*.joblib` (158 MB).

---

## 5. Summary of Reproducibility Gaps & Remediation

| Issue Identified | Impact | Remediation in Phase 6 |
|:---|:---|:---|
| Missing `requirements.txt` | External users cannot install pinned Python dependencies | Generate clean, pinned `requirements.txt` |
| Missing `LICENSE` | Unclear intellectual property / open-source usage terms | Add standard MIT License |
| Missing dataset guide | External researchers cannot obtain or prepare IO-VNBD | Create `data/README.md` with split & setup guide |
| Scatter in `scratch/` | Critical benchmark scripts mixed with scratch printouts | Promote benchmark scripts to `scripts/experiments/` |
| Dual `src/map` & `src/mapping` | Potential import confusion | Consolidate into `src/map/` |
| Walking recordings misconstrued | Walking data could be mistaken for vehicle validation | Explicitly qualify in `android/README.md` and `docs/` |
