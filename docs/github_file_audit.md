# GitHub Large File & Data Audit

> **Target Repository:** `24f2001869/NavSense`  
> **Policy:** GitHub file size limit is 100 MB hard limit; recommended <50 MB. No large binary datasets, checkpoints, or local caches should be committed.  
> **Date:** September 11, 2026  
> **Audit Status:** ✅ All large files identified, categorized, and filtered via `.gitignore`.

---

## 1. Executive Summary

A comprehensive scan of the repository discovered:
- **0 files > 500 MB**
- **4 files > 100 MB** (all located in `data/raw/IO-VNBD-repo/`, totaling ~800 MB compressed)
- **1 file > 50 MB and < 100 MB** (`android/app/build/outputs/apk/debug/app-debug.apk`, 69.78 MB)
- **72 files between 10 MB and 50 MB** (predominantly raw IO-VNBD CSVs in `data/raw/`, OSM map XMLs in `data/raw/maps/`, and legacy Random Forest `.joblib` checkpoints in `models/`)

**Crucial Finding:** Every single file > 10 MB belongs to directories or extensions that are explicitly excluded by `.gitignore`. The files proposed for Git tracking (source code, documentation, ONNX runtime models, curated evaluation results, and vector assets) total less than **35 MB**, ensuring fast cloning and complete GitHub compliance.

---

## 2. File Size Triage Matrix

| Size Range | Path / Pattern | Exact Size | Git Status | Reason / Justification |
| :--- | :--- | :--- | :--- | :--- |
| **> 100 MB** | `data/raw/IO-VNBD-repo/Unsynchronised V and S Dataset.zip` | 204.40 MB | 🔴 **EXCLUDED** (`data/raw/**`, `*.zip`) | External benchmark dataset. Must be downloaded via official repo. |
| **> 100 MB** | `data/raw/IO-VNBD-repo/Synchronised V abd S datasets.zip` | 194.17 MB | 🔴 **EXCLUDED** (`data/raw/**`, `*.zip`) | External benchmark dataset. Must be downloaded via official repo. |
| **> 100 MB** | `data/raw/IO-VNBD-repo/.git/lfs/objects/dc/...` | 204.40 MB | 🔴 **EXCLUDED** (`data/raw/**`) | Git-LFS object from cloned submodule. |
| **> 100 MB** | `data/raw/IO-VNBD-repo/.git/lfs/objects/62/...` | 194.17 MB | 🔴 **EXCLUDED** (`data/raw/**`) | Git-LFS object from cloned submodule. |
| **50–100 MB**| `android/app/build/outputs/apk/debug/app-debug.apk` | 69.78 MB | 🔴 **EXCLUDED** (`android/app/build/`, `*.apk`) | Android compilation build artifact. Can be built locally via `./gradlew assembleDebug`. |
| **10–50 MB** | `data/raw/maps/*.osm` (4 files) | 16.29–32.57 MB | 🔴 **EXCLUDED** (`data/raw/**`) | Raw OpenStreetMap XML exports for offline map-matching tests. |
| **10–50 MB** | `data/raw/IO-VNBD-repo/**/*.csv` (58 files) | 10.02–25.87 MB | 🔴 **EXCLUDED** (`data/raw/**`) | Raw uncompressed sensor logs from IO-VNBD. |
| **10–50 MB** | `models/**/*.joblib` (8 files) | 16.82–20.69 MB | 🔴 **EXCLUDED** (`models/**/*.joblib`) | Superseded Phase 4 Random Forest scikit-learn models (replaced by ONNX). |
| **10–50 MB** | `scratch/train_seq_cache.npz` | 15.75 MB | 🔴 **EXCLUDED** (`scratch/`, `*.npz`) | Temporary feature buffer cache. |

---

## 3. Production Model Artifacts Retained for GitHub

The operational deep learning models required for inference and Android deployment are lightweight and comply with GitHub's repository limits:

| File Path | Format | Size | Retained? | Purpose |
| :--- | :--- | :--- | :---: | :--- |
| `models/tcn_velocity_expanded.onnx` | ONNX | **162 KB** | 🟢 **YES** | Production deployment model for Android ONNX Runtime & Python inference. |
| `models/tcn_velocity_best.pt` | PyTorch | **1.54 MB** | 🟢 **YES** | PyTorch weights checkpoint for training validation & transfer learning. |
| `models/metadata/tcn_scaler_expanded.joblib` | Scikit-Learn Scaler | **1.2 KB** | 🟢 **YES** | 12-channel input feature normalization parameters. |
| `models/metadata/tcn_expanded_config.json` | JSON | **1.8 KB** | 🟢 **YES** | TCN architectural hyperparameters, receptive field, and dilation schedule. |

---

## 4. Verification Checklist

- [x] All files > 100 MB are ignored by `.gitignore`.
- [x] No Git-LFS dependencies are required for the published code repository.
- [x] Total trackable repository size is $< 35\text{ MB}$.
- [x] Cloned workspace will download in seconds on standard connections.
- [x] Instructions for fetching the 2.34 GB raw dataset are provided in [`data/README.md`](../data/README.md).
