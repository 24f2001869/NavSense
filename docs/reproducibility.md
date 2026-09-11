# Reproducibility & Benchmark Execution Guide

This document provides step-by-step instructions for reproducing all major experimental results, training pipelines, and forensic benchmarks published in this repository.

---

## 1. System Requirements & Environment Setup

* **Operating System**: Linux (Ubuntu 20.04/22.04) or Windows 10/11
* **Python Version**: Python 3.10 or 3.11
* **Hardware**:
  - CPU: 4+ cores (Intel Core i5/i7 or AMD Ryzen 5/7)
  - RAM: Minimum 8 GB (16 GB recommended for full dataset audits)
  - Disk Space: ~3.5 GB (including raw dataset)
  - GPU: Optional (all models are CPU-optimized and evaluate in seconds)

### Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/SIH26168-IDR.git
cd SIH26168-IDR

# Create isolated Python virtual environment
python -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Dataset Setup Verification

1. Download or obtain the **IO-VNBD** dataset as specified in [`data/README.md`](../data/README.md).
2. Extract the dataset so that `Categorised IOVNB Dataset` is located under:
   `data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset`
3. Verify discovery using:
   ```bash
   python -c "
   from src.data.dataset_builder import IOVNBDDatasetBuilder
   b = IOVNBDDatasetBuilder()
   trips = b.discover_trips()
   print(f'Discovered {len(trips)} total trips.')
   assert len(trips) >= 58, 'Missing expected IO-VNBD trips!'
   "
   ```

---

## 3. Running Authoritative Stage Benchmarks

### 3.1 Phase 5.6: Adaptive Regime-Aware Velocity Fusion Benchmark
Reproduces the multi-horizon comparison (10s, 20s, 30s, 60s) across all 19 held-out test trips:
```bash
python scripts/experiments/run_adaptive_fusion_benchmark.py
```
* **Outputs Generated**:
  - `results/adaptive_fusion/adaptive_per_trip_results.csv`
  - `results/adaptive_fusion/adaptive_aggregate_summary.csv`
  - `results/adaptive_fusion/adaptive_fusion_plots.png`
  - `results/adaptive_fusion/adaptive_fusion_summary.json`

### 3.2 Phase 5.5: Stateful Kinematic Integration Benchmark
Evaluates pure kinematics, fixed damped momentum, and static TCN across held-out trips:
```bash
python scripts/experiments/run_stateful_kinematic_benchmark.py
```
* **Outputs Generated**:
  - `results/stateful_kinematics/stateful_kinematics_summary.csv`
  - `results/stateful_kinematics/stateful_kinematics_plots.png`

### 3.3 Phase 5.4: Highway Steady-State Cruise Observability Audit
Evaluates statistical separability (ROC-AUC, mutual information) between 85 km/h and 105 km/h cruise:
```bash
python scripts/experiments/run_highway_speed_observability.py
```
* **Outputs Generated**:
  - `results/highway_observability/highway_observability_metrics.json`
  - `results/highway_observability/highway_observability_plots.png`

### 3.4 Phase 5.3: Spectral Vibration Speed-Information Audit
Executes the causal Welch PSD and FFT peak audit across 64 vehicle trips (>450,000 windows):
```bash
python scripts/experiments/run_vibration_speed_audit.py
```
* **Outputs Generated**:
  - `results/vibration_audit/vibration_audit_summary.json`
  - `results/vibration_audit/vibration_spectrum_plots.png`

### 3.5 Phase 5.1 & 5.2: Expanded TCN Forensics on Held-Out Test Set
Evaluates the Expanded TCN across all 19 held-out test trips:
```bash
python scripts/experiments/run_expanded_tcn_forensics.py
```
* **Outputs Generated**:
  - `results/expanded_tcn_benchmark/per_trip_forensic.csv`
  - `results/expanded_tcn_benchmark/overall_benchmark_summary.json`
  - `results/expanded_tcn_benchmark/expanded_tcn_benchmark_summary.png`

---

## 4. Running Verification Unit Tests

To verify zero data leakage and mathematical causality across all dataset loaders:
```bash
python tests/test_causality_and_leakage.py
```

---

## 5. Building the Android Edge Application

```bash
cd android

# Build debug APK
./gradlew assembleDebug

# Output APK path:
# android/app/build/outputs/apk/debug/app-debug.apk
```
