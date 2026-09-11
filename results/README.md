# Experimental Results Directory Guide

This directory contains the authoritative quantitative experimental outputs, benchmark metrics, and forensic visualizations generated across the research phases of the SIH26168 project.

---

## 1. Directory Structure & Results Hierarchy

```text
results/
├── README.md                          # This directory guide
├── RESULTS_INDEX.md                   # Master index of all authoritative experiment results
│
├── adaptive_fusion/                   # Phase 5.6: Causal Adaptive Regime Velocity Fusion
│   ├── adaptive_per_trip_results.csv  # 13-trip 60s & 18-trip 10s detailed outputs
│   ├── adaptive_aggregate_summary.csv # Multi-horizon benchmark summary (10s, 20s, 30s, 60s)
│   ├── adaptive_fusion_plots.png      # 4-panel diagnostic dashboard
│   ├── adaptive_fusion_summary.json   # Machine-readable JSON summary
│   └── adaptive_fusion_report.md      # Comprehensive scientific report
│
├── stateful_kinematics/               # Phase 5.5: Momentum & Stateful Kinematics
│   ├── stateful_kinematics_summary.csv# Pure kin vs momentum vs static TCN outputs
│   ├── stateful_kinematics_plots.png  # Comparison plots
│   └── stateful_kinematics_report.md  # Detailed forensic report
│
├── highway_observability/             # Phase 5.4: Steady-State Cruise Observability Forensics
│   ├── highway_observability_metrics.json # ROC-AUC & mutual information metrics
│   ├── highway_observability_plots.png    # Feature distribution & ROC curves
│   └── highway_observability_report.md    # Observability bound report
│
├── vibration_audit/                   # Phase 5.3: Spectral Vibration Speed Correlation Audit
│   ├── vibration_audit_summary.json   # Welch PSD & FFT correlation statistics
│   ├── vibration_spectrum_plots.png   # 4-panel spectrogram & frequency vs speed plots
│   └── vibration_audit_report.md      # Negative finding forensic report
│
├── balanced_tcn_benchmark/            # Phase 5.2: Target-Balanced TCN Training
│   ├── per_trip_balanced_benchmark.csv# Per-trip evaluation with inverse-density loss
│   ├── balanced_tcn_benchmark_summary.png # Pareto trade-off visualizations
│   └── balanced_tcn_weights.pth       # PyTorch checkpoint (65k parameters)
│
├── expanded_tcn_benchmark/            # Phase 4 & Phase 5: Expanded TCN Held-Out Benchmark
│   ├── per_trip_forensic.csv          # Complete 19 held-out trips master table
│   ├── overall_benchmark_summary.json # Aggregated metrics across categories
│   ├── expanded_tcn_scaler.json       # Feature normalization parameters
│   └── expanded_tcn_benchmark_summary.png # Per-trip drift & category waterfalls
│
├── phase4_6_map/                      # Phase 4.6: Closed-Loop Topological Map Matching
│   ├── fig1_map_matching_drift_waterfall.png # Drift comparison across M0–M5
│   ├── fig2_heading_and_crosstrack_decomposition.png
│   └── phase4_6_map_matching_report.md# Rejection of naive heading injection
│
└── figures/                           # Stage C5 to C10 diagnostic plots & dashboards
```

---

## 2. Authoritative Primary Files

When citing figures or numbers in papers or presentations, reference these primary source files:
* **Adaptive Regime Fusion**: [`adaptive_fusion/adaptive_aggregate_summary.csv`](adaptive_fusion/adaptive_aggregate_summary.csv)
* **19-Trip Forensic Table**: [`expanded_tcn_benchmark/per_trip_forensic.csv`](expanded_tcn_benchmark/per_trip_forensic.csv)
* **Highway Cruise Observability**: [`highway_observability/highway_observability_summary.json`](highway_observability/highway_observability_summary.json)
* **Vibration Spectrum Audit**: [`vibration_audit/vibration_audit_summary.json`](vibration_audit/vibration_audit_summary.json)
