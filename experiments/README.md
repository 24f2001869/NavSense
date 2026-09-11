# Experiments Code Directory Guide

This directory contains the historical sequence of automated experimental testbeds and stage benchmarks developed throughout the research campaign.

---

## Stage Script Index

| Stage Prefix | Scope / Scientific Focus | Key Benchmark Scripts | Primary Output Report |
|:---|:---|:---|:---|
| **Phase 0** | Reference Ground Truth Audit | `phase0_reference_audit.py` | `results/phase0_audit/` |
| **Phase 2** | Baseline ESKF Benchmark | `phase2_baseline_benchmark.py` | `results/phase2_baselines/` |
| **Phase 3** | Temporal Context Evaluation | `phase3_temporal_benchmark.py` | `results/phase3_temporal/` |
| **Phase 4** | Dilated TCN & NHC Ablation | `phase4_1_tcn_forensic_audit.py`<br>`phase4_2_controlled_ablation.py`<br>`phase4_3_standstill_gating.py`<br>`phase4_4_nhc_fusion_ablation.py`<br>`phase4_5_adaptive_nhc_fusion.py`<br>`phase4_6_map_matching_integration.py` | `results/phase4_*/` |
| **Phase 5** | Forensic Investigations | `phase5_1_field_forensic.py`<br>`train_balanced_tcn.py`<br>`evaluate_balanced_tcn.py` | `results/expanded_tcn_benchmark/`<br>`results/balanced_tcn_benchmark/` |
| **C5 Series** | Kinematics, Bias & Learned Speed | `validate_kinematic_integration_c5_2c1.py`<br>`run_rf_baseline_c5_3b2.py`<br>`run_vibration_characterization_c5_5_2.py` | `results/c5_*` |
| **C7 Series** | Kinematic Bounds & ZUPT | `run_zupt_oracle_c7_a1.py`<br>`run_kinematic_bounds_c7_b1.py` | `results/c7_*` |
| **C8 Series** | Multi-Hypothesis Map Matching | `run_closed_loop_map_matching_c8_4.py`<br>`audit_forward_axis_stability_c8_11.py` | `results/c8_*` |
| **C10 Series**| Field Telemetry Diagnostics | `c10_3_stationary_vibration_diagnostic.py`<br>`c10_7_vnhc_recovery_diagnostic.py` | `results/c10_*` |

---

## Active Benchmark Execution

To run the latest authoritative Phase 5 benchmarks, execute the unified scripts from `scripts/experiments/`:
* `scripts/experiments/run_adaptive_fusion_benchmark.py` (Phase 5.6)
* `scripts/experiments/run_stateful_kinematic_benchmark.py` (Phase 5.5)
* `scripts/experiments/run_highway_speed_observability.py` (Phase 5.4)
* `scripts/experiments/run_vibration_speed_audit.py` (Phase 5.3)
