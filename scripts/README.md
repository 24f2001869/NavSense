# Scripts Directory Guide

This directory houses executable utilities and standalone reproducibility scripts categorized by their role in the research pipeline.

---

## Directory Organization

```text
scripts/
├── README.md              # This directory guide
├── experiments/           # Primary scripts to reproduce published benchmarks
│   ├── run_adaptive_fusion_benchmark.py     # Phase 5.6 benchmark
│   ├── run_stateful_kinematic_benchmark.py  # Phase 5.5 benchmark
│   ├── run_highway_speed_observability.py   # Phase 5.4 audit
│   ├── run_vibration_speed_audit.py         # Phase 5.3 audit
│   └── run_expanded_tcn_forensics.py        # Phase 5.1/5.2 benchmark
├── evaluation/            # Forensic diagnostics and post-hoc analyses
│   ├── analyze_60_70_spike.py               # Pedestrian OOD 70 m/s diagnostic
│   └── field_forensics.py                   # Phone field telemetry evaluator
├── data/                  # Dataset analysis and split verification
│   └── audit_speed_distribution.py          # IO-VNBD speed density auditor
└── utilities/             # Helper tools and data converters
```

All scripts support standalone command-line execution with `--help` flags.
