# Technical Documentation Index

Welcome to the technical documentation for the **SIH26168 Intelligent Dead Reckoning (IDR)** engine.

---

## 1. System Specifications & Engineering Guides
* [**Problem Statement (`problem_statement.md`)**](problem_statement.md) — Official SIH26168 challenge scope, operational conditions, and precision targets.
* [**System Requirements (`system_requirements.md`)**](system_requirements.md) — Functional and edge non-functional requirements.
* [**System Architecture (`architecture.md`)**](architecture.md) — 5-layer system decomposition and data flows.
* [**Methodology & Mathematics (`methodology.md`)**](methodology.md) — Strapdown mechanization, ESKF Jacobians, and Dilated TCN derivations.
* [**Dataset Technical Reference (`dataset.md`)**](dataset.md) — Sensor channels, coordinate conventions, and IO-VNBD structure.
* [**Evaluation Protocol (`evaluation_protocol.md`)**](evaluation_protocol.md) — Rolling window extraction, metrics, and pass criteria.
* [**Reproducibility Guide (`reproducibility.md`)**](reproducibility.md) — How to set up environments and reproduce every benchmark.
* [**Known Limitations (`limitations.md`)**](limitations.md) — Honest catalog of system boundaries and failure modes.

---

## 2. Research History & Evidence Audits
* [**Current Status (`current_status.md`)**](current_status.md) — 🟢 WHAT WE KNOW, 🟡 WHAT WE THINK, 🔴 WHAT WE DON'T KNOW.
* [**The Research Story (`research_story.md`)**](research_story.md) — Narrative intellectual history from initial failures to physical truth.
* [**Chronological Research Log (`research_log.md`)**](research_log.md) — Detailed timeline of every experiment and decision.
* [**Numerical Integrity Audit (`numerical_audit.md`)**](numerical_audit.md) — Every published metric verified against primary source files.
* [**Claim / Evidence Matrix (`claim_evidence_matrix.md`)**](claim_evidence_matrix.md) — Safe wording and evidentiary support for all technical claims.
* [**Repository Audit (`repository_audit.md`)**](repository_audit.md) — Complete file inventory, triage, and exclusion map.

---

## 3. Detailed Experiment Logs (`docs/experiments/`)
* [**Master Experiment Index (`experiments/README.md`)**](experiments/README.md) — Lookup matrix for all research phases.
* [Phase 0: Dataset & Reference Audit](experiments/phase0_reference_audit.md)
* [Phase 1: Pure IMU Strapdown Baseline](experiments/phase1_baseline.md)
* [Phase 2: 15-State Error-State Kalman Filter](experiments/phase2_eskf.md)
* [Phase 3: 3D Attitude & Non-Holonomic Constraints](experiments/phase3_attitude.md)
* [Phase 4: AI Velocity Estimation & Expanded TCN](experiments/phase4_ai_velocity.md)
* [Phase 5.1: Field Telemetry & Pedestrian OOD Bug](experiments/phase5_1_field_forensics.md)
* [Phase 5.2: Target-Balanced TCN Training](experiments/phase5_2_target_balancing.md)
* [Phase 5.3: Spectral Vibration Speed Information Audit](experiments/phase5_3_vibration_audit.md)
* [Phase 5.5: Stateful Kinematics & Momentum](experiments/phase5_5_stateful_kinematics.md)
* [Phase 5.6: Causal Adaptive Regime-Aware Velocity Fusion](experiments/phase5_6_adaptive_fusion.md)

---

## 4. Architectural Decision Records (`docs/decisions/`)
* [**Accepted Approaches (`decisions/accepted_approaches.md`)**](decisions/accepted_approaches.md) — Validated core methods.
* [**Rejected Approaches (`decisions/rejected_approaches.md`)**](decisions/rejected_approaches.md) — What failed and why.
* [**Open Questions (`decisions/open_questions.md`)**](decisions/open_questions.md) — Unresolved physical and algorithmic challenges.
