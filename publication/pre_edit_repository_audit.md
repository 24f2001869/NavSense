# Pre-Edit Repository & File Inventory Audit

**Project:** NavSense / SIH26168  
**Repository Path:** `c:\Users\rk930\Desktop\SIH26168-IDR`  
**Auditor:** Senior Academic Publication Engineer  
**Audit Date:** 2026-09-12  

---

## A. Files Discovered (Repository Census)

* **Root Directory (5 files)**: `README.md`, `LICENSE`, `setup.py`, `pyproject.toml`, `.gitignore`.
* **Documentation (`docs/`, 54 files)**:
  * `docs/paper/` (10 files): Core manuscript draft and 9 comprehensive audit files.
  * `docs/experiments/` (18 files): Detailed Phase 0 to Phase 5.6 empirical experiment reports.
  * `docs/` root: `START_HERE.md`, `IO_VNBD_PAPER_NOTES.md`, `limitations.md`, `numerical_audit.md`, `reproducibility.md`, `research_log.md`, `system_architecture.md`.
* **Empirical Results (`results/`, 394 files across 24 directories)**:
  * Key summaries: `results/expanded_tcn_benchmark/`, `results/adaptive_fusion/`, `results/phase4_4_fusion/`, `results/phase4_5_adaptive/`, `results/phase4_6_map/`, `results/highway_observability/`, `results/vibration_audit/`, `results/field_analysis/`.
* **Visual Assets (`assets/`, 16 files)**:
  * `assets/architecture/system_architecture.svg`
  * `assets/experiments/` (14 high-resolution diagnostic PNG figures)
* **Scripts (`scripts/`, 11 files)**:
  * Core evaluation and data auditing: `run_highway_speed_observability.py`, `run_vibration_speed_census.py`, `run_adaptive_fusion_benchmark.py`, `audit_speed_distribution.py`, `evaluate_all_horizons.py`.
* **Android Edge Engine (`android/`, 1,117 files)**:
  * Native Kotlin/Java ESKF implementation, ONNX Runtime Mobile integration, `CompassRoseView.kt`, `MainActivity.kt`.
* **Test Verification Suite (`tests/`, 2 files)**:
  * `tests/test_causality_and_leakage.py`: Automated mathematical causality (future gradient $= 0$), trailing-edge window alignment, and trip-disjoint split tests.

---

## B. Files Relevant to Scientific Claims

1. `docs/paper/claim_evidence_audit.md`: Exhaustive 20-claim matrix classifying claims into Tier A (empirically demonstrated), Tier B (empirically bounded), Tier C (preliminary/superseded), and Tier D (strictly disclaimed/unsupported).
2. `docs/paper/methodology_audit.md`: Formalizes the IO-VNBD Racelogic VBOX Video HD2 logger (10 Hz) and Ford Fiesta CAN bus reference hierarchy (-100 ms latency compensated), single-driver Driver E census, and trip-disjoint partitioning.
3. `results/expanded_tcn_benchmark/held_out_evaluation_summary.json`: Multi-trip scaling evidence ($6 \to 39$ trips).
4. `results/phase4_4_fusion/phase4_4_fusion_report.md`: 8-way kinematic fusion ablation proving unconditional NHC degradation ($-45.47\%$).
5. `results/phase4_5_adaptive/phase4_5_adaptive_report.md`: Decoupled velocity damping formulation (+54.62% drift reduction).
6. `results/phase4_6_map/phase4_6_map_matching_report.md`: Closed-loop map feedback failure ($-125.47\%$).
7. `results/highway_observability/highway_observability_summary.json`: Motorway `V-Vfa02` empirical separability ($\text{ROC-AUC} = 0.625$, $-3.5\text{ m/s}$ high-speed bias).
8. `results/vibration_audit/vibration_speed_audit.csv`: >450,000 one-second FFT windows across 64 trips ($r = -0.032$, dominant peak 2.2–2.5 Hz).
9. `results/field_analysis/`: Pedestrian out-of-distribution dynamics (+26.08$\sigma$ yaw rate, 71.66 m/s spike).
10. `results/adaptive_fusion/adaptive_aggregate_summary.csv`: Multi-horizon benchmark across 19 held-out trips (3/13 passes at 60s, 16.76% macro-average normalized drift).

---

## C. Files Relevant to Publication

1. `docs/paper/paper_draft.md`: Authoritative full draft manuscript (93.5 KB, 12 IEEE-style sections).
2. `docs/paper/table_plan.md`: 13 detailed tables with schema and source data links.
3. `docs/paper/figure_plan.md`: 13 figures mapped to visual assets in `assets/`.
4. `docs/paper/literature_novelty_audit.md`: Prior art comparison table and literature contextualization.
5. `docs/paper/reviewer_attack.md`: 30 hostile reviewer Q&As.
6. `docs/paper/publication_readiness.md`: 20-point quality gate audit.
7. `docs/paper/author_input_required.md`: Administrative author checklist.
8. `publication/journal_of_navigation_requirements.md`: Verified Cambridge instructions.
9. `publication/zero_cost_publication_strategy.md`: Protocol for conventional zero-cost publication.

---

## D. Files Containing Numerical Results

* `results/expanded_tcn_benchmark/held_out_evaluation_summary.json`: 39-trip test MAE ($2.76\text{ m/s}$), RMSE ($3.59\text{ m/s}$), Bias ($-0.48\text{ m/s}$), 60s drift ($93.0\text{ m}$).
* `results/phase4_4_fusion/phase4_4_fusion_summary.csv`: NHC ablation ($843.4\text{ m} \to 1226.9\text{ m}$, $NIS_x$ surging $6.55 \to 82.12$).
* `results/phase4_5_adaptive/phase4_5_adaptive_summary.csv`: Variant V1 decoupled damping ($382.74\text{ m}$).
* `results/phase4_6_map/phase4_6_map_summary.csv`: Closed-loop map degradation ($578.3\text{ m} \to 1303.9\text{ m}$).
* `results/adaptive_fusion/adaptive_aggregate_summary.csv`: Aggregate multi-horizon benchmark (10s, 20s, 30s, 60s).
* `results/adaptive_fusion/adaptive_per_trip_results.csv`: Per-trip 60s breakdown across 19 test trips.
* `android/README.md` & `results/c10_5_test_protocol.md`: Android execution latency (4.2–7.8 ms ONNX, 0.8–1.4 ms ESKF, 8.63 Hz loop rate, numerical parity $\le 0.038\text{ m}$).

---

## E. Files Containing References & Citations

* `docs/paper/paper_draft.md`: Contains 20 numbered citations (`[1]` to `[20]`) and References section at end.
* `docs/paper/literature_novelty_audit.md`: Verified DOIs and comparative breakdown for AVNet (2025), Onyekpe et al. (2021), Wang et al. (2025), AI-IMU (2020), Brossard et al. (2020), RoNIN (2020).
* `docs/IO_VNBD_PAPER_NOTES.md`: Early literature notes and IO-VNBD dataset documentation.

---

## F. Existing Manuscript Versions & Authoritative Draft Determination

Two files with manuscript-like titles exist:
1. `docs/IO_VNBD_PAPER_NOTES.md` (39,641 bytes, modified 2026-09-08): Preliminary literature review and dataset extraction notes.
2. `docs/paper/paper_draft.md` (93,537 bytes, modified 2026-09-12): **Authoritative current draft**. Incorporates all recent forensic corrections, precise dataset denominators, softened physical observability wording, decoupled damping results, and sanitized nomenclature.

**Verdict**: `docs/paper/paper_draft.md` is unequivocally the single authoritative base manuscript.

---

## G. Existing Figure Sources

* Figure 1 (Architecture): `assets/architecture/system_architecture.svg`
* Figure 2 (Open-Loop Inertial): `assets/experiments/outage_track_drift_202714.png`
* Figure 3 (TCN Scaling): `assets/experiments/expanded_tcn_benchmark_summary.png`
* Figure 4 (TCN Receptive Field / Kinematics): `assets/experiments/tcn_kin_forensic_analysis.png`
* Figure 5 (NHC Lateral Instability): `assets/experiments/stateful_kinematics_plots.png`
* Figure 6 (Highway Bias / Observability): `assets/experiments/highway_observability_plots.png`
* Figure 7 (Target Balancing): `assets/experiments/balanced_tcn_benchmark_summary.png`
* Figure 8 (Map Matching Failure): `assets/experiments/c8_4_closed_loop_map_matching.png`
* Figure 9 (Multi-Trip Forensics): `assets/experiments/per_trip_forensic_plots.png`
* Figure 10 (Vibration Spectrum): `assets/experiments/vibration_spectrum_plots.png`
* Figure 11 (Adaptive Fusion): `assets/experiments/adaptive_fusion_plots.png`
* Figure 12 (Pedestrian OOD): `assets/experiments/rooftop_gait_speed_analysis.png`
* Figure 13 (Android Edge Timing): `assets/experiments/field_replay_improvements_comparison.png`

---

## H. Existing Table Sources

* Tables 1 & 2 (Census & Split): `docs/paper/table_plan.md` (derived from IO-VNBD dataset manifest).
* Table 3 (Inertial Baselines): `results/phase2_baselines/` & `adaptive_aggregate_summary.csv`.
* Table 4 (TCN Scaling): `results/expanded_tcn_benchmark/held_out_evaluation_summary.json`.
* Table 5 (Fusion Ablation): `results/phase4_4_fusion/phase4_4_fusion_summary.csv`.
* Table 6 (Map Ablation): `results/phase4_6_map/phase4_6_map_summary.csv`.
* Table 7 (Decoupled Recovery): `results/phase4_5_adaptive/phase4_5_adaptive_summary.csv`.
* Table 8 (Target Balancing): `results/balanced_tcn_benchmark/target_balanced_summary.json`.
* Table 9 (Vibration Census): `results/vibration_audit/vibration_speed_audit.csv`.
* Table 10 (Multi-Horizon Summary): `results/adaptive_fusion/adaptive_aggregate_summary.csv`.
* Table 11 (Per-Trip 60s Outage): `results/adaptive_fusion/adaptive_per_trip_results.csv`.
* Table 12 (Android Parity): `results/field_audit_abc/` & `android/README.md`.
* Table 13 (Pedestrian OOD): `results/field_analysis/`.

---

## I. Potential Contradictions Flagged for Verification

1. **Applied Sciences 2025 Speed Estimation Paper**: Referenced as both `[8]` and `[11]` in `paper_draft.md`. Must be cross-checked for duplicate authorship and unified into a single Harvard citation.
2. **Receptive Field vs Input Buffer**: Input window is 100 samples (10.0 s), but causal dilated convolution receptive field is 63 samples (6.3 s). The text must consistently distinguish the buffer length from the filter reach.
3. **Usable Model Samples vs Raw Synchronized Epochs**: Raw dataset contains 1,182,661 epochs across 64 trips; usable partitioned model training samples total 650,661 (478,210 train / 58,912 val / 113,539 test) due to 100-sample initialization buffer exclusions. Must never conflate raw epochs with usable samples.
4. **Android Replay vs Physical Drive**: Android evaluation was a 178.9 s golden telemetry replay on a OnePlus smartphone, not a physical vehicle drive. Physical testing was strictly bench and pedestrian walking.
5. **60-Second Denominator**: 13 usable trips at 60s ($T \ge 70\text{ s}$ duration threshold), passing trips = 3/13 (23.08%). Must never be written as 3/19.

---

## J. Missing Information Identified

1. **ORCID iD**: Missing in source repository; flagged in `publication/ORCID_REQUIRED.md` as a manual author action.
2. **Journal Submission Template (`nav.cls`)**: Not included in repository. As instructed, Microsoft Word (`.docx`) configured strictly to Cambridge 12pt Times New Roman specifications will serve as the primary submission vehicle.
3. **Keywords on Manuscript**: Journal of Navigation instructions explicitly mandate excluding keywords from the manuscript text (they are chosen in ScholarOne). Keywords will be removed from manuscript body and documented in `scholarone_metadata.md`.

---

## K. Stale or Generated Files

* Temporary scratch files in `scratch/` and experimental logs in `field_logs/` contain ephemeral testing traces. They are excluded from publication artifacts.
* All generated publication assets will be strictly isolated in `publication/` and `publication/journal_of_navigation_submission/`.
