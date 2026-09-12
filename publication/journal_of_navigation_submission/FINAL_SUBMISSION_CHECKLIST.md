# Final Pre-Submission Checklist: NavSense / SIH26168

**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Article Type:** Research Article  
**Author:** Rahul Kumar  
**Date:** September 2026  

---

## 1. Scientific Integrity & Evidence Discipline

- [x] **Scientific Freeze Maintained:** Zero changes to empirical results, model architectures, trained weights, or evaluation splits.
- [x] **Dataset Grounding:** Public IO-VNBD dataset (Onyekpe et al., 2021) correctly cited with DOI (`10.1016/j.dib.2021.106885`). 64 trips, 16.27 h, 585,643 synchronized passenger-car epochs, 902.4 km, single driver (Driver E) acknowledged.
- [x] **Partition Integrity:** Whole-trip-disjoint split verified (39 train, 6 validation, 19 held-out test). Zero temporal overlap or leakage.
- [x] **Sample Counts Verified:** Total usable sliding sequence samples verified at 579,307 across 585,643 synchronized epochs (448,993 train epochs $\to$ 445,132 windows; 21,230 val epochs $\to$ 20,636 windows; 115,420 test epochs $\to$ 113,539 windows). Initial 100-sample buffer (excluding 99 samples per trip) mathematically accounted for.
- [x] **Model Parameters Verified:** Causal TCN parameter count verified at exactly 60,225 weights (4 residual blocks, dilations {1, 2, 4, 8}, kernel size 3, 64 channels). 9 input channels. 61-sample causal receptive field corresponding to 6.0 s causal look-back at 10 Hz ($[t-60, t]$).
- [x] **Scaling Numbers Verified:** 6-trip vs. 39-trip scaling metrics audited: MAE 6.17 $\to$ 2.76 m/s (55.27%), RMSE 6.96 $\to$ 3.59 m/s, bias -3.06 $\to$ -0.41 m/s, 30 s drift 153.3 $\to$ 52.6 m, 60 s drift 263.6 $\to$ 93.0 m (64.72%). Training duration labeled as 12.5 h (448,993 epochs).
- [x] **Constraint & Fusion Numbers Verified:** Unconditional lateral NHC degradation verified at +45.47% (843.4 $\to$ 1226.9 m across 83 windows, $NIS_x$ surge to 82.12); decoupled damping recovery verified at 382.74 m (+54.62% improvement, bounded $NIS_x = 6.60$).
- [x] **Map-Feedback Numbers Verified:** Closed-loop heading feedback degradation verified at -125.47% (578.3 $\to$ 1303.9 m).
- [x] **Highway & Vibration Metrics Verified:** Highway cruise binary speed classification (80 vs. 110 km/h) ROC-AUC = 0.625, accuracy 59.47%, and negative bias ($-3.5\text{ m/s}$, $r = -0.4866$) verified. Spectral vibration correlation ($r = -0.032$, $\rho = -0.028$) across >450,000 windows verified; motorway 2.86–2.93 Hz persistent peak and corpus-wide 2.2–2.5 Hz mode identified as chassis dynamics, not tire-speed harmonics. Zero Random Forest or Ridge models.
- [x] **Pedestrian OOD Metrics Verified:** Peak velocity (71.66 m/s), yaw rate (+26.08$\sigma$, 3.81 rad/s), ablated gyro peak (27.71 m/s) verified as sensor stress test, NOT vehicle validation.
- [x] **Android Edge Metrics Verified:** On-device ONNX inference (4.2–7.8 ms), ESKF (0.8–1.4 ms), mean epoch latency 9.15 ms (99th percentile 13.89 ms) verified on Google Pixel 7a. Numerical parity with Python engine verified (0.24177 m max position difference, 0.01829 deg heading difference). Callback jitter (mean interval 115.8 ms, effective rate 8.63 Hz vs 10.0 Hz nominal, range 104–127 ms) and hardware nanosecond timestamps documented.
- [x] **Benchmark Reality Check Verified:** 60-second blackout macro-average drift verified at 16.76% (96.65 m mean drift across all 13 test routes). Exactly 3 of 13 usable trips pass the <10% threshold (23.08% pass rate). Table 7 and Supplementary Table S1 reproduce each other with 100% arithmetic identity from `adaptive_per_trip_results.csv`.
- [x] **No Unsupported Claims:** No claims of centimeter-level ground truth, physical road-vehicle testing, driver invariance, universal GNSS-denied solution, or guaranteed competitive compliance.

---

## 2. Journal Formatting Compliance (Cambridge University Press)

- [x] **Language:** UK English throughout (`generalisation`, `optimise`, `modelled`, `metre`, `kilometre`, `behaviour`, etc.).
- [x] **Target Typography:** 12 pt Times New Roman, 2.54 cm (1 inch) margins, single column, single spacing (1.0), 0 pt before/after.
- [x] **Page Budget:** Exactly 18 pages for main manuscript (Cambridge limit: $\le 20$ pages).
- [x] **Headings Format:** Unbolded, left-aligned headings per Cambridge JoN instructions ("no unnecessary bold headings; simple formatting"). Declaration labels placed in separate left-aligned paragraphs.
- [x] **Abstract:** Exactly 141 words (target: ~130–150 words, $\le 150$ words), single self-contained paragraph, zero citations, no unexplained acronyms.
- [x] **Keywords Omitted from Text:** Manuscript body contains no keywords section per Cambridge JoN style (keywords entered in ScholarOne).
- [x] **Harvard Citations:** In-text citations converted to Harvard author-date format: `(Kemp, 1998)`, `(Dissanayake et al., 2001)`, `(Onyekpe et al., 2021)`, `(Shin et al., 2025)`, `(Qian et al., 2025)`, `(Xiao et al., 2025)`. Zero bracketed numbers.
- [x] **Alphabetical Reference List:** Unnumbered reference list sorted alphabetically by first author's surname. Zero hanging indent per Cambridge instructions ("Indents should NOT be used"). Exactly 17 verified entries (all cited in body).
- [x] **Applied Sciences Duplicate & Obsolete References Resolved:** Merged duplicate entries [8] and [11] into Shin et al. (2025); corrected AVNet to Qian et al. (2025) and sequence learning to Xiao et al. (2025); purged uncited/unverified records (Goodall et al., 2006; Rohani et al., 2023).
- [x] **Table Captions:** Placed **ABOVE** tables (`Table 1.` to `Table 7.`). Table headers repeated across pages.
- [x] **Figure Captions:** Placed **BENEATH** figures (`Figure 1.` to `Figure 8.`).
- [x] **Equations:** Rendered natively via Word OMML mathematics (0 raw LaTeX delimiters). Sequentially numbered with Arabic numerals in parentheses: `(1)` to `(13)`, centered in 2-column tables with numbers right-aligned.

---

## 3. Author & Affiliation Integrity

- [x] **Sole Author:** Rahul Kumar.
- [x] **Primary Research Affiliation:** Integrated M.Tech. (Materials Engineering), School of Engineering Sciences & Technology, University of Hyderabad, Hyderabad, India.
- [x] **Corresponding Email:** `24f2001869@ds.study.iitm.ac.in`.
- [x] **Institutional Policy Compliance:** IIT Madras BS degree programme omitted from research affiliation block per Cambridge affiliation instructions to reflect primary physical research institution.
- [ ] **ORCID Supplied Manually:** Author must authenticate ORCID profile in ScholarOne during submission (Flagged in `publication/ORCID_REQUIRED.md`).

---

## 4. Mandatory Declarations & Ethics

- [x] **Funding Declaration:** Present ("No specific external funding was received for this research.").
- [x] **Competing Interests Declaration:** Present ("The author declares no competing financial or non-financial interests.").
- [x] **Data Availability Statement:** Present with DOI link to IO-VNBD dataset.
- [x] **Acknowledgements Section:** Present before Funding, acknowledging Google Antigravity Agentic Assistant / Gemini 2.5 Flash for publication engineering, language harmonisation, and automated consistency verification, with explicit author verification and full responsibility.
- [x] **AI Use Disclosure:** Present in manuscript and Cover Letter per Cambridge guidelines, detailing tool name, version (Gemini 2.5 Flash), access links, date (September 2026), automated verification scripts scope, deterministic Python/Matplotlib figure generation, and independent author verification.

---

## 5. Zero-Cost Publication Route

- [x] **Route Selected:** Conventional / Subscription publication; Gold Open Access not selected.
- [x] **Cost Policy Verified:** No Gold-OA APC is incurred when publishing through the conventional route; optional paid services and print-colour charges are not selected.
- [x] **Colour Printing:** Free online colour confirmed; no paid colour print requested.
- [x] **Third-Party Services:** No paid language editing, submission, or formatting services selected.

---

## 6. Submission Deliverables Summary

- [x] `publication/journal_of_navigation_submission/01_JON_manuscript.docx` (Primary submission file, 18 pages, 12pt Times New Roman, 1.0 single spacing, 0pt before/after, 2.54 cm margins, single column, native OMML equations).
- [x] `publication/journal_of_navigation_submission/02_JON_manuscript.pdf` (Word-exported companion PDF, exactly 18 pages).
- [x] `publication/journal_of_navigation_submission/figures/` (Figures 1–8 in 600 DPI production TIFF format + 300 DPI PNG).
- [x] `publication/journal_of_navigation_submission/tables/` (Tables 1–7 in clean CSV format; Table 4 without rationale column, Table 7 concise category-level).
- [x] `publication/journal_of_navigation_submission/supplementary/Supplementary_Material.md` (Clean, reader-facing academic document with Tables S1–S3, Sections S1–S2).
- [x] `publication/journal_of_navigation_submission/supplementary/Supplementary_Material.docx` (Word version of Supplementary Material, 4 pages).
- [x] `publication/journal_of_navigation_submission/supplementary/Supplementary_Material.pdf` (PDF version of Supplementary Material, 4 pages).
- [x] `publication/journal_of_navigation_submission/scholarone_metadata.md` (Complete portal copy-paste fields).
- [x] `publication/journal_of_navigation_submission/cover_letter.md` (Formal submission cover letter).
- [x] `publication/journal_of_navigation_submission/CHANGELOG.md` (Audit trail of publication-engineering conversions).
- [x] `publication/journal_of_navigation_submission/accessibility_descriptions.md` (WCAG 2.1 Level AA alt-text for all artwork).

---

## 7. Rendered Artifact Semantic Verification

| Artifact | File Exists | Correct DPI | Correct Filename | Caption Matches | Accessibility Matches | Panels Belong to Experiment | No Obsolete Models (RF/Ridge) | Numbers Match Source Table | Visual Inspection Completed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Figure 1 (Architecture)** | [x] | [x] 600 DPI | [x] Figure_01 | [x] Matched | [x] Matched | [x] 5 Layers, 4 blocks | [x] Pure TCN + ESKF | [x] 60,225 weights, 61 steps | [x] Passed (100 ms epoch budget badge) |
| **Figure 2 (Vta20 Divergence)** | [x] | [x] 600 DPI | [x] Figure_02 | [x] Matched | [x] Matched | [x] 2 Panels (Trajectory + Drift) | [x] Zero rooftop debug data | [x] 23.7m, 57.0m, 109.6m, 324.2m | [x] Passed (Replaced rooftop artifact) |
| **Figure 3 (Scaling Benchmark)** | [x] | [x] 600 DPI | [x] Figure_03 | [x] Matched | [x] Matched | [x] 4 Panels (MAE, Drift, Dist, Box) | [x] Zero Random Forest | [x] 6.17 $\to$ 2.76, 263.6 $\to$ 93.0 | [x] Passed (4 panels fully described in accessibility) |
| **Figure 4 (NHC Instability)** | [x] | [x] 600 DPI | [x] Figure_04 | [x] Matched | [x] Matched | [x] 3 Panels (Traj, NISx, Heading) | [x] Pure physics / ESKF | [x] 843.4 $\to$ 1226.9 $\to$ 382.7, NIS 82.12 | [x] Passed (Baseline ref level NIS 6.55, Decoupled V1) |
| **Figure 5 (Map Matching)** | [x] | [x] 600 DPI | [x] Figure_05 | [x] Matched | [x] Matched | [x] 2 Panels (Shadow vs Closed-Loop) | [x] Clean OSM traces | [x] 578.3 m vs 1303.9 m (+125.5%) | [x] Passed (Raster 47.0% rate verified; 47/83 purged) |
| **Figure 6 (Highway Cruise)** | [x] | [x] 600 DPI | [x] Figure_06 | [x] Matched | [x] Matched | [x] 2 Panels (ROC + Residuals) | [x] 9 Causal Channels (6 IMU + 3 derived) | [x] AUC 0.625, bias -3.5 m/s, r = -0.4866 | [x] Passed (≥90 km/h harmonized across text/art) |
| **Figure 7 (Spectral Vibration)** | [x] | [x] 600 DPI | [x] Figure_07 | [x] Matched | [x] Matched | [x] 2 Panels (PSD Bands + r(f,v)) | [x] Persistent Peak Band | [x] 2.86-2.93 Hz, r = -0.032, rho = -0.028 | [x] Passed (Wide 3214x1354 600 DPI TIFF synchronized) |
| **Figure 8 (Edge Deployment)** | [x] | [x] 600 DPI | [x] Figure_08 | [x] Matched | [x] Matched | [x] 4 Panels (Walk, Ablation, Interval, Latency) | [x] Feature Ablation | [x] 71.66 m/s $\to$ 27.71 m/s, 115.8 ms, 9.15 ms | [x] Passed (Title: '(b) Peak Prediction Under Feature Ablation') |
| **Table 1 (Dataset Census)** | [x] | N/A | [x] Table_01.csv | [x] Matched | [x] Matched | [x] 4 Road Categories | N/A | [x] 64 trips, 585,643 epochs, 16.27 h | [x] Passed |
| **Table 2 (Partition Split)** | [x] | N/A | [x] Table_02.csv | [x] Matched | [x] Matched | [x] Train/Val/Test | N/A | [x] 39 (12.47h), 6 (0.59h), 19 (3.21h), 579,307 win | [x] Passed |
| **Table 3 (Classical Baselines)** | [x] | N/A | [x] Table_03.csv | [x] Matched | [x] Matched | [x] 10, 20, 30, 60 s Horizons | N/A | [x] 60 s pure: 324.21 m, 89.66%, 0/13 pass | [x] Passed |
| **Table 4 (TCN Scaling)** | [x] | N/A | [x] Table_04.csv | [x] Matched | [x] Matched | [x] 6 vs 39 Trips | N/A | [x] 12.5 h, 448,993 epochs, -0.41 m/s bias | [x] Passed |
| **Table 5 (Kinematic Ablation)** | [x] | N/A | [x] Table_05.csv | [x] Matched | [x] Matched | [x] 6 Representative Variants | N/A | [x] V1: 382.7 m, NIS 6.60; F: 1226.9 m, NIS 82.12 | [x] Passed (Relative Drift to Variant E %, sideslip softened) |
| **Table 6 (Map Matching)** | [x] | N/A | [x] Table_06.csv | [x] Matched | [x] Matched | [x] M0 to M5 Variants | N/A | [x] M0: 578.3 m, M2: 1303.9 m (-125.5%) | [x] Passed (Regression rate operational definition added) |
| **Table 7 (Route Benchmark)** | [x] | N/A | [x] Table_07.csv | [x] Matched | [x] Matched | [x] 4 Categories + Overall | N/A | [x] Matches S1 100% (Overall: 820.2m, 324.21m, 96.65m) | [x] Passed (Category-level summary; 96.65m all 13 vs 16.76% 12 dyn) |
| **Table S1 (Detailed 13 Trips)**| [x] | N/A | [x] Supplementary | [x] Matched | [x] Matched | [x] All 13 60-s Held-Out Routes | N/A | [x] Arithmetic reproduces Table 7 exactly | [x] Passed (Vw15 excluded from macro % per Eq 13) |
