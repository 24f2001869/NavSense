# Final Publication-Engineering Audit Report: NavSense / SIH26168

**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Article Type:** Research Article  
**Article Title:** Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift  
**Author:** Rahul Kumar  
**Affiliation:** Integrated M.Tech. (Materials Engineering), School of Engineering Sciences & Technology, University of Hyderabad, Hyderabad, India  
**Date of Forensic Audit:** 12 September 2026  
**Auditor:** Lead Academic Publication Engineer & Forensic Scientific Reviewer  

---

## EXECUTIVE VERDICT: READY FOR AUTHOR FINAL REVIEW

The submission package for *The Journal of Navigation* has successfully completed an exhaustive forensic correction and journal-compliance pass. All previous formatting compromises (1.15 line spacing, 4 pt paragraph spacing, hypothetical double-column conversion arguments) have been **completely eliminated**. 

The actual single-column submission manuscript (`01_JON_manuscript.docx` and `02_JON_manuscript.pdf`) has been independently audited using Microsoft Word COM automation and PyMuPDF under the exact controlling Cambridge University Press author instructions:
- **Actual Word Page Count:** **19 pages** (strictly compliant with the Cambridge $\le 20$ page limit in its actual submission format)
- **Actual Word Word Count:** **6,790 words** (comfortably within the Cambridge 6,000–8,000 word range)
- **Scientific Freeze:** **100% FROZEN** (zero new experiments, zero new models, zero Phase 5.2 contamination)
- **Cost Safeguard:** Conventional / Subscription publication route specified (Gold Open Access not selected; zero author-facing fees)

No unsupported certainty statements ("guaranteed", "<5% desk rejection", "unconditional acceptance") remain in any file.

---

## 15-Point Forensic Specification Audit (Prompt Section 31)

| # | Audit Item | Verified Package Value | Controlling Cambridge Rule / Standard | Forensic Compliance Status |
|:---:|:---|:---:|:---|:---:|
| **1** | **Actual Word Page Count** | **19 pages** | Up to 20 pages including figures & tables | **FULL COMPLIANCE** (Audited via Word COM) |
| **2** | **Actual PDF Page Count** | **19 pages** | Matches Word document page layout | **FULL COMPLIANCE** (Audited via PyMuPDF) |
| **3** | **Actual Main-Text Word Count** | **6,790 words** | 6,000–8,000 words target range | **FULL COMPLIANCE** (Word COM Statistics) |
| **4** | **Abstract Word Count** | **141 words** | Approximately 150 words ($\le 150$), 1 paragraph | **FULL COMPLIANCE** (Single paragraph, 0 citations) |
| **5** | **Actual Spacing Settings** | **Line: 1.0 (Single)<br>Before: 0 pt<br>After: 0 pt** | Single line spacing, 0 pt before, 0 pt after, 2.54 cm margins, 12 pt Times New Roman, no indents, fully justified | **FULL COMPLIANCE** (Audited across all 159 paragraphs: 0 non-single, 0 non-zero before/after) |
| **6** | **Number of Figures** | **8 figures** | Embedded in text; captions placed beneath; TIFF/PNG production quality | **FULL COMPLIANCE** (8 × 600 DPI TIFF + 8 × 300 DPI PNG) |
| **7** | **Number of Tables** | **7 tables** | Embedded in text; captions placed above; booktabs styling | **FULL COMPLIANCE** (Table 4 cleaned; Table 7 concise category-level) |
| **8** | **Supplementary Items** | **3 Tables (S1–S3)<br>2 Sections (S1–S2)<br>3 Formats (DOCX, PDF, MD)** | Clean, formal reader-facing document; published uncopyedited as supplied | **FULL COMPLIANCE** (4 pages DOCX/PDF; zero internal audit language) |
| **9** | **Reference Count** | **19 references** | Harvard author-date, unnumbered, alphabetical, DOIs verified | **FULL COMPLIANCE** (100% bidirectional match; duplicate resolved) |
| **10** | **Numerical Audit Status** | **30 / 30 verified** | Exact empirical grounding in repository logs | **FULL COMPLIANCE** (24 exact, 6 rounding, 0 conflicts) |
| **11** | **Scientific Freeze Status** | **FROZEN** | Zero new experiments, splits, models, or claims | **FULL COMPLIANCE** (Single driver, CAN reference, no RTK) |
| **12** | **Phase 5.2 Contamination** | **0 MATCHES** | Absolute exclusion of target-balanced TCN and inverse-density loss weighting | **FULL COMPLIANCE** (Exhaustive search across submission dir: 0 matches) |
| **13** | **AI Disclosure Status** | **CONSISTENT** | Declare AI text generation and data/testing analysis in manuscript & methods | **FULL COMPLIANCE** (Identifies Antigravity/DeepMind; details all activities) |
| **14** | **ORCID Status** | **ACTION REQUIRED** | Mandatory for corresponding author in ScholarOne | **MANUAL AUTHOR ACTION FLAGGED** |
| **15** | **Remaining Author Actions** | **4 actions** | Final portal inputs during submission | **ACTIONABLE CHECKLIST PROVIDED** |

---

## Detailed Audit Summaries

### 1. Document Typography and Layout Audit
The Word document `01_JON_manuscript.docx` was built from the ground up using exact typography parameters:
- **Font Family:** Times New Roman throughout (Normal style, Headings, Captions, Tables, References).
- **Font Size:** Exactly 12 pt for body text, headings, author info, abstract, and references; 10 pt italic for figure captions; 11 pt for table captions; 8.5–9 pt for table cell data.
- **Line Spacing:** 1.0 (Single line spacing) enforced across 100% of paragraphs.
- **Paragraph Spacing:** Space Before = 0 pt, Space After = 0 pt across 100% of paragraphs.
- **Margins:** Exactly 2.54 cm (1.0 inch) on Top, Bottom, Left, and Right.
- **Column Format:** Single-column layout throughout. No hypothetical double-column conversion claims.
- **Section Heading Discipline:** Only true top-level section headings (`1. Introduction` through `7. Conclusions` and back-matter statements) receive `keep_with_next = True`. Numbered list items (contributions, limitations) are treated as regular body paragraphs, preventing artificial multi-inch page breaks.
- **Resulting Page Length:** Exactly **19 pages in Microsoft Word** and **19 pages in the exported PDF**.

### 2. Table and Figure Cleanup
- **Table 4:** The redundant "Scientific Rationale" column was eliminated. The table now cleanly contrasts the 6-trip model, 39-trip model, and relative performance delta, significantly improving visual density and page flow.
- **Table 7:** Replaced the sprawling 13-row route table in the main paper with a concise 4-row category-level summary (Suburban Town, Dense Urban, Winding Rural, Overall Macro-Average). The complete route-by-route breakdown (with reference distances, raw strapdown drift, and dynamic route characteristics) is cleanly preserved in **Supplementary Table S1**.
- **Dramatic Wording Purge:** Emotional descriptors such as "CATASTROPHIC open-loop cubic runaway" and "DEGRADATION" have been replaced with objective engineering terminology ("Severe open-loop drift growth", "Longitudinal scale compression").
- **Production Artwork:** In addition to high-resolution 300 DPI PNG files, all 8 figures have been exported as **600 DPI TIFF images with LZW lossless compression** (`Figure_01.tif` to `Figure_08.tif`) in accordance with Cambridge University Press artwork specifications.

### 3. Scientific Freeze and Phase 5.2 Purge
An automated forensic search for historical Phase 5.2 exploratory artifacts was executed across every text, markdown, CSV, and docx file in `publication/journal_of_navigation_submission/`:
- `Phase 5.2` / `phase 5.2`: **0 matches**
- `target-balanced` / `Target-Balanced`: **0 matches**
- `balanced TCN` / `Balanced TCN`: **0 matches**
- `inverse-density` / `Inverse-Density`: **0 matches**
- `speed-stratified` / `Speed-Stratified`: **0 matches**
- `loss_weighting`: **0 matches**
- `speed_stratified_sampler`: **0 matches**
A dedicated ledger document, `publication/final_scientific_freeze_audit.md`, certifies this complete purge.

### 4. Scientific Language and Boundary Softening
All causal and universal statements flagged during review were rigorously softened:
- **Chassis Vibration:** Replaced universal claims with bounded empirical statements: *"Under the evaluated 10 Hz feature construction, no reliable speed dependence was identified in the dominant vibration frequency ($r = -0.032$), consistent with low-frequency vehicle-body/chassis dynamics."*
- **Pedestrian Out-of-Distribution Dynamics:** Softened causal attribution: *"indicating that extreme rotational rates were a major contributor to the observed velocity spike (71.66 m/s)."*
- **Mobile Execution:** Scoped real-time phrasing: *"Total measured computational latency remained below 10.0 ms per 100 ms epoch, remaining within the nominal 10 Hz processing budget."*
- **Highway Cruising:** Scoped observability limits: *"weak statistical separability under the tested steady-state highway conditions (ROC-AUC = 0.625)."*
- **Ground Reference:** CAN bus forward velocity is strictly identified as an *engineering reference* subject to tyre slip and radius variation; the manuscript explicitly clarifies that IO-VNBD does not provide centimetre-level dual-frequency RTK GNSS ground truth.
- **Single Driver Boundary:** The manuscript prominently and repeatedly discloses that all 64 passenger-car trips were conducted by a single driver (Driver E); cross-driver generalisation is explicitly disclaimed.
- **Hardware vs. Software Timing:** The distinction between IMU hardware sampling (~400 Hz HAL) and Android navigation callback dispatch (~8.63 Hz jitter) is consistently maintained.

### 5. AI Disclosure Consistency
The AI disclosure statement has been fully harmonized across all package files (`01_JON_manuscript.md`, `01_JON_manuscript.docx`, `scholarone_metadata.md`, and Section 3.7 of the Methods):
- **Identity:** Google Antigravity Agentic Assistant (Google DeepMind).
- **Scope of Activities:** Manuscript restructuring into Cambridge submission format; language and UK English editorial harmonization; auditing and resolving bibliographic citations against CrossRef metadata; repository code and telemetry inspection; automated execution of numerical verification scripts, gradient leakage tests, and causality checks; and unit/cross-reference consistency auditing.
- **Integrity Statement:** Explicitly affirms that underlying experimental research, physical models, and scientific findings were established by the author; all automated numerical checks were independently verified against repository logs; no AI-generated scientific claim was accepted without verification; and the human author retains sole responsibility for the work.

### 6. Supplementary Material Document
`publication/journal_of_navigation_submission/supplementary/Supplementary_Material.md` has been compiled into standalone publication documents:
- `Supplementary_Material.docx` (4 pages, 1,294 words, 12 pt Times New Roman, single spacing, booktabs tables)
- `Supplementary_Material.pdf` (4 pages, generated via Word COM automation)
It contains zero internal audit jargon, zero development notes, zero Antigravity references, and zero "TODO" markers. It is completely ready to be published uncopyedited alongside the main article.

### 7. Zero-Cost Publication Route
The submission strategy adheres strictly to the conventional / subscription publishing model:
- **Publishing Option:** Conventional / Subscription publication (Gold Open Access NOT selected).
- **Author Fee:** Zero author-facing Article Processing Charges (APCs).
- **Colour Printing:** Free online colour confirmed; no optional paid print colour requested.
- **Commercial Add-Ons:** No paid language editing or fast-track services selected.

---

## Final Submission Package File Inventory

The primary submission package is organized in `publication/journal_of_navigation_submission/`:

```
publication/journal_of_navigation_submission/
├── 01_JON_manuscript.docx                 [Primary submission file: 19 pages, 6,790 words, 12pt TNR, single-spaced]
├── 02_JON_manuscript.pdf                  [Review companion PDF: exactly 19 pages]
├── cover_letter.md                        [Formal submission cover letter to Editor-in-Chief]
├── scholarone_metadata.md                 [Complete copy-paste metadata fields for ScholarOne portal]
├── FINAL_SUBMISSION_CHECKLIST.md          [Pre-submission sign-off checklist]
├── CHANGELOG.md                           [Audited transformation changelog]
├── figures/
│   ├── Figure_01.tif to Figure_08.tif     [8 × Production TIFF artwork, 600 DPI, LZW compressed]
│   └── Figure_01.png to Figure_08.png     [8 × High-resolution PNG artwork, 300 DPI]
├── tables/
│   ├── Table_01.csv to Table_07.csv       [7 × Clean CSV tables; Table 4 simplified, Table 7 category-level]
└── supplementary/
    ├── Supplementary_Material.docx        [Reader-facing Supplementary Material document, 4 pages]
    ├── Supplementary_Material.pdf         [Companion PDF of Supplementary Material, 4 pages]
    └── Supplementary_Material.md          [Markdown source with Tables S1–S3, Sections S1–S2]
```

Internal forensic audit files remain strictly separated in `publication/`:
- `publication/final_scientific_freeze_audit.md` (Phase 5.2 zero-match ledger)
- `publication/numerical_triple_check.md` (30-metric verification ledger)
- `publication/reference_audit.md` (19-reference CrossRef audit ledger)
- `publication/claim_ledger_final.md` (18-claim evidence traceability ledger)
- `publication/desk_rejection_risk.md` (Qualitative desk-rejection pre-audit)
- `publication/hostile_peer_review_final.md` (Adversarial simulated peer review)
- `publication/zero_cost_publication_strategy.md` (Conventional route policy breakdown)
- `publication/accessibility_descriptions.md` (WCAG 2.1 Level AA alt-text)
- `publication/before_after_summary.md` (Metric comparison ledger)
- `publication/ORCID_REQUIRED.md` (Author ORCID setup guidance)

---

## Remaining Author Actions (Pre-Submission Checklist)

Prior to clicking "Submit" in ScholarOne Manuscripts (`https://mc.manuscriptcentral.com/cup/nav`), the author must complete these four straightforward actions:

1. **Supply & Authenticate ORCID:** Log in to ScholarOne using your personal ORCID credentials or authorize your ORCID identifier when prompted.
2. **Select Conventional / Subscription Route:** When prompted on the publishing choice screen, select the **conventional subscription model**. Do **NOT** select Gold Open Access, ensuring no Article Processing Charges (APCs) are incurred.
3. **Select Online Keywords:** In the ScholarOne metadata step, select up to 4 keywords from the journal's taxonomy matching: *Inertial navigation, Dead reckoning, Smartphone navigation, GNSS outage*. (Keywords are omitted from the manuscript body per Cambridge style).
4. **Visual PDF Inspection:** Download and inspect the merged PDF automatically generated by ScholarOne to confirm that all 8 embedded figures and 7 tables render cleanly.

**Conclusion:** The NavSense / SIH26168 submission package is fully compliant with Cambridge University Press / *The Journal of Navigation* requirements and is genuinely ready for the author's final manual inspection and formal submission.
