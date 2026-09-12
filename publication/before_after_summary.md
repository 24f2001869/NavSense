# Before & After Publication-Engineering Summary

**Project:** NavSense / SIH26168  
**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Date:** September 2026  
**Auditor:** Senior Academic Publication Engineer  

---

## Metric Comparison Table

| Metric / Dimension | Baseline State (`docs/paper/paper_draft.md`) | Submission Package State (`publication/journal_of_navigation_submission/`) | Delta / Improvement | Rationale & Impact |
|:---|:---:|:---:|:---:|:---|
| **Target Journal Alignment** | Generic IEEE conference/transactions draft | *The Journal of Navigation* (Cambridge University Press) | 100% journal compliance | Transformed structure, style, typography, and citation system. |
| **Language & Spelling** | Mixed American English (`meter`, `generalization`) | Consistent UK English (`metre`, `generalisation`) | Standardized to UK English | Adheres to Cambridge University Press house style. |
| **Main Text Word Count** | ~8,450 words (sprawling draft) | **6,790 words** | -1,660 words (-19.6%) | Comfortably within Cambridge target range (6,000–8,000 words). |
| **Page Count (Layout)** | 42 pages unformatted | **19 pages** (single-column, 12pt Times New Roman, 1.0 single spacing, 0pt before/after, 2.54 cm margins) | -23 pages | Complies with Cambridge $\le 20$ page limit in its actual single-column submission layout. |
| **Abstract Word Count** | 358 words (3 sprawling paragraphs) | **141 words** (single self-contained paragraph) | -217 words (-60.6%) | Fully complies with JoN ~130–150 word guidance ($\le 150$ words). |
| **Manuscript Keywords** | 8 keywords in manuscript body | **0 keywords in body** (transferred to ScholarOne metadata) | Compliant with JoN | JoN policy requires selecting keywords online, not in text. |
| **Citation Style** | IEEE numeric bracketed (`[1]`, `[18]`) | **Harvard Author-Date** (`(Kemp, 1998)`, `(Onyekpe et al., 2021)`) | 100% Harvard conversion | Mandatory Cambridge JoN citation format. |
| **Total References** | 20 entries (with hidden duplicate) | **19 verified entries** (duplicate resolved) | -1 duplicate resolved | Clean, unnumbered alphabetical bibliography with verified DOIs. |
| **Bibliographic Duplicates** | 1 critical duplicate ([8] and [11] identical paper) | **0 duplicates** (merged and re-attributed to Shin et al., 2025) | 1 duplicate eliminated | CrossRef API verified true authors: Shin, Li, and Kim (2025). |
| **Citations Corrected** | 0 audited | **42 in-text citation instances converted** | 100% converted | Complete bidirectional audit (19/19 in-text to biblio, 0 orphans). |
| **Number of Main Figures** | 14 exploratory plots | **8 production figures** (600 DPI TIFF + 300 DPI PNG) | Consolidated to 8 figures | Streamlined main narrative; detailed plots moved to supplement. |
| **Figure Captions Placement** | Misplaced / mixed | **Captions placed BENEATH figures** | Fully compliant | Standard Cambridge JoN artwork style. |
| **Number of Main Tables** | 11 sprawling tables | **7 clean academic tables** (booktabs style) | Consolidated to 7 tables | Table 4 cleaned (no rationale column), Table 7 concise category-level. Full per-trip moved to Table S1. |
| **Table Captions Placement** | Misplaced / mixed | **Captions placed ABOVE tables** | Fully compliant | Standard Cambridge JoN table style. |
| **Supplementary Material** | Fragmented across `docs/` | **Unified formal document** (`Supplementary_Material.docx`, `Supplementary_Material.pdf`, `Supplementary_Material.md`) | Centralized supplement | Formal academic document with Tables S1–S3, Sections S1–S2. Zero Phase 5.2 mentions. |
| **Headings Typography** | Bold, all-caps, inconsistent | **Regular (unbolded) font weight** throughout | Compliant with JoN | Explicit Cambridge JoN instruction against unnecessary bold headings. |
| **Numerical Discrepancies** | 23 initial historical inconsistencies | **0 active numerical discrepancies** | Triple-check passed | 24 exact passes, 6 passes with documented rounding, 0 conflicts. |
| **Contradictions Identified** | Several draft ambiguities | **0 unresolved contradictions** | 15-point audit passed | Single-driver, CAN reference, 3/13 passes strictly consistent. |
| **Ground Reference Framing** | Unqualified "ground truth" | **"CAN bus engineering reference"** | Qualified throughout | Acknowledges lack of dual-frequency carrier-phase RTK GNSS. |
| **Hardware Timing Framing** | Idealized 10.0 Hz assumption | **8.63 Hz OS callback jitter vs ~400 Hz HAL** | Fully documented | Explains dynamic hardware nanosecond timestamping requirement. |
| **Publication Cost Route** | Unspecified / potential APC risk | **Conventional / Subscription route (Gold OA not selected)** | Fully safeguarded | Conventional route incurs zero author-facing fees. |
| **Remaining Manual Actions** | Unclear | **4 clearly identified author actions** | Clear actionable roadmap | Supply ORCID, confirm AI disclosure, select keywords, opt out Gold OA. |
