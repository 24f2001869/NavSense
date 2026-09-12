# Desk-Rejection Pre-Submission Audit: The Journal of Navigation

**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Article Type:** Research Article  
**Author:** Rahul Kumar  
**Audit Date:** September 2026  
**Auditor:** Senior Academic Publication Engineer  

---

## Executive Risk Assessment

* **Overall Pre-Submission Audit Status:** **NO OBVIOUS BLOCKER IDENTIFIED**
* **Findings:** No obvious scope, formatting, or ethical desk-rejection blocker was identified during the pre-submission audit. The manuscript adheres to empirical navigation engineering standards, avoiding promotional marketing and unsupported claims.
* **Primary Strength:** The manuscript presents a disciplined, failure-aware empirical study of sensor fusion failure modes, scaling effects, and deployment realities using public benchmark telemetry.
* **Format Alignment:** Formatted strictly to Cambridge University Press guidelines (12 pt Times New Roman, 1.0 single spacing, 0pt before/after, 2.54 cm margins, single-column submission, UK English, Harvard references, unbolded headings, table captions above, figure captions beneath, 141-word abstract).

---

## 11-Point Desk-Rejection Vulnerability Audit

| Factor | Critical Risk | Current Package State | Audit Finding |
|:---|:---|:---|:---|
| **1. Scope Fit** | Paper is perceived as pure computer science or computer vision with no navigation relevance. | Explicitly addresses terrestrial dead reckoning, 15-state Error-State Kalman Filtering, GNSS outages, coordinate frames, non-holonomic constraints, and road map matching. Aligns directly with *The Journal of Navigation* scope. | Well-aligned with journal scope |
| **2. Contribution Clarity** | Editor cannot discern what the paper actually accomplishes within the first page. | Section 1 contains an explicit 6-point contribution list. Abstract concisely delivers problem, method, scaling results, failure modes, and benchmark pass rates in 141 words. | Clear and verifiable |
| **3. Article Length** | Manuscript exceeds journal limits (>8,000 words or >20 submission pages). | Main text is 6,790 words (comfortably within the 6,000–8,000 range). The actual single-column submission manuscript is 19 pages (both DOCX and PDF), satisfying the $\le 20$ page limit without formatting compression hacks. | Compliant ($\le 20$ pages, 6,790 words) |
| **4. Novelty & Claims** | Exaggerated claims ("first ever", "solves navigation", "universal") trigger immediate editorial hostility. | Strictly conservative terminology enforced. No claims of universal GNSS-denied navigation. Physical road-validation and single-driver limitations prominently declared in abstract, methodology, results, discussion, and limitations sections. | Measured and bounded |
| **5. Empirical Grounding** | Claims lack verifiable numerical support or rely on unreleased private data. | Evaluated on the public IO-VNBD dataset (Onyekpe et al., 2021). All metrics triple-checked against raw CSV logs with 100% numerical verification. | Verified against public benchmark |
| **6. Reference Integrity** | Numbered citations, outdated formatting, missing DOIs, or duplicate entries. | Fully converted to Harvard author-date. Unnumbered alphabetical reference list. Resolved duplicate citation [8]/[11] to Shin et al. (2025). All DOIs verified via CrossRef. | 100% bidirectional Harvard compliance |
| **7. Formatting Compliance** | Generic IEEE layout, double-column submission, bold headings, misplaced captions. | Clean Cambridge layout: single column, 12 pt Times New Roman, 1.0 single spacing, 0pt before/after, 2.54 cm margins, unbolded headings, table captions ABOVE, figure captions BENEATH. | Exact Cambridge style enforced |
| **8. Ethics & Declarations** | Missing Data Availability, Funding, or Competing Interests statements. | All four mandatory statements (Data Availability, Code Availability, Funding, Competing Interests) are complete and present. | Complete |
| **9. AI Use Transparency** | Undisclosed AI writing assistance violates Cambridge publishing ethics policy. | Explicit AI Disclosure Statement included in end matter and methodology, detailing publication-engineering and automated verification activities. | Transparent and consistent |
| **10. Author & ORCID** | Missing institutional affiliation or unauthenticated corresponding author. | Single author Rahul Kumar, University of Hyderabad. IIT Madras BS programme omitted from research affiliation block per policy. ORCID requirement flagged for manual entry in ScholarOne. | Policy-compliant (ORCID flagged for author) |
| **11. Fees & Route** | Inadvertent selection of Gold OA triggers unexpected article processing charges (APC). | Conventional / subscription publication route selected (Gold Open Access not selected). Clear instructions provided for ScholarOne submission. | Documented conventional route |

---

## Actionable Takeaways for Author Before Final Submission

1. **ORCID Authentication:** Rahul must ensure his ORCID is authenticated when logging into ScholarOne Manuscripts.
2. **Gold OA Opt-Out:** When prompted during submission regarding Open Access publishing options, select the conventional / subscription publication option to ensure ₹0 publication cost.
3. **Keyword Dropdowns:** During ScholarOne step 2, select up to 4 keywords matching the recommended concepts: *inertial navigation, dead reckoning, smartphone navigation, GNSS outage*.
