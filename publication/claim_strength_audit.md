# Claim Strength & Tone Verification Audit

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Auditor:** Scientific Integrity & Tone Calibration Specialist  
**Audit Date:** 2026-09-12  

---

## 1. Dangerous Words Scan & Contextual Verification

Every occurrence of high-risk assertoric and superlative vocabulary in the manuscript was audited:

| Keyword | Total Counts | Contextual Inspection | Editorial Classification | Required Action in Restructured Manuscript |
|:---|:---:|:---|:---:|:---|
| `prove` / `proves` / `proven` | 1 | L82: *"While NHC has proven highly effective when paired with rigidly mounted..."* | Low Risk (Literature context) | Softened to: *"While NHC is widely documented to be effective in rigid automotive installations..."* |
| `guarantees` | 0 | Not present in manuscript. | **PASS ✅** | None. |
| `solves` | 0 | Not present in manuscript. | **PASS ✅** | None. |
| `universal` | 2 | L23, L46: Used strictly in negative context: *"universal <10% compliance was NOT achieved."* | **PASS ✅ (Honest Negative)** | Preserved; accurately frames empirical boundaries. |
| `always` / `never` | 0 | Absent from scientific claims. | **PASS ✅** | None. |
| `first` | 2 | L240: *"first-order truncation"* (Math); L369: *"We first establish..."* (Narrative sequence). | **PASS ✅ (Legitimate Mathematical)** | Preserved; no priority or historical claim made. |
| `novel` | 1 | L46: *"In this paper, we refrain from claiming the invention of a novel deep neural architecture..."* | **PASS ✅ (Honest Disclaimer)** | Preserved; explicitly disclaims architectural novelty. |
| `state-of-the-art` | 1 | In literature citation title (Quddus et al., 2007). | **PASS ✅ (Bibliographic)** | Preserved. |
| `optimal` | 1 | L279: *"optimal Kalman gain"* in standard linear quadratic estimation theory. | **PASS ✅ (Standard Mathematical)**| Preserved. |
| `fundamental` | 1 | L52: *"four fundamental failure modes"*. | Mild Risk | Softened to: *"four prominent empirical failure modes"*. |
| `impossible` | 0 | Not present in manuscript. | **PASS ✅** | None. |
| `ceiling` | 0 | Replaced in previous forensic pass with "weak statistical separability". | **PASS ✅** | None. |
| `robust` | 4 | L19, L51, L520: *"no robust generalizable vibration-based speed signal"*; L736: Title of RoNIN. | **PASS ✅ (Negative & Title)** | Preserved. |
| `generalisation` | 7 | Used strictly for trip-disjoint cross-route scaling, with explicit disclaimers regarding single-driver limitation. | **PASS ✅ (Correctly Scoped)** | UK English spelling normalized. |
| `ground truth` | 2 | L153–L154: Used strictly to caution against terminology abuse and clarify that IO-VNBD lacks RTK ground truth. | **PASS ✅ (Methodological Caution)** | Preserved. |

---

## 2. Tone Calibration Rules Applied to Cambridge Submission

1. **Avoid Superlatives**: Refrain from "breakthrough", "superior", "unprecedented", "state-of-the-art".
2. **Emphasize Evidence Wording**: Prefer:
   - *"The empirical evaluations demonstrate..."*
   - *"Under the tested 10-Hz processing rate and feature conditions..."*
   - *"In the evaluated configuration, unconditional constraints degraded..."*
   - *"This observation is consistent with unmodeled tire sideslip and phone misalignment..."*
3. **Preserve Negative Results as Primary Contributions**: Clearly communicate that the failure of unconditional NHC (-45.5%) and closed-loop map matching (-125.5%) are vital empirical findings that warn the navigation community against naive fusion assumptions.
