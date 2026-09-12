# Artificial Intelligence (AI) Disclosure & Publishing Ethics Review

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Policy Reference:** Cambridge University Press Research Publishing Ethics Guidelines on AI Tools  
**Auditor:** Academic Integrity & Compliance Specialist  
**Review Date:** 2026-09-12  

---

## 1. Cambridge University Press Official AI Policy Summary

Under Cambridge University Press publishing ethics policies:
1. **Authorship**: Artificial intelligence tools do not meet the legal or ethical accountability requirements of authorship. AI tools (e.g., ChatGPT, Claude, Gemini, Antigravity) **must never be listed as an author or co-author** on any submitted manuscript.
2. **Accountability**: The human author remains fully responsible for the originality, scientific validity, evidentiary integrity, reference authenticity, and factual accuracy of all submitted content.
3. **Mandatory Transparency**: Authors must explicitly declare any use of AI tools in the research design, code analysis, data parsing, literature search, or manuscript drafting process.
4. **Specific Disclosure Information**: The disclosure must specify:
   * The name and provider of the AI tool.
   * The model version.
   * The specific scope and nature of assistance (e.g., code verification, formatting restructuring, language editing).
   * A formal confirmation that all empirical numbers and technical conclusions were independently verified by the human author.

---

## 2. Assessment of AI Assistance in NavSense Preparation

During the preparation and auditing of this manuscript package:
* **Research Implementation & Experimental Data**: All neural network architectures (TCN-kin), Kalman filter C++/Java/Python implementations, Android native edge code, empirical evaluations on IO-VNBD, and test verification scripts were engineered and executed in the project workspace.
* **AI Tool Assistance**: Antigravity (powered by Google Gemini models) was utilized as an agentic software engineering and editorial audit assistant to:
  1. Perform automated forensic verification of experimental logs and numerical tables.
  2. Execute mathematical causality and data leakage tests (`test_causality_and_leakage.py`).
  3. Restructure manuscript sections and convert reference citations from IEEE numbered to Harvard author-date style.
  4. Perform language editing and verify compliance with Cambridge University Press author instructions.

---

## 3. Proposed Formal AI Declaration for Manuscript

The following statement is prepared for inclusion in the Declarations section of the main manuscript file (`01_JON_manuscript.docx`):

> **Declaration of Generative AI and AI-Assisted Technologies in the Writing Process:**  
> During the preparation of this work, the author utilized the Antigravity development environment (powered by Google Gemini large language models) for automated code and log audit verification, reference cross-checking, and editorial formatting assistance to comply with the journal's Harvard citation and single-column style guidelines. Following these automated verification and formatting steps, the author reviewed and edited the content thoroughly and takes full responsibility for the scientific integrity, numerical accuracy, and factual content of the published article.

---

## 4. Author Action Checklist (Manual Confirmation Required by Rahul)

Before final submission in ScholarOne, the author (Rahul Kumar) must:
- [ ] Confirm that the AI declaration accurately reflects the scope of AI assistance used.
- [ ] Verify that no AI-generated speculative text or ungrounded claims remain in the manuscript.
- [ ] Reconfirm that all empirical data originated strictly from the IO-VNBD dataset and repository logs.
