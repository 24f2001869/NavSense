# ScholarOne Manuscripts Submission Metadata: NavSense / SIH26168

**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Submission Portal:** ScholarOne Manuscripts (`https://mc.manuscriptcentral.com/cup/nav`)  
**Intended Publication Route:** Conventional / Subscription publication; Gold Open Access not selected (Author will not voluntarily incur Gold OA APC or optional paid services)  
**Date:** September 2026  
**Author:** Rahul Kumar  

---

## 1. Article Identification

* **Article Type:** Research Article
* **Full Title:** Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift
* **Short Running Title ($\le 40$ characters):** Failure-Aware Smartphone Dead Reckoning
* **Manuscript Word Count (Main Body):** 7,247 words (official Microsoft Word COM count; within Cambridge 6,000–8,000 range)
* **Actual Submission Page Count:** 20 pages (single-column, 12 pt Times New Roman, single spacing, 2.54 cm margins, compliant with $\le 20$ page limit)
* **Intended Publication Route:** Conventional / Subscription publication; Gold Open Access not selected. No Gold-OA APC is incurred when publishing through the conventional route; optional paid services and print-colour charges are not selected.

---

## 2. Author & Institution Details

* **Corresponding Author:** Rahul Kumar
* **Email Address:** `24f2001869@ds.study.iitm.ac.in`
* **Primary Research Affiliation:**  
  Integrated M.Tech. (Materials Engineering)  
  School of Engineering Sciences & Technology  
  University of Hyderabad  
  Prof. C.R. Rao Road, Gachibowli, Hyderabad 500046, Telangana, India
* **Additional Academic Programme Note:**  
  The author is concurrently enrolled in the BS Degree in Data Science and Applications (Diploma Level) at the Indian Institute of Technology Madras (IIT Madras). The formal affiliation reflects the primary academic institution associated with the research presented in this manuscript (University of Hyderabad).
* **ORCID ID:**  
  `[REQUIRED — AUTHOR MUST SUPPLY DURING SCHOLARONE SUBMISSION]`  
  *(Cambridge University Press mandates an authenticated ORCID for corresponding authors. See `publication/ORCID_REQUIRED.md` for instructions).*

---

## 3. Abstract

*(Exact word count: 148 words. Formatted as a single, self-contained paragraph without citations or unexplained acronyms).*

> During Global Navigation Satellite System (GNSS) outages, smartphone inertial dead reckoning suffers from rapid nonlinear open-loop position drift. We present an empirical study evaluating learned forward velocity estimation, sensor fusion, kinematic constraints, map feedback, and distribution shifts using the public IO-VNBD benchmark dataset (64 passenger-car trips, 16.27 hours). On 19 trip-disjoint test routes, scaling a causal temporal convolutional network from 6 to 39 training trips reduces velocity error by 55.27% (6.17 to 2.76 m/s) and 60-second drift by 64.72% (263.6 to 93.0 m). Unconditional lateral kinematic constraints degrade 60-second drift by 45.47%, and closed-loop map heading feedback degrades drift by 125.5%. Across 13 usable 60-second blackout trajectories, adaptive fusion achieves 96.65 m mean absolute drift and 16.76% macro-average normalized drift across 12 dynamic routes (excluding stationary control Vw15), satisfying a sub-10% drift benchmark on 3 of 13 routes (23.08%). Physical road validation and multi-driver generalisation remain open challenges.

---

## 4. Keywords & Subject Classification

*Note: Keywords are selected online within ScholarOne during submission and are NOT included in the manuscript text, in accordance with Cambridge instructions (maximum 4 keywords):*

1. **Inertial navigation**
2. **Dead reckoning**
3. **Smartphone navigation**
4. **GNSS outage**

*(Alternate terms if restricted by taxonomy: Sensor fusion, Kalman filtering, Sequence learning).*

---

## 5. Mandatory Declarations & Ethics Statements

### 5.1 Funding Statement
> "No specific external funding was received for this research."

### 5.2 Competing Interests Declaration
> "The author declares no competing financial or non-financial interests."

### 5.3 Data Availability Statement
> "The empirical evaluations in this study were conducted on the publicly available IO-VNBD benchmark dataset (Onyekpe et al., 2021), accessible under the Creative Commons Attribution 4.0 International licence at https://doi.org/10.1016/j.dib.2021.106885. Derived summary metrics and evaluation logs generated during this research are available within the project repository."

### 5.4 Code Availability Statement
> "The software code supporting the analyses, models, filtering pipelines, and Android deployment in this study is available under an open-source MIT licence at https://github.com/24f2001869/NavSense (mirror: https://github.com/24f2001869/SIH26168-IDR), specifically preserved under the frozen publication release tag paper-jon-v1.0."

### 5.5 AI-Assisted Technologies Disclosure
> "During the preparation and forensic auditing of this manuscript, the author utilized the Google Antigravity Agentic Assistant (Google DeepMind, powered by Gemini 2.5 Flash, accessed September 2026 via https://deepmind.google) for publication engineering. Automated verification scripts were executed in the Google Antigravity development environment to cross-check numerical tables, causality, and future-gradient leakage; the underlying experimental research, physical and mathematical formulations, models, filtering algorithms, and scientific findings were established by the author; all automated numerical verifications were independently verified against repository logs and experimental outputs. Assistance encompassed: manuscript restructuring into Journal of Navigation submission format; language and UK English editorial harmonization; auditing and resolving bibliographic citations against CrossRef metadata; repository code and telemetry inspection; and unit/cross-reference consistency auditing. No AI-generated scientific claim was accepted without independent verification. The author reviewed and approved all submitted text and retains sole responsibility for the contents of this manuscript."

---

## 6. Suggested Reviewers Guidance

*In strict compliance with scientific integrity rules, this system does NOT fabricate reviewer names, fake email addresses, or unverified affiliations. If ScholarOne requests suggested reviewers, the author may consider nominating recognized researchers in smartphone vehicular dead reckoning and inertial navigation who have no conflict of interest (e.g. researchers who have published on smartphone IMU velocity estimation or vehicular dead reckoning).*

**Candidate Domains for Author Consideration:**
* Researchers working on vehicular inertial navigation and non-holonomic constraints.
* Authors investigating deep sequence learning for dead reckoning using smartphone IMUs.
* Experts in Kalman filtering and terrestrial navigation under GNSS-denied environments.
