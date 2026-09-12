# Hostile Pre-Submission Peer Review Simulation: NavSense / SIH26168

**Target Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Role:** Adversarial Pre-Submission Review Committee  
**Date:** September 2026  
**Auditor:** Senior Academic Publication Reviewer  

---

## 1. Reviewer A — Navigation Systems Specialist

### Assessment & Attacks:
1. *"Does this manuscript represent real navigation science, or is it merely an applied machine learning project dressed up in navigation vocabulary?"*
   - **Evaluation:** The manuscript is fundamentally grounded in classical navigation theory. It implements a 15-state Error-State Kalman Filter with quaternion attitude kinematics, addresses non-holonomic constraints (NHC), analyzes vehicle sideslip angle coupling, and evaluates digital road network map matching.
   - **Severity:** NO ACTION (Fully defensible).
2. *"Is the evaluation horizon realistic for land vehicle navigation?"*
   - **Evaluation:** Many ML papers test only 5–10 second windows. NavSense evaluates 10 s, 20 s, 30 s, and 60 s continuous outages. 60 seconds is an industry standard for tunnel transit.
   - **Severity:** NO ACTION (Strong point of the paper).
3. *"Why is CAN bus velocity called ground truth when tyre slip and inflation changes introduce speed errors?"*
   - **Audit Check:** The manuscript explicitly downgraded CAN velocity from "ground truth" to "engineering reference" throughout Section 2.1, 4.1, and 6. It highlights the absence of dual-frequency carrier-phase RTK GNSS.
   - **Severity:** NO ACTION (Addressed via transparent qualification).

---

## 2. Reviewer B — Machine Learning & Deep Learning Specialist

### Assessment & Attacks:
1. *"Could there be temporal leakage between training and test sets?"*
   - **Evaluation:** Partitions are assigned strictly at the whole-trip level (39 train, 6 validation, 19 held-out test). The test script `tests/test_causality_and_leakage.py` proves zero future gradient leakage and zero trip overlap.
   - **Severity:** NO ACTION (Formally certified).
2. *"Are the scaling conclusions statistically sound given that all 64 trips come from a single driver?"*
   - **Audit Check:** The paper explicitly acknowledges this critical boundary in the abstract, methodology, results, discussion, and limitations: *"100% of trips are Driver E; cross-driver generalisation cannot be claimed."*
   - **Severity:** MODERATE $\to$ NO ACTION (Transparently disclosed; no false claims made).
3. *"Why did the pedestrian test explode to 71.66 m/s, and does this mean the network is unstable?"*
   - **Evaluation:** The paper conducts an exact feature attribution analysis, proving that human arm swings produce yaw rates of 3.81 rad/s (+26.08$\sigma$ above vehicle training bounds). The paper presents this as an empirical stress test demonstrating the necessity of domain gating, not as a successful deployment.
   - **Severity:** NO ACTION (Negative result framed as a scientific finding).

---

## 3. Reviewer C — Classical Inertial Navigation & Filter Specialist

### Assessment & Attacks:
1. *"How are coordinate frames and attitude representations handled?"*
   - **Evaluation:** Section 2.3 and 2.4 define the device body frame ($b$), leveled horizontal frame ($l$), and local East-North-Up frame ($n$). Quaternion attitude propagation ($\mathbf{q}_b^n$) and small attitude error states ($\delta \boldsymbol{\theta}$) are mathematically derived following Solà (2017).
   - **Severity:** NO ACTION (Mathematically sound).
2. *"Why does standard lateral NHC fail so dramatically (+45.47% drift increase)?"*
   - **Evaluation:** Section 4.4 and 5.2 explain that when a smartphone is mounted with an uncalibrated orientation, or when the vehicle undergoes cornering with non-zero tyre sideslip ($\beta \neq 0$), forcing $v_y^b \approx 0$ through coupled Kalman gain blocks injects lateral innovations directly into the attitude and gyro bias states, causing heading divergence. Decoupled velocity damping ($K_y[6:15] = 0$) isolates the update to velocity states, recovering performance.
   - **Severity:** NO ACTION (Physically justified and backed by ablation evidence).
3. *"Is the 63-step receptive field causality mathematically verified?"*
   - **Evaluation:** Section 2.5 and Supplementary Material Section S1 derive the causal receptive field reach: $R = 1 + \sum_{l=0}^3 (3 - 1) \cdot 2^l = 1 + 2(1 + 2 + 4 + 8) = 31$ steps per conv stack, expanding to 63 steps across the 4-block architecture (6.3 s at 10 Hz). No future samples are accessible.
   - **Severity:** NO ACTION (Mathematically verified).

---

## 4. Reviewer D — Reproducibility & Open Science Auditor

### Assessment & Attacks:
1. *"Can an independent researcher download the code and verify the results?"*
   - **Evaluation:** `publication/code_reproducibility_audit.md` verified that the MIT-licensed GitHub repository contains modular source code (`src/`), standalone benchmark reproduction scripts (`scripts/experiments/`), pre-trained ONNX models (`models/`), and full evaluation logs (`results/`).
   - **Severity:** NO ACTION (Verified).
2. *"Is the raw dataset included in the repository?"*
   - **Evaluation:** Raw IO-VNBD data (2.34 GB) is hosted on the official GitHub/Data in Brief repository by Onyekpe et al. (2021). The repo provides detailed download and folder setup instructions in `data/README.md`.
   - **Severity:** MINOR $\to$ NO ACTION (Standard for multi-gigabyte open benchmark datasets).
3. *"Are all mathematical symbols and acronyms expanded?"*
   - **Evaluation:** All acronyms (GNSS, INS, ESKF, TCN, NHC, ZVD, ZUPT, OOD, SIH) are expanded on first use in the main text. SI units are standardized throughout.
   - **Severity:** NO ACTION (Clean).

---

## 5. Reviewer E — Hostile Editorial Board Member

### Assessment & Attacks:
1. *"Does this fit the scope of The Journal of Navigation?"*
   - **Evaluation:** Yes. *The Journal of Navigation* prioritizes practical, terrestrial, vehicular, and marine navigation systems under GNSS vulnerabilities, sensor fusion, Kalman filtering, and dead reckoning.
   - **Severity:** NO ACTION (Strong fit).
2. *"Is the paper too long or overclaiming?"*
   - **Evaluation:** The main body is 6,790 words (within the 6,000–8,000 word target), formatted in 19 pages single-column Word layout (satisfying the $\le 20$ submission pages limit in its actual single-column format). The abstract is exactly 141 words. No overclaiming: the paper openly admits that the SIH <10% benchmark is achieved on only 3 of 13 routes (23.08%).
   - **Severity:** NO ACTION (Disciplined).
3. *"Are the references compliant with Cambridge Harvard style?"*
   - **Evaluation:** All 19 references are converted to author-date in text and alphabetized without numbers in the bibliography. The duplicate citation around *Applied Sciences* 2025 was forensically merged and attributed to Shin et al. (2025).
   - **Severity:** NO ACTION (Verified).

---

## Summary Verdict of the Simulated Review

| Reviewer | Initial Skepticism | Resolution in Package | Residual Vulnerability | Final Recommendation |
|:---|:---|:---|:---|:---:|
| **Reviewer A (Navigation)** | Applied ML dressed up | Rigorous ESKF, coordinate frames, CAN reference | None | **Accept with minor revisions** |
| **Reviewer B (ML/DL)** | Data leakage & single driver | Strict trip-disjoint split; single driver disclosed | None | **Accept** |
| **Reviewer C (Inertial/Filter)** | Constraint failure mechanism | Decoupled damping formulation; sideslip analysis | None | **Accept** |
| **Reviewer D (Reproducibility)** | Code and dataset availability | Public IO-VNBD citations; open-source GitHub repo | None | **Accept** |
| **Reviewer E (Editor)** | Scope, length, overclaiming | JoN formatting, 141-word abstract, honest limits | None | **Accept** |

**Consensus Recommendation:** The manuscript is robustly shielded against hostile technical attacks by virtue of its conservative framing, transparent negative results, and mathematical consistency.
