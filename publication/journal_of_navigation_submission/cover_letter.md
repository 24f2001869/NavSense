# Cover Letter

To:  
The Editor-in-Chief  
*The Journal of Navigation*  
Royal Institute of Navigation / Cambridge University Press  

**Date:** 12 September 2026  

**Subject:** Submission of Research Article: "Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift"

Dear Editor-in-Chief,

I am pleased to submit the above-titled original Research Article for consideration for publication in *The Journal of Navigation*.

### Fit and Scope
Vehicular positioning in GNSS-denied environments—such as subterranean corridors, road tunnels, and dense urban canyons—remains a critical navigation challenge. While tactical-grade inertial systems assisted by vehicle CAN-bus wheel encoders are well established, widespread consumer navigation applications rely on standalone smartphones lacking physical wheel odometry. Low-cost smartphone MEMS sensors exhibit severe thermal and bias instability, resulting in rapid nonlinear position divergence that exceeds 300 metres within 60 seconds of satellite loss. While recent sequence learning approaches have demonstrated promising preliminary results, existing studies often evaluate brief outages (5–15 s) or fail to examine operational failure modes when kinematic assumptions break down.

### Summary of Scientific Findings
This manuscript presents a systematic, failure-aware empirical study of smartphone inertial dead reckoning under sustained GNSS outages (10 to 60 seconds). Operating on the public IO-VNBD benchmark dataset (64 passenger-car trips, 16.27 hours of synchronized driving, 902.4 km) and native Android edge execution, the study reports the following key findings:
1. **Multi-Trip Scaling:** Scaling causal Temporal Convolutional Network (TCN) training from 6 to 39 whole trips across 19 strictly held-out test routes (240.6 km, 115,420 synchronized epochs) reduces velocity MAE by 55.27% (6.17 to 2.76 m/s) and decreases 60-second position drift by 64.72% (263.6 to 93.0 m).
2. **Kinematic Constraint Breakdown & Decoupled Recovery:** We demonstrate that standard unconditional lateral Non-Holonomic Constraints (NHC) degrade 60-second drift by 45.47% due to phone mounting misalignment and cornering sideslip angle coupling into attitude error states. We evaluate a decoupled lateral damping formulation ($K_y[6:15] = 0$) that restricts lateral innovations strictly to velocity decay, improving drift by 54.62% (382.7 m).
3. **Map-Feedback Failure:** In the tested implementation, downstream/open-loop map visualisation was safer than direct closed-loop heading feedback during outages, where candidate link ambiguity during turns caused 125.5% drift degradation (578.3 to 1303.9 m).
4. **Boundary and Distribution-Shift Realities:** Across >450,000 causal windows, spectral vibration features at 10 Hz show near-zero correlation with vehicle speed ($r = -0.032, \rho = -0.028$), consistent with low-frequency vehicle-body/chassis dynamics. Steady-state highway cruising exhibits weak statistical separability (ROC-AUC = 0.625) and negative regression bias ($-3.5\text{ m/s}$), while pedestrian arm swing induces extreme out-of-distribution velocity spikes (71.66 m/s), mitigated to 27.71 m/s via channel ablation.
5. **On-Device Edge Viability:** Native Android execution achieves mean epoch latency of 9.15 ms (4.2–7.8 ms ONNX inference, 0.8–1.4 ms 15-state ESKF, 99th percentile 13.89 ms) and confirms that hardware nanosecond timestamps were used to avoid time-scale distortion arising from mobile OS callback jitter (mean interval 115.8 ms, effective callback rate 8.63 Hz versus 10.0 Hz nominal).
6. **Benchmark Reality Check:** Across 13 usable 60-second blackout test routes, adaptive fusion achieves 96.65 m mean drift (16.76% normalized error) and satisfies the 10% drift benchmark on 3 of 13 routes (23.08%), highlighting both the measured improvements and the remaining empirical limitations of standalone smartphone navigation.

### Declarations
* This manuscript is original work and is not under consideration for publication elsewhere.
* The author declares no competing financial or non-financial interests.
* No external funding was received for this research.
* The author discloses the use of the Google Antigravity Agentic Assistant, powered by Gemini 2.5 Flash, for publication-engineering, language harmonisation, and automated consistency auditing; all scientific content, experimental analyses, and final manuscript decisions were independently verified by the author.
* The author has reviewed the manuscript, approves its submission to *The Journal of Navigation*, and takes full responsibility for the integrity of the work.

Thank you for your time and consideration of this work.

Sincerely,

**Rahul Kumar**  
Integrated M.Tech. (Materials Engineering)  
School of Engineering Sciences & Technology  
University of Hyderabad, Hyderabad 500046, India  
Email: `24f2001869@ds.study.iitm.ac.in`  
