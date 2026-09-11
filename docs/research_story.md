# The Research Story: From Black Boxes to Physical Truth

*A retrospective narrative on the engineering journey of the SIH26168 Intelligent Dead Reckoning project.*

---

## 1. The Naive Beginning: "Just Use AI"
Like many teams tackling the Smart India Hackathon challenge of GNSS-denied vehicle navigation, our starting premise seemed simple:
> *"Consumer smartphone IMUs drift when integrated. Modern deep neural networks can learn complex non-linear mappings. Therefore, let's train an AI model to predict vehicle velocity from raw smartphone sensor streams."*

The early experiments seemed promising. We trained small machine learning models on single vehicle trips. Within those single trips, the models reproduced speed profiles with low error. We exported an Android app, packaged the model, and assumed the problem was largely solved.

Then we began rigorous out-of-sample testing—and reality set in.

---

## 2. The First Wall: The Cross-Trip Generalization Collapse
When our initial tree-based models (Random Forests, GBDTs) were evaluated on trips driven by different motorists or in different vehicles, performance collapsed completely. Speed error doubled or tripled. 

We dug into the features and discovered why:
* The models had not learned universal vehicle dynamics.
* They had memorized vehicle-specific vibration spectra, mount resonance frequencies, and local throttle habits.
* When tested on a different car with a different suspension stiffness, the learned decision trees evaluated pure noise.

**The Lesson**: Machine learning for physical navigation cannot rely on unconstrained static feature histograms. It requires temporal convolutional architectures with causal receptive fields and strict, whole-trip leakage-proof evaluation splits.

---

## 3. The 70 m/s Ghost: The Danger of Walking with Automotive AI
To test our updated Dilated Temporal Convolutional Network (Expanded TCN) on Android, an engineer walked across campus with the phone in hand. Suddenly, the screen flashed with velocities of **60 to 71.66 m/s (over 250 km/h)** on a slow human walk!

It would have been easy to hide this result or dismiss it as a random glitch. Instead, we halted all deployment and conducted a forensic feature attribution audit.

The findings were illuminating:
* In a passenger vehicle, lateral cornering dynamics relate to forward speed through the centripetal relationship: $a_y = v^2/R = v \cdot \omega_{\text{yaw}}$.
* The TCN had correctly learned that when horizontal acceleration coincides with rapid yaw rate, the vehicle is navigating a high-speed curve.
* But during a pedestrian walk, human arm swing and gait dynamics produced measured yaw rates of **$+2.85\text{ rad/s}$** — which was **$+26.08\sigma$ (twenty-six standard deviations)** outside the automotive training envelope!
* The feedforward network dutifully multiplied this out-of-distribution rotation into a supersonic vehicle speed.

**The Lesson**: Neural networks are brittle outside their training manifold. Automotive models must be guarded by input $\sigma$-clipping, physical acceleration bounds, and out-of-distribution motion gating. Crucially, we established that **pedestrian testing can never substitute for true vehicle validation.**

---

## 4. The Highway Cruise Paradox: Where Information Vanishes
Having trained the Expanded TCN across 39 trips with zero leakage, we noticed an obstinate flaw:
* On winding mountain roads, the model was extraordinary: 60-second drift was **3.51% on `Vw12` and 4.91% on `Vw14a`**, passing the SIH <10% requirement with ease.
* But on straight motorways (`V-Vfa02`), actual speeds reached 110–120 km/h, while the TCN asymptotically plateaued around **85–88 km/h**, creating a persistent $-3.45\text{ m/s}$ negative bias and 11.2% drift.

Our initial instinct was that high speeds were under-represented in the dataset. We implemented continuous inverse-density loss weighting and stratified mini-batch sampling (Phase 5.2). Mountain accuracy increased further, but the highway ceiling didn't budge.

Next, we hypothesized that high-frequency chassis and tire vibrations might carry speed information (Phase 5.3). We audited 64 trips and >450,000 windows in the frequency domain. The correlation between dominant vibration frequency and vehicle speed was **$r = -0.032$** — zero. On the motorway, the vibration peak stayed locked at 2.9 Hz (suspension bounce) whether the car did 85 or 105 km/h.

Then came the definitive realization (Phase 5.4):
* During steady-state cruising on a straight, smooth highway:
  $$a_{\text{long}} \approx 0, \quad \omega_{\text{yaw}} \approx 0$$
* A smartphone sitting in a dashboard mount measures acceleration and angular rates that are mathematically indistinguishable from sensor noise ($ROC\text{-}AUC = 0.625$).
* **The neural network was not failing because it was poorly trained. It was failing because the information does not physically exist in the instantaneous IMU signal.**

---

## 5. The Synthesis: Physics, Memory, and Adaptive Regimes
This realization forced a fundamental pivot away from black-box AI toward **hybrid physical state estimation**:
* A real car does not enter a GNSS outage from zero velocity. It enters with a known GNSS speed vector ($v_0 = v_{\text{GNSS}}$).
* In high-speed cruise, **state continuity (momentum preservation)** is far more reliable than memoryless neural predictions. Integrating acceleration kinematically on `Vw12` achieved **1.78% drift at 60 seconds**!
* But in urban stop-and-go (`Vta26`), pure kinematics suffered catastrophic quadratic runaway (**574.6% drift**) due to integrating uncalibrated bias during stops.

The solution had to be **regime-aware**:
* At standstill $\implies$ clamp velocity to zero (ZVD).
* During stable cruise $\implies$ trust kinematic momentum ($\beta = 0.993$).
* During dynamic maneuvers $\implies$ blend kinematics with the Dilated TCN ($\beta = 0.970$).
* During high disagreement $\implies$ clip acceleration and bound updates.

In Phase 5.6, this adaptive fusion compressed urban runaway on `Vta26` from **575% down to 47%** and achieved the lowest overall 60-second drift (**16.76%**).

---

## 6. The Scientific Destination
Yet even with adaptive fusion, the cross-trip pass rate at 60 seconds is **3 out of 13 usable trips (23.1%)**.

We did not invent a miraculous algorithm that bypasses the laws of physics. But we achieved something much rarer in applied hackathons:
1. We systematically deconstructed every failure mode down to its governing equations.
2. We proved what smartphone IMUs can do (attitude stabilization, centripetal speed estimation on curves, short-outage dead reckoning).
3. We proved what standalone smartphone IMUs *cannot* do (distinguish steady-state straight cruise without physical scale references).

This repository is the honest, reproducible record of that investigation.
