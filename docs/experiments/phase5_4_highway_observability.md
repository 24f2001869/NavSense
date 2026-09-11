# Phase 5.4: Highway Steady-State Cruise Observability Analysis

## Question
Why does the Temporal Convolutional Network (TCN) underestimate vehicle forward speed during steady motorway cruising, consistently flatlining at $\sim 85\text{--}89\text{ km/h}$ even when the vehicle is traveling at 105–115 km/h? Is highway speed physically observable from smartphone inertial measurements alone?

## Hypothesis
During constant-speed, straight-line highway driving, longitudinal acceleration is zero ($\ddot{x} = 0$) and turn rates are zero ($\dot{\psi} = 0$). According to Galilean invariance, inertial sensors observe zero net specific force (aside from gravity). Therefore, smartphone IMU signals during steady cruising at 80 km/h and 115 km/h are statistically indistinguishable from the sensor noise floor.

## Dataset
* **Trip:** IO-VNBD Motorway benchmark trip `V-Vfa02` (Driver E, high-speed motorway).
* **Length:** 163.4 km, 446 rolling evaluation windows.
* **Ground Truth:** Vehicle CAN bus forward speed and reference GNSS.

## Inputs
* 10 Hz resampled tri-axial linear acceleration ($a_x, a_y, a_z$).
* 10 Hz resampled tri-axial angular rates ($\omega_x, \omega_y, \omega_z$).
* Derived total acceleration magnitude $\|a\|$, jerk, and curvature proxy $\kappa$.

## Method
1. **Steady-Cruise Window Extraction:** Identified temporal segments where CAN speed $> 70\text{ km/h}$ and acceleration variance $\sigma_a < 0.15\text{ m/s}^2$ for $\ge 6.0\text{ seconds}$ (exceeding the TCN's 6.1s receptive field).
2. **Speed Bin Stratification:** Split cruise segments into two speed regimes:
   - Low Cruise: $75\text{--}85\text{ km/h}$
   - High Cruise: $100\text{--}115\text{ km/h}$
3. **Statistical Observability Tests:**
   - Evaluated Receiver Operating Characteristic (ROC) area under the curve (AUC) to classify low vs. high cruise from IMU feature distributions.
   - Computed mutual information and Kolmogorov-Smirnov distance between feature distributions across speed bins.

## Experimental Configuration
* Receptive field: 61 time steps (6.1 seconds).
* Feature vectors: 12-dimensional normalized IMU statistics.
* Execution script: `scripts/experiments/run_highway_speed_observability.py`.

## Results
* **ROC-AUC for Cruise Speed Separation:** **0.625** (barely above a random coin flip of 0.500).
* **Feature Distributions:**
  - Longitudinal acceleration distribution at 80 km/h: Mean $= +0.02\text{ m/s}^2$, $\sigma = 0.08\text{ m/s}^2$.
  - Longitudinal acceleration distribution at 110 km/h: Mean $= +0.03\text{ m/s}^2$, $\sigma = 0.09\text{ m/s}^2$.
  - Gyroscope yaw rate distribution: Mean $= 0.001\text{ rad/s}$ across both regimes.
* **TCN Behavior:** In the absence of distinct inertial dynamics, the model's minimum Mean Squared Error (MSE) solution is to output the conditional expectation of its training set ($\sim 85\text{ km/h}$).

## Figures
Diagnostic distribution and ROC plots:
- Feature separation and ROC curves: [`assets/experiments/highway_observability_plots.png`](../../assets/experiments/highway_observability_plots.png)

## Interpretation
The flatlining of the TCN at 85 km/h is not a neural network training failure; it is an **inherent physical unobservability boundary**. A memoryless model receiving only inertial inputs cannot observe absolute velocity in unaccelerated motion. Once dynamic cues decay past the temporal receptive field, the model naturally regresses to the dataset mean.

## What Worked
* Conclusively explained the underlying cause of high-speed cruise under-prediction.
* Defined the theoretical necessity for stateful kinematic tracking.

## What Failed
* Confirmed that static, feedforward machine learning models cannot independently solve dead reckoning on long, straight highways without stateful momentum.

## Decision
**Observability Boundary Established.** We must augment feedforward ML with stateful kinematic velocity propagation ($v_{k+1} = v_k + a \Delta t$) initialized from the last known GNSS velocity fix prior to the blackout.

## Evidence
- Execution Script: [`scripts/experiments/run_highway_speed_observability.py`](../../scripts/experiments/run_highway_speed_observability.py)
- Metrics JSON: [`results/adaptive_fusion/highway_observability_metrics.json`](../../results/highway_observability/highway_observability_summary.json)
- Diagnostic Visual: [`assets/experiments/highway_observability_plots.png`](../../assets/experiments/highway_observability_plots.png)

## Reproduction
```bash
python scripts/experiments/run_highway_speed_observability.py
```

## Limitations
* Applies specifically to unaccelerated straight highway driving; when the vehicle makes lane changes or maneuvers, speed observability is partially restored through centripetal acceleration ($a_{\text{lat}} = v^2/R$).
