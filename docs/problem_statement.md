# SIH Problem Statement 26168: Context & Challenge Requirements

## 1. Official Problem Statement Overview

* **Problem ID**: SIH26168  
* **Title**: AI/ML-Based Intelligent Dead Reckoning (IDR) Navigation Engine for Ground Vehicles  
* **Domain**: Intelligent Transportation Systems / Defense & Strategic Navigation / Harsh Urban Environments  
* **Objective**: Develop an edge-deployable navigation system capable of sustaining accurate vehicle dead reckoning on commercial smartphones during prolonged GNSS outages (tunnels, urban canyons, dense foliage, electronic countermeasures).

---

## 2. Core Operational Challenges

### 2.1 The Environmental Reality
In modern urban and subterranean environments, Global Navigation Satellite System (GNSS) signals suffer from:
1. **Complete Signal Blockage**: Total loss of line-of-sight tracking in mountain tunnels, underground transit corridors, and subterranean parking structures.
2. **Severe Multipath & Reflection**: In urban canyons surrounded by high-rise buildings, satellite signals reflect off glass and concrete facades, producing pseudorange distortions of $30\text{--}100\text{ meters}$.
3. **Intentional / Unintentional Jamming & Spoofing**: Electronic interference in sensitive or contested operational zones rendering civilian GNSS completely unusable.

### 2.2 The Hardware Reality: Consumer Smartphone MEMS Sensors
In contrast to specialized military or autonomous vehicle inertial navigation systems (which utilize ring laser gyroscopes or fiber-optic gyroscopes costing $10,000–$100,000), this challenge demands an **accessible consumer solution**:
* **Sensor Grade**: Low-cost Micro-Electro-Mechanical Systems (MEMS) sensors integrated into commercial Android smartphones.
* **Bias Instability**: MEMS accelerometers exhibit bias instabilities of $0.05\text{--}0.2\text{ m/s}^2$; gyroscopes exhibit drift rates of $0.5\text{--}3.0^\circ/\text{s}$.
* **No Direct Wheel Access**: Unlike factory automotive navigation, the smartphone has **no direct physical connection to vehicle CAN bus wheel-speed encoders or transmission gearboxes**. It must deduce all movement strictly from inertial, magnetic, barometric, and topological signals.

---

## 3. The Target Benchmark Requirements

To be considered viable for practical deployment, the navigation engine must satisfy:
1. **Blackout Duration**: Sustain dead reckoning through continuous GNSS outages lasting up to **60 seconds**.
2. **Precision Target**: Maintain position drift below **10% of total distance travelled** ($e_{\text{drift}} / d_{\text{travelled}} \le 0.10$).
3. **Edge Execution**: Execute locally in real time on commercial Android hardware without cloud computation.
4. **Zero Prior In-Vehicle Calibration**: Operate dynamically regardless of the specific vehicle make or mount orientation.
