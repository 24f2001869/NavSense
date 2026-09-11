# Phase 5.5: Stateful Kinematic Integration & Momentum Preservation

## 1. Context & Motivation
Phase 5.4 proved why the static feedforward TCN fails on straight highways: with zero acceleration and zero yaw rate, an IMU window carries zero speed information ($ROC\text{-}AUC = 0.625$). The memoryless TCN inevitably reverts to its training distribution mean (~85 km/h).

However, in a real navigation system, **we are not starting from zero knowledge**. Before entering a tunnel or canyon, the vehicle was tracking a valid GNSS speed:
$$v(t_0) = v_{\text{GNSS}}$$

**The Hypothesis**: By preserving initial velocity and integrating longitudinal acceleration kinematically ($v_{k+1} = v_k + a_{\text{long}}\Delta t$), can we eliminate the highway cruising deficit?

---

## 2. Mathematical Models Evaluated

Three distinct velocity propagation architectures were benchmarked:

### Model 1: Static Dilated TCN (Memoryless Baseline)
$$v_{\text{M1}}(t) = f_{\text{TCN}}(\mathbf{x}_{t-10\text{s}:t})$$

### Model 2: Pure Kinematic Integration
$$v_{\text{M2}}(0) = v_{\text{GNSS}}(0)$$
$$v_{\text{M2}}(k+1) = \max\left(0, \, v_{\text{M2}}(k) + a_{\text{long}}(k) \Delta t\right)$$

### Model 3: Fixed Damped Momentum (Phase 5.5 Hybrid, $\beta = 0.985$)
$$v_{\text{pred}}(k+1) = \max\left(0, \, v_{\text{M3}}(k) + (a_{\text{long}}(k) - \hat{b}_a) \Delta t\right)$$
$$v_{\text{M3}}(k+1) = \beta \cdot v_{\text{pred}}(k+1) + (1 - \beta) \cdot v_{\text{TCN}}(k+1)$$

---

## 3. The Empirical Discovery: A Severe Regime Trade-Off

The benchmark across the held-out test trips revealed an extreme, regime-dependent divergence:

### 3.1 The Win: Mountain & High-Speed Highway Continuity
On open roads with high continuous speed, kinematic momentum produced exceptional performance:

| Test Trip | Category | Duration | Static TCN 60s Drift (%) | Damped Momentum 60s Drift (%) |
|:---|:---|:---:|:---:|:---:|
| **Vw12** | Mountain | 81.9 s | 6.68% | **1.78% (Stunning Pass ✅)** |
| **Vw14a** | Mountain | 303.9 s | 6.93% | **12.09%** |
| **V-Vfa02** (10s) | Motorway | 6,742 s | 27.31% | **10.62% (Pure Kin: 9.06% ✅)** |
| **V-Vfa02** (30s) | Motorway | 6,742 s | 13.52% | **12.10%** |

On `Vw12`, preserving momentum produced **1.78% drift at 60 seconds**, far superior to the static TCN.

### 3.2 The Disaster: Urban Stop-and-Go Phantom Motion
In dense suburban traffic (`Vta26`), the exact opposite occurred:
* When the vehicle stopped at a traffic light, true velocity dropped to $0.0\text{ m/s}$.
* However, residual uncalibrated accelerometer bias ($b_a \approx 0.15\text{ m/s}^2$) continued to be integrated:
  $$0 \longrightarrow 0.15 \longrightarrow 0.45 \longrightarrow 0.90 \longrightarrow 1.50\text{ m/s}$$
* **The Result on `Vta26`**: Pure kinematics created **phantom velocity**, producing an astronomical **574.6% drift at 60 seconds (292.8 m error on a 50 m trip)**!

```
Vta26 (Urban Congestion):
- Static TCN:        70.8% drift (Survived stop-and-go because it doesn't integrate)
- Pure Kinematics:  574.6% drift (CATASTROPHIC QUADRATIC DIVERGENCE)
- Damped Momentum:  197.6% drift (Severe Divergence)
```

---

## 4. Aggregate Benchmark Across All Horizons

Across the usable test trips:

| Outage Horizon | Pure Kinematics Mean Drift (%) | Damped Momentum Mean Drift (%) | Static TCN Mean Drift (%) |
|:---:|:---:|:---:|:---:|
| **10 Seconds** | **83.5%** (6/18 pass) | 111.4% (6/18 pass) | 106.4% (5/18 pass) |
| **20 Seconds** | 148.0% (4/17 pass) | 193.5% (4/17 pass) | **123.9%** (5/17 pass) |
| **30 Seconds** | 118.2% (2/16 pass) | 127.1% (2/16 pass) | **65.6%** (5/16 pass) |
| **60 Seconds** | 89.7% (0/13 pass) | 40.8% (1/13 pass) | **18.3%** (3/13 pass) |

---

## 5. Conclusion & The Core Insight

> [!IMPORTANT]
> **Core Scientific Finding**:  
> Kinematic momentum is **one half of the solution**.  
> - Pure kinematics excels during smooth open-road driving, but creates massive phantom motion during urban stops.  
> - The static TCN survives urban stop-and-go because its errors do not accumulate quadratically, but it collapses on straight highways.  
> - **A static transition based purely on elapsed time (e.g. 0–20s momentum $\to$ 20–60s TCN) is too simplistic.** The transition must be **condition-based (regime-aware)**.
