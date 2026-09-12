# Supplementary Material: Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages

**Article Title:** Failure-Aware Smartphone Inertial Dead Reckoning Under GNSS Outages: An Empirical Study of Learned Velocity, Sensor Fusion, Map Constraints, and Distribution Shift  
**Journal:** *The Journal of Navigation* (Cambridge University Press / Royal Institute of Navigation)  
**Author:** Rahul Kumar  
**Affiliation:** Integrated M.Tech. (Materials Engineering), School of Engineering Sciences & Technology, University of Hyderabad, Hyderabad, India  
**Corresponding Email:** 24f2001869@ds.study.iitm.ac.in  

---

### Overview
This supplementary document provides extended tabular data, detailed per-trip forensic metrics, hardware profiling logs, and mathematical derivations supporting the main article. All evaluations utilize the public IO-VNBD benchmark dataset (Onyekpe et al., 2021) and on-device Android mobile execution on a Google Pixel 7a smartphone.

---

## Supplementary Table S1: Per-Trip Performance Breakdown Under 60-Second GNSS Blackouts

The main manuscript presents aggregate and category-level metrics across the 13 usable held-out test routes. Table S1 provides the comprehensive route-by-route results for both unassisted strapdown inertial dead reckoning and the proposed NavSense adaptive fusion engine.

Table S1. Detailed route-by-route evaluation across all 13 usable held-out test trips under 60-second satellite outages.
| Trip ID | Route Environment | Reference Distance (m) | Pure Strapdown Drift (m) | Pure Strapdown Drift (%) | NavSense Adaptive Drift (m) | NavSense Adaptive Drift (%) | SIH Benchmark Status (<10%) | Observed Dynamic Characteristics |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Vta21** | Suburban Town | 786.7 | 406.8 | 51.45% | 61.5 | 7.88% | **MET (PASS)** | Steady suburban cruising along arterial corridor; bounded drift |
| **Vta22** | Suburban Town | 641.3 | 176.7 | 27.72% | 107.7 | 16.84% | Unmet | Frequent stop-and-go queue; velocity dampens with traffic |
| **Vta23** | Suburban Town | 570.1 | 352.9 | 62.40% | 156.8 | 26.16% | Unmet | Sharp 90-degree intersection turn during blackout window |
| **Vta24** | Suburban Town | 384.8 | 293.6 | 73.42% | 57.0 | 14.65% | Unmet | Commercial zone corridor; low-speed transient manoeuvres |
| **Vta26** | Suburban Town | 258.7 | 292.8 | 574.65% | 66.3 | 47.31% | Unmet | Short transit crawl; large percentage due to short travel distance |
| **Vta27** | Suburban Town | 818.9 | 306.5 | 37.18% | 95.9 | 12.68% | Unmet | Signalised arterial road with moderate speed transitions |
| **Vta28** | Suburban Town | 565.3 | 281.8 | 64.92% | 120.6 | 25.30% | Unmet | Roundabout entry and deceleration into residential zone |
| **Vw12** | Mountain Route | 1500.2 | 650.3 | 43.40% | 45.3 | 3.01% | **MET (PASS)** | Continuous mountain descent; highly dynamic centripetal cues |
| **Vw14a** | Mountain Route | 1511.1 | 312.6 | 21.06% | 130.4 | 8.54% | **MET (PASS)** | Rural mountain corridor with sweeping curves and steady throttle |
| **Vw14b** | Mountain Route | 1264.5 | 432.8 | 42.48% | 121.0 | 10.19% | Unmet | Consecutive hairpins; centripetal acceleration transitions |
| **Vw16a** | Mountain Route | 899.9 | 388.5 | 51.76% | 136.1 | 16.88% | Unmet | High curvature hill climb with frequent throttle modulations |
| **V-Vfa02** | Motorway | 1460.3 | 313.9 | 25.44% | 157.2 | 11.69% | Unmet | Unaccelerated high-speed cruising; negative TCN regression bias |
| **Vw15** | Stationary Control | 1.3 | 5.6 | 450.48% | 0.82 | 63.83% | Unmet | Zero-velocity control trip; standstill clamped by causal ZVD |
| **Mean / Total** | **All 13 Routes** | **820.2** | **324.21** | **89.66%** | **96.65** | **16.76%** | **3 / 13 (23.08%)** | **Complete Held-Out 60-s Evaluation Set** |

*Note: In accordance with the experimental protocol, trips with total durations shorter than 70.0 seconds are excluded from the 60-second evaluation set because they cannot accommodate the 10-second pre-outage calibration window plus the 60-second blackout duration. Macro-average normalized drift weights each dynamic trip equally, excluding the stationary control trip Vw15 from the percentage mean per Equation (13) to prevent division-by-zero distortion.*

---

## Supplementary Table S2: Spectral Vibration Speed Audit Across 64 Vehicle Trips

To evaluate the hypothesis that vehicle road and engine vibration captured by consumer smartphone IMUs provides a reliable proxy for vehicle forward speed, Table S2 summarizes the spectral census across all 64 passenger-car trips in the IO-VNBD dataset (>450,000 one-second FFT windows).

Table S2. Cross-trip spectral vibration census across four standardized road categories.
| Road Environment | Trips Analyzed (N) | Total 1-s Windows | Dominant Spectral Peak (Hz) | Pearson Correlation ($r$) with Speed | Spearman Correlation ($\rho$) with Speed | Physical Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Suburban Towns (Vta)** | 30 | 218450 | 2.24 ± 0.18 Hz | -0.028 | -0.024 | Consistent with low-frequency vehicle-body/chassis dynamics |
| **Dense Urban (Vtb)** | 12 | 40890 | 2.31 ± 0.22 Hz | -0.041 | -0.035 | Consistent with road-surface excitation and low-frequency vehicle dynamics |
| **Mountain Routes (Vw)** | 20 | 170920 | 2.45 ± 0.15 Hz | -0.019 | -0.016 | Consistent with chassis/body dynamics under curved-road motion |
| **Motorways (V-Vfa)** | 2 | 39180 | 2.86 ± 0.12 Hz | -0.038 | -0.031 | Persistent spectral peak across speed bands; physical mechanism not directly identified |
| **Complete Corpus** | **64** | **>450,000** | **2.2–2.5 Hz** | **-0.032** | **-0.028** | **No reliable speed signal identified at 10 Hz feature rate** |

*Methodological Note: Power spectral density was computed causally using Welch periodograms across 1.0-second sliding windows (10 samples, zero-padded to 64 points) on tri-axial specific forces and angular velocities.*

---

## Supplementary Table S3: On-Device Android Execution Profiling and Numerical Parity

Table S3 details the execution latency, memory utilization, and cross-platform numerical parity between the mobile Android deployment (Java/EJML/ONNX Runtime) on a Google Pixel 7a and the 64-bit Python reference engine.

Table S3. Mobile execution benchmarks and cross-platform numerical parity on Google Pixel 7a.
| Pipeline Subsystem | Implementation Language | Runtime Framework | Mean Execution Latency | Peak Latency (99th %tile) | Maximum Discrepancy vs. Python 64-bit Engine |
|:---|:---|:---|:---:|:---:|:---:|
| **IMU Ingestion & Decimation** | Kotlin | Android SensorManager HAL | 0.12 ms | 0.35 ms | Identical (Zero difference) |
| **Coordinate Leveling & PCA** | Java | Native Linear Algebra | 0.24 ms | 0.58 ms | Leveling Angle: <0.005° |
| **TCN Forward Inference** | C++ / Java Bindings | ONNX Runtime Mobile v1.16 | 5.82 ms | 7.84 ms | Speed: <0.001 m/s |
| **15-State ESKF Propagation** | Java | Efficient Java Matrix Library (EJML) | 0.94 ms | 1.38 ms | Position: 0.24177 m; Heading: 0.01829° |
| **Decoupled Velocity Damping** | Java | Custom EJML Matrix Slice | 0.18 ms | 0.32 ms | Velocity: 0.02117 m/s |
| **Open-Loop Vector Mapping** | Kotlin | OSMDroid Offline Engine | 1.85 ms | 3.42 ms | Render: <1 frame |
| **Total Navigation Epoch Loop** | **Mixed** | **End-to-End Android Pipeline** | **9.15 ms** | **13.89 ms** | **Within the 100-ms epoch budget** |

---

## Supplementary Section S1: Mathematical Derivation of Causal Receptive Field Reach

The causal Temporal Convolutional Network (TCN) operates over a sliding temporal buffer of $W = 100$ samples (10.0 s at 10 Hz). To ensure that predictions depend strictly on past motion dynamics with zero dependency on future epochs, causal convolutions are enforced via asymmetric left-padding.

For a 1D convolutional network with $L$ dilated residual blocks, kernel size $k$, and dilation factor $d_l$ at layer $l$, the mathematical receptive field reach $R$ (expressed in sample steps) is given by:

$$R = 1 + \sum_{l=0}^{L-1} (k - 1) \cdot d_l$$

In the TCN architecture, the parameters are:
* Number of dilated residual layers: $L = 4$
* Convolutional kernel size: $k = 3$
* Dilation schedule: $d_l \in \{1, 2, 4, 8\}$ for $l \in \{0, 1, 2, 3\}$

Each residual block contains two sequential dilated convolutional layers with identical dilation $d_l$. Substituting these values:

$$R_{\text{block},l} = 2 \cdot (k - 1) \cdot d_l = 2 \cdot (3 - 1) \cdot d_l = 4 d_l$$

Summing across all four dilated blocks:

$$R = 1 + \sum_{l=0}^3 4 d_l = 1 + 4 \cdot (1 + 2 + 4 + 8) = 1 + 4 \cdot 15 = 61 \text{ samples}$$

At the nominal 10.0 Hz sampling rate, 61 samples correspond to an effective causal look-back interval of:

$$\tau_{\text{causal}} = (61 - 1) \times 0.10\text{ s} = 6.0\text{ seconds}$$

representing input dependency on the causal interval $[t - 60, t]$.
