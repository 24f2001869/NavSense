# Phase 5.3: Spectral Vibration Speed-Information Forensic Audit

## 1. Context & Hypothesis
Following the discovery in Phase 5.2 that the motorway speed plateau is an information deficit rather than a training density issue, we evaluated an alternative sensor channel:  
**Does high-frequency vehicle vibration measured by a smartphone IMU correlate with vehicle road speed?**

In theory, tire rotation frequency ($f = v / (2\pi r)$), engine RPM harmonics, and road surface roughness generate chassis vibrations that increase in frequency or intensity with forward speed.

---

## 2. Experimental Methodology & Signal Processing

We conducted an exhaustive spectral audit across **all 64 available vehicle trips in IO-VNBD** (>450,000 rolling windows):
1. **Windowing**: Causal rolling analysis over 2-second, 5-second, and 10-second windows.
2. **Frequency Domain Decomposition**:
   - Power Spectral Density (PSD) using Welch's periodogram.
   - Fast Fourier Transform (FFT) peak detection across tri-axial accelerometer and gyroscope channels.
   - Spectral power integration across sub-bands: $[0.5–2\text{ Hz}]$, $[2–5\text{ Hz}]$, $[5–12\text{ Hz}]$, $[12–25\text{ Hz}]$, $[25–50\text{ Hz}]$.
3. **Hardware Bandwidth Bounds**:
   - Smartphone IMU was sampled at $100\text{ Hz} \implies$ **Nyquist frequency is $50.0\text{ Hz}$**.
   - Wheel rotation at $100\text{ km/h}$ occurs at $\sim 14.2\text{ Hz}$ ($100 / (3.6 \cdot 2\pi \cdot 0.312)$), well within the Nyquist limit.

---

## 3. Empirical Findings: Zero Generalizable Speed Observability

### 3.1 Dominant Frequency vs Forward Speed Correlation
Across all 64 vehicle trips and road categories:

$$\text{Pearson Correlation Coefficient } r(\text{Dominant Frequency}, v_{\text{true}}) = \mathbf{-0.032}$$

$$\text{Spearman Rank Correlation } \rho = \mathbf{-0.018}$$

The statistical relationship between dominant vibration frequency and forward vehicle speed was essentially zero.

### 3.2 The Motorway Constant-Frequency Paradox
On the motorway trip `V-Vfa02`:
* As vehicle speed varied widely between **$86\text{ km/h}$ and $104\text{ km/h}$** ($23.8\text{ to } 28.9\text{ m/s}$), the dominant vertical acceleration vibration peak remained locked at **$2.86\text{ to } 2.93\text{ Hz}$**.
* This stationary $\sim 2.9\text{ Hz}$ mode corresponds to the **vehicle chassis suspension bounce frequency**, not tire or wheel rotation.

### 3.3 High-Speed Motorway Ablation Benchmark
We trained a comparative Random Forest model with and without rolling spectral vibration features:

| Model Variant | Motorway MAE | Motorway RMSE | 60s Motorway Drift |
|:---|:---:|:---:|:---:|
| **Baseline (Kinematic Features Only)** | **4.39 m/s** | 5.21 m/s | 163.9 m (11.2%) |
| **With Vibration Spectrum (FFT Peaks + PSD Bands)** | **4.38 m/s** | 5.20 m/s | 163.4 m (11.2%) |
| **Delta Improvement** | **-0.01 m/s (-0.2%)** | -0.01 m/s | -0.5 m |

Adding spectral vibration features produced an imperceptible **$0.01\text{ m/s}$ ($0.2\%$) difference** in speed estimation error.

---

## 4. Scientifically Cautious Conclusion & Decision

> [!CAUTION]
> **Authoritative Decision: REJECTED as a primary vehicle speedometer.**  
> **Finding Formulation**:  
> *"No reliable, generalizable speed-specific vibration signal was demonstrated under the tested dataset and sampling conditions."*  
> While high-end automotive acoustic sensors mounted directly on wheel hubs may observe tire-road harmonics, commercial smartphone IMUs mounted on windshield or dashboard brackets are isolated by the vehicle suspension and mount damping. Vibration spectrum cannot be relied upon to solve the steady-state highway cruise deficit.
