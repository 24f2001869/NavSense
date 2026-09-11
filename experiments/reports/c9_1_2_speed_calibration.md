# Stage C9.1.2: Causal Pre-Outage Speed Calibration Benchmark Report

**Date:** September 8, 2026  
**Objective:** Controlled evaluation of pre-outage speed calibration models (Additive, Multiplicative, Affine) to determine whether along-track error can be reduced without urban degradation.

---

## Vta04 (Urban Loop)

### Horizon 10s (N = 14 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 32.58 m | 19.70 m | 19.55 m | 32.60% | 11.28° | 35.7% |
| **Model B: Additive Bias (v = v_ml + b)** | 31.55 m | 17.84 m | 20.86 m | 31.89% | 12.75° | 50.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 31.43 m | 17.82 m | 20.90 m | 31.78% | 12.07° | 42.9% |
| **Model D: Affine (v = s * v_ml + b)** | 31.13 m | 17.40 m | 20.93 m | 31.49% | 12.71° | 57.1% |
| **Reference: Ground-Truth Oracle** | 31.27 m | 25.22 m | 15.14 m | 27.58% | 1.00° | 7.1% |

### Horizon 20s (N = 7 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 104.54 m | 51.55 m | 72.83 m | 51.35% | 17.07° | 28.6% |
| **Model B: Additive Bias (v = v_ml + b)** | 102.96 m | 47.62 m | 77.06 m | 50.86% | 17.20° | 28.6% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 104.44 m | 46.94 m | 78.39 m | 51.70% | 15.91° | 28.6% |
| **Model D: Affine (v = s * v_ml + b)** | 101.91 m | 46.36 m | 76.85 m | 50.39% | 16.65° | 28.6% |
| **Reference: Ground-Truth Oracle** | 87.82 m | 76.99 m | 35.10 m | 39.13% | 1.12° | 0.0% |

### Horizon 30s (N = 4 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 119.10 m | 33.01 m | 106.89 m | 37.63% | 29.92° | 0.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 126.43 m | 39.31 m | 114.70 m | 39.89% | 31.10° | 0.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 126.74 m | 39.86 m | 113.93 m | 39.91% | 32.08° | 0.0% |
| **Model D: Affine (v = s * v_ml + b)** | 125.83 m | 39.91 m | 113.88 m | 39.71% | 30.75° | 0.0% |
| **Reference: Ground-Truth Oracle** | 113.48 m | 83.02 m | 59.86 m | 34.06% | 0.79° | 0.0% |

### Horizon 60s (N = 2 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 234.85 m | 213.64 m | 88.37 m | 35.33% | 54.44° | 0.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 414.02 m | 200.14 m | 302.51 m | 62.59% | 50.60° | 0.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 1256.15 m | 275.16 m | 1138.10 m | 190.54% | 74.17° | 0.0% |
| **Model D: Affine (v = s * v_ml + b)** | 170.20 m | 163.82 m | 41.11 m | 25.48% | 51.59° | 50.0% |
| **Reference: Ground-Truth Oracle** | 186.61 m | 121.81 m | 139.80 m | 28.26% | 0.63° | 0.0% |

## Vta02 (Highway Arterial)

### Horizon 10s (N = 99 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 39.69 m | 28.92 m | 21.24 m | 45.59% | 12.53° | 4.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 40.09 m | 29.84 m | 21.24 m | 46.29% | 12.55° | 5.1% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 40.05 m | 29.83 m | 21.21 m | 46.31% | 12.56° | 5.1% |
| **Model D: Affine (v = s * v_ml + b)** | 40.86 m | 30.93 m | 21.13 m | 46.59% | 12.54° | 5.1% |
| **Reference: Ground-Truth Oracle** | 1.27 m | 1.01 m | 0.57 m | 1.22% | 0.77° | 100.0% |

### Horizon 20s (N = 50 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 153.52 m | 112.15 m | 81.96 m | 84.89% | 22.69° | 0.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 156.42 m | 115.45 m | 82.42 m | 86.50% | 22.26° | 0.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 156.69 m | 115.79 m | 82.53 m | 86.70% | 22.43° | 0.0% |
| **Model D: Affine (v = s * v_ml + b)** | 157.92 m | 118.05 m | 82.01 m | 87.31% | 22.38° | 0.0% |
| **Reference: Ground-Truth Oracle** | 1.74 m | 1.13 m | 1.06 m | 0.87% | 0.63° | 100.0% |

### Horizon 30s (N = 34 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 296.98 m | 214.86 m | 159.90 m | 105.66% | 25.83° | 0.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 301.00 m | 222.77 m | 156.39 m | 107.97% | 25.76° | 0.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 301.15 m | 223.83 m | 155.20 m | 107.87% | 26.11° | 0.0% |
| **Model D: Affine (v = s * v_ml + b)** | 306.07 m | 226.65 m | 159.16 m | 110.14% | 25.22° | 0.0% |
| **Reference: Ground-Truth Oracle** | 3.41 m | 1.93 m | 2.23 m | 1.40% | 0.55° | 97.1% |

### Horizon 60s (N = 16 windows)

| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Raw ML Speed (v = v_ml)** | 1596.69 m | 1074.95 m | 959.29 m | 280.75% | 34.38° | 0.0% |
| **Model B: Additive Bias (v = v_ml + b)** | 1217.87 m | 969.38 m | 567.36 m | 200.39% | 45.27° | 0.0% |
| **Model C: Multiplicative Scale (v = s * v_ml)** | 1194.57 m | 948.63 m | 549.77 m | 195.97% | 40.85° | 0.0% |
| **Model D: Affine (v = s * v_ml + b)** | 1341.68 m | 1025.13 m | 681.64 m | 222.13% | 41.85° | 0.0% |
| **Reference: Ground-Truth Oracle** | 9.22 m | 7.73 m | 3.95 m | 1.77% | 0.66° | 93.8% |

---

## Physical Forensic Interpretation

