# Benchmark Evaluation Protocol

This document defines the strict, reproducible evaluation protocol used across all blackout benchmarks.

---

## 1. Rolling Window Generation Protocol

To avoid cherry-picking single easy or hard driving segments:
1. **Window Duration ($H$)**: Evaluated at four standard horizons: $H \in \{10\text{ s}, 20\text{ s}, 30\text{ s}, 60\text{ s}\}$.
2. **Pre-Outage Calibration Window ($T_{\text{pre}}$)**: Every episode requires a minimum of **$10.0\text{ seconds}$** of pre-outage driving to estimate initial velocity and accelerometer bias.
3. **Stride ($\Delta s$)**: Windows are extracted with a rolling stride of **$15.0\text{ seconds}$**.
4. **Trip Inclusion Rule**: A trip is evaluated at horizon $H$ if and only if:
   $$T_{\text{trip}} \ge T_{\text{pre}} + H$$
   For $H = 60\text{ s}$, trips must be at least **$70.0\text{ seconds}$** in duration.

---

## 2. Core Quantitative Metrics

### 2.1 Absolute Position Drift ($e_{\text{drift}}$)
Let $\hat{\mathbf{p}}(H)$ be the dead-reckoned 2D position at the end of the blackout and $\mathbf{p}_{\text{true}}(H)$ be the ground reference position:

$$e_{\text{drift}} = \|\hat{\mathbf{p}}(H) - \mathbf{p}_{\text{true}}(H)\|_2$$

### 2.2 Distance-Normalized Drift Percentage ($\%_{\text{drift}}$)
Let $d_{\text{true}}(H) = \sum_{k=0}^{N-1} \|\mathbf{p}_{\text{true}}(k+1) - \mathbf{p}_{\text{true}}(k)\|_2$ be the true distance traveled:

$$\%_{\text{drift}} = \frac{e_{\text{drift}}}{\max(d_{\text{true}}(H), \, 1.0\text{ m})} \times 100\%$$

### 2.3 The SIH <10% Pass Criterion
An episode is classified as an **Episode Pass** if:
$$\%_{\text{drift}} \le 10.00\%$$

A trip is classified as a **Trip Pass** if:
$$\operatorname{Mean}\left(\%_{\text{drift}}\right) \le 10.00\%$$
across all evaluated rolling episodes within that trip.
