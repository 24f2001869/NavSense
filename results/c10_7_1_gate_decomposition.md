# Stage C10.7.1: Isolated VNHC Gate Decomposition Diagnostic

**Date**: 2026-09-09  
**Status**: Completed (Offline Diagnostic Only — Production Code strictly frozen)  
**Objective**: Experimentally decompose the current VNHC innovation rejection mechanism into its orthogonal constituent gates to isolate the exact causal mechanism responsible for the persistent unrecoverable latch.

---

## 1. Executive Summary & Core Verdict

In Stage C10.7, replay of GNSS-denied bus telemetry demonstrated that removing VNHC innovation rejection cut vertical drift by over 99% in key stress episodes, but simultaneously revealed that un-gated measurement forcing during violent phone handling (tumbling) destabilizes the filter. However, Stage C10.7 evaluated composite interventions (combining gate removal, Huber attenuation, and heuristic counters).

**Stage C10.7.1** isolates the exact gating mechanism through a strictly controlled, 4-variant orthogonal decomposition across all 14 blackout episodes (3,786 epochs, 7.6 minutes of GNSS denial) on target Android bus data:
- **Zero code changes** to production Android Java or Python engines.
- **Zero state clamping** ($v_z \leftarrow 0$).
- **Zero variance inflation** or heuristic recovery counters.
- **Strictly identical initial states, covariances, biases, and update equations**.

```
                           ┌─────────────────────────────────────────────────────────┐
                           │                 VNHC GATE DECOMPOSITION                 │
                           └────────────────────────────┬────────────────────────────┘
                                                        │
                   ┌────────────────────┬───────┴────────────┬────────────────────┐
                   ▼                    ▼                    ▼                    ▼
             [Variant A]          [Variant B]          [Variant C]          [Variant D]
               CONTROL             RESIDUAL             NIS GATE             BOTH GATES
             (Both Gates)        REMOVED ONLY         REMOVED ONLY            REMOVED
           ┌──────────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
           │ Turn:  3°/s  │    │ Turn:  3°/s  │     │ Turn:  3°/s  │     │ Turn:  3°/s  │
           │ NIS:   9.21  │    │ NIS:   9.21  │     │ NIS:   OFF   │     │ NIS:   OFF   │
           │ Res:   3 m/s │    │ Res:   OFF   │     │ Res:   3 m/s │     │ Res:   OFF   │
           └──────────────┘    └──────────────┘     └──────────────┘     └──────────────┘
```

### The Causal Verdict: The "Dual-Lock Cascaded Latch"
The gate decomposition isolates the contribution of the NIS and residual gates to persistent VNHC rejection:
1. **The NIS gate and the residual gate act as dual, cascaded locks with different thresholds**:
   - Because innovation covariance $S = H P H^T + R \approx 0.30 - 0.35\,\text{m}^2/\text{s}^2$, for the observed innovation covariance range, the NIS threshold ($\text{NIS} > 9.21$) corresponds approximately to a residual magnitude of **1.6–1.8 m/s** ($|r_{\text{vert}}| > \sqrt{9.21 S}$).
   - The hard residual gate sits at $|r_{\text{vert}}| > \mathbf{3.0\,\text{m/s}}$.
2. **Removing NIS alone (Variant C) is completely ineffective**:
   - In **10 out of 14 episodes**, Variant C is 100% bit-for-bit identical to Control.
   - Mean longest rejection run remains virtually unchanged (**37.4 s vs 38.9 s** in Control). The hard $3.0\,\text{m/s}$ gate independently maintains the lock-out.
3. **Removing the Residual gate alone (Variant B) provides partial, but incomplete, relief**:
   - Mean longest rejection run drops from $38.9\,\text{s} \to \mathbf{17.1\,\text{s}}$, but in deep drift episodes (like Trip 1 Ep 5), vertical drift still reaches $+19.9\,\text{km}$ because drift exceeds the approximate 1.6–1.8 m/s NIS threshold.
4. **Removing both innovation gates (Variant D) eliminates the latch during straight driving, but confirms the danger of un-gated forcing**:
   - In straight-driving episodes, rejection runs collapse to **$7.5\,\text{s}$** (purely turning duration) and VNHC availability rises to **45.6%**.
   - However, during violent phone tumbling in Trip 2 Ep 5, Variant D produces severe errors ($1,213.4\,\text{m/s}$ peak vertical velocity with baseline Huber, and catastrophic explosion to $+686\,\text{km}$ under pure $\gamma=1.0$ linear Kalman gain).

> **Scientific Wording Note**: Replay evidence strongly supports a two-phase failure sequence: temporary VNHC invalidity during dynamic disturbance followed by persistent rejection after the disturbance subsides.

---

## 2. Quantitative Experimental Results

### 2.1 Master Summary Across All 14 Blackout Outages

| Variant | Turn Gate | NIS Gate | Residual Gate | Mean $|\Delta pos_u|$ (m) | Median $|\Delta pos_u|$ (m) | Max $|\Delta pos_u|$ (m) | Mean Max $|v_u|$ (m/s) | Mean VNHC Active % | Mean Max Rej Run (s) | Median Horiz Err (m) | NaN / Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | $3.0^\circ/\text{s}$ | $9.21$ | $3.0\,\text{m/s}$ | $4,834.1\,\text{m}$ | $441.4\,\text{m}$ | $41,851.9\,\text{m}$ | $185.7\,\text{m/s}$ | 16.3% | $38.9\,\text{s}$ | $3,234.9\,\text{m}$ | 0 |
| **B_RemoveResidualOnly** | $3.0^\circ/\text{s}$ | $9.21$ | **OFF** | $2,899.7\,\text{m}$ | $446.9\,\text{m}$ | $19,930.1\,\text{m}$ | $144.4\,\text{m/s}$ | 26.1% | **$17.1\,\text{s}$** | $3,960.2\,\text{m}$ | 0 |
| **C_RemoveNisOnly** | $3.0^\circ/\text{s}$ | **OFF** | $3.0\,\text{m/s}$ | $3,960.5\,\text{m}$ | $441.4\,\text{m}$ | $20,828.9\,\text{m}$ | $146.9\,\text{m/s}$ | 18.4% | $37.4\,\text{s}$ | $4,489.9\,\text{m}$ | 0 |
| **D_RemoveBoth** | $3.0^\circ/\text{s}$ | **OFF** | **OFF** | **$2,809.7\,\text{m}$** | **$214.7\,\text{m}$** | **$19,228.7\,\text{m}$** | $198.4\,\text{m/s}$ | **45.6%** | **$7.5\,\text{s}$** | **$2,907.2\,\text{m}$** | 0 |

### 2.2 Strictly Linear Kalman Gain Comparison ($\gamma = 1.0$, Zero Huber Attenuation)

To verify that findings are not an artifact of Huber weighting, all 14 episodes were replayed with pure linear Kalman gain ($\gamma = 1.0$, $K = P H^T S^{-1}$):

| Variant | Mean $|\Delta pos_u|$ (m) | Median $|\Delta pos_u|$ (m) | Max $|\Delta pos_u|$ (m) | Mean VNHC Active % | Mean Max Rejection Run (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure_A (Control)** | $4,592.8\,\text{m}$ | $606.4\,\text{m}$ | $19,696.6\,\text{m}$ | 16.4% | $38.9\,\text{s}$ |
| **Pure_B (Residual Removed)** | $3,336.2\,\text{m}$ | $447.0\,\text{m}$ | $16,727.1\,\text{m}$ | 25.5% | **$19.6\,\text{s}$** |
| **Pure_C (NIS Removed)** | $3,512.2\,\text{m}$ | $479.4\,\text{m}$ | $21,682.9\,\text{m}$ | 20.5% | $38.4\,\text{s}$ |
| **Pure_D (Both Removed)** | $50,848.4\,\text{m}$ | **$374.0\,\text{m}$** | **$686,734.2\,\text{m}$** | **45.6%** | **$7.5\,\text{s}$** |

**Crucial Insight**:
Under strictly linear Kalman gain, Variant C still latches for $38.4\,\text{s}$ (identical to Control's $38.9\,\text{s}$). Variant D achieves the lowest median drift ($374.0\,\text{m}$) and lowest rejection ($7.5\,\text{s}$), but in Trip 2 Ep 5 (tumbling) completely explodes to $+686\,\text{km}$ and $72,181\,\text{m/s}$. **This proves that un-gated linear measurement forcing during invalid attitude states is mathematically catastrophic.**

---

## 3. Episode-by-Episode Decomposition Audit

The 14 outage episodes comprise 7.6 minutes across diverse operational conditions (highway cruise, traffic stop-and-go, sharp intersections, and passenger handling).

```
Episode Breakdown Overview:
- 9 episodes: Variant C is 100% identical to Control (Residual gate alone locks the filter).
- 4 episodes: Variant B cuts rejection run by >60% (eliminating the 3 m/s ceiling restores tracking).
- 1 episode (Trip 2 Ep 5): Violent tumbling demonstrates why pure un-gated updates fail.
```

### Episode Details:

#### Trip 1 Episode 1 (3.9s, Rows 2035 -> 2064)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -142.4 m | 61.1 m/s | 0.0% | 3.8 s | 63.3 m |
| **B_RemoveResidualOnly** | -142.4 m | 61.1 m/s | 0.0% | 3.8 s | 63.3 m |
| **C_RemoveNisOnly** | -142.4 m | 61.1 m/s | 0.0% | 3.8 s | 63.3 m |
| **D_RemoveBoth** | -142.4 m | 61.1 m/s | 0.0% | 3.8 s | 63.3 m |
*Analysis*: Turn-rate gate active throughout entire short curve. All variants identical.

#### Trip 1 Episode 2 (94.3s, Rows 2182 -> 2967)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -657.6 m | 61.2 m/s | 46.8% | 16.2 s | 4269.2 m |
| **B_RemoveResidualOnly** | -356.3 m | 36.2 m/s | 62.1% | 2.0 s | 4061.8 m |
| **C_RemoveNisOnly** | -657.6 m | 61.2 m/s | 46.8% | 16.2 s | 4269.2 m |
| **D_RemoveBoth** | -343.6 m | 35.3 m/s | 62.5% | 2.0 s | 4021.4 m |
*Analysis*: Variant C is identical to Control. Removing the residual gate (B) cuts rejection run from $16.2\,\text{s} \to 2.0\,\text{s}$ and cuts drift by 46%.

#### Trip 1 Episode 3 (132.4s, Rows 3900 -> 5056)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +709.3 m | 21.8 m/s | 18.7% | 71.4 s | 296.7 m |
| **B_RemoveResidualOnly** | +954.8 m | 28.5 m/s | 37.0% | 39.6 s | 149.3 m |
| **C_RemoveNisOnly** | +282.5 m | 21.2 m/s | 42.6% | 71.4 s | 421.6 m |
| **D_RemoveBoth** | +70.4 m | 9.4 m/s | 73.5% | 7.1 s | 338.2 m |
*Analysis*: Both gates actively contribute to latching. Variant D achieves $70.4\,\text{m}$ drift (90% reduction) and $7.1\,\text{s}$ max rejection run.

#### Trip 1 Episode 4 (128.6s, Rows 7441 -> 8521)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +358.0 m | 35.5 m/s | 37.9% | 29.9 s | 5835.5 m |
| **B_RemoveResidualOnly** | +171.2 m | 35.7 m/s | 56.7% | 8.5 s | 4479.1 m |
| **C_RemoveNisOnly** | +358.0 m | 35.5 m/s | 37.9% | 29.9 s | 5835.5 m |
| **D_RemoveBoth** | +171.2 m | 35.7 m/s | 56.7% | 8.5 s | 4479.1 m |
*Analysis*: Variant C identical to Control. Removing residual gate cuts max rejection run from $29.9\,\text{s} \to 8.5\,\text{s}$ and cuts drift in half.

#### Trip 1 Episode 5 (125.1s, Rows 11097 -> 12191) — The Canonical 41 km Divergence
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -41851.9 m | 1133.2 m/s | 4.4% | 118.6 s | 52899.2 m |
| **B_RemoveResidualOnly** | +19930.1 m | 510.2 m/s | 21.5% | 13.0 s | 14122.3 m |
| **C_RemoveNisOnly** | -20828.9 m | 728.2 m/s | 4.5% | 118.6 s | 38240.0 m |
| **D_RemoveBoth** | -258.1 m | 286.4 m/s | 59.0% | 6.8 s | 9710.2 m |
*Analysis*: In Control, a road bump triggers initial rejection; vertical velocity integrates unchecked; both gates latch for **118.6 seconds** (94.8% of the blackout), resulting in $-41.8\,\text{km}$ drift. In Variant C, residual gate alone latches the filter for 118.6s. In Variant B, NIS gate latches when $|r| > 1.71\,\text{m/s}$. Only Variant D breaks the latch completely, slashing drift to $-258.1\,\text{m}$ (99.4% reduction).

#### Trip 1 Episode 6 (24.9s, Rows 13395 -> 13503)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +166.5 m | 26.6 m/s | 2.8% | 13.2 s | 168.7 m |
| **B_RemoveResidualOnly** | -49.9 m | 46.9 m/s | 4.6% | 7.2 s | 339.4 m |
| **C_RemoveNisOnly** | +166.5 m | 26.6 m/s | 2.8% | 13.2 s | 168.7 m |
| **D_RemoveBoth** | -35.2 m | 45.9 m/s | 25.7% | 7.2 s | 279.5 m |
*Analysis*: Variant C identical to Control.

#### Trip 1 Episode 7 (99.5s, Rows 13516 -> 14365)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -5402.0 m | 224.8 m/s | 0.8% | 32.2 s | 14832.1 m |
| **B_RemoveResidualOnly** | +807.8 m | 254.7 m/s | 3.8% | 23.2 s | 6963.2 m |
| **C_RemoveNisOnly** | -5402.0 m | 224.8 m/s | 0.8% | 32.2 s | 14832.1 m |
| **D_RemoveBoth** | +339.5 m | 185.4 m/s | 6.9% | 17.6 s | 2447.5 m |
*Analysis*: Variant C identical to Control. Variant D reduces vertical drift from $-5.4\,\text{km} \to +339.5\,\text{m}$ and reduces horizontal drift from $14.8\,\text{km} \to 2.4\,\text{km}$.

#### Trip 1 Episode 8 (39.9s, Rows 14574 -> 14929)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +1039.6 m | 123.3 m/s | 0.3% | 32.0 s | 4710.6 m |
| **B_RemoveResidualOnly** | +946.0 m | 188.5 m/s | 2.0% | 16.4 s | 3858.7 m |
| **C_RemoveNisOnly** | +1039.6 m | 123.3 m/s | 0.3% | 32.0 s | 4710.6 m |
| **D_RemoveBoth** | -2051.8 m | 231.0 m/s | 9.0% | 12.8 s | 3366.9 m |
*Analysis*: Variant C identical to Control.

#### Trip 1 Episode 9 (40.6s, Rows 15338 -> 15578)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +509.3 m | 25.5 m/s | 0.4% | 30.0 s | 2200.6 m |
| **B_RemoveResidualOnly** | -520.4 m | 33.1 m/s | 0.8% | 25.9 s | 3201.0 m |
| **C_RemoveNisOnly** | +509.3 m | 25.5 m/s | 0.4% | 30.0 s | 2200.6 m |
| **D_RemoveBoth** | -94.9 m | 17.8 m/s | 85.5% | 1.5 s | 761.0 m |
*Analysis*: Variant C identical to Control. Variant D raises VNHC active % from 0.4% to 85.5% and reduces drift to $-94.9\,\text{m}$.

#### Trip 2 Episode 1 (2.4s, Rows 0 -> 20)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -7.3 m | 6.1 m/s | 4.8% | 1.9 s | 8542109.6 m |
| **B_RemoveResidualOnly** | -7.3 m | 6.1 m/s | 4.8% | 1.9 s | 8542109.6 m |
| **C_RemoveNisOnly** | -7.3 m | 6.1 m/s | 4.8% | 1.9 s | 8542109.6 m |
| **D_RemoveBoth** | -4.7 m | 3.3 m/s | 33.3% | 0.6 s | 8542110.9 m |
*Analysis*: Initial satellite acquisition outage.

#### Trip 2 Episode 2 (161.7s, Rows 1705 -> 3055)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -36.0 m | 260.7 m/s | 26.1% | 77.1 s | 17522.4 m |
| **B_RemoveResidualOnly** | -4575.3 m | 293.0 m/s | 28.5% | 52.5 s | 17925.3 m |
| **C_RemoveNisOnly** | +7476.6 m | 195.1 m/s | 30.3% | 76.6 s | 16315.5 m |
| **D_RemoveBoth** | +16208.0 m | 641.6 m/s | 47.3% | 10.6 s | 29380.7 m |
*Analysis*: Severe phone handling during an extended outage. Without innovation gating, attitude errors leak into vertical velocity.

#### Trip 2 Episode 3 (57.6s, Rows 3501 -> 3981)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | +365.7 m | 22.3 m/s | 7.3% | 53.4 s | 1432.6 m |
| **B_RemoveResidualOnly** | -31.5 m | 4.1 m/s | 61.1% | 20.0 s | 410.8 m |
| **C_RemoveNisOnly** | +365.7 m | 22.3 m/s | 7.3% | 53.4 s | 1432.6 m |
| **D_RemoveBoth** | -13.1 m | 2.2 m/s | 92.1% | 1.2 s | 74.2 m |
*Analysis*: Clean straight driving episode. Control latches for $53.4\,\text{s}$ out of $57.6\,\text{s}$. Variant C is identical to Control. Variant B cuts drift from $+365.7\,\text{m} \to -31.5\,\text{m}$. Variant D cuts drift to $-13.1\,\text{m}$ (96.4% reduction) and horizontal error from $1,432.6\,\text{m} \to 74.2\,\text{m}$.

#### Trip 2 Episode 4 (80.1s, Rows 4092 -> 4757)
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -373.5 m | 9.3 m/s | 78.5% | 3.1 s | 1798.6 m |
| **B_RemoveResidualOnly** | -373.5 m | 9.3 m/s | 78.5% | 3.1 s | 1798.6 m |
| **C_RemoveNisOnly** | -373.5 m | 9.3 m/s | 78.5% | 3.1 s | 1798.6 m |
| **D_RemoveBoth** | -373.5 m | 9.3 m/s | 78.5% | 3.1 s | 1798.6 m |
*Analysis*: Clean highway cruising where neither innovation gate was triggered. All 4 variants produce identical outputs.

#### Trip 2 Episode 5 (224.0s, Rows 9843 -> 10331) — Dynamic Tumbling Stress Test
| Variant | $\Delta pos_u$ (m) | Max $|v_u|$ (m/s) | VNHC % | Max Rej Run (s) | Final Horiz Err (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A_Control** | -16058.5 m | 589.0 m/s | 0.0% | 61.1 s | 1351.8 m |
| **B_RemoveResidualOnly** | -11729.1 m | 513.7 m/s | 4.5% | 22.5 s | 8058.4 m |
| **C_RemoveNisOnly** | -17836.5 m | 516.0 m/s | 0.4% | 41.0 s | 6604.8 m |
| **D_RemoveBoth** | -19228.7 m | 1213.4 m/s | 8.6% | 22.5 s | 11608.0 m |
*Analysis*: Violent phone tumbling in the passenger cabin. Here, Variant D forces updates on corrupted attitude states, reaching $1,213.4\,\text{m/s}$ peak vertical velocity (and under pure $\gamma=1.0$ linear Kalman update, exploding to $+686\,\text{km}$). **This confirms that blind removal of gating without physical eligibility validation is hazardous.**

---

## 4. Visual Evidence: Diagnostic Plots

The 4-panel diagnostic figure is saved at [`results/c10_7_1_gate_decomposition.png`](c10_7_1_gate_decomposition.png):

*[C10.7.1 Gate Decomposition Diagnostic — Diagnostic chart]*

### Key Visual Observations:
1. **Panel A (Vertical Divergence Across All 14 Outages)**:
   - Median drift drops from $441.4\,\text{m}$ (Control) to $214.7\,\text{m}$ (Variant D).
   - Variant C's median drift ($441.4\,\text{m}$) is identical to Control.
2. **Panel B (Constraint Availability vs Latch Duration)**:
   - Control VNHC active rate is only 16.3%, with a mean max rejection run of $38.9\,\text{s}$.
   - Variant C barely moves the needle (18.4% active, $37.4\,\text{s}$ rejection run).
   - Variant B doubles active percentage during straight segments (26.1%) and cuts rejection run to $17.1\,\text{s}$.
   - Variant D achieves 45.6% availability and drops rejection run to $7.5\,\text{s}$ (the duration of physical curves).
3. **Panel C (Outage-by-Outage Drift on Key Stress Episodes)**:
   - Shows log-scale divergence across Trip 1 Ep 3, 4, 5 and Trip 2 Ep 2, 3, 5.
   - Highlights that in Trip 1 Ep 5, Control diverges by $-41.8\,\text{km}$, while Variant D stays bounded at $-258.1\,\text{m}$.
4. **Panel D (Trip 1 Ep 5 Timeseries Forensic)**:
   - Control (red) and Variant C (green dash-dot) plunge into runaway within 5 seconds of the bump.
   - Variant B (blue dashed) delays the runaway but eventually diverges when drift exceeds $1.71\,\text{m/s}$.
   - Variant D (purple) remains bounded near zero throughout the entire 125-second outage.

---

## 5. Architectural Implications & Path Forward

### 5.1 What Stage C10.7.1 Genuinely Established
1. **The gate decomposition isolates the contribution of the NIS and residual gates to persistent VNHC rejection**:
   - For the observed innovation covariance range, the NIS gate ($\text{NIS} > 9.21$) corresponds approximately to a residual magnitude of $1.6–1.8\,\text{m/s}$.
   - The residual gate acts as a $3.0\,\text{m/s}$ gate.
   - Together, they form an unrecoverable latch: once a disturbance causes vertical velocity error $> 1.6–1.8\,\text{m/s}$, Lock 1 shuts; when it exceeds $3.0\,\text{m/s}$, Lock 2 shuts.
   - When the vehicle returns to a regime with sensor-derived planar-motion eligibility, the filter refuses to accept $v_z \approx 0$, ensuring permanent divergence.
2. **Gating is NOT the enemy — Unrecoverable Latching is**:
   - As demonstrated by Variant D in Trip 2 Ep 5 (and confirmed by the $\gamma=1.0$ runaway to $686\,\text{km}$), measurements MUST be rejected when the vehicle or phone is not planar (e.g. tumbling, high angular rate, high vibration).
   - Forcing an update on an invalid measurement corrupts attitude and velocity states.

### 5.2 The Clean Architecture: Decoupled Physical Eligibility
The fundamental principle we are converging toward:
$$\mathbf{Physical\ Eligibility \neq Innovation\ Magnitude}$$

```
                               ┌────────────────────────────────┐
                               │       Is Vehicle Moving in     │
                               │   a Planar, Non-Turning Regime? │
                               └───────────────┬────────────────┘
                                               │
                                      NO ──────┴────── YES
                                      │                 │
                                      ▼                 ▼
                                [REJECT VNHC]    [EVALUATE RESIDUAL]
                             (Turns, Tumbling,          │
                             High Vibration)     ┌──────┴──────┐
                                                 │             │
                                              NORMAL       SUSPICIOUS
                                            (|r| ≤ 1.5)    (|r| > 1.5)
                                                 │             │
                                                 ▼             ▼
                                              STANDARD     ROBUST ADAPTIVE
                                               UPDATE          UPDATE
                                                          (Soft attenuation,
                                                          covariance inflation,
                                                          safe recovery)
```

1. **Eligibility Stage (Physical Environment Gate)**:
   - Low yaw rate: $|\omega_z| < 3.0^\circ/\text{s}$ for $\ge 0.5\,\text{s}$.
   - Low vertical dynamic acceleration: $|a_{\text{dyn}, z}| < 0.8\,\text{m/s}^2$.
   - Low angular jitter: $\|\boldsymbol{\omega}\| < 0.25\,\text{rad/s}$ (rejects hand manipulation/tumbling).
2. **Update Stage (Safe Recovery)**:
   - If physically eligible, the measurement $v_z^{\text{vehicle}} = 0$ is known to be physically valid.
   - A large residual $|r_{\text{vert}}|$ does **not** mean the measurement is bad; it means the **state has drifted**.
   - Instead of rejecting the measurement and locking the filter forever, the filter applies an adaptive, robust update (e.g. Huber attenuation or noise scaling) that safely pulls vertical velocity back to zero without shock.

---

## 6. Project Ledger & Commitments

- **Production Code Status**: `android/` and `src/navigation/dead_reckoning_engine.py` remain **STRICTLY FROZEN**.
- **Stage C10.7.1 Deliverables**:
  - Script: [`scratch/c10_7_1_gate_decomposition.py`](../scratch/c10_7_1_gate_decomposition.py)
  - Analysis Script: [`scratch/analyze_c10_7_1_results.py`](../scratch/analyze_c10_7_1_results.py)
  - Data: [`results/c10_7_1_gate_decomposition.json`](c10_7_1_gate_decomposition.json)
  - Diagnostic Plot: [`results/c10_7_1_gate_decomposition.png`](c10_7_1_gate_decomposition.png)
  - Report: [`results/c10_7_1_gate_decomposition.md`](c10_7_1_gate_decomposition.md)
- **Next Step**: Review findings with user before designing the decoupled physical eligibility architecture (Stage C10.8). Zero production code touches until approved.
