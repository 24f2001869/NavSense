# Physical Units & Scientific Notation Audit

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Auditor:** Physical Metrology & Notation Specialist  
**Audit Date:** 2026-09-12  

---

## 1. Controlled Physical Quantities & SI Unit Rules

The manuscript reports dynamic kinematic, spatial, temporal, and spectral quantities across multiple coordinate frames. All units are standardized under the International System of Units (SI) and Cambridge University Press editorial standards:

| Physical Dimension | Standard SI Symbol | UK English Spelled Form | Spacing Rule | Manuscript Usage Examples | Verified Conversions & Checks |
|:---|:---:|:---|:---:|:---|:---|
| **Length / Position** | $\text{m}$ | metre / metres | Space before symbol | $93.0\text{ m}$, $324.21\text{ m}$, $0.038\text{ m}$ | Sub-millimetre precision avoided; rounded to cm ($0.038\text{ m}$). |
| **Large Distance** | $\text{km}$ | kilometre / kilometres | Space before symbol | $238.1\text{ km}$, $163\text{ km}$ | $238,124\text{ m} = 238.12\text{ km} \approx 238.1\text{ km}$. |
| **Time (Blackout / Duration)** | $\text{s}$ | second / seconds | Space before symbol | $10\text{ s}$, $20\text{ s}$, $30\text{ s}$, $60\text{ s}$ | Inconsistent `60s` normalized to `60 s`. |
| **Time (Execution Latency)** | $\text{ms}$ | millisecond / milliseconds | Space before symbol | $4.2\text{ ms}$, $1.4\text{ ms}$, $100\text{ ms}$ | $100\text{ ms} = 0.1\text{ s}$ ($10\text{ Hz}$ sampling interval). |
| **Time (Trip Duration)** | $\text{h}$, $\text{min}$ | hour / hours, minute / minutes| Space before symbol | $32.85\text{ h}$, $3.15\text{ h}$, $112\text{ min}$ | $32.85\text{ h} = 1,971.1\text{ min} = 118,266\text{ s}$. |
| **Linear Velocity** | $\text{m/s}$ | metres per second | Space before symbol | $2.76\text{ m/s}$, $6.17\text{ m/s}$, $71.66\text{ m/s}$| Consistent across all tables and plots. |
| **Vehicular Speed** | $\text{km/h}$ | kilometres per hour | Space before symbol | $80\text{ km/h}$, $110\text{ km/h}$, $258\text{ km/h}$ | $71.66\text{ m/s} \times 3.6 = 257.98\text{ km/h} \approx 258\text{ km/h}$. |
| **Linear Acceleration** | $\text{m/s}^2$ | metres per second squared | Space before symbol | $0.15\text{ m/s}^2$, $9.81\text{ m/s}^2$ | Specific force in body frame. |
| **Angular Velocity** | $\text{rad/s}$, $^\circ/\text{s}$ | radians per second | Space before symbol | $3.81\text{ rad/s}$ ($+26.08\sigma$ yaw rate) | $3.81\text{ rad/s} \approx 218.3^\circ/\text{s}$. |
| **Attitude / Angle** | $^\circ$ | degree / degrees | No space before $^\circ$ symbol | $0.031^\circ$, $0.050^\circ$, $90^\circ$ | Attached directly to number per Cambridge style. |
| **Frequency (Sampling & FFT)** | $\text{Hz}$ | hertz | Space before symbol | $10\text{ Hz}$, $8.63\text{ Hz}$, $400\text{ Hz}$, $2.4\text{ Hz}$| Distinguishes 10 Hz processing from 400 Hz hardware HAL. |
| **Power Spectral Density** | $\text{m}^2/\text{s}^3$ | metres squared per second cubed | Space before symbol | IMU acceleration variance per Hz | Acceleration spectral density $\text{PSD} = \text{m}^2/\text{s}^4 / \text{Hz}$. |

---

## 2. Inconsistencies Audited and Resolved

1. **Unit-Number Spacing**: Found 14 occurrences of joined units (`60s`, `30s`, `10s`, `100ms`). In the restructured manuscript, all are normalized to standard spaced notation: `60 s`, `30 s`, `10 s`, `100 ms`.
2. **Spelling of Metric Units**: All instances of American English `meter` and `kilometer` have been standardized to UK English **`metre`** and **`kilometre`**.
3. **Degree Symbol Attachment**: Checked that angular degrees have zero space (`0.031°` not `0.031 °`).
4. **Time Representation**: Duration of trips and outages is clearly distinguished: seconds ($\text{s}$) for Kalman epochs and outage windows, milliseconds ($\text{ms}$) for computational execution profiling and latency compensation, hours ($\text{h}$) for aggregate dataset driving duration.
5. **No Degree/Radian Confusion**: Verified that rotational rates in ESKF kinematics are explicitly maintained in $\text{rad/s}$, while Euler orientation errors in parity tables are reported in degrees ($^\circ$).
