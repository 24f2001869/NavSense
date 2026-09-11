# IO-VNBD Dataset

## 1. Official Source & Provenance

* **Dataset Name:** IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)
* **Official GitHub Repository:** [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
* **Dataset Paper:**  
  U. Onyekpe, V. Palade, S. Kanarachos, A. Szkolnik, *"IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning"*, *Data in Brief*, 35, 106885, 2021.
* **DOI:** [`10.1016/j.dib.2021.106885`](https://doi.org/10.1016/j.dib.2021.106885)
* **Open-Access Article:** [PMC7907232](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/)
* **ScienceDirect Publication:** [ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S2352340921001694)

> [!IMPORTANT]
> **Data Redistribution Policy:**  
> The complete raw IO-VNBD dataset (~2.34 GB) is **not redistributed in this GitHub repository**.  
> Researchers wishing to reproduce our experiments must download the dataset directly from the official upstream repository above.

---

## 2. What the Dataset Contains

IO-VNBD provides comprehensive synchronized and unsynchronized data streams captured simultaneously from consumer smartphones and professional vehicle instrumentation during naturalistic driving across diverse geographic environments:

### Smartphone Sensor Streams
* **Tri-axial Accelerometer:** Specific force along phone body axes ($X, Y, Z$)
* **Tri-axial Gyroscope:** Angular rotation rates ($\omega_x, \omega_y, \omega_z$)
* **Magnetometer:** Ambient magnetic field vector ($\mu\text{T}$)
* **Gravity Sensor:** Estimated gravity vector in device frame
* **Smartphone GNSS Fixes:** Consumer-grade GPS coordinates (latitude, longitude, altitude), speed, and heading

### Vehicle & Reference Instrumentation
* **Vehicle CAN Bus:** Transmission speed, engine RPM, steering angle, throttle/brake commands
* **Wheel Speed Sensors:** Individual wheel speed encoder ticks / calibrated wheel speeds
* **Reference GNSS/INS:** High-precision dual-antenna RTK-GNSS and tactical-grade inertial navigation reference for ground-truth position and velocity

---

## 3. Sampling Rates & Signal Synchronization

* **Smartphone IMU:** Recorded at ~100 Hz native sampling rate.
* **Vehicle CAN / Wheel Speed:** Logged at 10–50 Hz native sampling rate.
* **Smartphone GPS:** Logged at 1 Hz.
* **Internal Synchronization:** The dataset authors aligned phone and vehicle clocks using hardware timestamp correlation and GPS time tags.
* **Project Preprocessing:** For this research, all sensor channels are uniformly interpolated and resampled to **10.0 Hz** using strictly causal anti-aliasing filters.

---

## 4. Dataset Structure: Synchronised vs. Unsynchronised

The official dataset distribution provides two archives:
1. **Unsynchronised V and S Dataset:** Raw independently logged vehicle (`V`) and smartphone (`S`) files.
2. **Synchronised V abd S datasets:** Pre-aligned paired recordings organized into:
   * **Categorised IOVNB Dataset:** Grouped by driving regime and driver behavior:
     - `Motorway` (High-speed open highways, Driver E)
     - `Dense urban` (Frequent stops, canyon multipath, Driver E)
     - `Suburban` (Arterial roads, moderate turns, Driver E)
     - `Winding` (Mountainous terrain, continuous turning dynamics, Driver E)
     - Additional drivers `A`, `B`, `C`, `D` (`S`, `M`, `St`, `Y` routes)
   * **Uncategorised IOVNB Dataset:** Chronological unclassified routes.

---

## 5. How This Project Used IO-VNBD

Our experiments used the **Synchronised Categorised IOVNB Dataset** across all 4 operational regimes to ensure zero cross-trip data contamination.

### Split Design (Strict Whole-Trip Partitioning)
To ensure zero data leakage, splits were made strictly at the **whole-trip level** (no overlapping time windows between splits):

* **Total Available Matched Trips:** 64 categorized trips.
* **Training Set (39 Trips):**
  - Suburban: 20 trips (`Vta01` to `Vta20`)
  - Dense Urban: 8 trips (`Vtb01` to `Vtb08`)
  - Winding Mountain: 10 trips (`Vw01` to `Vw11`, excluding `Vw03` due to logging gap)
  - Motorway: 1 trip (`V-Vfa01`)
* **Validation Set (6 Trips):** Used for early stopping and hyperparameter tuning (`Vta15`–`Vta18`, `Vtb07`, `Vw09`).
* **Held-Out Test Set (19 Trips):** Completely unseen during model training, tuning, and selection:
  - **Motorway (1 trip):** `V-Vfa02` (446 rolling evaluation windows, 163 km)
  - **Suburban (8 trips):** `Vta21`, `Vta22`, `Vta23`, `Vta24`, `Vta25`, `Vta26`, `Vta27`, `Vta28`
  - **Dense Urban (4 trips):** `Vtb09`, `Vtb10`, `Vtb11`, `Vtb12`
  - **Winding Mountain (6 trips):** `Vw12`, `Vw13`, `Vw14a`, `Vw14b`, `Vw15` (stationary control), `Vw16a`

### Trip Duration & Usable Evaluation Windows
Because rolling blackout evaluation requires trips to be longer than the blackout window plus the initialization buffer:
* **10-second Outage Horizon:** **18 / 19 trips usable** (`Vtb10` is 9.6 s, excluded).
* **20-second Outage Horizon:** **17 / 19 trips usable** (`Vw13` is 18.5 s, excluded).
* **30-second Outage Horizon:** **16 / 19 trips usable** (`Vtb11` is 26.2 s, excluded).
* **60-second Outage Horizon:** **13 / 19 trips usable** (`Vtb09` [35.3 s], `Vtb12` [34.8 s], `Vta25` [54.7 s] excluded because total duration $< 70\text{ s}$).

---

## 6. Local Installation & Setup Instructions

To download and configure the dataset for local reproduction:

### Step 1: Clone the Official Repository
Run from the root of this project:

```bash
mkdir -p data/raw
cd data/raw
git clone https://github.com/onyekpeu/IO-VNBD.git IO-VNBD-repo
cd ../..
```

### Step 2: Unzip the Synchronised Dataset
Extract the synchronized archive:

```bash
cd data/raw/IO-VNBD-repo
unzip "Synchronised V abd S datasets.zip"
cd ../../..
```

### Step 3: Verify Dataset Ingestion
Run the automated discovery verification script:

```bash
python -c "
from src.data.dataset_builder import IOVNBDDatasetBuilder
builder = IOVNBDDatasetBuilder()
trips = builder.discover_trips()
print(f'Discovered {len(trips)} valid synchronized trips.')
"
```

Expected output:
```text
Discovered 64 valid synchronized trips.
```
