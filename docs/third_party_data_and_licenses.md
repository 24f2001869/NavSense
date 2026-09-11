# Third-Party Data, Open-Source Software, and Licenses

> **Repository:** `SIH26168-IDR`  
> **Author / Maintainer:** `24f2001869`  
> **Repository License:** MIT License (see [`LICENSE`](../LICENSE))  

This document details the provenance, ownership, licensing terms, and redistribution status of external datasets, libraries, and tools utilized in this research project.

---

## 1. External Benchmark Dataset: IO-VNBD

### Citation & Ownership
- **Dataset Name:** IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)
- **Creators / Authors:** Uche Onyekpe, Vasile Palade, Spyros Kanarachos, Arkadiusz Szkolnik (Coventry University, UK)
- **Primary Publication:**  
  U. Onyekpe, V. Palade, S. Kanarachos, A. Szkolnik, *"IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning"*, *Data in Brief*, Vol. 35, Art. 106885, 2021.  
  DOI: [`10.1016/j.dib.2021.106885`](https://doi.org/10.1016/j.dib.2021.106885)
- **Open Access Article:** [PMC7907232](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907232/) / [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S2352340921001694)
- **Official GitHub Repository:** [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)

### Redistribution Policy & Disclaimer
- **Not Redistributed:** The complete raw IO-VNBD dataset (~2.34 GB) is **NOT** bundled or redistributed in this GitHub repository.
- **Acquisition:** Users and researchers wishing to replicate our experiments must clone or download the dataset directly from the authors' official repository: [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD).
- **Data Terms:** As documented in *Data in Brief*, the dataset is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. The creators retain original copyright of the raw measurements.

---

## 2. Core Open-Source Python Dependencies

The Python scientific and machine learning stack used across this repository operates under standard permissive open-source licenses:

| Library | Primary Use Case | Upstream License |
| :--- | :--- | :--- |
| **PyTorch** (`torch`) | Neural network training, Temporal Convolutional Network (TCN) layers | Modified BSD |
| **ONNX Runtime** (`onnxruntime`) | Cross-platform model inference & Android runtime engine | MIT License |
| **NumPy** (`numpy`) | Vectorized mathematical operations, matrix transformations | BSD 3-Clause |
| **SciPy** (`scipy`) | Signal processing, Butterworth filtering, Fast Fourier Transforms (FFT) | BSD 3-Clause |
| **Pandas** (`pandas`) | Telemetry parsing, CSV alignment, tabular aggregation | BSD 3-Clause |
| **Scikit-Learn** (`scikit-learn`) | Feature scaling (`StandardScaler`), metric computations | BSD 3-Clause |
| **Matplotlib** (`matplotlib`) | Evaluation trajectory plotting, frequency spectrum visualization | PSF-based License |
| **PyYAML** (`yaml`) | Configuration management | MIT License |

---

## 3. Android Mobile Application Components

The Android application in `android/` leverages the following libraries:

| Dependency | Purpose | License |
| :--- | :--- | :--- |
| **ONNX Runtime Android** (`com.microsoft.onnxruntime:onnxruntime-android`) | Native mobile inference of TCN velocity models | MIT License |
| **EJML (Efficient Java Matrix Library)** (`org.ejml:ejml-simple`) | High-performance linear algebra for the 15-state Java ESKF filter | Apache 2.0 |
| **OSMDroid** (`org.osmdroid:osmdroid-android`) | Offline / OpenStreetMap tile rendering for dead-reckoned trajectory visualization | Apache 2.0 |
| **AndroidX Core & AppCompat** | Modern Android architectural foundations | Apache 2.0 |

---

## 4. Proprietary & Field Data Notice

- Telemetry CSV files recorded during development (located in `data/field/`) were captured using real smartphone sensors (OnePlus Nord CE 3 / Android 14) on university campus premises.
- These sensor traces contain no personally identifiable information (PII) or user credentials and are provided under the repository's MIT License solely for sensor noise verification and timing diagnostics.
- They are strictly qualified as **hardware and motion stress tests**, not vehicle navigation validation.
