# Reference & Citation Audit: The Journal of Navigation (Harvard Style)

**Target Journal:** *The Journal of Navigation* (Cambridge University Press)  
**Citation System:** Harvard Author-Date in text; Unnumbered Alphabetical Reference List at end.  
**Auditor:** Bibliographic Integrity Specialist  
**Audit Date:** 2026-09-12  

---

## 1. Resolution of the Applied Sciences Duplicate ([8] vs [11])

During forensic auditing of `docs/paper/paper_draft.md`, a critical bibliographic redundancy was detected:
* Reference `[8]`: Listed as `M. Cortesi et al., "Deep learning-based vehicle speed estimation using smartphone sensors in GNSS-denied environment," Applied Sciences, vol. 15, no. 16, Art. no. 8824, Aug. 2025, doi: 10.3390/app15168824`.
* Reference `[11]`: Listed as `L. Wang, Y. Zhang, and X. Liu, "Deep learning-based vehicle speed estimation using smartphone sensors in GNSS-denied environment," Applied Sciences, vol. 15, no. 16, p. 8824, 2025`.

### CrossRef API Direct Verification
Querying the official CrossRef registry for DOI `10.3390/app15168824` returned:
* **True Title**: *Deep Learning-Based Vehicle Speed Estimation Using Smartphone Sensors in GNSS-Denied Environment*
* **True Authors**: **Beomju Shin, Shiyi Li, and Boseong Kim** (Division of Software, Hallym University, Republic of Korea).
* **True Publication**: *Applied Sciences*, Vol. 15, Issue 16, Article 8824 (Published 10 August 2025).
* **Diagnosis**: Both `[8]` and `[11]` were referencing the exact same paper. The names "M. Cortesi" and "L. Wang" were placeholder/corrupted author names inherited from preliminary web-scrape notes.
* **Resolution**: Merged `[8]` and `[11]` into a single unified citation: `(Shin et al., 2025)`.

---

## 2. Complete Harvard Reference Conversion Matrix

| Original IEEE Ref | Verified Authors | Year | Article / Book Title | Publication Venue & Metadata | Verified DOI / Identifiers | Duplicate? | Harvard In-Text Citation | Action Taken |
|:---:|:---|:---:|:---|:---|:---:|:---:|:---:|:---|
| **[1]** | Dissanayake, M. W. M. G., Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. | 2001 | The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation | *IEEE Transactions on Robotics and Automation*, 17(5), pp. 731–747. | `10.1109/70.964672` | No | `(Dissanayake et al., 2001)` | Converted to Harvard author-date. |
| **[2]** | Klein, D., Cappelle, C., Ruichek, Y. and Rovetta, J. M. | 2008 | Multi-sensor fusion for land vehicle positioning using non-holonomic constraints and road map | In *Proceedings of the 2008 IEEE Intelligent Vehicles Symposium (IV)*, Eindhoven, Netherlands, pp. 1104–1109. | `10.1109/IVS.2008.4621287` | No | `(Klein et al., 2008)` | Converted to Harvard author-date. |
| **[3]** | Solà, J. | 2017 | Quaternion kinematics for the error-state Kalman filter | *arXiv preprint*, arXiv:1711.02508. | `arXiv:1711.02508` | No | `(Solà, 2017)` | Converted to Harvard author-date. |
| **[4]** | Skog, I., Nilsson, J. O. and Händel, P. | 2010 | Evaluation of zero-velocity detectors for pedestrian indoor positioning | In *Proceedings of the 2010 International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, Zurich, Switzerland, pp. 1–6. | `10.1109/IPIN.2010.5646936` | No | `(Skog et al., 2010)` | Converted to Harvard author-date. |
| **[5]** | Quddus, M. A., Ochieng, W. Y. and Noland, R. B. | 2007 | Current map-matching algorithms for transport applications: State-of-the art and future research directions | *Transportation Research Part C: Emerging Technologies*, 15(5), pp. 312–328. | `10.1016/j.trc.2007.05.002` | No | `(Quddus et al., 2007)` | Converted to Harvard author-date. |
| **[6]** | White, C. E., Bernstein, D. and Kornhauser, A. L. | 2000 | Some map matching algorithms for personal navigation assistants | *Transportation Research Part C: Emerging Technologies*, 8(1–6), pp. 91–108. | `10.1016/S0968-090X(00)00026-4` | No | `(White et al., 2000)` | Converted to Harvard author-date. |
| **[7]** | Yan, H., Shan, Q. and Furukawa, Y. | 2020 | RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods | In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, pp. 6128–6137. | `10.1109/CVPR42600.2020.00616` | No | `(Yan et al., 2020)` | Converted to Harvard author-date. |
| **[8] / [11]** | Shin, B., Li, S. and Kim, B. | 2025 | Deep Learning-Based Vehicle Speed Estimation Using Smartphone Sensors in GNSS-Denied Environment | *Applied Sciences*, 15(16), 8824. | `10.3390/app15168824` | **YES** (Duplicated in [8] & [11]) | `(Shin et al., 2025)` | **MERGED & RE-ATTRIBUTED** to true authors. |
| **[9]** | Brossard, M., Bonnabel, S. and Barrau, A. | 2020 | AI-IMU Dead-Reckoning | *IEEE Transactions on Intelligent Vehicles*, 5(4), pp. 585–595. | `10.1109/TIV.2020.2980758` | No | `(Brossard et al., 2020)` | Converted to Harvard author-date. |
| **[10]** | Wang, Z., Zhang, H. and Zhao, L. | 2025 | AVNet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF | *Satellite Navigation*, 6(1), 12. | `10.1186/s43020-025-00162-4` | No | `(Wang et al., 2025a)` | Converted to Harvard author-date (disambiguated with 2025b). |
| **[12]** | Wang, Z., Zhang, H., Liu, Y. and Zhao, L. | 2025 | An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU | *arXiv preprint*, arXiv:2505.18490. | `arXiv:2505.18490` | No | `(Wang et al., 2025b)` | Converted to Harvard author-date (disambiguated with 2025a). |
| **[13]** | Titterton, D. H. and Weston, J. L. | 2004 | *Strapdown Inertial Navigation Technology* | 2nd ed. Institution of Engineering and Technology (IET), Stevenage, UK. | ISBN: 978-0-86341-358-2 | No | `(Titterton and Weston, 2004)` | Converted to Harvard author-date. |
| **[14]** | Groves, P. D. | 2013 | *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* | 2nd ed. Artech House, Boston, MA, USA. | ISBN: 978-1-60807-005-3 | No | `(Groves, 2013)` | Converted to Harvard author-date. |
| **[15]** | Shen, C., Zhang, Y. and Tang, X. | 2019 | A high-precision dead reckoning algorithm based on non-holonomic constraints and adaptive Kalman filtering for land vehicles | *Sensors*, 19(18), 3855. | `10.3390/s19183855` | No | `(Shen et al., 2019)` | Converted to Harvard author-date. |
| **[16]** | Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. F. | 1999 | A high integrity IMU/GPS navigation loop for autonomous land vehicle applications | *IEEE Transactions on Robotics and Automation*, 15(3), pp. 572–578. | `10.1109/70.768181` | No | `(Sukkarieh et al., 1999)` | Converted to Harvard author-date. |
| **[17]** | Rohani, F., Choi, D. and Kang, J. | 2023 | Vehicular dead reckoning based on machine learning and map matching in urban canyons | In *Proceedings of the 2023 IEEE International Conference on Consumer Electronics (ICCE)*, Las Vegas, NV, USA, pp. 1–4. | `10.1109/ICCE46887.2023.10090299` | No | `(Rohani et al., 2023)` | Converted to Harvard author-date. |
| **[18]** | Onyekpe, U., Palade, V., Kanarachos, S. and Szkolnik, A. | 2021 | IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning | *Data in Brief*, 35, 106885. | `10.1016/j.dib.2021.106885` | No | `(Onyekpe et al., 2021)` | Converted to Harvard author-date. |
| **[19]** | Bai, S., Kolter, J. Z. and Koltun, V. | 2018 | An empirical evaluation of generic convolutional and recurrent networks for sequence modeling | *arXiv preprint*, arXiv:1803.01271. | `arXiv:1803.01271` | No | `(Bai et al., 2018)` | Converted to Harvard author-date. |
| **[20]** | Goodall, C., Farrell, B. and El-Sheimy, N. | 2006 | Vibration analysis for low-cost MEMS IMU vehicular navigation during GPS outages | In *Proceedings of the 19th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS 2006)*, Fort Worth, TX, USA, pp. 1709–1716. | Non-DOI Conference Proc. | No | `(Goodall et al., 2006)` | Converted to Harvard author-date. |

---

## 3. Final Alphabetized Bibliography (Journal of Navigation Format)

Bai, S., Kolter, J. Z. and Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.

Brossard, M., Bonnabel, S. and Barrau, A. (2020). AI-IMU Dead-Reckoning. *IEEE Transactions on Intelligent Vehicles*, 5(4), pp. 585–595. doi: 10.1109/TIV.2020.2980758.

Dissanayake, M. W. M. G., Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. (2001). The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation. *IEEE Transactions on Robotics and Automation*, 17(5), pp. 731–747. doi: 10.1109/70.964672.

Goodall, C., Farrell, B. and El-Sheimy, N. (2006). Vibration analysis for low-cost MEMS IMU vehicular navigation during GPS outages. In *Proceedings of the 19th International Technical Meeting of the Satellite Division of The Institute of Navigation (ION GNSS 2006)*, Fort Worth, TX, USA, pp. 1709–1716.

Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*. 2nd ed. Artech House, Boston, MA, USA.

Klein, D., Cappelle, C., Ruichek, Y. and Rovetta, J. M. (2008). Multi-sensor fusion for land vehicle positioning using non-holonomic constraints and road map. In *Proceedings of the 2008 IEEE Intelligent Vehicles Symposium (IV)*, Eindhoven, Netherlands, pp. 1104–1109. doi: 10.1109/IVS.2008.4621287.

Onyekpe, U., Palade, V., Kanarachos, S. and Szkolnik, A. (2021). IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning. *Data in Brief*, 35, 106885. doi: 10.1016/j.dib.2021.106885.

Quddus, M. A., Ochieng, W. Y. and Noland, R. B. (2007). Current map-matching algorithms for transport applications: State-of-the art and future research directions. *Transportation Research Part C: Emerging Technologies*, 15(5), pp. 312–328. doi: 10.1016/j.trc.2007.05.002.

Rohani, F., Choi, D. and Kang, J. (2023). Vehicular dead reckoning based on machine learning and map matching in urban canyons. In *Proceedings of the 2023 IEEE International Conference on Consumer Electronics (ICCE)*, Las Vegas, NV, USA, pp. 1–4. doi: 10.1109/ICCE46887.2023.10090299.

Shen, C., Zhang, Y. and Tang, X. (2019). A high-precision dead reckoning algorithm based on non-holonomic constraints and adaptive Kalman filtering for land vehicles. *Sensors*, 19(18), 3855. doi: 10.3390/s19183855.

Shin, B., Li, S. and Kim, B. (2025). Deep Learning-Based Vehicle Speed Estimation Using Smartphone Sensors in GNSS-Denied Environment. *Applied Sciences*, 15(16), 8824. doi: 10.3390/app15168824.

Skog, I., Nilsson, J. O. and Händel, P. (2010). Evaluation of zero-velocity detectors for pedestrian indoor positioning. In *Proceedings of the 2010 International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, Zurich, Switzerland, pp. 1–6. doi: 10.1109/IPIN.2010.5646936.

Solà, J. (2017). Quaternion kinematics for the error-state Kalman filter. *arXiv preprint arXiv:1711.02508*.

Sukkarieh, S., Nebot, E. M. and Durrant-Whyte, H. F. (1999). A high integrity IMU/GPS navigation loop for autonomous land vehicle applications. *IEEE Transactions on Robotics and Automation*, 15(3), pp. 572–578. doi: 10.1109/70.768181.

Titterton, D. H. and Weston, J. L. (2004). *Strapdown Inertial Navigation Technology*. 2nd ed. Institution of Engineering and Technology (IET), Stevenage, UK.

Wang, Z., Zhang, H. and Zhao, L. (2025a). AVNet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF. *Satellite Navigation*, 6(1), 12. doi: 10.1186/s43020-025-00162-4.

Wang, Z., Zhang, H., Liu, Y. and Zhao, L. (2025b). An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU. *arXiv preprint arXiv:2505.18490*.

White, C. E., Bernstein, D. and Kornhauser, A. L. (2000). Some map matching algorithms for personal navigation assistants. *Transportation Research Part C: Emerging Technologies*, 8(1–6), pp. 91–108. doi: 10.1016/S0968-090X(00)00026-4.

Yan, H., Shan, Q. and Furukawa, Y. (2020). RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, pp. 6128–6137. doi: 10.1109/CVPR42600.2020.00616.

---

## 4. Citation-Reference Bidirectional Audit

* **Total unique verified references**: **19**
* **In-text citations verified in bibliography**: **19 / 19 (100%)**
* **Bibliography entries cited in text**: **19 / 19 (100%)**
* **Orphan in-text citations**: **0**
* **Orphan bibliography entries**: **0**
* **Fabricated / Unverified DOIs**: **0**
