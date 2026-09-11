"""
SIH26168 - Stage C: 3D Attitude Foundation
Module: src/navigation/attitude.py

A standalone 3D attitude estimator that processes the complete 3-axis smartphone
gyroscope vector (and optional accelerometer gravity reference) using a continuous
quaternion and Direction Cosine Matrix (DCM) representation.

===============================================================================
MATHEMATICAL FOUNDATION & COORDINATE CONTRACT SPECIFICATION
===============================================================================

1. COORDINATE FRAMES:
   ------------------
   a. Phone Sensor Frame (F_p):
      - As defined by Android AndroSensor and the IO-VNBD benchmark:
        * X_p: Across the smartphone screen (width, pointing to the right in portrait).
        * Y_p: Along the smartphone screen (length, pointing toward the top in portrait).
        * Z_p: Normal to the smartphone screen (pointing out of the front display).
      - Tri-axial angular rate: omega_p = [gyro_x, gyro_y, gyro_z]^T (rad/s).
      - Tri-axial specific force: f_p = [accel_x, accel_y, accel_z]^T (m/s^2).
      - Note: When the phone is placed display-up, f_p approx [0, 0, +9.81]^T m/s^2.

   b. Vehicle Body Frame (F_v):
      - Standard ISO 8855 / SAE vehicle coordinate system:
        * X_v: Longitudinal Forward (along the car's heading direction).
        * Y_v: Lateral Left (perpendicular to travel, port side).
        * Z_v: Vertical Up (normal to road surface, pointing toward vehicle roof).

   c. Navigation Frame (F_n):
      - Local East-North-Up (ENU) geodetic tangent plane:
        * X_n: Geodetic East (E).
        * Y_n: Geodetic North (N).
        * Z_n: Local Up (U, opposite gravity).
      - Vehicle Azimuth / Heading (psi):
        * Clockwise angle from True North (Y_n) toward East (X_n):
          0 deg = North, 90 deg = East, 180 deg = South, 270 deg = West.
        * Forward unit vector in ENU: u_fwd^n = [sin(psi), cos(psi), 0]^T.

2. MOUNTING ROTATION (R_vp / R_pv):
   ---------------------------------
   - The phone is installed in the vehicle with a mounting orientation R_vp in SO(3),
     transforming vectors from Phone frame to Vehicle frame:
       v_v = R_vp @ v_p
   - CRITICAL REAL-WORLD REQUIREMENT:
     * If the phone is rigidly fixed: R_vp approx constant.
     * If the phone is in a loose cradle, moves, or vibrates: R_vp = R_vp(t).
     * Therefore, mounting is maintained as an explicit separate calibration quantity
       and is never hardcoded or conflated with navigation attitude.

3. QUATERNION CONVENTION & ALGEBRA:
   ---------------------------------
   - Representation: Hamilton unit quaternion q = [q_w, q_x, q_y, q_z]^T,
     where q = q_w + q_x*i + q_y*j + q_z*k with i^2 = j^2 = k^2 = i*j*k = -1.
   - Rotation direction:
     q_nv represents the rotation from Vehicle Body frame (F_v) to Navigation frame (F_n).
   - Coordinate Transformation (Passive):
     A vector v_v in vehicle frame is expressed in navigation frame F_n by:
       v_n = C_v^n @ v_v = q_nv * [0, v_v]^T * q_nv^*
     where Direction Cosine Matrix C_v^n is:
       C_v^n = [
         [1 - 2(qy^2 + qz^2),   2(qx*qy - qw*qz),     2(qx*qz + qw*qy)],
         [2(qx*qy + qw*qz),     1 - 2(qx^2 + qz^2),   2(qy*qz - qw*qx)],
         [2(qx*qz - qw*qy),     2(qy*qz + qw*qx),     1 - 2(qx^2 + qy^2)]
       ]

4. PROPAGATION KINEMATICS (3D GYRO INTEGRATION):
   ----------------------------------------------
   - Given 3D angular rate in vehicle frame omega_v = [omega_vx, omega_vy, omega_vz]^T:
       Delta_theta = omega_v * dt
       theta = ||Delta_theta||
   - Incremental rotation quaternion:
       Delta_q = [cos(theta / 2), (Delta_theta / theta) * sin(theta / 2)]^T  (theta > 0)
   - Quaternion update:
       q_nv_{k} = q_nv_{k-1} (x) Delta_q
   - Strict unit normalization at every time step:
       q_nv_{k} <- q_nv_{k} / ||q_nv_{k}||

5. GRAVITY LEVELING REFERENCE (ROLL/PITCH CORRECTION ONLY):
   --------------------------------------------------------
   - Specific force under low dynamic acceleration measures reaction to gravity:
       g_meas^n = [0, 0, +g]^T  (pointing Up in ENU).
   - In vehicle frame, predicted gravity direction is:
       g_pred^v = (C_v^n)^T @ [0, 0, 1]^T = [C_{31}, C_{32}, C_{33}]^T
   - Tilt error vector:
       e_tilt = f_hat^v (x) g_pred^v
   - MATHEMATICAL PROOF OF YAW NON-OBSERVABILITY FROM GRAVITY:
       g_pred^v . e_tilt = g_pred^v . (f_hat^v (x) g_pred^v) = 0.
     The tilt error e_tilt is strictly perpendicular to the vertical axis g_pred^v!
     Therefore, accelerometer leveling provides ZERO observability of vehicle yaw (psi).
     Gravity is mathematically used ONLY to bound roll and pitch drift.
===============================================================================
"""

from typing import Dict, Optional, Tuple
import numpy as np


class AttitudeEstimator3D:
    """
    Continuous 3D Attitude Estimator tracking attitude quaternion q_nv and DCM C_v^n.
    Separates 3D rotational dynamics (pitch, roll, yaw) using full 3-axis gyro integration.
    """

    def __init__(
        self,
        init_heading_deg: float = 0.0,
        init_pitch_deg: float = 0.0,
        init_roll_deg: float = 0.0,
        R_vp: Optional[np.ndarray] = None
    ) -> None:
        """
        Initializes the 3D attitude estimator.

        Args:
            init_heading_deg: Initial vehicle heading in degrees (clockwise from North).
            init_pitch_deg: Initial vehicle pitch in degrees (positive nose up).
            init_roll_deg: Initial vehicle roll in degrees (positive right side down).
            R_vp: 3x3 mounting rotation matrix transforming Phone frame to Vehicle frame:
                  v_v = R_vp @ v_p. If None, defaults to Identity (phone aligned with vehicle).
        """
        if R_vp is not None:
            self.R_vp = np.asarray(R_vp, dtype=np.float64)
            if self.R_vp.shape != (3, 3):
                raise ValueError(f"R_vp must be a 3x3 matrix, got {self.R_vp.shape}")
        else:
            self.R_vp = np.eye(3, dtype=np.float64)

        # Initialize attitude quaternion q_nv from Euler angles
        self.q_nv = self._euler_to_quat(init_heading_deg, init_pitch_deg, init_roll_deg)
        self.q_nv = self._normalize_quat(self.q_nv)

        # Gravity constant (m/s^2)
        self.g_val = 9.80665

        # Tracking metrics
        self.last_omega_v = np.zeros(3, dtype=np.float64)
        self.last_f_v = np.zeros(3, dtype=np.float64)

    def set_mounting_matrix(self, R_vp: np.ndarray) -> None:
        """
        Updates the phone-to-vehicle mounting matrix R_vp.
        Used when the mounting calibration is determined or updated dynamically.
        """
        R = np.asarray(R_vp, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R_vp must be 3x3, got {R.shape}")
        self.R_vp = R

    def predict(
        self,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        dt: float
    ) -> np.ndarray:
        """
        Propagates 3D quaternion attitude using the complete 3-axis gyro vector.

        Args:
            gyro_x: Angular rate around phone X-axis (rad/s).
            gyro_y: Angular rate around phone Y-axis (rad/s).
            gyro_z: Angular rate around phone Z-axis (rad/s).
            dt: Timestep duration (seconds).

        Returns:
            Normalized attitude quaternion q_nv = [qw, qx, qy, qz].
        """
        if dt <= 0.0:
            return self.q_nv

        # 1. Full 3D angular rate in phone frame
        omega_p = np.array([gyro_x, gyro_y, gyro_z], dtype=np.float64)

        # 2. Transform into vehicle body frame: omega_v = R_vp @ omega_p
        omega_v = self.R_vp @ omega_p
        self.last_omega_v = omega_v

        # 3. Compute incremental rotation vector Delta_theta = omega_v * dt
        delta_theta = omega_v * dt
        theta = np.linalg.norm(delta_theta)

        # 4. Form incremental quaternion Delta_q
        if theta > 1e-12:
            s = np.sin(0.5 * theta) / theta
            dq = np.array([np.cos(0.5 * theta), s * delta_theta[0], s * delta_theta[1], s * delta_theta[2]])
        else:
            # Taylor series approximation for small angles
            dq = np.array([1.0 - (theta**2) / 8.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]])

        # 5. Quaternion propagation: q_nv <- q_nv (x) Delta_q
        self.q_nv = self._quat_mult(self.q_nv, dq)

        # 6. Strict unit quaternion normalization
        self.q_nv = self._normalize_quat(self.q_nv)

        return self.q_nv

    def update_gravity(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        ka: float = 0.02,
        accel_gate: float = 1.5
    ) -> bool:
        """
        Uses accelerometer specific force to level roll and pitch attitude.
        Explicitly does NOT correct yaw heading, as gravity has no horizontal component.

        Args:
            accel_x: Specific force along phone X-axis (m/s^2).
            accel_y: Specific force along phone Y-axis (m/s^2).
            accel_z: Specific force along phone Z-axis (m/s^2).
            ka: Complementary leveling gain (0 < ka << 1).
            accel_gate: Maximum allowable deviation from 1g (m/s^2) to reject dynamic motion.

        Returns:
            True if gravity update was accepted, False if gated out by dynamic acceleration.
        """
        f_p = np.array([accel_x, accel_y, accel_z], dtype=np.float64)
        f_v = self.R_vp @ f_p
        self.last_f_v = f_v

        f_norm = np.linalg.norm(f_v)
        # Gate check: only use accelerometer when magnitude is close to 1g (quasi-static or low dynamic accel)
        if abs(f_norm - self.g_val) > accel_gate:
            return False

        f_hat_v = f_v / f_norm

        # Predicted gravity in vehicle frame: C_v^n.T @ [0, 0, 1]^T = Row 2 of C_v^n
        C = self.get_dcm()
        g_pred_v = C[2, :]

        # Tilt error in vehicle frame: e_tilt = f_hat_v x g_pred_v
        e_tilt = np.cross(f_hat_v, g_pred_v)

        # Incremental tilt correction quaternion (strictly orthogonal to g_pred_v)
        corr_vec = ka * e_tilt
        corr_norm = np.linalg.norm(corr_vec)
        if corr_norm > 1e-12:
            s = np.sin(0.5 * corr_norm) / corr_norm
            dq_tilt = np.array([np.cos(0.5 * corr_norm), s * corr_vec[0], s * corr_vec[1], s * corr_vec[2]])
        else:
            dq_tilt = np.array([1.0, 0.5 * corr_vec[0], 0.5 * corr_vec[1], 0.5 * corr_vec[2]])

        self.q_nv = self._quat_mult(self.q_nv, dq_tilt)
        self.q_nv = self._normalize_quat(self.q_nv)
        return True

    def get_dcm(self) -> np.ndarray:
        """
        Returns the 3x3 Direction Cosine Matrix C_v^n transforming vectors
        from Vehicle frame to Navigation (ENU) frame: v_n = C_v^n @ v_v.
        """
        w, x, y, z = self.q_nv
        return np.array([
            [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z),         2.0*(x*z + w*y)],
            [2.0*(x*y + w*z),         1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)],
            [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x**2 + y**2)]
        ], dtype=np.float64)

    def get_yaw_deg(self) -> float:
        """
        Computes vehicle heading (azimuth) psi in degrees:
        Clockwise angle from True North toward East [0.0, 360.0).
        """
        C = self.get_dcm()
        # In ENU: Axis 0 of vehicle frame is forward vector: u_fwd^n = [C[0, 0], C[1, 0], C[2, 0]]^T
        # East component is C[0, 0], North component is C[1, 0]
        # psi = atan2(East, North)
        psi_rad = np.arctan2(C[0, 0], C[1, 0])
        return float(np.degrees(psi_rad) % 360.0)

    def get_pitch_deg(self) -> float:
        """
        Computes vehicle pitch angle theta in degrees:
        Elevation angle of vehicle forward vector above horizontal plane [-90.0, +90.0].
        Positive nose up.
        """
        C = self.get_dcm()
        # Up component of forward vector is C[2, 0]
        sin_pitch = np.clip(C[2, 0], -1.0, 1.0)
        return float(np.degrees(np.arcsin(sin_pitch)))

    def get_roll_deg(self) -> float:
        """
        Computes vehicle roll angle phi in degrees:
        Rotation about longitudinal forward axis [-180.0, +180.0].
        Positive right side down.
        """
        C = self.get_dcm()
        roll_rad = np.arctan2(C[2, 1], C[2, 2])
        return float(np.degrees(roll_rad))

    def get_quaternion(self) -> np.ndarray:
        """Returns the attitude quaternion [qw, qx, qy, qz]."""
        return self.q_nv.copy()

    def get_quaternion_norm(self) -> float:
        """Returns the Euclidean norm of the attitude quaternion."""
        return float(np.linalg.norm(self.q_nv))

    def get_attitude_state(self) -> Dict[str, float]:
        """Returns comprehensive attitude telemetry dictionary."""
        return {
            'yaw_deg': self.get_yaw_deg(),
            'pitch_deg': self.get_pitch_deg(),
            'roll_deg': self.get_roll_deg(),
            'q_norm': self.get_quaternion_norm(),
            'omega_mag_rads': float(np.linalg.norm(self.last_omega_v))
        }

    # =========================================================================
    # Internal Quaternion Utilities
    # =========================================================================
    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton quaternion multiplication: q = q1 (x) q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)

    @classmethod
    def _euler_to_quat(cls, heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """
        Constructs quaternion q_nv from heading (clockwise from North), pitch, and roll.
        """
        psi = np.radians(heading_deg)
        theta = np.radians(pitch_deg)
        phi = np.radians(roll_deg)

        # In ENU:
        # Rotation by psi (clockwise from North toward East) about Z_n:
        # u_fwd = [sin(psi), cos(psi), 0], u_lat = [-cos(psi), sin(psi), 0], u_up = [0, 0, 1]
        R_yaw = np.array([
            [np.sin(psi), -np.cos(psi), 0.0],
            [np.cos(psi),  np.sin(psi), 0.0],
            [0.0,          0.0,         1.0]
        ])

        # Pitch rotation (about lateral left Y_v axis)
        cp, sp = np.cos(theta), np.sin(theta)
        R_pitch = np.array([
            [cp,  0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp]
        ])

        # Roll rotation (about longitudinal forward X_v axis)
        cr, sr = np.cos(phi), np.sin(phi)
        R_roll = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr]
        ])

        C = R_yaw @ R_pitch @ R_roll
        return cls._dcm_to_quat(C)

    @staticmethod
    def _dcm_to_quat(C: np.ndarray) -> np.ndarray:
        """Converts a 3x3 rotation matrix to a Hamilton unit quaternion."""
        tr = np.trace(C)
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (C[2, 1] - C[1, 2]) / s
            qy = (C[0, 2] - C[2, 0]) / s
            qz = (C[1, 0] - C[0, 1]) / s
        elif (C[0, 0] > C[1, 1]) and (C[0, 0] > C[2, 2]):
            s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            qw = (C[2, 1] - C[1, 2]) / s
            qx = 0.25 * s
            qy = (C[0, 1] + C[1, 0]) / s
            qz = (C[0, 2] + C[2, 0]) / s
        elif C[1, 1] > C[2, 2]:
            s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            qw = (C[0, 2] - C[2, 0]) / s
            qx = (C[0, 1] + C[1, 0]) / s
            qy = 0.25 * s
            qz = (C[1, 2] + C[2, 1]) / s
        else:
            s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            qw = (C[1, 0] - C[0, 1]) / s
            qx = (C[0, 2] + C[2, 0]) / s
            qy = (C[1, 2] + C[2, 1]) / s
            qz = 0.25 * s

        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        return q / np.linalg.norm(q)
