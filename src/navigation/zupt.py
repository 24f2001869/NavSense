"""
SIH26168 - Stage C7-A: Zero-Velocity Updates (ZUPT)
Module: src/navigation/zupt.py

Implements:
1. ZeroVelocityUpdate: 3D zero-velocity measurement update for 15-state ESKF.
2. CausalStationaryDetector: Strictly causal, phone-only IMU standstill detector
   with multi-feature thresholding and temporal persistence gating.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

from src.navigation.nhc import rotvec_to_quat, quat_mult


class ZeroVelocityUpdate:
    """
    Applies a 3D Zero-Velocity Update (ZUPT) measurement to an ESKF3D instance.
    When a vehicle is stationary:
        z_zupt = [0, 0, 0]^T (m/s) in Navigation frame (ENU)
        h(x) = v^n
        r_zupt = z_zupt - v_hat^n = -v_hat^n
        H_zupt = [0_{3x3}, I_{3x3}, 0_{3x3}, 0_{3x3}, 0_{3x3}] in R^{3 x 15}
    """

    def __init__(self, sigma_vel: float = 0.05):
        """
        Args:
            sigma_vel: Measurement uncertainty std for zero velocity (m/s).
        """
        self.sigma_vel = float(sigma_vel)
        self.R_zupt = (self.sigma_vel**2) * np.eye(3, dtype=np.float64)

    def update_eskf(
        self,
        eskf: Any,
        custom_R: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Applies the 3D ZUPT update to the ESKF state and covariance using
        Joseph-stabilized numerical formulation.

        Args:
            eskf: An instance of ESKF3D.
            custom_R: Optional 3x3 measurement covariance matrix.

        Returns:
            Dictionary containing innovation, corrections, and covariance telemetry.
        """
        vel_n = eskf.vel_n.copy()
        r_zupt = -vel_n  # Innovation: 0 - v_hat^n

        # Measurement Jacobian H in R^{3 x 15}
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3, dtype=np.float64)

        R = custom_R if custom_R is not None else self.R_zupt

        # Innovation covariance S in R^{3 x 3}
        # S = H P H^T + R = P_{3:6, 3:6} + R
        S = eskf.P[3:6, 3:6] + R
        inv_S = np.linalg.inv(S)

        # Kalman gain K in R^{15 x 3}
        # K = P H^T S^{-1} = P_{:, 3:6} S^{-1}
        K = eskf.P[:, 3:6] @ inv_S

        # State error correction delta_x in R^15
        delta_x = K @ r_zupt

        # Apply state corrections
        # 1. Position correction
        eskf.pos_n += delta_x[0:3]
        # 2. Velocity correction (resets velocity toward zero)
        eskf.vel_n += delta_x[3:6]
        # 3. Attitude correction (body-frame right multiplication)
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        # 4. Accelerometer bias correction (calibrates bias online)
        eskf.ba += delta_x[9:12]
        # 5. Gyroscope bias correction
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update: P = (I - KH) P (I - KH)^T + K R K^T
        IKH = np.eye(15, dtype=np.float64) - K @ H
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {
            'innovation': r_zupt,
            'delta_v': delta_x[3:6],
            'delta_ba': delta_x[9:12],
            'delta_bg': delta_x[12:15],
            'pre_speed': float(np.linalg.norm(vel_n)),
            'post_speed': float(np.linalg.norm(eskf.vel_n))
        }


class CausalStationaryDetector:
    """
    Strictly causal, phone-only standstill detector using multi-feature
    thresholding and temporal persistence gating.

    Requires zero future samples, zero GPS, and zero reference ground truth.
    Features used over a trailing window W:
      1. Acceleration magnitude variance sigma_a^2
      2. Gyroscope magnitude variance sigma_omega^2
      3. Acceleration norm offset from local gravity: |mean(||a||) - g|
      4. Trailing jerk RMS J_rms
    """

    def __init__(
        self,
        dt: float = 0.1,
        window_sec: float = 0.5,
        persist_sec: float = 0.4,
        th_acc_var: float = 0.08,      # (m/s^2)^2
        th_gyro_var: float = 0.005,    # (rad/s)^2
        th_grav_diff: float = 0.35,    # m/s^2
        th_jerk_rms: float = 8.0,      # m/s^3
        gravity: float = 9.80665
    ):
        """
        Args:
            dt: Sampling interval in seconds (0.1 s = 10 Hz).
            window_sec: Trailing feature window in seconds (e.g. 0.5 s = 5 epochs).
            persist_sec: Required consecutive standstill duration before confirming.
            th_acc_var: Maximum allowable acceleration magnitude variance.
            th_gyro_var: Maximum allowable gyroscope magnitude variance.
            th_grav_diff: Maximum deviation of mean acceleration norm from 1g.
            th_jerk_rms: Maximum allowable jerk RMS.
            gravity: Expected gravitational magnitude.
        """
        self.dt = dt
        self.w_epochs = max(3, int(round(window_sec / dt)))
        self.persist_epochs = max(2, int(round(persist_sec / dt)))
        self.th_acc_var = float(th_acc_var)
        self.th_gyro_var = float(th_gyro_var)
        self.th_grav_diff = float(th_grav_diff)
        self.th_jerk_rms = float(th_jerk_rms)
        self.gravity = float(gravity)

        # Internal causal rolling buffers
        self.consecutive_standstill = 0
        self.is_currently_stationary = False

    def reset(self):
        """Resets the detector state for a new outage or window."""
        self.consecutive_standstill = 0
        self.is_currently_stationary = False

    def update(
        self,
        acc_window: np.ndarray,
        gyro_window: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates standstill state at the current epoch given the trailing window
        of acceleration and angular velocity.

        Args:
            acc_window: (K, 3) trailing acceleration array (K >= w_epochs).
            gyro_window: (K, 3) trailing gyroscope array (K >= w_epochs).

        Returns:
            Dictionary with is_stationary (bool), raw_candidate (bool), and metrics.
        """
        # Take the most recent w_epochs
        acc_w = acc_window[-self.w_epochs:]
        gyro_w = gyro_window[-self.w_epochs:]

        if len(acc_w) < self.w_epochs:
            return {
                'is_stationary': False,
                'raw_candidate': False,
                'consecutive_count': 0,
                'acc_var': 0.0,
                'gyro_var': 0.0,
                'grav_diff': 0.0,
                'jerk_rms': 0.0
            }

        # 1. Acceleration norm & variance
        acc_norms = np.linalg.norm(acc_w, axis=1)
        mean_acc_norm = float(np.mean(acc_norms))
        acc_var = float(np.var(acc_norms))
        grav_diff = float(abs(mean_acc_norm - self.gravity))

        # 2. Gyroscope norm & variance
        gyro_norms = np.linalg.norm(gyro_w, axis=1)
        gyro_var = float(np.var(gyro_norms))

        # 3. Trailing Jerk RMS
        jerk_vec = np.diff(acc_w, axis=0) / self.dt
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_vec**2, axis=1)))) if len(jerk_vec) > 0 else 0.0

        # Multi-feature candidate condition
        c_acc = acc_var < self.th_acc_var
        c_gyro = gyro_var < self.th_gyro_var
        c_grav = grav_diff < self.th_grav_diff
        c_jerk = jerk_rms < self.th_jerk_rms

        raw_candidate = c_acc and c_gyro and c_grav and c_jerk

        # Temporal persistence gating:
        # Require persist_epochs consecutive candidate detections to confirm stop.
        # Immediate exit on violation to avoid false braking/crawling ZUPT lock.
        if raw_candidate:
            self.consecutive_standstill += 1
            if self.consecutive_standstill >= self.persist_epochs:
                self.is_currently_stationary = True
        else:
            self.consecutive_standstill = 0
            self.is_currently_stationary = False

        return {
            'is_stationary': self.is_currently_stationary,
            'raw_candidate': raw_candidate,
            'consecutive_count': self.consecutive_standstill,
            'acc_var': acc_var,
            'gyro_var': gyro_var,
            'grav_diff': grav_diff,
            'jerk_rms': jerk_rms
        }
