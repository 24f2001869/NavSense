"""
SIH26168 - Stage C8-5: Chassis Wheel-Speed Odometry Fusion
Module: src/navigation/wheel_odometry.py

Implements physical chassis wheel-speed odometry measurements and fusion
for the 15-state 3D Error-State Kalman Filter (ESKF).

===============================================================================
PHYSICAL & MATHEMATICAL FOUNDATION
===============================================================================

1. PHYSICAL MEASUREMENT:
   For a typical front-wheel-drive land vehicle (e.g. Ford Fiesta):
     - Front wheels (FL, FR) transmit engine drive torque and experience slip
       during acceleration maneuvers.
     - Rear wheels (RL, RR) are non-driven and follow vehicle motion with
       negligible longitudinal slip during normal cruising and acceleration.
     - Effective rolling radius under load: r_w approx 0.2766 m.
     - Rear wheel speed average:
         v_wheel = 0.5 * (omega_RL + omega_RR) * r_w

2. MEASUREMENT MODELS:
   A. Forward Velocity Only (1-DOF):
      h_fwd(x) = v_x^v = C_n^v[0, :] * v^n
      z_fwd = v_wheel
      r_fwd = z_fwd - v_x^v
      H_fwd in R^{1 x 15}:
        H_fwd[0, 3:6] = C_n^v[0, :]
        H_fwd[0, 6:9] = [0.0, -v_z^v, v_y^v]

   B. Full 3D Body Velocity Fusion (Velocity + NHC, 3-DOF):
      h_3D(x) = v^v = C_n^v * v^n in R^3
      z_3D = [v_wheel, 0.0, 0.0]^T in R^3
      r_3D = z_3D - v^v = [v_wheel - v_x^v, -v_y^v, -v_z^v]^T
      H_3D in R^{3 x 15}:
        H_3D[:, 3:6] = C_n^v
        H_3D[:, 6:9] = [v^v]_x = [
            [ 0.0,      -v_z^v,  v_y^v],
            [ v_z^v,     0.0,   -v_x^v],
            [-v_y^v,     v_x^v,  0.0  ]
        ]
      R_3D = diag(sigma_wheel^2, sigma_lat^2, sigma_vert^2)

3. ERROR STATE INJECTION & JOSEPH COVARIANCE UPDATE:
   Joseph-stabilized covariance update guarantees symmetry and positive definiteness:
       P = (I - K*H) P (I - K*H)^T + K R K^T
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

from src.navigation.nhc import rotvec_to_dcm, rotvec_to_quat, quat_mult, skew

# Calibrated effective tire rolling radius on IO-VNBD Ford Fiesta
DEFAULT_TIRE_RADIUS_M = 0.2766
DEFAULT_SIGMA_WHEEL_MS = 0.20
DEFAULT_SIGMA_NHC_LAT_MS = 0.50
DEFAULT_SIGMA_NHC_VERT_MS = 0.50


def compute_wheel_speed_from_can(
    wheel_rl_rads: Any,
    wheel_rr_rads: Any,
    tire_radius: float = DEFAULT_TIRE_RADIUS_M
) -> Any:
    """Computes forward velocity from non-driven rear wheel angular speeds."""
    rl = np.asarray(wheel_rl_rads, dtype=np.float64)
    rr = np.asarray(wheel_rr_rads, dtype=np.float64)
    omega_rear = 0.5 * (rl + rr)
    speed = omega_rear * tire_radius
    if np.ndim(speed) == 0:
        return float(speed)
    return speed


def compute_velocity_residual_and_jacobian_1d(
    C_v_n: np.ndarray,
    vel_n: np.ndarray,
    v_wheel: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes 1-DOF forward velocity residual and measurement Jacobian.

    Args:
        C_v_n: 3x3 direction cosine matrix transforming Vehicle -> Navigation (ENU).
        vel_n: 3x1 velocity vector in Navigation frame (ENU).
        v_wheel: Measured forward wheel speed in m/s.

    Returns:
        r_fwd: Scalar innovation (v_wheel - v_x^v).
        H_fwd: 1x15 measurement Jacobian matrix.
        vel_v: 3x1 velocity in Vehicle body frame.
    """
    C_n_v = C_v_n.T
    vel_v = C_n_v @ vel_n

    r_fwd = float(v_wheel - vel_v[0])

    H_fwd = np.zeros((1, 15), dtype=np.float64)
    H_fwd[0, 3:6] = C_n_v[0, :]
    H_fwd[0, 6:9] = np.array([0.0, -vel_v[2], vel_v[1]], dtype=np.float64)

    return r_fwd, H_fwd, vel_v


def compute_velocity_residual_and_jacobian_3d(
    C_v_n: np.ndarray,
    vel_n: np.ndarray,
    v_wheel: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3-DOF coupled body velocity (Forward + Lateral NHC + Vertical NHC)
    residual and measurement Jacobian.

    Args:
        C_v_n: 3x3 direction cosine matrix transforming Vehicle -> Navigation (ENU).
        vel_n: 3x1 velocity vector in Navigation frame (ENU).
        v_wheel: Measured forward wheel speed in m/s.

    Returns:
        r_3d: 3x1 innovation [v_wheel - v_x^v, -v_y^v, -v_z^v]^T.
        H_3d: 3x15 measurement Jacobian matrix.
        vel_v: 3x1 velocity in Vehicle body frame.
    """
    C_n_v = C_v_n.T
    vel_v = C_n_v @ vel_n

    z_3d = np.array([v_wheel, 0.0, 0.0], dtype=np.float64)
    r_3d = z_3d - vel_v

    H_3d = np.zeros((3, 15), dtype=np.float64)
    H_3d[:, 3:6] = C_n_v
    H_3d[:, 6:9] = skew(vel_v)

    return r_3d, H_3d, vel_v


def validate_velocity_jacobian_finite_difference(
    C_v_n: Optional[np.ndarray] = None,
    vel_n: Optional[np.ndarray] = None,
    v_wheel: float = 15.0,
    eps: float = 1e-7
) -> Dict[str, Any]:
    """Validates 1D and 3D velocity Jacobians against central finite differences."""
    if C_v_n is None:
        C_v_n = rotvec_to_dcm(np.array([0.05, -0.08, 0.42], dtype=np.float64))
    if vel_n is None:
        vel_n = np.array([12.5, -5.2, 0.4], dtype=np.float64)

    # 1D Validation
    r_1d, H_1d, vel_v = compute_velocity_residual_and_jacobian_1d(C_v_n, vel_n, v_wheel)
    H_num_1d = np.zeros((1, 15), dtype=np.float64)

    for i in range(3):
        v_plus = vel_n.copy(); v_minus = vel_n.copy()
        v_plus[i] += eps; v_minus[i] -= eps
        h_plus = (C_v_n.T @ v_plus)[0]
        h_minus = (C_v_n.T @ v_minus)[0]
        H_num_1d[0, 3 + i] = (h_plus - h_minus) / (2.0 * eps)

    for i in range(3):
        dth_p = np.zeros(3); dth_m = np.zeros(3)
        dth_p[i] = eps; dth_m[i] = -eps
        C_p = C_v_n @ rotvec_to_dcm(dth_p)
        C_m = C_v_n @ rotvec_to_dcm(dth_m)
        h_plus = (C_p.T @ vel_n)[0]
        h_minus = (C_m.T @ vel_n)[0]
        H_num_1d[0, 6 + i] = (h_plus - h_minus) / (2.0 * eps)

    diff_1d = float(np.max(np.abs(H_1d - H_num_1d)))
    rel_1d = float(diff_1d / (np.max(np.abs(H_1d)) + 1e-12))

    # 3D Validation
    r_3d, H_3d, vel_v = compute_velocity_residual_and_jacobian_3d(C_v_n, vel_n, v_wheel)
    H_num_3d = np.zeros((3, 15), dtype=np.float64)

    for i in range(3):
        v_plus = vel_n.copy(); v_minus = vel_n.copy()
        v_plus[i] += eps; v_minus[i] -= eps
        h_plus = C_v_n.T @ v_plus
        h_minus = C_v_n.T @ v_minus
        H_num_3d[:, 3 + i] = (h_plus - h_minus) / (2.0 * eps)

    for i in range(3):
        dth_p = np.zeros(3); dth_m = np.zeros(3)
        dth_p[i] = eps; dth_m[i] = -eps
        C_p = C_v_n @ rotvec_to_dcm(dth_p)
        C_m = C_v_n @ rotvec_to_dcm(dth_m)
        h_plus = C_p.T @ vel_n
        h_minus = C_m.T @ vel_n
        H_num_3d[:, 6 + i] = (h_plus - h_minus) / (2.0 * eps)

    diff_3d = float(np.max(np.abs(H_3d - H_num_3d)))
    rel_3d = float(diff_3d / (np.max(np.abs(H_3d)) + 1e-12))

    passed = (rel_1d < 1e-4) and (rel_3d < 1e-4)
    return {
        'passed': passed,
        'diff_1d': diff_1d,
        'rel_1d': rel_1d,
        'diff_3d': diff_3d,
        'rel_3d': rel_3d
    }


class ChassisWheelSpeedFusion:
    """
    Manages Chassis Wheel-Speed Odometry measurement updates on an ESKF instance.
    Supports 1-DOF forward velocity update or coupled 3-DOF body velocity update (Velocity + NHC).
    """

    def __init__(
        self,
        sigma_wheel: float = DEFAULT_SIGMA_WHEEL_MS,
        sigma_lat: float = DEFAULT_SIGMA_NHC_LAT_MS,
        sigma_vert: float = DEFAULT_SIGMA_NHC_VERT_MS,
        tire_radius: float = DEFAULT_TIRE_RADIUS_M,
        nis_gate_1d: float = 6.635,   # 99% chi-square for 1 DOF
        nis_gate_3d: float = 11.345   # 99% chi-square for 3 DOF
    ):
        self.sigma_wheel = float(sigma_wheel)
        self.sigma_lat = float(sigma_lat)
        self.sigma_vert = float(sigma_vert)
        self.tire_radius = float(tire_radius)
        self.nis_gate_1d = float(nis_gate_1d)
        self.nis_gate_3d = float(nis_gate_3d)

        self.R_1d = np.array([[self.sigma_wheel**2]], dtype=np.float64)
        self.R_3d = np.diag([self.sigma_wheel**2, self.sigma_lat**2, self.sigma_vert**2])

    def update_eskf_forward_velocity(
        self,
        eskf: Any,
        v_wheel: float,
        apply_nis_gate: bool = True
    ) -> Dict[str, Any]:
        """
        Applies a 1-DOF forward velocity measurement update to the ESKF instance
        with Strict Attitude & Bias Freeze to prevent longitudinal error leakage into tilt.
        """
        C_v_n = eskf.attitude.get_dcm()
        vel_n = eskf.vel_n

        r_fwd, H_fwd, vel_v = compute_velocity_residual_and_jacobian_1d(C_v_n, vel_n, v_wheel)

        S = float(np.squeeze(H_fwd @ eskf.P @ H_fwd.T + self.R_1d))
        nis = float((r_fwd**2) / max(S, 1e-6))

        # Robust Huber adaptive scaling if NIS is elevated (e.g. transient slip or bumps)
        R_eff = self.R_1d.copy()
        gated_out = False
        if apply_nis_gate and nis > self.nis_gate_1d:
            gated_out = True
            scale = np.sqrt(nis / self.nis_gate_1d)
            R_eff = self.R_1d * (scale**2)
            S = float(np.squeeze(H_fwd @ eskf.P @ H_fwd.T + R_eff))

        K = (eskf.P @ H_fwd.T) / max(S, 1e-6)  # shape (15, 1)

        # STRICT ATTITUDE & BIAS FREEZE
        # Longitudinal speed has no direct observability of roll/pitch or sensor biases.
        # Zeroing out indices 6:15 prevents positive-feedback tilt/bias runaway.
        K[6:15, :] = 0.0

        delta_x = (K * r_fwd).flatten()

        # State error injection (Position and Velocity only)
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]

        # Joseph-form Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_fwd
        eskf.P = IKH @ eskf.P @ IKH.T + (K @ R_eff @ K.T)
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {
            'applied': True,
            'gated_out': gated_out,
            'innovation': r_fwd,
            'nis': nis,
            'delta_x': delta_x,
            'vel_v': vel_v
        }

    def update_eskf_3d_velocity(
        self,
        eskf: Any,
        v_wheel: float,
        apply_nis_gate: bool = True
    ) -> Dict[str, Any]:
        """
        Applies a 3-DOF coupled body velocity (Forward + Lateral NHC + Vertical NHC)
        measurement update to the ESKF instance with Strict Bias Freeze.
        """
        C_v_n = eskf.attitude.get_dcm()
        vel_n = eskf.vel_n

        r_3d, H_3d, vel_v = compute_velocity_residual_and_jacobian_3d(C_v_n, vel_n, v_wheel)

        S = H_3d @ eskf.P @ H_3d.T + self.R_3d
        S_inv = np.linalg.inv(S)
        nis = float(np.squeeze(r_3d.T @ S_inv @ r_3d))

        R_eff = self.R_3d.copy()
        gated_out = False
        if apply_nis_gate and nis > self.nis_gate_3d:
            gated_out = True
            scale = np.sqrt(nis / self.nis_gate_3d)
            R_eff = self.R_3d * (scale**2)
            S = H_3d @ eskf.P @ H_3d.T + R_eff
            S_inv = np.linalg.inv(S)

        K = eskf.P @ H_3d.T @ S_inv

        # Strict Longitudinal Attitude Freeze: Forward velocity column (col 0)
        # must NOT update attitude (indices 6:9).
        K[6:9, 0] = 0.0
        # Strict Bias Freeze: Velocity constraints must NOT update accelerometer or gyro biases.
        K[9:15, :] = 0.0

        delta_x = K @ r_3d

        # State error injection
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)

        # Joseph-form Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_3d
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R_eff @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {
            'applied': True,
            'gated_out': gated_out,
            'innovation': r_3d,
            'nis': nis,
            'delta_x': delta_x,
            'vel_v': vel_v
        }

