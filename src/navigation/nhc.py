"""
SIH26168 - Stage C4: Non-Holonomic Constraints (NHC)
Module: src/navigation/nhc.py

Implements physical Non-Holonomic Constraints (NHC) for land vehicles
integrated into the 15-state 3D Error-State Kalman Filter (ESKF).

===============================================================================
PHYSICAL & MATHEMATICAL FOUNDATION
===============================================================================

1. PHYSICAL MODEL:
   For a typical wheeled road vehicle moving on a road surface:
     - The vehicle cannot instantaneously slide sideways (no lateral slip):
         v_y^v approx 0
     - The vehicle cannot fly upwards or jump through the road (no vertical motion):
         v_z^v approx 0
   where the vehicle body frame is defined according to ISO 8855:
     +X_v: Longitudinal (Forward)
     +Y_v: Lateral (Left)
     +Z_v: Vertical (Up)

   Forward velocity v_x^v is unconstrained by NHC.

2. MEASUREMENT FUNCTION:
   Let v^n in R^3 be the velocity in the local navigation frame (East, North, Up).
   The velocity resolved in the vehicle body frame is:
       v^v = C_n^v v^n = (C_v^n)^T v^n

   The NHC measurement vector is:
       h_NHC(x) = [v_y^v, v_z^v]^T in R^2

   The pseudo-measurement is:
       z_NHC = [0, 0]^T in R^2

   The measurement innovation (residual) is:
       r_NHC = z_NHC - h_NHC(x_hat) = [-v_hat_y^v, -v_hat_z^v]^T

3. ERROR-STATE MEASUREMENT JACOBIAN DERIVATION:
   Under the right-multiplicative body-frame attitude error convention:
       C_v^n = C_hat_v^n Exp([delta_theta^v]_x) approx C_hat_v^n (I + [delta_theta^v]_x)
       C_n^v = (C_v^n)^T approx (I - [delta_theta^v]_x) C_hat_n^v

   The true vehicle-frame velocity is:
       v^v = C_n^v v^n = (I - [delta_theta^v]_x) C_hat_n^v (v_hat^n + delta_v^n)
           = C_hat_n^v v_hat^n + C_hat_n^v delta_v^n - [delta_theta^v]_x (C_hat_n^v v_hat^n) + h.o.t.
           = v_hat^v + C_hat_n^v delta_v^n + [v_hat^v]_x delta_theta^v

   Therefore:
       delta_v^v = C_hat_n^v delta_v^n + [v_hat^v]_x delta_theta^v

   Partial derivatives:
       d(v^v) / d(delta_v^n)     = C_hat_n^v = (C_hat_v^n)^T in R^{3 x 3}
       d(v^v) / d(delta_theta^v) = [v_hat^v]_x in R^{3 x 3}
       d(v^v) / d(delta_p^n)     = 0_{3 x 3}
       d(v^v) / d(delta_ba)      = 0_{3 x 3}
       d(v^v) / d(delta_bg)      = 0_{3 x 3}

   The complete 2 x 15 measurement matrix H_NHC is formed by extracting
   rows 1 and 2 (0-indexed: row 1 is Y_v, row 2 is Z_v):

       H_NHC = [
           [0_1x3,  C_hat_n^v[1, :],  [v_hat_z^v,   0,           -v_hat_x^v],  0_1x3,  0_1x3],
           [0_1x3,  C_hat_n^v[2, :],  [-v_hat_y^v,  v_hat_x^v,   0         ],  0_1x3,  0_1x3]
       ] in R^{2 x 15}

   CRUCIAL OBSERVABILITY LINK:
       H_NHC[0, 8] = -v_hat_x^v (forward velocity multiplying yaw error delta_theta_z).
       When the vehicle moves forward (v_hat_x^v > 0), any heading error rotates
       forward velocity into the lateral body axis, making yaw error directly
       observable through the zero-lateral-velocity constraint!
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    """Computes the 3x3 skew-symmetric matrix [v]_x."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)

def rotvec_to_dcm(v: np.ndarray) -> np.ndarray:
    """Computes rotation matrix from rotation vector using Rodrigues formula."""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(v)
    axis = v / theta
    c = np.cos(theta)
    s = np.sin(theta)
    return c * np.eye(3, dtype=np.float64) + (1.0 - c) * np.outer(axis, axis) + s * skew(axis)

def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Maps 3D small rotation vector to unit quaternion [w, x, y, z]."""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], dtype=np.float64)
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta) / theta
    return np.array([c, s * v[0], s * v[1], s * v[2]], dtype=np.float64)

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q = q1 (x) q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def compute_nhc_residual_and_jacobian(
    C_v_n: np.ndarray,
    vel_n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes NHC residual r_NHC in R^2, measurement matrix H_NHC in R^{2 x 15},
    and vehicle body-frame velocity v^v in R^3.

    Args:
        C_v_n: 3x3 direction cosine matrix transforming Vehicle -> Navigation (ENU).
        vel_n: 3x1 velocity vector in Navigation frame (ENU), [v_East, v_North, v_Up].

    Returns:
        r_nhc: 2x1 residual [-v_y^v, -v_z^v]^T.
        H_nhc: 2x15 measurement Jacobian matrix.
        vel_v: 3x1 velocity in Vehicle body frame [v_x^v, v_y^v, v_z^v]^T.
    """
    C_n_v = C_v_n.T
    vel_v = C_n_v @ vel_n

    # Residual: z - h(x) = [0, 0]^T - [v_y^v, v_z^v]^T
    r_nhc = np.array([-vel_v[1], -vel_v[2]], dtype=np.float64)

    # Measurement Jacobian H_NHC in R^{2 x 15}
    H_nhc = np.zeros((2, 15), dtype=np.float64)

    # Row 0: Lateral velocity constraint (v_y^v approx 0)
    H_nhc[0, 3:6] = C_n_v[1, :]
    H_nhc[0, 6:9] = np.array([vel_v[2], 0.0, -vel_v[0]], dtype=np.float64)

    # Row 1: Vertical velocity constraint (v_z^v approx 0)
    H_nhc[1, 3:6] = C_n_v[2, :]
    H_nhc[1, 6:9] = np.array([-vel_v[1], vel_v[0], 0.0], dtype=np.float64)

    return r_nhc, H_nhc, vel_v

def validate_nhc_jacobian_finite_difference(
    C_v_n: Optional[np.ndarray] = None,
    vel_n: Optional[np.ndarray] = None,
    eps: float = 1e-7
) -> Dict[str, Any]:
    """
    Validates the analytical NHC measurement Jacobian against numerical
    central finite differences across both velocity and attitude error states.
    """
    if C_v_n is None:
        C_v_n = rotvec_to_dcm(np.array([0.05, -0.08, 0.42], dtype=np.float64))
    if vel_n is None:
        vel_n = np.array([11.5, -4.2, 0.3], dtype=np.float64)

    r_analytic, H_analytic, vel_v = compute_nhc_residual_and_jacobian(C_v_n, vel_n)

    # Numerical Jacobian H_num (size 2 x 15)
    H_num = np.zeros((2, 15), dtype=np.float64)

    # 1. Perturb velocity delta_v^n
    for i in range(3):
        v_pert_plus = vel_n.copy()
        v_pert_minus = vel_n.copy()
        v_pert_plus[i] += eps
        v_pert_minus[i] -= eps

        v_v_plus = C_v_n.T @ v_pert_plus
        v_v_minus = C_v_n.T @ v_pert_minus

        h_plus = np.array([v_v_plus[1], v_v_plus[2]])
        h_minus = np.array([v_v_minus[1], v_v_minus[2]])
        H_num[:, 3 + i] = (h_plus - h_minus) / (2.0 * eps)

    # 2. Perturb attitude delta_theta^v (right-multiplicative body frame)
    for i in range(3):
        dtheta_plus = np.zeros(3)
        dtheta_minus = np.zeros(3)
        dtheta_plus[i] = eps
        dtheta_minus[i] = -eps

        C_plus = C_v_n @ rotvec_to_dcm(dtheta_plus)
        C_minus = C_v_n @ rotvec_to_dcm(dtheta_minus)

        v_v_plus = C_plus.T @ vel_n
        v_v_minus = C_minus.T @ vel_n

        h_plus = np.array([v_v_plus[1], v_v_plus[2]])
        h_minus = np.array([v_v_minus[1], v_v_minus[2]])
        H_num[:, 6 + i] = (h_plus - h_minus) / (2.0 * eps)

    max_abs_diff = float(np.max(np.abs(H_analytic - H_num)))
    rel_error = float(max_abs_diff / (np.max(np.abs(H_analytic)) + 1e-12))
    passed = rel_error < 1e-4

    return {
        'max_abs_diff': max_abs_diff,
        'rel_error': rel_error,
        'passed': passed,
        'H_analytic': H_analytic,
        'H_num': H_num
    }

class NonHolonomicConstraint:
    """
    Manages Non-Holonomic Constraint (NHC) measurement updates on an ESKF instance.
    """
    def __init__(
        self,
        sigma_lat: float = 0.5,
        sigma_vert: float = 0.5
    ):
        """
        Args:
            sigma_lat: Standard deviation of lateral velocity constraint (m/s).
            sigma_vert: Standard deviation of vertical velocity constraint (m/s).
        """
        self.sigma_lat = float(sigma_lat)
        self.sigma_vert = float(sigma_vert)
        self.R_nhc = np.diag([self.sigma_lat**2, self.sigma_vert**2])

    def update_eskf(
        self,
        eskf: Any,
        custom_R: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Applies an NHC measurement update to the provided ESKF instance.
        Corrects nominal states (position, velocity, attitude, biases)
        and updates covariance using the Joseph-stabilized formulation.

        Args:
            eskf: An instance of ESKF3D.
            custom_R: Optional 2x2 measurement covariance.

        Returns:
            Dictionary containing innovation, error correction, and telemetry.
        """
        C_v_n = eskf.attitude.get_dcm()
        vel_n = eskf.vel_n

        r_nhc, H_nhc, vel_v = compute_nhc_residual_and_jacobian(C_v_n, vel_n)

        R = custom_R if custom_R is not None else self.R_nhc

        # Innovation covariance S in R^{2 x 2}
        S = H_nhc @ eskf.P @ H_nhc.T + R
        K = eskf.P @ H_nhc.T @ np.linalg.inv(S)

        # State error correction
        delta_x = K @ r_nhc

        # Inject corrections into ESKF nominal state
        # 1. Position correction
        eskf.pos_n += delta_x[0:3]
        # 2. Velocity correction
        eskf.vel_n += delta_x[3:6]
        # 3. Attitude correction (body-frame right multiplication)
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        # 4. Accelerometer bias correction
        eskf.ba += delta_x[9:12]
        # 5. Gyroscope bias correction
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_nhc
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {
            'innovation': r_nhc,
            'delta_x': delta_x,
            'innovation_cov': S,
            'vel_v': vel_v,
            'H_nhc': H_nhc
        }

if __name__ == "__main__":
    print("Testing NHC Jacobian finite-difference agreement...")
    val = validate_nhc_jacobian_finite_difference()
    print(f"Max absolute diff: {val['max_abs_diff']:.3e}")
    print(f"Relative error:    {val['rel_error']:.3e}")
    print(f"Status:            {'PASS' if val['passed'] else 'FAIL'}")
