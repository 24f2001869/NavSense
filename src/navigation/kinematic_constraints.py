"""
SIH26168 - Stage C7-B1: Soft Kinematic Velocity & Acceleration Consistency Constraints
Module: src/navigation/kinematic_constraints.py

Implements:
1. SoftLongitudinalAccelerationConstraint:
   A soft kinematic constraint on vehicle forward acceleration in the 15-state ESKF.
   When the estimated body-frame longitudinal acceleration |a_x^v| exceeds an empirically
   derived plausibility threshold a_soft, a soft measurement update pulls the state back
   via Joseph-stabilized Kalman update.
   
2. OracleAccelerationUpdate:
   Diagnostic upper-bound reference update using ground-truth vehicle acceleration
   (e.g., from VBOX differentiated speed) to quantify the theoretical limit of drift
   recovery achievable through acceleration consistency.

Mathematical Formulation:
--------------------------
Estimated vehicle body-frame kinematic acceleration:
    a^v = C_n^v * a^n = f_corr^v - C_n^v * g^n
    a_x^v = e_1^T * (f_corr^v - C_n^v * g^n)

In Deployable Soft Mode:
    If |a_x^v| <= a_soft:
        r_a = 0 (inactive)
    If |a_x^v| > a_soft:
        z_a = sign(a_x^v) * a_soft
        r_a = z_a - a_x^v = -sign(a_x^v) * (|a_x^v| - a_soft)

In Oracle Mode:
    z_a = a_x_ref
    r_a = z_a - a_x^v

Measurement Jacobian H in R^{1 x 15}:
    a_x^v depends on:
      - attitude error delta_theta: -e_1^T * [g^v]_x = [0, g_z^v, -g_y^v]
      - accel bias error delta_ba:  -e_1^T = [-1, 0, 0]
      - velocity error delta_v (rate-coupled): e_1^T * C_n^v / dt
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

from src.navigation.nhc import rotvec_to_quat, quat_mult, skew


class SoftLongitudinalAccelerationConstraint:
    """
    Soft longitudinal acceleration consistency constraint for 15-state ESKF.
    """

    def __init__(
        self,
        a_soft: float = 4.0,
        sigma_a_meas: float = 0.50,
        sigma_excess: float = 1.0,
        mode: str = "coupled",  # "coupled" (updates vel, attitude, bias) or "bias_attitude"
        decouple_attitude: bool = False,
        strict_attitude_freeze: bool = False
    ):
        """
        Args:
            a_soft: Soft plausibility threshold in m/s^2 (e.g., 4.0 m/s^2 from empirical audit).
            sigma_a_meas: Nominal measurement noise std for the acceleration constraint (m/s^2).
            sigma_excess: Excess acceleration scale for dynamic covariance weighting.
            mode: "coupled" includes velocity rate Jacobian; "bias_attitude" updates bias/attitude.
            decouple_attitude: If True, sets H_theta = 0 (removes direct attitude Jacobian pathway).
            strict_attitude_freeze: If True, also sets K[6:9] = 0 to prevent indirect covariance attitude updates.
        """
        self.a_soft = float(a_soft)
        self.sigma_a_meas = float(sigma_a_meas)
        self.sigma_excess = float(sigma_excess)
        self.mode = mode
        self.decouple_attitude = bool(decouple_attitude)
        self.strict_attitude_freeze = bool(strict_attitude_freeze)
        self.R_nominal = float(self.sigma_a_meas**2)

    def compute_body_accel(self, eskf: Any) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the current body-frame kinematic acceleration a_x^v and gravity vector g^v.
        
        Returns:
            a_x_v: Forward kinematic acceleration in vehicle body frame (m/s^2).
            a_v: Full 3D body kinematic acceleration [a_x^v, a_y^v, a_z^v]^T.
            g_v: Body-frame gravity vector C_n^v * g^n.
        """
        C_v_n = eskf.attitude.get_dcm()
        C_n_v = C_v_n.T
        
        # Gravity vector in navigation frame: [0, 0, -g]
        # In eskf.py: a_n = f_corr_n - [0, 0, g_val]
        # Therefore g^n = [0, 0, eskf.g_val]
        g_n = np.array([0.0, 0.0, eskf.g_val], dtype=np.float64)
        g_v = C_n_v @ g_n
        
        # Body kinematic acceleration: a^v = f_corr^v - g^v
        a_v = eskf.last_f_corr_v - g_v
        a_x_v = float(a_v[0])
        return a_x_v, a_v, g_v

    def update_eskf(
        self,
        eskf: Any,
        dt: float = 0.1,
        oracle_a_ref: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Applies the soft acceleration constraint update to the ESKF if active.
        
        Args:
            eskf: Instance of ESKF3D.
            dt: Epoch time step in seconds.
            oracle_a_ref: If provided, operates in Oracle mode using ground-truth a_ref.
            
        Returns:
            Dictionary containing activation flag, innovation, corrections, and covariance.
        """
        a_x_v, a_v, g_v = self.compute_body_accel(eskf)
        
        # Determine whether constraint is active and compute innovation y
        if oracle_a_ref is not None:
            # Oracle Mode: exact reference acceleration
            is_active = True
            z_target = float(oracle_a_ref)
            r_a = z_target - a_x_v
            # In Oracle mode, nominal variance is tight
            R_k = float(0.10**2)
            weight = 1.0
        else:
            # Deployable Soft Mode
            abs_a = abs(a_x_v)
            if abs_a <= self.a_soft:
                # Plausible acceleration: constraint is completely inactive
                return {
                    'active': False,
                    'a_x_v': a_x_v,
                    'innovation': 0.0,
                    'delta_x': np.zeros(15, dtype=np.float64),
                    'R_k': self.R_nominal
                }
            
            is_active = True
            excess = abs_a - self.a_soft
            z_target = np.sign(a_x_v) * self.a_soft
            r_a = z_target - a_x_v  # Innovation: z_target - a_x_v
            
            # Dynamic weighting: higher excess -> stronger relative confidence
            weight = min(1.0, max(0.1, excess / self.sigma_excess))
            R_k = self.R_nominal / weight

        # Formulate Measurement Jacobian H in R^{1 x 15}
        # Error state: delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]
        H = np.zeros((1, 15), dtype=np.float64)
        
        # 1. Accelerometer bias Jacobian: d(a_x_v) / d(delta_ba) = -e_1^T = [-1, 0, 0]
        H[0, 9] = -1.0
        
        # 2. Attitude Jacobian: d(a_x_v) / d(delta_theta) = -e_1^T * [g^v]_x
        # [g^v]_x = [[0, -gz, gy], [gz, 0, -gx], [-gy, gx, 0]]
        # e_1^T * [g^v]_x = [0, -gz, gy]
        # So -e_1^T * [g^v]_x = [0, gz, -gy]
        if not self.decouple_attitude:
            H[0, 6] = 0.0
            H[0, 7] = g_v[2]   # maps to pitch error delta_theta_y
            H[0, 8] = -g_v[1]  # maps to yaw error delta_theta_z
        else:
            # Stage C7-B1-D: Diagnostic isolation of attitude channel (H_theta = 0)
            H[0, 6:9] = 0.0
        
        # 3. Velocity coupling (if enabled in coupled mode)
        # a_x_v connects to forward velocity change: delta_a_x_v = (e_1^T * C_n^v * delta_v) / dt
        if self.mode == "coupled" and dt > 0.0:
            C_v_n = eskf.attitude.get_dcm()
            C_n_v = C_v_n.T
            # Direction vector of vehicle forward axis in navigation frame
            # e_1^T * C_n_v is the 1st row of C_n_v = 1st column of C_v_n
            H[0, 3:6] = C_n_v[0, :] / max(dt, 0.05)

        # Kalman Filter Update
        # S = H P H^T + R in R^{1 x 1}
        S = float(H @ eskf.P @ H.T + R_k)
        if S <= 1e-12 or np.isnan(S):
            return {
                'active': False,
                'a_x_v': a_x_v,
                'innovation': r_a,
                'delta_x': np.zeros(15, dtype=np.float64),
                'K': np.zeros(15, dtype=np.float64),
                'H': H.flatten(),
                'R_k': R_k,
                'pos_corr': np.zeros(3, dtype=np.float64),
                'vel_corr': np.zeros(3, dtype=np.float64),
                'att_corr': np.zeros(3, dtype=np.float64),
                'ba_corr': np.zeros(3, dtype=np.float64),
                'bg_corr': np.zeros(3, dtype=np.float64)
            }
            
        inv_S = 1.0 / S
        # Kalman gain K in R^{15 x 1}: K = P H^T * inv_S
        K = (eskf.P @ H.T) * inv_S
        
        if self.strict_attitude_freeze:
            # Zero out attitude Kalman gain block to strictly prohibit any cross-covariance leakage
            K[6:9, :] = 0.0
        
        # Error state correction delta_x = K * r_a
        delta_x = (K * r_a).flatten()
        
        # Apply error-state injection to nominal states
        # 1. Position correction
        eskf.pos_n += delta_x[0:3]
        # 2. Velocity correction
        eskf.vel_n += delta_x[3:6]
        # 3. Attitude correction (body-frame right multiplication)
        dtheta_b = delta_x[6:9]
        if np.linalg.norm(dtheta_b) > 1e-12:
            dq_corr = rotvec_to_quat(dtheta_b)
            eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
            eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        # 4. Accelerometer bias correction
        eskf.ba += delta_x[9:12]
        # 5. Gyroscope bias correction
        eskf.bg += delta_x[12:15]
        
        # Joseph-stabilized Covariance Update: P = (I - KH) P (I - KH)^T + K R K^T
        IKH = np.eye(15, dtype=np.float64) - K @ H
        R_mat = np.array([[R_k]], dtype=np.float64)
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R_mat @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)
        
        return {
            'active': True,
            'a_x_v': a_x_v,
            'innovation': r_a,
            'delta_x': delta_x,
            'K': K.flatten(),
            'H': H.flatten(),
            'R_k': R_k,
            'weight': weight,
            'pos_corr': delta_x[0:3].copy(),
            'vel_corr': delta_x[3:6].copy(),
            'att_corr': delta_x[6:9].copy(),
            'ba_corr': delta_x[9:12].copy(),
            'bg_corr': delta_x[12:15].copy()
        }
