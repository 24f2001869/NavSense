"""
SIH26168 - Stage C3: 3D INS + GNSS Error-State Kalman Filter (ESKF)
Module: src/navigation/eskf.py

Implements a 15-state Error-State Kalman Filter (ESKF) for 3D strapdown inertial
navigation tightly integrated with GNSS position and Doppler velocity observations.

===============================================================================
MATHEMATICAL ARCHITECTURE & STATE FORMULATION
===============================================================================

1. NOMINAL STATE VECTOR (16 parameters):
   -------------------------------------
   - Position:    p^n in R^3       (East, North, Up in meters)
   - Velocity:    v^n in R^3       (v_East, v_North, v_Up in m/s)
   - Attitude:    q_nv in H        (Unit quaternion representing Vehicle -> Nav ENU)
   - Accel Bias:  b_a in R^3       (Accelerometer zero-bias in Vehicle body frame, m/s^2)
   - Gyro Bias:   b_g in R^3       (Gyroscope zero-bias in Vehicle body frame, rad/s)

2. ERROR STATE VECTOR (15 states):
   --------------------------------
   delta_x = [delta_p, delta_v, delta_theta, delta_ba, delta_bg]^T in R^15

   where:
   - delta_p in R^3:     Position error in local navigation frame (ENU).
   - delta_v in R^3:     Velocity error in local navigation frame (ENU).
   - delta_theta in R^3: True rotation vector error in Vehicle body frame (right-multiplicative):
                         q_true = q_nv (x) delta_q(delta_theta).
   - delta_ba in R^3:    Accelerometer bias error in Vehicle body frame.
   - delta_bg in R^3:    Gyroscope bias error in Vehicle body frame.

3. CONTINUOUS LINEARIZED ERROR-STATE DYNAMICS:
   --------------------------------------------
   delta_p_dot     = delta_v
   delta_v_dot     = -C_v^n * [f_corr^v]_x * delta_theta - C_v^n * delta_ba + w_v
   delta_theta_dot = -[omega_corr^v]_x * delta_theta - delta_bg + w_theta
   delta_ba_dot    = w_ba  (bias random walk)
   delta_bg_dot    = w_bg  (bias random walk)

   Continuous system matrix F_c in R^{15 x 15}:
   F_c = [
     [0_3,  I_3,  0_3,                         0_3,      0_3  ],
     [0_3,  0_3,  -C_v^n * [f_corr^v]_x,       -C_v^n,   0_3  ],
     [0_3,  0_3,  -[omega_corr^v]_x,           0_3,      -I_3 ],
     [0_3,  0_3,  0_3,                         0_3,      0_3  ],
     [0_3,  0_3,  0_3,                         0_3,      0_3  ]
   ]

4. DISCRETE TRANSITION MATRIX (Phi):
   ----------------------------------
   Phi approx I_15 + F_c * dt + 0.5 * (F_c * dt)^2

5. GNSS OBSERVATION MODEL:
   ------------------------
   Observation vector z in R^6:
     z_p = p_GNSS - p^n
     z_v = v_GNSS - v^n
     z = [z_p, z_v]^T

   Measurement matrix H in R^{6 x 15}:
     H = [
       [I_3, 0_3, 0_3, 0_3, 0_3],
       [0_3, I_3, 0_3, 0_3, 0_3]
     ]

6. ERROR-STATE INJECTION & RESET:
   -------------------------------
   After Kalman update delta_x_hat = K * (z - H * delta_x):
     p^n    <- p^n + delta_p_hat
     v^n    <- v^n + delta_v_hat
     q_nv   <- q_nv (x) delta_q(delta_theta_hat)
     q_nv   <- q_nv / ||q_nv||
     b_a    <- b_a + delta_ba_hat
     b_g    <- b_g + delta_bg_hat
   Covariance is updated via Joseph-stabilized form:
     P <- (I - K*H) * P * (I - K*H)^T + K * R * K^T
===============================================================================
"""

from typing import Dict, Optional, Tuple
import numpy as np

from src.navigation.attitude import AttitudeEstimator3D


def skew(v: np.ndarray) -> np.ndarray:
    """Computes the 3x3 skew-symmetric matrix [v]_x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Converts rotation vector v into a Hamilton unit quaternion [qw, qx, qy, qz]."""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    s = np.sin(0.5 * th) / th
    return np.array([np.cos(0.5 * th), s * v[0], s * v[1], s * v[2]], dtype=np.float64)


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion multiplication: q = q1 (x) q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


class ESKF3D:
    """
    15-State 3D Error-State Kalman Filter for INS/GNSS integration.
    """

    def __init__(
        self,
        init_pos_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_vel_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_heading_deg: float = 0.0,
        init_pitch_deg: float = 0.0,
        init_roll_deg: float = 0.0,
        R_vp: Optional[np.ndarray] = None,
        init_ba: Optional[np.ndarray] = None,
        init_bg: Optional[np.ndarray] = None,
        sigma_a: float = 0.15,        # Accel noise (m/s^2 / sqrt(Hz))
        sigma_g: float = 0.01,        # Gyro noise (rad/s / sqrt(Hz))
        sigma_ba: float = 0.001,      # Accel bias random walk (m/s^3 / sqrt(Hz))
        sigma_bg: float = 0.0001,     # Gyro bias random walk (rad/s^2 / sqrt(Hz))
        r_pos: float = 1.5,           # GNSS horizontal position std (m)
        r_pos_z: float = 3.0,         # GNSS vertical position std (m)
        r_vel: float = 0.1,           # GNSS horizontal velocity std (m/s)
        r_vel_z: float = 0.2,         # GNSS vertical velocity std (m/s)
        gravity: float = 9.80665
    ) -> None:
        # Nominal Navigation States
        self.pos_n = np.array(init_pos_enu, dtype=np.float64)
        self.vel_n = np.array(init_vel_enu, dtype=np.float64)
        self.ba = np.array(init_ba if init_ba is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.bg = np.array(init_bg if init_bg is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.g_val = float(gravity)

        # Stage C1 3D Attitude Engine
        self.attitude = AttitudeEstimator3D(
            init_heading_deg=init_heading_deg,
            init_pitch_deg=init_pitch_deg,
            init_roll_deg=init_roll_deg,
            R_vp=R_vp
        )

        # Error State Covariance P (15x15)
        # [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)]
        self.P = np.diag([
            1.5**2, 1.5**2, 3.0**2,          # pos (m)
            0.1**2, 0.1**2, 0.2**2,          # vel (m/s)
            np.radians(2.0)**2, np.radians(2.0)**2, np.radians(5.0)**2, # theta (rad)
            0.15**2, 0.15**2, 0.15**2,       # ba (m/s^2)
            np.radians(0.5)**2, np.radians(0.5)**2, np.radians(0.5)**2  # bg (rad/s)
        ])

        # Noise Parameters
        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_ba = sigma_ba
        self.sigma_bg = sigma_bg

        # Measurement Noise Covariance R (6x6)
        self.R = np.diag([
            r_pos**2, r_pos**2, r_pos_z**2,
            r_vel**2, r_vel**2, r_vel_z**2
        ])

        # Measurement Jacobian H (6x15)
        self.H = np.zeros((6, 15), dtype=np.float64)
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 3:6] = np.eye(3)

        # History tracking
        self.last_a_n = np.zeros(3, dtype=np.float64)
        self.last_f_corr_v = np.zeros(3, dtype=np.float64)
        self.last_omega_corr_v = np.zeros(3, dtype=np.float64)

    def predict(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        dt: float
    ) -> None:
        """
        Propagates nominal state and error-state covariance over time step dt.
        """
        if dt <= 0.0:
            return

        # 1. Measured vectors in vehicle body frame
        f_meas_p = np.array([accel_x, accel_y, accel_z], dtype=np.float64)
        omega_meas_p = np.array([gyro_x, gyro_y, gyro_z], dtype=np.float64)

        f_meas_v = self.attitude.R_vp @ f_meas_p
        omega_meas_v = self.attitude.R_vp @ omega_meas_p

        # 2. Subtract estimated sensor biases
        f_corr_v = f_meas_v - self.ba
        omega_corr_v = omega_meas_v - self.bg
        self.last_f_corr_v = f_corr_v
        self.last_omega_corr_v = omega_corr_v

        # 3. Transform specific force to navigation frame
        C_v_n = self.attitude.get_dcm()
        f_corr_n = C_v_n @ f_corr_v

        # 4. Kinematic acceleration with gravity compensation
        a_n = f_corr_n - np.array([0.0, 0.0, self.g_val], dtype=np.float64)
        self.last_a_n = a_n

        # 5. Nominal state propagation (Trapezoidal integration)
        self.pos_n += self.vel_n * dt + 0.5 * a_n * (dt**2)
        self.vel_n += a_n * dt

        # Propagate nominal attitude quaternion
        dq = rotvec_to_quat(omega_corr_v * dt)
        self.attitude.q_nv = quat_mult(self.attitude.q_nv, dq)
        self.attitude.q_nv /= np.linalg.norm(self.attitude.q_nv)

        # 6. Discrete State Transition Matrix (Phi in R^{15 x 15})
        Phi = np.eye(15, dtype=np.float64)
        # delta_p block
        Phi[0:3, 3:6] = np.eye(3) * dt
        Phi[0:3, 6:9] = -0.5 * C_v_n @ skew(f_corr_v) * (dt**2)
        Phi[0:3, 9:12] = -0.5 * C_v_n * (dt**2)
        # delta_v block
        Phi[3:6, 6:9] = -C_v_n @ skew(f_corr_v) * dt
        Phi[3:6, 9:12] = -C_v_n * dt
        # delta_theta block (body-frame attitude error dynamics)
        Phi[6:9, 6:9] = np.eye(3) - skew(omega_corr_v) * dt
        Phi[6:9, 12:15] = -np.eye(3) * dt

        # 7. Discrete Process Noise Covariance Q_d
        q_diag = np.concatenate([
            (1.0/3.0) * (self.sigma_a**2) * (dt**3) * np.ones(3),
            (self.sigma_a**2) * dt * np.ones(3),
            (self.sigma_g**2) * dt * np.ones(3),
            (self.sigma_ba**2) * dt * np.ones(3),
            (self.sigma_bg**2) * dt * np.ones(3)
        ])
        Q_d = np.diag(q_diag)

        # 8. Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)

    def update_gnss(
        self,
        gnss_pos_enu: np.ndarray,
        gnss_vel_enu: np.ndarray,
        custom_R: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Performs EKF measurement update using GNSS position and Doppler velocity.
        Injects estimated errors into nominal states and resets error state.
        """
        z_pos = np.asarray(gnss_pos_enu, dtype=np.float64)
        z_vel = np.asarray(gnss_vel_enu, dtype=np.float64)
        z = np.concatenate([z_pos, z_vel])

        # Measurement residual (innovation)
        y = z - np.concatenate([self.pos_n, self.vel_n])

        R = custom_R if custom_R is not None else self.R
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Compute error state correction
        delta_x = K @ y

        # Error State Injection
        # 1. Position correction
        self.pos_n += delta_x[0:3]
        # 2. Velocity correction
        self.vel_n += delta_x[3:6]
        # 3. Attitude correction (body-frame right multiplication)
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        self.attitude.q_nv = quat_mult(self.attitude.q_nv, dq_corr)
        self.attitude.q_nv /= np.linalg.norm(self.attitude.q_nv)
        # 4. Accelerometer bias correction
        self.ba += delta_x[9:12]
        # 5. Gyroscope bias correction
        self.bg += delta_x[12:15]

        # Joseph-form Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return {
            'innovation': y,
            'delta_x': delta_x,
            'innovation_cov': S
        }

    def get_state(self) -> Dict[str, any]:
        """Returns complete navigation state, biases, and telemetry."""
        return {
            'pos_n': self.pos_n.copy(),
            'vel_n': self.vel_n.copy(),
            'acc_n': self.last_a_n.copy(),
            'ba': self.ba.copy(),
            'bg': self.bg.copy(),
            'yaw_deg': self.attitude.get_yaw_deg(),
            'pitch_deg': self.attitude.get_pitch_deg(),
            'roll_deg': self.attitude.get_roll_deg(),
            'q_norm': self.attitude.get_quaternion_norm(),
            'dcm': self.attitude.get_dcm(),
            'cov_diag': np.diag(self.P).copy()
        }
