"""
SIH26168 - Step 5: Extended Kalman Filter (EKF) for INS/GNSS Integration
Loosely coupled 7-state error-state / direct EKF:
  State: x = [p_E, p_N, v_E, v_N, psi, b_a, b_omega]^T
  - Position in local ENU (m)
  - Velocity in local ENU (m/s)
  - Vehicle heading azimuth (rad, clockwise from North)
  - Forward accelerometer bias (m/s^2)
  - Gyroscope yaw rate bias (rad/s)
"""

import numpy as np

class NavigationEKF:
    def __init__(self, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0), init_yaw=0.0):
        # State vector [p_E, p_N, v_E, v_N, psi, b_a, b_w]
        self.x = np.zeros(7)
        self.x[0], self.x[1] = init_pos
        self.x[2], self.x[3] = init_vel
        self.x[4] = init_yaw

        # State Covariance Matrix P
        self.P = np.diag([
            5.0**2, 5.0**2,     # Position uncertainty (m^2)
            1.0**2, 1.0**2,     # Velocity uncertainty (m/s)^2
            np.radians(2.0)**2, # Heading uncertainty (rad^2)
            0.1**2,             # Accel bias uncertainty (m/s^2)^2
            0.005**2            # Gyro bias uncertainty (rad/s)^2
        ])

        # Process Noise Covariance Q
        self.q_pos = 0.01
        self.q_vel = 0.1
        self.q_heading = 0.002
        self.q_ba = 0.0001
        self.q_bw = 0.00001

        # Measurement Noise Covariances R
        self.R_gnss_pos = 2.5**2 # GNSS horizontal position accuracy (~2.5m)
        self.R_gnss_vel = 0.4**2 # GNSS speed accuracy (~0.4 m/s)
        self.R_gnss_head = np.radians(2.0)**2 # GNSS heading accuracy (~2 deg)

    def predict(self, a_fwd, omega_z, dt=0.1):
        """
        Prediction step using IMU forward acceleration and yaw rate.
        """
        p_e, p_n, v_e, v_n, psi, b_a, b_w = self.x

        # Corrected inertial inputs
        a_corr = a_fwd - b_a
        w_corr = omega_z - b_w

        # State integration
        self.x[0] += v_e * dt + 0.5 * (a_corr * np.sin(psi)) * dt**2
        self.x[1] += v_n * dt + 0.5 * (a_corr * np.cos(psi)) * dt**2
        self.x[2] += a_corr * np.sin(psi) * dt
        self.x[3] += a_corr * np.cos(psi) * dt
        self.x[4] = (psi + w_corr * dt + np.pi) % (2 * np.pi) - np.pi

        # Continuous-time Jacobian F_c
        F_c = np.zeros((7, 7))
        F_c[0, 2] = 1.0
        F_c[1, 3] = 1.0
        F_c[2, 4] = a_corr * np.cos(psi)
        F_c[2, 5] = -np.sin(psi)
        F_c[3, 4] = -a_corr * np.sin(psi)
        F_c[3, 5] = -np.cos(psi)
        F_c[4, 6] = -1.0

        F = np.eye(7) + F_c * dt

        Q_d = np.diag([
            self.q_pos * dt, self.q_pos * dt,
            self.q_vel * dt, self.q_vel * dt,
            self.q_heading * dt,
            self.q_ba * dt, self.q_bw * dt
        ])

        self.P = F @ self.P @ F.T + Q_d

    def update_gnss(self, pos_e, pos_n, vel_e=None, vel_n=None, heading_rad=None):
        """
        Measurement update using GNSS observations (position, velocity, course heading).
        """
        meas = [pos_e, pos_n]
        h_rows = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        r_diag = [self.R_gnss_pos, self.R_gnss_pos]

        if vel_e is not None and vel_n is not None:
            meas.extend([vel_e, vel_n])
            h_rows.extend([
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            ])
            r_diag.extend([self.R_gnss_vel, self.R_gnss_vel])

        if heading_rad is not None:
            # Wrap heading innovation to [-pi, pi]
            head_diff = (heading_rad - self.x[4] + np.pi) % (2 * np.pi) - np.pi
            meas.append(self.x[4] + head_diff)
            h_rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            r_diag.append(self.R_gnss_head)

        z = np.array(meas)
        H = np.array(h_rows)
        R = np.diag(r_diag)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.x[4] = (self.x[4] + np.pi) % (2 * np.pi) - np.pi

        I = np.eye(7)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
