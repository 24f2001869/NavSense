"""
SIH26168 - Step 7: AI + EKF Fusion Engine (Physics-AI Hybrid Navigation)
Fuses IMU inertial prediction, GNSS corrections (when present), AI predicted forward velocity,
and Non-Holonomic Constraints (NHC: lateral velocity ~= 0) during GNSS blackouts.
"""

import numpy as np
from src.fusion.ekf import NavigationEKF

class AIFusedEKF(NavigationEKF):
    def __init__(self, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0), init_yaw=0.0):
        super().__init__(init_pos, init_vel, init_yaw)
        self.sigma_ai_speed = 0.8 # m/s uncertainty on AI speed
        self.sigma_nhc = 0.1      # m/s lateral velocity constraint

    def update_ai_and_nhc(self, ai_forward_speed):
        """
        Updates EKF using AI forward velocity + Non-Holonomic Constraint.
        Projects body velocity [v_fwd, 0.0] into ENU frame:
            v_E = v_fwd * sin(psi)
            v_N = v_fwd * cos(psi)
        """
        psi = self.x[4]
        v_fwd = max(0.0, float(ai_forward_speed))

        # Pseudo-measurement vector in ENU
        z = np.array([
            v_fwd * np.sin(psi),
            v_fwd * np.cos(psi)
        ])

        H = np.zeros((2, 7))
        H[0, 2] = 1.0 # v_E
        H[1, 3] = 1.0 # v_N

        R = np.diag([self.sigma_ai_speed**2, self.sigma_ai_speed**2])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.x[4] = (self.x[4] + np.pi) % (2 * np.pi) - np.pi

        I = np.eye(7)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
