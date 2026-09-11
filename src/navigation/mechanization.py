"""
SIH26168 - Stage C2: 3D Inertial Mechanization
Module: src/navigation/mechanization.py

Implements classical 3D strapdown inertial navigation mechanization in the local
tangent Navigation Frame (East-North-Up, ENU). Fuses the validated Stage C1
quaternion/DCM attitude foundation with full 3-axis specific force transformation,
rigorous gravity subtraction, and consistent numerical integration.

===============================================================================
MATHEMATICAL AUDIT & FORMULATION
===============================================================================

1. ACCELEROMETER PHYSICAL CONVENTION:
   -----------------------------------
   - Physical Quantity: Specific force f_p = [accel_x, accel_y, accel_z]^T (m/s^2).
   - Sensor Output under Static Conditions (Display Up):
     Internal proof mass measures the upward normal contact force resisting gravity:
       f_p approx [0, 0, +9.80665]^T m/s^2.
   - Relation to Kinematic Acceleration:
       f = a - g
     where a = d^2 p / dt^2 is true kinematic acceleration, and g is the gravitational
     acceleration vector (pointing downward toward Earth center).

2. LOCAL NAVIGATION FRAME & GRAVITY COMPENSATION:
   ----------------------------------------------
   - Local Navigation Frame F_n: East-North-Up (ENU).
     * X_n: Geodetic East (E)
     * Y_n: Geodetic North (N)
     * Z_n: Local Up (U)
   - Gravitational Acceleration in ENU:
       g^n = [0, 0, -g]^T,  where g approx 9.80665 m/s^2.
   - Specific Force in ENU:
       f^n = a^n - g^n = a^n - [0, 0, -g]^T = a^n + [0, 0, g]^T.
   - EXACT DERIVATION OF KINEMATIC ACCELERATION:
       a^n = f^n + g^n = f^n - [0, 0, g]^T = [f_E^n, f_N^n, f_U^n - g]^T.
     * Note: We do NOT blindly assume a = f + g. The physical sign requires
       subtracting the upward reaction to gravity (+g) from the vertical specific
       force channel f_U^n.

3. FRAME TRANSFORMATIONS:
   -----------------------
   - Phone frame (F_p) -> Vehicle body frame (F_v):
       f^v = R_vp @ f^p
     where R_vp in SO(3) is the calibrated mounting matrix.
   - Vehicle body frame (F_v) -> Navigation frame (F_n):
       f^n = C_v^n @ f^v = C_v^n @ (R_vp @ f^p)
     where C_v^n is the Direction Cosine Matrix continuously maintained by the
     validated Stage C1 AttitudeEstimator3D.

4. NUMERICAL INTEGRATION:
   -----------------------
   - Velocity update (Trapezoidal / Heun's method, 2nd-order accurate):
       v_k^n = v_{k-1}^n + 0.5 * (a_{k-1}^n + a_k^n) * dt
   - Position update (Trapezoidal / consistent kinematic integration):
       p_k^n = p_{k-1}^n + 0.5 * (v_{k-1}^n + v_k^n) * dt
===============================================================================
"""

from typing import Dict, Optional, Tuple
import numpy as np

from src.navigation.attitude import AttitudeEstimator3D


class InertialMechanization3D:
    """
    Continuous 3D Strapdown Inertial Navigation Mechanization Engine.
    Propagates 3D position, velocity, and attitude in the local ENU navigation frame.
    """

    def __init__(
        self,
        init_pos_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_vel_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_heading_deg: float = 0.0,
        init_pitch_deg: float = 0.0,
        init_roll_deg: float = 0.0,
        R_vp: Optional[np.ndarray] = None,
        gravity: float = 9.80665
    ) -> None:
        """
        Initializes 3D inertial mechanization state.

        Args:
            init_pos_enu: Initial [East, North, Up] position in meters.
            init_vel_enu: Initial [v_East, v_North, v_Up] velocity in m/s.
            init_heading_deg: Initial vehicle heading in degrees (clockwise from North).
            init_pitch_deg: Initial vehicle pitch in degrees (positive nose up).
            init_roll_deg: Initial vehicle roll in degrees (positive right side down).
            R_vp: 3x3 mounting rotation matrix from Phone frame to Vehicle frame.
            gravity: Local gravitational acceleration magnitude (m/s^2).
        """
        self.pos_n = np.array(init_pos_enu, dtype=np.float64)
        self.vel_n = np.array(init_vel_enu, dtype=np.float64)
        self.acc_n = np.zeros(3, dtype=np.float64)
        self.g_val = float(gravity)

        # Stage C1 3D Attitude Foundation
        self.attitude = AttitudeEstimator3D(
            init_heading_deg=init_heading_deg,
            init_pitch_deg=init_pitch_deg,
            init_roll_deg=init_roll_deg,
            R_vp=R_vp
        )

        # History buffer for 2nd-order trapezoidal integration
        self._prev_acc_n = None
        self._prev_vel_n = None
        self._initialized = False

    def step(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        dt: float,
        apply_gravity_leveling: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Executes one discrete 3D inertial mechanization propagation step.

        Args:
            accel_x: Specific force along phone X-axis (m/s^2).
            accel_y: Specific force along phone Y-axis (m/s^2).
            accel_z: Specific force along phone Z-axis (m/s^2).
            gyro_x: Angular rate around phone X-axis (rad/s).
            gyro_y: Angular rate around phone Y-axis (rad/s).
            gyro_z: Angular rate around phone Z-axis (rad/s).
            dt: Timestep duration (seconds).
            apply_gravity_leveling: If True, applies Stage C1 accelerometer tilt leveling.

        Returns:
            Dictionary containing updated state:
              'pos_n': [p_East, p_North, p_Up] (m)
              'vel_n': [v_East, v_North, v_Up] (m/s)
              'acc_n': [a_East, a_North, a_Up] (m/s^2)
              'attitude': attitude telemetry dictionary
        """
        if dt <= 0.0:
            return self.get_state()

        # 1. Specific force vector in phone frame
        f_p = np.array([accel_x, accel_y, accel_z], dtype=np.float64)

        # 2. Transform to vehicle body frame: f_v = R_vp @ f_p
        f_v = self.attitude.R_vp @ f_p

        # 3. Transform to navigation frame (ENU) using current DCM: f_n = C_v^n @ f_v
        C_v_n = self.attitude.get_dcm()
        f_n = C_v_n @ f_v

        # 4. Rigorous Gravity Compensation: a^n = f^n - [0, 0, g]^T
        a_n = f_n - np.array([0.0, 0.0, self.g_val], dtype=np.float64)
        self.acc_n = a_n

        # 5. Numerical Velocity & Position Integration (Trapezoidal Rule)
        if not self._initialized:
            self._prev_acc_n = a_n.copy()
            self._prev_vel_n = self.vel_n.copy()
            self._initialized = True
        else:
            # v_k = v_{k-1} + 0.5 * (a_{k-1} + a_k) * dt
            v_curr = self.vel_n + 0.5 * (self._prev_acc_n + a_n) * dt
            # p_k = p_{k-1} + 0.5 * (v_{k-1} + v_k) * dt
            p_curr = self.pos_n + 0.5 * (self._prev_vel_n + v_curr) * dt

            self.vel_n = v_curr
            self.pos_n = p_curr
            self._prev_acc_n = a_n.copy()
            self._prev_vel_n = v_curr.copy()

        # 6. Propagate 3D Attitude for the next step using full 3-axis gyro
        self.attitude.predict(gyro_x, gyro_y, gyro_z, dt)
        if apply_gravity_leveling:
            self.attitude.update_gravity(accel_x, accel_y, accel_z, ka=0.02, accel_gate=1.5)

        return self.get_state()

    def get_state(self) -> Dict[str, any]:
        """Returns the complete navigation state."""
        return {
            'pos_n': self.pos_n.copy(),
            'vel_n': self.vel_n.copy(),
            'acc_n': self.acc_n.copy(),
            'yaw_deg': self.attitude.get_yaw_deg(),
            'pitch_deg': self.attitude.get_pitch_deg(),
            'roll_deg': self.attitude.get_roll_deg(),
            'q_norm': self.attitude.get_quaternion_norm(),
            'dcm': self.attitude.get_dcm()
        }

    def get_position(self) -> np.ndarray:
        """Returns current [East, North, Up] position vector (m)."""
        return self.pos_n.copy()

    def get_velocity(self) -> np.ndarray:
        """Returns current [v_East, v_North, v_Up] velocity vector (m/s)."""
        return self.vel_n.copy()

    def get_acceleration_nav(self) -> np.ndarray:
        """Returns current kinematic [a_East, a_North, a_Up] acceleration vector (m/s^2)."""
        return self.acc_n.copy()
