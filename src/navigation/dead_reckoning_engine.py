"""
SIH26168 - Stage C9: Unified Smartphone Dead-Reckoning Engine
Module: src/navigation/dead_reckoning_engine.py

Integrates the proven physical and algorithmic levers established across C8-12 and C8-13:
1. 15-state Error-State Kalman Filter (ESKF3D) inertial navigation core.
2. C8-12: 1-DOF Forward Velocity Update decoupled from vertical/lateral axes.
3. C8-12: Causal Pre-Outage Longitudinal Acceleration Bias Tracker (b_a,x).
4. C8-13: Decoupled 1-DOF Lateral Non-Holonomic Constraint (v_y^v approx 0)
   providing direct, undiluted first-order yaw observability (H[0, 8] = -v_fwd)
   with adaptive turn-rate noise scaling.
5. C8-13: Pre-Outage Straight-Cruising Zero Angular Rate Update (ZARU) Gyro Bias Tracker (b_g,z).
6. C8-13: Selective Environmental Quality-Gated Magnetometer Azimuth Update.
7. C8-13: Topological Road-Link Directional Guidance (OSM Edge Snapping).
8. Causal Stationary Detector & Zero Velocity Update (ZUPT).
9. GNSS Outage and Reacquisition State Machine with 6-DOF Chi-Square NIS gating.

Strict Deployment Boundary:
- Input sensors: Smartphone Accelerometer, Gyroscope, Magnetometer, Synthetic Orientation, GNSS.
- External inputs: Digital Road Vector Network (OSM).
- Absolutely zero CAN bus, zero wheel speeds, zero steering angle required in deployable mode.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.navigation.eskf import ESKF3D, rotvec_to_quat, quat_mult
from src.navigation.wheel_odometry import ChassisWheelSpeedFusion
from src.navigation.map_constraints import MapConstraintManager
from src.navigation.zupt import CausalStationaryDetector, ZeroVelocityUpdate


def update_compass_heading(eskf: ESKF3D, psi_meas_deg: float, sigma_psi_deg: float = 5.0) -> Dict[str, float]:
    """
    Performs Joseph-stabilized heading measurement update into 15-state ESKF.
    Enforces Strict Bias Freeze (K[9:15, :] = 0) to prevent heading errors from
    corrupting accelerometer and gyroscope bias states.
    """
    psi_eskf = eskf.attitude.get_yaw_deg()
    r_psi = np.radians((psi_meas_deg - psi_eskf + 180.0) % 360.0 - 180.0)

    C_v_n = eskf.attitude.get_dcm()
    H = np.zeros((1, 15), dtype=np.float64)
    H[0, 6:9] = -C_v_n[2, :]

    R = np.array([[np.radians(sigma_psi_deg)**2]], dtype=np.float64)
    S = H @ eskf.P @ H.T + R
    K = eskf.P @ H.T @ np.linalg.inv(S)

    # Strict Bias Freeze
    K[9:15, :] = 0.0

    dx = K.flatten() * r_psi

    eskf.pos_n += dx[0:3]
    eskf.vel_n += dx[3:6]
    dq = rotvec_to_quat(dx[6:9])
    eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq)
    eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)

    IKH = np.eye(15, dtype=np.float64) - K @ H
    eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
    eskf.P = 0.5 * (eskf.P + eskf.P.T)

    nis = float((r_psi**2) / S[0, 0])
    return {'r_psi_deg': float(np.degrees(r_psi)), 'nis': nis}


class NavigationState(Enum):
    ALIGNING = "ALIGNING"
    GNSS_LOCKED = "GNSS_LOCKED"
    DEAD_RECKONING = "DEAD_RECKONING"
    REACQUIRING = "REACQUIRING"


@dataclass
class EngineConfig:
    """Configuration parameters for the unified Dead-Reckoning Engine."""
    # Inertial noise densities
    sigma_accel: float = 0.291          # m/s^2 (calibrated phone accelerometer noise)
    sigma_gyro: float = 0.015           # rad/s
    gravity: float = 9.80665            # m/s^2
    dt_default: float = 0.1             # s (10 Hz nominal IMU rate)

    # C8-12 Forward Speed Fusion
    sigma_speed: float = 0.60           # m/s (1-DOF forward velocity noise)
    huber_speed_k: float = 3.0          # Huber threshold for forward speed

    # C8-13 Decoupled Lateral NHC
    sigma_lat_0: float = 0.50           # m/s (nominal lateral velocity constraint noise)
    nhc_gate_chi2: float = 9.0          # Chi-square 3-sigma gate on lateral residual
    turn_rate_scale_deg_s: float = 5.0  # Turn rate scaling for adaptive lateral noise

    # C8-13 Magnetometer Compass
    mag_norm_tol: float = 0.08          # Fraction of baseline Earth field (8%)
    mag_db_dt_tol: float = 5.0          # uT/s temporal gradient limit
    sigma_compass_deg: float = 5.0      # Compass heading noise
    compass_gate_deg: float = 25.0      # Maximum heading innovation for compass

    # C8-13 Road Guidance (OSM)
    map_search_radius_m: float = 25.0   # Search radius for candidate links
    map_heading_gate_deg: float = 30.0  # Orientation alignment gate
    sigma_map_heading_deg: float = 5.0  # Road link heading uncertainty
    map_update_interval_s: float = 1.0  # Road guidance update rate limit

    # C9.1 Decoupled Vertical NHC & Speed Bias Refinements
    enable_vert_nhc: bool = True        # Clamps vertical body velocity to prevent gravity leakage
    sigma_vert: float = 0.50            # m/s (vertical body velocity constraint noise)
    turn_rate_vnhc_gate_deg_s: float = 3.0 # deg/s (lock out vertical NHC during dynamic turns)
    turn_rate_compass_gate_deg_s: float = 3.0 # deg/s (lock out compass in turns)
    enable_causal_speed_bias: bool = False # Optional pre-outage speed bias correction
    max_speed_bias_ms: float = 1.5      # Maximum allowed speed bias clamp

    # Stationary Detection (ZUPT)
    zupt_sigma_vel: float = 0.05        # m/s velocity clamp at rest

    # Causal Lookback Buffers
    causal_lookback_s: float = 30.0     # Lookback window for pre-outage bias estimation
    min_straight_samples: int = 10      # Minimum epochs required for valid straight estimation


@dataclass
class GNSSMeasurement:
    """Standardized GNSS observation packet."""
    valid: bool
    pos_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [East, North, Up] meters
    vel_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [vE, vN, vU] m/s
    heading_deg: Optional[float] = None                                 # Ground track heading deg
    accuracy_m: float = 3.0                                            # 1-sigma horizontal accuracy


class DeadReckoningEngine:
    """
    Unified Smartphone Dead-Reckoning Navigation Engine.
    Executes real-time, strictly causal inertial dead reckoning during GNSS blockouts.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        road_index: Optional[Any] = None,
        init_pos_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_vel_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_heading_deg: float = 0.0,
        R_vp: Optional[np.ndarray] = None,
        ba_stat: Optional[np.ndarray] = None,
        bg_stat: Optional[np.ndarray] = None,
        baseline_mag_uT: float = 45.0,
        mag_declination_deg: float = 0.0
    ):
        self.cfg = config or EngineConfig()

        # Mount alignment matrix (Vehicle <- Phone)
        self.R_vp = R_vp if R_vp is not None else np.eye(3)

        # Static sensor biases
        self.ba_stat = ba_stat if ba_stat is not None else np.zeros(3)
        self.bg_stat = bg_stat if bg_stat is not None else np.zeros(3)

        # Baseline magnetic properties
        self.baseline_B = float(baseline_mag_uT)
        self.mag_declination = float(mag_declination_deg)

        # 1. Instantiate 15-state ESKF INS Core
        self.eskf = ESKF3D(
            init_pos_enu=init_pos_enu,
            init_vel_enu=init_vel_enu,
            init_heading_deg=float(init_heading_deg),
            init_pitch_deg=0.0,
            init_roll_deg=0.0,
            init_ba=self.ba_stat.copy(),
            init_bg=self.bg_stat.copy(),
            R_vp=np.eye(3),  # IMU inputs are pre-rotated by R_vp
            sigma_a=self.cfg.sigma_accel,
            sigma_g=self.cfg.sigma_gyro,
            gravity=self.cfg.gravity
        )

        # 2. Measurement Submodules
        self.speed_fusion = ChassisWheelSpeedFusion(sigma_wheel=self.cfg.sigma_speed)
        self.zupt = ZeroVelocityUpdate(sigma_vel=self.cfg.zupt_sigma_vel)
        self.stationary_detector = CausalStationaryDetector(dt=self.cfg.dt_default)

        # 3. Optional Map Constraint Manager
        self.map_mgr = None
        if road_index is not None:
            self.map_mgr = MapConstraintManager(
                road_index=road_index,
                search_radius_m=self.cfg.map_search_radius_m,
                heading_gate_deg=self.cfg.map_heading_gate_deg,
                sigma_psi_deg=self.cfg.sigma_map_heading_deg,
                update_interval_sec=self.cfg.map_update_interval_s
            )

        # State Machine Tracking
        self.nav_state = NavigationState.GNSS_LOCKED
        self.time_now_s = 0.0
        self.outage_duration_s = 0.0
        self.last_map_update_time = -1000.0

        # Causal Pre-Outage Tracking Buffers
        self.lookback_pts = int(self.cfg.causal_lookback_s / self.cfg.dt_default)
        self._buf_time: List[float] = []
        self._buf_phone_ax: List[float] = []
        self._buf_gnss_ax: List[float] = []
        self._buf_gyro_z: List[float] = []
        self._buf_speed: List[float] = []
        self._buf_turn_rate: List[float] = []

        # Locked Causal Biases for Outage Inoculation
        self.b_accel_x = 0.0    # C8-12 forward acceleration bias
        self.b_gyro_z = 0.0     # C8-13 straight ZARU gyro bias
        self.b_speed = 0.0      # C9.1 forward speed bias
        self._buf_ml_speed: List[float] = []

        # Previous magnetometer sample for gradient check
        self._last_mag = None

    def step(
        self,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        speed_est: float,
        dt: float = 0.1,
        mag_raw: Optional[np.ndarray] = None,
        gnss: Optional[GNSSMeasurement] = None,
        psi_mag_cal_deg: Optional[float] = None,
        db_dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a single navigation epoch:
        - Ingests phone IMU, magnetometer, causal speed prediction, and GNSS status.
        - Updates causal bias trackers when GNSS is available.
        - Executes decoupled 1-DOF speed and lateral NHC updates during outages.
        - Returns current state, covariance uncertainty, and diagnostic telemetry.
        """
        self.time_now_s += dt

        # 1. Transform raw phone IMU to vehicle body frame
        acc_v = self.R_vp @ accel_raw
        gyro_v = self.R_vp @ gyro_raw

        ax_raw, ay_raw, az_raw = acc_v
        gx_raw, gy_raw, gz_raw = gyro_v

        # Turn rate metric for kinematic constraints
        turn_rate_deg_s = float(np.abs(np.degrees(gz_raw)))

        # 2. Evaluate Stationary Rest Detector (ZUPT)
        det_res = self.stationary_detector.update(
            np.array([acc_v]),
            np.array([gyro_v])
        )
        is_stationary = bool(det_res['is_stationary'])

        # 3. State Machine Transitions
        is_outage = gnss is None or not gnss.valid
        if not is_outage:
            if self.nav_state == NavigationState.DEAD_RECKONING:
                self.nav_state = NavigationState.REACQUIRING
            else:
                self.nav_state = NavigationState.GNSS_LOCKED
            self.outage_duration_s = 0.0
        else:
            self.nav_state = NavigationState.DEAD_RECKONING
            self.outage_duration_s += dt

        # 4. Pre-Outage Causal Bias Tracking (Only during valid GNSS)
        if not is_outage and gnss is not None:
            v_gnss = float(np.linalg.norm(gnss.vel_enu[:2]))
            # Maintain rolling causal history
            self._buf_time.append(self.time_now_s)
            self._buf_phone_ax.append(ax_raw)
            self._buf_gyro_z.append(gz_raw)
            self._buf_speed.append(v_gnss)
            self._buf_turn_rate.append(turn_rate_deg_s)
            self._buf_ml_speed.append(speed_est)

            if len(self._buf_time) > self.lookback_pts:
                self._buf_time.pop(0)
                self._buf_phone_ax.pop(0)
                self._buf_gyro_z.pop(0)
                self._buf_speed.pop(0)
                self._buf_turn_rate.pop(0)
                self._buf_ml_speed.pop(0)

            # Update C8-12, C8-13, and C9.1 causal estimates
            if len(self._buf_speed) >= self.cfg.min_straight_samples:
                speeds = np.array(self._buf_speed)
                trs = np.array(self._buf_turn_rate)
                st_mask = (speeds >= 3.0) & (trs < 2.0)

                if np.sum(st_mask) >= self.cfg.min_straight_samples:
                    # GNSS speed derivative
                    a_gnss = np.gradient(speeds, dt)
                    p_ax = np.array(self._buf_phone_ax)
                    gzs = np.array(self._buf_gyro_z)
                    ml_spds = np.array(self._buf_ml_speed)

                    # C8-12 Causal Longitudinal Accel Bias
                    self.b_accel_x = float(np.mean(p_ax[st_mask] - a_gnss[st_mask]))

                    # C8-13 Causal Straight ZARU Gyro Bias
                    self.b_gyro_z = float(np.mean(gzs[st_mask]))

                    # C9.1 Causal Speed Bias
                    self.b_speed = float(np.clip(
                        np.mean(speeds[st_mask] - ml_spds[st_mask]),
                        -self.cfg.max_speed_bias_ms,
                        self.cfg.max_speed_bias_ms
                    ))

        # 5. Strapdown Inertial Prediction (With Locked Causal Biases)
        ax_corr = ax_raw
        gz_corr = gz_raw
        if self.nav_state == NavigationState.DEAD_RECKONING:
            ax_corr = ax_raw - self.b_accel_x
            gz_corr = gz_raw - self.b_gyro_z

        self.eskf.predict(ax_corr, ay_raw, az_raw, gx_raw, gy_raw, gz_corr, dt)

        # 6. Measurement Updates
        diagnostics: Dict[str, Any] = {
            'nav_state': self.nav_state.value,
            'outage_duration_s': self.outage_duration_s,
            'b_accel_x_applied': self.b_accel_x if is_outage else 0.0,
            'b_gyro_z_applied': self.b_gyro_z if is_outage else 0.0,
            'b_speed_applied': self.b_speed if is_outage else 0.0,
            'nhc_active': False,
            'vnhc_active': False,
            'compass_active': False,
            'map_active': False,
            'zupt_active': False,
            'gnss_active': False
        }

        if is_stationary:
            # Stationary ZUPT clamp
            self.zupt.update_eskf(self.eskf)
            diagnostics['zupt_active'] = True

        elif not is_outage and gnss is not None:
            # Full 6-DOF GNSS Update with Chi-square NIS Check
            z_pos = gnss.pos_enu
            z_vel = gnss.vel_enu
            z_meas = np.concatenate([z_pos, z_vel])
            y_innov = z_meas - np.concatenate([self.eskf.pos_n, self.eskf.vel_n])
            S = self.eskf.H @ self.eskf.P @ self.eskf.H.T + self.eskf.R
            try:
                S_inv = np.linalg.inv(S)
                nis = float(y_innov.T @ S_inv @ y_innov)
            except np.linalg.LinAlgError:
                nis = 0.0

            # Execute GNSS update injection
            self.eskf.update_gnss(z_pos, z_vel)
            diagnostics['gnss_active'] = True
            diagnostics['gnss_nis'] = nis

        else:
            # DEAD RECKONING MODE: Unified Sensor-Fusion Stack

            # A. C8-12 Decoupled 1-DOF Forward Velocity Update
            spd_use = speed_est
            if self.cfg.enable_causal_speed_bias:
                spd_use = max(0.0, speed_est + self.b_speed)
            self.speed_fusion.update_eskf_forward_velocity(self.eskf, spd_use, apply_nis_gate=True)

            # B. C8-13 Decoupled 1-DOF Lateral Non-Holonomic Constraint
            # Injects zero lateral velocity constraint with direct first-order yaw observability
            nhc_res = self._update_lateral_nhc(spd_use, turn_rate_deg_s)
            diagnostics['nhc_active'] = nhc_res['active']
            if nhc_res['active']:
                diagnostics['nhc_nis'] = nhc_res['nis']

            # C. C9.1 Decoupled 1-DOF Vertical Non-Holonomic Constraint
            # Clamps vertical velocity (v_z^v approx 0) to prevent gravity leakage runaway
            if self.cfg.enable_vert_nhc:
                vnhc_res = self._update_vertical_nhc(turn_rate_deg_s)
                diagnostics['vnhc_active'] = vnhc_res['active']
                if vnhc_res['active']:
                    diagnostics['vnhc_nis'] = vnhc_res['nis']

            # D. C8-13 Selective Quality-Gated Compass
            if mag_raw is not None and psi_mag_cal_deg is not None:
                compass_res = self._update_selective_compass(
                    mag_raw, psi_mag_cal_deg, dt, turn_rate_deg_s=turn_rate_deg_s, db_dt_override=db_dt
                )
                diagnostics['compass_active'] = compass_res['active']

            # E. C8-13 Topological OSM Road Heading Guidance
            if self.map_mgr is not None and (self.time_now_s - self.last_map_update_time >= self.cfg.map_update_interval_s - 1e-4):
                map_res = self._update_road_heading()
                diagnostics['map_active'] = map_res['active']
                if map_res['active']:
                    self.last_map_update_time = self.time_now_s

        # 7. Package Output Telemetry
        st = self.eskf.get_state()
        pos_cov = self.eskf.P[0:3, 0:3]
        pos_sigma = float(np.sqrt(np.trace(pos_cov[:2, :2])))
        yaw_sigma = float(np.degrees(np.sqrt(self.eskf.P[8, 8])))

        return {
            'time_s': self.time_now_s,
            'pos_enu': self.eskf.pos_n.copy(),
            'vel_enu': self.eskf.vel_n.copy(),
            'heading_deg': float(st['yaw_deg']),
            'pitch_deg': float(st['pitch_deg']),
            'roll_deg': float(st['roll_deg']),
            'pos_sigma_m': pos_sigma,
            'heading_sigma_deg': yaw_sigma,
            'diagnostics': diagnostics
        }

    def get_state(self) -> Dict[str, Any]:
        """Returns the current estimated navigation state and uncertainties."""
        st = self.eskf.get_state()
        pos_cov = self.eskf.P[0:3, 0:3]
        pos_sigma = float(np.sqrt(np.trace(pos_cov[:2, :2])))
        yaw_sigma = float(np.degrees(np.sqrt(self.eskf.P[8, 8])))
        return {
            'time_s': self.time_now_s,
            'pos_enu': self.eskf.pos_n.copy(),
            'vel_enu': self.eskf.vel_n.copy(),
            'heading_deg': float(st['yaw_deg']),
            'pitch_deg': float(st['pitch_deg']),
            'roll_deg': float(st['roll_deg']),
            'pos_sigma_m': pos_sigma,
            'heading_sigma_deg': yaw_sigma,
            'nav_state': self.nav_state.value
        }

    # =========================================================================
    # Internal Aiding Update Methods
    # =========================================================================

    def _update_lateral_nhc(self, v_fwd_est: float, tr_deg_s: float) -> Dict[str, Any]:
        """Executes decoupled 1-DOF lateral velocity constraint with adaptive turn-rate noise."""
        C_v_n = self.eskf.attitude.get_dcm()
        C_n_v = C_v_n.T
        vel_n = self.eskf.vel_n
        vel_v = C_n_v @ vel_n

        r_lat = -vel_v[1]

        # Adaptive lateral uncertainty during dynamic turns
        sigma_lat = self.cfg.sigma_lat_0 * (1.0 + (tr_deg_s / self.cfg.turn_rate_scale_deg_s)**2)
        R = np.array([[sigma_lat**2]], dtype=np.float64)

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 3:6] = C_n_v[1, :]
        vx_use = vel_v[0] if abs(vel_v[0]) > 0.5 else v_fwd_est
        H[0, 6:9] = np.array([vel_v[2], 0.0, -vx_use], dtype=np.float64)

        S = H @ self.eskf.P @ H.T + R
        S_inv = 1.0 / float(S[0, 0])
        nis = float((r_lat**2) * S_inv)

        if nis > self.cfg.nhc_gate_chi2 or abs(r_lat) > 2.5:
            return {'active': False, 'reason': 'gated', 'nis': nis}

        k_huber = 2.0
        gamma = min(1.0, k_huber / np.sqrt(max(nis, 1e-12))) if nis > (k_huber**2) else 1.0

        K = (self.eskf.P @ H.T * S_inv) * gamma
        K[9:15, :] = 0.0  # Strict Bias Freeze

        dx = (K * r_lat).flatten()
        self.eskf.pos_n += dx[0:3]
        self.eskf.vel_n += dx[3:6]
        dq = rotvec_to_quat(dx[6:9])
        self.eskf.attitude.q_nv = quat_mult(self.eskf.attitude.q_nv, dq)
        self.eskf.attitude.q_nv /= np.linalg.norm(self.eskf.attitude.q_nv)

        IKH = np.eye(15, dtype=np.float64) - K @ H
        self.eskf.P = IKH @ self.eskf.P @ IKH.T + K @ R @ K.T
        self.eskf.P = 0.5 * (self.eskf.P + self.eskf.P.T)

        return {'active': True, 'nis': nis, 'r_lat': r_lat}

    def _update_vertical_nhc(self, turn_rate_deg_s: float = 0.0) -> Dict[str, Any]:
        """
        Executes decoupled 1-DOF vertical body velocity constraint (v_z^v approx 0).
        Gated during dynamic turns (when roll/pitch dynamics induce apparent vertical motion).
        Enforces Strict Position, Attitude and Bias Freeze (K[0:3, :] = 0, K[6:15, :] = 0)
        so vertical velocity innovations can NEVER directly shift position or corrupt attitude.
        """
        if turn_rate_deg_s > self.cfg.turn_rate_vnhc_gate_deg_s:
            return {'active': False, 'reason': 'turning'}

        C_v_n = self.eskf.attitude.get_dcm()
        C_n_v = C_v_n.T
        vel_v = C_n_v @ self.eskf.vel_n

        r_vert = float(-vel_v[2])
        R = np.array([[self.cfg.sigma_vert**2]], dtype=np.float64)

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 3:6] = C_n_v[2, :]

        S = H @ self.eskf.P @ H.T + R
        S_inv = 1.0 / float(S[0, 0])
        nis = float((r_vert**2) * S_inv)

        if nis > self.cfg.nhc_gate_chi2 or abs(r_vert) > 3.0:
            return {'active': False, 'reason': 'gated', 'nis': nis}

        k_huber = 2.0
        gamma = min(1.0, k_huber / np.sqrt(max(nis, 1e-12))) if nis > (k_huber**2) else 1.0

        K = (self.eskf.P @ H.T * S_inv) * gamma
        # Strict Position, Attitude & Bias Freeze:
        # Vertical body velocity must ONLY update velocity states (indices 3:6)
        K[0:3, :] = 0.0
        K[6:15, :] = 0.0

        dx = (K * r_vert).flatten()
        self.eskf.vel_n += dx[3:6]

        IKH = np.eye(15, dtype=np.float64) - K @ H
        self.eskf.P = IKH @ self.eskf.P @ IKH.T + K @ R @ K.T
        self.eskf.P = 0.5 * (self.eskf.P + self.eskf.P.T)

        return {'active': True, 'nis': nis, 'r_vert': r_vert}

    def _update_selective_compass(
        self,
        mag_raw: np.ndarray,
        psi_mag_cal_deg: float,
        dt: float,
        turn_rate_deg_s: float = 0.0,
        db_dt_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Executes environmental quality-gated compass update with Strict Bias Freeze."""
        # Lock out compass during dynamic turns
        if turn_rate_deg_s > self.cfg.turn_rate_compass_gate_deg_s:
            return {'active': False, 'reason': 'turning'}

        mag_norm = float(np.linalg.norm(mag_raw))
        if db_dt_override is not None:
            db_dt = float(db_dt_override)
        else:
            db_dt = float(np.linalg.norm(mag_raw - self._last_mag) / dt) if self._last_mag is not None else 0.0
        self._last_mag = mag_raw.copy()

        # Strict environmental cleanliness gates
        norm_diff = abs(mag_norm - self.baseline_B) / self.baseline_B
        if norm_diff > self.cfg.mag_norm_tol or db_dt > self.cfg.mag_db_dt_tol:
            return {'active': False, 'reason': 'magnetic_anomaly'}

        # Innovation check against current ESKF heading
        cur_yaw = self.eskf.attitude.get_yaw_deg()
        yaw_diff = abs((psi_mag_cal_deg - cur_yaw + 180.0) % 360.0 - 180.0)
        if yaw_diff > self.cfg.compass_gate_deg:
            return {'active': False, 'reason': 'innovation_gate'}

        # Joseph-stabilized compass update
        res = update_compass_heading(self.eskf, psi_mag_cal_deg, sigma_psi_deg=self.cfg.sigma_compass_deg)
        return {'active': True, 'nis': res['nis']}

    def _update_road_heading(self) -> Dict[str, Any]:
        """Executes OSM road-link tangent heading update."""
        if self.map_mgr is None:
            return {'active': False, 'reason': 'no_map'}

        st = self.eskf.get_state()
        candidate = self.map_mgr.select_candidate(st['pos_n'], st['yaw_deg'])
        if candidate is None:
            return {'active': False, 'reason': 'no_candidate'}

        y_psi, H_psi = self.map_mgr.compute_heading_innovation(st['yaw_deg'], candidate)
        R_psi = np.array([[self.map_mgr.sigma_psi ** 2]], dtype=np.float64)
        S = H_psi @ self.eskf.P @ H_psi.T + R_psi
        S_inv = 1.0 / float(S[0, 0])
        nis = float((y_psi ** 2) * S_inv)

        if nis > self.map_mgr.nis_gate_1dof:
            return {'active': False, 'reason': 'nis_gated', 'nis': nis}

        K = self.eskf.P @ H_psi.T * S_inv
        K[9:15, :] = 0.0  # Strict Bias Freeze

        delta_x = (K * y_psi).flatten()
        self.eskf.pos_n += delta_x[0:3]
        self.eskf.vel_n += delta_x[3:6]
        dq_corr = rotvec_to_quat(delta_x[6:9])
        self.eskf.attitude.q_nv = quat_mult(self.eskf.attitude.q_nv, dq_corr)
        self.eskf.attitude.q_nv /= np.linalg.norm(self.eskf.attitude.q_nv)

        IKH = np.eye(15, dtype=np.float64) - K @ H_psi
        self.eskf.P = IKH @ self.eskf.P @ IKH.T + K @ R_psi @ K.T
        self.eskf.P = 0.5 * (self.eskf.P + self.eskf.P.T)

        return {'active': True, 'nis': nis}

    def reset(
        self,
        pos_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vel_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        heading_deg: float = 0.0
    ):
        """Resets engine state and covariance for a new trip or trajectory initialization."""
        self.eskf = ESKF3D(
            init_pos_enu=pos_enu,
            init_vel_enu=vel_enu,
            init_heading_deg=float(heading_deg),
            init_pitch_deg=0.0,
            init_roll_deg=0.0,
            init_ba=self.ba_stat.copy(),
            init_bg=self.bg_stat.copy(),
            R_vp=np.eye(3),
            sigma_a=self.cfg.sigma_accel,
            sigma_g=self.cfg.sigma_gyro,
            gravity=self.cfg.gravity
        )
        self.nav_state = NavigationState.GNSS_LOCKED
        self.time_now_s = 0.0
        self.outage_duration_s = 0.0
        self.b_accel_x = 0.0
        self.b_gyro_z = 0.0
        self.b_speed = 0.0
        self._buf_time.clear()
        self._buf_phone_ax.clear()
        self._buf_gyro_z.clear()
        self._buf_speed.clear()
        self._buf_turn_rate.clear()
        self._buf_ml_speed.clear()
        self._last_mag = None
