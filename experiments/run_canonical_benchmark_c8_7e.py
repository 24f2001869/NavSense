"""
SIH26168 - Stage C8-7E: Canonical Frozen Benchmark & Reproducibility Suite
Module: experiments/run_canonical_benchmark_c8_7e.py

Implements the single canonical, frozen evaluation protocol reconciling C8-7C and C8-7D:
- Enforces Strict Bias Freeze (K[9:15, :] = 0.0) across ALL velocity and NHC updates.
- Uses update_eskf_3d_velocity for joint Speed+NHC conditions (matching C8-7C to machine precision).
- Uses decoupled 1D Speed and 2D NHC (both with Strict Bias Freeze) for isolated ablation conditions.
- Evaluates across identical non-overlapping outage windows (5s, 10s, 20s, 30s, 60s) on Vta04 and Vta02.
- Exports canonical structured JSON results and benchmark tables.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D, rotvec_to_quat, quat_mult
from src.navigation.nhc import NonHolonomicConstraint, compute_nhc_residual_and_jacobian
from src.navigation.zupt import ZeroVelocityUpdate, CausalStationaryDetector
from src.navigation.wheel_odometry import (
    ChassisWheelSpeedFusion,
    compute_velocity_residual_and_jacobian_1d,
    compute_velocity_residual_and_jacobian_3d
)
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex
from src.navigation.map_constraints import MHTMapConstraintManager
from experiments.run_wheel_speed_fusion_c8_5 import compute_bcac_series
from experiments.evaluate_phone_speed_c8_7 import prepare_trip_phone_data

RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


class PhoneCompassFusion:
    """Confidence-Gated Smartphone Compass Fusion Engine for 15-State ESKF."""
    def __init__(self, sigma_psi_deg: float = 5.0):
        self.sigma_psi_rad = np.radians(sigma_psi_deg)

    def check_gates(
        self,
        b_norm: float,
        db_dt: float,
        gyro_yaw_degs: float,
        dpsi_mag_degs: float,
        dip_deg: float
    ) -> bool:
        """Evaluates the 4 experimental confidence hypotheses."""
        g1 = abs(b_norm - 48.8) <= 6.0          # Field Norm Anomaly
        g2 = db_dt <= 15.0                      # Temporal Gradient
        g3 = abs(dpsi_mag_degs - gyro_yaw_degs) <= 30.0 # Turn Rate Innovation
        g4 = abs(dip_deg - 68.1) <= 12.0        # Magnetic Dip Consistency
        return g1 and g2 and g3 and g4

    def update_eskf(self, eskf: ESKF3D, psi_mag_deg: float) -> Dict[str, Any]:
        """Performs Joseph-stabilized heading measurement update into ESKF."""
        psi_eskf = eskf.attitude.get_yaw_deg()
        r_psi = np.radians((psi_mag_deg - psi_eskf + 180.0) % 360.0 - 180.0)

        C_v_n = eskf.attitude.get_dcm()
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 6:9] = -C_v_n[2, :] # negative vertical component projected onto body frame

        R = np.array([[self.sigma_psi_rad**2]], dtype=np.float64)
        S = H @ eskf.P @ H.T + R
        K = eskf.P @ H.T @ np.linalg.inv(S)

        dx = K.flatten() * r_psi

        # Error state injection
        eskf.pos_n += dx[0:3]
        eskf.vel_n += dx[3:6]
        dq = rotvec_to_quat(dx[6:9])
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        eskf.ba += dx[9:12]
        eskf.bg += dx[12:15]

        # Joseph covariance update
        IKH = np.eye(15, dtype=np.float64) - K @ H
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {'r_psi_deg': float(np.degrees(r_psi)), 'nis': float((r_psi**2) / S[0, 0])}


def update_nhc_with_bias_freeze(eskf: ESKF3D, sigma_lat: float = 0.5, sigma_vert: float = 0.5) -> None:
    """Standalone 2-DOF NHC update with Strict Bias Freeze to prevent bias runaway."""
    C_v_n = eskf.attitude.get_dcm()
    vel_n = eskf.vel_n
    r_nhc, H_nhc, vel_v = compute_nhc_residual_and_jacobian(C_v_n, vel_n)

    R = np.diag([sigma_lat**2, sigma_vert**2])
    S = H_nhc @ eskf.P @ H_nhc.T + R
    K = eskf.P @ H_nhc.T @ np.linalg.inv(S)

    # Strict Bias Freeze: Velocity constraints must NOT update accelerometer or gyro biases
    K[9:15, :] = 0.0

    delta_x = K @ r_nhc

    # State error injection
    eskf.pos_n += delta_x[0:3]
    eskf.vel_n += delta_x[3:6]
    dtheta_b = delta_x[6:9]
    dq_corr = rotvec_to_quat(dtheta_b)
    eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
    eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)

    # Joseph covariance update
    IKH = np.eye(15, dtype=np.float64) - K @ H_nhc
    eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
    eskf.P = 0.5 * (eskf.P + eskf.P.T)


def prepare_canonical_data(trip_name: str, rf_model: RandomForestRegressor, dt: float = 0.1) -> Dict[str, Any]:
    """Ingests trip telemetry, OSM map, extracts causal ML speed, and precomputes series."""
    print(f"[{trip_name}] Loading trip data for canonical benchmark...", flush=True)
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    p_data = prepare_trip_phone_data(trip_name)
    acc_v = p_data['acc_v']
    gyro_v = p_data['gyro_v']
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    # Precomputed ML predicted forward speed
    ml_speed = np.maximum(0.0, rf_model.predict(p_data['features']))
    ml_speed[p_data['stat_mask']] = 0.0

    # CAN wheel speed reference
    wheel_rl = df_v['wheel_rl_rads'].values
    wheel_rr = df_v['wheel_rr_rads'].values
    wheel_speed = 0.5 * (wheel_rl + wheel_rr) * 0.2766

    # Magnetic channels for compass fusion
    raw_mag = df_p[['mag_x', 'mag_y', 'mag_z']].values.astype(np.float64)
    c_mag = np.array([-16.59, -33.41, 6.94])
    mag_c = raw_mag - c_mag
    raw_psi = np.degrees(np.arctan2(-mag_c[:, 0], -mag_c[:, 2])) % 360.0
    psi_rad = np.radians(raw_psi)
    dev = 1.60 + 27.20 * np.sin(psi_rad) - 0.53 * np.cos(psi_rad) + 1.05 * np.sin(2 * psi_rad) + 0.28 * np.cos(2 * psi_rad)
    cal_mag_heading = (raw_psi - dev) % 360.0

    b_norm = np.linalg.norm(raw_mag, axis=1)
    db_dt = np.zeros(n)
    db_dt[1:] = np.linalg.norm(np.diff(raw_mag, axis=0), axis=1) / dt
    dip_deg = np.degrees(np.arctan2(-raw_mag[:, 1], np.sqrt(raw_mag[:, 0]**2 + raw_mag[:, 2]**2)))
    dpsi_mag = p_data['dpsi_mag']

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    osm_file = REPO_ROOT / "data" / "raw" / "maps" / f"{trip_name}_network.osm"
    if not osm_file.exists():
        raise FileNotFoundError(f"OSM vector cache not found: {osm_file}")
    parsed_map = parse_osm_network(osm_file, lat0, lon0)
    road_index = RoadNetworkIndex(parsed_map['segments'])

    return {
        'trip_name': trip_name,
        'acc_v': acc_v,
        'gyro_v': gyro_v,
        'speed': speed,
        'heading': heading,
        'ml_speed': ml_speed,
        'wheel_speed': wheel_speed,
        'cal_mag_heading': cal_mag_heading,
        'b_norm': b_norm,
        'db_dt': db_dt,
        'dip_deg': dip_deg,
        'dpsi_mag': dpsi_mag,
        'gt_e': gt_e,
        'gt_n': gt_n,
        'gt_u': gt_u,
        'gt_ve': gt_ve,
        'gt_vn': gt_vn,
        'gt_vu': gt_vu,
        'ba_stat': p_data['ba_stat'],
        'bg_stat': p_data['bg_stat'],
        'sigma_series': sigma_series,
        'road_index': road_index,
        'regimes': p_data['regimes'],
        'n': n
    }


def simulate_canonical_window(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    flags: Dict[str, bool],
    dt: float = 0.1
) -> Dict[str, Any]:
    """
    Simulates a GNSS blackout window under the canonical evaluation protocol:
    - If use_speed and use_nhc are both True: uses coupled update_eskf_3d_velocity (matches C8-7C).
    - If use_speed is True and use_nhc is False: uses 1-DOF forward velocity update with Bias Freeze.
    - If use_nhc is True and use_speed is False: uses 2-DOF NHC update with Bias Freeze.
    - If use_compass is True: applies gated compass heading update.
    - If use_zupt is True: applies stationary ZUPT updates.
    - If use_map is True: applies MHT road network constraint (1.0s interval).
    """
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    speed = trip_data['speed']
    heading = trip_data['heading']
    ml_speed = trip_data['ml_speed']
    wheel_speed = trip_data['wheel_speed']
    cal_mag_heading = trip_data['cal_mag_heading']
    b_norm = trip_data['b_norm']
    db_dt = trip_data['db_dt']
    dip_deg = trip_data['dip_deg']
    dpsi_mag = trip_data['dpsi_mag']
    gt_e = trip_data['gt_e']
    gt_n = trip_data['gt_n']
    gt_u = trip_data['gt_u']
    gt_ve = trip_data['gt_ve']
    gt_vn = trip_data['gt_vn']
    gt_vu = trip_data['gt_vu']
    ba_stat = trip_data['ba_stat']
    sigma_series = trip_data['sigma_series']
    road_index = trip_data['road_index']
    regimes = trip_data['regimes']

    init_h = float(heading[kw])

    eskf = ESKF3D(
        init_pos_enu=(gt_e[kw], gt_n[kw], gt_u[kw]),
        init_vel_enu=(gt_ve[kw], gt_vn[kw], gt_vu[kw]),
        init_heading_deg=init_h,
        init_pitch_deg=0.0,
        init_roll_deg=0.0,
        init_ba=ba_stat.copy(),
        R_vp=np.eye(3),
        sigma_a=0.291,
        gravity=9.80665
    )

    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )

    phone_speed_fusion = ChassisWheelSpeedFusion(sigma_wheel=2.00, sigma_lat=0.50, sigma_vert=0.50)
    can_wheel_fusion = ChassisWheelSpeedFusion(sigma_wheel=0.20, sigma_lat=0.50, sigma_vert=0.50)
    compass_fusion = PhoneCompassFusion(sigma_psi_deg=5.0)

    map_mgr = None
    if flags.get('use_map', False):
        map_mgr = MHTMapConstraintManager(
            road_index=road_index,
            beam_width_K=3,
            search_radius_m=25.0,
            heading_gate_deg=30.0,
            commitment_margin_delta=0.20,
            sigma_lane_m=2.5,
            sigma_psi_deg=5.0,
            lane_offset_m=-0.90,
            n_persist=2,
            max_pos_correction_m=5.0,
            update_interval_sec=1.0
        )

    t_start = kw * dt
    last_map_time = -np.inf

    pos_errors = []
    along_errors = []
    cross_errors = []
    vel_errors = []
    heading_errors = []

    use_speed = flags.get('use_speed', False)
    use_nhc = flags.get('use_nhc', False)
    use_can_ref = flags.get('use_can_ref', False)

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]

        # 1. ESKF strapdown inertial propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Canonical Velocity / NHC Update
        cur_spd = float(wheel_speed[step_k]) if use_can_ref else float(ml_speed[step_k])
        fusion_engine = can_wheel_fusion if use_can_ref else phone_speed_fusion

        if use_speed and use_nhc:
            # Joint 3D Body Velocity Update (coupled with Strict Bias Freeze)
            fusion_engine.update_eskf_3d_velocity(eskf, cur_spd, apply_nis_gate=True)
        elif use_speed and not use_nhc:
            # 1-DOF Forward Velocity Only (Strict Bias Freeze)
            fusion_engine.update_eskf_forward_velocity(eskf, cur_spd, apply_nis_gate=True)
        elif not use_speed and use_nhc:
            # 2-DOF NHC Only (Strict Bias Freeze)
            update_nhc_with_bias_freeze(eskf, sigma_lat=0.50, sigma_vert=0.50)

        # 3. Confidence-Gated Compass Heading Update
        if flags.get('use_compass', False):
            if compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], np.degrees(gz), dpsi_mag[step_k], dip_deg[step_k]):
                compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        # 4. Zero Velocity Update (ZUPT)
        if flags.get('use_zupt', False):
            k_start = max(0, step_k - 10)
            det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
            if det_res['is_stationary']:
                zupt.update_eskf(eskf)

        # 5. Map Matching Constraint (1.0s interval)
        if map_mgr is not None and (t_now - last_map_time >= 0.999):
            last_map_time = t_now
            map_mgr.evaluate_and_update(
                eskf=eskf,
                t_now=t_now,
                mode='joint',
                speed_ms=cur_spd,
                gyro_z_rads=gz,
                shadow_mode=False
            )

        # Metrics recording
        e_err = eskf.pos_n[0] - gt_e[step_k]
        n_err = eskf.pos_n[1] - gt_n[step_k]
        pos_2d_err = np.hypot(e_err, n_err)
        pos_errors.append(pos_2d_err)

        h_gt_rad = np.radians(heading[step_k])
        unit_along = np.array([np.sin(h_gt_rad), np.cos(h_gt_rad)])
        unit_cross = np.array([np.cos(h_gt_rad), -np.sin(h_gt_rad)])
        err_vec = np.array([e_err, n_err])
        along_errors.append(float(np.dot(err_vec, unit_along)))
        cross_errors.append(float(np.dot(err_vec, unit_cross)))

        ve_err = eskf.vel_n[0] - gt_ve[step_k]
        vn_err = eskf.vel_n[1] - gt_vn[step_k]
        vel_errors.append(float(np.hypot(ve_err, vn_err)))

        psi_eskf = eskf.attitude.get_yaw_deg()
        dh = abs((psi_eskf - heading[step_k] + 180.0) % 360.0 - 180.0)
        heading_errors.append(float(dh))

    dist_traveled = float(np.sum(speed[kw : kw + w_dur]) * dt)
    final_pos_err = pos_errors[-1]
    drift_pct = (final_pos_err / max(dist_traveled, 10.0)) * 100.0

    mid_reg = regimes[kw + w_dur // 2]
    primary_regime = mid_reg[0] if (mid_reg and len(mid_reg) > 0) else 'general'

    return {
        'kw': kw,
        'duration_s': w_dur * dt,
        'dist_traveled_m': dist_traveled,
        'final_pos_err_m': float(final_pos_err),
        'mean_pos_err_m': float(np.mean(pos_errors)),
        'final_along_err_m': float(along_errors[-1]),
        'final_cross_err_m': float(cross_errors[-1]),
        'final_vel_err_ms': float(vel_errors[-1]),
        'final_heading_err_deg': float(heading_errors[-1]),
        'drift_pct': float(drift_pct),
        'primary_regime': primary_regime
    }


def run_canonical_benchmark():
    """Executes the master Stage C8-7E Canonical Benchmark."""
    print("=================================================================", flush=True)
    print("STAGE C8-7E: CANONICAL BENCHMARK & REPRODUCIBILITY AUDIT", flush=True)
    print("=================================================================", flush=True)

    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])

    d_vta04 = prepare_canonical_data('Vta04', rf_model)
    d_vta02 = prepare_canonical_data('Vta02', rf_model)

    horizons = [5, 10, 20, 30, 60]

    CONDITIONS = {
        'A0_Baseline_PureIMU': {
            'use_speed': False, 'use_compass': False, 'use_nhc': False, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Pure Strapdown IMU Dead Reckoning'
        },
        'B1_Speed_Only': {
            'use_speed': True, 'use_compass': False, 'use_nhc': False, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + RF Speed Alone (1D Forward)'
        },
        'B2_Compass_Only': {
            'use_speed': False, 'use_compass': True, 'use_nhc': False, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + Gated Compass'
        },
        'B3_NHC_Only': {
            'use_speed': False, 'use_compass': False, 'use_nhc': True, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + NHC Alone (2D with Bias Freeze)'
        },
        'C1_Speed_Compass': {
            'use_speed': True, 'use_compass': True, 'use_nhc': False, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + Speed + Compass'
        },
        'C2_Speed_Compass_NHC': {
            'use_speed': True, 'use_compass': True, 'use_nhc': True, 'use_zupt': False, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + Speed + Compass + NHC'
        },
        'C3_Speed_Compass_NHC_ZUPT': {
            'use_speed': True, 'use_compass': True, 'use_nhc': True, 'use_zupt': True, 'use_map': False, 'use_can_ref': False,
            'desc': 'Baseline + Speed + Compass + NHC + ZUPT'
        },
        'C4_Full_Smartphone_Map': {
            'use_speed': True, 'use_compass': True, 'use_nhc': True, 'use_zupt': True, 'use_map': True, 'use_can_ref': False,
            'desc': 'Full Smartphone Stack (+ Map Matching)'
        },
        'D1_Ablate_Minus_Speed': {
            'use_speed': False, 'use_compass': True, 'use_nhc': True, 'use_zupt': True, 'use_map': True, 'use_can_ref': False,
            'desc': 'Full Stack MINUS Speed'
        },
        'D2_Ablate_Minus_Compass': {
            'use_speed': True, 'use_compass': False, 'use_nhc': True, 'use_zupt': True, 'use_map': True, 'use_can_ref': False,
            'desc': 'Full Stack MINUS Compass'
        },
        'D3_Ablate_Minus_NHC': {
            'use_speed': True, 'use_compass': True, 'use_nhc': False, 'use_zupt': True, 'use_map': True, 'use_can_ref': False,
            'desc': 'Full Stack MINUS NHC'
        },
        'D5_Ablate_Minus_Map': {
            'use_speed': True, 'use_compass': True, 'use_nhc': True, 'use_zupt': True, 'use_map': False, 'use_can_ref': False,
            'desc': 'Full Stack MINUS Map'
        },
        'REF_CAN_Wheel_Full': {
            'use_speed': True, 'use_compass': True, 'use_nhc': True, 'use_zupt': True, 'use_map': True, 'use_can_ref': True,
            'desc': 'Reference Ceiling (CAN Wheel Speed + Full Stack)'
        }
    }

    all_canonical_results = {
        'metadata': {
            'stage': 'C8-7E',
            'date': 'September 7, 2026',
            'horizons_s': horizons,
            'conditions': {k: v['desc'] for k, v in CONDITIONS.items()}
        },
        'trips': {}
    }

    for trip_data, tname in [(d_vta04, 'Vta04'), (d_vta02, 'Vta02')]:
        print(f"\n=================================================================", flush=True)
        print(f"RUNNING CANONICAL PROTOCOL ON: {tname}", flush=True)
        print(f"=================================================================", flush=True)

        trip_summary = {}

        for dur_s in horizons:
            w_dur = int(dur_s / 0.1)
            step_stride = w_dur
            n_windows = (trip_data['n'] - w_dur) // step_stride
            if n_windows == 0:
                continue

            cond_metrics = {}

            for c_key, c_cfg in CONDITIONS.items():
                win_res_list = []
                for w_idx in range(n_windows):
                    kw = w_idx * step_stride
                    res = simulate_canonical_window(trip_data, kw, w_dur, c_cfg)
                    win_res_list.append(res)

                final_errs = [r['final_pos_err_m'] for r in win_res_list]
                along_errs = [abs(r['final_along_err_m']) for r in win_res_list]
                cross_errs = [abs(r['final_cross_err_m']) for r in win_res_list]
                vel_errs = [r['final_vel_err_ms'] for r in win_res_list]
                head_errs = [r['final_heading_err_deg'] for r in win_res_list]
                drift_pcts = [r['drift_pct'] for r in win_res_list]

                cond_metrics[c_key] = {
                    'description': c_cfg['desc'],
                    'windows_count': n_windows,
                    'mean_pos_err_m': float(np.mean(final_errs)),
                    'p50_pos_err_m': float(np.percentile(final_errs, 50)),
                    'p90_pos_err_m': float(np.percentile(final_errs, 90)),
                    'p95_pos_err_m': float(np.percentile(final_errs, 95)),
                    'mean_along_err_m': float(np.mean(along_errs)),
                    'mean_cross_err_m': float(np.mean(cross_errs)),
                    'mean_vel_err_ms': float(np.mean(vel_errs)),
                    'mean_head_err_deg': float(np.mean(head_errs)),
                    'mean_drift_pct': float(np.mean(drift_pcts)),
                    'p50_drift_pct': float(np.percentile(drift_pcts, 50)),
                    'p90_drift_pct': float(np.percentile(drift_pcts, 90)),
                    'pass_sih_10pct': bool(np.mean(drift_pcts) < 10.0)
                }

            trip_summary[f"{dur_s}s"] = cond_metrics

            print(f"\n--- {tname} [{dur_s}s Horizon, {n_windows} Windows] ---", flush=True)
            print(f"{'Condition':<28} | {'Pos Err (m)':<11} | {'Along (m)':<10} | {'Cross (m)':<10} | {'Head (deg)':<10} | {'Drift %':<8}", flush=True)
            print("-" * 88, flush=True)
            for c_key in ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'B3_NHC_Only', 'C2_Speed_Compass_NHC',
                          'C4_Full_Smartphone_Map', 'D1_Ablate_Minus_Speed', 'D2_Ablate_Minus_Compass',
                          'D3_Ablate_Minus_NHC', 'D5_Ablate_Minus_Map', 'REF_CAN_Wheel_Full']:
                m = cond_metrics[c_key]
                print(f"{c_key:<28} | {m['mean_pos_err_m']:>10.2f}m | {m['mean_along_err_m']:>9.2f}m | {m['mean_cross_err_m']:>9.2f}m | {m['mean_head_err_deg']:>9.1f}° | {m['mean_drift_pct']:>7.2f}%", flush=True)

        all_canonical_results['trips'][tname] = trip_summary

    out_file = RES_DIR / "c8_7e_canonical_benchmark.json"
    with open(out_file, 'w') as f:
        json.dump(all_canonical_results, f, indent=2)
    print(f"\nSaved canonical benchmark results to: {out_file}", flush=True)

    return all_canonical_results


if __name__ == '__main__':
    run_canonical_benchmark()
