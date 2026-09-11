"""
SIH26168 - Stage C8-7C: End-to-End Smartphone-Only Navigation Stack Integration
Module: experiments/run_phone_navigation_c8_7c.py

Integrates the standalone smartphone sensor suite into the full navigation stack:
  Smartphone Sensors -> Alignment -> Speed Estimator -> 15-State ESKF -> NHC -> ZUPT -> Confidence-Gated Compass -> Map Matching

Evaluates across rolling GNSS outages (5s, 10s, 20s, 30s, 60s) on:
- Vta02 (Suburban Benchmark)
- Vta04 (Highway Benchmark)

Evaluated Conditions:
- C0_Baseline: Pure Strapdown IMU + NHC + ZUPT (No speed update, No compass, No map)
- C1_PhoneSpeed_Only: 15-state ESKF + Smartphone ML Speed + NHC + ZUPT
- C2_Compass_Only: 15-state ESKF + Confidence-Gated Calibrated Compass + NHC + ZUPT
- C3_PhoneSpeed_Compass: 15-state ESKF + Smartphone ML Speed + Calibrated Compass + NHC + ZUPT
- C4_Full_Smartphone_Map: Complete Standalone Stack (Phone Speed + Compass + NHC + ZUPT + MHT Map)
- C_Ref_WheelCAN: Reference ceiling using physical CAN wheel speed odometry

Strict Deployment Boundary:
- Zero vehicle CAN, wheel speed, or steering used in Conditions C0-C4.
- All deployable model inputs restricted to smartphone IMU, Magnetometer, Synthetic Orientation, and Pre-outage GNSS.
"""

import sys
import time
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
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D, rotvec_to_quat, quat_mult
from src.navigation.nhc import NonHolonomicConstraint
from src.navigation.zupt import ZeroVelocityUpdate, CausalStationaryDetector
from src.navigation.kinematic_constraints import SoftLongitudinalAccelerationConstraint
from src.navigation.wheel_odometry import ChassisWheelSpeedFusion
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex
from src.navigation.map_constraints import MHTMapConstraintManager
from experiments.run_wheel_speed_fusion_c8_5 import compute_bcac_series
from experiments.evaluate_phone_speed_c8_7 import prepare_trip_phone_data

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


class PhoneCompassFusion:
    """
    Confidence-Gated Smartphone Compass Fusion Engine for 15-State ESKF.
    Implements the 4-factor gating engine verified in Stage C8-6.
    """
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

        # Yaw perturbation Jacobian:
        # In ENU clockwise azimuth, turning left (+wz) decreases azimuth psi.
        # Thus H_theta = [0, 0, -1] in vehicle horizontal plane.
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


def prepare_navigation_data_c8_7(trip_name: str, rf_model: RandomForestRegressor, dt: float = 0.1) -> Dict[str, Any]:
    """Loads telemetry, OSM map, extracts causal ML speed, and precomputes navigation series."""
    print(f"[{trip_name}] Ingesting trip for navigation integration...", flush=True)
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    # Pre-extract causal phone features and static calibration
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

    # Temporal rate and magnetic dip
    b_norm = np.linalg.norm(raw_mag, axis=1)
    db_dt = np.zeros(n)
    db_dt[1:] = np.linalg.norm(np.diff(raw_mag, axis=0), axis=1) / dt
    dip_deg = np.degrees(np.arctan2(-raw_mag[:, 1], np.sqrt(raw_mag[:, 0]**2 + raw_mag[:, 2]**2)))
    dpsi_mag = p_data['dpsi_mag']

    # Ground truth ENU coordinates
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    # Process noise series (BCAC)
    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    # OSM road network
    osm_file = REPO_ROOT / "data" / "raw" / "maps" / f"{trip_name}_network.osm"
    if not osm_file.exists():
        raise FileNotFoundError(f"OSM vector cache not found: {osm_file}")
    parsed_map = parse_osm_network(osm_file, lat0, lon0)
    road_index = RoadNetworkIndex(parsed_map['segments'])
    print(f"[{trip_name}] Road network ready with {road_index.num_segments} directed segments.", flush=True)

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


def simulate_outage_navigation_c8_7(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    condition: str,
    dt: float = 0.1
) -> Dict[str, Any]:
    """
    Simulates a GNSS blackout window under the specified C8-7 navigation condition.
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

    nhc = NonHolonomicConstraint(sigma_lat=0.50, sigma_vert=0.50)
    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )

    # Speed fusion engines
    phone_speed_fusion = ChassisWheelSpeedFusion(sigma_wheel=2.00, sigma_lat=0.50, sigma_vert=0.50)
    can_wheel_fusion = ChassisWheelSpeedFusion(sigma_wheel=0.20, sigma_lat=0.50, sigma_vert=0.50)
    compass_fusion = PhoneCompassFusion(sigma_psi_deg=5.0)

    # MHT Map Manager (only active for C4 and C_Ref_WheelCAN if requested)
    map_mgr = None
    if condition == 'C4_Full_Smartphone_Map':
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

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]

        # 1. ESKF strapdown inertial propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Velocity & NHC updates depending on condition
        if condition == 'C0_Baseline':
            nhc.update_eskf(eskf)

        elif condition == 'C1_PhoneSpeed_Only':
            cur_speed_est = float(ml_speed[step_k])
            phone_speed_fusion.update_eskf_3d_velocity(eskf, cur_speed_est, apply_nis_gate=True)

        elif condition == 'C2_Compass_Only':
            nhc.update_eskf(eskf)
            # Compass heading update with confidence gate
            if compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], np.degrees(gz), dpsi_mag[step_k], dip_deg[step_k]):
                compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        elif condition in ['C3_PhoneSpeed_Compass', 'C4_Full_Smartphone_Map']:
            cur_speed_est = float(ml_speed[step_k])
            phone_speed_fusion.update_eskf_3d_velocity(eskf, cur_speed_est, apply_nis_gate=True)
            # Compass heading update with confidence gate
            if compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], np.degrees(gz), dpsi_mag[step_k], dip_deg[step_k]):
                compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        elif condition == 'C_Ref_WheelCAN':
            cur_wheel_spd = float(wheel_speed[step_k])
            can_wheel_fusion.update_eskf_3d_velocity(eskf, cur_wheel_spd, apply_nis_gate=True)
            if compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], np.degrees(gz), dpsi_mag[step_k], dip_deg[step_k]):
                compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        # 3. ZUPT update
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 4. Map matching constraint (if active, 1.0s cadence)
        if map_mgr is not None and (t_now - last_map_time >= 0.999):
            last_map_time = t_now
            spd_for_map = float(ml_speed[step_k]) if condition != 'C_Ref_WheelCAN' else float(wheel_speed[step_k])
            map_mgr.evaluate_and_update(
                eskf=eskf,
                t_now=t_now,
                mode='joint',
                speed_ms=spd_for_map,
                gyro_z_rads=gz,
                shadow_mode=False
            )

        # 5. Error telemetry
        st = eskf.get_state()
        true_pos = np.array([gt_e[step_k], gt_n[step_k]])
        est_pos = st['pos_n'][:2]
        delta_p = est_pos - true_pos
        pos_err = float(np.linalg.norm(delta_p))

        # Along-track and cross-track projections
        true_h_rad = np.radians(heading[step_k])
        u_along = np.array([np.sin(true_h_rad), np.cos(true_h_rad)])
        u_cross = np.array([-np.cos(true_h_rad), np.sin(true_h_rad)])

        along_err = float(np.dot(delta_p, u_along))
        cross_err = float(np.dot(delta_p, u_cross))

        # Velocity and heading errors
        true_v = speed[step_k]
        est_v = float(np.linalg.norm(st['vel_n'][:2]))
        v_err = float(abs(est_v - true_v))

        h_err = float(abs((st['yaw_deg'] - heading[step_k] + 180.0) % 360.0 - 180.0))

        pos_errors.append(pos_err)
        along_errors.append(along_err)
        cross_errors.append(cross_err)
        vel_errors.append(v_err)
        heading_errors.append(h_err)

    # Final window metrics
    dist_traveled = float(np.sum(speed[kw : kw + w_dur]) * dt)
    final_pos_err = pos_errors[-1]
    drift_pct = float((final_pos_err / (dist_traveled + 1e-4)) * 100.0)

    return {
        'condition': condition,
        'window_idx': kw,
        'dist_traveled_m': dist_traveled,
        'final_pos_err_m': final_pos_err,
        'max_pos_err_m': float(np.max(pos_errors)),
        'mean_pos_err_m': float(np.mean(pos_errors)),
        'final_along_err_m': along_errors[-1],
        'final_cross_err_m': cross_errors[-1],
        'mean_vel_err_ms': float(np.mean(vel_errors)),
        'final_vel_err_ms': vel_errors[-1],
        'mean_heading_err_deg': float(np.mean(heading_errors)),
        'final_heading_err_deg': heading_errors[-1],
        'drift_pct': drift_pct,
        'primary_regime': regimes[kw + w_dur // 2][0] if regimes[kw + w_dur // 2] else 'general'
    }


def run_c8_7c_full_navigation_benchmark():
    """Master benchmark orchestrator for C8-7C."""
    print("=================================================================", flush=True)
    print("STAGE C8-7C: END-TO-END SMARTPHONE-ONLY NAVIGATION BENCHMARK", flush=True)
    print("=================================================================", flush=True)

    # 1. Train ML model on Vta02
    print("[Pipeline] Training Random Forest Speed Estimator on Vta02...", flush=True)
    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])
    print("[Pipeline] ML Speed Estimator trained.", flush=True)

    # 2. Prepare navigation datasets
    d_vta02 = prepare_navigation_data_c8_7('Vta02', rf)
    d_vta04 = prepare_navigation_data_c8_7('Vta04', rf)

    horizons = [5, 10, 20, 30, 60]
    conditions = [
        'C0_Baseline',
        'C1_PhoneSpeed_Only',
        'C2_Compass_Only',
        'C3_PhoneSpeed_Compass',
        'C4_Full_Smartphone_Map',
        'C_Ref_WheelCAN'
    ]

    all_nav_results = {
        'metadata': {
            'stage': 'C8-7C',
            'date': 'September 6, 2026',
            'horizons_s': horizons,
            'conditions': conditions
        },
        'trips': {}
    }

    for trip_data, tname in [(d_vta02, 'Vta02'), (d_vta04, 'Vta04')]:
        print(f"\n=================================================================", flush=True)
        print(f"EVALUATING NAVIGATION BENCHMARK: {tname}", flush=True)
        print(f"=================================================================", flush=True)

        trip_summary = {}

        for dur_s in horizons:
            w_dur = int(dur_s / 0.1)
            step_stride = w_dur
            n_windows = (trip_data['n'] - w_dur) // step_stride
            if n_windows == 0:
                continue

            cond_metrics = {}

            for cond in conditions:
                win_res_list = []
                for w_idx in range(n_windows):
                    kw = w_idx * step_stride
                    res = simulate_outage_navigation_c8_7(trip_data, kw, w_dur, cond)
                    win_res_list.append(res)

                final_errs = [r['final_pos_err_m'] for r in win_res_list]
                along_errs = [abs(r['final_along_err_m']) for r in win_res_list]
                cross_errs = [abs(r['final_cross_err_m']) for r in win_res_list]
                vel_errs = [r['final_vel_err_ms'] for r in win_res_list]
                head_errs = [r['final_heading_err_deg'] for r in win_res_list]
                drift_pcts = [r['drift_pct'] for r in win_res_list]

                # Regime stratification
                regime_drift = {}
                for r_name in ['cruising', 'acceleration', 'braking', 'cornering', 'rough_road', 'stationary', 'general']:
                    match = [r['drift_pct'] for r in win_res_list if r['primary_regime'] == r_name]
                    if match:
                        regime_drift[r_name] = float(np.mean(match))

                cond_metrics[cond] = {
                    'windows_count': n_windows,
                    'mean_pos_err_m': float(np.mean(final_errs)),
                    'p50_pos_err_m': float(np.median(final_errs)),
                    'p90_pos_err_m': float(np.percentile(final_errs, 90)),
                    'p95_pos_err_m': float(np.percentile(final_errs, 95)),
                    'mean_along_err_m': float(np.mean(along_errs)),
                    'mean_cross_err_m': float(np.mean(cross_errs)),
                    'mean_vel_err_ms': float(np.mean(vel_errs)),
                    'mean_head_err_deg': float(np.mean(head_errs)),
                    'mean_drift_pct': float(np.mean(drift_pcts)),
                    'p50_drift_pct': float(np.median(drift_pcts)),
                    'p90_drift_pct': float(np.percentile(drift_pcts, 90)),
                    'p95_drift_pct': float(np.percentile(drift_pcts, 95)),
                    'pass_sih_10pct': bool(np.mean(drift_pcts) < 10.0),
                    'regime_drift_pct': regime_drift
                }

            trip_summary[f"{dur_s}s"] = cond_metrics

            print(f"\n--- Horizon T = {dur_s:2d}s ({n_windows} windows) on {tname} ---", flush=True)
            print(f"{'Condition':24s} | {'Pos Err (m)':11s} | {'Along (m)':9s} | {'Cross (m)':9s} | {'Head (deg)':10s} | {'Drift %':9s} | {'SIH <10%':8s}", flush=True)
            print("-" * 92, flush=True)
            for cond in conditions:
                m = cond_metrics[cond]
                pass_str = "PASS" if m['pass_sih_10pct'] else "FAIL"
                print(f"{cond:24s} | {m['mean_pos_err_m']:7.2f}m (P90:{m['p90_pos_err_m']:5.1f}) | {m['mean_along_err_m']:7.2f}m | {m['mean_cross_err_m']:7.2f}m | {m['mean_head_err_deg']:8.2f}° | {m['mean_drift_pct']:7.2f}% | {pass_str:8s}", flush=True)

        all_nav_results['trips'][tname] = trip_summary

    # Save to JSON
    json_path = RES_DIR / "c8_7_navigation_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_nav_results, f, indent=2)
    print(f"\n[Artifact] Navigation benchmark exported to {json_path}", flush=True)

    return all_nav_results


if __name__ == "__main__":
    run_c8_7c_full_navigation_benchmark()
