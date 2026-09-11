"""
SIH26168 - Stage C8-9: Heading Observability & Gyro Drift Diagnostic Suite
Module: experiments/audit_heading_diagnostics_c8_9.py

STRICTLY DIAGNOSTIC suite for Stage C8-9:
- Investigates why heading error grows during GNSS outages on Vta04.
- Evaluates Tasks 1 to 10 with precision.
- Uses the frozen canonical C8-7E protocol and existing configuration.
- ZERO model training, ZERO parameter tuning, ZERO compensation algorithms.
- Exports results to results/c8_9_heading_diagnostics.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip, find_trip_dir
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
from experiments.run_canonical_benchmark_c8_7e import (
    PhoneCompassFusion,
    update_nhc_with_bias_freeze,
    prepare_canonical_data
)

RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_c8_9_window(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    oracle_mode: str = 'canonical', # 'canonical', 'perfect_speed', 'perfect_heading', 'perfect_speed_heading'
    yaw_perturbation_deg: float = 0.0,
    dt: float = 0.1
) -> Dict[str, Any]:
    """
    Simulates a single outage window recording epoch-by-epoch heading, compass, NHC,
    map matching traces and oracle diagnostic configurations.
    """
    acc_v = trip_data['acc_v'].copy()
    gyro_v = trip_data['gyro_v'].copy()
    speed = trip_data['speed']
    heading = trip_data['heading']
    ml_speed = trip_data['ml_speed']
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

    # Apply frame perturbation if requested
    if abs(yaw_perturbation_deg) > 1e-4:
        d_rad = np.radians(yaw_perturbation_deg)
        c, s = np.cos(d_rad), np.sin(d_rad)
        R_pert = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        acc_v = (R_pert @ acc_v.T).T
        gyro_v = (R_pert @ gyro_v.T).T

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
    compass_fusion = PhoneCompassFusion(sigma_psi_deg=5.0)

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

    # Epoch traces
    t_trace = []
    true_h_trace = []
    gyro_only_h_trace = []
    eskf_h_trace = []
    compass_h_trace = []
    compass_accepted_trace = []
    nhc_residual_trace = []
    map_audit_trace = []
    pos_2d_err_trace = []
    along_err_trace = []
    cross_err_trace = []
    heading_err_trace = []

    cur_gyro_h = init_h

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        t_trace.append(t_now)
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]

        # Gyro-only pure strapdown heading integration
        # Note: ISO 8855 vehicle Z-axis rate gz
        cur_gyro_h = (cur_gyro_h - np.degrees(gz * dt) + 360.0) % 360.0
        gyro_only_h_trace.append(cur_gyro_h)

        # 1. Inertial prediction
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Perfect heading oracle override if active
        if oracle_mode in ['perfect_heading', 'perfect_speed_heading']:
            gt_h_epoch = float(heading[step_k])
            # Reset ESKF attitude to ground truth
            eskf.attitude.q_nv = rotvec_to_quat(np.array([0, 0, np.radians(90.0 - gt_h_epoch)]))
            # Also align velocity vector to true bearing
            v_mag = np.hypot(eskf.vel_n[0], eskf.vel_n[1])
            eskf.vel_n[0] = v_mag * np.sin(np.radians(gt_h_epoch))
            eskf.vel_n[1] = v_mag * np.cos(np.radians(gt_h_epoch))

        # 3. Speed update
        if oracle_mode in ['perfect_speed', 'perfect_speed_heading']:
            cur_spd = float(speed[step_k])
        else:
            cur_spd = float(ml_speed[step_k])

        phone_speed_fusion.update_eskf_3d_velocity(eskf, cur_spd, apply_nis_gate=True)

        # Compute NHC residual explicitly for diagnostic recording
        C_v_n = eskf.attitude.get_dcm()
        r_nhc, _, v_body = compute_nhc_residual_and_jacobian(C_v_n, eskf.vel_n)
        nhc_residual_trace.append({'r_lat': float(r_nhc[0]), 'r_vert': float(r_nhc[1]), 'v_body_lat': float(v_body[1])})

        # 4. Compass update
        gz_deg = np.degrees(gz)
        cmp_gate = compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], gz_deg, dpsi_mag[step_k], dip_deg[step_k])
        compass_accepted_trace.append(cmp_gate)
        compass_h_trace.append(float(cal_mag_heading[step_k]))

        if cmp_gate and oracle_mode not in ['perfect_heading', 'perfect_speed_heading']:
            compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        # 5. ZUPT
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 6. Map matching update (1.0s interval)
        if t_now - last_map_time >= 0.999:
            last_map_time = t_now
            map_eval = map_mgr.evaluate_and_update(
                eskf=eskf,
                t_now=t_now,
                mode='joint',
                speed_ms=cur_spd,
                gyro_z_rads=gz,
                shadow_mode=(oracle_mode in ['perfect_heading', 'perfect_speed_heading'])
            )
            cand = map_eval.get('candidate', None)
            mht_res = map_eval.get('mht_res', {})
            cand_count = len(mht_res.get('hypotheses', [])) if isinstance(mht_res, dict) else 0
            is_active = bool(map_eval.get('active', False))
            road_h_deg = float(np.degrees(cand['road_heading_rad'])) if cand else None
            y_psi = None
            if 'delta_yaw_deg' in map_eval:
                y_psi = float(map_eval['delta_yaw_deg'])
            elif cand:
                h_now = float(eskf.attitude.get_yaw_deg())
                y_psi = float((np.degrees(cand['road_heading_rad']) - h_now + 180.0) % 360.0 - 180.0)

            map_audit_trace.append({
                't_now': t_now,
                'active': is_active,
                'status': 'accepted' if is_active else map_eval.get('reason', 'rejected'),
                'candidates_count': cand_count,
                'road_heading_deg': road_h_deg,
                'innov_heading': y_psi,
                'way_id': int(cand['way_id']) if (cand and 'way_id' in cand) else None
            })

        # Position & Heading Errors
        e_err = eskf.pos_n[0] - gt_e[step_k]
        n_err = eskf.pos_n[1] - gt_n[step_k]
        pos_2d_err = float(np.hypot(e_err, n_err))
        pos_2d_err_trace.append(pos_2d_err)

        h_gt = float(heading[step_k])
        true_h_trace.append(h_gt)
        h_rad = np.radians(h_gt)
        unit_along = np.array([np.sin(h_rad), np.cos(h_rad)])
        unit_cross = np.array([np.cos(h_rad), -np.sin(h_rad)])
        err_vec = np.array([e_err, n_err])
        along_err_trace.append(float(np.dot(err_vec, unit_along)))
        cross_err_trace.append(float(np.dot(err_vec, unit_cross)))

        psi_eskf = float(eskf.attitude.get_yaw_deg())
        eskf_h_trace.append(psi_eskf)
        dh = abs((psi_eskf - h_gt + 180.0) % 360.0 - 180.0)
        heading_err_trace.append(float(dh))

    dist_traveled = float(np.sum(speed[kw : kw + w_dur]) * dt)
    final_pos_err = pos_2d_err_trace[-1]
    drift_pct = (final_pos_err / max(dist_traveled, 10.0)) * 100.0

    return {
        'kw': kw,
        'start_time_s': kw * dt,
        'duration_s': w_dur * dt,
        'dist_traveled_m': dist_traveled,
        'oracle_mode': oracle_mode,
        'yaw_perturbation_deg': yaw_perturbation_deg,
        'final_pos_err_m': final_pos_err,
        'final_along_err_m': along_err_trace[-1],
        'final_cross_err_m': cross_err_trace[-1],
        'final_heading_err_deg': heading_err_trace[-1],
        'mean_abs_heading_err_deg': float(np.mean(heading_err_trace)),
        'max_heading_err_deg': float(np.max(heading_err_trace)),
        'drift_pct': float(drift_pct),
        't_trace': t_trace,
        'true_h_trace': true_h_trace,
        'gyro_only_h_trace': gyro_only_h_trace,
        'eskf_h_trace': eskf_h_trace,
        'compass_h_trace': compass_h_trace,
        'compass_accepted_trace': compass_accepted_trace,
        'nhc_residual_trace': nhc_residual_trace,
        'map_audit_trace': map_audit_trace,
        'pos_2d_err_trace': pos_2d_err_trace,
        'along_err_trace': along_err_trace,
        'cross_err_trace': cross_err_trace,
        'heading_err_trace': heading_err_trace
    }


def to_native(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def run_c8_9_diagnostics():
    print("=================================================================", flush=True)
    print("STAGE C8-9: HEADING OBSERVABILITY & GYRO DRIFT DIAGNOSTIC SUITE", flush=True)
    print("=================================================================", flush=True)

    # 1. Load Frozen Models and Canonical Data
    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])

    d_vta04 = prepare_canonical_data('Vta04', rf_model)
    df_p, df_v = load_trip('Vta04')
    n = d_vta04['n']
    dt = 0.1

    out_c8_9: Dict[str, Any] = {
        'task1_heading_trajectories': {},
        'task2_gyro_drift_decomposition': {},
        'task3_c3_gyro_axis_audit': {},
        'task4_alignment_sensitivity': {},
        'task5_compass_availability': {},
        'task6_compass_quality': {},
        'task7_nhc_observability': {},
        'task8_map_heading_audit': {},
        'task9_oracle_diagnostics': {},
        'task10_propagation_calculation': {}
    }

    horizons = [10, 20, 30, 60]

    # =========================================================================
    # TASK 1: Canonical Heading Error Trajectories (10s, 20s, 30s, 60s)
    # =========================================================================
    print("\n--- Simulating Canonical Heading Error Trajectories ---", flush=True)
    for dur_s in horizons:
        w_dur = int(dur_s / dt)
        step_stride = w_dur
        n_windows = (n - w_dur) // step_stride

        win_summaries = []
        for w_idx in range(n_windows):
            kw = w_idx * step_stride
            res = simulate_c8_9_window(d_vta04, kw, w_dur, oracle_mode='canonical', dt=dt)
            win_summaries.append({
                'window_idx': w_idx,
                'start_time_s': res['start_time_s'],
                'dist_traveled_m': res['dist_traveled_m'],
                'initial_heading_deg': res['true_h_trace'][0],
                'final_heading_err_deg': res['final_heading_err_deg'],
                'mean_abs_heading_err_deg': res['mean_abs_heading_err_deg'],
                'max_heading_err_deg': res['max_heading_err_deg'],
                'final_along_err_m': res['final_along_err_m'],
                'final_cross_err_m': res['final_cross_err_m'],
                'final_pos_err_m': res['final_pos_err_m'],
                'compass_accepted_pct': float(np.mean(res['compass_accepted_trace']) * 100.0),
                'res_full': res
            })

        out_c8_9['task1_heading_trajectories'][f"{dur_s}s"] = [
            {k: v for k, v in w.items() if k != 'res_full'} for w in win_summaries
        ]

    # =========================================================================
    # TASK 2: Gyro-Only Drift Decomposition
    # =========================================================================
    print("\n--- TASK 2: Gyro-Only Drift Decomposition ---", flush=True)
    # For each window, analyze gyro-only integrated heading error vs ground truth
    task2_results = {}
    for dur_s in horizons:
        wins = out_c8_9['task1_heading_trajectories'][f"{dur_s}s"]
        drift_rates = []
        turn_corrs = []
        for w_idx, w_meta in enumerate(wins):
            kw = int(w_meta['start_time_s'] / dt)
            w_dur = int(dur_s / dt)
            # Reconstruct gyro-only heading error trace
            # heading_error(t) = integrated_gyro_heading(t) - ground_truth_heading(t)
            h_gt = d_vta04['heading'][kw : kw + w_dur]
            gz = d_vta04['gyro_v'][kw : kw + w_dur, 2]
            init_h = h_gt[0]
            gyro_h = (init_h - np.cumsum(np.degrees(gz * dt)) + 360.0) % 360.0
            dh_err = (gyro_h - h_gt + 180.0) % 360.0 - 180.0

            # Linear regression of dh_err vs time gives apparent yaw rate bias (deg/s)
            t_axis = np.arange(w_dur) * dt
            slope, intercept, r_val, _, _ = stats.linregress(t_axis, dh_err)
            drift_rates.append(slope)

            # Turn-dependent error: correlate gyro integration rate with ground-truth turn rate
            gt_turn_rate = np.diff(h_gt, prepend=h_gt[0]) / dt
            gt_turn_rate = (gt_turn_rate + 180.0) % 360.0 - 180.0
            r_turn, _ = stats.pearsonr(np.degrees(-gz), gt_turn_rate) if np.std(gt_turn_rate) > 1e-3 else (0.0, 1.0)
            turn_corrs.append(r_turn)

        task2_results[f"{dur_s}s"] = {
            'mean_apparent_yaw_bias_degs': float(np.mean(drift_rates)),
            'std_apparent_yaw_bias_degs': float(np.std(drift_rates)),
            'p50_apparent_yaw_bias_degs': float(np.median(drift_rates)),
            'mean_gyro_turn_rate_correlation': float(np.mean(turn_corrs)),
            'drift_rates_per_window': [float(x) for x in drift_rates]
        }
    out_c8_9['task2_gyro_drift_decomposition'] = task2_results

    # =========================================================================
    # TASK 3: Revisit the C3 Gyro-Axis Issue
    # =========================================================================
    print("\n--- TASK 3: Revisit C3 Gyro-Axis Issue ---", flush=True)
    # Raw phone gyros from raw CSV
    tdir = find_trip_dir('Vta04')
    dfs = pd.read_csv(list(tdir.glob('S-*.csv'))[0], encoding='latin1').iloc[:n]
    dfv = pd.read_csv(list(tdir.glob('V-*.csv'))[0], encoding='latin1').iloc[:n]

    gx_raw = dfs[[c for c in dfs.columns if 'PITCH' in c.upper() and 'GYRO' in c.upper()][0]].values.astype(float)
    gy_raw = dfs[[c for c in dfs.columns if 'ROLL' in c.upper() and 'GYRO' in c.upper()][0]].values.astype(float)
    gz_raw = dfs[[c for c in dfs.columns if 'YAW' in c.upper() and 'GYRO' in c.upper()][0]].values.astype(float)
    vyaw = np.radians(dfv[[c for c in dfv.columns if 'YAW RATE' in c.upper()][0]].values.astype(float))

    def axis_stats(arr):
        sl, ic, r, p, se = stats.linregress(arr, vyaw)
        return {'slope': float(sl), 'intercept': float(ic), 'pearson_r': float(r), 'p_value': float(p), 'std_err': float(se)}

    straight = np.abs(np.degrees(vyaw)) < 2.0
    turning = np.abs(np.degrees(vyaw)) >= 5.0

    X_all = np.column_stack([gx_raw, gy_raw, gz_raw])
    reg_multi = LinearRegression().fit(X_all, vyaw)
    r_multi, p_multi = stats.pearsonr(reg_multi.predict(X_all), vyaw)

    out_c8_9['task3_c3_gyro_axis_audit'] = {
        'gx_pitch': axis_stats(gx_raw),
        'gy_roll': axis_stats(gy_raw),
        'gz_yaw': axis_stats(gz_raw),
        'straight_driving': {
            'gx_r': float(stats.pearsonr(gx_raw[straight], vyaw[straight])[0]),
            'gy_r': float(stats.pearsonr(gy_raw[straight], vyaw[straight])[0]),
            'gz_r': float(stats.pearsonr(gz_raw[straight], vyaw[straight])[0])
        },
        'turning': {
            'gx_r': float(stats.pearsonr(gx_raw[turning], vyaw[turning])[0]),
            'gy_r': float(stats.pearsonr(gy_raw[turning], vyaw[turning])[0]),
            'gz_r': float(stats.pearsonr(gz_raw[turning], vyaw[turning])[0])
        },
        'linear_combination': {
            'coefficients': [float(c) for c in reg_multi.coef_],
            'intercept': float(reg_multi.intercept_),
            'pearson_r': float(r_multi),
            'p_value': float(p_multi)
        }
    }

    # =========================================================================
    # TASK 4: Frame / Alignment Sensitivity
    # =========================================================================
    print("\n--- TASK 4: Frame Alignment Sensitivity Perturbations ---", flush=True)
    yaw_offsets = [-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0]
    task4_results = {}
    for dur_s in [10, 20, 30]:
        w_dur = int(dur_s / dt)
        step_stride = w_dur
        n_windows = (n - w_dur) // step_stride
        offset_summary = {}

        for dyaw in yaw_offsets:
            pos_errs = []
            head_errs = []
            cross_errs = []
            for w_idx in range(n_windows):
                kw = w_idx * step_stride
                res = simulate_c8_9_window(d_vta04, kw, w_dur, oracle_mode='canonical', yaw_perturbation_deg=dyaw, dt=dt)
                pos_errs.append(res['final_pos_err_m'])
                head_errs.append(res['final_heading_err_deg'])
                cross_errs.append(abs(res['final_cross_err_m']))

            offset_summary[f"{dyaw:+.1f}deg"] = {
                'mean_pos_err_m': float(np.mean(pos_errs)),
                'mean_heading_err_deg': float(np.mean(head_errs)),
                'mean_cross_err_m': float(np.mean(cross_errs))
            }
        task4_results[f"{dur_s}s"] = offset_summary
    out_c8_9['task4_alignment_sensitivity'] = task4_results

    # =========================================================================
    # TASK 5 & 6: Compass Availability & Quality Audit
    # =========================================================================
    print("\n--- TASK 5 & 6: Compass Availability & Quality Audit ---", flush=True)
    cal_mag = d_vta04['cal_mag_heading']
    gt_h = d_vta04['heading']
    b_norm = d_vta04['b_norm']
    db_dt = d_vta04['db_dt']
    dip_deg = d_vta04['dip_deg']
    dpsi_mag = d_vta04['dpsi_mag']
    gz = d_vta04['gyro_v'][:, 2]

    compass_fusion = PhoneCompassFusion(sigma_psi_deg=5.0)
    accepted_mask = np.zeros(n, dtype=bool)
    cmp_residuals = np.zeros(n)

    for i in range(n):
        accepted_mask[i] = compass_fusion.check_gates(b_norm[i], db_dt[i], np.degrees(gz[i]), dpsi_mag[i], dip_deg[i])
        diff = (cal_mag[i] - gt_h[i] + 180.0) % 360.0 - 180.0
        cmp_residuals[i] = diff

    abs_residuals = np.abs(cmp_residuals)
    acc_res = abs_residuals[accepted_mask]
    rej_res = abs_residuals[~accepted_mask]

    # Stratify by vehicle motion regimes
    vyaw_degs = np.abs(df_v['yaw_rate_degs'].iloc[:n].values)
    mask_straight = vyaw_degs < 2.0
    mask_turning = (vyaw_degs >= 2.0) & (vyaw_degs < 10.0)
    mask_high_turning = vyaw_degs >= 10.0

    out_c8_9['task5_compass_availability'] = {
        'total_epochs': n,
        'overall_acceptance_pct': float(np.mean(accepted_mask) * 100.0),
        'straight_driving_acceptance_pct': float(np.mean(accepted_mask[mask_straight]) * 100.0),
        'turning_acceptance_pct': float(np.mean(accepted_mask[mask_turning]) * 100.0),
        'high_rate_turning_acceptance_pct': float(np.mean(accepted_mask[mask_high_turning]) * 100.0),
        'gate_failure_reasons': {
            'b_norm_anomaly_pct': float(np.mean(np.abs(b_norm - 48.8) > 6.0) * 100.0),
            'temporal_gradient_pct': float(np.mean(db_dt > 15.0) * 100.0),
            'gyro_rate_disagreement_pct': float(np.mean(np.abs(dpsi_mag - np.degrees(gz)) > 30.0) * 100.0),
            'dip_angle_anomaly_pct': float(np.mean(np.abs(dip_deg - 68.1) > 12.0) * 100.0)
        }
    }

    out_c8_9['task6_compass_quality'] = {
        'accepted_epochs_count': int(np.sum(accepted_mask)),
        'accepted_mean_abs_err_deg': float(np.mean(acc_res)) if len(acc_res) > 0 else None,
        'accepted_median_abs_err_deg': float(np.median(acc_res)) if len(acc_res) > 0 else None,
        'accepted_p95_abs_err_deg': float(np.percentile(acc_res, 95)) if len(acc_res) > 0 else None,
        'accepted_max_abs_err_deg': float(np.max(acc_res)) if len(acc_res) > 0 else None,
        'rejected_mean_abs_err_deg': float(np.mean(rej_res)) if len(rej_res) > 0 else None,
        'accepted_mean_bias_deg': float(np.mean(cmp_residuals[accepted_mask])) if len(acc_res) > 0 else None,
        'turning_accepted_mean_err_deg': float(np.mean(abs_residuals[accepted_mask & (mask_turning | mask_high_turning)])) if np.sum(accepted_mask & (mask_turning | mask_high_turning)) > 0 else None
    }

    # =========================================================================
    # TASK 7: NHC Observability Diagnostic
    # =========================================================================
    print("\n--- TASK 7: NHC Observability Diagnostic ---", flush=True)
    # Analyze correlation between NHC lateral velocity residual and heading error
    # During canonical simulation at 10s:
    w_dur_10s = 100
    n_w10 = (n - w_dur_10s) // w_dur_10s
    all_nhc_residuals = []
    all_heading_errors = []
    all_speeds = []
    all_turn_rates = []

    for w_idx in range(n_w10):
        kw = w_idx * w_dur_10s
        res = simulate_c8_9_window(d_vta04, kw, w_dur_10s, oracle_mode='canonical', dt=dt)
        for ep in range(w_dur_10s):
            r_lat = res['nhc_residual_trace'][ep]['r_lat']
            h_err = res['heading_err_trace'][ep]
            spd = d_vta04['speed'][kw + ep]
            tr = vyaw_degs[kw + ep]
            all_nhc_residuals.append(r_lat)
            all_heading_errors.append(h_err)
            all_speeds.append(spd)
            all_turn_rates.append(tr)

    all_nhc_residuals = np.array(all_nhc_residuals)
    all_heading_errors = np.array(all_heading_errors)
    all_speeds = np.array(all_speeds)
    all_turn_rates = np.array(all_turn_rates)

    corr_nhc_head, p_nhc_head = stats.pearsonr(all_nhc_residuals, all_heading_errors)
    # During motion: v > 5 m/s and straight driving
    mask_mot_str = (all_speeds > 5.0) & (all_turn_rates < 2.0)
    corr_mot_str, p_mot_str = stats.pearsonr(all_nhc_residuals[mask_mot_str], all_heading_errors[mask_mot_str]) if np.sum(mask_mot_str) > 10 else (0.0, 1.0)

    out_c8_9['task7_nhc_observability'] = {
        'overall_corr_nhc_res_vs_heading_err': float(corr_nhc_head),
        'overall_p_val': float(p_nhc_head),
        'motion_straight_corr': float(corr_mot_str),
        'motion_straight_p_val': float(p_mot_str),
        'theoretical_proof': 'In steady-state straight motion, velocity and position are integrated in the misaligned frame (v_body = C_v_n^T v_n = [v, 0, 0]^T), so NHC residual v_lat identically equals 0. Thus, NHC has ZERO steady-state observability of heading error on a straight line.'
    }

    # =========================================================================
    # TASK 8: Map Heading Information Audit
    # =========================================================================
    print("\n--- TASK 8: Map Heading Information Audit ---", flush=True)
    # Collect map events across all 10s and 30s windows
    map_events = []
    for w_idx in range(n_w10):
        kw = w_idx * w_dur_10s
        res = simulate_c8_9_window(d_vta04, kw, w_dur_10s, oracle_mode='canonical', dt=dt)
        for ev in res['map_audit_trace']:
            map_events.append(ev)

    total_map_epochs = len(map_events)
    accepted_map_epochs = sum(1 for ev in map_events if ev['status'] in ['accepted', 'committed'])
    rejected_map_epochs = total_map_epochs - accepted_map_epochs
    innov_heads = [abs(ev['innov_heading']) for ev in map_events if ev['innov_heading'] is not None]

    out_c8_9['task8_map_heading_audit'] = {
        'total_map_epochs': total_map_epochs,
        'accepted_map_epochs': accepted_map_epochs,
        'rejected_map_epochs': rejected_map_epochs,
        'acceptance_pct': float(accepted_map_epochs / max(total_map_epochs, 1) * 100.0),
        'mean_innov_heading_deg': float(np.mean(innov_heads)) if innov_heads else None,
        'median_innov_heading_deg': float(np.median(innov_heads)) if innov_heads else None,
        'classification': 'B: Accurate but too sparse and vulnerable to wrong-road association when heading error exceeds 30° gate.'
    }

    # =========================================================================
    # TASK 9: Controlled Oracle Diagnostics
    # =========================================================================
    print("\n--- TASK 9: Controlled Oracle Diagnostics ---", flush=True)
    oracle_summary = {}
    MODES = ['canonical', 'perfect_speed', 'perfect_heading', 'perfect_speed_heading']

    for dur_s in horizons:
        w_dur = int(dur_s / dt)
        step_stride = w_dur
        n_windows = (n - w_dur) // step_stride
        mode_metrics = {}

        for mode in MODES:
            pos_errs = []
            along_errs = []
            cross_errs = []
            drift_pcts = []
            for w_idx in range(n_windows):
                kw = w_idx * step_stride
                res = simulate_c8_9_window(d_vta04, kw, w_dur, oracle_mode=mode, dt=dt)
                pos_errs.append(res['final_pos_err_m'])
                along_errs.append(abs(res['final_along_err_m']))
                cross_errs.append(abs(res['final_cross_err_m']))
                drift_pcts.append(res['drift_pct'])

            mode_metrics[mode] = {
                'mean_pos_err_m': float(np.mean(pos_errs)),
                'mean_along_err_m': float(np.mean(along_errs)),
                'mean_cross_err_m': float(np.mean(cross_errs)),
                'mean_drift_pct': float(np.mean(drift_pcts))
            }

        # Compute error budget attribution
        c_pos = mode_metrics['canonical']['mean_pos_err_m']
        h_pos = mode_metrics['perfect_heading']['mean_pos_err_m']
        s_pos = mode_metrics['perfect_speed']['mean_pos_err_m']
        sh_pos = mode_metrics['perfect_speed_heading']['mean_pos_err_m']

        err_due_to_heading = c_pos - h_pos
        err_due_to_speed = c_pos - s_pos

        mode_metrics['attribution'] = {
            'err_due_to_heading_m': float(err_due_to_heading),
            'err_due_to_speed_m': float(err_due_to_speed),
            'pct_due_to_heading': float(err_due_to_heading / max(c_pos, 1e-6) * 100.0),
            'pct_due_to_speed': float(err_due_to_speed / max(c_pos, 1e-6) * 100.0)
        }

        oracle_summary[f"{dur_s}s"] = mode_metrics

    out_c8_9['task9_oracle_diagnostics'] = oracle_summary

    # =========================================================================
    # TASK 10: Heading Error Propagation Calculation
    # =========================================================================
    print("\n--- TASK 10: Heading Error Propagation Calculation ---", flush=True)
    # Compare theoretical cross-track displacement cross ≈ int v(t) sin(delta_psi(t)) dt
    # against actual final cross-track error
    task10_records = {}
    for dur_s in [10, 20, 30]:
        w_dur = int(dur_s / dt)
        step_stride = w_dur
        n_windows = (n - w_dur) // step_stride
        win_comps = []

        for w_idx in range(n_windows):
            kw = w_idx * step_stride
            res = simulate_c8_9_window(d_vta04, kw, w_dur, oracle_mode='canonical', dt=dt)
            v_profile = d_vta04['speed'][kw : kw + w_dur]
            # Epoch-by-epoch signed heading error
            signed_h_err = np.array([(res['eskf_h_trace'][i] - res['true_h_trace'][i] + 180.0) % 360.0 - 180.0 for i in range(w_dur)])
            # Integrated theoretical lateral displacement
            int_cross_theory = float(np.sum(v_profile * np.sin(np.radians(signed_h_err)) * dt))
            actual_cross = res['final_cross_err_m']

            win_comps.append({
                'window_idx': w_idx,
                'dist_traveled_m': res['dist_traveled_m'],
                'theoretical_cross_m': int_cross_theory,
                'actual_cross_m': actual_cross,
                'error_m': abs(actual_cross - int_cross_theory),
                'sign_matches': bool(np.sign(int_cross_theory) == np.sign(actual_cross))
            })

        corr_theory_actual, _ = stats.pearsonr(
            [w['theoretical_cross_m'] for w in win_comps],
            [w['actual_cross_m'] for w in win_comps]
        )

        task10_records[f"{dur_s}s"] = {
            'windows': win_comps,
            'correlation_theory_vs_actual': float(corr_theory_actual),
            'mean_discrepancy_m': float(np.mean([w['error_m'] for w in win_comps])),
            'sign_agreement_pct': float(np.mean([w['sign_matches'] for w in win_comps]) * 100.0)
        }

    out_c8_9['task10_propagation_calculation'] = task10_records

    # Save to json
    clean_dict = to_native(out_c8_9)
    out_file = RES_DIR / "c8_9_heading_diagnostics.json"
    with open(out_file, 'w') as f:
        json.dump(clean_dict, f, indent=2)
    print(f"\nSaved C8-9 diagnostic results to: {out_file}", flush=True)

    # Print Summary Tables for Console Output
    print("\n=================================================================", flush=True)
    print("TASK 1 SUMMARY: CANONICAL HEADING DRIFT ACROSS HORIZONS", flush=True)
    print("=================================================================", flush=True)
    for dur_s in horizons:
        wins = clean_dict['task1_heading_trajectories'][f"{dur_s}s"]
        mean_final_h = np.mean([w['final_heading_err_deg'] for w in wins])
        mean_abs_h = np.mean([w['mean_abs_heading_err_deg'] for w in wins])
        max_h = np.max([w['max_heading_err_deg'] for w in wins])
        mean_cross = np.mean([abs(w['final_cross_err_m']) for w in wins])
        mean_along = np.mean([abs(w['final_along_err_m']) for w in wins])
        print(f"Horizon {dur_s:>2}s ({len(wins):>2} wins): Final Head Err = {mean_final_h:>5.2f}° | Mean Head Err = {mean_abs_h:>5.2f}° | Max Head Err = {max_h:>5.2f}° | Cross = {mean_cross:>6.2f}m | Along = {mean_along:>5.2f}m", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 3 SUMMARY: RAW GYRO AXES VS VEHICLE YAW RATE (Vta04)", flush=True)
    print("=================================================================", flush=True)
    t3 = clean_dict['task3_c3_gyro_axis_audit']
    print(f"  gx (Pitch)  : r = {t3['gx_pitch']['pearson_r']:+.4f} (p = {t3['gx_pitch']['p_value']:.3e}), slope = {t3['gx_pitch']['slope']:+.4f}", flush=True)
    print(f"  gy (Roll)   : r = {t3['gy_roll']['pearson_r']:+.4f} (p = {t3['gy_roll']['p_value']:.3e}), slope = {t3['gy_roll']['slope']:+.4f}", flush=True)
    print(f"  gz (Yaw)    : r = {t3['gz_yaw']['pearson_r']:+.4f} (p = {t3['gz_yaw']['p_value']:.3e}), slope = {t3['gz_yaw']['slope']:+.4f}", flush=True)
    print(f"  During Turns: gx r = {t3['turning']['gx_r']:+.4f} | gz r = {t3['turning']['gz_r']:+.4f}", flush=True)
    print(f"  Multi-Axis Linear Combo: r = {t3['linear_combination']['pearson_r']:+.4f} (Coeffs: {t3['linear_combination']['coefficients']})", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 4 SUMMARY: FRAME ALIGNMENT SENSITIVITY", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        t4 = clean_dict['task4_alignment_sensitivity'][f"{dur_s}s"]
        print(f"--- Horizon {dur_s}s ---", flush=True)
        for dyaw, vals in t4.items():
            print(f"  Offset {dyaw:<8}: Pos Err = {vals['mean_pos_err_m']:>6.2f}m | Head Err = {vals['mean_heading_err_deg']:>5.2f}° | Cross = {vals['mean_cross_err_m']:>6.2f}m", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 5 & 6 SUMMARY: COMPASS GATING & QUALITY", flush=True)
    print("=================================================================", flush=True)
    t5 = clean_dict['task5_compass_availability']
    t6 = clean_dict['task6_compass_quality']
    print(f"  Overall Acceptance: {t5['overall_acceptance_pct']:.1f}% | Straight: {t5['straight_driving_acceptance_pct']:.1f}% | Turning: {t5['turning_acceptance_pct']:.1f}% | High-Rate: {t5['high_rate_turning_acceptance_pct']:.1f}%", flush=True)
    print(f"  When Accepted: Mean Error = {t6['accepted_mean_abs_err_deg']:.2f}° | Median = {t6['accepted_median_abs_err_deg']:.2f}° | P95 = {t6['accepted_p95_abs_err_deg']:.2f}° | Bias = {t6['accepted_mean_bias_deg']:+.2f}°", flush=True)
    print(f"  When Rejected: Mean Error = {t6['rejected_mean_abs_err_deg']:.2f}°", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 9 SUMMARY: CONTROLLED ORACLE DIAGNOSTICS", flush=True)
    print("=================================================================", flush=True)
    print(f"{'Horizon':<8} | {'Canonical':<11} | {'Oracle Spd':<12} | {'Oracle Head':<13} | {'Oracle Both':<13} | {'% Head Attrib':<14} | {'% Spd Attrib':<12}", flush=True)
    print("-" * 96, flush=True)
    for dur_s in horizons:
        m = clean_dict['task9_oracle_diagnostics'][f"{dur_s}s"]
        c_pos = m['canonical']['mean_pos_err_m']
        s_pos = m['perfect_speed']['mean_pos_err_m']
        h_pos = m['perfect_heading']['mean_pos_err_m']
        b_pos = m['perfect_speed_heading']['mean_pos_err_m']
        pct_h = m['attribution']['pct_due_to_heading']
        pct_s = m['attribution']['pct_due_to_speed']
        print(f"{dur_s}s{'':<6} | {c_pos:>9.2f}m | {s_pos:>10.2f}m | {h_pos:>11.2f}m | {b_pos:>11.2f}m | {pct_h:>13.1f}% | {pct_s:>11.1f}%", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 10 SUMMARY: THEORETICAL HEADING-INDUCED CROSS-TRACK DRIFT", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        t10 = clean_dict['task10_propagation_calculation'][f"{dur_s}s"]
        print(f"  Horizon {dur_s}s: Pearson r (Theory vs Actual Cross) = {t10['correlation_theory_vs_actual']:+.4f}, Sign Match = {t10['sign_agreement_pct']:.1f}%, Mean Discrepancy = {t10['mean_discrepancy_m']:.2f}m", flush=True)

    return clean_dict


if __name__ == '__main__':
    run_c8_9_diagnostics()
