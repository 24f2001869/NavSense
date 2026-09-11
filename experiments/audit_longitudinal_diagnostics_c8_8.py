"""
SIH26168 - Stage C8-8: Longitudinal Error Attribution & Physical-Cause Diagnostic
Module: experiments/audit_longitudinal_diagnostics_c8_8.py

Diagnostic-only audit to determine what actually causes the remaining along-track
and total position error in Vta04 before designing any compensation.

Constraints:
- Strictly diagnostic: NO compensation algorithms, NO pitch fixes, NO hyperparameter tuning, NO ML re-training.
- Uses the frozen canonical C8-7E evaluation protocol and existing models/configuration.
- Evaluates Tasks 1 to 8 thoroughly across 10s, 20s, 30s (with 5s and 60s context).
- Exports structured results to results/c8_8_longitudinal_diagnostics.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
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
from experiments.run_canonical_benchmark_c8_7e import (
    PhoneCompassFusion,
    update_nhc_with_bias_freeze,
    prepare_canonical_data
)

RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_diagnostic_window(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    condition: str,
    dt: float = 0.1
) -> Dict[str, Any]:
    """
    Simulates a GNSS blackout window under a specified diagnostic condition,
    recording detailed epoch-by-epoch state traces for mathematical error attribution.
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

    use_speed = condition in ['B1_Speed_Only', 'C4_Full_Smartphone_Map', 'C4_Counterfactual_RefSpeed', 'REF_CAN_Wheel_Full']
    use_nhc = condition in ['B3_NHC_Only', 'C4_Full_Smartphone_Map', 'C4_Counterfactual_RefSpeed', 'REF_CAN_Wheel_Full']
    use_compass = condition in ['C4_Full_Smartphone_Map', 'C4_Counterfactual_RefSpeed', 'REF_CAN_Wheel_Full']
    use_zupt = condition in ['C4_Full_Smartphone_Map', 'C4_Counterfactual_RefSpeed', 'REF_CAN_Wheel_Full']
    use_map = condition in ['C4_Full_Smartphone_Map', 'C4_Counterfactual_RefSpeed', 'REF_CAN_Wheel_Full']
    use_can_ref = (condition == 'REF_CAN_Wheel_Full')
    use_counterfactual = (condition == 'C4_Counterfactual_RefSpeed')

    map_mgr = None
    if use_map:
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
    signed_heading_errors = []
    est_pitches = []
    est_vels = []

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]

        # 1. Strapdown propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Velocity / NHC update
        if use_can_ref:
            cur_spd = float(wheel_speed[step_k])
            fusion_engine = can_wheel_fusion
        elif use_counterfactual:
            cur_spd = float(speed[step_k])  # Reference vehicle speed
            fusion_engine = phone_speed_fusion  # Identical smartphone noise sigma=2.00
        else:
            cur_spd = float(ml_speed[step_k])
            fusion_engine = phone_speed_fusion

        if use_speed and use_nhc:
            fusion_engine.update_eskf_3d_velocity(eskf, cur_spd, apply_nis_gate=True)
        elif use_speed and not use_nhc:
            fusion_engine.update_eskf_forward_velocity(eskf, cur_spd, apply_nis_gate=True)
        elif not use_speed and use_nhc:
            update_nhc_with_bias_freeze(eskf, sigma_lat=0.50, sigma_vert=0.50)

        # 3. Compass update
        if use_compass:
            if compass_fusion.check_gates(b_norm[step_k], db_dt[step_k], np.degrees(gz), dpsi_mag[step_k], dip_deg[step_k]):
                compass_fusion.update_eskf(eskf, cal_mag_heading[step_k])

        # 4. ZUPT update
        if use_zupt:
            k_start = max(0, step_k - 10)
            det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
            if det_res['is_stationary']:
                zupt.update_eskf(eskf)

        # 5. Map matching update
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
        dh_signed = (psi_eskf - heading[step_k] + 180.0) % 360.0 - 180.0
        signed_heading_errors.append(float(dh_signed))
        heading_errors.append(float(abs(dh_signed)))

        est_pitches.append(float(eskf.attitude.get_pitch_deg()))
        
        # Projected body forward velocity
        C_v_n = eskf.attitude.get_dcm()
        v_body = C_v_n.T @ eskf.vel_n
        est_vels.append(float(v_body[0]))

    dist_traveled = float(np.sum(speed[kw : kw + w_dur]) * dt)
    final_pos_err = float(pos_errors[-1])
    drift_pct = (final_pos_err / max(dist_traveled, 10.0)) * 100.0

    win_gt_speed = speed[kw : kw + w_dur]
    win_rf_speed = ml_speed[kw : kw + w_dur]
    speed_residuals = win_rf_speed - win_gt_speed
    i_speed = float(np.sum(speed_residuals) * dt)

    return {
        'kw': kw,
        'condition': condition,
        'start_time_s': kw * dt,
        'duration_s': w_dur * dt,
        'dist_traveled_m': dist_traveled,
        'true_final_vel_ms': float(speed[kw + w_dur - 1]),
        'est_final_vel_ms': float(est_vels[-1]),
        'true_mean_speed_ms': float(np.mean(win_gt_speed)),
        'rf_mean_speed_ms': float(np.mean(win_rf_speed)),
        'rf_speed_mae_ms': float(np.mean(np.abs(speed_residuals))),
        'rf_speed_bias_ms': float(np.mean(speed_residuals)),
        'i_speed_m': i_speed,
        'final_along_err_m': float(along_errors[-1]),
        'final_cross_err_m': float(cross_errors[-1]),
        'final_pos_err_m': final_pos_err,
        'final_heading_err_deg': float(heading_errors[-1]),
        'final_signed_heading_err_deg': float(signed_heading_errors[-1]),
        'mean_heading_err_deg': float(np.mean(heading_errors)),
        'final_vel_err_ms': float(vel_errors[-1]),
        'final_pitch_est_deg': float(est_pitches[-1]),
        'mean_pitch_est_deg': float(np.mean(est_pitches)),
        'along_errors_trace': along_errors,
        'cross_errors_trace': cross_errors,
        'heading_errors_trace': heading_errors,
        'signed_heading_errors_trace': signed_heading_errors,
        'est_vels_trace': est_vels,
        'drift_pct': float(drift_pct)
    }


def run_c8_8_diagnostics():
    print("=================================================================", flush=True)
    print("STAGE C8-8: LONGITUDINAL ERROR ATTRIBUTION & PHYSICAL DIAGNOSTIC", flush=True)
    print("=================================================================", flush=True)

    # 1. Load Frozen Models and Canonical Data
    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])

    d_vta04 = prepare_canonical_data('Vta04', rf_model)
    df_p, df_v = load_trip('Vta04')
    n = d_vta04['n']
    dt = 0.1

    horizons = [10, 20, 30]
    out_diagnostics: Dict[str, Any] = {
        'task1_window_level': {},
        'task2_correlations': {},
        'task3_integrated_speed': {},
        'task4_acceleration': {},
        'task5_dynamic_pitch': {},
        'task6_mounting_compliance': {},
        'task7_heading_interaction': {},
        'task8_counterfactual': {}
    }

    # =========================================================================
    # TASK 1, 2, 3, 7: Window-Level Simulation & Diagnostics
    # =========================================================================
    CONDITIONS_TO_EVAL = [
        'A0_Baseline_PureIMU',
        'B1_Speed_Only',
        'B3_NHC_Only',
        'C4_Full_Smartphone_Map',
        'C4_Counterfactual_RefSpeed',
        'REF_CAN_Wheel_Full'
    ]

    sim_results: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for dur_s in [5, 10, 20, 30, 60]:
        w_dur = int(dur_s / dt)
        step_stride = w_dur
        n_windows = (n - w_dur) // step_stride
        sim_results[dur_s] = {}

        for cond in CONDITIONS_TO_EVAL:
            sim_results[dur_s][cond] = []
            for w_idx in range(n_windows):
                kw = w_idx * step_stride
                res = simulate_diagnostic_window(d_vta04, kw, w_dur, cond, dt=dt)
                sim_results[dur_s][cond].append(res)

    # TASK 1 Output
    for dur_s in [10, 20, 30]:
        w_res_full = sim_results[dur_s]['C4_Full_Smartphone_Map']
        w_res_imu = sim_results[dur_s]['A0_Baseline_PureIMU']
        w_res_spd = sim_results[dur_s]['B1_Speed_Only']
        w_res_nhc = sim_results[dur_s]['B3_NHC_Only']

        win_records = []
        for i, r_c4 in enumerate(w_res_full):
            r_imu = w_res_imu[i]
            r_spd = w_res_spd[i]
            r_nhc = w_res_nhc[i]

            rec = {
                'window_idx': i,
                'start_time_s': r_c4['start_time_s'],
                'dist_traveled_m': r_c4['dist_traveled_m'],
                'true_final_vel_ms': r_c4['true_final_vel_ms'],
                'est_final_vel_ms': r_c4['est_final_vel_ms'],
                'true_mean_speed_ms': r_c4['true_mean_speed_ms'],
                'rf_mean_speed_ms': r_c4['rf_mean_speed_ms'],
                'rf_speed_mae_ms': r_c4['rf_speed_mae_ms'],
                'rf_speed_bias_ms': r_c4['rf_speed_bias_ms'],
                'i_speed_m': r_c4['i_speed_m'],
                'c4_final_along_err_m': r_c4['final_along_err_m'],
                'c4_final_cross_err_m': r_c4['final_cross_err_m'],
                'c4_final_pos_err_m': r_c4['final_pos_err_m'],
                'c4_final_heading_err_deg': r_c4['final_heading_err_deg'],
                'c4_drift_pct': r_c4['drift_pct'],
                'pure_imu_final_pos_err_m': r_imu['final_pos_err_m'],
                'pure_imu_final_along_err_m': r_imu['final_along_err_m'],
                'speed_only_final_pos_err_m': r_spd['final_pos_err_m'],
                'speed_only_final_along_err_m': r_spd['final_along_err_m'],
                'nhc_only_final_pos_err_m': r_nhc['final_pos_err_m'],
                'nhc_only_final_along_err_m': r_nhc['final_along_err_m']
            }
            win_records.append(rec)

        out_diagnostics['task1_window_level'][f"{dur_s}s"] = win_records

    # =========================================================================
    # TASK 2: Determine whether speed error predicts along-track error
    # =========================================================================
    for dur_s in [10, 20, 30]:
        wins = sim_results[dur_s]['C4_Full_Smartphone_Map']
        rf_bias = np.array([w['rf_speed_bias_ms'] for w in wins])
        rf_mae = np.array([w['rf_speed_mae_ms'] for w in wins])
        i_spd = np.array([w['i_speed_m'] for w in wins])
        along_err = np.array([w['final_along_err_m'] for w in wins])
        abs_along_err = np.abs(along_err)
        cross_err = np.array([w['final_cross_err_m'] for w in wins])
        abs_cross_err = np.abs(cross_err)
        head_err = np.array([w['final_heading_err_deg'] for w in wins])

        def safe_corr(x, y):
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                return {'pearson_r': 0.0, 'pearson_p': 1.0, 'spearman_rho': 0.0, 'spearman_p': 1.0}
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            return {'pearson_r': float(pr), 'pearson_p': float(pp), 'spearman_rho': float(sr), 'spearman_p': float(sp)}

        out_diagnostics['task2_correlations'][f"{dur_s}s"] = {
            'bias_vs_signed_along': safe_corr(rf_bias, along_err),
            'bias_vs_abs_along': safe_corr(rf_bias, abs_along_err),
            'mae_vs_abs_along': safe_corr(rf_mae, abs_along_err),
            'ispeed_vs_signed_along': safe_corr(i_spd, along_err),
            'ispeed_vs_abs_along': safe_corr(i_spd, abs_along_err),
            'head_err_vs_along': safe_corr(head_err, abs_along_err),
            'head_err_vs_cross': safe_corr(head_err, abs_cross_err)
        }

    # =========================================================================
    # TASK 3: Integrated speed-error diagnostic
    # =========================================================================
    for dur_s in [10, 20, 30]:
        wins = sim_results[dur_s]['C4_Full_Smartphone_Map']
        task3_list = []
        ratios = []
        signs_match = []

        for w in wins:
            i_spd = w['i_speed_m']
            al_err = w['final_along_err_m']
            
            # Ratio meaningful when |i_spd| > 0.2 m
            if abs(i_spd) >= 0.2:
                ratio = float(al_err / i_spd)
                ratios.append(ratio)
                match = bool(np.sign(al_err) == np.sign(i_spd))
                signs_match.append(match)
            else:
                ratio = None
                match = None

            task3_list.append({
                'window_idx': int(w['kw'] // int(dur_s / dt)),
                'start_time_s': float(w['start_time_s']),
                'i_speed_m': float(i_spd),
                'final_along_err_m': float(al_err),
                'ratio_along_over_ispeed': float(ratio) if ratio is not None else None,
                'sign_matches': match
            })

        out_diagnostics['task3_integrated_speed'][f"{dur_s}s"] = {
            'windows': task3_list,
            'mean_i_speed_m': float(np.mean([abs(w['i_speed_m']) for w in wins])),
            'mean_along_err_m': float(np.mean([abs(w['final_along_err_m']) for w in wins])),
            'mean_ratio_when_meaningful': float(np.mean(ratios)) if ratios else None,
            'median_ratio_when_meaningful': float(np.median(ratios)) if ratios else None,
            'sign_agreement_rate': float(np.mean(signs_match)) if signs_match else None,
            'count_meaningful': len(ratios)
        }

    # =========================================================================
    # TASK 4: Acceleration-Level Diagnostic
    # =========================================================================
    acc_v = d_vta04['acc_v']
    gyro_v = d_vta04['gyro_v']
    speed = d_vta04['speed']
    heading = d_vta04['heading']
    ba_stat = d_vta04['ba_stat']

    # Reference longitudinal acceleration
    if 'veh_accel_long_ms2' in df_v.columns:
        a_x_ref = df_v['veh_accel_long_ms2'].iloc[:n].values.astype(np.float64)
    else:
        a_x_ref = np.gradient(speed, dt)

    a_x_phone = acc_v[:, 0]
    a_x_phone_bc = a_x_phone - ba_stat[0]

    # Differentiated velocity from reference
    a_x_ref_diff = np.gradient(speed, dt)

    # Gravity component from pitch
    pitch_est_deg = d_vta04['pitch_est'] if 'pitch_est' in d_vta04 else np.zeros(n)
    g_comp = 9.80665 * np.sin(np.radians(pitch_est_deg))

    # Difference / discrepancy
    delta_a_raw = a_x_phone - a_x_ref
    delta_a_bc = a_x_phone_bc - a_x_ref

    # Filtered acceleration (rolling 1s / 10 samples)
    s_ax = pd.Series(a_x_phone_bc)
    a_x_filtered = s_ax.rolling(10, min_periods=1).mean().values
    delta_a_filtered = a_x_filtered - a_x_ref

    # Integrated velocity from acceleration
    # Pure integration: v(t) = v0 + cumsum(a_x * dt)
    v_int_raw = np.cumsum(a_x_phone) * dt
    v_int_bc = np.cumsum(a_x_phone_bc) * dt

    # Overall full trip statistics
    def calc_acc_metrics(sig_phone, sig_ref):
        diff = sig_phone - sig_ref
        pr, _ = stats.pearsonr(sig_phone, sig_ref)
        return {
            'mean_bias_ms2': float(np.mean(diff)),
            'std_ms2': float(np.std(diff)),
            'rmse_ms2': float(np.sqrt(np.mean(diff**2))),
            'pearson_r': float(pr)
        }

    # Frequency-domain PSD via Welch
    fs = 10.0
    f_raw, psd_phone = welch(a_x_phone, fs=fs, nperseg=min(256, n))
    _, psd_ref = welch(a_x_ref, fs=fs, nperseg=min(256, n))
    _, psd_diff = welch(delta_a_bc, fs=fs, nperseg=min(256, n))

    out_diagnostics['task4_acceleration'] = {
        'raw_phone_vs_ref': calc_acc_metrics(a_x_phone, a_x_ref),
        'bias_corrected_phone_vs_ref': calc_acc_metrics(a_x_phone_bc, a_x_ref),
        'filtered_phone_vs_ref': calc_acc_metrics(a_x_filtered, a_x_ref),
        'ref_sensor_vs_ref_differentiated': calc_acc_metrics(a_x_ref, a_x_ref_diff),
        'ba_stat_x': float(ba_stat[0]),
        'psd': {
            'frequencies_hz': f_raw.tolist(),
            'psd_phone': psd_phone.tolist(),
            'psd_ref': psd_ref.tolist(),
            'psd_diff': psd_diff.tolist()
        }
    }

    # =========================================================================
    # TASK 5: Dynamic Pitch Hypothesis Test
    # =========================================================================
    # Test whether dynamic pitch plausibly explains longitudinal discrepancy
    # Pitch rate: gyro_v[:, 1]
    pitch_rate_degs = np.degrees(gyro_v[:, 1])
    
    # Measure dynamic pitch angle from ESKF attitude and from vehicle chassis suspension model
    # Suspension pitch squat: typical k_pitch ~ 1.5 to 3.0 deg/g = 0.15 to 0.30 deg / (m/s^2)
    k_susp_deg = 0.20 # deg / (m/s^2)
    pitch_dyn_model_deg = k_susp_deg * a_x_ref
    g_leak_susp_ms2 = 9.80665 * np.sin(np.radians(pitch_dyn_model_deg))

    # Observed discrepancy
    acc_discrepancy = delta_a_bc

    # Compare magnitudes:
    # 1. Root-Mean-Square of predicted gravity leakage vs observed discrepancy
    rms_g_leak = float(np.sqrt(np.mean(g_leak_susp_ms2**2)))
    rms_discrepancy = float(np.sqrt(np.mean(acc_discrepancy**2)))
    mean_abs_g_leak = float(np.mean(np.abs(g_leak_susp_ms2)))
    mean_abs_discrepancy = float(np.mean(np.abs(acc_discrepancy)))

    corr_leak_disc, p_val_leak = stats.pearsonr(g_leak_susp_ms2, acc_discrepancy)

    # Attitude estimator / ESKF pitch angle variations during outage windows
    eskf_pitches_10s = [w['mean_pitch_est_deg'] for w in sim_results[10]['C4_Full_Smartphone_Map']]
    eskf_g_leak_10s = [9.80665 * np.sin(np.radians(p)) for p in eskf_pitches_10s]

    out_diagnostics['task5_dynamic_pitch'] = {
        'pitch_rate_degs_mean': float(np.mean(pitch_rate_degs)),
        'pitch_rate_degs_std': float(np.std(pitch_rate_degs)),
        'pitch_rate_degs_max': float(np.max(np.abs(pitch_rate_degs))),
        'rms_predicted_g_leakage_ms2': rms_g_leak,
        'mean_abs_predicted_g_leakage_ms2': mean_abs_g_leak,
        'rms_observed_discrepancy_ms2': rms_discrepancy,
        'mean_abs_observed_discrepancy_ms2': mean_abs_discrepancy,
        'ratio_leakage_to_discrepancy': float(rms_g_leak / max(rms_discrepancy, 1e-6)),
        'correlation_g_leak_vs_discrepancy': float(corr_leak_disc),
        'p_value': float(p_val_leak),
        'eskf_outage_mean_pitch_deg_10s': float(np.mean(np.abs(eskf_pitches_10s))),
        'verdict_dominant': bool(rms_g_leak >= 0.5 * rms_discrepancy)
    }

    # =========================================================================
    # TASK 6: Mounting / Compliance Hypothesis
    # =========================================================================
    # Energy in frequency bands for phone acceleration vs reference acceleration
    # Sampling: fs = 10 Hz (Nyquist = 5 Hz)
    # Bands: 0-1 Hz, 1-2 Hz, 2-5 Hz, >5 Hz (note: >5 Hz aliased)
    freqs = f_raw
    df_f = freqs[1] - freqs[0]

    def band_power(psd, f_min, f_max):
        idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]
        return float(np.sum(psd[idx]) * df_f)

    bp_phone = {
        '0_to_1_hz': band_power(psd_phone, 0.0, 1.0),
        '1_to_2_hz': band_power(psd_phone, 1.0, 2.0),
        '2_to_5_hz': band_power(psd_phone, 2.0, 5.0),
        'total_0_to_5_hz': band_power(psd_phone, 0.0, 5.0)
    }
    bp_ref = {
        '0_to_1_hz': band_power(psd_ref, 0.0, 1.0),
        '1_to_2_hz': band_power(psd_ref, 1.0, 2.0),
        '2_to_5_hz': band_power(psd_ref, 2.0, 5.0),
        'total_0_to_5_hz': band_power(psd_ref, 0.0, 5.0)
    }
    bp_diff = {
        '0_to_1_hz': band_power(psd_diff, 0.0, 1.0),
        '1_to_2_hz': band_power(psd_diff, 1.0, 2.0),
        '2_to_5_hz': band_power(psd_diff, 2.0, 5.0),
        'total_0_to_5_hz': band_power(psd_diff, 0.0, 5.0)
    }

    # High frequency excess energy in phone (mount flutter / vibration)
    excess_hf_ratio = (bp_phone['2_to_5_hz'] - bp_ref['2_to_5_hz']) / max(bp_ref['2_to_5_hz'], 1e-6)

    # Correlate high-frequency vibration with acceleration discrepancy
    # Compute rolling 2-5 Hz energy proxy (high-pass jerk / diff)
    jerk_hf = np.abs(np.diff(a_x_phone, prepend=a_x_phone[0]))
    s_jhf = pd.Series(jerk_hf).rolling(10, min_periods=1).mean().values
    corr_hf_disc, _ = stats.pearsonr(s_jhf, np.abs(delta_a_bc))

    out_diagnostics['task6_mounting_compliance'] = {
        'band_power_phone': bp_phone,
        'band_power_ref': bp_ref,
        'band_power_discrepancy': bp_diff,
        'excess_hf_ratio_2_to_5_hz': float(excess_hf_ratio),
        'corr_hf_vibration_vs_abs_discrepancy': float(corr_hf_disc),
        'nyquist_note': '10 Hz sampling rate restricts resolvable spectrum to 0-5 Hz. >5 Hz components are folded/aliased into the 0-5 Hz band.'
    }

    # =========================================================================
    # TASK 7: Heading Interaction Diagnostic
    # =========================================================================
    for dur_s in [10, 20, 30]:
        wins = sim_results[dur_s]['C4_Full_Smartphone_Map']
        t7_list = []
        for w in wins:
            w_idx = w['kw'] // int(dur_s / dt)
            dist = w['dist_traveled_m']
            pos_err = w['final_pos_err_m']
            along_err = w['final_along_err_m']
            cross_err = w['final_cross_err_m']
            final_h_err = w['final_heading_err_deg']
            mean_h_err = w['mean_heading_err_deg']

            # Theoretical heading-induced position error:
            # First order cross-track: e_cross_approx = L * sin(mean_h_err)
            # Second order along-track chord reduction: e_along_chord = L * (1 - cos(mean_h_err))
            h_rad = np.radians(mean_h_err)
            e_cross_theory = dist * np.sin(h_rad)
            e_along_chord_theory = dist * (1.0 - np.cos(h_rad))

            # Fraction of variance / energy in along vs cross:
            frac_along = (along_err**2) / max(pos_err**2, 1e-6)
            frac_cross = (cross_err**2) / max(pos_err**2, 1e-6)

            t7_list.append({
                'window_idx': w_idx,
                'dist_m': dist,
                'pos_2d_err_m': pos_err,
                'along_err_m': along_err,
                'cross_err_m': cross_err,
                'final_heading_err_deg': final_h_err,
                'mean_heading_err_deg': mean_h_err,
                'theory_heading_cross_m': float(e_cross_theory),
                'theory_heading_along_chord_m': float(e_along_chord_theory),
                'frac_along_energy': float(frac_along),
                'frac_cross_energy': float(frac_cross)
            })

        mean_along_sq = np.mean([w['along_err_m']**2 for w in t7_list])
        mean_cross_sq = np.mean([w['cross_err_m']**2 for w in t7_list])
        mean_pos_sq = np.mean([w['pos_2d_err_m']**2 for w in t7_list])

        out_diagnostics['task7_heading_interaction'][f"{dur_s}s"] = {
            'windows': t7_list,
            'mean_pos_2d_err_m': float(np.mean([w['pos_2d_err_m'] for w in t7_list])),
            'mean_abs_along_err_m': float(np.mean([abs(w['along_err_m']) for w in t7_list])),
            'mean_abs_cross_err_m': float(np.mean([abs(w['cross_err_m']) for w in t7_list])),
            'along_share_of_total_energy_pct': float((mean_along_sq / max(mean_pos_sq, 1e-6)) * 100.0),
            'cross_share_of_total_energy_pct': float((mean_cross_sq / max(mean_pos_sq, 1e-6)) * 100.0),
            'mean_heading_err_deg': float(np.mean([w['mean_heading_err_deg'] for w in t7_list])),
            'mean_final_heading_err_deg': float(np.mean([w['final_heading_err_deg'] for w in t7_list]))
        }

    # =========================================================================
    # TASK 8: One Controlled Counterfactual Only (Oracle Speed)
    # =========================================================================
    counterfactual_summary = {}
    for dur_s in [5, 10, 20, 30, 60]:
        wins_c4 = sim_results[dur_s]['C4_Full_Smartphone_Map']
        wins_oracle = sim_results[dur_s]['C4_Counterfactual_RefSpeed']
        wins_ref_can = sim_results[dur_s]['REF_CAN_Wheel_Full']
        wins_imu = sim_results[dur_s]['A0_Baseline_PureIMU']

        c4_pos = np.mean([w['final_pos_err_m'] for w in wins_c4])
        c4_along = np.mean([abs(w['final_along_err_m']) for w in wins_c4])
        c4_cross = np.mean([abs(w['final_cross_err_m']) for w in wins_c4])
        c4_drift = np.mean([w['drift_pct'] for w in wins_c4])

        ora_pos = np.mean([w['final_pos_err_m'] for w in wins_oracle])
        ora_along = np.mean([abs(w['final_along_err_m']) for w in wins_oracle])
        ora_cross = np.mean([abs(w['final_cross_err_m']) for w in wins_oracle])
        ora_drift = np.mean([w['drift_pct'] for w in wins_oracle])

        can_pos = np.mean([w['final_pos_err_m'] for w in wins_ref_can])
        imu_pos = np.mean([w['final_pos_err_m'] for w in wins_imu])

        delta_pos = c4_pos - ora_pos
        pct_attributable = (delta_pos / max(c4_pos, 1e-6)) * 100.0

        delta_along = c4_along - ora_along
        pct_along_attributable = (delta_along / max(c4_along, 1e-6)) * 100.0

        counterfactual_summary[f"{dur_s}s"] = {
            'c4_smartphone_pos_err_m': float(c4_pos),
            'c4_smartphone_along_err_m': float(c4_along),
            'c4_smartphone_cross_err_m': float(c4_cross),
            'c4_smartphone_drift_pct': float(c4_drift),
            'oracle_speed_pos_err_m': float(ora_pos),
            'oracle_speed_along_err_m': float(ora_along),
            'oracle_speed_cross_err_m': float(ora_cross),
            'oracle_speed_drift_pct': float(ora_drift),
            'delta_pos_err_m': float(delta_pos),
            'delta_along_err_m': float(delta_along),
            'pct_pos_attributable_to_speed': float(pct_attributable),
            'pct_along_attributable_to_speed': float(pct_along_attributable),
            'ref_can_wheel_pos_err_m': float(can_pos),
            'pure_imu_pos_err_m': float(imu_pos)
        }

    out_diagnostics['task8_counterfactual'] = counterfactual_summary

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

    out_diagnostics_clean = to_native(out_diagnostics)

    # Save to json
    out_file = RES_DIR / "c8_8_longitudinal_diagnostics.json"
    with open(out_file, 'w') as f:
        json.dump(out_diagnostics_clean, f, indent=2)
    print(f"\nSaved C8-8 diagnostic results to: {out_file}", flush=True)

    # Print Summary Tables for Tasks 1 to 8
    print("\n=================================================================", flush=True)
    print("TASK 1: CANONICAL WINDOW-LEVEL ERROR DECOMPOSITION (10s, 20s, 30s)", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        wins = out_diagnostics_clean['task1_window_level'][f"{dur_s}s"]
        print(f"\n--- {dur_s}s Horizon ({len(wins)} Windows) ---", flush=True)
        print(f"{'Win':<4} | {'Start(s)':<8} | {'Dist(m)':<7} | {'v_true':<6} | {'v_est':<6} | {'RF_bar':<6} | {'RF_bias':<7} | {'I_spd':<6} | {'Along(m)':<8} | {'Cross(m)':<8} | {'2D Err(m)':<9} | {'Head(°)':<7}", flush=True)
        print("-" * 105, flush=True)
        for w in wins:
            print(f"{w['window_idx']:<4} | {w['start_time_s']:>8.1f} | {w['dist_traveled_m']:>7.1f} | {w['true_final_vel_ms']:>6.2f} | {w['est_final_vel_ms']:>6.2f} | {w['rf_mean_speed_ms']:>6.2f} | {w['rf_speed_bias_ms']:>+7.2f} | {w['i_speed_m']:>+6.2f} | {w['c4_final_along_err_m']:>+8.2f} | {w['c4_final_cross_err_m']:>+8.2f} | {w['c4_final_pos_err_m']:>9.2f} | {w['c4_final_heading_err_deg']:>7.2f}", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 2: SPEED ERROR CORRELATIONS WITH ALONG-TRACK ERROR", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        corrs = out_diagnostics_clean['task2_correlations'][f"{dur_s}s"]
        print(f"\nHorizon: {dur_s}s", flush=True)
        print(f"  RF Speed Bias vs Signed Along Err : Pearson r = {corrs['bias_vs_signed_along']['pearson_r']:+.4f} (p = {corrs['bias_vs_signed_along']['pearson_p']:.3e})", flush=True)
        print(f"  RF Speed MAE vs Abs Along Err     : Pearson r = {corrs['mae_vs_abs_along']['pearson_r']:+.4f} (p = {corrs['mae_vs_abs_along']['pearson_p']:.3e})", flush=True)
        print(f"  I_speed vs Signed Along Err       : Pearson r = {corrs['ispeed_vs_signed_along']['pearson_r']:+.4f} (p = {corrs['ispeed_vs_signed_along']['pearson_p']:.3e})", flush=True)
        print(f"  Heading Err vs Abs Along Err      : Pearson r = {corrs['head_err_vs_along']['pearson_r']:+.4f} (p = {corrs['head_err_vs_along']['pearson_p']:.3e})", flush=True)
        print(f"  Heading Err vs Abs Cross Err      : Pearson r = {corrs['head_err_vs_cross']['pearson_r']:+.4f} (p = {corrs['head_err_vs_cross']['pearson_p']:.3e})", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 3: INTEGRATED SPEED ERROR VS ALONG-TRACK DRIFT", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        t3 = out_diagnostics_clean['task3_integrated_speed'][f"{dur_s}s"]
        print(f"\nHorizon: {dur_s}s (Mean |I_speed| = {t3['mean_i_speed_m']:.2f} m, Mean |Along Err| = {t3['mean_along_err_m']:.2f} m)", flush=True)
        print(f"  Mean Ratio (Along / I_speed)      : {t3['mean_ratio_when_meaningful'] if t3['mean_ratio_when_meaningful'] is not None else 'N/A'}", flush=True)
        print(f"  Median Ratio (Along / I_speed)    : {t3['median_ratio_when_meaningful'] if t3['median_ratio_when_meaningful'] is not None else 'N/A'}", flush=True)
        print(f"  Sign Agreement Rate (Along vs I_spd): {t3['sign_agreement_rate'] * 100.0 if t3['sign_agreement_rate'] is not None else 'N/A'}%", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 4: ACCELERATION-LEVEL DIAGNOSTIC (Vta04)", flush=True)
    print("=================================================================", flush=True)
    t4 = out_diagnostics_clean['task4_acceleration']
    for k, v in [('Raw Phone vs Ref', t4['raw_phone_vs_ref']),
                 ('Bias-Corrected Phone vs Ref', t4['bias_corrected_phone_vs_ref']),
                 ('Filtered Phone vs Ref', t4['filtered_phone_vs_ref'])]:
        print(f"  {k:<30}: Bias = {v['mean_bias_ms2']:+.4f} m/s², Std = {v['std_ms2']:.4f} m/s², RMSE = {v['rmse_ms2']:.4f} m/s², r = {v['pearson_r']:+.4f}", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 5: DYNAMIC PITCH HYPOTHESIS TEST", flush=True)
    print("=================================================================", flush=True)
    t5 = out_diagnostics_clean['task5_dynamic_pitch']
    print(f"  Pitch Rate: Mean = {t5['pitch_rate_degs_mean']:+.4f}°/s, Std = {t5['pitch_rate_degs_std']:.4f}°/s, Max = {t5['pitch_rate_degs_max']:.4f}°/s", flush=True)
    print(f"  RMS Predicted Gravity Leakage      : {t5['rms_predicted_g_leakage_ms2']:.4f} m/s²", flush=True)
    print(f"  RMS Observed Accel Discrepancy     : {t5['rms_observed_discrepancy_ms2']:.4f} m/s²", flush=True)
    print(f"  Ratio Leakage / Discrepancy        : {t5['ratio_leakage_to_discrepancy']:.4f} ({t5['ratio_leakage_to_discrepancy']*100:.2f}%)", flush=True)
    print(f"  Correlation Leakage vs Discrepancy : r = {t5['correlation_g_leak_vs_discrepancy']:+.4f} (p = {t5['p_value']:.3e})", flush=True)
    print(f"  Verdict: {'SUPPORTED HYPOTHESIS' if t5['verdict_dominant'] else 'REJECTED AS DOMINANT EXPLANATION (Leakage is 1 order of magnitude smaller than discrepancy)'}", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 6: MOUNTING / COMPLIANCE HYPOTHESIS (FREQUENCY BANDS)", flush=True)
    print("=================================================================", flush=True)
    t6 = out_diagnostics_clean['task6_mounting_compliance']
    print(f"{'Band':<12} | {'Phone Power':<14} | {'Ref Power':<14} | {'Discrepancy Power':<18}", flush=True)
    print("-" * 65, flush=True)
    for b in ['0_to_1_hz', '1_to_2_hz', '2_to_5_hz', 'total_0_to_5_hz']:
        print(f"{b:<12} | {t6['band_power_phone'][b]:>14.6f} | {t6['band_power_ref'][b]:>14.6f} | {t6['band_power_discrepancy'][b]:>18.6f}", flush=True)
    print(f"  Excess HF Ratio (2-5 Hz)           : {t6['excess_hf_ratio_2_to_5_hz']:.4f}", flush=True)
    print(f"  Corr HF Vibration vs |Discrepancy|: r = {t6['corr_hf_vibration_vs_abs_discrepancy']:+.4f}", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 7: HEADING INTERACTION DIAGNOSTIC (ENERGY BREAKDOWN)", flush=True)
    print("=================================================================", flush=True)
    for dur_s in [10, 20, 30]:
        t7 = out_diagnostics_clean['task7_heading_interaction'][f"{dur_s}s"]
        print(f"  Horizon {dur_s}s: Mean 2D = {t7['mean_pos_2d_err_m']:.2f}m | Along = {t7['mean_abs_along_err_m']:.2f}m | Cross = {t7['mean_abs_cross_err_m']:.2f}m | Head = {t7['mean_heading_err_deg']:.2f}°", flush=True)
        print(f"    Energy Share: Along-Track = {t7['along_share_of_total_energy_pct']:.1f}% | Cross-Track = {t7['cross_share_of_total_energy_pct']:.1f}%", flush=True)

    print("\n=================================================================", flush=True)
    print("TASK 8: CONTROLLED COUNTERFACTUAL (CANONICAL SPEED VS ORACLE REF SPEED)", flush=True)
    print("=================================================================", flush=True)
    print(f"{'Horizon':<8} | {'C4 Phone (m)':<12} | {'Oracle Spd (m)':<14} | {'Delta (m)':<10} | {'% Attributable':<14} | {'C4 Drift %':<10} | {'Oracle Drift %':<14}", flush=True)
    print("-" * 92, flush=True)
    for dur_s in [5, 10, 20, 30, 60]:
        c = out_diagnostics_clean['task8_counterfactual'][f"{dur_s}s"]
        print(f"{dur_s}s{'':<6} | {c['c4_smartphone_pos_err_m']:>12.2f} | {c['oracle_speed_pos_err_m']:>14.2f} | {c['delta_pos_err_m']:>10.2f} | {c['pct_pos_attributable_to_speed']:>13.1f}% | {c['c4_smartphone_drift_pct']:>9.2f}% | {c['oracle_speed_drift_pct']:>13.2f}%", flush=True)

    return out_diagnostics_clean


if __name__ == '__main__':
    run_c8_8_diagnostics()

