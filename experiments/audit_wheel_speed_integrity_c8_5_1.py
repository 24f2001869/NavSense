"""
SIH26168 - Stage C8-5.1: Wheel-Speed Integrity & Observability Audit
Module: experiments/audit_wheel_speed_integrity_c8_5_1.py

Forensic robustness and filter consistency audit for chassis CAN wheel-speed fusion:
- Sub-Audit A: Wheel-speed accuracy vs RTK VBOX ground truth (speed/accel/braking stratified).
- Sub-Audit B: Wheel slip & rear-wheel differential audit across operational regimes
               (acceleration, hard braking, cornering, low-speed, ABS events).
- Sub-Audit C: Tire-radius sensitivity sweep (r_w * [0.98, 0.99, 1.00, 1.01, 1.02]) on 30s/60s drift.
- Sub-Audit D: Measurement covariance sensitivity sweep (sigma_wheel in [0.10, 0.20, 0.30, 0.50, 1.00] m/s).
- Sub-Audit E: Huber adaptive variance scaling sensitivity (None, Weak, Nominal, Strong).
- Sub-Audit F: Filter consistency & overconfidence audit (NEES & actual error vs 3-sigma envelope).
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D
from src.navigation.nhc import NonHolonomicConstraint
from src.navigation.zupt import ZeroVelocityUpdate, CausalStationaryDetector
from src.navigation.kinematic_constraints import SoftLongitudinalAccelerationConstraint
from src.navigation.wheel_odometry import ChassisWheelSpeedFusion, compute_wheel_speed_from_can
from experiments.run_wheel_speed_fusion_c8_5 import compute_bcac_series

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def prepare_trip_data_c8_5_1(trip_name: str, dt: float = 0.1) -> Dict[str, Any]:
    """Loads trip telemetry, IMU, CAN wheel speeds, and constructs window indices."""
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    wheel_rl = df_v['wheel_rl_rads'].values
    wheel_rr = df_v['wheel_rr_rads'].values
    wheel_fl = df_v['wheel_fl_rads'].values
    wheel_fr = df_v['wheel_fr_rads'].values
    wheel_speed = 0.5 * (wheel_rl + wheel_rr) * 0.2766

    indicated_speed = df_v['indicated_speed_kmh'].values / 3.6 if 'indicated_speed_kmh' in df_v else wheel_speed
    accel_long = df_v['veh_accel_long_ms2'].values if 'veh_accel_long_ms2' in df_v else np.gradient(speed, dt)
    accel_lat = df_v['veh_accel_lat_ms2'].values if 'veh_accel_lat_ms2' in df_v else np.zeros_like(speed)
    yaw_rate = df_v['yaw_rate_degs'].values if 'yaw_rate_degs' in df_v else np.gradient(heading, dt)
    steer = df_v['steer_angle_deg'].values if 'steer_angle_deg' in df_v else np.zeros_like(speed)
    brake_psi = df_v['brake_pressure_psi'].values if 'brake_pressure_psi' in df_v else np.zeros_like(speed)
    time_s = df_v['time_s'].values if 'time_s' in df_v else np.arange(n) * dt

    acc_v, gyro_v, R_vp, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124], dtype=np.float64)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    # Window stride matching C8-4 and C8-5
    stride = 150 if trip_name == 'Vta04' else 200
    windows = {}
    for h in [10, 20, 30, 60]:
        w_dur = int(round(h / dt))
        max_start = n - w_dur
        windows[h] = list(range(0, max_start, stride))

    return {
        'trip_name': trip_name,
        'dt': dt,
        'acc_v': acc_v,
        'gyro_v': gyro_v,
        'speed': speed,
        'wheel_speed': wheel_speed,
        'indicated_speed': indicated_speed,
        'wheel_rl': wheel_rl,
        'wheel_rr': wheel_rr,
        'wheel_fl': wheel_fl,
        'wheel_fr': wheel_fr,
        'accel_long': accel_long,
        'accel_lat': accel_lat,
        'yaw_rate': yaw_rate,
        'steer': steer,
        'brake_psi': brake_psi,
        'time_s': time_s,
        'heading': heading,
        'gt_e': gt_e,
        'gt_n': gt_n,
        'gt_u': gt_u,
        'gt_ve': gt_ve,
        'gt_vn': gt_vn,
        'gt_vu': gt_vu,
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'windows': windows,
        'n_epochs': n
    }



# =============================================================================
# 1. SUB-AUDIT A: WHEEL SPEED ACCURACY VS VBOX GROUND TRUTH
# =============================================================================

def audit_wheel_speed_accuracy(trip_data: Dict[str, Any], trip_name: str) -> Dict[str, Any]:
    """Audits wheel-speed accuracy vs RTK VBOX speed."""
    v_vbox = trip_data['speed']
    v_wheel = trip_data['wheel_speed']
    v_ind = trip_data['indicated_speed']
    acc_long = trip_data['accel_long']
    brake_psi = trip_data['brake_psi']

    err_wheel = v_wheel - v_vbox
    err_ind = v_ind - v_vbox

    # Speed bins: [0, 5), [5, 10), [10, 15), [15, 20), [20, 25), >= 25 m/s
    speed_bins = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 100.0)]
    speed_strat = {}
    for lo, hi in speed_bins:
        mask = (v_vbox >= lo) & (v_vbox < hi)
        lbl = f"[{lo:.0f}, {hi:.0f}) m/s" if hi < 100 else f">= {lo:.0f} m/s"
        if np.sum(mask) > 10:
            e = err_wheel[mask]
            speed_strat[lbl] = {
                'count': int(np.sum(mask)),
                'mean_bias_ms': float(np.mean(e)),
                'std_ms': float(np.std(e)),
                'mae_ms': float(np.mean(np.abs(e))),
                'p95_abs_err_ms': float(np.percentile(np.abs(e), 95))
            }

    # Acceleration bins: < -2.0, [-2.0, -0.5), [-0.5, 0.5), [0.5, 2.0), >= 2.0 m/s^2
    acc_bins = [(-100.0, -2.0), (-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0), (2.0, 100.0)]
    acc_strat = {}
    for lo, hi in acc_bins:
        mask = (acc_long >= lo) & (acc_long < hi)
        if lo == -100.0:
            lbl = "< -2.0 m/s² (hard brake)"
        elif hi == 100.0:
            lbl = ">= 2.0 m/s² (hard accel)"
        else:
            lbl = f"[{lo:.1f}, {hi:.1f}) m/s²"

        if np.sum(mask) > 10:
            e = err_wheel[mask]
            acc_strat[lbl] = {
                'count': int(np.sum(mask)),
                'mean_bias_ms': float(np.mean(e)),
                'std_ms': float(np.std(e)),
                'mae_ms': float(np.mean(np.abs(e)))
            }

    # Braking specific audit (accel < -1.0 or brake pressure > 50 psi)
    brake_mask = (acc_long < -1.0) | (brake_psi > 50.0)
    brake_stats = {}
    if np.sum(brake_mask) > 10:
        eb = err_wheel[brake_mask]
        brake_stats = {
            'count': int(np.sum(brake_mask)),
            'mean_bias_ms': float(np.mean(eb)),
            'std_ms': float(np.std(eb)),
            'mae_ms': float(np.mean(np.abs(eb))),
            'p95_abs_err_ms': float(np.percentile(np.abs(eb), 95)),
            'max_abs_err_ms': float(np.max(np.abs(eb)))
        }

    return {
        'trip_name': trip_name,
        'total_samples': len(v_vbox),
        'overall': {
            'mean_bias_ms': float(np.mean(err_wheel)),
            'std_ms': float(np.std(err_wheel)),
            'mae_ms': float(np.mean(np.abs(err_wheel))),
            'median_ms': float(np.median(err_wheel)),
            'p90_abs_err_ms': float(np.percentile(np.abs(err_wheel), 90)),
            'p95_abs_err_ms': float(np.percentile(np.abs(err_wheel), 95)),
            'p99_abs_err_ms': float(np.percentile(np.abs(err_wheel), 99)),
            'max_abs_err_ms': float(np.max(np.abs(err_wheel)))
        },
        'indicated_speed_overall': {
            'mean_bias_ms': float(np.mean(err_ind)),
            'std_ms': float(np.std(err_ind)),
            'mae_ms': float(np.mean(np.abs(err_ind)))
        },
        'speed_stratified': speed_strat,
        'acceleration_stratified': acc_strat,
        'braking_regime': brake_stats
    }


# =============================================================================
# 2. SUB-AUDIT B: WHEEL SLIP & REAR DIFFERENTIAL AUDIT
# =============================================================================

def audit_wheel_slip_and_differentials(trip_data: Dict[str, Any], trip_name: str) -> Dict[str, Any]:
    """Audits rear wheel differential (RL vs RR), front-rear slip, and operational regimes."""
    rw = 0.2766
    w_rl = trip_data['wheel_rl']
    w_rr = trip_data['wheel_rr']
    w_fl = trip_data['wheel_fl']
    w_fr = trip_data['wheel_fr']
    v_vbox = trip_data['speed']
    acc_long = trip_data['accel_long']
    acc_lat = trip_data['accel_lat']
    yaw_rate = trip_data['yaw_rate']
    steer = trip_data['steer']
    brake_psi = trip_data['brake_psi']
    time_s = trip_data['time_s']

    v_rl = w_rl * rw
    v_rr = w_rr * rw
    v_rear = 0.5 * (v_rl + v_rr)
    v_front = 0.5 * (w_fl + w_fr) * rw

    delta_v_rear = np.abs(v_rl - v_rr)
    delta_v_fr = v_front - v_rear  # positive = front wheel drive slip
    slip_ratio = (v_rear - v_vbox) / np.maximum(v_vbox, 1.0)

    # Operational Regimes:
    # 1. Acceleration: acc_long > 1.0 m/s^2
    # 2. Hard Braking: acc_long < -2.5 m/s^2
    # 3. Cornering: |yaw_rate| > 12.0 deg/s or |acc_lat| > 1.5 m/s^2
    # 4. Low Speed: 0.1 < v_vbox < 2.0 m/s
    # 5. Cruising: |acc_long| <= 0.5 and |acc_lat| <= 0.5 and v_vbox >= 5.0
    # 6. Standstill: v_vbox <= 0.1

    regimes = {
        'Cruising': (np.abs(acc_long) <= 0.5) & (np.abs(acc_lat) <= 0.5) & (v_vbox >= 5.0),
        'Acceleration': (acc_long > 1.0),
        'Hard Braking': (acc_long < -2.5),
        'Cornering': (np.abs(yaw_rate) > 12.0) | (np.abs(acc_lat) > 1.5),
        'Low Speed': (v_vbox > 0.1) & (v_vbox < 2.0),
        'Standstill': (v_vbox <= 0.1)
    }

    regime_results = {}
    for r_name, r_mask in regimes.items():
        if np.sum(r_mask) > 5:
            d_r = delta_v_rear[r_mask]
            s_r = slip_ratio[r_mask]
            d_fr = delta_v_fr[r_mask]
            regime_results[r_name] = {
                'count': int(np.sum(r_mask)),
                'delta_rear_median_ms': float(np.median(d_r)),
                'delta_rear_p90_ms': float(np.percentile(d_r, 90)),
                'delta_rear_p99_ms': float(np.percentile(d_r, 99)),
                'delta_rear_max_ms': float(np.max(d_r)),
                'mean_slip_ratio': float(np.mean(s_r)),
                'front_rear_diff_mean_ms': float(np.mean(d_fr)),
                'front_rear_diff_max_ms': float(np.max(d_fr))
            }

    # Percentile distribution of delta_v_rear
    p_dist = {
        'p50_ms': float(np.percentile(delta_v_rear, 50)),
        'p75_ms': float(np.percentile(delta_v_rear, 75)),
        'p90_ms': float(np.percentile(delta_v_rear, 90)),
        'p95_ms': float(np.percentile(delta_v_rear, 95)),
        'p99_ms': float(np.percentile(delta_v_rear, 99)),
        'p99_9_ms': float(np.percentile(delta_v_rear, 99.9)),
        'max_ms': float(np.max(delta_v_rear))
    }

    # Forensic audit of maximum rear differential event
    idx_max = int(np.argmax(delta_v_rear))
    max_event = {
        'timestamp_s': float(time_s[idx_max]),
        'delta_v_rear_ms': float(delta_v_rear[idx_max]),
        'v_vbox_ms': float(v_vbox[idx_max]),
        'v_rl_ms': float(v_rl[idx_max]),
        'v_rr_ms': float(v_rr[idx_max]),
        'accel_long_ms2': float(acc_long[idx_max]),
        'accel_lat_ms2': float(acc_lat[idx_max]),
        'yaw_rate_degs': float(yaw_rate[idx_max]),
        'steer_deg': float(steer[idx_max]),
        'brake_pressure_psi': float(brake_psi[idx_max])
    }

    # Count of extreme events
    count_gt_1ms = int(np.sum(delta_v_rear > 1.0))
    count_gt_2ms = int(np.sum(delta_v_rear > 2.0))
    count_gt_3ms = int(np.sum(delta_v_rear > 3.0))

    return {
        'trip_name': trip_name,
        'percentile_distribution': p_dist,
        'regimes': regime_results,
        'extreme_event_counts': {
            'gt_1_0_ms': count_gt_1ms,
            'gt_1_0_ms_pct': float(count_gt_1ms / len(delta_v_rear) * 100.0),
            'gt_2_0_ms': count_gt_2ms,
            'gt_2_0_ms_pct': float(count_gt_2ms / len(delta_v_rear) * 100.0),
            'gt_3_0_ms': count_gt_3ms,
            'gt_3_0_ms_pct': float(count_gt_3ms / len(delta_v_rear) * 100.0)
        },
        'maximum_differential_event': max_event
    }


# =============================================================================
# 3. FAST SIMULATION ENGINE FOR SENSITIVITY SWEEPS
# =============================================================================

def run_fast_window_sim(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    tire_radius: float = 0.2766,
    sigma_wheel: float = 0.20,
    nis_gate_3d: float = 11.345,
    apply_nis_gate: bool = True
) -> Dict[str, Any]:
    """Runs a single window simulation under Condition C2 (Velocity + NHC) with specific parameters."""
    dt = trip_data['dt']
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    wheel_rl = trip_data['wheel_rl']
    wheel_rr = trip_data['wheel_rr']
    gt_e = trip_data['gt_e']
    gt_n = trip_data['gt_n']
    gt_u = trip_data['gt_u']
    gt_ve = trip_data['gt_ve']
    gt_vn = trip_data['gt_vn']
    gt_vu = trip_data['gt_vu']
    heading = trip_data['heading']
    sigma_series = trip_data['sigma_series']
    ba_stat = trip_data['ba_stat']

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

    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )
    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    b1_constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )
    wheel_fusion = ChassisWheelSpeedFusion(
        sigma_wheel=sigma_wheel,
        sigma_lat=0.50,
        sigma_vert=0.50,
        tire_radius=tire_radius,
        nis_gate_3d=nis_gate_3d
    )

    # Wheel speed array with custom radius
    v_wheel_arr = 0.5 * (wheel_rl + wheel_rr) * tire_radius

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]
        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]
        cur_wheel_spd = v_wheel_arr[step_k]

        # 1. ESKF strapdown propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Coupled 3D Body Velocity update (Forward wheel speed + NHC)
        wheel_fusion.update_eskf_3d_velocity(eskf, cur_wheel_spd, apply_nis_gate=apply_nis_gate)

        # 3. ZUPT update
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 4. B1 Strict Attitude Freeze
        b1_constraint.update_eskf(eskf)

    # Compute terminal errors
    k_end = kw + w_dur - 1
    p_est = eskf.pos_n[0:2]
    p_true = np.array([gt_e[k_end], gt_n[k_end]])
    err_vec = p_est - p_true
    total_drift = float(np.linalg.norm(err_vec))

    # Along-track / Cross-track projection
    h_end = np.radians(heading[k_end])
    u_fwd = np.array([np.sin(h_end), np.cos(h_end)])
    u_lat = np.array([np.cos(h_end), -np.sin(h_end)])

    along_track = float(abs(np.dot(err_vec, u_fwd)))
    cross_track = float(abs(np.dot(err_vec, u_lat)))

    P_2d = eskf.P[0:2, 0:2]
    cov_trace = float(np.trace(P_2d))

    # NEES calculation: e^T * P^-1 * e
    try:
        P_inv = np.linalg.inv(P_2d)
        nees = float(err_vec.T @ P_inv @ err_vec)
    except Exception:
        nees = np.nan

    return {
        'total_drift_m': total_drift,
        'along_track_m': along_track,
        'cross_track_m': cross_track,
        'cov_trace_m2': cov_trace,
        'sigma_p_m': float(np.sqrt(cov_trace)),
        'nees': nees,
        'err_vec': err_vec.tolist()
    }


# =============================================================================
# 4. SUB-AUDIT C: TIRE-RADIUS SENSITIVITY SWEEP
# =============================================================================

def audit_tire_radius_sensitivity(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Sweeps tire rolling radius r_w * [0.98, 0.99, 1.00, 1.01, 1.02] across 30s and 60s."""
    nominal_rw = 0.2766
    scale_factors = [0.98, 0.99, 1.00, 1.01, 1.02]
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}
        for dur_s in [30, 60]:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"
            trip_res[dur_key] = {}

            for s_factor in scale_factors:
                r_val = nominal_rw * s_factor
                drifts = []
                alongs = []
                crosses = []

                for kw in w_indices:
                    res = run_fast_window_sim(trip_data, kw, w_dur, tire_radius=r_val)
                    drifts.append(res['total_drift_m'])
                    alongs.append(res['along_track_m'])
                    crosses.append(res['cross_track_m'])

                trip_res[dur_key][f"{s_factor:.2f}"] = {
                    'scale_factor': s_factor,
                    'tire_radius_m': r_val,
                    'mean_drift_m': float(np.mean(drifts)),
                    'mean_along_m': float(np.mean(alongs)),
                    'mean_cross_m': float(np.mean(crosses)),
                    'std_drift_m': float(np.std(drifts))
                }

            # Compute sensitivity gradient: d(drift) / d(% radius error)
            d_nom = trip_res[dur_key]["1.00"]['mean_drift_m']
            d_p2 = trip_res[dur_key]["1.02"]['mean_drift_m']
            d_m2 = trip_res[dur_key]["0.98"]['mean_drift_m']
            sens_pct = (abs(d_p2 - d_nom) + abs(d_m2 - d_nom)) / (2.0 * max(d_nom, 1e-3) * 0.02) * 100.0
            trip_res[dur_key]['sensitivity_pct_drift_per_pct_radius'] = float(sens_pct)

        results[t_name] = trip_res

    return results


# =============================================================================
# 5. SUB-AUDIT D: WHEEL-SPEED MEASUREMENT COVARIANCE SENSITIVITY
# =============================================================================

def audit_covariance_sensitivity(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Sweeps measurement noise sigma_wheel in [0.10, 0.20, 0.30, 0.50, 1.00] m/s."""
    sigmas = [0.10, 0.20, 0.30, 0.50, 1.00]
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}
        for dur_s in [30, 60]:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"
            trip_res[dur_key] = {}

            for sig in sigmas:
                drifts = []
                alongs = []
                crosses = []
                cov_traces = []

                for kw in w_indices:
                    res = run_fast_window_sim(trip_data, kw, w_dur, sigma_wheel=sig)
                    drifts.append(res['total_drift_m'])
                    alongs.append(res['along_track_m'])
                    crosses.append(res['cross_track_m'])
                    cov_traces.append(res['cov_trace_m2'])

                trip_res[dur_key][f"{sig:.2f}"] = {
                    'sigma_wheel_ms': sig,
                    'R_var_m2s2': sig**2,
                    'mean_drift_m': float(np.mean(drifts)),
                    'mean_along_m': float(np.mean(alongs)),
                    'mean_cross_m': float(np.mean(crosses)),
                    'mean_cov_trace_m2': float(np.mean(cov_traces))
                }

        results[t_name] = trip_res

    return results


# =============================================================================
# 6. SUB-AUDIT E: HUBER ADAPTIVE VARIANCE SCALING SENSITIVITY
# =============================================================================

def audit_huber_scaling_sensitivity(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluates filter robustness across Huber threshold settings."""
    modes = {
        'No_Huber': {'apply_gate': False, 'gate_3d': 11.345},
        'Weak_Huber': {'apply_gate': True, 'gate_3d': 16.27},   # 99.9% gate
        'Nominal_Huber': {'apply_gate': True, 'gate_3d': 11.345}, # 99% gate
        'Strong_Huber': {'apply_gate': True, 'gate_3d': 7.815}   # 95% gate
    }
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}
        for dur_s in [30, 60]:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"
            trip_res[dur_key] = {}

            for m_name, m_cfg in modes.items():
                drifts = []
                alongs = []
                crosses = []

                for kw in w_indices:
                    res = run_fast_window_sim(
                        trip_data, kw, w_dur,
                        nis_gate_3d=m_cfg['gate_3d'],
                        apply_nis_gate=m_cfg['apply_gate']
                    )
                    drifts.append(res['total_drift_m'])
                    alongs.append(res['along_track_m'])
                    crosses.append(res['cross_track_m'])

                trip_res[dur_key][m_name] = {
                    'mode': m_name,
                    'mean_drift_m': float(np.mean(drifts)),
                    'std_drift_m': float(np.std(drifts)),
                    'max_drift_m': float(np.max(drifts)),
                    'mean_along_m': float(np.mean(alongs)),
                    'mean_cross_m': float(np.mean(crosses))
                }

        results[t_name] = trip_res

    return results


# =============================================================================
# 7. SUB-AUDIT F: FILTER CONSISTENCY & OVERCONFIDENCE AUDIT (NEES)
# =============================================================================

def audit_filter_consistency(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Audits actual position error vs filter covariance envelope and computes NEES."""
    horizons = [10, 20, 30, 60]
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}
        for dur_s in horizons:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"

            errors = []
            sigmas_1 = []
            sigmas_2 = []
            sigmas_3 = []
            nees_list = []
            ratios = []

            for kw in w_indices:
                res = run_fast_window_sim(trip_data, kw, w_dur)
                e = res['total_drift_m']
                sig1 = res['sigma_p_m']
                errors.append(e)
                sigmas_1.append(sig1)
                sigmas_2.append(2.0 * sig1)
                sigmas_3.append(3.0 * sig1)
                ratios.append(e / max(sig1, 1e-6))
                if not np.isnan(res['nees']):
                    nees_list.append(res['nees'])

            n_wins = len(errors)
            in_1sig = int(np.sum(np.array(errors) <= np.array(sigmas_1)))
            in_2sig = int(np.sum(np.array(errors) <= np.array(sigmas_2)))
            in_3sig = int(np.sum(np.array(errors) <= np.array(sigmas_3)))

            mean_nees = float(np.mean(nees_list)) if nees_list else 0.0
            overconfidence_factor = float(np.sqrt(mean_nees / 2.0)) if mean_nees > 0 else 0.0

            trip_res[dur_key] = {
                'window_count': n_wins,
                'mean_actual_error_m': float(np.mean(errors)),
                'mean_sigma_p_m': float(np.mean(sigmas_1)),
                'mean_3sigma_p_m': float(np.mean(sigmas_3)),
                'mean_error_to_sigma_ratio': float(np.mean(ratios)),
                'median_error_to_sigma_ratio': float(np.median(ratios)),
                'nees_statistics': {
                    'expected_2d_nees': 2.0,
                    'empirical_mean_nees': mean_nees,
                    'empirical_median_nees': float(np.median(nees_list)) if nees_list else 0.0,
                    'empirical_p90_nees': float(np.percentile(nees_list, 90)) if nees_list else 0.0,
                    'overconfidence_factor': overconfidence_factor
                },
                'envelope_containment': {
                    'within_1sigma_count': in_1sig,
                    'within_1sigma_pct': float(in_1sig / n_wins * 100.0),
                    'theoretical_1sigma_pct': 39.3,
                    'within_2sigma_count': in_2sig,
                    'within_2sigma_pct': float(in_2sig / n_wins * 100.0),
                    'theoretical_2sigma_pct': 86.5,
                    'within_3sigma_count': in_3sig,
                    'within_3sigma_pct': float(in_3sig / n_wins * 100.0),
                    'theoretical_3sigma_pct': 98.9
                }
            }

        results[t_name] = trip_res

    return results


# =============================================================================
# 8. PUBLICATION DIAGNOSTIC VISUALIZATION
# =============================================================================

def generate_diagnostic_figure(
    audit_a_vta02: Dict[str, Any],
    audit_a_vta04: Dict[str, Any],
    audit_b_vta02: Dict[str, Any],
    audit_b_vta04: Dict[str, Any],
    audit_c: Dict[str, Any],
    audit_d: Dict[str, Any],
    audit_e: Dict[str, Any],
    audit_f: Dict[str, Any],
    out_path: Path
):
    """Generates 6-panel publication diagnostic figure."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Stage C8-5.1: Wheel-Speed Integrity, Observability & Filter Consistency Audit", fontsize=15, fontweight='bold')

    # Panel 1: Sub-Audit A: Wheel Speed Error vs Speed Bins
    ax = axes[0, 0]
    all_bins_order = ['[0, 5) m/s', '[5, 10) m/s', '[10, 15) m/s', '[15, 20) m/s', '[20, 25) m/s', '>= 25 m/s']
    strat_02 = audit_a_vta02.get('speed_stratified', {})
    strat_04 = audit_a_vta04.get('speed_stratified', {})
    bins_common = [b for b in all_bins_order if (b in strat_02) or (b in strat_04)]
    if not bins_common:
        bins_common = list(strat_02.keys())

    maes_02 = [strat_02.get(b, {}).get('mae_ms', np.nan) for b in bins_common]
    biases_02 = [strat_02.get(b, {}).get('mean_bias_ms', np.nan) for b in bins_common]
    maes_04 = [strat_04.get(b, {}).get('mae_ms', np.nan) for b in bins_common]
    biases_04 = [strat_04.get(b, {}).get('mean_bias_ms', np.nan) for b in bins_common]

    x = np.arange(len(bins_common))
    width = 0.35
    ax.bar(x - width/2, np.nan_to_num(maes_02), width, label='Vta02 MAE', color='#2b5c8f', alpha=0.85)
    ax.bar(x + width/2, np.nan_to_num(maes_04), width, label='Vta04 MAE', color='#d95f02', alpha=0.85)

    valid_02 = ~np.isnan(biases_02)
    valid_04 = ~np.isnan(biases_04)
    if np.any(valid_02):
        ax.plot(x[valid_02], np.array(biases_02)[valid_02], 'o--', color='#1b9e77', label='Vta02 Bias')
    if np.any(valid_04):
        ax.plot(x[valid_04], np.array(biases_04)[valid_04], 's--', color='#e7298a', label='Vta04 Bias')

    ax.set_xticks(x)
    ax.set_xticklabels(bins_common, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Error (m/s)")
    ax.set_title("Sub-Audit A: Wheel Speed Error vs Vehicle Speed", fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)

    # Panel 2: Sub-Audit B: Rear Wheel Differential by Regime
    ax = axes[0, 1]
    reg_names = ['Cruising', 'Acceleration', 'Hard Braking', 'Cornering', 'Low Speed']
    p90_02 = [audit_b_vta02['regimes'].get(r, {}).get('delta_rear_p90_ms', 0) for r in reg_names]
    max_02 = [audit_b_vta02['regimes'].get(r, {}).get('delta_rear_max_ms', 0) for r in reg_names]
    p90_04 = [audit_b_vta04['regimes'].get(r, {}).get('delta_rear_p90_ms', 0) for r in reg_names]
    max_04 = [audit_b_vta04['regimes'].get(r, {}).get('delta_rear_max_ms', 0) for r in reg_names]

    x = np.arange(len(reg_names))
    ax.bar(x - width/2, p90_02, width, label='Vta02 P90 Diff', color='#4575b4', alpha=0.85)
    ax.bar(x + width/2, p90_04, width, label='Vta04 P90 Diff', color='#fc8d59', alpha=0.85)
    ax.scatter(x - width/2, max_02, color='#313695', zorder=5, label='Vta02 Max')
    ax.scatter(x + width/2, max_04, color='#d73027', zorder=5, label='Vta04 Max (3.59 m/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(reg_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("Rear Differential |v_RL - v_RR| (m/s)")
    ax.set_title("Sub-Audit B: Rear Differential across Regimes", fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 3: Sub-Audit C: Tire-Radius Sensitivity Curves
    ax = axes[0, 2]
    scales = [-2, -1, 0, 1, 2]  # % delta
    scale_keys = ["0.98", "0.99", "1.00", "1.01", "1.02"]
    d30_02 = [audit_c['Vta02_suburban']['30s'][k]['mean_drift_m'] for k in scale_keys]
    d60_02 = [audit_c['Vta02_suburban']['60s'][k]['mean_drift_m'] for k in scale_keys]
    d30_04 = [audit_c['Vta04_highway']['30s'][k]['mean_drift_m'] for k in scale_keys]
    d60_04 = [audit_c['Vta04_highway']['60s'][k]['mean_drift_m'] for k in scale_keys]

    ax.plot(scales, d30_02, 'o-', color='#2b5c8f', label='Vta02 30s Drift')
    ax.plot(scales, d60_02, 's-', color='#d95f02', label='Vta02 60s Drift')
    ax.plot(scales, d30_04, '^--', color='#1b9e77', label='Vta04 30s Drift')
    ax.plot(scales, d60_04, 'd--', color='#7570b3', label='Vta04 60s Drift')
    ax.set_xlabel("Tire Radius Error (%)")
    ax.set_ylabel("Mean Position Drift (m)")
    ax.set_title("Sub-Audit C: Tire Radius Sensitivity (r_w +/- 2%)", fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 4: Sub-Audit D: Measurement Covariance Sensitivity
    ax = axes[1, 0]
    sig_keys = ["0.10", "0.20", "0.30", "0.50", "1.00"]
    sig_vals = [0.10, 0.20, 0.30, 0.50, 1.00]
    cd30_02 = [audit_d['Vta02_suburban']['30s'][k]['mean_drift_m'] for k in sig_keys]
    cd60_02 = [audit_d['Vta02_suburban']['60s'][k]['mean_drift_m'] for k in sig_keys]
    cd30_04 = [audit_d['Vta04_highway']['30s'][k]['mean_drift_m'] for k in sig_keys]
    cd60_04 = [audit_d['Vta04_highway']['60s'][k]['mean_drift_m'] for k in sig_keys]

    ax.plot(sig_vals, cd30_02, 'o-', color='#2b5c8f', label='Vta02 30s')
    ax.plot(sig_vals, cd60_02, 's-', color='#d95f02', label='Vta02 60s')
    ax.plot(sig_vals, cd30_04, '^--', color='#1b9e77', label='Vta04 30s')
    ax.plot(sig_vals, cd60_04, 'd--', color='#7570b3', label='Vta04 60s')
    ax.axvline(0.20, color='red', linestyle=':', label='Default 0.20 m/s')
    ax.set_xlabel("sigma_wheel (m/s)")
    ax.set_ylabel("Mean Position Drift (m)")
    ax.set_title("Sub-Audit D: Noise Covariance Sensitivity", fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 5: Sub-Audit E: Huber Scaling Sensitivity
    ax = axes[1, 1]
    h_modes = ['No_Huber', 'Weak_Huber', 'Nominal_Huber', 'Strong_Huber']
    h_labels = ['No Huber', 'Weak (99.9%)', 'Nominal (99%)', 'Strong (95%)']
    hd60_02 = [audit_e['Vta02_suburban']['60s'][m]['mean_drift_m'] for m in h_modes]
    hd30_02 = [audit_e['Vta02_suburban']['30s'][m]['mean_drift_m'] for m in h_modes]
    hd60_04 = [audit_e['Vta04_highway']['60s'][m]['mean_drift_m'] for m in h_modes]

    x = np.arange(len(h_modes))
    ax.bar(x - width/2, hd60_02, width, label='Vta02 60s Drift', color='#d95f02', alpha=0.85)
    ax.bar(x + width/2, hd60_04, width, label='Vta04 60s Drift', color='#7570b3', alpha=0.85)
    ax.plot(x, hd30_02, 'o--', color='#2b5c8f', label='Vta02 30s Drift')
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Mean Position Drift (m)")
    ax.set_title("Sub-Audit E: Huber Variance Scaling Sensitivity", fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 6: Sub-Audit F: Filter Consistency & Overconfidence
    ax = axes[1, 2]
    horiz_labels = ['10s', '20s', '30s', '60s']
    act_err_02 = [audit_f['Vta02_suburban'][h]['mean_actual_error_m'] for h in horiz_labels]
    sigma3_02 = [audit_f['Vta02_suburban'][h]['mean_3sigma_p_m'] for h in horiz_labels]
    overconf_02 = [audit_f['Vta02_suburban'][h]['nees_statistics']['overconfidence_factor'] for h in horiz_labels]

    x = np.arange(len(horiz_labels))
    ax.plot(x, act_err_02, 'o-', color='#d73027', lw=2, label='Actual Error (Vta02)')
    ax.plot(x, sigma3_02, 's--', color='#4575b4', lw=2, label='Filter 3*sigma Envelope')
    ax.set_xticks(x)
    ax.set_xticklabels(horiz_labels)
    ax.set_ylabel("Position (m)")
    ax.set_title("Sub-Audit F: Actual Drift vs Filter 3-sigma Envelope", fontweight='bold')

    # Annotate overconfidence factor on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x, overconf_02, 'd:', color='#762a83', lw=2, label='Overconfidence Factor')
    ax2.set_ylabel("Overconfidence Factor (x)", color='#762a83')
    ax2.tick_params(axis='y', labelcolor='#762a83')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[C8-5.1] Diagnostic figure saved to: {out_path}")


# =============================================================================
# 9. MASTER ORCHESTRATION
# =============================================================================

def main():
    json_path = RES_DIR / "c8_5_1_wheel_speed_integrity.json"
    fig_path = FIG_DIR / "c8_5_1_wheel_speed_integrity.png"

    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only" and json_path.exists():
        print(f"[C8-5.1] Loading existing JSON from {json_path} for plotting...")
        with open(json_path, 'r') as f:
            res = json.load(f)
        generate_diagnostic_figure(
            res['sub_audit_a_accuracy']['Vta02_suburban'],
            res['sub_audit_a_accuracy']['Vta04_highway'],
            res['sub_audit_b_slip_and_differentials']['Vta02_suburban'],
            res['sub_audit_b_slip_and_differentials']['Vta04_highway'],
            res['sub_audit_c_tire_radius_sensitivity'],
            res['sub_audit_d_covariance_sensitivity'],
            res['sub_audit_e_huber_sensitivity'],
            res['sub_audit_f_filter_consistency'],
            fig_path
        )
        return

    print("=" * 80)
    print("SIH26168 - Stage C8-5.1: Wheel-Speed Integrity & Observability Audit")
    print("=" * 80)
    t0 = time.time()

    print("[1/6] Ingesting telemetry and IMU data for Vta02 and Vta04...")
    data_vta02 = prepare_trip_data_c8_5_1("Vta02", dt=0.1)
    data_vta04 = prepare_trip_data_c8_5_1("Vta04", dt=0.1)

    print("[2/6] Executing Sub-Audit A: Wheel-Speed Accuracy vs VBOX Ground Truth...")
    audit_a_02 = audit_wheel_speed_accuracy(data_vta02, "Vta02_suburban")
    audit_a_04 = audit_wheel_speed_accuracy(data_vta04, "Vta04_highway")

    print("[3/6] Executing Sub-Audit B: Wheel Slip & Rear Differential across Regimes...")
    audit_b_02 = audit_wheel_slip_and_differentials(data_vta02, "Vta02_suburban")
    audit_b_04 = audit_wheel_slip_and_differentials(data_vta04, "Vta04_highway")

    print("[4/6] Executing Sub-Audit C: Tire-Radius Sensitivity Sweep (r_w +/- 2%)...")
    audit_c = audit_tire_radius_sensitivity(data_vta02, data_vta04)

    print("[5/6] Executing Sub-Audit D: Measurement Noise Covariance Sensitivity...")
    audit_d = audit_covariance_sensitivity(data_vta02, data_vta04)

    print("[6/6] Executing Sub-Audit E (Huber Scaling) and Sub-Audit F (Filter Consistency)...")
    audit_e = audit_huber_scaling_sensitivity(data_vta02, data_vta04)
    audit_f = audit_filter_consistency(data_vta02, data_vta04)

    # Master consolidation
    master_results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'stage': 'C8-5.1',
            'trips_evaluated': ['Vta02', 'Vta04'],
            'total_windows': len(data_vta02['windows'].get(30, [])) + len(data_vta04['windows'].get(30, [])),
            'execution_time_sec': float(time.time() - t0)
        },
        'sub_audit_a_accuracy': {
            'Vta02_suburban': audit_a_02,
            'Vta04_highway': audit_a_04
        },
        'sub_audit_b_slip_and_differentials': {
            'Vta02_suburban': audit_b_02,
            'Vta04_highway': audit_b_04
        },
        'sub_audit_c_tire_radius_sensitivity': audit_c,
        'sub_audit_d_covariance_sensitivity': audit_d,
        'sub_audit_e_huber_sensitivity': audit_e,
        'sub_audit_f_filter_consistency': audit_f
    }

    # Save JSON
    json_path = RES_DIR / "c8_5_1_wheel_speed_integrity.json"
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"[C8-5.1] Master results saved to: {json_path}")

    # Generate Figure
    fig_path = FIG_DIR / "c8_5_1_wheel_speed_integrity.png"
    generate_diagnostic_figure(
        audit_a_02, audit_a_04,
        audit_b_02, audit_b_04,
        audit_c, audit_d, audit_e, audit_f,
        fig_path
    )

    print("=" * 80)
    print(f"Stage C8-5.1 Audit completed in {time.time() - t0:.2f} seconds.")
    print("=" * 80)


if __name__ == '__main__':
    main()
