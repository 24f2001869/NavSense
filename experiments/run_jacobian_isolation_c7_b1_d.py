"""
SIH26168 - Stage C7-B1-D: Diagnostic Jacobian-Isolation Experiment (H_theta = 0)
Module: experiments/run_jacobian_isolation_c7_b1_d.py

Evaluates the causal effect of isolating the attitude block in the acceleration
pseudo-measurement Jacobian H:
    Setting H_theta = [0, 0, 0] vs coupled H_theta = [0, g_z^v, -g_y^v].

Experimental Conditions:
1. C0 (Frozen Baseline): ESKF + NHC (0.5 m/s) + BCAC (w_b=10s) + Deployable ZUPT.
2. B1_3_Coupled: Nominal C7-B1 soft acceleration constraint (a_soft=3.0 m/s^2, H_theta coupled).
3. B1_3_Decoupled_Htheta: Soft acceleration constraint with H_theta = 0_1x3.
4. B1_3_Strict_Freeze: Soft acceleration constraint with H_theta = 0_1x3 AND K_theta = 0_3x1.

Frozen Configuration:
- Trips: Vta02 (54 windows) and Vta04 (9 windows).
- Horizons: 10s, 20s, 30s, 60s.
- Measurement noise: sigma_a_meas = 0.50 m/s^2, a_soft = 3.0 m/s^2.
- No other changes: zero tuning, zero new thresholds, no CAN speed, no lateral constraint.

Telemetry Recorded per Window:
- Total drift, along-track error, cross-track error, velocity error.
- Final heading error, net attitude change (yaw, pitch, roll).
- Cumulative absolute attitude corrections (sum |delta_psi|, sum |delta_theta|, sum |delta_phi|).
- Accelerometer bias correction (cumulative absolute and net change).
- Acceleration innovation and activation percentage.
- Kalman gain norm by state block (Kp, Kv, Ktheta, Kba, Kbg).
- NHC innovation and correction.
"""

import sys
import time
import json
from pathlib import Path
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

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def compute_bcac_series(acc_v, dt=0.1, w_b_s=10.0, sigma_c0=0.291):
    """Computes frozen BCAC process noise series sigma_a(k)."""
    n = len(acc_v)
    w_k = 10
    wb_epochs = int(round(w_b_s / dt))

    jerk_rms = np.zeros(n)
    for i in range(n):
        i_start = max(0, i - w_k + 1)
        win_acc = acc_v[i_start : i + 1]
        if len(win_acc) > 1:
            jerk = np.diff(win_acc, axis=0) / dt
            jerk_rms[i] = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        else:
            jerk_rms[i] = 0.0

    b = np.zeros(n)
    for i in range(n):
        i_start = max(0, i - wb_epochs + 1)
        b[i] = np.median(jerk_rms[i_start : i + 1])

    j_norm = jerk_rms / (b + 1e-4)
    mult = np.clip(np.sqrt(j_norm), 0.75, 1.35)
    return sigma_c0 * mult


def prepare_trip_data(trip_name, dt=0.1):
    """Loads, standardizes, and computes reference quantities for a trip."""
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    acc_v, gyro_v, R_vp, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124], dtype=np.float64)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    a_vbox_diff = np.gradient(speed, dt)
    can_acc = df_v['veh_accel_long_g'].values * 9.80665 if 'veh_accel_long_g' in df_v.columns else a_vbox_diff
    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    return {
        'trip_name': trip_name,
        'acc_v': acc_v,
        'gyro_v': gyro_v,
        'speed': speed,
        'heading': heading,
        'gt_e': gt_e,
        'gt_n': gt_n,
        'gt_u': gt_u,
        'gt_ve': gt_ve,
        'gt_vn': gt_vn,
        'gt_vu': gt_vu,
        'a_vbox_diff': a_vbox_diff,
        'can_acc': can_acc,
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'n_epochs': n
    }


def run_single_window_isolation(
    kw, w_dur, dt,
    trip_data,
    condition='B1_3_Coupled',
    record_timeseries=False
):
    """
    Executes a single outage window with complete state correction telemetry.
    """
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    heading = trip_data['heading']
    gt_e = trip_data['gt_e']
    gt_n = trip_data['gt_n']
    gt_u = trip_data['gt_u']
    gt_ve = trip_data['gt_ve']
    gt_vn = trip_data['gt_vn']
    gt_vu = trip_data['gt_vu']
    ba_stat = trip_data['ba_stat']
    sigma_series = trip_data['sigma_series']

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

    nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)
    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )

    # Instantiate Constraint according to condition
    if condition == 'C0':
        constraint = None
    elif condition == 'B1_3_Coupled':
        constraint = SoftLongitudinalAccelerationConstraint(
            a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
            decouple_attitude=False, strict_attitude_freeze=False
        )
    elif condition == 'B1_3_Decoupled_Htheta':
        constraint = SoftLongitudinalAccelerationConstraint(
            a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
            decouple_attitude=True, strict_attitude_freeze=False
        )
    elif condition == 'B1_3_Strict_Freeze':
        constraint = SoftLongitudinalAccelerationConstraint(
            a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
            decouple_attitude=True, strict_attitude_freeze=True
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Telemetry accumulators
    act_count = 0
    cum_abs_roll_corr = 0.0
    cum_abs_pitch_corr = 0.0
    cum_abs_yaw_corr = 0.0
    cum_abs_ba_corr = 0.0

    sum_Kp_norm = 0.0
    sum_Kv_norm = 0.0
    sum_Ktheta_norm = 0.0
    sum_Kba_norm = 0.0
    sum_Kbg_norm = 0.0

    acc_innovations = []
    nhc_innovations_lat = []
    nhc_innovations_vert = []
    cum_abs_nhc_vel_corr = 0.0

    init_st = eskf.get_state()
    init_roll = init_st['roll_deg']
    init_pitch = init_st['pitch_deg']
    init_yaw = init_st['yaw_deg']
    init_ba = eskf.ba.copy()

    # Optional timeseries recording
    ts_time = []
    ts_long_err = []
    ts_lat_err = []
    ts_vel_err = []
    ts_heading_err = []
    ts_cum_yaw_corr = []
    ts_cum_pitch_corr = []
    ts_nhc_lat_innov = []
    ts_acc_innov = []

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]

        ax = acc_v[step_k, 0]
        ay = acc_v[step_k, 1]
        az = acc_v[step_k, 2]
        gx = gyro_v[step_k, 0]
        gy = gyro_v[step_k, 1]
        gz = gyro_v[step_k, 2]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 1. NHC update
        nhc_res = nhc.update_eskf(eskf)
        nhc_innovations_lat.append(abs(float(nhc_res['innovation'][0])))
        nhc_innovations_vert.append(abs(float(nhc_res['innovation'][1])))
        cum_abs_nhc_vel_corr += float(np.linalg.norm(nhc_res['delta_x'][3:6]))

        # 2. Deployable ZUPT update
        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        # 3. Acceleration Constraint update
        a_innov_val = 0.0
        if constraint is not None:
            c_res = constraint.update_eskf(eskf, dt=dt)
            if c_res['active']:
                act_count += 1
                a_innov_val = float(c_res['innovation'])
                acc_innovations.append(abs(a_innov_val))

                d_att = c_res['att_corr']  # [dtheta_x, dtheta_y, dtheta_z] in rad
                cum_abs_roll_corr += float(abs(d_att[0])) * (180.0 / np.pi)
                cum_abs_pitch_corr += float(abs(d_att[1])) * (180.0 / np.pi)
                cum_abs_yaw_corr += float(abs(d_att[2])) * (180.0 / np.pi)

                d_ba = c_res['ba_corr']
                cum_abs_ba_corr += float(np.linalg.norm(d_ba))

                K = c_res['K']
                sum_Kp_norm += float(np.linalg.norm(K[0:3]))
                sum_Kv_norm += float(np.linalg.norm(K[3:6]))
                sum_Ktheta_norm += float(np.linalg.norm(K[6:9]))
                sum_Kba_norm += float(np.linalg.norm(K[9:12]))
                sum_Kbg_norm += float(np.linalg.norm(K[12:15]))

        if record_timeseries:
            t_rel = (step_k - kw + 1) * dt
            st = eskf.get_state()
            p_curr = st['pos_n'][:2]
            v_curr = st['vel_n'][:2]
            gt_p_curr = np.array([gt_e[step_k + 1], gt_n[step_k + 1]])
            gt_v_curr = np.array([gt_ve[step_k + 1], gt_vn[step_k + 1]])
            curr_h_deg = float(heading[step_k + 1])
            curr_h_rad = np.radians(curr_h_deg)

            e_vec = p_curr - gt_p_curr
            u_fwd = np.array([np.sin(curr_h_rad), np.cos(curr_h_rad)])
            u_lat = np.array([-np.cos(curr_h_rad), np.sin(curr_h_rad)])

            est_yaw = st['yaw_deg']
            h_err = (est_yaw - curr_h_deg + 180.0) % 360.0 - 180.0

            ts_time.append(t_rel)
            ts_long_err.append(float(np.dot(e_vec, u_fwd)))
            ts_lat_err.append(float(np.dot(e_vec, u_lat)))
            ts_vel_err.append(float(np.linalg.norm(v_curr - gt_v_curr)))
            ts_heading_err.append(abs(float(h_err)))
            ts_cum_yaw_corr.append(cum_abs_yaw_corr)
            ts_cum_pitch_corr.append(cum_abs_pitch_corr)
            ts_nhc_lat_innov.append(abs(float(nhc_res['innovation'][0])))
            ts_acc_innov.append(abs(a_innov_val))

    # Final state metrics
    st_end = eskf.get_state()
    gt_p_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur]])
    gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur]])
    end_h_deg = float(heading[kw + w_dur])
    end_h_rad = np.radians(end_h_deg)

    e_vec_end = st_end['pos_n'][:2] - gt_p_end
    u_fwd_end = np.array([np.sin(end_h_rad), np.cos(end_h_rad)])
    u_lat_end = np.array([-np.cos(end_h_rad), np.sin(end_h_rad)])

    final_yaw = st_end['yaw_deg']
    final_pitch = st_end['pitch_deg']
    final_roll = st_end['roll_deg']
    final_h_err = abs(float((final_yaw - end_h_deg + 180.0) % 360.0 - 180.0))

    # Net attitude change over outage window
    net_roll_change = abs(float((final_roll - init_roll + 180.0) % 360.0 - 180.0))
    net_pitch_change = abs(float((final_pitch - init_pitch + 180.0) % 360.0 - 180.0))
    net_yaw_change = abs(float((final_yaw - init_yaw + 180.0) % 360.0 - 180.0))

    # Net accelerometer bias change
    net_ba_change = float(np.linalg.norm(eskf.ba - init_ba))

    mean_Kp = (sum_Kp_norm / act_count) if act_count > 0 else 0.0
    mean_Kv = (sum_Kv_norm / act_count) if act_count > 0 else 0.0
    mean_Ktheta = (sum_Ktheta_norm / act_count) if act_count > 0 else 0.0
    mean_Kba = (sum_Kba_norm / act_count) if act_count > 0 else 0.0
    mean_Kbg = (sum_Kbg_norm / act_count) if act_count > 0 else 0.0

    res = {
        'total_drift_m': float(np.linalg.norm(e_vec_end)),
        'along_track_m': float(abs(np.dot(e_vec_end, u_fwd_end))),
        'cross_track_m': float(abs(np.dot(e_vec_end, u_lat_end))),
        'along_track_signed_m': float(np.dot(e_vec_end, u_fwd_end)),
        'cross_track_signed_m': float(np.dot(e_vec_end, u_lat_end)),
        'vel_err_ms': float(np.linalg.norm(st_end['vel_n'][:2] - gt_v_end)),
        'final_heading_err_deg': final_h_err,
        'net_yaw_change_deg': net_yaw_change,
        'net_pitch_change_deg': net_pitch_change,
        'net_roll_change_deg': net_roll_change,
        'cum_abs_yaw_corr_deg': cum_abs_yaw_corr,
        'cum_abs_pitch_corr_deg': cum_abs_pitch_corr,
        'cum_abs_roll_corr_deg': cum_abs_roll_corr,
        'cum_abs_ba_corr': cum_abs_ba_corr,
        'net_ba_change': net_ba_change,
        'act_count': act_count,
        'act_pct': float(act_count / w_dur) * 100.0,
        'mean_acc_innov': float(np.mean(acc_innovations)) if acc_innovations else 0.0,
        'mean_Kp': mean_Kp,
        'mean_Kv': mean_Kv,
        'mean_Ktheta': mean_Ktheta,
        'mean_Kba': mean_Kba,
        'mean_Kbg': mean_Kbg,
        'mean_nhc_innov_lat': float(np.mean(nhc_innovations_lat)) if nhc_innovations_lat else 0.0,
        'mean_nhc_innov_vert': float(np.mean(nhc_innovations_vert)) if nhc_innovations_vert else 0.0,
        'cum_abs_nhc_vel_corr': cum_abs_nhc_vel_corr
    }

    if record_timeseries:
        res['timeseries'] = {
            'time': ts_time,
            'long_err': ts_long_err,
            'lat_err': ts_lat_err,
            'vel_err': ts_vel_err,
            'heading_err': ts_heading_err,
            'cum_yaw_corr': ts_cum_yaw_corr,
            'cum_pitch_corr': ts_cum_pitch_corr,
            'nhc_lat_innov': ts_nhc_lat_innov,
            'acc_innov': ts_acc_innov
        }

    return res


def evaluate_trip_isolation(trip_data, conditions, horizons=[10.0, 20.0, 30.0, 60.0], stride_s=20.0, dt=0.1):
    """
    Evaluates all isolation conditions across all horizons for a given trip.
    """
    stride_k = int(round(stride_s / dt))
    results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, trip_data['n_epochs'] - w_dur - 1, stride_k))
        n_win = len(window_starts)

        results[h_key] = {
            'n_windows': n_win,
            'conditions': {}
        }

        for c in conditions:
            win_outputs = []
            for kw in window_starts:
                out = run_single_window_isolation(kw, w_dur, dt, trip_data, condition=c, record_timeseries=False)
                win_outputs.append(out)

            # Aggregate statistics
            drifts = [w['total_drift_m'] for w in win_outputs]
            long_errs = [w['along_track_m'] for w in win_outputs]
            lat_errs = [w['cross_track_m'] for w in win_outputs]
            vel_errs = [w['vel_err_ms'] for w in win_outputs]
            h_errs = [w['final_heading_err_deg'] for w in win_outputs]

            net_yaw = [w['net_yaw_change_deg'] for w in win_outputs]
            net_pitch = [w['net_pitch_change_deg'] for w in win_outputs]
            cum_yaw = [w['cum_abs_yaw_corr_deg'] for w in win_outputs]
            cum_pitch = [w['cum_abs_pitch_corr_deg'] for w in win_outputs]

            cum_ba = [w['cum_abs_ba_corr'] for w in win_outputs]
            net_ba = [w['net_ba_change'] for w in win_outputs]
            act_pct = [w['act_pct'] for w in win_outputs]
            acc_innov = [w['mean_acc_innov'] for w in win_outputs if w['mean_acc_innov'] > 0]

            Kp = [w['mean_Kp'] for w in win_outputs]
            Kv = [w['mean_Kv'] for w in win_outputs]
            Ktheta = [w['mean_Ktheta'] for w in win_outputs]
            Kba = [w['mean_Kba'] for w in win_outputs]
            Kbg = [w['mean_Kbg'] for w in win_outputs]

            nhc_lat = [w['mean_nhc_innov_lat'] for w in win_outputs]
            nhc_corr = [w['cum_abs_nhc_vel_corr'] for w in win_outputs]

            results[h_key]['conditions'][c] = {
                'drift_mean': float(np.mean(drifts)),
                'drift_median': float(np.median(drifts)),
                'drift_p90': float(np.percentile(drifts, 90)),
                'along_track_mean': float(np.mean(long_errs)),
                'cross_track_mean': float(np.mean(lat_errs)),
                'vel_err_mean': float(np.mean(vel_errs)),
                'final_heading_err_mean': float(np.mean(h_errs)),
                'net_yaw_change_mean': float(np.mean(net_yaw)),
                'net_pitch_change_mean': float(np.mean(net_pitch)),
                'cum_abs_yaw_corr_mean': float(np.mean(cum_yaw)),
                'cum_abs_pitch_corr_mean': float(np.mean(cum_pitch)),
                'cum_abs_ba_corr_mean': float(np.mean(cum_ba)),
                'net_ba_change_mean': float(np.mean(net_ba)),
                'act_pct_mean': float(np.mean(act_pct)),
                'mean_acc_innov': float(np.mean(acc_innov)) if acc_innov else 0.0,
                'mean_Kp': float(np.mean(Kp)),
                'mean_Kv': float(np.mean(Kv)),
                'mean_Ktheta': float(np.mean(Ktheta)),
                'mean_Kba': float(np.mean(Kba)),
                'mean_Kbg': float(np.mean(Kbg)),
                'mean_nhc_innov_lat': float(np.mean(nhc_lat)),
                'cum_abs_nhc_vel_corr_mean': float(np.mean(nhc_corr)),
                'windows': win_outputs
            }

    return results


def run_forensic_windows(trip_data, conditions, dt=0.1):
    """
    Runs forensic window timeseries logging on representative Vta04 windows:
    - 30s: Window 0, Window 2, Window 4
    - 60s: Window 4
    """
    forensics = {}

    # 30-s windows (kw = 50, 50+200*2=450, 50+200*4=850)
    w_dur_30 = int(round(30.0 / dt))
    win_targets_30 = [50, 450, 850]
    win_names_30 = ['w0_30s', 'w2_30s', 'w4_30s']

    for name, kw in zip(win_names_30, win_targets_30):
        forensics[name] = {}
        for c in conditions:
            forensics[name][c] = run_single_window_isolation(
                kw, w_dur_30, dt, trip_data, condition=c, record_timeseries=True
            )

    # 60-s window (kw = 850)
    w_dur_60 = int(round(60.0 / dt))
    forensics['w4_60s'] = {}
    for c in conditions:
        forensics['w4_60s'][c] = run_single_window_isolation(
            850, w_dur_60, dt, trip_data, condition=c, record_timeseries=True
        )

    return forensics


def plot_isolation_summary(all_results):
    """
    Generates high-impact diagnostic figures comparing C0, B1_3 Coupled,
    B1_3 Decoupled-Htheta, and B1_3 Strict-Freeze.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    horizons = ['10s', '20s', '30s', '60s']
    conds = ['C0', 'B1_3_Coupled', 'B1_3_Decoupled_Htheta', 'B1_3_Strict_Freeze']
    colors = {
        'C0': '#7f8c8d',
        'B1_3_Coupled': '#e74c3c',
        'B1_3_Decoupled_Htheta': '#2980b9',
        'B1_3_Strict_Freeze': '#27ae60'
    }
    labels = {
        'C0': 'C0 Baseline',
        'B1_3_Coupled': 'B1_3 (Coupled H_θ)',
        'B1_3_Decoupled_Htheta': 'B1_3 (Decoupled H_θ=0)',
        'B1_3_Strict_Freeze': 'B1_3 (Strict Freeze K_θ=0)'
    }

    # Row 0: Vta04 (Highway) - Total Drift, Along-Track, Cross-Track
    # 1. Total Drift
    ax = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.20
    for i, c in enumerate(conds):
        vals = [all_results['Vta04'][h]['conditions'][c]['drift_mean'] for h in horizons]
        ax.bar(x + (i - 1.5) * width, vals, width, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Drift Mean (m)', fontsize=11, fontweight='bold')
    ax.set_title('Vta04 Highway: Total Drift across Horizons', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 2. Along-Track Error
    ax = axes[0, 1]
    for i, c in enumerate(conds):
        vals = [all_results['Vta04'][h]['conditions'][c]['along_track_mean'] for h in horizons]
        ax.bar(x + (i - 1.5) * width, vals, width, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Along-Track Error (m)', fontsize=11, fontweight='bold')
    ax.set_title('Vta04 Highway: Along-Track Error (Longitudinal)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 3. Cross-Track Error
    ax = axes[0, 2]
    for i, c in enumerate(conds):
        vals = [all_results['Vta04'][h]['conditions'][c]['cross_track_mean'] for h in horizons]
        ax.bar(x + (i - 1.5) * width, vals, width, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Cross-Track Error (m)', fontsize=11, fontweight='bold')
    ax.set_title('Vta04 Highway: Cross-Track Error (Lateral Explosion Test)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Row 1: Vta02 (Suburban) and Key Diagnostics
    # 4. Vta02 Total Drift
    ax = axes[1, 0]
    for i, c in enumerate(conds):
        vals = [all_results['Vta02'][h]['conditions'][c]['drift_mean'] for h in horizons]
        ax.bar(x + (i - 1.5) * width, vals, width, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Drift Mean (m)', fontsize=11, fontweight='bold')
    ax.set_title('Vta02 Suburban: Total Drift across Horizons', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. Cumulative Absolute Yaw Correction (Vta04)
    ax = axes[1, 1]
    for i, c in enumerate(conds[1:], 1):  # C0 has 0 constraint corrections
        vals = [all_results['Vta04'][h]['conditions'][c]['cum_abs_yaw_corr_mean'] for h in horizons]
        ax.bar(x + (i - 2) * (width * 1.3), vals, width * 1.3, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative |δψ| (deg)', fontsize=11, fontweight='bold')
    ax.set_title('Vta04: Cumulative Absolute Yaw Correction', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 6. Kalman Gain Block Norm ||K_theta|| (Vta04)
    ax = axes[1, 2]
    for i, c in enumerate(conds[1:], 1):
        vals = [all_results['Vta04'][h]['conditions'][c]['mean_Ktheta'] for h in horizons]
        ax.bar(x + (i - 2) * (width * 1.3), vals, width * 1.3, label=labels[c], color=colors[c], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Attitude Kalman Gain ||K_θ||', fontsize=11, fontweight='bold')
    ax.set_title('Vta04: Attitude Kalman Gain Magnitude', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = FIG_DIR / "c7_b1_d_jacobian_isolation_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[C7-B1-D] Saved comparison figure to: {fig_path}")


def plot_forensic_timeseries(forensics):
    """
    Plots epoch-by-epoch timeseries for Vta04 Window 2 (30s) and Window 4 (60s)
    to visually demonstrate trajectory and state behavior under Jacobian isolation.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = {
        'C0': '#7f8c8d',
        'B1_3_Coupled': '#e74c3c',
        'B1_3_Decoupled_Htheta': '#2980b9',
        'B1_3_Strict_Freeze': '#27ae60'
    }
    labels = {
        'C0': 'C0 Baseline',
        'B1_3_Coupled': 'B1_3 Coupled H_θ',
        'B1_3_Decoupled_Htheta': 'B1_3 Decoupled H_θ=0',
        'B1_3_Strict_Freeze': 'B1_3 Strict Freeze K_θ=0'
    }

    # Left Column: Window 2 (30s)
    w2 = forensics['w2_30s']
    for c in ['C0', 'B1_3_Coupled', 'B1_3_Decoupled_Htheta', 'B1_3_Strict_Freeze']:
        ts = w2[c]['timeseries']
        axes[0, 0].plot(ts['time'], ts['long_err'], label=labels[c], color=colors[c], lw=2.0)
        axes[1, 0].plot(ts['time'], ts['lat_err'], label=labels[c], color=colors[c], lw=2.0)
        axes[2, 0].plot(ts['time'], ts['cum_yaw_corr'], label=labels[c], color=colors[c], lw=2.0)

    axes[0, 0].set_title('Window 2 (30s): Along-Track Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Along-Track Error (m)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(fontsize=8)

    axes[1, 0].set_title('Window 2 (30s): Cross-Track Error', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Cross-Track Error (m)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    axes[2, 0].set_title('Window 2 (30s): Cumulative Yaw Correction', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel('Cumulative |δψ| (deg)')
    axes[2, 0].set_xlabel('Outage Time (s)')
    axes[2, 0].grid(True, linestyle='--', alpha=0.5)

    # Right Column: Window 4 (60s)
    w4 = forensics['w4_60s']
    for c in ['C0', 'B1_3_Coupled', 'B1_3_Decoupled_Htheta', 'B1_3_Strict_Freeze']:
        ts = w4[c]['timeseries']
        axes[0, 1].plot(ts['time'], ts['long_err'], label=labels[c], color=colors[c], lw=2.0)
        axes[1, 1].plot(ts['time'], ts['lat_err'], label=labels[c], color=colors[c], lw=2.0)
        axes[2, 1].plot(ts['time'], ts['cum_yaw_corr'], label=labels[c], color=colors[c], lw=2.0)

    axes[0, 1].set_title('Window 4 (60s): Along-Track Error', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Along-Track Error (m)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 1].set_title('Window 4 (60s): Cross-Track Error (Highway Blowup Test)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Cross-Track Error (m)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    axes[2, 1].set_title('Window 4 (60s): Cumulative Yaw Correction', fontsize=11, fontweight='bold')
    axes[2, 1].set_ylabel('Cumulative |δψ| (deg)')
    axes[2, 1].set_xlabel('Outage Time (s)')
    axes[2, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = FIG_DIR / "c7_b1_d_timeseries_forensic.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[C7-B1-D] Saved forensic timeseries figure to: {fig_path}")


def main():
    print("=" * 80)
    print("SIH26168 - Stage C7-B1-D: Diagnostic Jacobian-Isolation Experiment")
    print("Testing Hypothesis: Does H_theta = 0 preserve longitudinal drift reduction")
    print("while eliminating cross-track lateral error expansion?")
    print("=" * 80)

    t0 = time.time()
    dt = 0.1
    conditions = ['C0', 'B1_3_Coupled', 'B1_3_Decoupled_Htheta', 'B1_3_Strict_Freeze']
    horizons = [10.0, 20.0, 30.0, 60.0]

    all_results = {}

    for trip_name in ['Vta02', 'Vta04']:
        print(f"\n[C7-B1-D] Loading trip: {trip_name}...")
        trip_data = prepare_trip_data(trip_name, dt=dt)
        print(f"Loaded {trip_data['n_epochs']} epochs.")

        print(f"[C7-B1-D] Running multi-horizon evaluation across {conditions}...")
        trip_eval = evaluate_trip_isolation(trip_data, conditions, horizons=horizons, stride_s=20.0, dt=dt)
        all_results[trip_name] = trip_eval

        # Quick summary display for 30s
        print(f"\n--- {trip_name} 30-s Summary ---")
        h30 = trip_eval['30s']['conditions']
        print(f"{'Condition':<25} | {'Drift (m)':<10} | {'Along (m)':<10} | {'Cross (m)':<10} | {'Vel Err':<8} | {'Head Err':<8} | {'Cum |dpsi|':<10} | {'||K_theta||':<10}")
        print("-" * 105)
        for c in conditions:
            d = h30[c]
            print(f"{c:<25} | {d['drift_mean']:<10.2f} | {d['along_track_mean']:<10.2f} | {d['cross_track_mean']:<10.2f} | {d['vel_err_mean']:<8.2f} | {d['final_heading_err_mean']:<8.2f} | {d['cum_abs_yaw_corr_mean']:<10.2f} | {d['mean_Ktheta']:<10.3e}")

    # Forensic timeseries on Vta04
    print("\n[C7-B1-D] Running forensic timeseries on representative Vta04 windows...")
    trip_data_04 = prepare_trip_data('Vta04', dt=dt)
    forensics = run_forensic_windows(trip_data_04, conditions, dt=dt)

    # Plot figures
    print("\n[C7-B1-D] Generating publication figures...")
    plot_isolation_summary(all_results)
    plot_forensic_timeseries(forensics)

    # Save JSON results (stripping timeseries from main json to keep file clean)
    clean_results = {}
    for t_name in ['Vta02', 'Vta04']:
        clean_results[t_name] = {}
        for h_key in ['10s', '20s', '30s', '60s']:
            clean_results[t_name][h_key] = {
                'n_windows': all_results[t_name][h_key]['n_windows'],
                'conditions': {}
            }
            for c in conditions:
                cond_dict = dict(all_results[t_name][h_key]['conditions'][c])
                # remove full window array for summary or keep compact summary
                summary_keys = [
                    'drift_mean', 'drift_median', 'drift_p90',
                    'along_track_mean', 'cross_track_mean', 'vel_err_mean',
                    'final_heading_err_mean', 'net_yaw_change_mean', 'net_pitch_change_mean',
                    'cum_abs_yaw_corr_mean', 'cum_abs_pitch_corr_mean',
                    'cum_abs_ba_corr_mean', 'net_ba_change_mean',
                    'act_pct_mean', 'mean_acc_innov',
                    'mean_Kp', 'mean_Kv', 'mean_Ktheta', 'mean_Kba', 'mean_Kbg',
                    'mean_nhc_innov_lat', 'cum_abs_nhc_vel_corr_mean'
                ]
                clean_results[t_name][h_key]['conditions'][c] = {
                    k: cond_dict[k] for k in summary_keys
                }

    out_json = RES_DIR / "c7_b1_d_jacobian_isolation.json"
    with open(out_json, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"[C7-B1-D] Master JSON saved to: {out_json}")

    elapsed = time.time() - t0
    print(f"\n[C7-B1-D] Completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
