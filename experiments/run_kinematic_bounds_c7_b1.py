"""
SIH26168 - Stage C7-B1: Soft Longitudinal Acceleration-Consistency Constraint Evaluation
Module: experiments/run_kinematic_bounds_c7_b1.py

Evaluates the effect of soft longitudinal acceleration-consistency constraints in the
15-state ESKF to target continuous-motion dead-reckoning drift.

Conditions Evaluated:
1. C0 (Frozen Baseline): ESKF + NHC (0.5 m/s) + BCAC (w_b=10s) + Deployable ZUPT (0.8s persist).
2. B1 (Soft Accel Constraint, a_soft=4.0 m/s^2): C0 + soft longitudinal acceleration constraint.
3. B1-3.0 (Sensitivity, a_soft=3.0 m/s^2): C0 + tighter soft constraint.
4. B1-5.0 (Sensitivity, a_soft=5.0 m/s^2): C0 + looser soft constraint.
5. Oracle-B1: C0 + ground-truth reference acceleration update from differentiated VBOX speed.

Evaluated across:
- Horizons: 5s, 10s, 20s, 30s, 60s.
- Primary Diagnostic Trip: Vta02 (54 windows).
- Held-Out Confirmation Trip: Vta04 (9 windows, 100% continuous highway).
- Regimes: Steady Cruising, Acceleration, Severe Braking, Rough Road, Standstill.
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

    # Differentiated VBOX speed as reference longitudinal acceleration
    a_vbox_diff = np.gradient(speed, dt)

    # CAN chassis acceleration if available
    can_acc = df_v['veh_accel_long_g'].values * 9.80665 if 'veh_accel_long_g' in df_v.columns else a_vbox_diff

    # BCAC process noise series
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


def run_single_outage(
    kw, w_dur, dt,
    trip_data,
    condition='C0',  # 'C0', 'B1', 'B1_3', 'B1_5', 'Oracle_B1'
    record_trajectory=False
):
    """
    Executes a single outage simulation under the specified experimental condition.
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
    a_vbox_diff = trip_data['a_vbox_diff']
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

    # Initialize constraint if applicable
    if condition == 'B1':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=4.0, sigma_a_meas=0.50)
    elif condition == 'B1_3':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=3.0, sigma_a_meas=0.50)
    elif condition == 'B1_5':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=5.0, sigma_a_meas=0.50)
    elif condition == 'Oracle_B1':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=4.0, sigma_a_meas=0.10)
    else:
        constraint = None

    constraint_activations = 0
    constraint_innovations = []
    max_vel_attained = 0.0

    traj_time = []
    traj_pos_err = []
    traj_vel_err = []
    traj_long_err = []
    traj_lat_err = []
    traj_a_x_v = []
    traj_constraint_active = []
    traj_innov = []

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]

        ax = acc_v[step_k, 0]
        ay = acc_v[step_k, 1]
        az = acc_v[step_k, 2]
        gx = gyro_v[step_k, 0]
        gy = gyro_v[step_k, 1]
        gz = gyro_v[step_k, 2]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 1. NHC update (always active in frozen baseline)
        nhc.update_eskf(eskf)

        # 2. Deployable ZUPT update (active during detected standstill)
        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        # 3. Acceleration Constraint update
        c_active = False
        c_innov = 0.0
        if constraint is not None:
            if condition == 'Oracle_B1':
                res_c = constraint.update_eskf(eskf, dt=dt, oracle_a_ref=a_vbox_diff[step_k])
            else:
                res_c = constraint.update_eskf(eskf, dt=dt)

            if res_c['active']:
                constraint_activations += 1
                constraint_innovations.append(abs(float(res_c['innovation'])))
                c_active = True
                c_innov = float(res_c['innovation'])

        # Track velocity magnitude
        cur_vel_norm = float(np.linalg.norm(eskf.vel_n[:2]))
        if cur_vel_norm > max_vel_attained:
            max_vel_attained = cur_vel_norm

        if record_trajectory:
            t_curr = (step_k - kw + 1) * dt
            st = eskf.get_state()
            p_curr = st['pos_n'][:2]
            v_curr = st['vel_n'][:2]
            gt_p_curr = np.array([gt_e[step_k + 1], gt_n[step_k + 1]])
            gt_v_curr = np.array([gt_ve[step_k + 1], gt_vn[step_k + 1]])
            curr_h_rad = np.radians(heading[step_k + 1])

            e_vec = p_curr - gt_p_curr
            u_fwd = np.array([np.sin(curr_h_rad), np.cos(curr_h_rad)])
            u_lat = np.array([-np.cos(curr_h_rad), np.sin(curr_h_rad)])

            a_x_now, _, _ = constraint.compute_body_accel(eskf) if constraint else (0.0, None, None)

            traj_time.append(t_curr)
            traj_pos_err.append(float(np.linalg.norm(e_vec)))
            traj_vel_err.append(float(np.linalg.norm(v_curr - gt_v_curr)))
            traj_long_err.append(float(np.dot(e_vec, u_fwd)))
            traj_lat_err.append(float(np.dot(e_vec, u_lat)))
            traj_a_x_v.append(float(a_x_now))
            traj_constraint_active.append(c_active)
            traj_innov.append(c_innov)

    # End-of-window metrics
    st_end = eskf.get_state()
    gt_p_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur]])
    gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur]])
    end_h_rad = np.radians(heading[kw + w_dur])

    e_vec_end = st_end['pos_n'][:2] - gt_p_end
    u_fwd_end = np.array([np.sin(end_h_rad), np.cos(end_h_rad)])
    u_lat_end = np.array([-np.cos(end_h_rad), np.sin(end_h_rad)])

    out = {
        'final_drift_m': float(np.linalg.norm(e_vec_end)),
        'final_vel_err_ms': float(np.linalg.norm(st_end['vel_n'][:2] - gt_v_end)),
        'final_long_err_m': float(np.dot(e_vec_end, u_fwd_end)),
        'final_lat_err_m': float(np.dot(e_vec_end, u_lat_end)),
        'max_vel_attained': max_vel_attained,
        'constraint_activation_count': constraint_activations,
        'constraint_activation_pct': float(constraint_activations / w_dur) * 100.0,
        'mean_constraint_innov': float(np.mean(constraint_innovations)) if constraint_innovations else 0.0
    }

    if record_trajectory:
        out['traj'] = {
            'time': traj_time,
            'pos_err': traj_pos_err,
            'vel_err': traj_vel_err,
            'long_err': traj_long_err,
            'lat_err': traj_lat_err,
            'a_x_v': traj_a_x_v,
            'constraint_active': traj_constraint_active,
            'innov': traj_innov
        }

    return out


def evaluate_trip(trip_data, conditions, horizons=[5.0, 10.0, 20.0, 30.0, 60.0], stride_s=20.0, dt=0.1):
    """
    Runs multi-horizon, multi-condition evaluation across a trip.
    """
    stride_k = int(round(stride_s / dt))
    trip_results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, trip_data['n_epochs'] - w_dur - 1, stride_k))
        n_win = len(window_starts)

        trip_results[h_key] = {
            'n_windows': n_win,
            'conditions': {c: {} for c in conditions},
            'regimes': {c: {'cruising': [], 'braking': [], 'acceleration': [], 'rough_road': [], 'standstill': []} for c in conditions}
        }

        # Pre-tag regimes for each window
        regime_tags = []
        for kw in window_starts:
            win_spd = trip_data['speed'][kw : kw + w_dur + 1]
            win_acc = trip_data['can_acc'][kw : kw + w_dur + 1]
            win_vert = trip_data['acc_v'][kw : kw + w_dur + 1, 2]

            is_standstill = float(np.mean(win_spd)) < 0.15
            is_braking = float(np.min(win_acc)) < -1.5
            is_accel = (float(np.max(win_acc)) > 1.0) and (not is_braking)
            is_cruising = (not is_standstill) and (not is_braking) and (not is_accel)
            is_rough = float(np.var(win_vert)) > 0.514

            regime_tags.append({
                'standstill': is_standstill,
                'braking': is_braking,
                'acceleration': is_accel,
                'cruising': is_cruising,
                'rough_road': is_rough
            })

        for c in conditions:
            drifts = []
            long_errs = []
            lat_errs = []
            vel_errs = []
            max_vels = []
            act_pcts = []
            innovs = []

            for idx, kw in enumerate(window_starts):
                r = run_single_outage(kw, w_dur, dt, trip_data, condition=c, record_trajectory=False)
                drifts.append(r['final_drift_m'])
                long_errs.append(r['final_long_err_m'])
                lat_errs.append(r['final_lat_err_m'])
                vel_errs.append(r['final_vel_err_ms'])
                max_vels.append(r['max_vel_attained'])
                act_pcts.append(r['constraint_activation_pct'])
                if r['mean_constraint_innov'] > 0:
                    innovs.append(r['mean_constraint_innov'])

                # Tag regime
                rt = regime_tags[idx]
                if rt['standstill']: trip_results[h_key]['regimes'][c]['standstill'].append(r['final_drift_m'])
                if rt['braking']: trip_results[h_key]['regimes'][c]['braking'].append(r['final_drift_m'])
                if rt['acceleration']: trip_results[h_key]['regimes'][c]['acceleration'].append(r['final_drift_m'])
                if rt['cruising']: trip_results[h_key]['regimes'][c]['cruising'].append(r['final_drift_m'])
                if rt['rough_road']: trip_results[h_key]['regimes'][c]['rough_road'].append(r['final_drift_m'])

            trip_results[h_key]['conditions'][c] = {
                'drift_mean': float(np.mean(drifts)),
                'drift_median': float(np.median(drifts)),
                'drift_p90': float(np.percentile(drifts, 90)),
                'along_track_mean': float(np.mean(np.abs(long_errs))),
                'cross_track_mean': float(np.mean(np.abs(lat_errs))),
                'vel_err_mean': float(np.mean(vel_errs)),
                'max_vel_mean': float(np.mean(max_vels)),
                'activation_pct_mean': float(np.mean(act_pcts)),
                'mean_innov': float(np.mean(innovs)) if innovs else 0.0,
                'raw_drifts': drifts
            }

    return trip_results


def main():
    print("===============================================================================")
    print("STAGE C7-B1: SOFT LONGITUDINAL ACCELERATION-CONSISTENCY CONSTRAINT BENCHMARK")
    print("===============================================================================")
    start_total_time = time.time()
    dt = 0.1

    print("\n[1] Preparing Trip Datasets (Vta02 Primary & Vta04 Held-Out)...", flush=True)
    trip_v2 = prepare_trip_data("Vta02", dt=dt)
    trip_v4 = prepare_trip_data("Vta04", dt=dt)

    conditions = ['C0', 'B1', 'B1_3', 'B1_5', 'Oracle_B1']
    horizons = [5.0, 10.0, 20.0, 30.0, 60.0]

    # -------------------------------------------------------------
    # 2. RUN NAVIGATION BENCHMARK
    # -------------------------------------------------------------
    print("\n[2] Executing Multi-Horizon Benchmark on Vta02...", flush=True)
    t0 = time.time()
    res_v2 = evaluate_trip(trip_v2, conditions, horizons=horizons, stride_s=20.0, dt=dt)
    print(f"    Vta02 benchmark completed in {time.time() - t0:.1f}s", flush=True)

    print("\n[3] Executing Multi-Horizon Benchmark on Vta04...", flush=True)
    t0 = time.time()
    res_v4 = evaluate_trip(trip_v4, conditions, horizons=horizons, stride_s=20.0, dt=dt)
    print(f"    Vta04 benchmark completed in {time.time() - t0:.1f}s", flush=True)

    # -------------------------------------------------------------
    # 3. PRINT DETAILED SUMMARY TABLES
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print("STAGE C7-B1 RESULTS SUMMARY: 30-SECOND HORIZON")
    print("="*85)
    print(f"{'Condition':<12} | {'Vta02 Mean':<11} | {'Vta02 Med':<10} | {'Vta02 P90':<10} | {'Vta04 Mean':<11} | {'Vta04 Med':<10} | {'Vta04 P90':<10}")
    print("-" * 85)
    for c in conditions:
        m2 = res_v2['30s']['conditions'][c]
        m4 = res_v4['30s']['conditions'][c]
        print(f"{c:<12} | {m2['drift_mean']:<9.2f} m | {m2['drift_median']:<8.2f} m | {m2['drift_p90']:<8.2f} m | {m4['drift_mean']:<9.2f} m | {m4['drift_median']:<8.2f} m | {m4['drift_p90']:<8.2f} m")

    print("\n" + "="*85)
    print("LONGITUDINAL VS CROSS-TRACK BREAKDOWN (30s HORIZON)")
    print("="*85)
    print(f"{'Condition':<12} | {'Vta02 Along':<12} | {'Vta02 Cross':<12} | {'Vta04 Along':<12} | {'Vta04 Cross':<12} | {'Act %':<8}")
    print("-" * 85)
    for c in conditions:
        m2 = res_v2['30s']['conditions'][c]
        m4 = res_v4['30s']['conditions'][c]
        print(f"{c:<12} | {m2['along_track_mean']:<10.2f} m | {m2['cross_track_mean']:<10.2f} m | {m4['along_track_mean']:<10.2f} m | {m4['cross_track_mean']:<10.2f} m | {m2['activation_pct_mean']:<6.1f}%")

    # -------------------------------------------------------------
    # 4. FORENSIC TIME-SERIES EXTRACTION (Window on Vta04)
    # -------------------------------------------------------------
    print("\n[4] Extracting Forensic Outage Time-Series on Vta04 Window 4 (t=80-140s)...", flush=True)
    kw_forensic = 800  # 80 s
    w_dur_forensic = 600  # 60 s
    forensic_c0 = run_single_outage(kw_forensic, w_dur_forensic, dt, trip_v4, condition='C0', record_trajectory=True)
    forensic_b1 = run_single_outage(kw_forensic, w_dur_forensic, dt, trip_v4, condition='B1', record_trajectory=True)
    forensic_ora = run_single_outage(kw_forensic, w_dur_forensic, dt, trip_v4, condition='Oracle_B1', record_trajectory=True)

    # -------------------------------------------------------------
    # 5. GENERATE PUBLICATION FIGURES
    # -------------------------------------------------------------
    print("\n[5] Generating Publication Figures...", flush=True)

    # Figure 1: Empirical Acceleration Distributions (Audit C7-B0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Panel A: VBOX vs CAN acceleration distribution on Vta02
    ax = axes[0, 0]
    ax.hist(np.abs(trip_v2['a_vbox_diff']), bins=50, range=(0, 6), density=True, alpha=0.6, color='tab:blue', label='VBOX Differentiated Speed')
    ax.hist(np.abs(trip_v2['can_acc']), bins=50, range=(0, 6), density=True, alpha=0.6, color='tab:orange', label='CAN Chassis Accel')
    ax.axvline(3.0, color='gold', linestyle='--', linewidth=1.5, label='Soft 3.0 m/s²')
    ax.axvline(4.0, color='red', linestyle='--', linewidth=1.5, label='Nominal 4.0 m/s²')
    ax.set_title("Vta02 Longitudinal Accel |a_x| Distribution", fontsize=11, fontweight='bold')
    ax.set_xlabel("|a_x| (m/s²)")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Cumulative Distribution Function (CDF)
    ax = axes[0, 1]
    sorted_v2 = np.sort(np.abs(trip_v2['a_vbox_diff']))
    sorted_v4 = np.sort(np.abs(trip_v4['a_vbox_diff']))
    cdf_v2 = np.arange(len(sorted_v2)) / len(sorted_v2)
    cdf_v4 = np.arange(len(sorted_v4)) / len(sorted_v4)
    ax.plot(sorted_v2, cdf_v2, color='tab:blue', linewidth=2.0, label='Vta02 (Suburban/Stops)')
    ax.plot(sorted_v4, cdf_v4, color='tab:green', linewidth=2.0, label='Vta04 (Continuous Highway)')
    ax.axvline(4.0, color='red', linestyle='--', label='a_soft = 4.0 m/s²')
    ax.axhline(0.99, color='gray', linestyle=':', label='99th Percentile')
    ax.set_xlim(0, 7)
    ax.set_title("CDF of Longitudinal Acceleration |a_x|", fontsize=11, fontweight='bold')
    ax.set_xlabel("|a_x| (m/s²)")
    ax.set_ylabel("Cumulative Probability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Severe Braking Transients on Vta02
    ax = axes[1, 0]
    brake_mask = trip_v2['a_vbox_diff'] < -1.5
    ax.hist(trip_v2['a_vbox_diff'][brake_mask], bins=30, color='tab:red', alpha=0.7, edgecolor='black')
    ax.axvline(-4.0, color='black', linestyle='--', linewidth=2.0, label='a_soft Threshold (-4 m/s²)')
    ax.set_title("Vta02 Severe Braking Distribution (a_x < -1.5 m/s²)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Deceleration a_x (m/s²)")
    ax.set_ylabel("Epoch Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Planar Kinematic Residual e_kin Comparison
    ax = axes[1, 1]
    # Phone vs CAN lateral residual
    e_kin_phone = trip_v2['acc_v'][:, 1] - trip_v2['speed'] * trip_v2['gyro_v'][:, 2]
    e_kin_can = trip_v2['can_acc'] - trip_v2['speed'] * np.radians(trip_v2['heading'])  # approx
    ax.hist(np.clip(e_kin_phone, -10, 10), bins=50, density=True, alpha=0.6, color='tab:purple', label='Phone IMU e_kin (Std=2.65)')
    ax.set_title("Offline Kinematic Residual e_kin = a_y - v_x*omega_z", fontsize=11, fontweight='bold')
    ax.set_xlabel("Residual e_kin (m/s²)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = FIG_DIR / "c7_b0_empirical_acceleration_and_residual_distributions.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"   Saved Figure 1: {fig1_path}")

    # Figure 2: Drift Comparison Across Horizons (Vta02 & Vta04)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h_vals = [5, 10, 20, 30, 60]
    h_labels = ["5s", "10s", "20s", "30s", "60s"]

    # Vta02
    ax = axes[0]
    c_styles = {
        'C0': ('tab:blue', 'o-', 'C0 Baseline (BCAC+NHC+ZUPT)'),
        'B1': ('tab:red', 's-', 'B1 Nominal (a_soft=4.0)'),
        'B1_3': ('tab:orange', '^--', 'B1 Tighter (a_soft=3.0)'),
        'B1_5': ('tab:purple', 'v--', 'B1 Looser (a_soft=5.0)'),
        'Oracle_B1': ('tab:green', 'd-.', 'Oracle-B1 (True a_ref)')
    }
    for c in conditions:
        color, marker, label = c_styles[c]
        vals = [res_v2[f"{h}s"]['conditions'][c]['drift_mean'] for h in h_vals]
        ax.plot(h_vals, vals, marker, color=color, linewidth=2.0, label=label)
    ax.set_title("Vta02 (Primary Diagnostic): Mean Drift vs Outage Horizon", fontsize=11, fontweight='bold')
    ax.set_xlabel("Outage Duration (seconds)")
    ax.set_ylabel("Mean Position Drift (meters)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Vta04
    ax = axes[1]
    for c in conditions:
        color, marker, label = c_styles[c]
        vals = [res_v4[f"{h}s"]['conditions'][c]['drift_mean'] for h in h_vals]
        ax.plot(h_vals, vals, marker, color=color, linewidth=2.0, label=label)
    ax.set_title("Vta04 (Held-Out Continuous Highway): Mean Drift vs Horizon", fontsize=11, fontweight='bold')
    ax.set_xlabel("Outage Duration (seconds)")
    ax.set_ylabel("Mean Position Drift (meters)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig2_path = FIG_DIR / "c7_b1_oracle_vs_soft_acceleration_drift_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"   Saved Figure 2: {fig2_path}")

    # Figure 3: Forensic Time-Series on Vta04 Window 4 (60s continuous outage)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    t_f = forensic_c0['traj']['time']

    # Panel A: Position Error
    ax = axes[0]
    ax.plot(t_f, forensic_c0['traj']['pos_err'], color='tab:blue', linewidth=2.0, label='C0 Baseline (No Accel Bound)')
    ax.plot(t_f, forensic_b1['traj']['pos_err'], color='tab:red', linewidth=2.0, label='B1 Soft Accel Constraint')
    ax.plot(t_f, forensic_ora['traj']['pos_err'], color='tab:green', linewidth=2.0, label='Oracle-B1 (True a_ref)')
    ax.set_title("Forensic Time-Series (Vta04 Window 4, 60s Continuous Highway Driving)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Position Error (m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Velocity Error
    ax = axes[1]
    ax.plot(t_f, forensic_c0['traj']['vel_err'], color='tab:blue', linewidth=1.8, label='C0 Velocity Error')
    ax.plot(t_f, forensic_b1['traj']['vel_err'], color='tab:red', linewidth=1.8, label='B1 Velocity Error')
    ax.plot(t_f, forensic_ora['traj']['vel_err'], color='tab:green', linewidth=1.8, label='Oracle-B1 Velocity Error')
    ax.set_ylabel("Velocity Error (m/s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Body Acceleration and Constraint Innovation
    ax = axes[2]
    ax.plot(t_f, forensic_b1['traj']['a_x_v'], color='black', linewidth=1.2, alpha=0.7, label='Estimated Body a_x^v')
    ax.axhline(4.0, color='red', linestyle='--', linewidth=1.5, label='a_soft = +4.0 m/s²')
    ax.axhline(-4.0, color='red', linestyle='--', linewidth=1.5, label='a_soft = -4.0 m/s²')
    # Shade active epochs
    active_mask = np.array(forensic_b1['traj']['constraint_active'])
    if np.sum(active_mask) > 0:
        ax.fill_between(t_f, -6, 6, where=active_mask, color='red', alpha=0.2, label='Constraint Active Epochs')
    ax.set_xlabel("Outage Time (seconds)")
    ax.set_ylabel("Longitudinal a_x^v (m/s²)")
    ax.set_ylim(-6, 6)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = FIG_DIR / "c7_b1_acceleration_timeseries_and_constraint_innovations.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"   Saved Figure 3: {fig3_path}")

    # Figure 4: Regime Breakdown (Vta02 30s horizon)
    fig, ax = plt.subplots(figsize=(12, 6))
    regimes = ['cruising', 'acceleration', 'braking', 'rough_road']
    x = np.arange(len(regimes))
    width = 0.25

    c0_means = [np.mean(res_v2['30s']['regimes']['C0'][r]) if res_v2['30s']['regimes']['C0'][r] else 0.0 for r in regimes]
    b1_means = [np.mean(res_v2['30s']['regimes']['B1'][r]) if res_v2['30s']['regimes']['B1'][r] else 0.0 for r in regimes]
    ora_means = [np.mean(res_v2['30s']['regimes']['Oracle_B1'][r]) if res_v2['30s']['regimes']['Oracle_B1'][r] else 0.0 for r in regimes]

    ax.bar(x - width, c0_means, width, label='C0 Baseline', color='tab:blue', alpha=0.8)
    ax.bar(x, b1_means, width, label='B1 Soft Accel (a_soft=4.0)', color='tab:red', alpha=0.8)
    ax.bar(x + width, ora_means, width, label='Oracle-B1 (True a_ref)', color='tab:green', alpha=0.8)

    ax.set_title("Vta02: 30-Second Drift by Driving Regime (C0 vs B1 vs Oracle-B1)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Position Drift (m)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Steady Cruising\n(N={len(res_v2['30s']['regimes']['C0']['cruising'])})",
                        f"Acceleration\n(N={len(res_v2['30s']['regimes']['C0']['acceleration'])})",
                        f"Severe Braking\n(N={len(res_v2['30s']['regimes']['C0']['braking'])})",
                        f"Rough Road\n(N={len(res_v2['30s']['regimes']['C0']['rough_road'])})"], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig4_path = FIG_DIR / "c7_b1_regime_breakdown_performance.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"   Saved Figure 4: {fig4_path}")

    # -------------------------------------------------------------
    # 6. EXPORT STRUCTURED JSON DELIVERABLE
    # -------------------------------------------------------------
    output_data = {
        'metadata': {
            'stage': 'C7-B1',
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'conditions': conditions,
            'horizons': horizons,
            'a_soft_nominal': 4.0,
            'sigma_a_meas': 0.50
        },
        'Vta02': res_v2,
        'Vta04': res_v4,
        'forensic_w4_summary': {
            'c0_drift_m': forensic_c0['final_drift_m'],
            'b1_drift_m': forensic_b1['final_drift_m'],
            'ora_drift_m': forensic_ora['final_drift_m'],
            'c0_vel_err_ms': forensic_c0['final_vel_err_ms'],
            'b1_vel_err_ms': forensic_b1['final_vel_err_ms'],
            'ora_vel_err_ms': forensic_ora['final_vel_err_ms']
        }
    }

    json_path = RES_DIR / "c7_b1_kinematic_bounds.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[6] Saved Structured Deliverable: {json_path}")
    print(f"Total Stage C7-B1 execution time: {time.time() - start_total_time:.1f}s")


if __name__ == '__main__':
    main()
