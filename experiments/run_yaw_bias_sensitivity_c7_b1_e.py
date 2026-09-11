"""
SIH26168 - Stage C7-B1-E: Controlled Yaw Gyro Bias Sensitivity Experiment
Module: experiments/run_yaw_bias_sensitivity_c7_b1_e.py

Hypothesis Test:
    Is the remaining cross-track error (~138 m on Vta04 at 30 s) under strict-freeze
    acceleration consistency actually sensitive to unobserved yaw gyro bias (b_{g,z})?

Experimental Formulation:
    - Baseline Filter: Stage C7-B1-D Winning Architecture (B1_3 Strict Freeze):
        * ESKF 15-state + NHC (0.5 m/s) + BCAC (w_b=10s) + Deployable ZUPT
        * Soft acceleration constraint a_soft = 3.0 m/s^2, sigma_a_meas = 0.50 m/s^2
        * Decoupled Jacobian H_theta = 0_1x3 and Strict Attitude Freeze K_theta = 0_3x1
        * Joseph-stabilized covariance update with K_actual
    - Controlled Perturbation:
        delta_b_gz in {-0.010, -0.005, -0.0025, -0.001, 0.0, +0.001, +0.0025, +0.005, +0.010} rad/s
    - Testbeds: Vta04 (9 windows highway) and Vta02 (54 windows suburban)
    - Horizons: 10 s, 20 s, 30 s, 60 s
    - Recorded Metrics:
        * Final heading error (deg)
        * Cross-track error (signed and absolute mean)
        * Along-track error (signed and absolute mean)
        * Total position drift (mean, median, P90)
        * Velocity error (m/s)
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
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'n_epochs': n
    }


def run_single_window_perturbation(
    kw, w_dur, dt,
    trip_data,
    delta_bg_z=0.0
):
    """
    Runs a single outage window under B1_3 Strict Freeze with an injected yaw gyro bias.
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

    constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]

        ax = acc_v[step_k, 0]
        ay = acc_v[step_k, 1]
        az = acc_v[step_k, 2]
        gx = gyro_v[step_k, 0]
        gy = gyro_v[step_k, 1]
        # Inject controlled yaw gyro bias delta_bg_z (rad/s)
        gz = gyro_v[step_k, 2] + delta_bg_z

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 1. NHC update
        nhc.update_eskf(eskf)

        # 2. Deployable ZUPT update
        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        # 3. Soft Acceleration constraint with Strict Attitude Freeze
        constraint.update_eskf(eskf, dt=dt)

    # End-of-window metrics
    st_end = eskf.get_state()
    gt_p_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur]])
    gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur]])
    end_h_deg = float(heading[kw + w_dur])
    end_h_rad = np.radians(end_h_deg)

    e_vec_end = st_end['pos_n'][:2] - gt_p_end
    u_fwd_end = np.array([np.sin(end_h_rad), np.cos(end_h_rad)])
    u_lat_end = np.array([-np.cos(end_h_rad), np.sin(end_h_rad)])

    final_yaw = st_end['yaw_deg']
    final_h_err = abs(float((final_yaw - end_h_deg + 180.0) % 360.0 - 180.0))

    return {
        'total_drift_m': float(np.linalg.norm(e_vec_end)),
        'along_track_abs_m': float(abs(np.dot(e_vec_end, u_fwd_end))),
        'cross_track_abs_m': float(abs(np.dot(e_vec_end, u_lat_end))),
        'along_track_signed_m': float(np.dot(e_vec_end, u_fwd_end)),
        'cross_track_signed_m': float(np.dot(e_vec_end, u_lat_end)),
        'vel_err_ms': float(np.linalg.norm(st_end['vel_n'][:2] - gt_v_end)),
        'final_heading_err_deg': final_h_err
    }


def evaluate_bias_sweep(trip_data, bias_values, horizons=[10.0, 20.0, 30.0, 60.0], stride_s=20.0, dt=0.1):
    """
    Evaluates the yaw bias perturbation grid across all horizons for a given trip.
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
            'biases': {}
        }

        for b in bias_values:
            b_key = f"{b:+.4f}"
            win_res = []
            for kw in window_starts:
                out = run_single_window_perturbation(kw, w_dur, dt, trip_data, delta_bg_z=b)
                win_res.append(out)

            drifts = [w['total_drift_m'] for w in win_res]
            long_abs = [w['along_track_abs_m'] for w in win_res]
            cross_abs = [w['cross_track_abs_m'] for w in win_res]
            long_sgn = [w['along_track_signed_m'] for w in win_res]
            cross_sgn = [w['cross_track_signed_m'] for w in win_res]
            vel_err = [w['vel_err_ms'] for w in win_res]
            h_err = [w['final_heading_err_deg'] for w in win_res]

            results[h_key]['biases'][b_key] = {
                'delta_bg_z': float(b),
                'drift_mean': float(np.mean(drifts)),
                'drift_median': float(np.median(drifts)),
                'drift_p90': float(np.percentile(drifts, 90)),
                'along_track_abs_mean': float(np.mean(long_abs)),
                'cross_track_abs_mean': float(np.mean(cross_abs)),
                'along_track_signed_mean': float(np.mean(long_sgn)),
                'cross_track_signed_mean': float(np.mean(cross_sgn)),
                'vel_err_mean': float(np.mean(vel_err)),
                'final_heading_err_mean': float(np.mean(h_err))
            }

    return results


def plot_bias_sensitivity(all_results, bias_values):
    """
    Generates multi-panel publication diagnostic plot of yaw bias sensitivity.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    horizons = ['10s', '20s', '30s', '60s']
    h_colors = {'10s': '#2ecc71', '20s': '#3498db', '30s': '#f39c12', '60s': '#e74c3c'}

    b_array = np.array(bias_values)
    b_deg_s = b_array * (180.0 / np.pi)

    # 1. Vta04 Cross-Track Error vs delta_bg_z
    ax = axes[0, 0]
    for h in horizons:
        vals = [all_results['Vta04'][h]['biases'][f"{b:+.4f}"]['cross_track_abs_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='o', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Cross-Track Error (m)', fontsize=10, fontweight='bold')
    ax.set_title('Vta04 Highway: Cross-Track Error Sensitivity', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 2. Vta04 Total Drift vs delta_bg_z
    ax = axes[0, 1]
    for h in horizons:
        vals = [all_results['Vta04'][h]['biases'][f"{b:+.4f}"]['drift_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='s', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Total Drift (m)', fontsize=10, fontweight='bold')
    ax.set_title('Vta04 Highway: Total Position Drift Sensitivity', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 3. Vta04 Final Heading Error vs delta_bg_z
    ax = axes[0, 2]
    for h in horizons:
        vals = [all_results['Vta04'][h]['biases'][f"{b:+.4f}"]['final_heading_err_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='^', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Heading Error (deg)', fontsize=10, fontweight='bold')
    ax.set_title('Vta04 Highway: Final Heading Error Sensitivity', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 4. Vta04 Along-Track Error vs delta_bg_z (Should remain relatively flat!)
    ax = axes[1, 0]
    for h in horizons:
        vals = [all_results['Vta04'][h]['biases'][f"{b:+.4f}"]['along_track_abs_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='d', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Along-Track Error (m)', fontsize=10, fontweight='bold')
    ax.set_title('Vta04 Highway: Along-Track Error Invariance', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 5. Vta02 Suburban Cross-Track Error vs delta_bg_z
    ax = axes[1, 1]
    for h in horizons:
        vals = [all_results['Vta02'][h]['biases'][f"{b:+.4f}"]['cross_track_abs_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='o', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Cross-Track Error (m)', fontsize=10, fontweight='bold')
    ax.set_title('Vta02 Suburban: Cross-Track Error Sensitivity', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 6. Vta02 Suburban Total Drift vs delta_bg_z
    ax = axes[1, 2]
    for h in horizons:
        vals = [all_results['Vta02'][h]['biases'][f"{b:+.4f}"]['drift_mean'] for b in bias_values]
        ax.plot(b_deg_s, vals, marker='s', lw=2.0, color=h_colors[h], label=f'{h} Horizon')
    ax.set_xlabel('Injected Yaw Gyro Bias δb_gz (deg/s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Total Drift (m)', fontsize=10, fontweight='bold')
    ax.set_title('Vta02 Suburban: Total Position Drift Sensitivity', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = FIG_DIR / "c7_b1_e_yaw_bias_sensitivity.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[C7-B1-E] Saved sensitivity figure to: {fig_path}")


def main():
    print("=" * 80)
    print("SIH26168 - Stage C7-B1-E: Controlled Yaw Gyro Bias Sensitivity Experiment")
    print("Testing Hypothesis: Is remaining cross-track error directly sensitive to delta_bg_z?")
    print("=" * 80)

    t0 = time.time()
    dt = 0.1
    horizons = [10.0, 20.0, 30.0, 60.0]
    bias_values = [-0.010, -0.005, -0.0025, -0.001, 0.0, +0.001, +0.0025, +0.005, +0.010]

    all_results = {}

    for trip_name in ['Vta04', 'Vta02']:
        print(f"\n[C7-B1-E] Loading trip: {trip_name}...")
        trip_data = prepare_trip_data(trip_name, dt=dt)
        print(f"Loaded {trip_data['n_epochs']} epochs.")

        print(f"[C7-B1-E] Running bias sweep across {bias_values} rad/s...")
        res = evaluate_bias_sweep(trip_data, bias_values, horizons=horizons, stride_s=20.0, dt=dt)
        all_results[trip_name] = res

        # Display summary table for 30s
        print(f"\n--- {trip_name} 30-s Yaw Bias Sensitivity Summary ---")
        h30 = res['30s']['biases']
        print(f"{'delta_bg_z (rad/s)':<20} | {'deg/s':<8} | {'Drift (m)':<10} | {'Cross (m)':<10} | {'Along (m)':<10} | {'Head Err (deg)':<14} | {'Vel Err (m/s)':<12}")
        print("-" * 96)
        for b in bias_values:
            b_k = f"{b:+.4f}"
            row = h30[b_k]
            b_deg = b * (180.0 / np.pi)
            print(f"{row['delta_bg_z']:<+20.4f} | {b_deg:<+8.3f} | {row['drift_mean']:<10.2f} | {row['cross_track_abs_mean']:<10.2f} | {row['along_track_abs_mean']:<10.2f} | {row['final_heading_err_mean']:<14.2f} | {row['vel_err_mean']:<12.2f}")

    # Generate figure
    print("\n[C7-B1-E] Generating publication figure...")
    plot_bias_sensitivity(all_results, bias_values)

    # Save master JSON
    out_json = RES_DIR / "c7_b1_e_yaw_bias_sensitivity.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[C7-B1-E] Master JSON saved to: {out_json}")

    elapsed = time.time() - t0
    print(f"\n[C7-B1-E] Completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
