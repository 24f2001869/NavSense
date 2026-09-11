"""
SIH26168 - Stage C7-A1: Oracle Zero-Velocity Updates (ZUPT) Diagnostic
Module: experiments/run_zupt_oracle_c7_a1.py

Evaluates the theoretical upper bound of Zero-Velocity Updates (ZUPT) on the
frozen Stage C5/C6 architecture (Bounded Adaptive C0 / BCAC + NHC) by using
offline VBOX ground-truth standstill intervals on primary diagnostic trip Vta02.

Protocol:
1. Ground Truth Stop Identification: VBOX vehicle speed < 0.05 m/s.
2. Outage Matrix on Vta02 across 10s, 20s, 30s, 60s horizons:
   - Group A (Stop-Intersecting): Outages containing >= 1.0 s of true standstill.
   - Group B (Continuous-Motion): Outages with 0 standstill (control group).
3. Evaluates Baseline BCAC (No ZUPT) vs. Oracle ZUPT (3D velocity update at stop epochs).
4. Extracts high-resolution forensic time-series on Stop 7 (45.9 s traffic stop):
   - Pre-stop vs. post-stop velocity error
   - End-of-outage position drift reduction
   - Forward accelerometer bias convergence
   - Post-stop error recovery duration
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
from src.navigation.zupt import ZeroVelocityUpdate

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


def run_single_outage(
    kw, w_dur, dt,
    acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
    gt_speed,
    apply_oracle_zupt=False,
    record_trajectory=False
):
    """Simulates a single outage window with optional Oracle ZUPT updates."""
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

    traj_time = []
    traj_pos_err = []
    traj_vel_err = []
    traj_lat_err = []
    traj_long_err = []
    traj_ba_x = []
    zupt_applied_epochs = []

    # Standstill detection in window
    window_gt_speeds = gt_speed[kw : kw + w_dur]
    standstill_epochs_in_win = np.sum(window_gt_speeds < 0.05)

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

        # 2. Oracle ZUPT update if vehicle is truly stationary (VBOX speed < 0.05 m/s)
        is_true_stop = gt_speed[step_k] < 0.05
        if apply_oracle_zupt and is_true_stop:
            zupt.update_eskf(eskf)
            zupt_applied_epochs.append(step_k - kw)

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

            traj_time.append(t_curr)
            traj_pos_err.append(float(np.linalg.norm(e_vec)))
            traj_vel_err.append(float(np.linalg.norm(v_curr - gt_v_curr)))
            traj_long_err.append(float(np.dot(e_vec, u_fwd)))
            traj_lat_err.append(float(np.dot(e_vec, u_lat)))
            traj_ba_x.append(float(eskf.ba[0]))

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
        'standstill_sec': float(standstill_epochs_in_win * dt),
        'has_qualifying_stop': bool(standstill_epochs_in_win >= 10),  # >= 1.0 s stop
        'zupt_count': len(zupt_applied_epochs)
    }

    if record_trajectory:
        out['traj'] = {
            'time': traj_time,
            'pos_err': traj_pos_err,
            'vel_err': traj_vel_err,
            'long_err': traj_long_err,
            'lat_err': traj_lat_err,
            'ba_x': traj_ba_x,
            'zupt_epochs': zupt_applied_epochs
        }

    return out


def main():
    print("===============================================================================")
    print("STAGE C7-A1: ORACLE ZERO-VELOCITY UPDATES (ZUPT) DIAGNOSTIC")
    print("===============================================================================")
    start_total_time = time.time()
    dt = 0.1

    print("\nLoading trip Vta02...", flush=True)
    df_p, df_v = load_trip("Vta02")

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    acc_v, gyro_v, R_vp, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124], dtype=np.float64)
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    n_total = len(speed)
    print(f"Vta02 loaded: {n_total} epochs ({n_total*dt:.1f} s), Total stop time = {np.sum(speed < 0.05)*dt:.1f} s", flush=True)

    # -------------------------------------------------------------
    # 1. EVALUATE MULTI-HORIZON OUTAGES (10s, 20s, 30s, 60s)
    # -------------------------------------------------------------
    horizons = [10.0, 20.0, 30.0, 60.0]
    stride_s = 20.0  # Dense sampling to capture stop windows
    stride_k = int(round(stride_s / dt))

    results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, n_total - w_dur - 1, stride_k))
        n_win = len(window_starts)

        print(f"\n--- Simulating Horizon {int(h)}s ({n_win} windows) ---", flush=True)
        t_h0 = time.time()

        base_dr = []
        zupt_dr = []
        base_ve = []
        zupt_ve = []
        base_long = []
        zupt_long = []
        has_stop = []
        stop_secs = []

        for w_idx, kw in enumerate(window_starts):
            # 1. Baseline BCAC (No ZUPT)
            r_base = run_single_outage(
                kw, w_dur, dt,
                acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
                speed, apply_oracle_zupt=False, record_trajectory=False
            )
            # 2. Oracle ZUPT
            r_zupt = run_single_outage(
                kw, w_dur, dt,
                acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
                speed, apply_oracle_zupt=True, record_trajectory=False
            )

            base_dr.append(r_base['final_drift_m'])
            zupt_dr.append(r_zupt['final_drift_m'])
            base_ve.append(r_base['final_vel_err_ms'])
            zupt_ve.append(r_zupt['final_vel_err_ms'])
            base_long.append(abs(r_base['final_long_err_m']))
            zupt_long.append(abs(r_zupt['final_long_err_m']))
            has_stop.append(r_base['has_qualifying_stop'])
            stop_secs.append(r_base['standstill_sec'])

            if (w_idx + 1) % 15 == 0 or (w_idx + 1) == n_win:
                print(f"   Window {w_idx+1}/{n_win} ({((w_idx+1)/n_win)*100:.0f}%) complete ({time.time()-t_h0:.1f} s)", flush=True)

        has_stop = np.array(has_stop, dtype=bool)
        base_dr = np.array(base_dr)
        zupt_dr = np.array(zupt_dr)
        base_ve = np.array(base_ve)
        zupt_ve = np.array(zupt_ve)
        base_long = np.array(base_long)
        zupt_long = np.array(zupt_long)

        n_stop_win = int(np.sum(has_stop))
        n_motion_win = int(np.sum(~has_stop))

        # Group A: Stop-Intersecting
        grp_a = {
            'n_windows': n_stop_win,
            'baseline_drift_mean': float(np.mean(base_dr[has_stop])) if n_stop_win > 0 else 0.0,
            'zupt_drift_mean': float(np.mean(zupt_dr[has_stop])) if n_stop_win > 0 else 0.0,
            'baseline_drift_median': float(np.median(base_dr[has_stop])) if n_stop_win > 0 else 0.0,
            'zupt_drift_median': float(np.median(zupt_dr[has_stop])) if n_stop_win > 0 else 0.0,
            'baseline_vel_mean': float(np.mean(base_ve[has_stop])) if n_stop_win > 0 else 0.0,
            'zupt_vel_mean': float(np.mean(zupt_ve[has_stop])) if n_stop_win > 0 else 0.0,
            'baseline_long_mean': float(np.mean(base_long[has_stop])) if n_stop_win > 0 else 0.0,
            'zupt_long_mean': float(np.mean(zupt_long[has_stop])) if n_stop_win > 0 else 0.0,
        }
        if grp_a['baseline_drift_mean'] > 0:
            grp_a['reduction_m'] = grp_a['baseline_drift_mean'] - grp_a['zupt_drift_mean']
            grp_a['reduction_pct'] = (grp_a['reduction_m'] / grp_a['baseline_drift_mean']) * 100.0

        # Group B: Continuous-Motion
        grp_b = {
            'n_windows': n_motion_win,
            'baseline_drift_mean': float(np.mean(base_dr[~has_stop])),
            'zupt_drift_mean': float(np.mean(zupt_dr[~has_stop])),
            'baseline_drift_median': float(np.median(base_dr[~has_stop])),
            'zupt_drift_median': float(np.median(zupt_dr[~has_stop])),
        }

        # Overall Aggregate
        agg = {
            'n_windows_total': n_win,
            'baseline_drift_mean': float(np.mean(base_dr)),
            'zupt_drift_mean': float(np.mean(zupt_dr)),
            'overall_reduction_m': float(np.mean(base_dr) - np.mean(zupt_dr)),
            'overall_reduction_pct': float(((np.mean(base_dr) - np.mean(zupt_dr)) / np.mean(base_dr)) * 100.0)
        }

        results[h_key] = {
            'horizon_s': h,
            'group_a_stop_intersecting': grp_a,
            'group_b_continuous_motion': grp_b,
            'aggregate': agg
        }

        print(f"   [Horizon {int(h)}s Results]:")
        if n_stop_win > 0:
            print(f"      Group A (Stop-Intersecting, N={n_stop_win}): Base={grp_a['baseline_drift_mean']:.2f}m -> ZUPT={grp_a['zupt_drift_mean']:.2f}m ({grp_a['reduction_pct']:.1f}% reduction / -{grp_a['reduction_m']:.2f}m)")
            print(f"         Along-Track Error: Base={grp_a['baseline_long_mean']:.2f}m -> ZUPT={grp_a['zupt_long_mean']:.2f}m")
            print(f"         Final Vel Error: Base={grp_a['baseline_vel_mean']:.2f}m/s -> ZUPT={grp_a['zupt_vel_mean']:.2f}m/s")
        print(f"      Group B (Continuous Motion, N={n_motion_win}): Base={grp_b['baseline_drift_mean']:.2f}m -> ZUPT={grp_b['zupt_drift_mean']:.2f}m (Difference: {grp_b['baseline_drift_mean'] - grp_b['zupt_drift_mean']:.4f}m)")
        print(f"      Overall Trip Aggregate (N={n_win}): Base={agg['baseline_drift_mean']:.2f}m -> ZUPT={agg['zupt_drift_mean']:.2f}m ({agg['overall_reduction_pct']:.1f}% reduction)")

    # -------------------------------------------------------------
    # 2. FORENSIC TIME-SERIES AUDIT ON STOP 7 (45.9s Standstill)
    # -------------------------------------------------------------
    print("\n[PHASE 2] Extracting High-Resolution Forensic Trajectory on Stop 7...", flush=True)
    # Stop 7 starts at t = 707.3s. Let's pick an outage starting at t = 700.0s (kw = 7000), 60s horizon.
    # Vehicle drives for 7.3s, stops for 45.9s (707.3 - 753.2s), then resumes driving for 6.8s.
    kw_stop7 = 7000
    w_dur_stop7 = 600  # 60s

    res_base_s7 = run_single_outage(
        kw_stop7, w_dur_stop7, dt,
        acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
        speed, apply_oracle_zupt=False, record_trajectory=True
    )
    res_zupt_s7 = run_single_outage(
        kw_stop7, w_dur_stop7, dt,
        acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
        speed, apply_oracle_zupt=True, record_trajectory=True
    )

    t_base = res_base_s7['traj']['time']
    t_zupt = res_zupt_s7['traj']['time']

    # Pre-stop and post-stop metrics
    # Stop starts at epoch index ~ 73 (7.3s)
    idx_pre = 72
    # Stop ends at epoch index ~ 532 (53.2s)
    idx_during = 300  # midpoint of stop
    idx_end = len(t_base) - 1

    forensic_summary = {
        'window_start_time_s': kw_stop7 * dt,
        'outage_duration_s': w_dur_stop7 * dt,
        'stop_start_relative_s': 7.3,
        'stop_end_relative_s': 53.2,
        'pre_stop_vel_err_base': res_base_s7['traj']['vel_err'][idx_pre],
        'pre_stop_vel_err_zupt': res_zupt_s7['traj']['vel_err'][idx_pre],
        'mid_stop_vel_err_base': res_base_s7['traj']['vel_err'][idx_during],
        'mid_stop_vel_err_zupt': res_zupt_s7['traj']['vel_err'][idx_during],
        'end_vel_err_base': res_base_s7['traj']['vel_err'][idx_end],
        'end_vel_err_zupt': res_zupt_s7['traj']['vel_err'][idx_end],
        'end_drift_base_m': res_base_s7['traj']['pos_err'][idx_end],
        'end_drift_zupt_m': res_zupt_s7['traj']['pos_err'][idx_end],
        'end_drift_reduction_m': res_base_s7['traj']['pos_err'][idx_end] - res_zupt_s7['traj']['pos_err'][idx_end],
        'end_drift_reduction_pct': ((res_base_s7['traj']['pos_err'][idx_end] - res_zupt_s7['traj']['pos_err'][idx_end]) / res_base_s7['traj']['pos_err'][idx_end]) * 100.0,
        'final_ba_x_base': res_base_s7['traj']['ba_x'][idx_end],
        'final_ba_x_zupt': res_zupt_s7['traj']['ba_x'][idx_end]
    }

    print(f"Stop 7 Forensic Telemetry (60s Outage):")
    print(f"   Pre-Stop Velocity Error (t=7.2s): Base = {forensic_summary['pre_stop_vel_err_base']:.2f} m/s | ZUPT = {forensic_summary['pre_stop_vel_err_zupt']:.2f} m/s")
    print(f"   Mid-Stop Velocity Error (t=30.0s): Base = {forensic_summary['mid_stop_vel_err_base']:.2f} m/s | ZUPT = {forensic_summary['mid_stop_vel_err_zupt']:.4f} m/s (Collapsed to ~0!)")
    print(f"   End Velocity Error (t=60.0s): Base = {forensic_summary['end_vel_err_base']:.2f} m/s | ZUPT = {forensic_summary['end_vel_err_zupt']:.2f} m/s")
    print(f"   End Position Drift (t=60.0s): Base = {forensic_summary['end_drift_base_m']:.2f} m -> ZUPT = {forensic_summary['end_drift_zupt_m']:.2f} m")
    print(f"   >>> NET DRIFT REDUCTION: -{forensic_summary['end_drift_reduction_m']:.2f} m (-{forensic_summary['end_drift_reduction_pct']:.1f}%) <<<")

    # -------------------------------------------------------------
    # 3. GENERATE DIAGNOSTIC FIGURES
    # -------------------------------------------------------------
    print("\n[PHASE 3] Generating Diagnostic Figures...", flush=True)

    # FIGURE 1: Forensic Time-Series on Stop 7
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Panel A: Velocity Error Trace
    axes[0, 0].plot(t_base, res_base_s7['traj']['vel_err'], label='Baseline BCAC (No ZUPT)', color='crimson', lw=2)
    axes[0, 0].plot(t_zupt, res_zupt_s7['traj']['vel_err'], label='Oracle ZUPT', color='navy', lw=2.2)
    axes[0, 0].axvspan(7.3, 53.2, color='gray', alpha=0.2, label='True Standstill Window (VBOX < 0.05 m/s)')
    axes[0, 0].set_title('(A) 3D Velocity Error Trace over 60s Outage', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Outage Elapsed Time (s)', fontsize=10)
    axes[0, 0].set_ylabel('Velocity Error (m/s)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=9)

    # Panel B: Position Drift Growth
    axes[0, 1].plot(t_base, res_base_s7['traj']['pos_err'], label='Baseline BCAC (No ZUPT)', color='crimson', lw=2)
    axes[0, 1].plot(t_zupt, res_zupt_s7['traj']['pos_err'], label='Oracle ZUPT', color='navy', lw=2.2)
    axes[0, 1].axvspan(7.3, 53.2, color='gray', alpha=0.2, label='True Standstill Window')
    axes[0, 1].set_title('(B) Position Error Accumulation (Drift Frozen at Stop)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Outage Elapsed Time (s)', fontsize=10)
    axes[0, 1].set_ylabel('Total Position Drift (m)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(fontsize=9)

    # Panel C: Along-Track vs. Cross-Track Error
    axes[1, 0].plot(t_base, res_base_s7['traj']['long_err'], label='Baseline Along-Track', color='darkred', ls='--', lw=1.8)
    axes[1, 0].plot(t_zupt, res_zupt_s7['traj']['long_err'], label='Oracle ZUPT Along-Track', color='blue', lw=2)
    axes[1, 0].plot(t_zupt, res_zupt_s7['traj']['lat_err'], label='Oracle ZUPT Cross-Track', color='green', lw=1.8)
    axes[1, 0].axvspan(7.3, 53.2, color='gray', alpha=0.2)
    axes[1, 0].set_title('(C) Along-Track vs. Cross-Track Error Growth', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Outage Elapsed Time (s)', fontsize=10)
    axes[1, 0].set_ylabel('Error Component (m)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(fontsize=9)

    # Panel D: Accelerometer Bias State Estimate b_a,x
    axes[1, 1].plot(t_base, res_base_s7['traj']['ba_x'], label='Baseline ba_x (Static Init)', color='crimson', ls='--', lw=1.8)
    axes[1, 1].plot(t_zupt, res_zupt_s7['traj']['ba_x'], label='Oracle ZUPT ba_x (Online Calibrated)', color='purple', lw=2.2)
    axes[1, 1].axvspan(7.3, 53.2, color='gray', alpha=0.2)
    axes[1, 1].set_title(r'(D) Accelerometer Forward Bias Estimate $\hat{b}_{a,x}(t)$', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Outage Elapsed Time (s)', fontsize=10)
    axes[1, 1].set_ylabel(r'Forward Bias $\hat{b}_{a,x}$ (m/s²)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    fig1_path = FIG_DIR / "c7_a1_oracle_zupt_timeseries_forensic.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 1: {fig1_path}", flush=True)

    # FIGURE 2: Multi-Horizon Drift Reduction: Group A (Stop) vs. Group B (Continuous)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    h_labels = [f"{int(h)}s" for h in horizons]
    x = np.arange(len(h_labels))

    # Panel A: Group A (Stop-Intersecting Outages)
    base_a = [results[hl]['group_a_stop_intersecting']['baseline_drift_mean'] for hl in h_labels]
    zupt_a = [results[hl]['group_a_stop_intersecting']['zupt_drift_mean'] for hl in h_labels]
    axes[0].bar(x - 0.18, base_a, 0.35, label='Baseline BCAC', color='coral', alpha=0.85)
    axes[0].bar(x + 0.18, zupt_a, 0.35, label='Oracle ZUPT', color='teal', alpha=0.85)
    axes[0].set_title('(A) Group A: Outages Intersecting a True Stop', fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h_labels)
    axes[0].set_ylabel('Mean Final Position Drift (m)', fontsize=10)
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(fontsize=9.5)

    # Panel B: Overall Vta02 Aggregate vs Group B
    base_agg = [results[hl]['aggregate']['baseline_drift_mean'] for hl in h_labels]
    zupt_agg = [results[hl]['aggregate']['zupt_drift_mean'] for hl in h_labels]
    axes[1].bar(x - 0.18, base_agg, 0.35, label='Trip Baseline (All Windows)', color='crimson', alpha=0.85)
    axes[1].bar(x + 0.18, zupt_agg, 0.35, label='Trip with ZUPT (All Windows)', color='navy', alpha=0.85)
    axes[1].set_title('(B) Entire Trip Aggregate Performance', fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(h_labels)
    axes[1].set_ylabel('Mean Final Position Drift (m)', fontsize=10)
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout()
    fig2_path = FIG_DIR / "c7_a1_oracle_zupt_drift_comparison.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 2: {fig2_path}", flush=True)

    # -------------------------------------------------------------
    # 4. EXPORT STRUCTURED JSON DELIVERABLE
    # -------------------------------------------------------------
    deliverable = {
        'status': 'Complete',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sub_phase': 'Stage C7-A1: Oracle ZUPT Diagnostic',
        'trip': 'Vta02',
        'horizons_evaluation': results,
        'stop7_forensic_case_study': forensic_summary,
        'figures': [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT))
        ]
    }

    json_path = RES_DIR / "c7_a1_zupt_oracle.json"
    with open(json_path, 'w') as f:
        json.dump(deliverable, f, indent=2)
    print(f"\nSaved structured JSON deliverable: {json_path}", flush=True)

    print(f"\n>>> Stage C7-A1 Oracle Diagnostic Successfully Completed in {time.time() - start_total_time:.1f} s <<<", flush=True)


if __name__ == "__main__":
    main()
