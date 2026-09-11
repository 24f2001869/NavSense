"""
SIH26168 - Stage C7-B1 Mechanism Audit: Inspecting ESKF State Corrections & NHC Coupling
Module: experiments/audit_c7_b1_mechanism.py

Performs a controlled, forensic mechanism audit on representative Vta04 windows (30s and 60s)
to quantify the exact mathematical channel through which the soft acceleration constraint
affects each ESKF state and couples into NHC.

Records epoch-by-epoch:
1. Acceleration constraint innovation r_a
2. Acceleration constraint Kalman gain K_a across all 15 states
3. Corrections to:
   - delta_v (velocity)
   - delta_theta (attitude, specifically roll, pitch, and yaw)
   - delta_ba (accelerometer bias)
   - delta_bg (gyroscope bias)
4. NHC innovation before and after update
5. NHC state correction
6. Heading trajectory psi(t) and heading error delta_psi(t)
7. Along-track velocity v_x^v(t) vs VBOX ground truth
8. Cross-track velocity v_y^v(t) vs ground truth (0 m/s)
9. Trajectory along-track and cross-track position errors

Compares C0 (Baseline), B1_3 (Soft 3.0 m/s^2), and Oracle-B1 (True a_ref).
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


def prepare_trip_data(trip_name="Vta04", dt=0.1):
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
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'n_epochs': n
    }


def run_mechanism_audit_window(kw, w_dur, dt, trip_data, condition='B1_3'):
    """
    Executes a forensic window run recording full internal Kalman state telemetry.
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
    speed = trip_data['speed']

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
    detector = CausalStationaryDetector(dt=dt, window_sec=0.5, persist_sec=0.8)

    if condition == 'B1_3':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=3.0, sigma_a_meas=0.50)
    elif condition == 'Oracle_B1':
        constraint = SoftLongitudinalAccelerationConstraint(a_soft=4.0, sigma_a_meas=0.10)
    else:
        constraint = None

    # Telemetry storage
    times = []
    innov_a = []
    kalman_gain_vel_x = []
    kalman_gain_pitch = []
    kalman_gain_yaw = []
    kalman_gain_ba_x = []

    corr_vel_fwd = []
    corr_vel_lat = []
    corr_pitch_deg = []
    corr_yaw_deg = []
    corr_ba_x = []

    nhc_innov_before = []
    nhc_innov_after = []
    nhc_corr_yaw_deg = []

    heading_est_deg = []
    heading_gt_deg = []
    heading_err_deg = []

    v_fwd_est = []
    v_fwd_gt = []
    v_lat_est = []
    v_lat_gt = []

    err_along_m = []
    err_cross_m = []
    err_pos_total_m = []

    for step_k in range(kw, kw + w_dur):
        t_now = (step_k - kw + 1) * dt
        times.append(t_now)
        eskf.sigma_a = sigma_series[step_k]

        ax = acc_v[step_k, 0]
        ay = acc_v[step_k, 1]
        az = acc_v[step_k, 2]
        gx = gyro_v[step_k, 0]
        gy = gyro_v[step_k, 1]
        gz = gyro_v[step_k, 2]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # Record pre-constraint heading and velocity
        C_v_n_pre = eskf.attitude.get_dcm()
        C_n_v_pre = C_v_n_pre.T
        v_v_pre = C_n_v_pre @ eskf.vel_n

        # Acceleration constraint update
        act_a = False
        r_a_val = 0.0
        delta_x_a = np.zeros(15)
        K_a_val = np.zeros(15)

        if constraint is not None:
            a_x_v, a_v, g_v = constraint.compute_body_accel(eskf)
            if condition == 'Oracle_B1':
                res_c = constraint.update_eskf(eskf, dt=dt, oracle_a_ref=a_vbox_diff[step_k])
            else:
                res_c = constraint.update_eskf(eskf, dt=dt)

            if res_c['active']:
                act_a = True
                r_a_val = float(res_c['innovation'])
                delta_x_a = res_c['delta_x'].copy()
                if abs(r_a_val) > 1e-12:
                    K_a_val = delta_x_a / r_a_val

        innov_a.append(r_a_val)
        kalman_gain_vel_x.append(float(K_a_val[3]))
        kalman_gain_pitch.append(float(np.degrees(K_a_val[7])))
        kalman_gain_yaw.append(float(np.degrees(K_a_val[8])))
        kalman_gain_ba_x.append(float(K_a_val[9]))

        corr_vel_fwd.append(float(delta_x_a[3]))
        corr_vel_lat.append(float(delta_x_a[4]))
        corr_pitch_deg.append(float(np.degrees(delta_x_a[7])))
        corr_yaw_deg.append(float(np.degrees(delta_x_a[8])))
        corr_ba_x.append(float(delta_x_a[9]))

        # NHC update
        C_v_n_mid = eskf.attitude.get_dcm()
        C_n_v_mid = C_v_n_mid.T
        v_v_mid = C_n_v_mid @ eskf.vel_n
        nhc_innov_before.append(float(-v_v_mid[1]))

        res_nhc = nhc.update_eskf(eskf)
        nhc_corr = res_nhc.get('delta_x', np.zeros(15))
        nhc_corr_yaw_deg.append(float(np.degrees(nhc_corr[8])))

        C_v_n_post = eskf.attitude.get_dcm()
        C_n_v_post = C_v_n_post.T
        v_v_post = C_n_v_post @ eskf.vel_n
        nhc_innov_after.append(float(-v_v_post[1]))

        # ZUPT update
        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        # Vehicle-frame velocities & errors
        v_fwd_est.append(float(v_v_post[0]))
        v_fwd_gt.append(float(speed[step_k]))
        v_lat_est.append(float(v_v_post[1]))
        v_lat_gt.append(0.0)

        # Attitude & heading
        yaw_est = eskf.attitude.get_yaw_deg()
        yaw_gt = float(heading[step_k + 1])
        heading_est_deg.append(yaw_est)
        heading_gt_deg.append(yaw_gt)
        h_diff = (yaw_est - yaw_gt + 180.0) % 360.0 - 180.0
        heading_err_deg.append(h_diff)

        # Position errors
        st = eskf.get_state()
        p_curr = st['pos_n'][:2]
        gt_p_curr = np.array([gt_e[step_k + 1], gt_n[step_k + 1]])
        gt_h_rad = np.radians(yaw_gt)
        u_fwd = np.array([np.sin(gt_h_rad), np.cos(gt_h_rad)])
        u_lat = np.array([-np.cos(gt_h_rad), np.sin(gt_h_rad)])

        e_vec = p_curr - gt_p_curr
        err_along_m.append(float(np.dot(e_vec, u_fwd)))
        err_cross_m.append(float(np.dot(e_vec, u_lat)))
        err_pos_total_m.append(float(np.linalg.norm(e_vec)))

    return {
        'time': times,
        'innov_a': innov_a,
        'K_vel_x': kalman_gain_vel_x,
        'K_pitch': kalman_gain_pitch,
        'K_yaw': kalman_gain_yaw,
        'K_ba_x': kalman_gain_ba_x,
        'corr_vel_fwd': corr_vel_fwd,
        'corr_vel_lat': corr_vel_lat,
        'corr_pitch_deg': corr_pitch_deg,
        'corr_yaw_deg': corr_yaw_deg,
        'corr_ba_x': corr_ba_x,
        'nhc_innov_before': nhc_innov_before,
        'nhc_innov_after': nhc_innov_after,
        'nhc_corr_yaw_deg': nhc_corr_yaw_deg,
        'heading_est': heading_est_deg,
        'heading_gt': heading_gt_deg,
        'heading_err': heading_err_deg,
        'v_fwd_est': v_fwd_est,
        'v_fwd_gt': v_fwd_gt,
        'v_lat_est': v_lat_est,
        'err_along': err_along_m,
        'err_cross': err_cross_m,
        'err_pos': err_pos_total_m,
        'final_drift': err_pos_total_m[-1],
        'final_along': err_along_m[-1],
        'final_cross': err_cross_m[-1],
        'final_heading_err': heading_err_deg[-1]
    }


def main():
    print("===============================================================================")
    print("STAGE C7-B1 MECHANISM AUDIT: ESKF STATE CORRECTIONS & NHC COUPLING")
    print("===============================================================================")
    t0 = time.time()
    dt = 0.1

    trip_v4 = prepare_trip_data("Vta04", dt=dt)

    # Auditing representative windows on Vta04:
    # Window 0 (t=5-35s), Window 2 (t=45-75s), Window 4 (t=85-115s)
    test_windows = [
        {'name': 'Window 0 (t=5-35s)', 'kw': 50, 'w_dur': 300},
        {'name': 'Window 2 (t=45-75s)', 'kw': 450, 'w_dur': 300},
        {'name': 'Window 4 (t=85-115s)', 'kw': 850, 'w_dur': 300},
        {'name': 'Window 4 Long (t=85-145s, 60s)', 'kw': 850, 'w_dur': 600},
    ]

    audit_results = {}

    for twin in test_windows:
        wname = twin['name']
        kw = twin['kw']
        wdur = twin['w_dur']
        print(f"\nEvaluating {wname}...", flush=True)

        res_c0 = run_mechanism_audit_window(kw, wdur, dt, trip_v4, condition='C0')
        res_b13 = run_mechanism_audit_window(kw, wdur, dt, trip_v4, condition='B1_3')
        res_ora = run_mechanism_audit_window(kw, wdur, dt, trip_v4, condition='Oracle_B1')

        audit_results[wname] = {
            'C0': res_c0,
            'B1_3': res_b13,
            'Oracle_B1': res_ora
        }

        print(f"  [End Metrics for {wname}]:")
        print(f"    C0:        Drift={res_c0['final_drift']:.2f} m | Along={res_c0['final_along']:.2f} m | Cross={res_c0['final_cross']:.2f} m | HeadErr={res_c0['final_heading_err']:.2f}°")
        print(f"    B1_3:      Drift={res_b13['final_drift']:.2f} m | Along={res_b13['final_along']:.2f} m | Cross={res_b13['final_cross']:.2f} m | HeadErr={res_b13['final_heading_err']:.2f}°")
        print(f"    Oracle-B1: Drift={res_ora['final_drift']:.2f} m | Along={res_ora['final_along']:.2f} m | Cross={res_ora['final_cross']:.2f} m | HeadErr={res_ora['final_heading_err']:.2f}°")

        # Accumulate corrections made by B1_3
        b13_yaw_corr_sum = float(np.sum(np.abs(res_b13['corr_yaw_deg'])))
        b13_pitch_corr_sum = float(np.sum(np.abs(res_b13['corr_pitch_deg'])))
        b13_vel_corr_sum = float(np.sum(np.abs(res_b13['corr_vel_fwd'])))
        b13_ba_corr_sum = float(np.sum(np.abs(res_b13['corr_ba_x'])))
        print(f"    B1_3 Cumulative State Corrections: |d_yaw|={b13_yaw_corr_sum:.3f}° | |d_pitch|={b13_pitch_corr_sum:.3f}° | |d_vel|={b13_vel_corr_sum:.3f} m/s | |d_ba|={b13_ba_corr_sum:.3f} m/s²")

    # -------------------------------------------------------------
    # GENERATE 4-PANEL FORENSIC COUPLING FIGURE (Window 2 t=45-75s)
    # -------------------------------------------------------------
    w_target = audit_results['Window 2 (t=45-75s)']
    t_axis = w_target['C0']['time']

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Position Error Decomposition (Along vs Cross)
    ax = axes[0]
    ax.plot(t_axis, w_target['C0']['err_along'], 'b--', label='C0 Along-Track Error', linewidth=1.5)
    ax.plot(t_axis, w_target['C0']['err_cross'], 'b:', label='C0 Cross-Track Error', linewidth=1.5)
    ax.plot(t_axis, w_target['B1_3']['err_along'], 'r-', label='B1_3 Along-Track Error (Suppressed)', linewidth=2.0)
    ax.plot(t_axis, w_target['B1_3']['err_cross'], 'r:', label='B1_3 Cross-Track Error (Exploded)', linewidth=2.0)
    ax.plot(t_axis, w_target['Oracle_B1']['err_along'], 'g-', label='Oracle Along-Track Error', linewidth=1.5)
    ax.plot(t_axis, w_target['Oracle_B1']['err_cross'], 'g:', label='Oracle Cross-Track Error', linewidth=1.5)
    ax.set_title("Vta04 Window 2 (t=45-75s): Along-Track vs Cross-Track Position Divergence", fontsize=11, fontweight='bold')
    ax.set_ylabel("Error (m)")
    ax.legend(fontsize=8, loc='upper left', ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: Heading Error Trajectory
    ax = axes[1]
    ax.plot(t_axis, w_target['C0']['heading_err'], 'b--', label='C0 Heading Error', linewidth=1.5)
    ax.plot(t_axis, w_target['B1_3']['heading_err'], 'r-', label='B1_3 Heading Error', linewidth=2.0)
    ax.plot(t_axis, w_target['Oracle_B1']['heading_err'], 'g-', label='Oracle Heading Error', linewidth=1.5)
    ax.set_title("Vehicle Heading Error δψ (Attitude Drift)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Heading Error (°)")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 3: Acceleration Constraint Corrections to Attitude (Pitch & Yaw)
    ax = axes[2]
    ax.plot(t_axis, w_target['B1_3']['corr_pitch_deg'], 'm-', label='B1_3 Accel Correction to Pitch δθ_y', linewidth=1.2)
    ax.plot(t_axis, w_target['B1_3']['corr_yaw_deg'], 'c-', label='B1_3 Accel Correction to Yaw δθ_z', linewidth=1.2)
    ax.set_title("Attitude Corrections Injected by Acceleration Constraint (δθ)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Correction (°)")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 4: NHC Lateral Innovation and Yaw Correction
    ax = axes[3]
    ax.plot(t_axis, w_target['C0']['nhc_innov_before'], 'b--', label='C0 NHC Innovation (-v_y^v)', linewidth=1.2)
    ax.plot(t_axis, w_target['B1_3']['nhc_innov_before'], 'r-', label='B1_3 NHC Innovation (-v_y^v)', linewidth=1.5)
    ax.plot(t_axis, w_target['B1_3']['nhc_corr_yaw_deg'], 'k:', label='B1_3 NHC Correction to Yaw', linewidth=1.2)
    ax.set_title("NHC Lateral Velocity Innovation & Subsequent Yaw Correction", fontsize=11, fontweight='bold')
    ax.set_xlabel("Outage Time (seconds)")
    ax.set_ylabel("NHC Innovation (m/s) / Yaw (°)")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "c7_b1_forensic_mechanism_audit.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved Forensic Mechanism Figure: {fig_path}")

    # Export structured JSON
    export_summary = {}
    for wname, wdata in audit_results.items():
        export_summary[wname] = {
            'C0': {
                'drift': wdata['C0']['final_drift'],
                'along': wdata['C0']['final_along'],
                'cross': wdata['C0']['final_cross'],
                'head_err': wdata['C0']['final_heading_err']
            },
            'B1_3': {
                'drift': wdata['B1_3']['final_drift'],
                'along': wdata['B1_3']['final_along'],
                'cross': wdata['B1_3']['final_cross'],
                'head_err': wdata['B1_3']['final_heading_err'],
                'total_d_yaw_deg': float(np.sum(np.abs(wdata['B1_3']['corr_yaw_deg']))),
                'total_d_pitch_deg': float(np.sum(np.abs(wdata['B1_3']['corr_pitch_deg']))),
                'total_d_vel_ms': float(np.sum(np.abs(wdata['B1_3']['corr_vel_fwd']))),
                'total_d_ba_ms2': float(np.sum(np.abs(wdata['B1_3']['corr_ba_x'])))
            },
            'Oracle_B1': {
                'drift': wdata['Oracle_B1']['final_drift'],
                'along': wdata['Oracle_B1']['final_along'],
                'cross': wdata['Oracle_B1']['final_cross'],
                'head_err': wdata['Oracle_B1']['final_heading_err']
            }
        }

    out_json = RES_DIR / "c7_b1_mechanism_audit.json"
    with open(out_json, 'w') as f:
        json.dump(export_summary, f, indent=2)
    print(f"Saved Structured Audit Data: {out_json}")
    print(f"Mechanism Audit Completed in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
