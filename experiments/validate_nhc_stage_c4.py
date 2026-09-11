"""
SIH26168 - Stage C4: Non-Holonomic Constraints (NHC) Validation
Validation Script: experiments/validate_nhc_stage_c4.py

Evaluates the 15-state 3D ESKF with Non-Holonomic Constraints (v_y^v ~ 0, v_z^v ~ 0)
on IO-VNBD trip Vta04 during the GNSS outage interval t = 25.1 s to 55.0 s.
Compares head-to-head against:
  - Stage C2: 3D Pure Mechanization (139.64 m / 40.53% drift)
  - Stage C3: 3D ESKF without NHC (2,300.23 m / 667.71% drift failure case)
  - Stage C4: 3D ESKF with NHC across 3 sensitivity covariance levels (sigma = 1.0, 0.5, 0.2 m/s)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.orientation import geodetic_to_enu
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.navigation.mechanization import InertialMechanization3D
from src.navigation.eskf import ESKF3D
from src.navigation.nhc import (
    NonHolonomicConstraint,
    compute_nhc_residual_and_jacobian,
    validate_nhc_jacobian_finite_difference,
    skew
)
from src.evaluation.metrics import compute_cumulative_distance


def compute_observability_analysis(C_v_n, f_v, omega_v, speed_fwd, has_gnss=True, has_nhc=True):
    """
    Computes singular values and rank of the 15-state observability matrix
    under specified measurement configurations.
    """
    F_c = np.zeros((15, 15), dtype=np.float64)
    F_c[0:3, 3:6] = np.eye(3)
    F_c[3:6, 6:9] = -C_v_n @ skew(f_v)
    F_c[3:6, 9:12] = -C_v_n
    F_c[6:9, 6:9] = -skew(omega_v)
    F_c[6:9, 12:15] = -np.eye(3)

    H_blocks = []
    if has_gnss:
        H_gnss = np.zeros((6, 15), dtype=np.float64)
        H_gnss[0:3, 0:3] = np.eye(3)
        H_gnss[3:6, 3:6] = np.eye(3)
        H_blocks.append(H_gnss)

    if has_nhc:
        H_nhc = np.zeros((2, 15), dtype=np.float64)
        C_n_v = C_v_n.T
        # Lateral velocity constraint
        H_nhc[0, 3:6] = C_n_v[1, :]
        H_nhc[0, 6:9] = np.array([0.0, 0.0, -speed_fwd], dtype=np.float64)
        # Vertical velocity constraint
        H_nhc[1, 3:6] = C_n_v[2, :]
        H_nhc[1, 6:9] = np.array([0.0, speed_fwd, 0.0], dtype=np.float64)
        H_blocks.append(H_nhc)

    if not H_blocks:
        return 0, np.zeros(15)

    H = np.vstack(H_blocks)

    # Observability matrix O = [H; H*F; H*F^2; H*F^3; H*F^4; H*F^5]
    O_blocks = [H]
    F_pow = np.eye(15)
    for i in range(1, 6):
        F_pow = F_pow @ F_c
        O_blocks.append(H @ F_pow)
    O = np.vstack(O_blocks)

    _, s, _ = np.linalg.svd(O)
    rank = int(np.sum(s > 1e-5))
    return rank, s


def run_stage_c4_experiment():
    print("=" * 80)
    print("      SIH26168 STAGE C4: NON-HOLONOMIC CONSTRAINTS (NHC) VALIDATION     ")
    print("=" * 80)

    # 1. Finite-Difference Jacobian Validation Check
    print("\n--- 1. NUMERICAL NHC JACOBIAN VALIDATION ---")
    val = validate_nhc_jacobian_finite_difference()
    print(f"Max Absolute Difference: {val['max_abs_diff']:.3e}")
    print(f"Relative Error:          {val['rel_error']:.3e}")
    print(f"Validation Status:       {'PASS' if val['passed'] else 'FAIL'}")
    assert val['passed'], "NHC Jacobian validation failed finite difference test!"

    # 2. Dataset Ingestion & Ground Truth Setup
    print("\n--- 2. DATASET INGESTION & GROUND TRUTH SETUP ---")
    df_p, df_v = load_trip('Vta04')
    t = df_p['time_s'].values
    dt = 0.1

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values

    acc_v, gyro_v, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    print(f"Loaded Vta04: {len(t)} samples ({t[-1]:.1f} s).")
    print(f"Alignment Angles: Roll={angles['roll_deg']:.2f}°, Pitch={angles['pitch_deg']:.2f}°, Yaw={angles['yaw_deg']:.2f}°")

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, _ = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    vbox_heading_rad = np.radians(df_v['veh_heading_deg'].values)
    gt_ve = speed * np.sin(vbox_heading_rad)
    gt_vn = speed * np.cos(vbox_heading_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6

    t_outage_start = 25.1
    t_outage_end = 55.0
    k_outage_start = int(round(t_outage_start / dt))
    k_outage_end = int(round(t_outage_end / dt))

    cum_dist = compute_cumulative_distance(gt_e, gt_n)
    outage_dist = float(cum_dist[k_outage_end] - cum_dist[k_outage_start])
    print(f"Outage Window: t = {t_outage_start:.1f} s to {t_outage_end:.1f} s ({t_outage_end - t_outage_start:.1f} s)")
    print(f"Distance Traveled During Outage: {outage_dist:.2f} m\n")

    # =========================================================================
    # 3. BASELINE RUNS: C2 (Open-Loop 3D Mechanization) & C3 (ESKF without NHC)
    # =========================================================================
    print("--- 3. RUNNING BASELINES (C2 and C3) ---")

    # C2 Open-Loop Mechanization
    ins_c2 = InertialMechanization3D(
        init_pos_enu=(gt_e[k_outage_start], gt_n[k_outage_start], 0.0),
        init_vel_enu=(gt_ve[k_outage_start], gt_vn[k_outage_start], gt_vu[k_outage_start]),
        init_heading_deg=float(df_v['veh_heading_deg'].iloc[k_outage_start]),
        R_vp=R_pv,
        gravity=9.80665
    )
    c2_pos_e, c2_pos_n = [], []
    c2_vel_e, c2_vel_n = [], []
    for k in range(k_outage_start, k_outage_end + 1):
        st = ins_c2.get_state()
        c2_pos_e.append(st['pos_n'][0])
        c2_pos_n.append(st['pos_n'][1])
        c2_vel_e.append(st['vel_n'][0])
        c2_vel_n.append(st['vel_n'][1])
        ins_c2.step(raw_acc[k, 0], raw_acc[k, 1], raw_acc[k, 2],
                    raw_gyro[k, 0], raw_gyro[k, 1], raw_gyro[k, 2], dt, apply_gravity_leveling=True)

    c2_final_err = np.linalg.norm([c2_pos_e[-1] - gt_e[k_outage_end], c2_pos_n[-1] - gt_n[k_outage_end]])
    print(f"C2 Baseline Error: {c2_final_err:.2f} m ({c2_final_err / outage_dist * 100:.2f}% drift)")

    # C3 ESKF (Pre-outage GNSS calibration, No NHC)
    eskf_c3 = ESKF3D(
        init_pos_enu=(gt_e[0], gt_n[0], 0.0),
        init_vel_enu=(gt_ve[0], gt_vn[0], gt_vu[0]),
        init_heading_deg=float(df_v['veh_heading_deg'].iloc[0]),
        R_vp=R_pv,
        gravity=9.80665
    )
    for k in range(k_outage_end + 1):
        eskf_c3.predict(raw_acc[k, 0], raw_acc[k, 1], raw_acc[k, 2],
                        raw_gyro[k, 0], raw_gyro[k, 1], raw_gyro[k, 2], dt)
        if k <= k_outage_start and (k % 10 == 0):
            eskf_c3.update_gnss(np.array([gt_e[k], gt_n[k], 0.0]),
                                np.array([gt_ve[k], gt_vn[k], gt_vu[k]]))

    st_c3 = eskf_c3.get_state()
    c3_final_err = np.linalg.norm([st_c3['pos_n'][0] - gt_e[k_outage_end], st_c3['pos_n'][1] - gt_n[k_outage_end]])
    print(f"C3 Failure Error:  {c3_final_err:.2f} m ({c3_final_err / outage_dist * 100:.2f}% drift)")

    # =========================================================================
    # 4. STAGE C4: ESKF + NHC SENSITIVITY EXPERIMENTS
    # =========================================================================
    print("\n--- 4. RUNNING STAGE C4 SENSITIVITY EXPERIMENTS (ESKF + NHC) ---")
    nhc_levels = [
        {'name': 'Conservative (sigma = 1.0 m/s)', 'sigma_lat': 1.0, 'sigma_vert': 1.0},
        {'name': 'Nominal      (sigma = 0.5 m/s)', 'sigma_lat': 0.5, 'sigma_vert': 0.5},
        {'name': 'Tight        (sigma = 0.2 m/s)', 'sigma_lat': 0.2, 'sigma_vert': 0.2}
    ]

    c4_results = {}

    for lvl in nhc_levels:
        s_lat = lvl['sigma_lat']
        s_vert = lvl['sigma_vert']
        lvl_name = lvl['name']
        print(f"\nEvaluating C4 with {lvl_name}...")

        nhc_module = NonHolonomicConstraint(sigma_lat=s_lat, sigma_vert=s_vert)
        eskf_c4 = ESKF3D(
            init_pos_enu=(gt_e[0], gt_n[0], 0.0),
            init_vel_enu=(gt_ve[0], gt_vn[0], gt_vu[0]),
            init_heading_deg=float(df_v['veh_heading_deg'].iloc[0]),
            R_vp=R_pv,
            gravity=9.80665
        )

        records = []
        pre_outage_res = []
        nhc_innovations = []
        p_eigenvalues = []

        for k in range(k_outage_end + 1):
            cur_t = float(t[k])
            ax, ay, az = raw_acc[k]
            gx, gy, gz = raw_gyro[k]

            # 1. IMU propagation
            eskf_c4.predict(ax, ay, az, gx, gy, gz, dt)

            # 2. GNSS update (1 Hz, only pre-outage t < 25.1 s)
            if k <= k_outage_start and (k % 10 == 0):
                eskf_c4.update_gnss(
                    np.array([gt_e[k], gt_n[k], 0.0]),
                    np.array([gt_ve[k], gt_vn[k], gt_vu[k]])
                )

            # 3. NHC update (10 Hz, both pre-outage and during outage)
            nhc_res = nhc_module.update_eskf(eskf_c4)
            nhc_innovations.append(nhc_res['innovation'])

            # 4. Telemetry snapshot
            st = eskf_c4.get_state()
            eigs = np.linalg.eigvalsh(eskf_c4.P)
            p_eigenvalues.append(np.min(eigs))

            rec = {
                'k': k,
                'time_s': cur_t,
                'pos_e': st['pos_n'][0],
                'pos_n': st['pos_n'][1],
                'pos_u': st['pos_n'][2],
                'vel_e': st['vel_n'][0],
                'vel_n': st['vel_n'][1],
                'vel_u': st['vel_n'][2],
                'yaw_deg': st['yaw_deg'],
                'pitch_deg': st['pitch_deg'],
                'roll_deg': st['roll_deg'],
                'b_ax': st['ba'][0],
                'b_ay': st['ba'][1],
                'b_az': st['ba'][2],
                'b_gx_degs': np.degrees(st['bg'][0]),
                'b_gy_degs': np.degrees(st['bg'][1]),
                'b_gz_degs': np.degrees(st['bg'][2]),
                'vel_vx': nhc_res['vel_v'][0],
                'vel_vy': nhc_res['vel_v'][1],
                'vel_vz': nhc_res['vel_v'][2],
                'nhc_inn_y': nhc_res['innovation'][0],
                'nhc_inn_z': nhc_res['innovation'][1],
                'gt_e': gt_e[k],
                'gt_n': gt_n[k],
                'gt_ve': gt_ve[k],
                'gt_vn': gt_vn[k],
                'gt_vu': gt_vu[k],
                'gt_yaw_deg': float(df_v['veh_heading_deg'].iloc[k])
            }
            records.append(rec)

        df_rec = pd.DataFrame(records)

        # Pre-outage metrics (k in [0, k_outage_start])
        pre_m = df_rec['k'] <= k_outage_start
        pre_pos_err = np.sqrt((df_rec.loc[pre_m, 'pos_e'] - df_rec.loc[pre_m, 'gt_e'])**2 +
                              (df_rec.loc[pre_m, 'pos_n'] - df_rec.loc[pre_m, 'gt_n'])**2)
        pre_vel_err = np.sqrt((df_rec.loc[pre_m, 'vel_e'] - df_rec.loc[pre_m, 'gt_ve'])**2 +
                              (df_rec.loc[pre_m, 'vel_n'] - df_rec.loc[pre_m, 'gt_vn'])**2)
        pre_pos_rmse = float(np.sqrt(np.mean(pre_pos_err**2)))
        pre_vel_rmse = float(np.sqrt(np.mean(pre_vel_err**2)))

        pre_yaw_diff = (df_rec.loc[pre_m, 'yaw_deg'] - df_rec.loc[pre_m, 'gt_yaw_deg'] + 180.0) % 360.0 - 180.0
        pre_yaw_rmse = float(np.sqrt(np.mean(pre_yaw_diff**2)))

        final_pre_ba = [df_rec.loc[k_outage_start, 'b_ax'],
                        df_rec.loc[k_outage_start, 'b_ay'],
                        df_rec.loc[k_outage_start, 'b_az']]
        final_pre_bg = [df_rec.loc[k_outage_start, 'b_gx_degs'],
                        df_rec.loc[k_outage_start, 'b_gy_degs'],
                        df_rec.loc[k_outage_start, 'b_gz_degs']]

        # Outage metrics (k in [k_outage_start, k_outage_end])
        out_m = (df_rec['k'] >= k_outage_start) & (df_rec['k'] <= k_outage_end)
        df_out = df_rec.loc[out_m].reset_index(drop=True)

        pos_err_out = np.sqrt((df_out['pos_e'] - df_out['gt_e'])**2 +
                              (df_out['pos_n'] - df_out['gt_n'])**2)
        vel_err_out = np.sqrt((df_out['vel_e'] - df_out['gt_ve'])**2 +
                              (df_out['vel_n'] - df_out['gt_vn'])**2)
        yaw_diff_out = (df_out['yaw_deg'] - df_out['gt_yaw_deg'] + 180.0) % 360.0 - 180.0

        final_pos_err = float(pos_err_out.iloc[-1])
        drift_pct = float(final_pos_err / outage_dist * 100.0)
        rmse_pos = float(np.sqrt(np.mean(pos_err_out**2)))
        final_vel_err = float(vel_err_out.iloc[-1])
        rmse_vel = float(np.sqrt(np.mean(vel_err_out**2)))
        max_vel_err = float(np.max(vel_err_out))
        final_yaw_err = float(yaw_diff_out.iloc[-1])
        max_yaw_err = float(np.max(np.abs(yaw_diff_out)))

        c4_results[lvl_name] = {
            'df_rec': df_rec,
            'df_out': df_out,
            'pre_pos_rmse': pre_pos_rmse,
            'pre_vel_rmse': pre_vel_rmse,
            'pre_yaw_rmse': pre_yaw_rmse,
            'final_pre_ba': final_pre_ba,
            'final_pre_bg': final_pre_bg,
            'final_pos_err': final_pos_err,
            'drift_pct': drift_pct,
            'rmse_pos': rmse_pos,
            'final_vel_err': final_vel_err,
            'rmse_vel': rmse_vel,
            'max_vel_err': max_vel_err,
            'final_yaw_err': final_yaw_err,
            'max_yaw_err': max_yaw_err,
            'min_eig': float(np.min(p_eigenvalues))
        }

        print(f"  Final Position Error: {final_pos_err:.2f} m ({drift_pct:.2f}% drift)")
        print(f"  Final Velocity Error: {final_vel_err:.2f} m/s, Max Vel Error: {max_vel_err:.2f} m/s")
        print(f"  Final Yaw Error:      {final_yaw_err:.2f}°, Max Yaw Error: {max_yaw_err:.2f}°")
        print(f"  Pre-outage b_gz:      {final_pre_bg[2]:.3f} deg/s")

    # =========================================================================
    # 5. FORENSIC WINDOW ANALYSIS (Across the Road Bump)
    # =========================================================================
    print("\n--- 5. FORENSIC ROAD DISTURBANCE ANALYSIS (Nominal sigma = 0.5 m/s) ---")
    nom_df = c4_results['Nominal      (sigma = 0.5 m/s)']['df_rec']

    windows = [
        ('A. Before Disturbance (25.1 - 29.6 s)', 25.1, 29.6),
        ('B. During Disturbance (29.6 - 33.2 s)', 29.6, 33.2),
        ('C. After Disturbance  (33.2 - 55.0 s)', 33.2, 55.0)
    ]

    forensic_records = []
    for w_name, s_t, e_t in windows:
        w_mask = (nom_df['time_s'] >= s_t) & (nom_df['time_s'] <= e_t)
        sub = nom_df.loc[w_mask]

        pos_e = np.sqrt((sub['pos_e'] - sub['gt_e'])**2 + (sub['pos_n'] - sub['gt_n'])**2)
        vel_e = np.sqrt((sub['vel_e'] - sub['gt_ve'])**2 + (sub['vel_n'] - sub['gt_vn'])**2)
        yaw_e = (sub['yaw_deg'] - sub['gt_yaw_deg'] + 180.0) % 360.0 - 180.0

        forensic_records.append({
            'Window': w_name,
            'Mean v_y^v (m/s)': float(np.mean(sub['vel_vy'])),
            'Max |v_y^v| (m/s)': float(np.max(np.abs(sub['vel_vy']))),
            'Mean v_z^v (m/s)': float(np.mean(sub['vel_vz'])),
            'Max |v_z^v| (m/s)': float(np.max(np.abs(sub['vel_vz']))),
            'Mean NHC Inn_y (m/s)': float(np.mean(sub['nhc_inn_y'])),
            'Mean NHC Inn_z (m/s)': float(np.mean(sub['nhc_inn_z'])),
            'End Pos Err (m)': float(pos_e.iloc[-1]),
            'Max Vel Err (m/s)': float(np.max(vel_e)),
            'End Yaw Err (deg)': float(yaw_e.iloc[-1])
        })

    df_forensic = pd.DataFrame(forensic_records)
    print(df_forensic.to_string(index=False))

    # =========================================================================
    # 6. OBSERVABILITY ANALYSIS (Rank & Singular Values Comparison)
    # =========================================================================
    print("\n--- 6. OBSERVABILITY COMPARISON ---")
    C_nom = np.eye(3)
    f_nom = np.array([0.0, 0.0, 9.80665])
    omega_nom = np.array([0.0, 0.0, 0.0])
    speed_nom = 11.5 # m/s typical Vta04 speed

    r_c3, s_c3 = compute_observability_analysis(C_nom, f_nom, omega_nom, speed_nom, has_gnss=True, has_nhc=False)
    r_c4_gnss, s_c4_gnss = compute_observability_analysis(C_nom, f_nom, omega_nom, speed_nom, has_gnss=True, has_nhc=True)
    r_c4_outage, s_c4_outage = compute_observability_analysis(C_nom, f_nom, omega_nom, speed_nom, has_gnss=False, has_nhc=True)

    print(f"C3 (GNSS only, straight motion):      Rank = {r_c3:2d}/15, Smallest SVs: {s_c3[-4]:.2e}, {s_c3[-3]:.2e}, {s_c3[-2]:.2e}, {s_c3[-1]:.2e}")
    print(f"C4 (GNSS + NHC, straight motion):     Rank = {r_c4_gnss:2d}/15, Smallest SVs: {s_c4_gnss[-4]:.2e}, {s_c4_gnss[-3]:.2e}, {s_c4_gnss[-2]:.2e}, {s_c4_gnss[-1]:.2e}")
    print(f"C4 (Outage NHC only, straight motion): Rank = {r_c4_outage:2d}/15, Smallest SVs: {s_c4_outage[-4]:.2e}, {s_c4_outage[-3]:.2e}, {s_c4_outage[-2]:.2e}, {s_c4_outage[-1]:.2e}")

    # =========================================================================
    # 7. GENERATE COMPREHENSIVE COMPARISON TABLE & FIGURES
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    nom_out = c4_results['Nominal      (sigma = 0.5 m/s)']['df_out']
    t_out = nom_out['time_s']

    # Subplot 1: Position Error Comparison
    c2_err_series = np.sqrt((c2_pos_e - gt_e[k_outage_start:k_outage_end + 1])**2 +
                            (c2_pos_n - gt_n[k_outage_start:k_outage_end + 1])**2)
    axes[0].plot(t_out, c2_err_series, 'k--', label='C2 Open-Loop Mechanization (139.6 m)')
    for lvl in nhc_levels:
        l_name = lvl['name']
        res = c4_results[l_name]['df_out']
        p_err = np.sqrt((res['pos_e'] - res['gt_e'])**2 + (res['pos_n'] - res['gt_n'])**2)
        axes[0].plot(t_out, p_err, label=f"C4 {l_name} ({p_err.iloc[-1]:.1f} m)")

    axes[0].set_ylabel('Position Error (m)')
    axes[0].set_title('Stage C4: 3D ESKF + Non-Holonomic Constraints Outage Performance (Vta04)')
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    # Subplot 2: Lateral and Vertical Velocity (NHC channels)
    axes[1].plot(t_out, nom_out['vel_vy'], 'b-', label='Lateral Velocity v_y^v (NHC ~ 0)')
    axes[1].plot(t_out, nom_out['vel_vz'], 'g-', label='Vertical Velocity v_z^v (NHC ~ 0)')
    axes[1].axvspan(29.6, 33.2, color='red', alpha=0.15, label='Rough-Road Disturbance')
    axes[1].set_ylabel('Velocity in Body Frame (m/s)')
    axes[1].grid(True, linestyle=':')
    axes[1].legend()

    # Subplot 3: Estimated Biases
    axes[2].plot(t_out, nom_out['b_gx_degs'], label='b_gx (deg/s)')
    axes[2].plot(t_out, nom_out['b_gy_degs'], label='b_gy (deg/s)')
    axes[2].plot(t_out, nom_out['b_gz_degs'], label='b_gz (deg/s)', color='red', linewidth=2)
    axes[2].axvspan(29.6, 33.2, color='red', alpha=0.15)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Gyro Bias (deg/s)')
    axes[2].grid(True, linestyle=':')
    axes[2].legend()

    fig_dir = REPO_ROOT / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig_path = fig_dir / 'stage_c4_nhc_validation.png'
    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=150)
    plt.close()
    print(f"\nSaved diagnostic plot to: {out_fig_path}")

    return {
        'c2_final_err': c2_final_err,
        'c3_final_err': c3_final_err,
        'c4_results': c4_results,
        'df_forensic': df_forensic,
        'observability': {
            'c3': (r_c3, s_c3),
            'c4_gnss': (r_c4_gnss, s_c4_gnss),
            'c4_outage': (r_c4_outage, s_c4_outage)
        }
    }


if __name__ == "__main__":
    run_stage_c4_experiment()
