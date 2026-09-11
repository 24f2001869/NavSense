"""
SIH26168 - Step 7: Multi-Architecture Benchmark & SIH Target Verification
Evaluates 4 navigation modalities under simulated GNSS outages:
  1. Pure IMU Dead Reckoning
  2. Baseline EKF (unassisted during outage)
  3. AI Speed Dead Reckoning
  4. AI + EKF Fusion (Hybrid Navigation with NHC)

Generates the full SIH evaluation matrix:
  - Distances: 50m, 100m, 500m, 1000m
  - Outages: 5s, 10s, 30s, 60s
  - Target: Drift Percentage < 10.0%
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.dead_reckoning import run_pure_imu_dead_reckoning, run_heading_velocity_dead_reckoning
from src.fusion.ekf import NavigationEKF
from src.fusion.ai_ekf import AIFusedEKF
from src.evaluation.outage_simulator import GNSSOutageSimulator
from src.evaluation.metrics import calculate_navigation_metrics, compute_cumulative_distance
from src.ml.predict_velocity import AIVelocityPredictor
from src.ml.dataset import extract_window_features

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_four_mode_comparison(trip_name="Vta04", outage_start=25.0, outage_dur=30.0):
    """
    Runs a head-to-head trajectory comparison of all 4 architectures on trip_name
    during a 30s GNSS blackout.
    """
    print(f"\n=======================================================")
    print(f"  Running 4-Mode Benchmark on {trip_name} (Outage: {outage_dur}s)")
    print(f"=======================================================")

    df_p, df_v = load_trip(trip_name)
    n_total = len(df_p)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, _ = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    # 1. Alignment
    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values

    acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)

    # Dataset inspection finding: gyro_x is the physical turning axis in this phone mounting
    yaw_rate = df_p['gyro_x'].values

    # 2. Extract AI Predicted Speeds
    predictor = AIVelocityPredictor(ema_alpha=0.08)
    X_feat, _, _ = extract_window_features(acc_v, gyro_v, speed, window_size=10, step_size=1)
    ai_speeds_window = predictor.predict_batch(X_feat)
    pad = len(acc_v) - len(ai_speeds_window)
    ai_speed_full = np.concatenate([np.full(pad, ai_speeds_window[0]), ai_speeds_window])

    # Outage mask
    time_s = df_p['time_s'].values
    outage_sim = GNSSOutageSimulator().add_outage(outage_start, outage_dur)
    gnss_avail = outage_sim.generate_outage_mask(time_s)

    # Initial conditions
    init_yaw = np.radians(df_v['veh_heading_deg'].iloc[0])
    init_ve = speed[0] * np.sin(init_yaw)
    init_vn = speed[0] * np.cos(init_yaw)

    # -------------------------------------------------------------
    # Mode 1: Pure IMU Dead Reckoning
    # -------------------------------------------------------------
    dr_e, dr_n, _, _, dr_yaw = run_pure_imu_dead_reckoning(
        acc_v, np.column_stack([np.zeros(n_total), np.zeros(n_total), yaw_rate]), dt=0.1,
        init_pos=(gt_e[0], gt_n[0]),
        init_vel=(init_ve, init_vn),
        init_yaw_rad=init_yaw
    )

    # -------------------------------------------------------------
    # Mode 2: Baseline EKF (Blind during outage)
    # -------------------------------------------------------------
    base_ekf = NavigationEKF(init_pos=(gt_e[0], gt_n[0]), init_vel=(init_ve, init_vn), init_yaw=init_yaw)
    bekf_e, bekf_n = np.zeros(n_total), np.zeros(n_total)

    for k in range(n_total):
        base_ekf.predict(acc_v[k, 0], yaw_rate[k], dt=0.1)
        if gnss_avail[k] and (k % 10 == 0):
            h_rad = np.radians(df_v['veh_heading_deg'].iloc[k])
            meas_ve = speed[k] * np.sin(h_rad)
            meas_vn = speed[k] * np.cos(h_rad)
            base_ekf.update_gnss(gt_e[k], gt_n[k], meas_ve, meas_vn, heading_rad=h_rad)
        bekf_e[k], bekf_n[k] = base_ekf.x[0], base_ekf.x[1]

    # -------------------------------------------------------------
    # Mode 3: AI Speed + Heading DR
    # -------------------------------------------------------------
    ai_dr_e, ai_dr_n = run_heading_velocity_dead_reckoning(
        ai_speed_full, dr_yaw, dt=0.1, init_pos=(gt_e[0], gt_n[0])
    )

    # -------------------------------------------------------------
    # Mode 4: AI + EKF Fusion with NHC (Our Proposed System)
    # -------------------------------------------------------------
    fused_ekf = AIFusedEKF(init_pos=(gt_e[0], gt_n[0]), init_vel=(init_ve, init_vn), init_yaw=init_yaw)
    fused_e, fused_n = np.zeros(n_total), np.zeros(n_total)

    for k in range(n_total):
        fused_ekf.predict(acc_v[k, 0], yaw_rate[k], dt=0.1)
        if gnss_avail[k] and (k % 10 == 0):
            h_rad = np.radians(df_v['veh_heading_deg'].iloc[k])
            meas_ve = speed[k] * np.sin(h_rad)
            meas_vn = speed[k] * np.cos(h_rad)
            fused_ekf.update_gnss(gt_e[k], gt_n[k], meas_ve, meas_vn, heading_rad=h_rad)
        elif not gnss_avail[k]:
            # During outage: apply AI velocity and NHC at 10 Hz
            fused_ekf.update_ai_and_nhc(ai_speed_full[k])
        fused_e[k], fused_n[k] = fused_ekf.x[0], fused_ekf.x[1]

    # Evaluate during the outage window
    outage_indices = np.where(~gnss_avail)[0]
    outage_dist = float(compute_cumulative_distance(gt_e[outage_indices], gt_n[outage_indices])[-1])

    modes = {
        "1. Pure IMU DR": (dr_e, dr_n),
        "2. Baseline EKF (Blind)": (bekf_e, bekf_n),
        "3. AI Speed DR": (ai_dr_e, ai_dr_n),
        "4. AI + EKF Fusion (Ours)": (fused_e, fused_n)
    }

    comparison_results = []
    for mode_name, (est_e, est_n) in modes.items():
        sub_est_e = est_e[outage_indices]
        sub_est_n = est_n[outage_indices]
        sub_gt_e = gt_e[outage_indices]
        sub_gt_n = gt_n[outage_indices]

        m, _ = calculate_navigation_metrics(sub_est_e, sub_est_n, sub_gt_e, sub_gt_n)
        drift_during_outage = (m['fpe_m'] / max(outage_dist, 1.0)) * 100.0
        pass_sih = drift_during_outage < 10.0

        comparison_results.append({
            "Mode": mode_name,
            "Outage Dist (m)": round(outage_dist, 1),
            "Max Error (m)": round(m['max_error_m'], 2),
            "Final Error (m)": round(m['fpe_m'], 2),
            "RMSE (m)": round(m['rmse_m'], 2),
            "Drift %": round(drift_during_outage, 2),
            "SIH Target (<10%)": "PASS" if pass_sih else "FAIL"
        })

    df_comp = pd.DataFrame(comparison_results)
    print(df_comp.to_string(index=False))

    # Plot Trajectory Comparison
    plt.figure(figsize=(12, 9))
    plt.plot(gt_e, gt_n, 'g-', lw=3.0, label='Ground Truth Path', zorder=5)
    plt.plot(bekf_e, bekf_n, 'm--', lw=2.0, label='Baseline EKF (Diverges in Outage)')
    plt.plot(ai_dr_e, ai_dr_n, 'c-.', lw=2.0, label='AI Speed DR')
    plt.plot(fused_e, fused_n, 'b-', lw=2.5, label='AI + EKF Fusion (Proposed System)')

    plt.plot(gt_e[outage_indices], gt_n[outage_indices], 'r-', lw=4.5, alpha=0.6, label=f'Simulated Tunnel / Outage ({outage_dur}s)')
    plt.plot(gt_e[0], gt_n[0], 'ko', markersize=9, label='Start')

    plt.xlabel('East (meters)', fontsize=11)
    plt.ylabel('North (meters)', fontsize=11)
    plt.title(f'SIH26168 Architecture Comparison | {outage_dur}s GNSS Outage on Unseen Trip ({trip_name})', fontsize=13, fontweight='bold')
    plt.grid(True)
    plt.axis('equal')
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

    out_file = FIG_DIR / f"{trip_name}_architecture_comparison.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved architecture comparison: {out_file}")

    return df_comp

def generate_sih_benchmark_matrix(trip_name="Vta04"):
    """
    Evaluates our proposed AI+EKF Fusion system across the explicit SIH benchmarks:
      - Distances: ~50m, ~100m, ~350m, ~700m
      - Outages: 5s, 10s, 30s, 60s
    """
    print(f"\n=======================================================")
    print(f"  Generating SIH26168 Target Evaluation Matrix")
    print(f"=======================================================")

    df_p, df_v = load_trip(trip_name)
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, _ = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    yaw_rate = df_p['gyro_x'].values

    acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)

    predictor = AIVelocityPredictor(ema_alpha=0.08)
    X_feat, _, _ = extract_window_features(acc_v, gyro_v, speed, window_size=10, step_size=1)
    ai_speeds_window = predictor.predict_batch(X_feat)
    pad = len(acc_v) - len(ai_speeds_window)
    ai_speed_full = np.concatenate([np.full(pad, ai_speeds_window[0]), ai_speeds_window])

    test_outages = [5.0, 10.0, 30.0, 60.0]
    results_matrix = []
    cum_dist = compute_cumulative_distance(gt_e, gt_n)

    for dur in test_outages:
        outage_start = 25.0
        time_s = df_p['time_s'].values

        outage_sim = GNSSOutageSimulator().add_outage(outage_start, dur)
        gnss_avail = outage_sim.generate_outage_mask(time_s)

        init_yaw = np.radians(df_v['veh_heading_deg'].iloc[0])
        fused_ekf = AIFusedEKF(
            init_pos=(gt_e[0], gt_n[0]),
            init_vel=(speed[0]*np.sin(init_yaw), speed[0]*np.cos(init_yaw)),
            init_yaw=init_yaw
        )

        n = len(time_s)
        est_e, est_n = np.zeros(n), np.zeros(n)

        for k in range(n):
            fused_ekf.predict(acc_v[k, 0], yaw_rate[k], dt=0.1)
            if gnss_avail[k] and (k % 10 == 0):
                h_rad = np.radians(df_v['veh_heading_deg'].iloc[k])
                meas_ve = speed[k] * np.sin(h_rad)
                meas_vn = speed[k] * np.cos(h_rad)
                fused_ekf.update_gnss(gt_e[k], gt_n[k], meas_ve, meas_vn, heading_rad=h_rad)
            elif not gnss_avail[k]:
                fused_ekf.update_ai_and_nhc(ai_speed_full[k])
            est_e[k], est_n[k] = fused_ekf.x[0], fused_ekf.x[1]

        outage_idx = np.where(~gnss_avail)[0]
        actual_outage_dist = float(cum_dist[outage_idx[-1]] - cum_dist[outage_idx[0]])
        final_err = float(np.sqrt((est_e[outage_idx[-1]] - gt_e[outage_idx[-1]])**2 + (est_n[outage_idx[-1]] - gt_n[outage_idx[-1]])**2))
        max_err = float(np.max(np.sqrt((est_e[outage_idx] - gt_e[outage_idx])**2 + (est_n[outage_idx] - gt_n[outage_idx])**2)))
        drift_pct = (final_err / max(actual_outage_dist, 1.0)) * 100.0

        results_matrix.append({
            "Outage Duration (s)": int(dur),
            "Distance Travelled (m)": round(actual_outage_dist, 1),
            "Final Error (m)": round(final_err, 2),
            "Max Error (m)": round(max_err, 2),
            "Drift %": round(drift_pct, 2),
            "SIH Target (<10%)": "PASS" if drift_pct < 10.0 else "FAIL"
        })

    df_matrix = pd.DataFrame(results_matrix)
    print(df_matrix.to_string(index=False))

    matrix_file = RESULTS_DIR / "benchmark_matrix.csv"
    df_matrix.to_csv(matrix_file, index=False)
    print(f"\nSaved SIH Benchmark Matrix to: {matrix_file}")

    return df_matrix

if __name__ == "__main__":
    run_four_mode_comparison("Vta04", outage_start=25.0, outage_dur=30.0)
    generate_sih_benchmark_matrix("Vta04")
