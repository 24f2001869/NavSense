"""
SIH26168 - Step 5: Extended Kalman Filter Baseline Evaluation
Simulates navigation with 1 Hz GNSS corrections and an artificial 30-second outage.
Compares Classical EKF vs Ground Truth and Pure Dead Reckoning.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.fusion.ekf import NavigationEKF
from src.evaluation.outage_simulator import GNSSOutageSimulator
from src.evaluation.metrics import calculate_navigation_metrics

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def run_ekf_simulation(trip_name="Vta02", duration_s=150, outage_start=50.0, outage_dur=30.0):
    df_p, df_v = load_trip(trip_name)

    mask = df_p['time_s'] <= duration_s
    df_p = df_p[mask].reset_index(drop=True)
    df_v = df_v[mask].reset_index(drop=True)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, _ = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    # Phone alignment
    raw_accel = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values

    acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_accel, raw_gyro, speed)

    # Setup Outage Simulator
    outage_sim = GNSSOutageSimulator().add_outage(outage_start, outage_dur)
    time_s = df_p['time_s'].values
    gnss_avail = outage_sim.generate_outage_mask(time_s)

    # Initialize EKF
    init_yaw = np.radians(df_v['veh_heading_deg'].iloc[0])
    init_ve = speed[0] * np.sin(init_yaw)
    init_vn = speed[0] * np.cos(init_yaw)

    ekf = NavigationEKF(
        init_pos=(gt_e[0], gt_n[0]),
        init_vel=(init_ve, init_vn),
        init_yaw=init_yaw
    )

    n = len(time_s)
    ekf_e = np.zeros(n)
    ekf_n = np.zeros(n)
    ekf_ve = np.zeros(n)
    ekf_vn = np.zeros(n)
    ekf_yaw = np.zeros(n)

    dt = 0.1

    for k in range(n):
        # 1. IMU Prediction at 10 Hz
        a_fwd = acc_v[k, 0]
        w_z = gyro_v[k, 2]
        ekf.predict(a_fwd, w_z, dt=dt)

        # 2. GNSS Correction at 1 Hz (every 10 samples) IF AVAILABLE
        if gnss_avail[k] and (k % 10 == 0):
            # GNSS measurement (simulate sensor noise ~1.5m, 0.2 m/s)
            meas_e = gt_e[k] + np.random.normal(0, 1.0)
            meas_n = gt_n[k] + np.random.normal(0, 1.0)
            meas_ve = speed[k] * np.sin(np.radians(df_v['veh_heading_deg'].iloc[k]))
            meas_vn = speed[k] * np.cos(np.radians(df_v['veh_heading_deg'].iloc[k]))
            ekf.update_gnss(meas_e, meas_n, meas_ve, meas_vn)

        ekf_e[k] = ekf.x[0]
        ekf_n[k] = ekf.x[1]
        ekf_ve[k] = ekf.x[2]
        ekf_vn[k] = ekf.x[3]
        ekf_yaw[k] = ekf.x[4]

    # Metrics
    metrics, errors = calculate_navigation_metrics(ekf_e, ekf_n, gt_e, gt_n)
    print(f"=== EKF Metrics with {outage_dur}s GNSS Outage ===")
    for k_m, v_m in metrics.items():
        print(f"  {k_m}: {v_m:.2f}" if isinstance(v_m, float) else f"  {k_m}: {v_m}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(gt_e, gt_n, 'g-', lw=2.5, label='Ground Truth')
    plt.plot(ekf_e, ekf_n, 'b--', lw=2.0, label='Baseline EKF')

    # Highlight outage window
    outage_idx = np.where(~gnss_avail)[0]
    if len(outage_idx) > 0:
        plt.plot(ekf_e[outage_idx], ekf_n[outage_idx], 'r-', lw=3.0, label=f'GNSS Outage ({outage_dur:.0f}s)')

    plt.plot(0, 0, 'ko', markersize=8, label='Start')
    plt.xlabel('East (meters)')
    plt.ylabel('North (meters)')
    plt.title(f'[{trip_name}] Baseline EKF with {outage_dur}s GNSS Outage (t={outage_start}-{outage_start+outage_dur}s)', fontsize=13, fontweight='bold')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    out_file = FIG_DIR / f"{trip_name}_ekf_outage_simulation.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved EKF outage figure: {out_file}")

    return metrics

if __name__ == "__main__":
    run_ekf_simulation("Vta02", duration_s=150, outage_start=50.0, outage_dur=30.0)
