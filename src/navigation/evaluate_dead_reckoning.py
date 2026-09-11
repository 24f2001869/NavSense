"""
SIH26168 - Step 4: Pure Dead Reckoning Benchmark Evaluation
Runs pure IMU dead reckoning and speed+heading dead reckoning against ground truth,
computes metrics, and saves the comparison trajectory figure.
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
from src.navigation.dead_reckoning import run_pure_imu_dead_reckoning, run_heading_velocity_dead_reckoning
from src.evaluation.metrics import calculate_navigation_metrics

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_dead_reckoning(trip_name="Vta02", duration_s=120):
    df_p, df_v = load_trip(trip_name)

    # Limit to initial test window (e.g. 120s or full)
    mask = df_p['time_s'] <= duration_s
    df_p = df_p[mask].reset_index(drop=True)
    df_v = df_v[mask].reset_index(drop=True)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, _ = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    # 1. Align phone to vehicle frame
    raw_accel = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values

    acc_v, gyro_v, R_pv, _ = align_phone_to_vehicle(raw_accel, raw_gyro, speed)

    # Initial conditions from ground truth
    init_yaw = np.radians(df_v['veh_heading_deg'].iloc[0])
    init_v_e = speed[0] * np.sin(init_yaw)
    init_v_n = speed[0] * np.cos(init_yaw)

    # 2. Run Pure IMU DR
    dr_e, dr_n, _, _, _ = run_pure_imu_dead_reckoning(
        acc_v, gyro_v, dt=0.1,
        init_pos=(0.0, 0.0),
        init_vel=(init_v_e, init_v_n),
        init_yaw_rad=init_yaw
    )

    # 3. Compute Metrics
    metrics_pure, err_pure = calculate_navigation_metrics(dr_e, dr_n, gt_e, gt_n)
    print("=== Pure IMU Dead Reckoning Metrics ===")
    for k, v in metrics_pure.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # 4. Plot Comparison
    plt.figure(figsize=(10, 8))
    plt.plot(gt_e, gt_n, 'g-', lw=2.5, label='Ground Truth Trajectory')
    plt.plot(dr_e, dr_n, 'r--', lw=2.0, label='Pure IMU Dead Reckoning (Unconstrained)')
    plt.plot(0, 0, 'ko', markersize=8, label='Start Point')
    plt.xlabel('East (meters)')
    plt.ylabel('North (meters)')
    plt.title(f'[{trip_name}] Pure IMU Dead Reckoning vs Ground Truth ({duration_s}s Window)', fontsize=13, fontweight='bold')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    out_file = FIG_DIR / f"{trip_name}_pure_dead_reckoning_drift.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved drift comparison: {out_file}")

    return metrics_pure

if __name__ == "__main__":
    evaluate_dead_reckoning("Vta02", duration_s=120)
