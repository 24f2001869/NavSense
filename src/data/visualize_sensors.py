"""
SIH26168 - Step 2: Sensor Visualization & 2D Ground Truth Route Mapping
Generates publication-quality diagnostic plots for smartphone IMU streams
and vehicle ground truth reference trajectories.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data.loader import load_trip

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_sensor_timeseries(df_phone, df_veh, trip_name="Vta02", duration_s=120):
    """Plots initial window of IMU and reference velocity streams."""
    mask_phone = df_phone['time_s'] <= duration_s
    mask_veh = df_veh['time_rel_s'] <= duration_s

    p_sub = df_phone[mask_phone]
    v_sub = df_veh[mask_veh]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Accelerometer
    axes[0].plot(p_sub['time_s'], p_sub['accel_x'], label='Accel X (Lateral)', color='#1f77b4', alpha=0.85)
    axes[0].plot(p_sub['time_s'], p_sub['accel_y'], label='Accel Y (Longitudinal)', color='#2ca02c', alpha=0.85)
    axes[0].plot(p_sub['time_s'], p_sub['accel_z'], label='Accel Z (Vertical + g)', color='#d62728', alpha=0.85)
    axes[0].set_ylabel('Accel ($m/s^2$)')
    axes[0].set_title(f'[{trip_name}] Smartphone 3-Axis Accelerometer (First {duration_s}s)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')

    # 2. Gyroscope
    axes[1].plot(p_sub['time_s'], p_sub['gyro_x'], label='Gyro X (Pitch rate)', color='#9467bd', alpha=0.85)
    axes[1].plot(p_sub['time_s'], p_sub['gyro_y'], label='Gyro Y (Roll rate)', color='#8c564b', alpha=0.85)
    axes[1].plot(p_sub['time_s'], p_sub['gyro_z'], label='Gyro Z (Yaw rate)', color='#e377c2', alpha=0.85)
    axes[1].set_ylabel('Angular Vel ($rad/s$)')
    axes[1].set_title(f'[{trip_name}] Smartphone 3-Axis Gyroscope', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')

    # 3. Magnetometer
    axes[2].plot(p_sub['time_s'], p_sub['mag_x'], label='Mag X', color='#ff7f0e', alpha=0.85)
    axes[2].plot(p_sub['time_s'], p_sub['mag_y'], label='Mag Y', color='#17becf', alpha=0.85)
    axes[2].plot(p_sub['time_s'], p_sub['mag_z'], label='Mag Z', color='#bcbd22', alpha=0.85)
    axes[2].set_ylabel(r'Mag Field ($\mu T$)')
    axes[2].set_title(f'[{trip_name}] Smartphone 3-Axis Magnetometer', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')

    # 4. Forward Velocity: Smartphone GPS vs Vehicle True VBOX
    axes[3].plot(v_sub['time_rel_s'], v_sub['veh_speed_ms'], label='True Vehicle Speed (VBOX CAN, 10Hz)', color='#2ca02c', lw=2)
    if 'phone_speed_ms' in p_sub.columns:
        axes[3].plot(p_sub['time_s'], p_sub['phone_speed_ms'], label='Phone GPS Speed (1Hz coarse)', color='#d62728', lw=1.5, ls='--')
    axes[3].set_ylabel('Speed ($m/s$)')
    axes[3].set_xlabel('Time ($s$)')
    axes[3].set_title(f'[{trip_name}] Forward Speed: Ground Truth Reference vs Phone GPS', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    out_file = FIG_DIR / f"{trip_name}_sensor_timeseries.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved sensor timeseries: {out_file}")

def plot_2d_route_map(df_veh, trip_name="Vta02"):
    """Plots 2D geodetic trajectory of vehicle ground truth route."""
    valid_mask = (df_veh['veh_lat'].notnull()) & (df_veh['veh_lon'].notnull()) & (df_veh['veh_lat'] != 0)
    df_valid = df_veh[valid_mask]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df_valid['veh_lon'],
        df_valid['veh_lat'],
        c=df_valid['veh_speed_kmh'],
        cmap='viridis',
        s=8,
        alpha=0.85
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ground Truth Speed ($km/h$)', rotation=270, labelpad=15)

    # Start and End markers
    plt.plot(df_valid['veh_lon'].iloc[0], df_valid['veh_lat'].iloc[0], 'go', markersize=10, label='Start Point')
    plt.plot(df_valid['veh_lon'].iloc[-1], df_valid['veh_lat'].iloc[-1], 'rs', markersize=10, label='End Point')

    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title(f'[{trip_name}] Ground Truth Vehicle Trajectory (IO-VNBD Benchmark)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.tight_layout()

    out_file = FIG_DIR / f"{trip_name}_2d_route_map.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved 2D route map: {out_file}")

if __name__ == "__main__":
    df_phone, df_veh = load_trip("Vta02")
    plot_sensor_timeseries(df_phone, df_veh, "Vta02", duration_s=120)
    plot_2d_route_map(df_veh, "Vta02")
