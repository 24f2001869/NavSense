"""
SIH26168 - Stage C5.2-C0: Longitudinal-Frame Kinematic Integrity Audit
Script: experiments/audit_kinematics_c5_2c0.py

Comprehensive offline physical audit across Vta02, Vta03, and Vta04:
  1. Compares candidate coordinate transformations (Phone, Leveled, R_pv, Optimal Azimuth)
  2. Evaluates correlation, scale factor, and sign consistency between IMU acceleration and VBOX dv/dt
  3. Verifies gyroscope yaw rate mapping (resolving the C3 gyro_x vs gyro_z question)
  
STRICT RULE: VBOX dv/dt and yaw rate are used solely as offline diagnostic references.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import compute_leveling_matrix, align_phone_to_vehicle

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float, fs: float = 10.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def compute_optimal_horizontal_azimuth(ax_level: np.ndarray, ay_level: np.ndarray, dv_dt_ref: np.ndarray):
    """
    Sweeps azimuth theta in [-180, 180] deg to find the forward axis that maximizes
    correlation with reference dv/dt.
    """
    angles_deg = np.linspace(-180, 180, 361)
    corrs = []
    for deg in angles_deg:
        rad = np.radians(deg)
        a_proj = ax_level * np.cos(rad) + ay_level * np.sin(rad)
        r = float(np.corrcoef(a_proj, dv_dt_ref)[0, 1])
        corrs.append(r)
    corrs = np.array(corrs)
    best_idx = np.argmax(corrs)
    best_angle_deg = float(angles_deg[best_idx])
    max_corr = float(corrs[best_idx])
    return best_angle_deg, max_corr, angles_deg, corrs


def audit_trip_longitudinal_kinematics(trip_name: str) -> dict:
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    dt = 0.1
    time_s = df_p['time_s'].to_numpy()

    # Ground truth speed and acceleration
    v_true = df_v['veh_speed_ms'].to_numpy()
    dv_dt_raw = np.gradient(v_true, dt)
    dv_dt_filt = butter_lowpass_filter(dv_dt_raw, cutoff_hz=0.5, fs=10.0)

    # Ground truth vehicle yaw rate (rad/s)
    if 'yaw_rate_degs' in df_v.columns:
        yaw_rate_true_rads = np.radians(df_v['yaw_rate_degs'].to_numpy())
    else:
        heading_rad = np.unwrap(np.radians(df_v['veh_heading_deg'].to_numpy()))
        yaw_rate_true_rads = np.gradient(heading_rad, dt)
    yaw_rate_true_filt = butter_lowpass_filter(yaw_rate_true_rads, cutoff_hz=0.5, fs=10.0)

    # Raw Phone IMU
    ax_p = df_p['accel_x'].to_numpy()
    ay_p = df_p['accel_y'].to_numpy()
    az_p = df_p['accel_z'].to_numpy()

    gx_p = df_p['gyro_x'].to_numpy()
    gy_p = df_p['gyro_y'].to_numpy()
    gz_p = df_p['gyro_z'].to_numpy()

    acc_p_stack = np.vstack([ax_p, ay_p, az_p])
    gyro_p_stack = np.vstack([gx_p, gy_p, gz_p])

    # 1. Candidate 1: Raw Phone Axes
    corr_ax_p = float(np.corrcoef(butter_lowpass_filter(ax_p, 0.5), dv_dt_filt)[0, 1])
    corr_ay_p = float(np.corrcoef(butter_lowpass_filter(ay_p, 0.5), dv_dt_filt)[0, 1])
    corr_az_p = float(np.corrcoef(butter_lowpass_filter(az_p, 0.5), dv_dt_filt)[0, 1])

    corr_gx_p = float(np.corrcoef(butter_lowpass_filter(gx_p, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gy_p = float(np.corrcoef(butter_lowpass_filter(gy_p, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gz_p = float(np.corrcoef(butter_lowpass_filter(gz_p, 0.5), yaw_rate_true_filt)[0, 1])

    # 2. Candidate 2: Static Gravity-Leveled
    if 'grav_x' in df_p.columns:
        g_vec = df_p[['grav_x', 'grav_y', 'grav_z']].values.mean(axis=0)
    else:
        g_vec = np.array([ax_p.mean(), ay_p.mean(), az_p.mean()])
    R_level, roll_rad, pitch_rad = compute_leveling_matrix(g_vec)
    acc_l_stack = R_level @ acc_p_stack
    gyro_l_stack = R_level @ gyro_p_stack

    ax_l, ay_l, az_l = acc_l_stack[0], acc_l_stack[1], acc_l_stack[2]
    gx_l, gy_l, gz_l = gyro_l_stack[0], gyro_l_stack[1], gyro_l_stack[2]

    ax_l_filt = butter_lowpass_filter(ax_l, 0.5)
    ay_l_filt = butter_lowpass_filter(ay_l, 0.5)
    az_l_filt = butter_lowpass_filter(az_l, 0.5)

    corr_ax_l = float(np.corrcoef(ax_l_filt, dv_dt_filt)[0, 1])
    corr_ay_l = float(np.corrcoef(ay_l_filt, dv_dt_filt)[0, 1])
    corr_az_l = float(np.corrcoef(az_l_filt, dv_dt_filt)[0, 1])

    corr_gx_l = float(np.corrcoef(butter_lowpass_filter(gx_l, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gy_l = float(np.corrcoef(butter_lowpass_filter(gy_l, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gz_l = float(np.corrcoef(butter_lowpass_filter(gz_l, 0.5), yaw_rate_true_filt)[0, 1])

    # 3. Candidate 3: Pre-calibrated R_pv from C1/C3
    acc_v, gyro_v, R_pv, angles_calib = align_phone_to_vehicle(acc_p_stack.T, gyro_p_stack.T, v_true)
    ax_v = acc_v[:, 0]
    ay_v = acc_v[:, 1]
    az_v = acc_v[:, 2]
    gx_v = gyro_v[:, 0]
    gy_v = gyro_v[:, 1]
    gz_v = gyro_v[:, 2]

    ax_v_filt = butter_lowpass_filter(ax_v, 0.5)
    ay_v_filt = butter_lowpass_filter(ay_v, 0.5)
    az_v_filt = butter_lowpass_filter(az_v, 0.5)

    corr_ax_v = float(np.corrcoef(ax_v_filt, dv_dt_filt)[0, 1])
    corr_ay_v = float(np.corrcoef(ay_v_filt, dv_dt_filt)[0, 1])
    corr_az_v = float(np.corrcoef(az_v_filt, dv_dt_filt)[0, 1])

    corr_gx_v = float(np.corrcoef(butter_lowpass_filter(gx_v, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gy_v = float(np.corrcoef(butter_lowpass_filter(gy_v, 0.5), yaw_rate_true_filt)[0, 1])
    corr_gz_v = float(np.corrcoef(butter_lowpass_filter(gz_v, 0.5), yaw_rate_true_filt)[0, 1])

    # 4. Candidate 4: Optimal Horizontal Azimuth theta*
    best_theta_deg, max_corr_theta, sweep_angles, sweep_corrs = compute_optimal_horizontal_azimuth(
        ax_l_filt, ay_l_filt, dv_dt_filt
    )
    rad_opt = np.radians(best_theta_deg)
    ax_opt_filt = ax_l_filt * np.cos(rad_opt) + ay_l_filt * np.sin(rad_opt)

    # Scale factor and sign consistency for best candidate vs dv/dt
    scale_factor_opt = float(np.std(ax_opt_filt) / (np.std(dv_dt_filt) + 1e-12))
    scale_factor_raw = float(np.std(ax_p) / (np.std(dv_dt_filt) + 1e-12))

    accel_mask = dv_dt_filt > 0.3
    brake_mask = dv_dt_filt < -0.3
    sign_match_accel = float(np.mean(np.sign(ax_opt_filt[accel_mask] - np.mean(ax_opt_filt)) == np.sign(dv_dt_filt[accel_mask]))) if np.sum(accel_mask) > 0 else 0.0
    sign_match_brake = float(np.mean(np.sign(ax_opt_filt[brake_mask] - np.mean(ax_opt_filt)) == np.sign(dv_dt_filt[brake_mask]))) if np.sum(brake_mask) > 0 else 0.0

    return {
        'trip': trip_name,
        'duration_s': float(time_s[-1] - time_s[0]),
        'sample_count': int(n),
        'mean_speed_ms': float(np.mean(v_true)),
        'std_speed_ms': float(np.std(v_true)),
        'std_dv_dt_ms2': float(np.std(dv_dt_filt)),
        'angles_calib': angles_calib,
        'R_pv_calib': R_pv.tolist(),
        'optimal_azimuth_deg': best_theta_deg,
        'max_azimuth_corr': max_corr_theta,
        'scale_factor_opt_vs_dvdt': scale_factor_opt,
        'scale_factor_raw_vs_dvdt': scale_factor_raw,
        'sign_consistency': {
            'accel_bursts': sign_match_accel,
            'braking': sign_match_brake
        },
        'correlations': {
            'phone_accel_vs_dvdt': {'ax': corr_ax_p, 'ay': corr_ay_p, 'az': corr_az_p},
            'phone_gyro_vs_yawrate': {'gx': corr_gx_p, 'gy': corr_gy_p, 'gz': corr_gz_p},
            'leveled_accel_vs_dvdt': {'ax': corr_ax_l, 'ay': corr_ay_l, 'az': corr_az_l},
            'leveled_gyro_vs_yawrate': {'gx': corr_gx_l, 'gy': corr_gy_l, 'gz': corr_gz_l},
            'calib_v_accel_vs_dvdt': {'ax': corr_ax_v, 'ay': corr_ay_v, 'az': corr_az_v},
            'calib_v_gyro_vs_yawrate': {'gx': corr_gx_v, 'gy': corr_gy_v, 'gz': corr_gz_v}
        },
        'time_s': time_s,
        'v_true': v_true,
        'dv_dt_filt': dv_dt_filt,
        'yaw_rate_true_filt': yaw_rate_true_filt,
        'ax_opt_filt': ax_opt_filt,
        'ax_p_filt': butter_lowpass_filter(ax_p, 0.5),
        'ax_v_filt': ax_v_filt,
        'gz_v_filt': butter_lowpass_filter(gz_v, 0.5),
        'gx_p_filt': butter_lowpass_filter(gx_p, 0.5),
        'gz_p_filt': butter_lowpass_filter(gz_p, 0.5),
        'sweep_angles': sweep_angles,
        'sweep_corrs': sweep_corrs
    }


def main():
    print("=" * 90)
    print("STAGE C5.2-C0: LONGITUDINAL-FRAME & GYROSCOPE KINEMATIC INTEGRITY AUDIT")
    print("=" * 90)

    trips = ["Vta02", "Vta03", "Vta04"]
    results = {}

    for t in trips:
        print(f"\nAuditing Trip: {t}...")
        res = audit_trip_longitudinal_kinematics(t)
        results[t] = res

        corr = res['correlations']
        print(f"  Calibrated Angles: Roll={res['angles_calib']['roll_deg']:.2f} deg, Pitch={res['angles_calib']['pitch_deg']:.2f} deg, Yaw={res['angles_calib']['yaw_deg']:.2f} deg")
        print(f"  Optimal Forward Azimuth theta*: {res['optimal_azimuth_deg']:+.1f} deg (Peak r = {res['max_azimuth_corr']:+.3f})")
        print(f"  Acceleration Correlations with dv/dt:")
        print(f"    - Raw Phone:   ax={corr['phone_accel_vs_dvdt']['ax']:+.3f}, ay={corr['phone_accel_vs_dvdt']['ay']:+.3f}, az={corr['phone_accel_vs_dvdt']['az']:+.3f}")
        print(f"    - Leveled:     ax={corr['leveled_accel_vs_dvdt']['ax']:+.3f}, ay={corr['leveled_accel_vs_dvdt']['ay']:+.3f}, az={corr['leveled_accel_vs_dvdt']['az']:+.3f}")
        print(f"    - Calib (Rpv): ax={corr['calib_v_accel_vs_dvdt']['ax']:+.3f}, ay={corr['calib_v_accel_vs_dvdt']['ay']:+.3f}, az={corr['calib_v_accel_vs_dvdt']['az']:+.3f}")
        print(f"  Gyroscope Correlations with Vehicle Yaw Rate:")
        print(f"    - Raw Phone:   gx={corr['phone_gyro_vs_yawrate']['gx']:+.3f}, gy={corr['phone_gyro_vs_yawrate']['gy']:+.3f}, gz={corr['phone_gyro_vs_yawrate']['gz']:+.3f}")
        print(f"    - Calib (Rpv): gx={corr['calib_v_gyro_vs_yawrate']['gx']:+.3f}, gy={corr['calib_v_gyro_vs_yawrate']['gy']:+.3f}, gz={corr['calib_v_gyro_vs_yawrate']['gz']:+.3f}")
        print(f"  Scale Factor (std(a_opt) / std(dv/dt)): {res['scale_factor_opt_vs_dvdt']:.2f}")
        print(f"  Sign Consistency: Accel bursts={res['sign_consistency']['accel_bursts']*100:.1f}%, Braking={res['sign_consistency']['braking']*100:.1f}%")

    # Save summary JSON (excluding numpy arrays)
    summary_json = {}
    for t in trips:
        d = dict(results[t])
        for k in ['time_s', 'v_true', 'dv_dt_filt', 'yaw_rate_true_filt', 'ax_opt_filt', 'ax_p_filt', 'ax_v_filt', 'gz_v_filt', 'gx_p_filt', 'gz_p_filt', 'sweep_angles', 'sweep_corrs']:
            del d[k]
        summary_json[t] = d
    json_path = RES_DIR / "c5_2c0_kinematic_audit.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved kinematic audit metrics to: {json_path}")

    # Publication-Grade Diagnostic Figure: 3 trips x 3 rows
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    for idx, t in enumerate(trips):
        res = results[t]
        t_arr = res['time_s']

        # Column 1: Azimuth sweep r(theta, dv/dt)
        axs[0, idx].plot(res['sweep_angles'], res['sweep_corrs'], 'b-', lw=1.5)
        axs[0, idx].axvline(res['optimal_azimuth_deg'], color='r', linestyle='--', label=f"theta* = {res['optimal_azimuth_deg']:+.1f} deg (r={res['max_azimuth_corr']:+.2f})")
        axs[0, idx].axvline(-res['angles_calib']['yaw_deg'], color='g', linestyle=':', label=f"Calib Yaw = {-res['angles_calib']['yaw_deg']:+.1f} deg")
        axs[0, idx].axhline(0, color='k', linestyle='-', alpha=0.3)
        axs[0, idx].set_title(f"{t}: Horizontal Forward Azimuth Sweep", fontweight='bold')
        axs[0, idx].set_xlabel("Azimuth Angle (deg)")
        axs[0, idx].set_ylabel("Correlation r(a_proj, dv/dt)")
        axs[0, idx].grid(True, alpha=0.3)
        axs[0, idx].legend(fontsize=8, loc='lower center')

        # Column 2: Longitudinal Acceleration vs dv/dt
        axs[1, idx].plot(t_arr, res['dv_dt_filt'], 'k-', lw=1.5, label='VBOX dv/dt')
        axs[1, idx].plot(t_arr, res['ax_opt_filt'] - np.mean(res['ax_opt_filt']), 'r--', lw=1.2, label=f"IMU a_long (r={res['max_azimuth_corr']:+.2f})")
        axs[1, idx].set_title(f"{t}: IMU a_long vs VBOX dv/dt", fontweight='bold')
        axs[1, idx].set_xlabel("Time (s)")
        axs[1, idx].set_ylabel("Acceleration (m/s^2)")
        axs[1, idx].set_ylim([-3.0, 3.0])
        axs[1, idx].grid(True, alpha=0.3)
        axs[1, idx].legend(fontsize=8, loc='upper right')

        # Column 3: Gyroscope vs Vehicle Yaw Rate
        axs[2, idx].plot(t_arr, np.degrees(res['yaw_rate_true_filt']), 'k-', lw=1.5, label='CAN/VBOX Yaw Rate')
        axs[2, idx].plot(t_arr, np.degrees(res['gz_v_filt']), 'g--', lw=1.2, label=f"gz_v (r={res['correlations']['calib_v_gyro_vs_yawrate']['gz']:+.2f})")
        axs[2, idx].plot(t_arr, np.degrees(res['gx_p_filt']), 'm:', lw=1.0, alpha=0.7, label=f"Raw gx_p (r={res['correlations']['phone_gyro_vs_yawrate']['gx']:+.2f})")
        axs[2, idx].set_title(f"{t}: Gyroscope vs Vehicle Yaw Rate", fontweight='bold')
        axs[2, idx].set_xlabel("Time (s)")
        axs[2, idx].set_ylabel("Yaw Rate (deg/s)")
        axs[2, idx].grid(True, alpha=0.3)
        axs[2, idx].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig_path = FIG_DIR / "c5_2c0_longitudinal_correlation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved publication diagnostic figure to: {fig_path}")


if __name__ == "__main__":
    main()
