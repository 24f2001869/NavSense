"""
SIH26168 - Stage C5.1: Causal Feature Extraction for AI Forward-Speed Estimation
Module: src/ml/speed_features.py

Builds a transparent, causal, multi-domain feature set from raw smartphone IMU channels.
Strictly adheres to causal temporal windows [t - W + 1, ..., t].
No VBOX, GNSS, or future information is used in the feature extraction.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.data.loader import load_trip


def extract_causal_features_for_trip(
    df_phone: pd.DataFrame,
    df_veh: pd.DataFrame,
    window_size: int = 20,
    stride: int = 1,
    frame: str = "phone"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Extracts causal window features from smartphone IMU data in either raw phone frame
    (Representation A) or sensor-only gravity-leveled frame (Representation B).
    Associates each window with ground-truth scalar VBOX vehicle speed magnitude at the trailing edge.

    Args:
        df_phone: DataFrame with clean smartphone channels (must include accel, gyro, and grav).
        df_veh: DataFrame with synchronized VBOX vehicle channels.
        window_size: Number of samples in causal window (e.g. 20 samples = 2.0 s at 10 Hz).
        stride: Stride in samples between consecutive windows (e.g. 1 sample = 0.1 s).
        frame: "phone" (Raw Phone-Frame, C5.1) or "gravity_leveled" (Sensor-Only Leveling, C5.2-A).

    Returns:
        X: Feature matrix of shape (N_windows, 72).
        y: Target speed array of shape (N_windows,) in m/s (scalar vehicle speed magnitude).
        timestamps: Array of timestamps corresponding to window end (t).
        feature_names: List of descriptive feature names.
        meta: Dictionary containing leveling metadata (angles, R_level).
    """
    from src.preprocessing.gravity_alignment import compute_leveling_matrix

    # 1. Extract raw smartphone sensor channels
    ax_p = df_phone['accel_x'].values.astype(np.float64)
    ay_p = df_phone['accel_y'].values.astype(np.float64)
    az_p = df_phone['accel_z'].values.astype(np.float64)

    gx_p = df_phone['gyro_x'].values.astype(np.float64)
    gy_p = df_phone['gyro_y'].values.astype(np.float64)
    gz_p = df_phone['gyro_z'].values.astype(np.float64)

    meta = {'frame': frame}

    if frame.lower() == "gravity_leveled":
        # Compute static leveling rotation from smartphone-only gravity vector
        if 'grav_x' in df_phone.columns and 'grav_y' in df_phone.columns and 'grav_z' in df_phone.columns:
            g_vec = df_phone[['grav_x', 'grav_y', 'grav_z']].values.mean(axis=0)
            g_source = "df_phone gravity sensor mean"
        else:
            from src.preprocessing.gravity_alignment import detect_stationary_period
            acc_stack_raw = np.vstack([ax_p, ay_p, az_p]).T
            stat_idx = detect_stationary_period(acc_stack_raw, speed=None)
            g_vec = np.mean(acc_stack_raw[stat_idx], axis=0)
            g_source = "stationary accelerometer mean"

        R_level, roll_rad, pitch_rad = compute_leveling_matrix(g_vec)
        meta['R_level'] = R_level
        meta['roll_deg'] = float(np.degrees(roll_rad))
        meta['pitch_deg'] = float(np.degrees(pitch_rad))
        meta['gravity_source'] = g_source

        # Rotate 3-axis accelerometer and gyroscope into leveled frame
        acc_p_stack = np.vstack([ax_p, ay_p, az_p])  # (3, N)
        gyro_p_stack = np.vstack([gx_p, gy_p, gz_p]) # (3, N)

        acc_l_stack = R_level @ acc_p_stack
        gyro_l_stack = R_level @ gyro_p_stack

        ax_l, ay_l, az_l = acc_l_stack[0], acc_l_stack[1], acc_l_stack[2]
        gx_l, gy_l, gz_l = gyro_l_stack[0], gyro_l_stack[1], gyro_l_stack[2]

        # Horizontal acceleration magnitude in leveled plane
        a_horiz_mag = np.sqrt(ax_l**2 + ay_l**2)
        g_level_mag = np.sqrt(gx_l**2 + gy_l**2 + gz_l**2)

        channels = {
            'accel_x_level': ax_l,
            'accel_y_level': ay_l,
            'accel_z_level': az_l,
            'accel_horizontal_magnitude': a_horiz_mag,
            'gyro_x_level': gx_l,
            'gyro_y_level': gy_l,
            'gyro_z_level': gz_l,
            'gyro_level_magnitude': g_level_mag
        }
    elif frame.lower() == "phone":
        a_mag = np.sqrt(ax_p**2 + ay_p**2 + az_p**2)
        g_mag = np.sqrt(gx_p**2 + gy_p**2 + gz_p**2)
        channels = {
            'acc_x': ax_p,
            'acc_y': ay_p,
            'acc_z': az_p,
            'acc_mag': a_mag,
            'gyro_x': gx_p,
            'gyro_y': gy_p,
            'gyro_z': gz_p,
            'gyro_mag': g_mag
        }
    else:
        raise ValueError(f"Unknown frame '{frame}'. Must be 'phone' or 'gravity_leveled'.")

    # Target: Scalar VBOX vehicle speed magnitude (m/s)
    vbox_speed = df_veh['veh_speed_ms'].values.astype(np.float64)
    time_s = df_phone['time_s'].values.astype(np.float64)

    n_samples = len(ax_p)

    # Prepare feature names
    stat_names = [
        'mean', 'std', 'rms', 'min', 'max', 'ptp', 'abs_mean', 'energy', 'diff_var'
    ]
    feature_names = []
    for ch_name in channels.keys():
        for s_name in stat_names:
            feature_names.append(f"{ch_name}_{s_name}")

    X_list = []
    y_list = []
    t_list = []

    # Causal sliding window: window covers [i, i + window_size - 1]
    # Prediction time is at index: k_pred = i + window_size - 1
    for i in range(0, n_samples - window_size + 1, stride):
        k_end = i + window_size
        k_pred = k_end - 1

        window_features = []
        for ch_name, sig in channels.items():
            w = sig[i:k_end]

            w_mean = float(np.mean(w))
            w_std = float(np.std(w))
            w_rms = float(np.sqrt(np.mean(w**2)))
            w_min = float(np.min(w))
            w_max = float(np.max(w))
            w_ptp = float(w_max - w_min)
            w_abs_mean = float(np.mean(np.abs(w)))
            w_energy = float(np.mean(w**2))
            w_diff_var = float(np.var(np.diff(w))) if len(w) > 1 else 0.0

            window_features.extend([
                w_mean, w_std, w_rms, w_min, w_max, w_ptp, w_abs_mean, w_energy, w_diff_var
            ])

        X_list.append(window_features)
        y_list.append(vbox_speed[k_pred])
        t_list.append(time_s[k_pred])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    timestamps = np.array(t_list, dtype=np.float64)

    return X, y, timestamps, feature_names, meta


def load_and_extract_speed_dataset(
    trip_name: str,
    window_size: int = 20,
    stride: int = 1,
    frame: str = "phone"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads trip, audits time alignment, and extracts causal features in phone or gravity_leveled frame.
    """
    df_p, df_v = load_trip(trip_name)

    # Time synchronization inspection
    dt_p = np.diff(df_p['time_s'].values)
    dt_v = np.diff(df_v['time_s'].values) if 'time_s' in df_v.columns else dt_p

    sync_info = {
        'trip_name': trip_name,
        'n_samples_phone': len(df_p),
        'n_samples_veh': len(df_v),
        'dt_mean_phone': float(np.mean(dt_p)),
        'dt_std_phone': float(np.std(dt_p)),
        'duration_s': float(df_p['time_s'].iloc[-1] - df_p['time_s'].iloc[0]),
        'min_vbox_speed': float(df_v['veh_speed_ms'].min()),
        'max_vbox_speed': float(df_v['veh_speed_ms'].max()),
        'mean_vbox_speed': float(df_v['veh_speed_ms'].mean()),
        'frame': frame
    }

    X, y, timestamps, feature_names, meta = extract_causal_features_for_trip(
        df_p, df_v, window_size=window_size, stride=stride, frame=frame
    )
    sync_info.update(meta)

    return X, y, timestamps, feature_names, sync_info


if __name__ == "__main__":
    print("Testing speed feature extraction on Vta02...")
    X, y, t, names, sync = load_and_extract_speed_dataset("Vta02", window_size=20, stride=1)
    print(f"Vta02 Extracted: X={X.shape}, y={y.shape}, Features={len(names)}")
    print(f"Sync Info: {sync}")
