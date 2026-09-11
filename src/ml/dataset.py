"""
SIH26168 - Step 6: IMU Window Feature Extraction for AI Velocity Estimation
Extracts statistical, kinematic, and vibration features over sliding temporal windows
to predict true vehicle forward velocity without GNSS.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle

def extract_window_features(accel_veh, gyro_veh, speed_ms, window_size=10, step_size=1):
    """
    Extracts multi-domain features over sliding windows of IMU data.
    window_size: 10 samples = 1.0 second at 10 Hz
    step_size: 1 sample = 0.1 second step
    """
    n = len(accel_veh)
    X = []
    y = []

    a_norm = np.linalg.norm(accel_veh, axis=1)
    g_norm = np.linalg.norm(gyro_veh, axis=1)

    for i in range(0, n - window_size + 1, step_size):
        w_acc = accel_veh[i : i + window_size]      # (W, 3)
        w_gyro = gyro_veh[i : i + window_size]      # (W, 3)
        w_anorm = a_norm[i : i + window_size]       # (W,)
        w_gnorm = g_norm[i : i + window_size]       # (W,)

        features = [
            # 1. Acceleration Means
            np.mean(w_acc[:, 0]), # Forward
            np.mean(w_acc[:, 1]), # Lateral
            np.mean(w_acc[:, 2]), # Vertical
            np.mean(w_anorm),

            # 2. Acceleration Stds (vibration & engine excitement)
            np.std(w_acc[:, 0]),
            np.std(w_acc[:, 1]),
            np.std(w_acc[:, 2]),
            np.std(w_anorm),

            # 3. Acceleration Extrema & Energy
            np.max(w_acc[:, 0]) - np.min(w_acc[:, 0]),
            np.max(w_acc[:, 2]) - np.min(w_acc[:, 2]),
            np.sqrt(np.mean(w_acc[:, 0]**2)), # RMS forward
            np.sqrt(np.mean(w_acc[:, 2]**2)), # RMS vertical

            # 4. Gyroscope Means & Stds
            np.mean(w_gyro[:, 0]),
            np.mean(w_gyro[:, 1]),
            np.mean(w_gyro[:, 2]), # Yaw rate mean
            np.std(w_gyro[:, 2]),  # Yaw rate std
            np.mean(w_gnorm),
            np.std(w_gnorm),

            # 5. Frequency Energy (Variance of high-pass diff)
            np.var(np.diff(w_acc[:, 0])),
            np.var(np.diff(w_acc[:, 2]))
        ]

        # Target is ground truth velocity at the end of window
        target_speed = speed_ms[i + window_size - 1]

        X.append(features)
        y.append(target_speed)

    feature_names = [
        'acc_fwd_mean', 'acc_lat_mean', 'acc_vert_mean', 'acc_norm_mean',
        'acc_fwd_std', 'acc_lat_std', 'acc_vert_std', 'acc_norm_std',
        'acc_fwd_range', 'acc_vert_range', 'acc_fwd_rms', 'acc_vert_rms',
        'gyro_x_mean', 'gyro_y_mean', 'gyro_z_mean', 'gyro_z_std', 'gyro_norm_mean', 'gyro_norm_std',
        'acc_fwd_diff_var', 'acc_vert_diff_var'
    ]

    return np.array(X), np.array(y), feature_names

def load_and_preprocess_dataset(trip_names, window_size=10, step_size=1):
    """
    Loads one or more trips, applies coordinate alignment,
    and returns concatenated feature matrix X and target y.
    """
    X_all, y_all = [], []
    for trip in trip_names:
        df_p, df_v = load_trip(trip)
        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        speed = df_v['veh_speed_ms'].values

        acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
        X_trip, y_trip, names = extract_window_features(acc_v, gyro_v, speed, window_size, step_size)
        X_all.append(X_trip)
        y_all.append(y_trip)

    return np.vstack(X_all), np.concatenate(y_all), names
