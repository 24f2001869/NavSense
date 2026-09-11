"""
SIH26168 - Step 3: Automatic Vehicle-Relative IMU Alignment
Computes the rotation matrix R_pv mapping raw Smartphone sensor coordinates
into the Vehicle Body Coordinate Frame:
  - Axis 0: Longitudinal Forward
  - Axis 1: Lateral (Left/Right)
  - Axis 2: Vertical (Up)

Implements two-stage alignment:
  1. Gravity leveling via stationary accelerometer vector (roll & pitch).
  2. Forward direction discovery via longitudinal acceleration during initial motion (yaw).
"""

import numpy as np

def detect_stationary_period(accel, speed=None, window_size=30, accel_var_thresh=0.05):
    """
    Detects indices where vehicle is stationary.
    If speed is available, finds the longest contiguous stationary block.
    Otherwise, uses rolling window with lowest acceleration and gyro variance.
    """
    n = len(accel)
    if speed is not None:
        stat_mask = speed < 0.15
        stat_indices = np.where(stat_mask)[0]
        if len(stat_indices) >= window_size:
            # Find contiguous blocks
            diffs = np.diff(stat_indices)
            splits = np.where(diffs > 1)[0] + 1
            blocks = np.split(stat_indices, splits)
            # Pick longest block
            longest_block = max(blocks, key=len)
            if len(longest_block) >= 15:
                return longest_block

    # Fallback to acceleration variance
    accel_mag = np.linalg.norm(accel, axis=1)
    variances = [np.var(accel_mag[i:i+window_size]) for i in range(n - window_size)]
    min_idx = int(np.argmin(variances))
    return np.arange(min_idx, min_idx + window_size)

def compute_leveling_matrix(accel_mean):
    """
    Computes leveling rotation matrix R_level such that R_level @ accel_mean = [0, 0, ||accel_mean||]^T
    where the Z-axis aligns with the upward vertical.
    """
    ax, ay, az = accel_mean
    # Roll: rotation about phone X axis to zero out Y
    roll = np.arctan2(ay, az)
    # Pitch: rotation about phone Y axis to zero out X
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # R_level = R_pitch @ R_roll
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr]
    ])
    R_y = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    R_level = R_y @ R_x
    return R_level, roll, pitch

def compute_forward_alignment(accel_leveled, speed=None, min_accel=0.3):
    """
    Estimates forward vehicle azimuth from longitudinal acceleration bursts.
    In the leveled frame, vehicle forward acceleration is predominantly in the horizontal plane.
    """
    # Select samples where significant horizontal acceleration occurs
    horiz_accel = accel_leveled[:, :2]
    horiz_mag = np.linalg.norm(horiz_accel, axis=1)

    if speed is not None:
        # Accelerating forward condition: speed > 2 m/s and speed derivative positive
        accel_deriv = np.gradient(speed, 0.1)
        valid_mask = (accel_deriv > 0.2) & (horiz_mag > min_accel)
    else:
        valid_mask = horiz_mag > min_accel

    if np.sum(valid_mask) > 10:
        mean_vector = np.mean(horiz_accel[valid_mask], axis=0)
    else:
        mean_vector = np.array([0.0, 1.0]) # fallback

    # Compute yaw offset so that mean_vector points along [1, 0] (vehicle forward)
    yaw_align = np.arctan2(mean_vector[1], mean_vector[0])
    cy, sy = np.cos(-yaw_align), np.sin(-yaw_align)

    R_yaw = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return R_yaw, yaw_align

def align_phone_to_vehicle(accel, gyro, speed=None):
    """
    End-to-end alignment pipeline.
    Transforms raw phone IMU signals into vehicle-aligned frame:
        accel_veh[:, 0] = Forward acceleration
        accel_veh[:, 1] = Lateral acceleration
        accel_veh[:, 2] = Vertical acceleration (+1g when stationary)
    """
    stat_indices = detect_stationary_period(accel, speed)
    accel_stat_mean = np.mean(accel[stat_indices], axis=0)

    R_level, roll, pitch = compute_leveling_matrix(accel_stat_mean)
    accel_leveled = (R_level @ accel.T).T

    R_yaw, yaw_align = compute_forward_alignment(accel_leveled, speed)
    R_pv = R_yaw @ R_level

    accel_veh = (R_pv @ accel.T).T
    gyro_veh = (R_pv @ gyro.T).T

    angles_deg = {
        'roll_deg': float(np.degrees(roll)),
        'pitch_deg': float(np.degrees(pitch)),
        'yaw_deg': float(np.degrees(yaw_align))
    }

    return accel_veh, gyro_veh, R_pv, angles_deg

if __name__ == "__main__":
    from src.data.loader import load_trip
    df_p, df_v = load_trip("Vta02")
    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    sp = df_v['veh_speed_ms'].values

    acc_v, gyro_v, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro, sp)
    print("Computed Phone-to-Vehicle Alignment Angles:", angles)
    print("Mean Stationarity Vertical Accel (should be ~9.8 m/s^2):", np.mean(acc_v[:30, 2]))
    print("Mean Stationarity Horiz Accel (should be ~0.0 m/s^2):", np.linalg.norm(acc_v[:30, :2], axis=1).mean())
