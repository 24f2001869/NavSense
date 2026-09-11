"""
Dual-Representation Dataset Builder for IO-VNBD.

Generates:
1. Branch A (Engineered Features):
   - 2.0-second sliding windows (20 samples @ 10 Hz)
   - Statistical features per IMU channel (mean, std, rms, min, max, jerk, energy)
   - Used for classical baselines (Ridge, Random Forest)
2. Branch B (Raw Temporal Sequences):
   - 10.0-second sliding windows (100 samples @ 10 Hz)
   - 6-channel raw IMU tensors [a_lin_x, a_lin_y, a_lin_z, gyro_yaw, gyro_pitch, gyro_roll]
   - Used for temporal deep learning (GRU, TCN, AVNet)
3. Aligned Ground Truth Speed:
   - True zero clamping during verified stops (brake + wheel speed == 0)
   - Audited CAN velocity (m/s)
4. Offline Regime Labels:
   - Synchronized regime mask per window for granular post-hoc slicing.

Trajectory-Aware Splitting:
- Split A (Unseen Trips): 70% train, 10% val, 20% test across trips within a driver category.
- Split B (Unseen Driver): Train on driver subset, evaluate out-of-domain on unseen drivers.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.ml.regime_segmentation import extract_regime_masks


def compute_window_features(window_imu: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Computes ~56 statistical and kinematic features from a (W, 8) IMU window.
    Channels: [a_lin_x, a_lin_y, a_lin_z, a_lin_mag, gyro_yaw, gyro_pitch, gyro_roll, gyro_mag]
    """
    features = []
    w_len = window_imu.shape[0]
    
    for c in range(window_imu.shape[1]):
        col = window_imu[:, c]
        mean_val = np.mean(col)
        std_val = np.std(col)
        rms_val = np.sqrt(np.mean(col**2))
        min_val = np.min(col)
        max_val = np.max(col)
        range_val = max_val - min_val
        energy_val = np.sum(col**2) / w_len
        
        # Jerk (derivative)
        diff_val = np.diff(col) / dt if w_len > 1 else np.array([0.0])
        jerk_mean = np.mean(np.abs(diff_val))
        jerk_std = np.std(diff_val)
        
        features.extend([mean_val, std_val, rms_val, min_val, max_val, range_val, energy_val, jerk_mean, jerk_std])
        
    return np.array(features, dtype=np.float32)


class IOVNBDDatasetBuilder:
    """Builds and manages Branch A (features) and Branch B (sequences) datasets."""
    
    def __init__(
        self,
        data_roots: List[Path],
        window_size_sec_branch_a: float = 2.0,
        window_stride_sec_branch_a: float = 0.5,
        window_size_sec_branch_b: float = 10.0,
        window_stride_sec_branch_b: float = 1.0,
        hz: float = 10.0,
    ):
        self.data_roots = data_roots
        self.hz = hz
        self.w_a_samples = int(window_size_sec_branch_a * hz)
        self.s_a_samples = int(window_stride_sec_branch_a * hz)
        self.w_b_samples = int(window_size_sec_branch_b * hz)
        self.s_b_samples = int(window_stride_sec_branch_b * hz)
        
    def discover_trips(self) -> List[Dict]:
        """Scans all data roots for non-LFS paired CSV files."""
        trips = []
        for root_dir in self.data_roots:
            if not root_dir.exists():
                continue
            for root, dirs, files in os.walk(root_dir):
                s_files = [f for f in files if f.startswith('S-') and f.endswith('.csv')]
                v_files = [f for f in files if f.startswith('V-') and f.endswith('.csv')]
                if s_files and v_files:
                    s_path = Path(root) / s_files[0]
                    v_path = Path(root) / v_files[0]
                    # Filter out Git LFS pointer stubs (< 1000 bytes)
                    if s_path.stat().st_size > 1000 and v_path.stat().st_size > 1000:
                        rel = Path(root).relative_to(root_dir)
                        driver = rel.parts[0] if len(rel.parts) > 1 else 'Driver'
                        trips.append({
                            'driver': driver,
                            'trip_name': rel.name,
                            's_path': s_path,
                            'v_path': v_path,
                        })
        return trips
        
    def load_clean_trip(self, trip_info: Dict) -> pd.DataFrame:
        """Loads and prepares trip telemetry with ground-truth zero clamping."""
        s_df = pd.read_csv(trip_info['s_path'], encoding='latin1', skipinitialspace=True)
        v_df = pd.read_csv(trip_info['v_path'], encoding='latin1', skipinitialspace=True)
        
        s_df.columns = [c.strip() for c in s_df.columns]
        v_df.columns = [c.strip() for c in v_df.columns]
        
        min_len = min(len(s_df), len(v_df))
        s_df = s_df.iloc[:min_len].copy()
        v_df = v_df.iloc[:min_len].copy()
        
        # Column lookups
        ax = [c for c in s_df.columns if 'ACCELEROMETER X' in c][0]
        ay = [c for c in s_df.columns if 'ACCELEROMETER Y' in c][0]
        az = [c for c in s_df.columns if 'ACCELEROMETER Z' in c][0]
        gx = [c for c in s_df.columns if 'GRAVITY X' in c][0]
        gy = [c for c in s_df.columns if 'GRAVITY Y' in c][0]
        gz = [c for c in s_df.columns if 'GRAVITY Z' in c][0]
        gy_yaw = [c for c in s_df.columns if 'GYROSCOPE Yaw' in c][0]
        gy_pitch = [c for c in s_df.columns if 'GYROSCOPE Pitch' in c][0]
        gy_roll = [c for c in s_df.columns if 'GYROSCOPE Roll' in c][0]
        gps_spd = [c for c in s_df.columns if 'GPS SPEED' in c][0]
        
        v_spd = [c for c in v_df.columns if c == 'Velocity (km/hr)'][0]
        w_rl = [c for c in v_df.columns if 'Rear Left' in c][0]
        w_rr = [c for c in v_df.columns if 'Rear Right' in c][0]
        lon_acc = [c for c in v_df.columns if 'Longitudinal Acceleration' in c][0]
        lat_acc = [c for c in v_df.columns if 'Lateral Acceleration' in c][0]
        yaw_r = [c for c in v_df.columns if 'Yaw Rate' in c][0]
        steer = [c for c in v_df.columns if 'Steering Angle' in c][0]
        brake_press = [c for c in v_df.columns if 'Brake Pressure' in c][0]
        brake_pos = [c for c in v_df.columns if 'Brake Position' in c][0]
        handbrake = [c for c in v_df.columns if 'Handbrake' in c][0]
        
        df = pd.DataFrame({
            # Raw & Linear IMU
            'acc_x': s_df[ax].values,
            'acc_y': s_df[ay].values,
            'acc_z': s_df[az].values,
            'grav_x': s_df[gx].values,
            'grav_y': s_df[gy].values,
            'grav_z': s_df[gz].values,
            'gyro_yaw': s_df[gy_yaw].values,
            'gyro_pitch': s_df[gy_pitch].values,
            'gyro_roll': s_df[gy_roll].values,
            
            # Vehicle Reference Channels
            'can_speed_kmh': v_df[v_spd].values,
            'wheel_rl': v_df[w_rl].values,
            'wheel_rr': v_df[w_rr].values,
            'veh_lon_acc_g': v_df[lon_acc].values,
            'veh_lat_acc_g': v_df[lat_acc].values,
            'veh_yaw_rate': v_df[yaw_r].values,
            'veh_steer_deg': v_df[steer].values,
            'brake_pressure_psi': v_df[brake_press].values,
            'brake_pos': v_df[brake_pos].values,
            'handbrake': v_df[handbrake].values,
            'phone_gps_speed_kmh': s_df[gps_spd].values,
        })
        
        df['can_speed_mps'] = df['can_speed_kmh'] / 3.6
        df['wheel_rear_rads'] = (df['wheel_rl'] + df['wheel_rr']) / 2.0
        df['lin_acc_x'] = df['acc_x'] - df['grav_x']
        df['lin_acc_y'] = df['acc_y'] - df['grav_y']
        df['lin_acc_z'] = df['acc_z'] - df['grav_z']
        df['lin_acc_mag'] = np.sqrt(df['lin_acc_x']**2 + df['lin_acc_y']**2 + df['lin_acc_z']**2)
        df['gyro_mag'] = np.sqrt(df['gyro_yaw']**2 + df['gyro_pitch']**2 + df['gyro_roll']**2)
        
        # Ground Truth Speed: Apply true zero clamping when brake + wheel stopped
        v_gt = df['can_speed_mps'].values.copy()
        stop_mask = (df['brake_pos'] == 1.0) & (df['wheel_rear_rads'] < 0.1) & (df['can_speed_mps'] < 0.1)
        v_gt[stop_mask] = 0.0
        df['v_gt'] = v_gt
        
        # Extract offline regime metadata
        masks, primary = extract_regime_masks(df)
        df['primary_regime'] = primary
        for k, m in masks.items():
            df[f'regime_{k}'] = m
            
        return df

    def build_branch_a(self, df: pd.DataFrame, trip_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Builds Branch A (Engineered Features) dataset for a single trip.
        Returns: (X_features, y_speed, regimes)
        """
        imu_channels = df[['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'lin_acc_mag',
                           'gyro_yaw', 'gyro_pitch', 'gyro_roll', 'gyro_mag']].values
        speeds = df['v_gt'].values
        regimes = df['primary_regime'].values
        
        n_samples = len(df)
        X_list, y_list, reg_list = [], [], []
        
        for start_idx in range(0, n_samples - self.w_a_samples + 1, self.s_a_samples):
            end_idx = start_idx + self.w_a_samples
            center_idx = (start_idx + end_idx) // 2
            
            w_data = imu_channels[start_idx:end_idx]
            feats = compute_window_features(w_data, dt=1.0/self.hz)
            
            X_list.append(feats)
            y_list.append(speeds[center_idx])
            reg_list.append(regimes[center_idx])
            
        if not X_list:
            return np.empty((0, 72), dtype=np.float32), np.empty(0, dtype=np.float32), []
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), reg_list

    def build_branch_b(self, df: pd.DataFrame, trip_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Builds Branch B (Raw 100-step Temporal Sequences) dataset for a single trip.
        Returns: (X_sequences, y_speed, regimes)
        Tensors shape: (N, 100, 6)
        """
        imu_6ch = df[['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'gyro_yaw', 'gyro_pitch', 'gyro_roll']].values
        speeds = df['v_gt'].values
        regimes = df['primary_regime'].values
        
        n_samples = len(df)
        X_list, y_list, reg_list = [], [], []
        
        for start_idx in range(0, n_samples - self.w_b_samples + 1, self.s_b_samples):
            end_idx = start_idx + self.w_b_samples
            target_idx = end_idx - 1  # Causal: predict instantaneous speed at end of window
            
            seq_data = imu_6ch[start_idx:end_idx]
            X_list.append(seq_data)
            y_list.append(speeds[target_idx])
            reg_list.append(regimes[target_idx])
            
        if not X_list:
            return np.empty((0, self.w_b_samples, 6), dtype=np.float32), np.empty(0, dtype=np.float32), []
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), reg_list


def create_split_a(trips: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Trajectory-aware Split A (Unseen Trips within primary driver domain).
    Partitions trips into Train (70%), Val (10%), Test (20%).
    """
    # Sort trips by name for determinism
    sorted_trips = sorted(trips, key=lambda x: x['trip_name'])
    n = len(sorted_trips)
    
    if n == 1:
        return sorted_trips, sorted_trips, sorted_trips
    elif n == 2:
        return [sorted_trips[0]], [sorted_trips[1]], [sorted_trips[1]]
    elif n == 3:
        # Vta02 (largest, ~18.3 min) -> Train
        # Vta03 (~1.1 min) -> Val
        # Vta04 (~3.0 min) -> Test (completely untouched unseen trajectory)
        return [sorted_trips[0]], [sorted_trips[1]], [sorted_trips[2]]
    else:
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.10 * n))
        train_trips = sorted_trips[:n_train]
        val_trips = sorted_trips[n_train:n_train + n_val]
        test_trips = sorted_trips[n_train + n_val:]
        return train_trips, val_trips, test_trips
