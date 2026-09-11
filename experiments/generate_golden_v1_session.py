"""
SIH26168 - Phase 5.0: Generate Golden Reference Session V1 for Android Parity
Script: experiments/generate_golden_v1_session.py

Executes the frozen Python ESKF3D + TCN-kin + V1 Decoupled Pure-Velocity Damping
navigation engine on the canonical 1789-epoch Vta04 highway corridor and serializes
both the raw inputs and ground-truth filter states at every single epoch.

Android GoldenReferenceVerifier will ingest this JSON and assert exact numerical parity:
- Position Error: <= 0.05 m
- Velocity Error: <= 0.01 m/s
- Heading Error : <= 0.05 deg
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnxruntime as ort

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D, rotvec_to_quat, quat_mult
from src.map.geometry import wrap_angle_rad

ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
DATA_DIR = PROJECT_ROOT / "data"

DATA_ROOTS = [
    Path('data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset'),
    Path('data/raw/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset'),
]


def extract_tcn_kin_features(df: pd.DataFrame) -> np.ndarray:
    """Locked 9-channel TCN-kin features."""
    lin_acc_x = df['lin_acc_x'].values.astype(np.float32)
    lin_acc_y = df['lin_acc_y'].values.astype(np.float32)
    lin_acc_z = df['lin_acc_z'].values.astype(np.float32)
    gyro_yaw = df['gyro_yaw'].values.astype(np.float32)
    gyro_pitch = df['gyro_pitch'].values.astype(np.float32)
    gyro_roll = df['gyro_roll'].values.astype(np.float32)
    
    grav_x = df['grav_x'].values.astype(np.float32)
    grav_y = df['grav_y'].values.astype(np.float32)
    grav_z = df['grav_z'].values.astype(np.float32)
    grav_norm = np.maximum(np.sqrt(grav_x**2 + grav_y**2 + grav_z**2), 1e-6)
    u_gx, u_gy, u_gz = grav_x / grav_norm, grav_y / grav_norm, grav_z / grav_norm
    
    a_vert = lin_acc_x * u_gx + lin_acc_y * u_gy + lin_acc_z * u_gz
    a_tot_sq = lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2
    a_horiz = np.sqrt(np.maximum(a_tot_sq - a_vert**2, 0.0))
    omega_mag = np.sqrt(gyro_yaw**2 + gyro_pitch**2 + gyro_roll**2)
    kappa = omega_mag * a_horiz
    
    return np.column_stack([
        lin_acc_x, lin_acc_y, lin_acc_z,
        gyro_yaw, gyro_pitch, gyro_roll,
        a_horiz, a_vert, kappa
    ])


def main():
    print("=" * 80)
    print("PHASE 5.0: GENERATING GOLDEN REFERENCE SESSION V1 FOR ANDROID PARITY")
    print("=" * 80)
    
    # 1. Load ONNX Model & Scaler
    onnx_path = ASSETS_DIR / "tcn_kin_speed.onnx"
    scaler_path = ASSETS_DIR / "tcn_scaler.json"
    
    if not onnx_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("ONNX model or scaler missing! Run export_tcn_to_onnx.py first.")
        
    print(f"[1] Loading ONNX model from: {onnx_path.name}")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler_meta = json.load(f)
    scaler_mean = np.array(scaler_meta['mean'], dtype=np.float32)
    scaler_scale = np.array(scaler_meta['scale'], dtype=np.float32)
    
    # 2. Load Vta04 Trip Telemetry
    builder = IOVNBDDatasetBuilder(data_roots=DATA_ROOTS)
    all_trips = builder.discover_trips()
    vta04_trips = [t for t in all_trips if t['trip_name'] == 'Vta04']
    if not vta04_trips:
        raise FileNotFoundError("Vta04 trip not found in dataset!")
    vta04_info = vta04_trips[0]
    
    print(f"[2] Ingesting canonical Vta04 trip telemetry...")
    df = builder.load_clean_trip(vta04_info)
    df_v = pd.read_csv(vta04_info['v_path'], encoding='latin1')
    n_samples = min(len(df), len(df_v))
    df = df.iloc[:n_samples].copy()
    df_v = df_v.iloc[:n_samples].copy()
    
    raw_acc = np.column_stack([df['acc_x'].values, df['acc_y'].values, df['acc_z'].values])
    raw_gyro = np.column_stack([df['gyro_yaw'].values, df['gyro_pitch'].values, df['gyro_roll'].values])
    
    lats = df_v[' Latitude (degrees)'].values
    lons = df_v[' Longitude (degrees)'].values
    headings = df_v[' Heading (degrees)'].values
    speeds = df_v[' Velocity (km/hr)'].values / 3.6
    lat0, lon0 = float(lats[0]), float(lons[0])
    
    e, n_pos, u = geodetic_to_enu(lats, lons, lat0, lon0)
    pos_enu = np.column_stack([e, n_pos, u])
    
    h_rad = np.radians(headings)
    vel_enu = np.column_stack([speeds * np.sin(h_rad), speeds * np.cos(h_rad), np.zeros(n_samples)])
    
    _, _, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro)
    R_vp = R_pv
    
    # Precompute 9-feature series
    feats = extract_tcn_kin_features(df)
    
    # Precompute ONNX predictions at every 10 epochs (1 Hz)
    print(f"[3] Generating 1 Hz ONNX TCN-kin speed predictions over {n_samples} epochs...")
    tcn_predictions = {}
    for i in range(100, n_samples, 10):
        feat_win = feats[i - 100 : i]
        scaled_win = (feat_win - scaler_mean) / scaler_scale
        input_tensor = scaled_win.reshape(1, 100, 9).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        pred_spd = float(ort_session.run(None, ort_inputs)[0][0, 0])
        pred_spd = max(0.0, pred_spd) # Physical non-negative speed
        tcn_predictions[i] = pred_spd
        
    print(f"  -> Generated {len(tcn_predictions)} speed predictions.")
    
    # 3. Simulate Frozen ESKF + TCN + V1 Engine
    print(f"[4] Executing step-by-step frozen Python ESKF engine...")
    dt = 0.1
    sigma_tcn = 1.0
    sigma_nhc_y = 0.5
    
    eskf = ESKF3D(
        sigma_a=0.3,
        sigma_g=0.01,
        sigma_ba=0.001,
        sigma_bg=0.0001
    )
    
    eskf.pos_n = pos_enu[0].copy().astype(np.float64)
    eskf.vel_n = vel_enu[0].copy().astype(np.float64)
    eskf.attitude.R_vp = R_vp.copy()
    
    init_yaw_rad = np.radians(headings[0])
    C_init = np.array([
        [np.sin(init_yaw_rad), -np.cos(init_yaw_rad), 0.0],
        [np.cos(init_yaw_rad),  np.sin(init_yaw_rad), 0.0],
        [0.0,                   0.0,                  1.0]
    ], dtype=np.float64)
    eskf.attitude.q_nv = eskf.attitude._dcm_to_quat(C_init)
    
    eskf.P = np.eye(15, dtype=np.float64) * 0.01
    eskf.P[0:3, 0:3] = np.eye(3) * 1.0
    eskf.P[3:6, 3:6] = np.eye(3) * 0.25
    eskf.P[6:9, 6:9] = np.eye(3) * np.radians(2.0)**2
    
    epoch_records = []
    
    # Run GNSS lock for first 300 epochs (30s), then simulate 600 epochs of outage (60s), then reacquisition
    outage_start = 300
    outage_end = 900
    
    for i in range(n_samples):
        t_sec = i * dt
        ax_p, ay_p, az_p = raw_acc[i]
        gx_p, gy_p, gz_p = raw_gyro[i]
        
        # 1. Inertial Propagation
        eskf.predict(ax_p, ay_p, az_p, gx_p, gy_p, gz_p, dt)
        
        # 2. Gravity Leveling
        eskf.attitude.update_gravity(ax_p, ay_p, az_p, ka=0.02, accel_gate=1.5)
        
        C_v_n = eskf.attitude.get_dcm()
        C_n_v = C_v_n.T
        vel_v = C_n_v @ eskf.vel_n
        
        gnss_valid = not (outage_start <= i < outage_end)
        nav_state = "DEAD_RECKONING" if not gnss_valid else ("REACQUIRING" if i == outage_end else "GNSS_LOCKED")
        
        # GNSS Update when available
        gnss_nis = 0.0
        if gnss_valid and (i % 10 == 0):
            z = np.concatenate([pos_enu[i], vel_enu[i]])
            y_gnss = z - np.concatenate([eskf.pos_n, eskf.vel_n])
            H_gnss = np.zeros((6, 15), dtype=np.float64)
            H_gnss[0:3, 0:3] = np.eye(3)
            H_gnss[3:6, 3:6] = np.eye(3)
            R_gnss = np.diag([4.0, 4.0, 9.0, 0.04, 0.04, 0.25])
            S_gnss = H_gnss @ eskf.P @ H_gnss.T + R_gnss
            S_inv = np.linalg.inv(S_gnss)
            gnss_nis = float(y_gnss.T @ S_inv @ y_gnss)
            K_gnss = eskf.P @ H_gnss.T @ S_inv
            dx = (K_gnss @ y_gnss).flatten()
            eskf.pos_n += dx[0:3]
            eskf.vel_n += dx[3:6]
            dtheta = dx[6:9]
            dq = rotvec_to_quat(dtheta)
            eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq)
            eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
            eskf.ba += dx[9:12]
            eskf.bg += dx[12:15]
            IKH = np.eye(15, dtype=np.float64) - K_gnss @ H_gnss
            eskf.P = IKH @ eskf.P @ IKH.T + K_gnss @ R_gnss @ K_gnss.T
            eskf.P = 0.5 * (eskf.P + eskf.P.T)
            
        # TCN Forward Speed Update (during outage or continuously at 1 Hz)
        tcn_speed_applied = False
        v_hat_tcn = 0.0
        if (not gnss_valid) and (i in tcn_predictions):
            v_hat_tcn = float(tcn_predictions[i])
            r_x = v_hat_tcn - vel_v[0]
            H_x = np.zeros((1, 15), dtype=np.float64)
            H_x[0, 3:6] = C_n_v[0, :]
            S_x = float(np.squeeze(H_x @ eskf.P @ H_x.T) + sigma_tcn**2)
            nis_x = (r_x**2) / max(S_x, 1e-6)
            if nis_x <= 9.0:
                tcn_speed_applied = True
                K_x = (eskf.P @ H_x.T) / max(S_x, 1e-6)
                K_x[6:15, :] = 0.0 # Freeze attitude feedback strictly
                dx_x = (K_x * r_x).flatten()
                eskf.pos_n += dx_x[0:3]
                eskf.vel_n += dx_x[3:6]
                IKH_x = np.eye(15, dtype=np.float64) - K_x @ H_x
                eskf.P = IKH_x @ eskf.P @ IKH_x.T + K_x @ np.array([[sigma_tcn**2]]) @ K_x.T
                eskf.P = 0.5 * (eskf.P + eskf.P.T)
                
        # V1 Decoupled Pure-Velocity Damping (during outage at 10 Hz)
        nhc_applied = False
        if not gnss_valid:
            vel_v_curr = C_n_v @ eskf.vel_n
            H_y = np.zeros((1, 15), dtype=np.float64)
            H_y[0, 3:6] = C_n_v[1, :]
            r_y = 0.0 - vel_v_curr[1]
            S_y = float(np.squeeze(H_y @ eskf.P @ H_y.T) + sigma_nhc_y**2)
            K_y = (eskf.P @ H_y.T) / max(S_y, 1e-6)
            K_y[6:15, :] = 0.0 # Freeze attitude feedback strictly
            dx_y = (K_y * r_y).flatten()
            eskf.vel_n += dx_y[3:6]
            IKH_y = np.eye(15, dtype=np.float64) - K_y @ H_y
            eskf.P = IKH_y @ eskf.P @ IKH_y.T + K_y @ np.array([[sigma_nhc_y**2]]) @ K_y.T
            eskf.P = 0.5 * (eskf.P + eskf.P.T)
            nhc_applied = True
            
        st = eskf.get_state()
        
        epoch_records.append({
            'epoch': i,
            'time_s': round(t_sec, 2),
            'inputs': {
                'accel_p': [float(ax_p), float(ay_p), float(az_p)],
                'gyro_p': [float(gx_p), float(gy_p), float(gz_p)],
                'gnss_valid': gnss_valid,
                'gnss_pos_enu': [float(pos_enu[i, 0]), float(pos_enu[i, 1]), float(pos_enu[i, 2])],
                'gnss_vel_enu': [float(vel_enu[i, 0]), float(vel_enu[i, 1]), float(vel_enu[i, 2])],
                'tcn_speed_mps': float(v_hat_tcn) if tcn_speed_applied else None
            },
            'expected_outputs': {
                'nav_state': nav_state,
                'pos_enu': [float(eskf.pos_n[0]), float(eskf.pos_n[1]), float(eskf.pos_n[2])],
                'vel_enu': [float(eskf.vel_n[0]), float(eskf.vel_n[1]), float(eskf.vel_n[2])],
                'yaw_deg': float(st['yaw_deg']),
                'pitch_deg': float(st['pitch_deg']),
                'roll_deg': float(st['roll_deg']),
                'ba': [float(eskf.ba[0]), float(eskf.ba[1]), float(eskf.ba[2])],
                'bg': [float(eskf.bg[0]), float(eskf.bg[1]), float(eskf.bg[2])],
                'q_nv': [float(eskf.attitude.q_nv[0]), float(eskf.attitude.q_nv[1]), float(eskf.attitude.q_nv[2]), float(eskf.attitude.q_nv[3])],
                'pos_sigma_m': float(np.sqrt(np.trace(eskf.P[0:2, 0:2])))
            }
        })
        
    session_dict = {
        'meta': {
            'trip_name': 'Vta04',
            'origin_lat': lat0,
            'origin_lon': lon0,
            'n_epochs': len(epoch_records),
            'outage_start_epoch': outage_start,
            'outage_end_epoch': outage_end,
            'init_pos_enu': pos_enu[0].tolist(),
            'init_vel_enu': vel_enu[0].tolist(),
            'init_heading_deg': float(headings[0]),
            'R_vp': R_vp.tolist(),
            'dt': dt
        },
        'epochs': epoch_records
    }
    
    out_assets = ASSETS_DIR / "golden_reference_session_v1.json"
    out_data = DATA_DIR / "golden_reference_session_v1.json"
    
    for outp in [out_assets, out_data]:
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2)
            
    print(f"[5] Golden Reference Session V1 successfully serialized:")
    print(f"  - File: {out_assets} ({out_assets.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  - Total Epochs: {len(epoch_records)} ({len(epoch_records)*0.1:.1f} seconds of driving)")
    print(f"  - Outage Window: Epoch {outage_start} to {outage_end} (60.0s blackout)")
    print("=" * 80)


if __name__ == '__main__':
    main()
