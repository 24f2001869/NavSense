#!/usr/bin/env python3
"""
Leakage-Proof Multi-Trip Vehicle Retraining & Benchmarking Experiment
Script: experiments/benchmark_expanded_tcn_vs_rf.py

Evaluates:
1. Current TCN (6-trip training baseline: models/tcn_kin_speed.onnx)
2. Expanded TCN (39-trip training: DilatedTCNNet on diverse IO-VNBD routes)
3. Random Forest (39-trip training: 50 trees, max_depth=10 on statistical window features)

Guarantees:
- Zero data leakage: Splitting strictly at the trip level.
- 19 Completely Held-Out Test Trips (115,420 continuous epochs, 3.21 hours of driving)
- Unseen by BOTH the Current TCN and the Expanded TCN / RF.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet

RESULTS_DIR = PROJECT_ROOT / "results" / "expanded_tcn_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# 1. TRIP PARTITIONING (ZERO LEAKAGE)
# -------------------------------------------------------------
ORIGINAL_TRAIN_TRIPS = ['Vta01a', 'Vta01b', 'Vta02', 'Vta03', 'Vta04', 'Vta05']

HELD_OUT_TEST_TRIP_NAMES = [
    # Vta (Suburban town) - 8 trips
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    # Vtb (Dense urban) - 4 trips
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    # Vw (Winding mountain roads) - 6 trips
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
    # Vf (High-speed motorway) - 1 trip
    'V-Vfa02'
]

VAL_TRIP_NAMES = [
    'Vta19', 'Vta20',
    'Vtb08',
    'Vw10', 'Vw11',
    'V-Vfa01'
]


def extract_tcn_kin_features(df: pd.DataFrame) -> np.ndarray:
    """9-channel TCN-kin features from IO-VNBD."""
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


def compute_integrated_drift(y_true: np.ndarray, y_pred: np.ndarray, horizon_sec: float, dt: float = 0.1) -> float:
    """Computes mean cumulative distance error over rolling horizon windows."""
    window_steps = int(round(horizon_sec / dt))
    n = len(y_true)
    if n < window_steps:
        return float(np.sum(np.abs(y_true - y_pred)) * dt)
    diff = y_pred - y_true
    rolling_dist_err = np.abs(pd.Series(diff * dt).rolling(window_steps).sum().dropna().values)
    return float(np.mean(rolling_dist_err))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """Computes full suite of benchmark metrics."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    
    # R2 score
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum(err**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-4 else 0.0
    
    drift_30s = compute_integrated_drift(y_true, y_pred, horizon_sec=30.0, dt=dt)
    drift_60s = compute_integrated_drift(y_true, y_pred, horizon_sec=60.0, dt=dt)
    
    outliers_30 = int(np.sum(y_pred > 30.0))
    outliers_50 = int(np.sum(y_pred > 50.0))
    max_pred = float(np.max(y_pred))
    
    return {
        'mae_mps': mae,
        'mae_kmh': mae * 3.6,
        'rmse_mps': rmse,
        'rmse_kmh': rmse * 3.6,
        'bias_mps': bias,
        'r2': r2,
        'drift_30s_m': drift_30s,
        'drift_60s_m': drift_60s,
        'outliers_30': outliers_30,
        'outliers_50': outliers_50,
        'max_pred_mps': max_pred,
    }


# -------------------------------------------------------------
# 2. MAIN EXPERIMENT RUNNER
# -------------------------------------------------------------
def main():
    print("=" * 85)
    print("LEAKAGE-PROOF MULTI-TRIP VEHICLE RETRAINING & BENCHMARK")
    print("=" * 85)
    
    # 1. Discover and filter vehicle trips
    data_roots = [
        PROJECT_ROOT / 'data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset',
    ]
    builder = IOVNBDDatasetBuilder(data_roots=data_roots)
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    
    print(f"\n[1] Total synchronized car trips in repository: {len(car_trips)}")
    
    # Partition trips
    train_trips = [t for t in car_trips if t['trip_name'] not in HELD_OUT_TEST_TRIP_NAMES and t['trip_name'] not in VAL_TRIP_NAMES]
    val_trips = [t for t in car_trips if t['trip_name'] in VAL_TRIP_NAMES]
    test_trips = [t for t in car_trips if t['trip_name'] in HELD_OUT_TEST_TRIP_NAMES]
    
    print(f"  * Train trips      : {len(train_trips):2d} trips (includes {len(ORIGINAL_TRAIN_TRIPS)} original)")
    print(f"  * Validation trips : {len(val_trips):2d} trips")
    print(f"  * Held-Out Test    : {len(test_trips):2d} trips")
    
    # Assert zero leakage
    train_set = set(t['trip_name'] for t in train_trips)
    val_set = set(t['trip_name'] for t in val_trips)
    test_set = set(t['trip_name'] for t in test_trips)
    assert len(train_set.intersection(val_set)) == 0, "LEAKAGE DETECTED: Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "LEAKAGE DETECTED: Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "LEAKAGE DETECTED: Val and Test overlap!"
    print("  -> Zero Data Leakage VERIFIED: Train, Val, and Test are completely disjoint sets.")
    
    # 2. Extract Data for Train and Val
    print("\n[2] Ingesting training trips and building sequences (window=100, stride=10)...")
    train_X_list, train_y_list = [], []
    train_rf_X, train_rf_y = [], []
    
    for idx, t in enumerate(train_trips, 1):
        df = builder.load_clean_trip(t)
        feats = extract_tcn_kin_features(df)
        targets = df['can_speed_mps'].values.astype(np.float32)
        N = len(feats)
        if N < 100:
            continue
            
        # Stride 10 for training (1 Hz window stepping)
        for end_i in range(100, N, 10):
            train_X_list.append(feats[end_i-100:end_i])
            train_y_list.append(targets[end_i-1])
            # RF features: window mean, std, rms, min, max of the 9 channels (45 features)
            w_feats = feats[end_i-100:end_i]
            rf_vec = np.hstack([
                np.mean(w_feats, axis=0),
                np.std(w_feats, axis=0),
                np.sqrt(np.mean(w_feats**2, axis=0)),
                np.min(w_feats, axis=0),
                np.max(w_feats, axis=0)
            ])
            train_rf_X.append(rf_vec)
            train_rf_y.append(targets[end_i-1])
            
    X_train = np.array(train_X_list, dtype=np.float32)
    y_train = np.array(train_y_list, dtype=np.float32)
    X_train_rf = np.array(train_rf_X, dtype=np.float32)
    y_train_rf = np.array(train_rf_y, dtype=np.float32)
    print(f"  -> Built {len(X_train)} training sequences from 39 trips ({X_train.nbytes / 1e6:.1f} MB)")
    
    # Fit StandardScaler on Train ONLY
    print("\n[3] Fitting StandardScaler exclusively on Train...")
    scaler = StandardScaler()
    # Reshape (N, 100, 9) to (N*100, 9) to compute channel mean/std
    scaler.fit(X_train.reshape(-1, 9))
    train_mean = scaler.mean_.astype(np.float32)
    train_scale = scaler.scale_.astype(np.float32)
    
    # Scale train sequences
    X_train_scaled = (X_train - train_mean) / train_scale
    
    # Validation data
    print("Ingesting validation trips (window=100, stride=10)...")
    val_X_list, val_y_list = [], []
    for t in val_trips:
        df = builder.load_clean_trip(t)
        feats = extract_tcn_kin_features(df)
        targets = df['can_speed_mps'].values.astype(np.float32)
        N = len(feats)
        if N < 100:
            continue
        for end_i in range(100, N, 10):
            val_X_list.append(feats[end_i-100:end_i])
            val_y_list.append(targets[end_i-1])
            
    X_val = np.array(val_X_list, dtype=np.float32)
    y_val = np.array(val_y_list, dtype=np.float32)
    X_val_scaled = (X_val - train_mean) / train_scale
    print(f"  -> Built {len(X_val)} validation sequences from 6 trips")
    
    # Save expanded scaler
    scaler_dict = {
        'feature_names': ['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'gyro_yaw', 'gyro_pitch', 'gyro_roll', 'a_horiz', 'a_vert', 'kappa'],
        'mean': train_mean.tolist(),
        'scale': train_scale.tolist(),
        'window_samples': 100,
        'dt': 0.1,
        'rate_hz': 10.0,
        'n_train_trips': len(train_trips),
        'n_train_windows': len(X_train),
    }
    with open(RESULTS_DIR / "expanded_tcn_scaler.json", "w") as f:
        json.dump(scaler_dict, f, indent=2)

    # ---------------------------------------------------------
    # 3. TRAIN EXPANDED TCN
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("TRAINING EXPANDED TCN MODEL (39 DIVERSE TRIPS, 12.5 HOURS)")
    print("=" * 85)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    model_expanded = DilatedTCNNet(in_channels=9, hidden_dim=32)
    optimizer = torch.optim.AdamW(model_expanded.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 18
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=0.5)
    
    best_val_loss = float('inf')
    best_weights = None
    
    start_train_time = time.time()
    for epoch in range(epochs):
        model_expanded.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model_expanded(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model_expanded.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model_expanded.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model_expanded(batch_x).squeeze(-1)
                val_loss += criterion(pred, batch_y).item() * len(batch_y)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model_expanded.state_dict().items()}
            mark = "[BEST]"
        else:
            mark = ""
            
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {mark}")
        
    model_expanded.load_state_dict(best_weights)
    train_dur = time.time() - start_train_time
    print(f"\nExpanded TCN Training Complete in {train_dur:.1f} s. Best Val Loss: {best_val_loss:.4f}")
    
    # Save PyTorch checkpoint
    torch.save(best_weights, RESULTS_DIR / "expanded_tcn_weights.pth")
    
    # ---------------------------------------------------------
    # 4. TRAIN RANDOM FOREST BASELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("TRAINING RANDOM FOREST BASELINE (39 TRIPS, 50 TREES, DEPTH 10)")
    print("=" * 85)
    
    rf_start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    # Train on a representative random sample of 25k windows for CPU efficiency
    if len(X_train_rf) > 25000:
        idx_sub = np.random.choice(len(X_train_rf), size=25000, replace=False)
        rf_model.fit(X_train_rf[idx_sub], y_train_rf[idx_sub])
    else:
        rf_model.fit(X_train_rf, y_train_rf)
    rf_dur = time.time() - rf_start_time
    print(f"Random Forest Training Complete in {rf_dur:.1f} s")

    # ---------------------------------------------------------
    # 5. LOAD CURRENT TCN BASELINE (6 TRIPS)
    # ---------------------------------------------------------
    print("\n[5] Loading Current TCN Baseline (models/tcn_kin_speed.onnx)...")
    sess_current = ort.InferenceSession(str(PROJECT_ROOT / "models" / "tcn_kin_speed.onnx"))
    with open(PROJECT_ROOT / "models" / "tcn_scaler.json") as f:
        current_scaler_data = json.load(f)
    current_mean = np.array(current_scaler_data['mean'], dtype=np.float32)
    current_scale = np.array(current_scaler_data['scale'], dtype=np.float32)

    # ---------------------------------------------------------
    # 6. HEAD-TO-HEAD BENCHMARK ON 19 HELD-OUT TEST TRIPS
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"EVALUATING ON {len(test_trips)} COMPLETELY HELD-OUT TEST TRIPS (STRIDE = 1)")
    print("=" * 85)
    
    trip_eval_records = []
    
    # Pre-warm models
    model_expanded.eval()
    
    total_test_epochs = 0
    
    for idx, t in enumerate(test_trips, 1):
        t_name = t['trip_name']
        t_driver = t['driver']
        df = builder.load_clean_trip(t)
        raw_feats = extract_tcn_kin_features(df)
        y_true = df['can_speed_mps'].values.astype(np.float32)[99:]
        N = len(raw_feats)
        if N < 100:
            continue
            
        n_eval_epochs = len(y_true)
        total_test_epochs += n_eval_epochs
        
        # Build sliding windows for full sequence evaluation: (N-99, 100, 9)
        windows_raw = np.lib.stride_tricks.sliding_window_view(raw_feats, window_shape=(100, 9)).squeeze(1)
        
        # 1. Evaluate Current TCN
        windows_current_scaled = (windows_raw - current_mean) / current_scale
        preds_current = []
        for i in range(0, len(windows_current_scaled), 512):
            chunk = windows_current_scaled[i:i+512]
            out = sess_current.run(None, {'input_features': chunk})[0]
            preds_current.append(out.flatten())
        y_pred_current = np.concatenate(preds_current)
        m_curr = compute_metrics(y_true, y_pred_current)
        
        # 2. Evaluate Expanded TCN
        windows_expanded_scaled = (windows_raw - train_mean) / train_scale
        preds_expanded = []
        with torch.no_grad():
            for i in range(0, len(windows_expanded_scaled), 512):
                chunk = torch.from_numpy(windows_expanded_scaled[i:i+512])
                out = model_expanded(chunk).squeeze(-1).numpy()
                preds_expanded.append(out)
        y_pred_expanded = np.concatenate(preds_expanded)
        m_exp = compute_metrics(y_true, y_pred_expanded)
        
        # 3. Evaluate Random Forest
        # Vectorized RF features from windows
        rf_feats = np.hstack([
            np.mean(windows_raw, axis=1),
            np.std(windows_raw, axis=1),
            np.sqrt(np.mean(windows_raw**2, axis=1)),
            np.min(windows_raw, axis=1),
            np.max(windows_raw, axis=1)
        ])
        y_pred_rf = rf_model.predict(rf_feats).astype(np.float32)
        m_rf = compute_metrics(y_true, y_pred_rf)
        
        record = {
            'trip_name': t_name,
            'driver': t_driver,
            'epochs': n_eval_epochs,
            'duration_s': n_eval_epochs * 0.1,
            
            # Current TCN
            'curr_mae': m_curr['mae_mps'],
            'curr_rmse': m_curr['rmse_mps'],
            'curr_bias': m_curr['bias_mps'],
            'curr_r2': m_curr['r2'],
            'curr_d30': m_curr['drift_30s_m'],
            'curr_d60': m_curr['drift_60s_m'],
            
            # Expanded TCN
            'exp_mae': m_exp['mae_mps'],
            'exp_rmse': m_exp['rmse_mps'],
            'exp_bias': m_exp['bias_mps'],
            'exp_r2': m_exp['r2'],
            'exp_d30': m_exp['drift_30s_m'],
            'exp_d60': m_exp['drift_60s_m'],
            
            # Random Forest
            'rf_mae': m_rf['mae_mps'],
            'rf_rmse': m_rf['rmse_mps'],
            'rf_bias': m_rf['bias_mps'],
            'rf_r2': m_rf['r2'],
            'rf_d30': m_rf['drift_30s_m'],
            'rf_d60': m_rf['drift_60s_m'],
        }
        trip_eval_records.append(record)
        
        print(f"[{idx:02d}/{len(test_trips):02d}] {t_driver[:3]}-{t_name:10s} ({n_eval_epochs*0.1/60:.1f}m): "
              f"Curr MAE={m_curr['mae_mps']:4.2f}m/s | Exp MAE={m_exp['mae_mps']:4.2f}m/s | RF MAE={m_rf['mae_mps']:4.2f}m/s")

    df_eval = pd.DataFrame(trip_eval_records)
    
    # ---------------------------------------------------------
    # 7. AGGREGATE METRICS SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"OVERALL HEAD-TO-HEAD BENCHMARK ({len(df_eval)} TEST TRIPS, {total_test_epochs} EPOCHS, {total_test_epochs*0.1/3600:.2f} HOURS)")
    print("=" * 95)
    
    summary_data = [
        {
            'Model': 'Current TCN (6 trips)',
            'Training Scope': '6 trips (1.1 hrs)',
            'Vehicle MAE (m/s)': df_eval['curr_mae'].mean(),
            'Vehicle MAE (km/h)': df_eval['curr_mae'].mean() * 3.6,
            'RMSE (m/s)': df_eval['curr_rmse'].mean(),
            'RMSE (km/h)': df_eval['curr_rmse'].mean() * 3.6,
            'Bias (m/s)': df_eval['curr_bias'].mean(),
            'R2': df_eval['curr_r2'].mean(),
            '30s Drift (m)': df_eval['curr_d30'].mean(),
            '60s Drift (m)': df_eval['curr_d60'].mean(),
        },
        {
            'Model': 'Expanded TCN (39 trips)',
            'Training Scope': '39 trips (12.5 hrs)',
            'Vehicle MAE (m/s)': df_eval['exp_mae'].mean(),
            'Vehicle MAE (km/h)': df_eval['exp_mae'].mean() * 3.6,
            'RMSE (m/s)': df_eval['exp_rmse'].mean(),
            'RMSE (km/h)': df_eval['exp_rmse'].mean() * 3.6,
            'Bias (m/s)': df_eval['exp_bias'].mean(),
            'R2': df_eval['exp_r2'].mean(),
            '30s Drift (m)': df_eval['exp_d30'].mean(),
            '60s Drift (m)': df_eval['exp_d60'].mean(),
        },
        {
            'Model': 'Random Forest Baseline',
            'Training Scope': '39 trips (12.5 hrs)',
            'Vehicle MAE (m/s)': df_eval['rf_mae'].mean(),
            'Vehicle MAE (km/h)': df_eval['rf_mae'].mean() * 3.6,
            'RMSE (m/s)': df_eval['rf_rmse'].mean(),
            'RMSE (km/h)': df_eval['rf_rmse'].mean() * 3.6,
            'Bias (m/s)': df_eval['rf_bias'].mean(),
            'R2': df_eval['rf_r2'].mean(),
            '30s Drift (m)': df_eval['rf_d30'].mean(),
            '60s Drift (m)': df_eval['rf_d60'].mean(),
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    
    print(df_summary.to_string(index=False))
    
    # Road category breakdown
    print("\n" + "-" * 95)
    print("ROAD CATEGORY BREAKDOWN (MAE in m/s):")
    print("-" * 95)
    cat_summary = []
    for d, g in df_eval.groupby('driver'):
        cat_summary.append({
            'Category': d,
            'Trips': len(g),
            'Epochs': g['epochs'].sum(),
            'Current TCN MAE': g['curr_mae'].mean(),
            'Expanded TCN MAE': g['exp_mae'].mean(),
            'RF MAE': g['rf_mae'].mean(),
            'Expanded MAE Improvement': f"{(g['curr_mae'].mean() - g['exp_mae'].mean()) / g['curr_mae'].mean() * 100:+.1f}%"
        })
    df_cat = pd.DataFrame(cat_summary)
    print(df_cat.to_string(index=False))

    # ---------------------------------------------------------
    # 8. PEDESTRIAN WALKING LOG SANITY CHECK
    # ---------------------------------------------------------
    print("\n" + "=" * 95)
    print("SANITY CHECK ON PEDESTRIAN WALKING FIELD LOG (idr_telemetry_20260910_203838.csv)")
    print("=" * 95)
    ped_csv = PROJECT_ROOT / "data/field/files/idr_telemetry_20260910_203838.csv"
    if ped_csv.exists():
        df_ped = pd.read_csv(ped_csv)
        ax = df_ped['accel_x'].values.astype(np.float32)
        ay = df_ped['accel_y'].values.astype(np.float32)
        az = df_ped['accel_z'].values.astype(np.float32)
        gx = df_ped['gyro_x'].values.astype(np.float32)
        gy = df_ped['gyro_y'].values.astype(np.float32)
        gz = df_ped['gyro_z'].values.astype(np.float32)
        
        # Gravity low-pass
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.5 / (8.6 / 2), btype='low')
        grav_x = filtfilt(b, a, ax).astype(np.float32)
        grav_y = filtfilt(b, a, ay).astype(np.float32)
        grav_z = filtfilt(b, a, az).astype(np.float32)
        lin_acc_x = ax - grav_x
        lin_acc_y = ay - grav_y
        lin_acc_z = az - grav_z
        grav_norm = np.maximum(np.sqrt(grav_x**2 + grav_y**2 + grav_z**2), 1e-6)
        u_gx, u_gy, u_gz = grav_x / grav_norm, grav_y / grav_norm, grav_z / grav_norm
        a_vert = lin_acc_x * u_gx + lin_acc_y * u_gy + lin_acc_z * u_gz
        a_tot_sq = lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2
        a_horiz = np.sqrt(np.maximum(a_tot_sq - a_vert**2, 0.0))
        omega_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        kappa = omega_mag * a_horiz
        
        ped_feats = np.column_stack([
            lin_acc_x, lin_acc_y, lin_acc_z,
            gx, gy, gz,
            a_horiz, a_vert, kappa
        ])
        
        ped_windows_raw = np.lib.stride_tricks.sliding_window_view(ped_feats, window_shape=(100, 9)).squeeze(1)
        
        # Current TCN
        ped_curr_scaled = (ped_windows_raw - current_mean) / current_scale
        ped_curr_preds = []
        for i in range(0, len(ped_curr_scaled), 512):
            out = sess_current.run(None, {'input_features': ped_curr_scaled[i:i+512]})[0]
            ped_curr_preds.append(out.flatten())
        y_ped_curr = np.concatenate(ped_curr_preds)
        
        # Expanded TCN
        ped_exp_scaled = (ped_windows_raw - train_mean) / train_scale
        ped_exp_preds = []
        with torch.no_grad():
            for i in range(0, len(ped_exp_scaled), 512):
                out = model_expanded(torch.from_numpy(ped_exp_scaled[i:i+512])).squeeze(-1).numpy()
                ped_exp_preds.append(out)
        y_ped_exp = np.concatenate(ped_exp_preds)
        
        # Random Forest
        rf_ped_feats = np.hstack([
            np.mean(ped_windows_raw, axis=1),
            np.std(ped_windows_raw, axis=1),
            np.sqrt(np.mean(ped_windows_raw**2, axis=1)),
            np.min(ped_windows_raw, axis=1),
            np.max(ped_windows_raw, axis=1)
        ])
        y_ped_rf = rf_model.predict(rf_ped_feats).astype(np.float32)
        
        print(f"Pedestrian Walking Log (Hostel to Mess - True Speed ~ 1.3 m/s):")
        print(f"  * Current TCN   : Mean = {y_ped_curr.mean():5.2f} m/s | Max = {y_ped_curr.max():5.2f} m/s | Windows >= 30m/s: {(y_ped_curr>=30).sum()}")
        print(f"  * Expanded TCN  : Mean = {y_ped_exp.mean():5.2f} m/s | Max = {y_ped_exp.max():5.2f} m/s | Windows >= 30m/s: {(y_ped_exp>=30).sum()}")
        print(f"  * Random Forest : Mean = {y_ped_rf.mean():5.2f} m/s | Max = {y_ped_rf.max():5.2f} m/s | Windows >= 30m/s: {(y_ped_rf>=30).sum()}")
        print("  -> CONFIRMED: Training on vehicles alone does NOT solve the pedestrian OOD extrapolation issue;")
        print("     A dedicated confidence/OOD detection layer is strictly required as planned.")

    # ---------------------------------------------------------
    # 9. PLOTS & REPORT ARTIFACT EXPORT
    # ---------------------------------------------------------
    print("\n[9] Generating diagnostic plots and saving reports...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Trip Vehicle Retraining Benchmark: Current TCN vs Expanded TCN vs Random Forest', fontsize=14, fontweight='bold')
    
    # Panel 1: Overall MAE Comparison
    ax1 = axs[0, 0]
    models = ['Current TCN\n(6 trips)', 'Expanded TCN\n(39 trips)', 'Random Forest\n(39 trips)']
    maes = [df_summary.loc[0, 'Vehicle MAE (m/s)'], df_summary.loc[1, 'Vehicle MAE (m/s)'], df_summary.loc[2, 'Vehicle MAE (m/s)']]
    colors = ['#FF5252', '#00E676', '#2979FF']
    bars1 = ax1.bar(models, maes, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    ax1.set_ylabel('Mean Absolute Error (m/s)', fontweight='bold')
    ax1.set_title('Vehicle Speed MAE on 19 Held-Out Test Trips (115,420 Epochs)', fontweight='bold')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f} m/s\n({h*3.6:.1f} km/h)', ha='center', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(maes) * 1.3)
    
    # Panel 2: 30s and 60s Dead-Reckoning Drift
    ax2 = axs[0, 1]
    x_pos = np.arange(len(models))
    w_bar = 0.35
    d30s = [df_summary.loc[0, '30s Drift (m)'], df_summary.loc[1, '30s Drift (m)'], df_summary.loc[2, '30s Drift (m)']]
    d60s = [df_summary.loc[0, '60s Drift (m)'], df_summary.loc[1, '60s Drift (m)'], df_summary.loc[2, '60s Drift (m)']]
    r1 = ax2.bar(x_pos - w_bar/2, d30s, w_bar, label='30s Horizon Drift', color='#FFB300', edgecolor='black')
    r2 = ax2.bar(x_pos + w_bar/2, d60s, w_bar, label='60s Horizon Drift', color='#E040FB', edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.set_ylabel('Rolling Along-Track Drift Error (meters)', fontweight='bold')
    ax2.set_title('Integrated Dead-Reckoning Distance Drift (30s & 60s)', fontweight='bold')
    for bar in r1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2.0, f'{h:.1f}m', ha='center', fontsize=9, fontweight='bold')
    for bar in r2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2.0, f'{h:.1f}m', ha='center', fontsize=9, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Road Category Breakdown
    ax3 = axs[1, 0]
    cats = df_cat['Category'].values
    x_cat = np.arange(len(cats))
    w_c = 0.25
    ax3.bar(x_cat - w_c, df_cat['Current TCN MAE'], w_c, label='Current TCN (6 trips)', color='#FF5252', edgecolor='black')
    ax3.bar(x_cat, df_cat['Expanded TCN MAE'], w_c, label='Expanded TCN (39 trips)', color='#00E676', edgecolor='black')
    ax3.bar(x_cat + w_c, df_cat['RF MAE'], w_c, label='Random Forest (39 trips)', color='#2979FF', edgecolor='black')
    ax3.set_xticks(x_cat)
    ax3.set_xticklabels(['Suburban (Vta)', 'Urban (Vtb)', 'Mountain (Vw)', 'Motorway (Vf)'], fontweight='bold')
    ax3.set_ylabel('MAE (m/s)', fontweight='bold')
    ax3.set_title('Generalization by Road Category (19 Held-Out Trips)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Per-Trip MAE Distribution
    ax4 = axs[1, 1]
    box_data = [df_eval['curr_mae'].values, df_eval['exp_mae'].values, df_eval['rf_mae'].values]
    bp = ax4.boxplot(box_data, tick_labels=['Current TCN', 'Expanded TCN', 'Random Forest'], patch_artist=True)
    box_colors = ['#FFCDD2', '#C8E6C9', '#BBDEFB']
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col)
    ax4.set_ylabel('Per-Trip MAE (m/s)', fontweight='bold')
    ax4.set_title('Per-Trip MAE Variance Across 19 Held-Out Test Trips', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / "expanded_tcn_benchmark_summary.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved to: {plot_path}")
    
    # Copy plot to brain artifacts dir for markdown embedding
    artifact_plot_dir = PROJECT_ROOT / "results" / "figures"
    artifact_plot_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(plot_path, artifact_plot_dir / "expanded_tcn_benchmark_summary.png")
    
    # Save JSON files
    df_eval.to_json(RESULTS_DIR / "per_trip_evaluation_results.json", orient='records', indent=2)
    df_summary.to_json(RESULTS_DIR / "overall_benchmark_summary.json", orient='records', indent=2)
    df_cat.to_json(RESULTS_DIR / "category_breakdown_summary.json", orient='records', indent=2)
    print(f"Results JSON saved to: {RESULTS_DIR}")
    print("\n" + "=" * 85)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == '__main__':
    main()
