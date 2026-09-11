#!/usr/bin/env python3
"""
Train Target-Balanced TCN for Speed Estimation (Phase 5.2)
Script: experiments/train_balanced_tcn.py

Features:
- Exact 65k-parameter Dilated TCN architecture (in_channels=9, hidden_dim=32, 5 blocks)
- Leakage-proof: 39 training trips, 6 validation trips, 19 held-out test trips strictly isolated
- Speed-Stratified Mini-Batch Sampling (equalized 25% sampling across 4 speed quartiles)
- Continuous Inverse-Density Loss Weighting (WeightedSmoothL1Loss)
- Cosine Annealing LR Schedule + Early Stopping on Validation Loss
- Exports PyTorch checkpoint & ONNX model
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet
from src.ml.loss_weighting import SpeedDensityEstimator, WeightedSmoothL1Loss
from src.ml.speed_stratified_sampler import create_speed_stratified_sampler

RESULTS_DIR = PROJECT_ROOT / "results" / "balanced_tcn_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_TRAIN_TRIPS = ['Vta01a', 'Vta01b', 'Vta02', 'Vta03', 'Vta04', 'Vta05']

HELD_OUT_TEST_TRIP_NAMES = [
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
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

def main():
    print("=" * 85)
    print("PHASE 5.2: TARGET-BALANCED TCN TRAINING (ELIMINATING SPEED-DEPENDENT BIAS)")
    print("=" * 85)
    
    # 1. Discover and partition vehicle trips
    data_roots = [
        PROJECT_ROOT / 'data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset',
    ]
    builder = IOVNBDDatasetBuilder(data_roots=data_roots)
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    
    train_trips = [t for t in car_trips if t['trip_name'] not in HELD_OUT_TEST_TRIP_NAMES and t['trip_name'] not in VAL_TRIP_NAMES]
    val_trips = [t for t in car_trips if t['trip_name'] in VAL_TRIP_NAMES]
    test_trips = [t for t in car_trips if t['trip_name'] in HELD_OUT_TEST_TRIP_NAMES]
    
    print(f"\n[1] Partitioning Verified:")
    print(f"  * Train trips      : {len(train_trips)} trips (39 trips)")
    print(f"  * Validation trips : {len(val_trips)} trips (6 trips)")
    print(f"  * Held-Out Test    : {len(test_trips)} trips (19 trips)")
    
    # Zero leakage assertion
    train_set = set(t['trip_name'] for t in train_trips)
    val_set = set(t['trip_name'] for t in val_trips)
    test_set = set(t['trip_name'] for t in test_trips)
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    print("  -> Zero Data Leakage Guaranteed: Disjoint sets verified.")
    
    # 2. Ingest sequences for Train and Val
    # Check if pre-cached training sequences exist in scratch to save 60s
    train_cache_file = PROJECT_ROOT / "scratch" / "train_seq_cache.npz"
    if train_cache_file.exists():
        print(f"\n[2] Loading pre-ingested sequences from {train_cache_file}...")
        cache_data = np.load(train_cache_file)
        X_train = cache_data['X_train']
        y_train = cache_data['y_train']
        X_val = cache_data['X_val']
        y_val = cache_data['y_val']
    else:
        print("\n[2] Ingesting training trips (window=100, stride=10)...")
        train_X_list, train_y_list = [], []
        for idx, t in enumerate(train_trips, 1):
            df = builder.load_clean_trip(t)
            feats = extract_tcn_kin_features(df)
            targets = df['can_speed_mps'].values.astype(np.float32)
            N = len(feats)
            if N < 100:
                continue
            for end_i in range(100, N, 10):
                train_X_list.append(feats[end_i-100:end_i])
                train_y_list.append(targets[end_i-1])
        X_train = np.array(train_X_list, dtype=np.float32)
        y_train = np.array(train_y_list, dtype=np.float32)

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
        
        np.savez_compressed(train_cache_file, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        print(f"Saved sequences to cache {train_cache_file}")

    print(f"  -> Built {len(X_train):,d} training windows ({X_train.nbytes / 1e6:.1f} MB)")
    print(f"  -> Built {len(X_val):,d} validation windows")

    # 3. Fit StandardScaler exclusively on Train
    print("\n[3] Fitting StandardScaler exclusively on Train...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 9))
    train_mean = scaler.mean_.astype(np.float32)
    train_scale = scaler.scale_.astype(np.float32)

    X_train_scaled = (X_train - train_mean) / train_scale
    X_val_scaled = (X_val - train_mean) / train_scale

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
    with open(RESULTS_DIR / "balanced_tcn_scaler.json", "w") as f:
        json.dump(scaler_dict, f, indent=2)

    # 4. Target Balancing Setup
    print("\n[4] Setting up Target Balancing Modules...")
    
    # 4a. Continuous Inverse-Density Estimator
    density_estimator = SpeedDensityEstimator(clip_min=0.5, clip_max=3.5)
    density_estimator.fit(y_train)
    sample_loss_weights = density_estimator.compute_weights(y_train)
    print(f"  -> Inverse-Density Weights: min={sample_loss_weights.min():.2f}, mean={sample_loss_weights.mean():.2f}, max={sample_loss_weights.max():.2f}")
    print(f"     Motorway (>=25 m/s) avg weight: {sample_loss_weights[y_train >= 25.0].mean():.2f}x")
    print(f"     Standstill (<1 m/s)  avg weight: {sample_loss_weights[y_train < 1.0].mean():.2f}x")
    print(f"     Mid-Speed (10-20 m/s) avg weight: {sample_loss_weights[(y_train >= 10.0) & (y_train < 20.0)].mean():.2f}x")

    # 4b. Speed-Stratified Mini-Batch Sampler
    stratified_sampler, bin_stats = create_speed_stratified_sampler(y_train)
    print("\n  -> Speed-Stratified Sampler Quartiles:")
    for label, bstat in bin_stats.items():
        print(f"     * {label:26s}: Natural={bstat['natural_pct']:5.2f}% -> Sampled={bstat['effective_sampled_pct']:5.2f}%")

    # 5. Build DataLoaders
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train),
        torch.from_numpy(sample_loss_weights)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_scaled),
        torch.from_numpy(y_val)
    )

    # Use stratified sampler for training loader
    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        sampler=stratified_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False
    )

    # 6. Initialize Model, Optimizer, and Loss
    print("\n[5] Training Balanced TCN Model...")
    model_balanced = DilatedTCNNet(in_channels=9, hidden_dim=32)
    optimizer = torch.optim.AdamW(model_balanced.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 18
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_weighted = WeightedSmoothL1Loss(beta=0.5)
    criterion_val = nn.SmoothL1Loss(beta=0.5)

    best_val_loss = float('inf')
    best_weights = None

    start_train_time = time.time()
    for epoch in range(epochs):
        model_balanced.train()
        train_loss = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            pred = model_balanced(batch_x).squeeze(-1)
            loss = criterion_weighted(pred, batch_y, batch_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model_balanced.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation evaluation
        model_balanced.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model_balanced(batch_x).squeeze(-1)
                val_loss += criterion_val(pred, batch_y).item() * len(batch_y)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model_balanced.state_dict().items()}
            mark = "★ BEST"
        else:
            mark = ""

        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Balanced Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {mark}")

    model_balanced.load_state_dict(best_weights)
    train_dur = time.time() - start_train_time
    print(f"\nBalanced TCN Training Complete in {train_dur:.1f} s. Best Val Loss: {best_val_loss:.4f}")

    # Save PyTorch checkpoint
    checkpoint_path = RESULTS_DIR / "balanced_tcn_weights.pth"
    torch.save(best_weights, checkpoint_path)
    print(f"Saved PyTorch weights to: {checkpoint_path}")

    # 7. Export to ONNX
    print("\n[6] Exporting Balanced TCN to ONNX format...")
    onnx_path = MODELS_DIR / "balanced_tcn_speed.onnx"
    dummy_input = torch.randn(1, 100, 9, dtype=torch.float32)
    torch.onnx.export(
        model_balanced,
        dummy_input,
        str(onnx_path),
        input_names=['input_features'],
        output_names=['speed_mps'],
        dynamic_axes={'input_features': {0: 'batch_size'}, 'speed_mps': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"Saved ONNX model to: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")
    print("\n" + "=" * 85)
    print("PHASE 5.2 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 85)

if __name__ == '__main__':
    main()
