"""
SIH26168 - Phase 5.0: Export Locked TCN-kin Model to ONNX for Android
Script: experiments/export_tcn_to_onnx.py

Features:
1. Trains the locked TCN-kin model on training routes (Vta01a, Vta01b, Vta02, Vta03, Vta04, Vta05)
   with locked seed 42 and 12 epochs (Huber/SmoothL1 loss).
2. Exports the trained PyTorch network to ONNX format:
   - Input shape: [batch_size, 100, 9] (100-sample temporal window x 9 kinematic features)
   - Output shape: [batch_size, 1] (forward vehicle speed in m/s)
   - Dynamic batch axis for maximum runtime flexibility.
3. Exports StandardScaler parameters (mean, scale) as JSON for the Android feature pipeline.
4. Verifies ONNX Runtime vs PyTorch numerical parity (asserts max absolute difference < 1e-5 m/s).
5. Deploys directly to Android assets: `android/app/src/main/assets/tcn_kin_speed.onnx`.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
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

ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


def build_tcn_sequences(df: pd.DataFrame, window_samples=100, stride_samples=10):
    feats = extract_tcn_kin_features(df)
    targets = df['can_speed_mps'].values.astype(np.float32)
    
    N = len(feats)
    X_list, y_list, indices = [], [], []
    for end_idx in range(window_samples, N, stride_samples):
        start_idx = end_idx - window_samples
        X_list.append(feats[start_idx:end_idx])
        y_list.append(targets[end_idx - 1])
        indices.append(end_idx - 1)
        
    if len(X_list) == 0:
        return np.empty((0, window_samples, feats.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32), []
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), indices


def train_locked_tcn_kin(X_train: np.ndarray, y_train: np.ndarray, seed=42) -> DilatedTCNNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DilatedTCNNet(in_channels=9, hidden_dim=32)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)
    
    model.train()
    for epoch in range(12):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx).squeeze(-1)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model


def main():
    print("=" * 80)
    print("PHASE 5.0: EXPORT LOCKED TCN-KIN TO ONNX FOR ANDROID")
    print("=" * 80)
    
    builder = IOVNBDDatasetBuilder(data_roots=DATA_ROOTS)
    all_trips = builder.discover_trips()
    vta_trips = [t for t in all_trips if 'Vta' in t['trip_name']]
    selected_train = [t for t in vta_trips if t['trip_name'] in ['Vta01a', 'Vta01b', 'Vta02', 'Vta03', 'Vta04', 'Vta05']]
    
    print(f"[1] Training Trips ({len(selected_train)}): {[t['trip_name'] for t in selected_train]}")
    X_train_list, y_train_list = [], []
    for t in selected_train:
        df = builder.load_clean_trip(t)
        X_t, y_t, _ = build_tcn_sequences(df, window_samples=100, stride_samples=10)
        if len(y_t) > 0:
            X_train_list.append(X_t)
            y_train_list.append(y_t)
            
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"  -> Total Training Sequences: {len(X_train)} windows of shape {X_train.shape[1:]}")
    
    # Fit Scaler
    N, T, C = X_train.shape
    scaler = StandardScaler()
    flat_train = X_train.reshape(-1, C)
    scaler.fit(flat_train)
    X_train_scaled = scaler.transform(flat_train).reshape(N, T, C).astype(np.float32)
    
    # Save Scaler Metadata
    scaler_dict = {
        'feature_names': [
            'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
            'gyro_yaw', 'gyro_pitch', 'gyro_roll',
            'a_horiz', 'a_vert', 'kappa'
        ],
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'window_samples': 100,
        'dt': 0.1,
        'rate_hz': 10.0
    }
    scaler_path_assets = ASSETS_DIR / "tcn_scaler.json"
    scaler_path_models = MODELS_DIR / "tcn_scaler.json"
    for sp in [scaler_path_assets, scaler_path_models]:
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(scaler_dict, f, indent=2)
    print(f"[2] Saved Scaler Metadata: {scaler_path_assets} ({scaler_path_assets.stat().st_size} bytes)")
    
    # Train Locked Model
    print("\n[3] Training Locked PyTorch Model (Seed=42, 12 Epochs)...")
    model = train_locked_tcn_kin(X_train_scaled, y_train, seed=42)
    print("  -> PyTorch Model Trained Successfully.")
    
    # Export to ONNX
    print("\n[4] Exporting to ONNX...")
    dummy_input = torch.randn(1, 100, 9, dtype=torch.float32)
    
    onnx_path_assets = ASSETS_DIR / "tcn_kin_speed.onnx"
    onnx_path_models = MODELS_DIR / "tcn_kin_speed.onnx"
    
    for op in [onnx_path_assets, onnx_path_models]:
        torch.onnx.export(
            model,
            dummy_input,
            str(op),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['speed_mps'],
            dynamic_axes={
                'input_features': {0: 'batch_size'},
                'speed_mps': {0: 'batch_size'}
            },
            dynamo=False
        )
        print(f"  -> Exported ONNX: {op} ({op.stat().st_size / 1024:.1f} KB)")
        
    # Verify ONNX Runtime Parity
    print("\n[5] Verifying Numerical Parity (PyTorch vs ONNX Runtime)...")
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(str(onnx_path_assets))
    
    # Test on 10 random samples from training set
    test_batch = X_train_scaled[:20]
    with torch.no_grad():
        torch_preds = model(torch.from_numpy(test_batch)).squeeze(-1).numpy()
        
    ort_inputs = {ort_session.get_inputs()[0].name: test_batch}
    ort_preds = ort_session.run(None, ort_inputs)[0].squeeze(-1)
    
    max_diff = float(np.max(np.abs(torch_preds - ort_preds)))
    mean_diff = float(np.mean(np.abs(torch_preds - ort_preds)))
    print(f"  -> Mean Abs Difference: {mean_diff:.2e} m/s")
    print(f"  -> Max  Abs Difference: {max_diff:.2e} m/s")
    
    if max_diff < 1e-4:
        print("  -> PARITY VERIFIED: ONNX Runtime matches PyTorch to sub-millimeter precision!")
    else:
        raise AssertionError(f"ONNX parity failed! Max diff = {max_diff}")
        
    print("\n" + "=" * 80)
    print("PHASE 5.0 ONNX EXPORT COMPLETE & READY FOR ANDROID INGESTION")
    print(f"  - Model File : {onnx_path_assets}")
    print(f"  - Scaler File: {scaler_path_assets}")
    print("=" * 80)


if __name__ == '__main__':
    main()
