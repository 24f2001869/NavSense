import pandas as pd
import numpy as np
import onnxruntime as ort
import json
from scipy.signal import butter, filtfilt

session = ort.InferenceSession('models/tcn_kin_speed.onnx')
with open('models/tcn_scaler.json') as f:
    scaler = json.load(f)

mean = np.array(scaler['mean'], dtype=np.float32)
scale = np.array(scaler['scale'], dtype=np.float32)
feature_names = scaler['feature_names']

# Load 203838 (hostel to mess walk)
df = pd.read_csv('data/field/files/idr_telemetry_20260910_203838.csv')
ax = df['accel_x'].values.astype(np.float32)
ay = df['accel_y'].values.astype(np.float32)
az = df['accel_z'].values.astype(np.float32)
gx = df['gyro_x'].values.astype(np.float32)
gy = df['gyro_y'].values.astype(np.float32)
gz = df['gyro_z'].values.astype(np.float32)

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

raw_feats = np.column_stack([
    lin_acc_x, lin_acc_y, lin_acc_z,
    gx, gy, gz,
    a_horiz, a_vert, kappa
])

scaled_feats = (raw_feats - mean) / scale

N = len(raw_feats)
window_preds = []
extreme_windows = []

for i in range(100, N):
    w_scaled = scaled_feats[i-100:i].reshape(1, 100, 9)
    pred = session.run(None, {'input_features': w_scaled})[0][0, 0]
    window_preds.append((i, pred))
    if pred >= 60.0:
        w_raw = raw_feats[i-100:i]
        extreme_windows.append((i, pred, w_raw, w_scaled[0]))

print(f"Total evaluated windows: {len(window_preds)}")
print(f"Windows >= 60 m/s: {len(extreme_windows)}")

# Pick the absolute maximum prediction window
max_idx, max_pred, max_w_raw, max_w_scaled = max(extreme_windows, key=lambda x: x[1])
t_start = df['timestamp_ms'].iloc[max_idx-100] / 1000.0 - df['timestamp_ms'].iloc[0] / 1000.0
t_end = df['timestamp_ms'].iloc[max_idx] / 1000.0 - df['timestamp_ms'].iloc[0] / 1000.0
print(f"\nPEAK EXTREME PREDICTION: {max_pred:.2f} m/s ({max_pred*3.6:.1f} km/h)")
print(f"Window: Epoch {max_idx-100} to {max_idx} (Time: {t_start:.1f}s to {t_end:.1f}s)")

# Analyze feature statistics in this extreme window
print("\nFeature Summary in Peak Window (Raw vs Scaler Distribution):")
print(f"{'Feature':12s} | {'Raw Mean':>10s} | {'Raw Min':>10s} | {'Raw Max':>10s} | {'Raw Std':>10s} | {'Z-Score Mean':>12s} | {'Z-Score Max':>12s}")
print("-" * 85)
for f_i, name in enumerate(feature_names):
    col_raw = max_w_raw[:, f_i]
    col_scaled = max_w_scaled[:, f_i]
    print(f"{name:12s} | {col_raw.mean():10.3f} | {col_raw.min():10.3f} | {col_raw.max():10.3f} | {col_raw.std():10.3f} | {col_scaled.mean():12.2f} | {col_scaled.max():12.2f}")

# Feature Ablation Sensitivity
print("\nFeature Ablation on Peak Window (Zeroing out each feature channel):")
base_pred = max_pred
for f_i, name in enumerate(feature_names):
    w_ablated = max_w_scaled.copy().reshape(1, 100, 9)
    # Zero in scaled space corresponds to the training mean
    w_ablated[0, :, f_i] = 0.0
    pred_abl = session.run(None, {'input_features': w_ablated})[0][0, 0]
    drop = base_pred - pred_abl
    pct_drop = (drop / base_pred) * 100.0
    print(f"  Ablate {name:12s} -> New Pred: {pred_abl:6.2f} m/s (Drop: {drop:+6.2f} m/s, {pct_drop:5.1f}%)")
