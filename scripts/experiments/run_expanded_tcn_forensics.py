#!/usr/bin/env python3
"""
Comprehensive Forensic Audit of Expanded TCN on 19 Held-Out Test Trips
Script: scratch/run_expanded_tcn_forensics.py
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet

RESULTS_DIR = PROJECT_ROOT / "results" / "expanded_tcn_benchmark"

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

def compute_rolling_drift(y_true: np.ndarray, y_pred: np.ndarray, horizon_sec: float, dt: float = 0.1):
    """Computes rolling horizon along-track drift series and mean."""
    w = int(round(horizon_sec / dt))
    n = len(y_true)
    if n < w:
        err = float(np.sum(np.abs(y_true - y_pred)) * dt)
        return np.array([err]), err, np.array([float(np.sum(y_true) * dt)]), float(np.sum(y_true) * dt)
    
    diff = y_pred - y_true
    # Rolling sum of diff * dt
    rolling_drift = np.abs(pd.Series(diff * dt).rolling(w).sum().dropna().values)
    rolling_dist = pd.Series(y_true * dt).rolling(w).sum().dropna().values
    
    mean_drift = float(np.mean(rolling_drift))
    mean_dist = float(np.mean(rolling_dist))
    
    return rolling_drift, mean_drift, rolling_dist, mean_dist

def main():
    print("=" * 80)
    print("RUNNING IN-DEPTH FORENSIC AUDIT OF EXPANDED TCN (19 HELD-OUT TRIPS)")
    print("=" * 80)
    
    # 1. Load builder and discover trips
    data_roots = [
        PROJECT_ROOT / 'data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset',
    ]
    builder = IOVNBDDatasetBuilder(data_roots=data_roots)
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    
    train_trips = [t for t in car_trips if t['trip_name'] not in HELD_OUT_TEST_TRIP_NAMES and t['trip_name'] not in VAL_TRIP_NAMES]
    val_trips = [t for t in car_trips if t['trip_name'] in VAL_TRIP_NAMES]
    test_trips = [t for t in car_trips if t['trip_name'] in HELD_OUT_TEST_TRIP_NAMES]
    
    # Load Scaler and Model Weights
    with open(RESULTS_DIR / "expanded_tcn_scaler.json") as f:
        scaler_data = json.load(f)
    train_mean = np.array(scaler_data['mean'], dtype=np.float32)
    train_scale = np.array(scaler_data['scale'], dtype=np.float32)
    
    weights_path = RESULTS_DIR / "expanded_tcn_weights.pth"
    model_expanded = DilatedTCNNet(in_channels=9, hidden_dim=32)
    model_expanded.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model_expanded.eval()
    
    # Load Current TCN and Scaler
    import onnxruntime as ort
    sess_current = ort.InferenceSession(str(PROJECT_ROOT / "models" / "tcn_kin_speed.onnx"))
    with open(PROJECT_ROOT / "models" / "tcn_scaler.json") as f:
        curr_scaler = json.load(f)
    curr_mean = np.array(curr_scaler['mean'], dtype=np.float32)
    curr_scale = np.array(curr_scaler['scale'], dtype=np.float32)
    
    # Load existing RF model if available or train quickly
    from sklearn.ensemble import RandomForestRegressor
    print("Preparing Random Forest baseline for head-to-head comparison...")
    # Gather training features for RF
    rf_X_tr, rf_y_tr = [], []
    for t in train_trips:
        df = builder.load_clean_trip(t)
        f = extract_tcn_kin_features(df)
        y = df['can_speed_mps'].values.astype(np.float32)
        for end_i in range(100, len(f), 15):
            w_f = f[end_i-100:end_i]
            rf_vec = np.hstack([
                np.mean(w_f, axis=0), np.std(w_f, axis=0),
                np.sqrt(np.mean(w_f**2, axis=0)), np.min(w_f, axis=0), np.max(w_f, axis=0)
            ])
            rf_X_tr.append(rf_vec)
            rf_y_tr.append(y[end_i-1])
    rf_model = RandomForestRegressor(n_estimators=40, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(np.array(rf_X_tr[:25000]), np.array(rf_y_tr[:25000]))
    print("Random Forest baseline ready.")
    
    # 2. Iterate through each of the 19 test trips
    per_trip_records = []
    all_eval_data = [] # stores granular sample-level predictions and features for failure mode analysis
    
    for idx, t in enumerate(test_trips, 1):
        t_name = t['trip_name']
        t_driver = t['driver']
        
        # Categorize road type
        if 'Vfa' in t_name or 'Vf' in t_driver:
            road_cat = 'Motorway (Vf)'
        elif 'Vtb' in t_name or 'Vtb' in t_driver:
            road_cat = 'Dense Urban (Vtb)'
        elif 'Vw' in t_name or 'Vw' in t_driver:
            road_cat = 'Winding Mountain (Vw)'
        else:
            road_cat = 'Suburban Town (Vta)'
            
        df = builder.load_clean_trip(t)
        raw_feats = extract_tcn_kin_features(df)
        y_true = df['can_speed_mps'].values.astype(np.float32)[99:]
        N = len(raw_feats)
        if N < 100:
            continue
            
        n_epochs = len(y_true)
        duration_s = n_epochs * 0.1
        duration_min = duration_s / 60.0
        
        # Total distance travelled in the evaluation period
        dist_travelled_m = float(np.sum(y_true) * 0.1)
        dist_travelled_km = dist_travelled_m / 1000.0
        
        # Sliding windows: (n_epochs, 100, 9)
        windows_raw = np.lib.stride_tricks.sliding_window_view(raw_feats, window_shape=(100, 9)).squeeze(1)
        
        # Expanded TCN predictions
        w_exp_scaled = (windows_raw - train_mean) / train_scale
        preds_exp = []
        with torch.no_grad():
            for i in range(0, len(w_exp_scaled), 512):
                chunk = torch.from_numpy(w_exp_scaled[i:i+512])
                out = model_expanded(chunk).squeeze(-1).numpy()
                preds_exp.append(out)
        y_pred_exp = np.concatenate(preds_exp)
        
        # Current TCN predictions
        w_curr_scaled = (windows_raw - curr_mean) / curr_scale
        preds_curr = []
        for i in range(0, len(w_curr_scaled), 512):
            out = sess_current.run(None, {'input_features': w_curr_scaled[i:i+512]})[0]
            preds_curr.append(out.flatten())
        y_pred_curr = np.concatenate(preds_curr)
        
        # Random Forest predictions
        rf_feats = np.hstack([
            np.mean(windows_raw, axis=1), np.std(windows_raw, axis=1),
            np.sqrt(np.mean(windows_raw**2, axis=1)), np.min(windows_raw, axis=1), np.max(windows_raw, axis=1)
        ])
        y_pred_rf = rf_model.predict(rf_feats).astype(np.float32)
        
        # Error metrics for Expanded TCN
        err_exp = y_pred_exp - y_true
        mae_exp = float(np.mean(np.abs(err_exp)))
        rmse_exp = float(np.sqrt(np.mean(err_exp**2)))
        bias_exp = float(np.mean(err_exp))
        
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_res_exp = np.sum(err_exp**2)
        r2_exp = float(1.0 - ss_res_exp / (ss_tot + 1e-8)) if ss_tot > 1e-4 else 0.0
        
        # RF and Current TCN MAE
        mae_rf = float(np.mean(np.abs(y_pred_rf - y_true)))
        mae_curr = float(np.mean(np.abs(y_pred_curr - y_true)))
        
        # Rolling drift computations
        r_drift_30, mean_d30, r_dist_30, mean_dist30 = compute_rolling_drift(y_true, y_pred_exp, horizon_sec=30.0, dt=0.1)
        r_drift_60, mean_d60, r_dist_60, mean_dist60 = compute_rolling_drift(y_true, y_pred_exp, horizon_sec=60.0, dt=0.1)
        
        # Drift-to-distance ratios
        # Definition 1: Mean window drift / Mean window distance travelled in that window
        # (This avoids zero division when stopped)
        d30_over_dist_mean_pct = (mean_d30 / max(mean_dist30, 1e-4)) * 100.0
        d60_over_dist_mean_pct = (mean_d60 / max(mean_dist60, 1e-4)) * 100.0
        
        # Definition 2: Per-window ratio strictly during motion (where window dist >= 15m for 30s or 30m for 60s)
        mov_mask_30 = r_dist_30 >= 15.0 # ~0.5 m/s
        mov_mask_60 = r_dist_60 >= 30.0 # ~0.5 m/s
        
        if np.sum(mov_mask_30) > 0:
            pw_ratio_30_pct = float(np.median(r_drift_30[mov_mask_30] / r_dist_30[mov_mask_30])) * 100.0
            p90_ratio_30_pct = float(np.percentile(r_drift_30[mov_mask_30] / r_dist_30[mov_mask_30], 90)) * 100.0
            pct_windows_pass_30 = float(np.mean(r_drift_30[mov_mask_30] / r_dist_30[mov_mask_30] < 0.10)) * 100.0
        else:
            pw_ratio_30_pct = 0.0
            p90_ratio_30_pct = 0.0
            pct_windows_pass_30 = 100.0 if mean_d30 < 3.0 else 0.0
            
        if np.sum(mov_mask_60) > 0:
            pw_ratio_60_pct = float(np.median(r_drift_60[mov_mask_60] / r_dist_60[mov_mask_60])) * 100.0
            p90_ratio_60_pct = float(np.percentile(r_drift_60[mov_mask_60] / r_dist_60[mov_mask_60], 90)) * 100.0
            pct_windows_pass_60 = float(np.mean(r_drift_60[mov_mask_60] / r_dist_60[mov_mask_60] < 0.10)) * 100.0
        else:
            pw_ratio_60_pct = 0.0
            p90_ratio_60_pct = 0.0
            pct_windows_pass_60 = 100.0 if mean_d60 < 6.0 else 0.0
            
        # Definition 3: Window drift / Total trip distance
        d30_over_trip_dist_pct = (mean_d30 / max(dist_travelled_m, 1e-4)) * 100.0
        d60_over_trip_dist_pct = (mean_d60 / max(dist_travelled_m, 1e-4)) * 100.0
        
        max_pred = float(np.max(y_pred_exp))
        p95_pred = float(np.percentile(y_pred_exp, 95))
        max_true = float(np.max(y_true))
        p95_true = float(np.percentile(y_true, 95))
        
        # Check TCN vs RF winner
        winner = 'Expanded TCN' if mae_exp < mae_rf else 'Random Forest'
        diff_vs_rf = mae_rf - mae_exp # positive means TCN is better
        
        rec = {
            'trip_name': t_name,
            'road_category': road_cat,
            'driver': t_driver,
            'epochs': n_epochs,
            'duration_s': round(duration_s, 1),
            'duration_min': round(duration_min, 2),
            'distance_m': round(dist_travelled_m, 1),
            'distance_km': round(dist_travelled_km, 3),
            'avg_speed_mps': round(float(np.mean(y_true)), 2),
            'avg_speed_kmh': round(float(np.mean(y_true)) * 3.6, 1),
            'mae_mps': round(mae_exp, 3),
            'mae_kmh': round(mae_exp * 3.6, 2),
            'rmse_mps': round(rmse_exp, 3),
            'bias_mps': round(bias_exp, 3),
            'r2': round(r2_exp, 4),
            'drift_30s_m': round(mean_d30, 2),
            'drift_60s_m': round(mean_d60, 2),
            'd30_over_dist_mean_pct': round(d30_over_dist_mean_pct, 2),
            'd60_over_dist_mean_pct': round(d60_over_dist_mean_pct, 2),
            'median_pw_ratio_30_pct': round(pw_ratio_30_pct, 2),
            'median_pw_ratio_60_pct': round(pw_ratio_60_pct, 2),
            'pct_windows_pass_30': round(pct_windows_pass_30, 1),
            'pct_windows_pass_60': round(pct_windows_pass_60, 1),
            'pass_sih_60s_mean': bool(d60_over_dist_mean_pct < 10.0),
            'pass_sih_30s_mean': bool(d30_over_dist_mean_pct < 10.0),
            'max_pred_mps': round(max_pred, 2),
            'p95_pred_mps': round(p95_pred, 2),
            'max_true_mps': round(max_true, 2),
            'p95_true_mps': round(p95_true, 2),
            'curr_tcn_mae': round(mae_curr, 3),
            'rf_mae': round(mae_rf, 3),
            'winner': winner,
            'tcn_advantage_mps': round(diff_vs_rf, 3)
        }
        per_trip_records.append(rec)
        
        # Save sample-level features and errors
        # Subsample by 2 to keep memory manageable (56k points)
        step_sub = 1 if n_epochs < 2000 else 4
        for s_i in range(0, n_epochs, step_sub):
            all_eval_data.append({
                'trip': t_name,
                'category': road_cat,
                'v_true': y_true[s_i],
                'v_pred': y_pred_exp[s_i],
                'err': err_exp[s_i],
                'abs_err': abs(err_exp[s_i]),
                'lin_acc_x': raw_feats[s_i+99, 0],
                'lin_acc_y': raw_feats[s_i+99, 1], # longitudinal acceleration
                'lin_acc_z': raw_feats[s_i+99, 2],
                'gyro_yaw': raw_feats[s_i+99, 3],
                'gyro_pitch': raw_feats[s_i+99, 4],
                'gyro_roll': raw_feats[s_i+99, 5],
                'a_horiz': raw_feats[s_i+99, 6],
                'a_vert': raw_feats[s_i+99, 7],
                'kappa': raw_feats[s_i+99, 8],
                'accel_mag': np.sqrt(raw_feats[s_i+99, 0]**2 + raw_feats[s_i+99, 1]**2 + raw_feats[s_i+99, 2]**2),
                'omega_mag': np.sqrt(raw_feats[s_i+99, 3]**2 + raw_feats[s_i+99, 4]**2 + raw_feats[s_i+99, 5]**2)
            })

    df_trips = pd.DataFrame(per_trip_records)
    df_samples = pd.DataFrame(all_eval_data)
    
    # 3. Rank trips BEST to WORST by 60s drift percentage (d60_over_dist_mean_pct)
    df_trips_ranked = df_trips.sort_values(by='d60_over_dist_mean_pct', ascending=True).reset_index(drop=True)
    df_trips_ranked['rank_60s'] = df_trips_ranked.index + 1
    
    # Export CSV
    csv_path = RESULTS_DIR / "per_trip_forensic.csv"
    df_trips_ranked.to_csv(csv_path, index=False)
    print(f"Per-trip forensic CSV saved to: {csv_path}")
    
    # 4. Compute Category-level forensic metrics
    cat_records = []
    for cat, g in df_trips.groupby('road_category'):
        tot_dist_km = g['distance_km'].sum()
        tot_epochs = g['epochs'].sum()
        mean_mae = g['mae_mps'].mean()
        mean_rmse = g['rmse_mps'].mean()
        mean_bias = g['bias_mps'].mean()
        mean_d30 = g['drift_30s_m'].mean()
        mean_d60 = g['drift_60s_m'].mean()
        mean_pct_60 = g['d60_over_dist_mean_pct'].mean()
        pass_count_60 = (g['d60_over_dist_mean_pct'] < 10.0).sum()
        pass_count_30 = (g['d30_over_dist_mean_pct'] < 10.0).sum()
        tcn_wins = (g['winner'] == 'Expanded TCN').sum()
        rf_wins = (g['winner'] == 'Random Forest').sum()
        
        cat_records.append({
            'road_category': cat,
            'n_trips': len(g),
            'total_epochs': tot_epochs,
            'total_dist_km': round(tot_dist_km, 2),
            'mean_mae_mps': round(mean_mae, 3),
            'mean_mae_kmh': round(mean_mae * 3.6, 2),
            'mean_rmse_mps': round(mean_rmse, 3),
            'mean_bias_mps': round(mean_bias, 3),
            'mean_drift_30s_m': round(mean_d30, 2),
            'mean_drift_60s_m': round(mean_d60, 2),
            'mean_d60_pct': round(mean_pct_60, 2),
            'trips_pass_sih_60s': f"{pass_count_60}/{len(g)}",
            'trips_pass_sih_30s': f"{pass_count_30}/{len(g)}",
            'tcn_wins': tcn_wins,
            'rf_wins': rf_wins
        })
    df_cat_forensic = pd.DataFrame(cat_records)

    # 5. Dominance analysis on the aggregate 93.0 m drift
    macro_mean_d60 = df_trips['drift_60s_m'].mean()
    epoch_weighted_d60 = np.sum(df_trips['drift_60s_m'] * df_trips['epochs']) / df_trips['epochs'].sum()
    dist_weighted_d60 = np.sum(df_trips['drift_60s_m'] * df_trips['distance_m']) / df_trips['distance_m'].sum()
    
    print("\n--- AGGREGATE 60s DRIFT DECOMPOSITION ---")
    print(f"Macro (Unweighted) Mean 60s Drift : {macro_mean_d60:.2f} m")
    print(f"Epoch-Weighted Mean 60s Drift     : {epoch_weighted_d60:.2f} m")
    print(f"Distance-Weighted Mean 60s Drift  : {dist_weighted_d60:.2f} m")
    print(f"Single Trip Drift Range           : {df_trips['drift_60s_m'].min():.2f} m (Vw15) to {df_trips['drift_60s_m'].max():.2f} m (V-Vfa02)")
    
    # 6. Feature Correlation and Regime Failure Mode Analysis
    print("\n--- CORRELATION BETWEEN FEATURES AND PREDICTION ERROR ---")
    feat_cols = ['v_true', 'accel_mag', 'omega_mag', 'a_horiz', 'a_vert', 'kappa', 'lin_acc_y', 'gyro_yaw']
    corr_results = []
    for c in feat_cols:
        r_abs, _ = stats.pearsonr(df_samples[c], df_samples['abs_err'])
        r_sgn, _ = stats.pearsonr(df_samples[c], df_samples['err'])
        rho_abs, _ = stats.spearmanr(df_samples[c], df_samples['abs_err'])
        corr_results.append({
            'feature': c,
            'pearson_r_abs_err': round(float(r_abs), 4),
            'spearman_rho_abs_err': round(float(rho_abs), 4),
            'pearson_r_signed_bias': round(float(r_sgn), 4),
        })
    df_corr = pd.DataFrame(corr_results)
    print(df_corr.to_string(index=False))

    # 7. Regime Breakdown Analysis
    print("\n--- SPECIFIC REGIME FAILURE MODE ANALYSIS ---")
    regimes = {
        'Standstill (v < 0.5 m/s)': df_samples['v_true'] < 0.5,
        'Low Speed (0.5 <= v < 5.0 m/s)': (df_samples['v_true'] >= 0.5) & (df_samples['v_true'] < 5.0),
        'Medium Speed (5.0 <= v < 15.0 m/s)': (df_samples['v_true'] >= 5.0) & (df_samples['v_true'] < 15.0),
        'High Speed (15.0 <= v < 25.0 m/s)': (df_samples['v_true'] >= 15.0) & (df_samples['v_true'] < 25.0),
        'Motorway Extreme (v >= 25.0 m/s)': df_samples['v_true'] >= 25.0,
        'Hard Acceleration (a_y > 1.5 m/s2)': df_samples['lin_acc_y'] > 1.5,
        'Hard Braking (a_y < -1.5 m/s2)': df_samples['lin_acc_y'] < -1.5,
        'Sharp Turning (|yaw_rate| > 0.1 rad/s)': np.abs(df_samples['gyro_yaw']) > 0.1,
        'Straight Cruising (|yaw| < 0.02 & |a_y| < 0.3)': (np.abs(df_samples['gyro_yaw']) < 0.02) & (np.abs(df_samples['lin_acc_y']) < 0.3),
    }
    regime_results = []
    for reg_name, mask in regimes.items():
        n_pts = int(np.sum(mask))
        if n_pts > 0:
            sub_err = df_samples.loc[mask, 'err']
            sub_abs = df_samples.loc[mask, 'abs_err']
            sub_true = df_samples.loc[mask, 'v_true']
            sub_pred = df_samples.loc[mask, 'v_pred']
            
            under_cnt = int(np.sum(sub_err < -3.0))
            over_cnt = int(np.sum(sub_err > 3.0))
            
            regime_results.append({
                'regime': reg_name,
                'samples': n_pts,
                'pct_of_eval': round(n_pts / len(df_samples) * 100.0, 1),
                'mean_true_mps': round(float(np.mean(sub_true)), 2),
                'mean_pred_mps': round(float(np.mean(sub_pred)), 2),
                'mae_mps': round(float(np.mean(sub_abs)), 3),
                'bias_mps': round(float(np.mean(sub_err)), 3),
                'under_est_pct': round(under_cnt / n_pts * 100.0, 1),
                'over_est_pct': round(over_cnt / n_pts * 100.0, 1),
            })
    df_regimes = pd.DataFrame(regime_results)
    print(df_regimes.to_string(index=False))

    # 8. Produce Forensic Plots
    print("\n--- GENERATING 6-PANEL FORENSIC PLOT ---")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    fig.suptitle('Forensic Evaluation of Expanded TCN on 19 Held-Out Test Trips (IO-VNBD)', fontsize=15, fontweight='bold')
    
    # Panel 1: Ranked 60s Drift Percentage by Trip
    ax1 = fig.add_subplot(gs[0, 0])
    trips_sorted = df_trips_ranked.sort_values(by='d60_over_dist_mean_pct', ascending=True)
    bar_colors = ['#00E676' if p < 10.0 else ('#FFB300' if p < 20.0 else '#FF5252') for p in trips_sorted['d60_over_dist_mean_pct']]
    bars = ax1.barh(trips_sorted['trip_name'], trips_sorted['d60_over_dist_mean_pct'], color=bar_colors, edgecolor='black', alpha=0.85)
    ax1.axvline(10.0, color='red', linestyle='--', linewidth=1.5, label='SIH 10% Threshold')
    ax1.set_xlabel('60s Drift / Distance Travelled (%)', fontweight='bold')
    ax1.set_title('Trip Ranking: 60s Dead-Reckoning Drift % (Best to Worst)', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    for b in bars:
        w = b.get_width()
        ax1.text(w + 0.5, b.get_y() + b.get_height()/2, f'{w:.1f}%', va='center', fontsize=8, fontweight='bold')
    ax1.set_xlim(0, max(trips_sorted['d60_over_dist_mean_pct']) * 1.15)
    
    # Panel 2: TCN vs Random Forest MAE Head-to-Head
    ax2 = fig.add_subplot(gs[0, 1])
    x_idx = np.arange(len(df_trips))
    w_b = 0.38
    ax2.bar(x_idx - w_b/2, df_trips['mae_mps'], w_b, label='Expanded TCN', color='#00E676', edgecolor='black', alpha=0.85)
    ax2.bar(x_idx + w_b/2, df_trips['rf_mae'], w_b, label='Random Forest', color='#2979FF', edgecolor='black', alpha=0.85)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(df_trips['trip_name'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('MAE (m/s)', fontweight='bold')
    ax2.set_title('Head-to-Head MAE per Trip: Expanded TCN vs Random Forest', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Error vs True Speed (Binned Scatter)
    ax3 = fig.add_subplot(gs[1, 0])
    v_bins = np.linspace(0, 35, 15)
    df_samples['v_bin'] = pd.cut(df_samples['v_true'], bins=v_bins)
    bin_stats = df_samples.groupby('v_bin', observed=False).agg({
        'v_true': 'mean',
        'abs_err': 'mean',
        'err': 'mean',
        'v_pred': 'mean'
    }).dropna()
    ax3.plot(bin_stats['v_true'], bin_stats['abs_err'], 'o-', color='#D50000', linewidth=2, label='Mean Absolute Error (m/s)')
    ax3.plot(bin_stats['v_true'], bin_stats['err'], 's--', color='#2979FF', linewidth=2, label='Signed Bias (m/s)')
    ax3.axhline(0, color='gray', linestyle=':')
    ax3.set_xlabel('True Vehicle Speed (m/s)', fontweight='bold')
    ax3.set_ylabel('Speed Error (m/s)', fontweight='bold')
    ax3.set_title('Prediction Error & Bias vs True Vehicle Speed', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Error vs Longitudinal Acceleration (Braking vs Throttle)
    ax4 = fig.add_subplot(gs[1, 1])
    a_bins = np.linspace(-4, 4, 17)
    df_samples['a_bin'] = pd.cut(df_samples['lin_acc_y'], bins=a_bins)
    a_stats = df_samples.groupby('a_bin', observed=False).agg({
        'lin_acc_y': 'mean',
        'abs_err': 'mean',
        'err': 'mean'
    }).dropna()
    ax4.plot(a_stats['lin_acc_y'], a_stats['abs_err'], 'o-', color='#FF6D00', linewidth=2, label='MAE (m/s)')
    ax4.plot(a_stats['lin_acc_y'], a_stats['err'], 's--', color='#00B0FF', linewidth=2, label='Bias (m/s)')
    ax4.axvline(0, color='gray', linestyle=':')
    ax4.axhline(0, color='gray', linestyle=':')
    ax4.set_xlabel('Longitudinal Acceleration lin_acc_y (m/s²)', fontweight='bold')
    ax4.set_ylabel('Speed Error (m/s)', fontweight='bold')
    ax4.set_title('Error Behavior during Braking (<0) vs Acceleration (>0)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Error vs Yaw Rate (Turning Dynamics)
    ax5 = fig.add_subplot(gs[2, 0])
    yaw_bins = np.linspace(0, 0.4, 15)
    df_samples['yaw_bin'] = pd.cut(np.abs(df_samples['gyro_yaw']), bins=yaw_bins)
    yaw_stats = df_samples.groupby('yaw_bin', observed=False).agg({
        'gyro_yaw': lambda x: np.mean(np.abs(x)),
        'abs_err': 'mean',
        'err': 'mean'
    }).dropna()
    ax5.plot(yaw_stats['gyro_yaw'] * 180.0 / np.pi, yaw_stats['abs_err'], 'd-', color='#7C4DFF', linewidth=2, label='MAE (m/s)')
    ax5.plot(yaw_stats['gyro_yaw'] * 180.0 / np.pi, yaw_stats['err'], '^--', color='#00BFA5', linewidth=2, label='Signed Bias (m/s)')
    ax5.set_xlabel('Absolute Yaw Rate |gyro_yaw| (deg/s)', fontweight='bold')
    ax5.set_ylabel('Speed Error (m/s)', fontweight='bold')
    ax5.set_title('Error Behavior during Turns (Cornering Dynamics)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: SIH Pass Rates across 4 Road Categories
    ax6 = fig.add_subplot(gs[2, 1])
    cats = df_cat_forensic['road_category'].values
    pass_counts = [int(p.split('/')[0]) for p in df_cat_forensic['trips_pass_sih_60s']]
    fail_counts = [int(p.split('/')[1]) - int(p.split('/')[0]) for p in df_cat_forensic['trips_pass_sih_60s']]
    x_c = np.arange(len(cats))
    ax6.bar(x_c, pass_counts, label='Satisfies <10% SIH Drift', color='#00E676', edgecolor='black', alpha=0.85)
    ax6.bar(x_c, fail_counts, bottom=pass_counts, label='Exceeds 10% SIH Drift', color='#FF5252', edgecolor='black', alpha=0.85)
    ax6.set_xticks(x_c)
    ax6.set_xticklabels(['Motorway\n(Vf)', 'Urban\n(Vtb)', 'Suburban\n(Vta)', 'Mountain\n(Vw)'], fontweight='bold')
    ax6.set_ylabel('Number of Trips', fontweight='bold')
    ax6.set_title('SIH <10% Distance Drift Compliance by Road Category', fontweight='bold')
    for i in range(len(cats)):
        tot = pass_counts[i] + fail_counts[i]
        pct = pass_counts[i] / tot * 100.0
        ax6.text(i, tot + 0.15, f'{pass_counts[i]}/{tot}\n({pct:.0f}%)', ha='center', fontweight='bold', fontsize=9)
    ax6.set_ylim(0, 10.5)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / "per_trip_forensic_plots.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved to: {plot_path}")
    
    # Copy plot to brain artifacts dir for markdown embedding
    artifact_plot_dir = PROJECT_ROOT / "results" / "figures"
    artifact_plot_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(plot_path, artifact_plot_dir / "per_trip_forensic_plots.png")
    
    # Save intermediate JSONs for report generation
    df_trips_ranked.to_json(RESULTS_DIR / "per_trip_forensic.json", orient='records', indent=2)
    df_cat_forensic.to_json(RESULTS_DIR / "category_forensic.json", orient='records', indent=2)
    df_corr.to_json(RESULTS_DIR / "feature_correlations.json", orient='records', indent=2)
    df_regimes.to_json(RESULTS_DIR / "regime_failure_modes.json", orient='records', indent=2)
    
    print("\n" + "=" * 80)
    print("FORENSIC AUDIT COMPLETE. READY FOR REPORT GENERATION.")
    print("=" * 80)

if __name__ == '__main__':
    main()
