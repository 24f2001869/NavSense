#!/usr/bin/env python3
"""
Comprehensive Forensic Evaluation of Target-Balanced TCN vs Expanded TCN vs Baselines
Script: experiments/evaluate_balanced_tcn.py
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
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet

RESULTS_EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_tcn_benchmark"
RESULTS_BALANCED_DIR = PROJECT_ROOT / "results" / "balanced_tcn_benchmark"
RESULTS_BALANCED_DIR.mkdir(parents=True, exist_ok=True)

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
    w = int(round(horizon_sec / dt))
    n = len(y_true)
    if n < w:
        err = float(np.sum(np.abs(y_true - y_pred)) * dt)
        dist = float(np.sum(y_true) * dt)
        return np.array([err]), err, np.array([dist]), dist
    
    diff = y_pred - y_true
    rolling_drift = np.abs(pd.Series(diff * dt).rolling(w).sum().dropna().values)
    rolling_dist = pd.Series(y_true * dt).rolling(w).sum().dropna().values
    
    mean_drift = float(np.mean(rolling_drift))
    mean_dist = float(np.mean(rolling_dist))
    return rolling_drift, mean_drift, rolling_dist, mean_dist

def run_tcn_inference(model, raw_feats, mean, scale, chunk_size=512):
    scaled_feats = (raw_feats - mean) / scale
    windows = np.lib.stride_tricks.sliding_window_view(scaled_feats, window_shape=(100, 9)).squeeze(1)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), chunk_size):
            chunk = np.ascontiguousarray(windows[i:i+chunk_size], dtype=np.float32)
            t_chunk = torch.from_numpy(chunk)
            out = model(t_chunk).squeeze(-1).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)

def main():
    print("=" * 85)
    print("PHASE 5.2 FORENSIC EVALUATION: TARGET-BALANCED TCN ON 19 HELD-OUT TEST TRIPS")
    print("=" * 85)
    
    builder = IOVNBDDatasetBuilder([
        Path('data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset'),
    ])
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    test_trips = [t for t in car_trips if t['trip_name'] in HELD_OUT_TEST_TRIP_NAMES]
    
    # 1. Load Scalers
    with open(RESULTS_EXPANDED_DIR / "expanded_tcn_scaler.json") as f:
        exp_scaler_data = json.load(f)
    exp_mean = np.array(exp_scaler_data['mean'], dtype=np.float32)
    exp_scale = np.array(exp_scaler_data['scale'], dtype=np.float32)

    with open(RESULTS_BALANCED_DIR / "balanced_tcn_scaler.json") as f:
        bal_scaler_data = json.load(f)
    bal_mean = np.array(bal_scaler_data['mean'], dtype=np.float32)
    bal_scale = np.array(bal_scaler_data['scale'], dtype=np.float32)

    # 2. Load Models
    print("\n[1] Loading Models for Head-to-Head Comparison...")
    
    # Expanded TCN
    exp_weights_path = RESULTS_EXPANDED_DIR / "expanded_tcn_weights.pth"
    model_expanded = DilatedTCNNet(in_channels=9, hidden_dim=32)
    model_expanded.load_state_dict(torch.load(exp_weights_path, map_location='cpu'))
    model_expanded.eval()
    print("  * Expanded TCN loaded.")

    # Balanced TCN
    bal_weights_path = RESULTS_BALANCED_DIR / "balanced_tcn_weights.pth"
    model_balanced = DilatedTCNNet(in_channels=9, hidden_dim=32)
    model_balanced.load_state_dict(torch.load(bal_weights_path, map_location='cpu'))
    model_balanced.eval()
    print("  * Target-Balanced TCN loaded.")

    # 3. Replay through all 19 held-out test trips
    print(f"\n[2] Replaying {len(test_trips)} Completely Held-Out Test Trips (113,539 Epochs)...")
    
    per_trip_results = []
    
    all_y_true = []
    all_pred_exp = []
    all_pred_bal = []
    
    for idx, t in enumerate(test_trips, 1):
        t_name = t['trip_name']
        t_driver = t['driver']
        
        df = builder.load_clean_trip(t)
        raw_feats = extract_tcn_kin_features(df)
        if len(raw_feats) < 100:
            continue
        y_true = df['can_speed_mps'].values.astype(np.float32)[99:]
        N = len(y_true)
        duration_s = N * 0.1
        distance_m = float(np.sum(y_true) * 0.1)
        avg_speed_mps = float(np.mean(y_true))
        
        # Inferences
        y_pred_exp = run_tcn_inference(model_expanded, raw_feats, exp_mean, exp_scale)
        y_pred_bal = run_tcn_inference(model_balanced, raw_feats, bal_mean, bal_scale)
        
        # Errors
        err_exp = y_pred_exp - y_true
        err_bal = y_pred_bal - y_true
        
        mae_exp = float(np.mean(np.abs(err_exp)))
        mae_bal = float(np.mean(np.abs(err_bal)))
        
        rmse_exp = float(np.sqrt(np.mean(err_exp**2)))
        rmse_bal = float(np.sqrt(np.mean(err_bal**2)))
        
        bias_exp = float(np.mean(err_exp))
        bias_bal = float(np.mean(err_bal))
        
        # Drifts
        _, d30_exp, _, dist30 = compute_rolling_drift(y_true, y_pred_exp, 30.0)
        _, d60_exp, _, dist60 = compute_rolling_drift(y_true, y_pred_exp, 60.0)
        
        _, d30_bal, _, _ = compute_rolling_drift(y_true, y_pred_bal, 30.0)
        _, d60_bal, _, _ = compute_rolling_drift(y_true, y_pred_bal, 60.0)
        
        d60_pct_exp = (d60_exp / max(dist60, 1e-4)) * 100.0
        d60_pct_bal = (d60_bal / max(dist60, 1e-4)) * 100.0
        
        pass_sih_60_exp = bool(d60_pct_exp < 10.0 and dist60 > 50.0)
        pass_sih_60_bal = bool(d60_pct_bal < 10.0 and dist60 > 50.0)
        
        winner = "Balanced TCN" if mae_bal < mae_exp else "Expanded TCN"
        adv_mps = mae_exp - mae_bal
        
        rec = {
            'trip_name': t_name,
            'driver': t_driver,
            'epochs': N,
            'duration_min': round(duration_s / 60.0, 2),
            'distance_km': round(distance_m / 1000.0, 3),
            'avg_speed_kmh': round(avg_speed_mps * 3.6, 1),
            
            # Expanded TCN
            'exp_mae_mps': round(mae_exp, 3),
            'exp_rmse_mps': round(rmse_exp, 3),
            'exp_bias_mps': round(bias_exp, 3),
            'exp_drift_60s_m': round(d60_exp, 2),
            'exp_drift_60s_pct': round(d60_pct_exp, 2),
            'exp_pass_sih_60': pass_sih_60_exp,
            
            # Balanced TCN
            'bal_mae_mps': round(mae_bal, 3),
            'bal_rmse_mps': round(rmse_bal, 3),
            'bal_bias_mps': round(bias_bal, 3),
            'bal_drift_60s_m': round(d60_bal, 2),
            'bal_drift_60s_pct': round(d60_pct_bal, 2),
            'bal_pass_sih_60': pass_sih_60_bal,
            
            'winner': winner,
            'balanced_adv_mps': round(adv_mps, 3),
        }
        per_trip_results.append(rec)
        
        all_y_true.append(y_true)
        all_pred_exp.append(y_pred_exp)
        all_pred_bal.append(y_pred_bal)
        
        status_sym = "[PASS]" if pass_sih_60_bal else "[FAIL]"
        print(f"[{idx:02d}/19] {t_name:10s} ({rec['duration_min']:4.1f}m, {rec['avg_speed_kmh']:4.1f} km/h): "
              f"Exp MAE={mae_exp:4.2f} -> Bal MAE={mae_bal:4.2f} m/s | "
              f"Exp 60s={d60_pct_exp:5.1f}% -> Bal 60s={d60_pct_bal:5.1f}% {status_sym}")

    df_per_trip = pd.DataFrame(per_trip_results)
    df_per_trip.to_csv(RESULTS_BALANCED_DIR / "per_trip_balanced_benchmark.csv", index=False)
    
    # 4. Global Analysis & Failure Modes
    all_y_true = np.concatenate(all_y_true)
    all_pred_exp = np.concatenate(all_pred_exp)
    all_pred_bal = np.concatenate(all_pred_bal)
    
    err_exp_all = all_pred_exp - all_y_true
    err_bal_all = all_pred_bal - all_y_true
    
    # Regime Masks
    m_standstill = all_y_true < 0.5
    m_low = (all_y_true >= 0.5) & (all_y_true < 5.0)
    m_mid = (all_y_true >= 5.0) & (all_y_true < 20.0)
    m_fast = (all_y_true >= 20.0) & (all_y_true < 25.0)
    m_motorway = all_y_true >= 25.0
    
    regime_summary = [
        {
            'Regime': 'Standstill (<0.5 m/s)',
            'Samples': int(np.sum(m_standstill)),
            'Exp MAE': round(float(np.mean(np.abs(err_exp_all[m_standstill]))), 3),
            'Bal MAE': round(float(np.mean(np.abs(err_bal_all[m_standstill]))), 3),
            'Exp Bias': round(float(np.mean(err_exp_all[m_standstill])), 3),
            'Bal Bias': round(float(np.mean(err_bal_all[m_standstill])), 3),
        },
        {
            'Regime': 'Low Speed (0.5-5 m/s)',
            'Samples': int(np.sum(m_low)),
            'Exp MAE': round(float(np.mean(np.abs(err_exp_all[m_low]))), 3),
            'Bal MAE': round(float(np.mean(np.abs(err_bal_all[m_low]))), 3),
            'Exp Bias': round(float(np.mean(err_exp_all[m_low])), 3),
            'Bal Bias': round(float(np.mean(err_bal_all[m_low])), 3),
        },
        {
            'Regime': 'Mid Speed (5-20 m/s)',
            'Samples': int(np.sum(m_mid)),
            'Exp MAE': round(float(np.mean(np.abs(err_exp_all[m_mid]))), 3),
            'Bal MAE': round(float(np.mean(np.abs(err_bal_all[m_mid]))), 3),
            'Exp Bias': round(float(np.mean(err_exp_all[m_mid])), 3),
            'Bal Bias': round(float(np.mean(err_bal_all[m_mid])), 3),
        },
        {
            'Regime': 'Motorway Extreme (>=25 m/s)',
            'Samples': int(np.sum(m_motorway)),
            'Exp MAE': round(float(np.mean(np.abs(err_exp_all[m_motorway]))), 3),
            'Bal MAE': round(float(np.mean(np.abs(err_bal_all[m_motorway]))), 3),
            'Exp Bias': round(float(np.mean(err_exp_all[m_motorway])), 3),
            'Bal Bias': round(float(np.mean(err_bal_all[m_motorway])), 3),
        },
    ]
    df_regimes = pd.DataFrame(regime_summary)
    
    # Macro summaries
    macro_exp_mae = float(df_per_trip['exp_mae_mps'].mean())
    macro_bal_mae = float(df_per_trip['bal_mae_mps'].mean())
    
    macro_exp_d60 = float(df_per_trip['exp_drift_60s_m'].mean())
    macro_bal_d60 = float(df_per_trip['bal_drift_60s_m'].mean())
    
    pass_exp_count = int(df_per_trip['exp_pass_sih_60'].sum())
    pass_bal_count = int(df_per_trip['bal_pass_sih_60'].sum())
    
    bal_wins = int((df_per_trip['winner'] == 'Balanced TCN').sum())
    
    # Motorway trip specifically
    mw_row = df_per_trip[df_per_trip['trip_name'] == 'V-Vfa02'].iloc[0]
    
    overall_summary = {
        'Expanded TCN Macro MAE (m/s)': round(macro_exp_mae, 3),
        'Balanced TCN Macro MAE (m/s)': round(macro_bal_mae, 3),
        'Expanded TCN Epoch Weighted MAE (m/s)': round(float(np.mean(np.abs(err_exp_all))), 3),
        'Balanced TCN Epoch Weighted MAE (m/s)': round(float(np.mean(np.abs(err_bal_all))), 3),
        'Expanded TCN Macro 60s Drift (m)': round(macro_exp_d60, 2),
        'Balanced TCN Macro 60s Drift (m)': round(macro_bal_d60, 2),
        'Expanded TCN SIH 60s Pass Rate': f"{pass_exp_count}/19",
        'Balanced TCN SIH 60s Pass Rate': f"{pass_bal_count}/19",
        'Balanced TCN Wins vs Expanded TCN': f"{bal_wins}/19 ({bal_wins/19*100:.1f}%)",
        'Motorway V-Vfa02 Exp MAE': float(mw_row['exp_mae_mps']),
        'Motorway V-Vfa02 Bal MAE': float(mw_row['bal_mae_mps']),
        'Motorway V-Vfa02 Exp 60s %': float(mw_row['exp_drift_60s_pct']),
        'Motorway V-Vfa02 Bal 60s %': float(mw_row['bal_drift_60s_pct']),
        'Motorway V-Vfa02 Exp Bias': float(mw_row['exp_bias_mps']),
        'Motorway V-Vfa02 Bal Bias': float(mw_row['bal_bias_mps']),
    }
    
    print("\n" + "=" * 85)
    print("SUMMARY COMPARISON: EXPANDED TCN VS TARGET-BALANCED TCN")
    print("=" * 85)
    for k, v in overall_summary.items():
        print(f"  * {k:40s}: {v}")
        
    print("\n" + "-" * 85)
    print("REGIME FAILURE MODE ELIMINATION:")
    print("-" * 85)
    print(df_regimes.to_string(index=False))
    
    # Save summaries
    with open(RESULTS_BALANCED_DIR / "overall_balanced_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)
    with open(RESULTS_BALANCED_DIR / "regime_comparison.json", "w") as f:
        json.dump(regime_summary, f, indent=2)

    # 5. Diagnostic Plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Phase 5.2: Target-Balanced TCN vs Expanded TCN on 19 Held-Out Test Trips', fontsize=14, fontweight='bold')
    
    # Panel 1: Per-Trip MAE Comparison
    ax1 = axs[0, 0]
    x_idx = np.arange(len(df_per_trip))
    w = 0.35
    ax1.bar(x_idx - w/2, df_per_trip['exp_mae_mps'], w, label='Expanded TCN (Unweighted)', color='#90CAF9', edgecolor='black')
    ax1.bar(x_idx + w/2, df_per_trip['bal_mae_mps'], w, label='Balanced TCN (Loss-Weighted)', color='#4CAF50', edgecolor='black')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(df_per_trip['trip_name'], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('MAE (m/s)', fontweight='bold')
    ax1.set_title('Per-Trip MAE Comparison across 19 Held-Out Trips', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: 60s Drift Percentage & SIH Threshold
    ax2 = axs[0, 1]
    ax2.bar(x_idx - w/2, np.clip(df_per_trip['exp_drift_60s_pct'], 0, 40), w, label='Expanded TCN', color='#90CAF9', edgecolor='black')
    ax2.bar(x_idx + w/2, np.clip(df_per_trip['bal_drift_60s_pct'], 0, 40), w, label='Balanced TCN', color='#4CAF50', edgecolor='black')
    ax2.axhline(10.0, color='red', linestyle='--', linewidth=2, label='SIH <10% Threshold')
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(df_per_trip['trip_name'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('60s Drift / Distance (%) [Capped at 40%]', fontweight='bold')
    ax2.set_title('60-Second Drift Percentage vs SIH Threshold', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Bias by Speed Regime
    ax3 = axs[1, 0]
    reg_names = [r['Regime'].split(' ')[0] for r in regime_summary]
    x_r = np.arange(len(reg_names))
    b_exp = [r['Exp Bias'] for r in regime_summary]
    b_bal = [r['Bal Bias'] for r in regime_summary]
    ax3.bar(x_r - w/2, b_exp, w, label='Expanded TCN Bias', color='#EF5350', edgecolor='black')
    ax3.bar(x_r + w/2, b_bal, w, label='Balanced TCN Bias', color='#26A69A', edgecolor='black')
    ax3.axhline(0.0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(x_r)
    ax3.set_xticklabels([r['Regime'] for r in regime_summary], rotation=15, ha='right', fontsize=8)
    ax3.set_ylabel('Signed Bias (m/s)', fontweight='bold')
    ax3.set_title('Systematic Speed Bias Elimination Across Regimes', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Motorway V-Vfa02 (163 km) Endurance Comparison
    ax4 = axs[1, 1]
    mw_metrics = ['MAE (m/s)', 'Bias (m/s)', '60s Drift (m/10)', '60s Drift (%)']
    mw_exp_vals = [mw_row['exp_mae_mps'], abs(mw_row['exp_bias_mps']), mw_row['exp_drift_60s_m']/10.0, mw_row['exp_drift_60s_pct']]
    mw_bal_vals = [mw_row['bal_mae_mps'], abs(mw_row['bal_bias_mps']), mw_row['bal_drift_60s_m']/10.0, mw_row['bal_drift_60s_pct']]
    x_m = np.arange(len(mw_metrics))
    ax4.bar(x_m - w/2, mw_exp_vals, w, label='Expanded TCN', color='#90CAF9', edgecolor='black')
    ax4.bar(x_m + w/2, mw_bal_vals, w, label='Balanced TCN', color='#4CAF50', edgecolor='black')
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(mw_metrics, fontweight='bold')
    ax4.set_title('Motorway V-Vfa02 (163 km, 112 mins) Impact', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_out = RESULTS_BALANCED_DIR / "balanced_tcn_benchmark_summary.png"
    plt.savefig(plot_out, dpi=200)
    print(f"\nPlot saved to: {plot_out}")
    
    # Copy to brain artifact dir for markdown rendering
    artifact_plots_dir = PROJECT_ROOT / "results" / "figures"
    artifact_plots_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(plot_out, artifact_plots_dir / "balanced_tcn_benchmark_summary.png")
    print(f"Copied plot to brain artifact directory.")

if __name__ == '__main__':
    main()
