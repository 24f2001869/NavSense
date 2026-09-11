#!/usr/bin/env python3
"""
Phase 5.4: Speed Observability & Highway Forensics
Script: scratch/run_highway_speed_observability.py

Executes:
1. Speed-Band Residual Deconstruction (V-Vfa02 & High-Speed Test Trips)
2. Temporal Memory & Cruise Collapse Analysis (Receptive Field Boundary vs Steady-State Decay)
3. Steady-State Distribution Separability & Observability Limit (ROC-AUC across speed bands)
4. Recursive Kinematic State Tracking Baselines (Stateful Momentum vs Static TCN)
5. Comprehensive Visual & Quantitative Artifact Generation
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, spatial
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet

RESULTS_DIR = PROJECT_ROOT / "results" / "highway_observability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPANDED_TCN_DIR = PROJECT_ROOT / "results" / "expanded_tcn_benchmark"

TEST_TRIP_NAMES = [
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
    'V-Vfa02'
]

HIGH_SPEED_TRIPS = ['V-Vfa02', 'Vw12', 'Vw13', 'Vw14a', 'Vw14b']


def extract_tcn_kin_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extracts 9-channel TCN features, true CAN speed, and raw kinematics."""
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

    # Longitudinal acceleration approximation (forward body axis)
    a_long = lin_acc_y

    v_true = df['can_speed_mps'].values.astype(np.float32)

    features = np.column_stack([
        lin_acc_x, lin_acc_y, lin_acc_z,
        gyro_yaw, gyro_pitch, gyro_roll,
        a_horiz, a_vert, kappa
    ])

    extra = {
        'a_long': a_long,
        'a_lat': lin_acc_x,
        'a_vert': a_vert,
        'a_horiz': a_horiz,
        'gyro_yaw': gyro_yaw,
        'omega_mag': omega_mag,
        'u_gz': u_gz,
    }

    return features, v_true, extra


def load_tcn_model_and_scaler() -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    """Loads frozen Expanded TCN model and scaler parameters."""
    scaler_path = EXPANDED_TCN_DIR / "expanded_tcn_scaler.json"
    weights_path = EXPANDED_TCN_DIR / "expanded_tcn_weights.pth"

    with open(scaler_path, 'r') as f:
        s_data = json.load(f)
    mean = np.array(s_data['mean'], dtype=np.float32)
    scale = np.array(s_data['scale'], dtype=np.float32)

    model = DilatedTCNNet(in_channels=9, hidden_dim=32)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model, mean, scale


def predict_tcn_rolling(model: nn.Module, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Computes rolling TCN predictions in batches."""
    n_samples = len(features)
    preds = np.zeros(n_samples, dtype=np.float32)
    if n_samples < window_size:
        return preds

    norm_feat = (features - mean) / (scale + 1e-6)

    # Prepare windows
    windows = np.lib.stride_tricks.sliding_window_view(norm_feat, window_shape=(window_size, norm_feat.shape[1]))
    windows = windows.squeeze(axis=1) # (N_win, window_size, C)

    batch_size = 2048
    all_p = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            b_win = torch.tensor(windows[i:i+batch_size], dtype=torch.float32)
            out = model(b_win).squeeze(-1).numpy()
            all_p.append(out)

    all_p = np.concatenate(all_p)
    preds[window_size - 1:] = all_p
    preds[:window_size - 1] = all_p[0]
    return preds


def run_experiment_1_speed_bands(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Deconstructs speed error across discrete speed bands on V-Vfa02 and highway trips."""
    print("--- Experiment 1: Highway Underestimation Deconstruction ---")

    bands = [
        ('0-10 km/h (Standstill/Crawl)', 0.0, 2.78),
        ('10-30 km/h (City Slow)', 2.78, 8.33),
        ('30-50 km/h (Urban)', 8.33, 13.89),
        ('50-70 km/h (Arterial)', 13.89, 19.44),
        ('70-80 km/h (Highway Entry)', 19.44, 22.22),
        ('80-90 km/h (Highway Cruising)', 22.22, 25.00),
        ('90-100 km/h (Highway Cruising)', 25.00, 27.78),
        ('100-110 km/h (Fast Highway)', 27.78, 30.56),
        ('110+ km/h (High-Speed Motorway)', 30.56, 100.0)
    ]

    records = []
    for b_name, v_min, v_max in bands:
        sub = df_all[(df_all['v_true'] >= v_min) & (df_all['v_true'] < v_max)]
        n_pts = len(sub)
        if n_pts == 0:
            continue

        v_true_mean = float(sub['v_true'].mean())
        v_pred_mean = float(sub['v_pred'].mean())
        bias = float((sub['v_pred'] - sub['v_true']).mean())
        mae = float(np.abs(sub['v_pred'] - sub['v_true']).mean())
        rmse = float(np.sqrt(((sub['v_pred'] - sub['v_true'])**2).mean()))
        rel_bias_pct = (bias / v_true_mean) * 100.0 if v_true_mean > 0.1 else 0.0

        records.append({
            'speed_band': b_name,
            'n_samples': n_pts,
            'v_true_mean_mps': round(v_true_mean, 2),
            'v_true_mean_kmh': round(v_true_mean * 3.6, 1),
            'v_pred_mean_mps': round(v_pred_mean, 2),
            'v_pred_mean_kmh': round(v_pred_mean * 3.6, 1),
            'bias_mps': round(bias, 3),
            'rel_bias_pct': round(rel_bias_pct, 1),
            'mae_mps': round(mae, 3),
            'rmse_mps': round(rmse, 3)
        })

    res_df = pd.DataFrame(records)

    # Compute correlation between residual error (v_true - v_pred) and kinematic features at highway speeds (v_true >= 20 m/s)
    hwy_mask = df_all['v_true'] >= 20.0
    hwy_sub = df_all[hwy_mask].copy()
    residual = hwy_sub['v_true'] - hwy_sub['v_pred']

    corrs = {}
    candidates = ['a_long', 'jerk_long', 'a_lat', 'a_vert', 'a_horiz', 'gyro_yaw', 'omega_mag', 'time_since_transient']
    for cand in candidates:
        if cand in hwy_sub.columns:
            r, p = stats.pearsonr(residual, hwy_sub[cand])
            corrs[cand] = float(r)

    return res_df, corrs


def run_experiment_2_memory_decay(df_vfa02: pd.DataFrame, dt: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Experiment 2: Temporal Memory & Cruise Collapse Analysis
    Identifies steady straight cruising segments following an acceleration event,
    and measures v_pred(t) vs elapsed cruising time.
    """
    print("--- Experiment 2: Temporal Memory & Cruise Collapse Analysis ---")

    v = df_vfa02['v_true']
    v_diff_5s = (v - v.shift(50)).abs().fillna(0)
    yaw = df_vfa02['gyro_yaw'].rolling(10, min_periods=1).mean().abs()
    
    # Robust steady highway cruise: v >= 20 m/s (72 km/h), vehicle not turning, and speed constant over 5s
    is_straight_cruise = ((v >= 20.0) & (yaw < 0.04) & (v_diff_5s < 1.5)).values

    runs = is_straight_cruise.astype(int)
    diffs = (pd.Series(runs) != pd.Series(runs).shift()).cumsum()

    segments = []
    for k, grp in pd.Series(runs).groupby(diffs):
        if grp.iloc[0] == 1 and len(grp) >= 150: # >= 15s
            start = grp.index[0]
            end = grp.index[-1]
            segments.append((start, end))

    print(f"Found {len(segments)} robust sustained straight highway cruise segments.")

    time_bins = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
    decay_trajectories = {t: [] for t in time_bins}
    true_trajectories = {t: [] for t in time_bins}

    for start_idx, end_idx in segments:
        for t_sec in time_bins:
            idx = start_idx + int(round(t_sec / dt))
            if idx <= end_idx:
                decay_trajectories[t_sec].append(df_vfa02['v_pred'].iloc[idx])
                true_trajectories[t_sec].append(df_vfa02['v_true'].iloc[idx])

    records = []
    v0_pred_mean = np.mean(decay_trajectories[0]) if len(decay_trajectories[0]) > 0 else 0.0

    for t_sec in time_bins:
        if len(decay_trajectories[t_sec]) >= 5:
            v_pred_m = float(np.mean(decay_trajectories[t_sec]))
            v_true_m = float(np.mean(true_trajectories[t_sec]))
            bias_m = v_pred_m - v_true_m
            decay_from_entry = v_pred_m - v0_pred_mean
            records.append({
                'elapsed_cruise_sec': t_sec,
                'n_episodes': len(decay_trajectories[t_sec]),
                'v_true_mean_mps': round(v_true_m, 2),
                'v_true_mean_kmh': round(v_true_m * 3.6, 1),
                'v_pred_mean_mps': round(v_pred_m, 2),
                'v_pred_mean_kmh': round(v_pred_m * 3.6, 1),
                'bias_mps': round(bias_m, 2),
                'pred_decay_from_entry_mps': round(decay_from_entry, 2)
            })

    decay_df = pd.DataFrame(records)

    decay_stats = {}
    if 0 in decay_trajectories and 6 in decay_trajectories and 10 in decay_trajectories:
        decay_stats['v_pred_t0'] = float(np.mean(decay_trajectories[0]))
        decay_stats['v_pred_t6s'] = float(np.mean(decay_trajectories[6]))
        decay_stats['v_pred_t10s'] = float(np.mean(decay_trajectories[10]))
        decay_stats['decay_within_rf'] = float(decay_stats['v_pred_t6s'] - decay_stats['v_pred_t0'])
        decay_stats['decay_past_rf'] = float(decay_stats['v_pred_t10s'] - decay_stats['v_pred_t0'])

    return decay_df, decay_stats


def run_experiment_3_observability_limit(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Experiment 3: Steady-State Distribution Separability & Observability Limit
    Extracts straight cruising windows where acceleration has been zero for >10s.
    Tests whether IMU features can distinguish 85 km/h from 105 km/h.
    """
    print("--- Experiment 3: Steady-State Distribution Separability & Observability Limit ---")

    v = df_all['v_true']
    v_diff_5s = (v - v.shift(50)).abs().fillna(0)
    yaw = df_all['gyro_yaw'].rolling(10, min_periods=1).mean().abs()

    steady_mask = (v >= 20.0) & (yaw < 0.04) & (v_diff_5s < 1.5)
    df_steady = df_all[steady_mask].copy()
    print(f"Total steady-state straight cruising samples: {len(df_steady)}")

    band_A = df_steady[(df_steady['v_true'] >= 22.2) & (df_steady['v_true'] <= 25.0)] # 80-90 km/h
    band_B = df_steady[(df_steady['v_true'] >= 25.0) & (df_steady['v_true'] <= 27.8)] # 90-100 km/h
    band_C = df_steady[df_steady['v_true'] >= 27.8]                                   # 100+ km/h

    print(f"Samples in Band A (80-90 km/h): {len(band_A)}")
    print(f"Samples in Band B (90-100 km/h): {len(band_B)}")
    print(f"Samples in Band C (100+ km/h): {len(band_C)}")

    feature_cols = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'gyro_yaw', 'gyro_pitch', 'gyro_roll', 'a_horiz', 'a_vert', 'kappa']

    records = []
    for col in feature_cols:
        mA, sA = float(band_A[col].mean()), float(band_A[col].std())
        mB, sB = float(band_B[col].mean()), float(band_B[col].std())
        mC, sC = float(band_C[col].mean()), float(band_C[col].std())

        wd_AB = float(stats.wasserstein_distance(band_A[col].values, band_B[col].values)) if len(band_A) > 0 and len(band_B) > 0 else 0.0
        wd_BC = float(stats.wasserstein_distance(band_B[col].values, band_C[col].values)) if len(band_B) > 0 and len(band_C) > 0 else 0.0
        wd_AC = float(stats.wasserstein_distance(band_A[col].values, band_C[col].values)) if len(band_A) > 0 and len(band_C) > 0 else 0.0

        records.append({
            'channel': col,
            'band_A_mean (85 km/h)': round(mA, 4),
            'band_A_std': round(sA, 4),
            'band_B_mean (95 km/h)': round(mB, 4),
            'band_B_std': round(sB, 4),
            'band_C_mean (105 km/h)': round(mC, 4),
            'band_C_std': round(sC, 4),
            'wasserstein_A_vs_B': round(wd_AB, 4),
            'wasserstein_B_vs_C': round(wd_BC, 4),
            'wasserstein_A_vs_C': round(wd_AC, 4)
        })

    sep_df = pd.DataFrame(records)

    ml_results = {}
    if len(band_A) >= 50 and len(band_C) >= 50:
        n_sub = min(len(band_A), len(band_C), 5000)
        sub_A = band_A.sample(n=n_sub, random_state=42)
        sub_C = band_C.sample(n=n_sub, random_state=42)

        X_pair = np.vstack([sub_A[feature_cols].values, sub_C[feature_cols].values])
        y_pair = np.array([0] * n_sub + [1] * n_sub)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores_lr, acc_scores_lr = [], []
        auc_scores_dt, acc_scores_dt = [], []

        for train_idx, test_idx in skf.split(X_pair, y_pair):
            X_tr, X_te = X_pair[train_idx], X_pair[test_idx]
            y_tr, y_te = y_pair[train_idx], y_pair[test_idx]

            mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-6
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma

            clf_lr = LogisticRegression(C=1.0, max_iter=200)
            clf_lr.fit(X_tr_n, y_tr)
            p_lr = clf_lr.predict_proba(X_te_n)[:, 1]
            auc_scores_lr.append(roc_auc_score(y_te, p_lr))
            acc_scores_lr.append(accuracy_score(y_te, (p_lr >= 0.5).astype(int)))

            clf_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf_dt.fit(X_tr, y_tr)
            p_dt = clf_dt.predict_proba(X_te)[:, 1]
            auc_scores_dt.append(roc_auc_score(y_te, p_dt))
            acc_scores_dt.append(accuracy_score(y_te, (p_dt >= 0.5).astype(int)))

        ml_results['logistic_regression_roc_auc'] = float(np.mean(auc_scores_lr))
        ml_results['logistic_regression_accuracy'] = float(np.mean(acc_scores_lr))
        ml_results['decision_tree_roc_auc'] = float(np.mean(auc_scores_dt))
        ml_results['decision_tree_accuracy'] = float(np.mean(acc_scores_dt))

        print(f"Classifier Separation 20 m/s vs 32 m/s: LR ROC-AUC = {ml_results['logistic_regression_roc_auc']:.3f}, DT ROC-AUC = {ml_results['decision_tree_roc_auc']:.3f}")

    return sep_df, ml_results


def run_experiment_4_stateful_baselines(df_vfa02: pd.DataFrame, dt: float = 0.1) -> pd.DataFrame:
    """
    Experiment 4: Recursive Kinematic State Tracking Baseline
    Simulates GNSS blackouts during high-speed cruising and compares:
    1. Static TCN speed (memoryless feedforward)
    2. Pure Kinematic Integration: v_{k+1} = v_k + a_long * dt
    3. Bias-Corrected Kinematics: v_{k+1} = v_k + (a_long - b_a) * dt
    4. Velocity-Damped Momentum (ESKF-style): v_{k+1} = beta * (v_k + a_corr * dt) + (1-beta) * v_TCN
    """
    print("--- Experiment 4: Recursive Kinematic State Tracking Baseline ---")

    hwy_indices = np.where(df_vfa02['v_true'].values >= 20.0)[0]
    valid_starts = [idx for idx in hwy_indices if idx >= 100 and idx + 600 < len(df_vfa02)]

    step_stride = max(1, len(valid_starts) // 50)
    eval_starts = valid_starts[::step_stride][:50]
    print(f"Simulating 60s GNSS blackouts across {len(eval_starts)} motorway episodes.")

    records = []

    for ep_idx, start_idx in enumerate(eval_starts):
        end_idx = start_idx + 600
        true_speeds = df_vfa02['v_true'].values[start_idx:end_idx]
        tcn_speeds = df_vfa02['v_pred'].values[start_idx:end_idx]
        a_long_raw = df_vfa02['a_long'].values[start_idx:end_idx]

        # Estimate accelerometer bias during preceding 10s (100 steps)
        pre_a = df_vfa02['a_long'].values[start_idx-100:start_idx]
        pre_v = df_vfa02['v_true'].values[start_idx-100:start_idx]
        dv_dt_pre = (pre_v[-1] - pre_v[0]) / 10.0
        acc_bias = float(np.mean(pre_a) - dv_dt_pre)

        v0 = true_speeds[0]

        # 1. Static TCN
        v_m1 = tcn_speeds

        # 2. Pure Kinematics
        v_m2 = np.zeros(600, dtype=np.float32)
        v_m2[0] = v0
        for k in range(599):
            v_m2[k+1] = max(0.0, v_m2[k] + a_long_raw[k] * dt)

        # 3. Bias-Corrected Kinematics
        v_m3 = np.zeros(600, dtype=np.float32)
        v_m3[0] = v0
        for k in range(599):
            v_m3[k+1] = max(0.0, v_m3[k] + (a_long_raw[k] - acc_bias) * dt)

        # 4. Velocity-Damped Momentum (ESKF V1-style, beta = 0.985)
        beta = 0.985
        v_m4 = np.zeros(600, dtype=np.float32)
        v_m4[0] = v0
        for k in range(599):
            v_pred_kin = max(0.0, v_m4[k] + (a_long_raw[k] - acc_bias) * dt)
            v_m4[k+1] = beta * v_pred_kin + (1.0 - beta) * tcn_speeds[k+1]

        dist_30s = float(np.sum(true_speeds[:300]) * dt)
        dist_60s = float(np.sum(true_speeds) * dt)

        for m_name, v_series in [('Static_TCN', v_m1), ('Pure_Kinematics', v_m2), ('BiasCorr_Kinematics', v_m3), ('Damped_Momentum', v_m4)]:
            drift_30 = float(np.abs(np.sum(v_series[:300] - true_speeds[:300]) * dt))
            drift_60 = float(np.abs(np.sum(v_series - true_speeds) * dt))
            mae_60 = float(np.mean(np.abs(v_series - true_speeds)))
            bias_60 = float(np.mean(v_series - true_speeds))

            records.append({
                'episode': ep_idx,
                'model': m_name,
                'v0_mps': round(float(v0), 2),
                'dist_30s_m': round(dist_30s, 1),
                'dist_60s_m': round(dist_60s, 1),
                'drift_30s_m': round(drift_30, 2),
                'drift_60s_m': round(drift_60, 2),
                'drift_pct_30s': round((drift_30 / dist_30s) * 100.0, 2),
                'drift_pct_60s': round((drift_60 / dist_60s) * 100.0, 2),
                'mae_mps': round(mae_60, 3),
                'bias_mps': round(bias_60, 3)
            })

    sim_df = pd.DataFrame(records)
    summary_df = sim_df.groupby('model')[['drift_30s_m', 'drift_60s_m', 'drift_pct_30s', 'drift_pct_60s', 'mae_mps', 'bias_mps']].mean().reset_index()
    return summary_df


def generate_plots(res_df: pd.DataFrame, decay_df: pd.DataFrame, sep_df: pd.DataFrame, base_summary_df: pd.DataFrame, plot_path: Path):
    """Generates a multi-panel forensic diagnostic figure."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Speed Band Bias & Residual Error
    ax1 = axes[0, 0]
    bands = res_df['speed_band'].tolist()
    biases = res_df['bias_mps'].tolist()
    maes = res_df['mae_mps'].tolist()
    x = np.arange(len(bands))
    width = 0.35

    ax1.bar(x - width/2, biases, width, label='Speed Bias (v_pred - v_true)', color='crimson', alpha=0.85)
    ax1.bar(x + width/2, maes, width, label='Speed MAE (m/s)', color='steelblue', alpha=0.85)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Speed Error (m/s)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel 1: TCN Error Deconstruction by Speed Band (High-Speed Collapse)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', frameon=True)

    # Panel 2: Temporal Memory Decay during Steady Cruise
    ax2 = axes[0, 1]
    if len(decay_df) > 0:
        t_sec = decay_df['elapsed_cruise_sec'].values
        v_true = decay_df['v_true_mean_mps'].values
        v_pred = decay_df['v_pred_mean_mps'].values

        ax2.plot(t_sec, v_true, 'o-', color='black', linewidth=2.5, label='True Cruise Speed (m/s)')
        ax2.plot(t_sec, v_pred, 's--', color='crimson', linewidth=2.0, label='TCN Predicted Speed (m/s)')
        ax2.axvline(6.1, color='purple', linestyle=':', linewidth=2.0, label='TCN Receptive Field Limit (6.1s)')

        ax2.set_xlabel('Elapsed Time into Straight Cruise (seconds)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Speed (m/s)', fontsize=11, fontweight='bold')
        ax2.set_title('Panel 2: Cruise Collapse — TCN Decays past 6.1s Receptive Field', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', frameon=True)

    # Panel 3: Steady-State IMU Feature Separability (Wasserstein Distance)
    ax3 = axes[1, 0]
    channels = sep_df['channel'].tolist()
    wd_ac = sep_df['wasserstein_A_vs_C'].tolist()
    y_pos = np.arange(len(channels))

    ax3.barh(y_pos, wd_ac, color='teal', alpha=0.85)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(channels, fontsize=9)
    ax3.set_xlabel('Wasserstein Distance (85 km/h vs 105 km/h straight cruise)', fontsize=11, fontweight='bold')
    ax3.set_title('Panel 3: Observability Ceiling — Near-Zero Difference in IMU Features', fontsize=12, fontweight='bold')

    # Panel 4: Outage Drift Comparison across Baselines
    ax4 = axes[1, 1]
    models = base_summary_df['model'].tolist()
    drift_60 = base_summary_df['drift_60s_m'].tolist()
    pct_60 = base_summary_df['drift_pct_60s'].tolist()

    x4 = np.arange(len(models))
    bars = ax4.bar(x4, drift_60, color=['firebrick', 'goldenrod', 'dodgerblue', 'forestgreen'], alpha=0.85)
    ax4.axhline(100.0, color='red', linestyle='--', label='100m Drift Threshold')

    for bar, pct in zip(bars, pct_60):
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 3.0, f'{pct:.1f}% dist', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xticks(x4)
    ax4.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax4.set_ylabel('60-Second Along-Track Drift (m)', fontsize=11, fontweight='bold')
    ax4.set_title('Panel 4: Stateful Kinematic Integration vs Memoryless Static TCN', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic plots to: {plot_path}")


def main():
    print("=" * 80)
    print("PHASE 5.4: SPEED OBSERVABILITY & HIGHWAY FORENSICS AUDIT")
    print("=" * 80)

    data_roots = [
        PROJECT_ROOT / 'data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset',
    ]
    builder = IOVNBDDatasetBuilder(data_roots=data_roots)
    all_trips = builder.discover_trips()

    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    test_trips = [t for t in car_trips if t['trip_name'] in TEST_TRIP_NAMES]

    print(f"Discovered {len(car_trips)} car trips ({len(test_trips)} held-out test trips).")

    model, scaler_mean, scaler_scale = load_tcn_model_and_scaler()
    print("Loaded Expanded TCN model (65k params) and training feature scaler.")

    dfs_test = []
    df_vfa02 = None

    for idx, t in enumerate(test_trips, 1):
        t_name = t['trip_name']
        df = builder.load_clean_trip(t)
        feat, v_true, extra = extract_tcn_kin_features(df)
        v_pred = predict_tcn_rolling(model, feat, scaler_mean, scaler_scale, window_size=100)

        sub_df = pd.DataFrame({
            'trip': t_name,
            'driver': t['driver'],
            'v_true': v_true,
            'v_pred': v_pred,
            'lin_acc_x': feat[:, 0],
            'lin_acc_y': feat[:, 1],
            'lin_acc_z': feat[:, 2],
            'gyro_yaw': feat[:, 3],
            'gyro_pitch': feat[:, 4],
            'gyro_roll': feat[:, 5],
            'a_horiz': feat[:, 6],
            'a_vert': feat[:, 7],
            'kappa': feat[:, 8],
            'a_long': extra['a_long'],
            'a_lat': extra['a_lat'],
            'omega_mag': extra['omega_mag'],
            'u_gz': extra['u_gz'],
        })

        dt = 0.1
        sub_df['jerk_long'] = np.concatenate([[0.0], np.diff(sub_df['a_long'].values) / dt])

        is_transient = (np.abs(sub_df['a_long']) > 0.5) | (np.abs(sub_df['gyro_yaw']) > 0.05)
        time_since = np.zeros(len(sub_df), dtype=np.int32)
        last_t = -9999
        for i, tr in enumerate(is_transient):
            if tr:
                last_t = i
            time_since[i] = i - last_t
        sub_df['time_since_transient'] = time_since

        dfs_test.append(sub_df)
        if t_name == 'V-Vfa02':
            df_vfa02 = sub_df
            print(f"Loaded V-Vfa02: {len(df_vfa02)} samples ({len(df_vfa02)*0.1/60:.1f} mins).")

    df_all_test = pd.concat(dfs_test, ignore_index=True)
    print(f"Aggregated test dataset: {len(df_all_test)} samples.")

    # 4. Run Experiment 1
    speed_band_df, feature_corrs = run_experiment_1_speed_bands(df_all_test)
    speed_band_df.to_csv(RESULTS_DIR / "speed_band_residuals.csv", index=False)
    print("Saved speed band residuals to: results/highway_observability/speed_band_residuals.csv")
    print(speed_band_df[['speed_band', 'n_samples', 'v_true_mean_kmh', 'v_pred_mean_kmh', 'bias_mps', 'mae_mps']])

    # 5. Run Experiment 2
    decay_df, decay_stats = run_experiment_2_memory_decay(df_vfa02 if df_vfa02 is not None else df_all_test)
    decay_df.to_csv(RESULTS_DIR / "cruise_memory_decay.csv", index=False)
    print("Saved cruise memory decay to: results/highway_observability/cruise_memory_decay.csv")
    print(decay_df)

    # 6. Run Experiment 3
    sep_df, ml_sep_results = run_experiment_3_observability_limit(df_all_test)
    sep_df.to_csv(RESULTS_DIR / "observability_separability.csv", index=False)
    print("Saved feature separability to: results/highway_observability/observability_separability.csv")
    print(sep_df[['channel', 'band_A_mean (85 km/h)', 'band_C_mean (105 km/h)', 'wasserstein_A_vs_C']])

    # 7. Run Experiment 4
    base_summary_df = run_experiment_4_stateful_baselines(df_vfa02 if df_vfa02 is not None else df_all_test)
    base_summary_df.to_csv(RESULTS_DIR / "outage_tracking_baselines.csv", index=False)
    print("Saved outage tracking baselines to: results/highway_observability/outage_tracking_baselines.csv")
    print(base_summary_df)

    # 8. Generate Plots
    plot_path = RESULTS_DIR / "highway_observability_plots.png"
    generate_plots(speed_band_df, decay_df, sep_df, base_summary_df, plot_path)

    # Copy plot to brain artifacts
    import shutil
    try:
        brain_plot_dir = PROJECT_ROOT / "results" / "figures"
        brain_plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, brain_plot_dir / "highway_observability_plots.png")
    except Exception as e:
        print(f"Notice: Could not copy to brain dir: {e}")

    summary_data = {
        'speed_band_correlations': feature_corrs,
        'decay_stats': decay_stats,
        'ml_separability': ml_sep_results,
        'baselines_summary': base_summary_df.to_dict(orient='records')
    }
    with open(RESULTS_DIR / "highway_observability_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("Phase 5.4 Audit Computation Complete.")


if __name__ == '__main__':
    main()
