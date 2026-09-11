#!/usr/bin/env python3
"""
Phase 5.6: Adaptive Regime-Aware Velocity Fusion Benchmark
Script: scratch/run_adaptive_fusion_benchmark.py

Evaluates:
- Model 1: Static Dilated TCN (Memoryless baseline)
- Model 2: Pure Kinematic Integration (v_{k+1} = max(0, v_k + a_long * dt))
- Model 3: Fixed Damped Momentum (Phase 5.5 model, beta = 0.985)
- Model 4: Adaptive Regime-Aware Velocity Fusion (Phase 5.6: ZVD + Cruise Momentum + Dynamic Blending)
Across the held-out test trips at 10s, 20s, 30s, and 60s horizons.
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

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder
from src.ml.models.temporal_speed_net import DilatedTCNNet

RESULTS_DIR = PROJECT_ROOT / "results" / "adaptive_fusion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPANDED_TCN_DIR = PROJECT_ROOT / "results" / "expanded_tcn_benchmark"

TEST_TRIP_NAMES = [
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
    'V-Vfa02'
]

TRIP_CATEGORIES = {
    'V-Vfa02': 'Motorway',
    'Vta21': 'Suburban', 'Vta22': 'Suburban', 'Vta23': 'Suburban', 'Vta24': 'Suburban',
    'Vta25': 'Suburban', 'Vta26': 'Suburban', 'Vta27': 'Suburban', 'Vta28': 'Suburban',
    'Vtb09': 'Dense Urban', 'Vtb10': 'Dense Urban', 'Vtb11': 'Dense Urban', 'Vtb12': 'Dense Urban',
    'Vw12': 'Mountain', 'Vw13': 'Mountain', 'Vw14a': 'Mountain', 'Vw14b': 'Mountain',
    'Vw15': 'Stationary', 'Vw16a': 'Mountain'
}


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


def extract_features_and_raw(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extracts 9-channel features, CAN speed, forward acceleration, and raw IMU signals."""
    lin_acc_x = df['lin_acc_x'].values.astype(np.float32)
    lin_acc_y = df['lin_acc_y'].values.astype(np.float32)
    lin_acc_z = df['lin_acc_z'].values.astype(np.float32)
    gyro_yaw = df['gyro_yaw'].values.astype(np.float32)
    gyro_pitch = df['gyro_pitch'].values.astype(np.float32)
    gyro_roll = df['gyro_roll'].values.astype(np.float32)

    acc_x = df['acc_x'].values.astype(np.float32)
    acc_y = df['acc_y'].values.astype(np.float32)
    acc_z = df['acc_z'].values.astype(np.float32)

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

    features = np.column_stack([
        lin_acc_x, lin_acc_y, lin_acc_z,
        gyro_yaw, gyro_pitch, gyro_roll,
        a_horiz, a_vert, kappa
    ])

    v_true = df['can_speed_mps'].values.astype(np.float32)
    a_long = lin_acc_y

    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    raw_dict = {
        'acc_mag': acc_mag,
        'omega_mag': omega_mag,
        'a_long': a_long,
        'gyro_yaw': gyro_yaw
    }

    return features, v_true, a_long, raw_dict


def predict_tcn_rolling(model: nn.Module, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Computes rolling TCN predictions in batches."""
    n_samples = len(features)
    preds = np.zeros(n_samples, dtype=np.float32)
    if n_samples < window_size:
        return preds

    norm_feat = (features - mean) / (scale + 1e-6)
    windows = np.lib.stride_tricks.sliding_window_view(norm_feat, window_shape=(window_size, norm_feat.shape[1])).squeeze(1)

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


def evaluate_trip_adaptive(v_true: np.ndarray, v_tcn: np.ndarray, a_long: np.ndarray, raw_dict: Dict[str, np.ndarray], dt: float = 0.1, stride_sec: float = 15.0) -> List[Dict]:
    """
    Evaluates simulated blackouts across sliding windows on a single trip.
    Compares Static TCN, Pure Kinematics, Fixed Damped Momentum, and Adaptive Fusion.
    """
    n_samples = len(v_true)
    max_horizon_steps = 600
    pre_window = 100

    stride_steps = int(round(stride_sec / dt))
    valid_starts = range(pre_window, n_samples - max_horizon_steps, stride_steps)

    # Check if trip is short but can support smaller horizons
    can_full_60s = len(valid_starts) > 0
    if not can_full_60s:
        # Check if can support 10s, 20s, or 30s
        if n_samples >= pre_window + 100: # at least 20s
            max_avail_steps = min(max_horizon_steps, n_samples - pre_window - 5)
            valid_starts = range(pre_window, n_samples - max_avail_steps, stride_steps)
        else:
            return []

    # Precompute causal rolling metrics over 15 samples (1.5s)
    acc_mag_series = pd.Series(raw_dict['acc_mag'])
    omega_mag_series = pd.Series(raw_dict['omega_mag'])
    roll_std_a = acc_mag_series.rolling(15, min_periods=1).std().fillna(0).values
    roll_mean_a = acc_mag_series.rolling(15, min_periods=1).mean().values
    roll_omega = omega_mag_series.rolling(15, min_periods=1).mean().values

    horizons = [10, 20, 30, 60]
    episode_records = []

    for ep_idx, start_idx in enumerate(valid_starts):
        avail_steps = min(max_horizon_steps, n_samples - start_idx)
        if avail_steps < 100: # less than 10s
            continue

        end_idx = start_idx + avail_steps
        y_true = v_true[start_idx:end_idx]
        y_tcn = v_tcn[start_idx:end_idx]
        acc_raw = a_long[start_idx:end_idx]

        std_a_ep = roll_std_a[start_idx:end_idx]
        mean_a_ep = roll_mean_a[start_idx:end_idx]
        omega_ep = roll_omega[start_idx:end_idx]

        # Estimate pre-outage bias
        pre_a = a_long[start_idx-pre_window:start_idx]
        pre_v = v_true[start_idx-pre_window:start_idx]
        dv_dt = (pre_v[-1] - pre_v[0]) / (pre_window * dt) if len(pre_v) > 1 else 0.0
        acc_bias = float(np.mean(pre_a) - dv_dt)

        v0 = float(y_true[0])

        # Model 1: Static TCN
        v_m1 = y_tcn

        # Model 2: Pure Kinematics
        v_m2 = np.zeros(avail_steps, dtype=np.float32)
        v_m2[0] = v0
        for k in range(avail_steps - 1):
            v_m2[k+1] = max(0.0, v_m2[k] + acc_raw[k] * dt)

        # Model 3: Fixed Damped Momentum (Phase 5.5, beta = 0.985)
        v_m3 = np.zeros(avail_steps, dtype=np.float32)
        v_m3[0] = v0
        beta_fixed = 0.985
        for k in range(avail_steps - 1):
            v_pred_kin = max(0.0, v_m3[k] + (acc_raw[k] - acc_bias) * dt)
            v_m3[k+1] = beta_fixed * v_pred_kin + (1.0 - beta_fixed) * y_tcn[k+1]

        # Model 4: Adaptive Regime-Aware Velocity Fusion (Phase 5.6)
        v_m4 = np.zeros(avail_steps, dtype=np.float32)
        v_m4[0] = v0
        
        regime_history = []
        for k in range(avail_steps - 1):
            curr_v = v_m4[k]
            # Causal Regime Classification:
            is_stopped = (omega_ep[k] < 0.05) and (std_a_ep[k] < 0.16) and (abs(mean_a_ep[k] - 9.81) < 0.28) and (curr_v < 2.5 or y_tcn[k] < 2.5)
            is_stable_cruise = (curr_v > 15.0) and (omega_ep[k] < 0.04) and (std_a_ep[k] < 0.75) and (abs(acc_raw[k]) < 0.45)
            is_high_disagree = abs(curr_v - y_tcn[k]) > 4.5

            if is_stopped:
                # Regime 1: STATIONARY (ZVD)
                v_m4[k+1] = 0.0
                regime_history.append('STATIONARY')
            elif is_stable_cruise:
                # Regime 2: STABLE HIGHWAY CRUISE (High momentum weight)
                beta_adapt = 0.993
                v_kin = max(0.0, curr_v + (acc_raw[k] - acc_bias) * dt)
                v_m4[k+1] = beta_adapt * v_kin + (1.0 - beta_adapt) * y_tcn[k+1]
                regime_history.append('CRUISE')
            elif is_high_disagree:
                # Regime 4: HIGH DISAGREEMENT (Bounded blending)
                a_eff = np.clip(acc_raw[k] - acc_bias, -1.5, 1.5)
                v_kin = max(0.0, curr_v + a_eff * dt)
                beta_adapt = 0.950
                v_m4[k+1] = beta_adapt * v_kin + (1.0 - beta_adapt) * y_tcn[k+1]
                regime_history.append('DISAGREE')
            else:
                # Regime 3: MOVING DYNAMIC (Balanced complementary blending)
                beta_adapt = 0.965
                v_kin = max(0.0, curr_v + (acc_raw[k] - acc_bias) * dt)
                v_m4[k+1] = beta_adapt * v_kin + (1.0 - beta_adapt) * y_tcn[k+1]
                regime_history.append('DYNAMIC')

        models = {
            'Static_TCN': v_m1,
            'Pure_Kinematics': v_m2,
            'Fixed_Damped_Momentum': v_m3,
            'Adaptive_Regime_Fusion': v_m4
        }

        for h in horizons:
            h_steps = int(round(h / dt))
            if avail_steps < h_steps:
                continue

            dist_true = float(np.sum(y_true[:h_steps]) * dt)

            for m_name, v_series in models.items():
                dist_est = float(np.sum(v_series[:h_steps]) * dt)
                drift_m = float(np.abs(dist_est - dist_true))
                drift_pct = (drift_m / dist_true * 100.0) if dist_true > 1.0 else 0.0

                episode_records.append({
                    'episode': ep_idx,
                    'model': m_name,
                    'horizon_sec': h,
                    'v0_mps': round(v0, 2),
                    'dist_true_m': round(dist_true, 2),
                    'drift_m': round(drift_m, 2),
                    'drift_pct': round(drift_pct, 2),
                    'is_pass_10pct': 1 if (drift_pct < 10.0 and dist_true > 1.0) else (1 if dist_true <= 1.0 and drift_m < 5.0 else 0)
                })

    return episode_records


def generate_diagnostic_plots(per_trip_df: pd.DataFrame, agg_summary_df: pd.DataFrame, plot_path: Path):
    """Generates a 4-panel diagnostic figure comparing adaptive fusion against baselines."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    horizons = [10, 20, 30, 60]
    models = ['Pure_Kinematics', 'Static_TCN', 'Fixed_Damped_Momentum', 'Adaptive_Regime_Fusion']
    colors = ['gray', 'crimson', 'dodgerblue', 'forestgreen']

    # Panel 1: Multi-Horizon Drift % Across Test Trips
    ax1 = axes[0, 0]
    for m, c in zip(models, colors):
        sub = agg_summary_df[agg_summary_df['model'] == m]
        if len(sub) > 0:
            pcts = []
            for h in horizons:
                h_sub = sub[sub['horizon_sec'] == h]
                pcts.append(float(h_sub['mean_drift_pct'].iloc[0]) if len(h_sub) > 0 else np.nan)
            ax1.plot(horizons, pcts, 'o-', linewidth=2.5, label=m, color=c)

    ax1.axhline(10.0, color='black', linestyle='--', linewidth=1.5, label='SIH 10% Benchmark')
    ax1.set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Drift (% of distance)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel 1: Aggregate Drift % Across Tested Horizons', fontsize=12, fontweight='bold')
    ax1.set_xticks(horizons)
    ax1.set_ylim(0, 45)
    ax1.legend(loc='upper left', frameon=True)

    # Panel 2: SIH Pass Count (<10% Drift) across Horizons
    ax2 = axes[0, 1]
    width = 0.18
    x = np.arange(len(horizons))

    for idx, (m, c) in enumerate(zip(models, colors)):
        sub = agg_summary_df[agg_summary_df['model'] == m]
        passes = []
        for h in horizons:
            h_sub = sub[sub['horizon_sec'] == h]
            passes.append(int(h_sub['trips_passed_10pct'].iloc[0]) if len(h_sub) > 0 else 0)
        bars = ax2.bar(x + (idx - 1.5)*width, passes, width, label=m, color=c, alpha=0.85)
        for b, val in zip(bars, passes):
            ax2.text(b.get_x() + b.get_width()/2.0, val + 0.3, f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h}s Horizon' for h in horizons], fontsize=10, fontweight='bold')
    ax2.set_ylabel('Passing Trips (<10% Drift)', fontsize=11, fontweight='bold')
    ax2.set_title('Panel 2: Number of Trips Passing SIH <10% Criterion', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 15)
    ax2.legend(loc='upper right', frameon=True)

    # Panel 3: Suburban Drift Suppression (Vta21 to Vta28 at 30s)
    ax3 = axes[1, 0]
    sub_30s = per_trip_df[(per_trip_df['horizon_sec'] == 30) & (per_trip_df['category'] == 'Suburban')]
    sub_trips = sorted(sub_30s['trip'].unique().tolist())
    x3 = np.arange(len(sub_trips))

    for idx, (m, c) in enumerate(zip(['Pure_Kinematics', 'Fixed_Damped_Momentum', 'Adaptive_Regime_Fusion'], ['gray', 'dodgerblue', 'forestgreen'])):
        vals = []
        for tr in sub_trips:
            v_sub = sub_30s[(sub_30s['trip'] == tr) & (sub_30s['model'] == m)]['drift_pct'].values
            vals.append(float(v_sub[0]) if len(v_sub) > 0 else 0.0)
        ax3.bar(x3 + (idx - 1)*0.25, vals, 0.25, label=m, color=c, alpha=0.85)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(sub_trips, rotation=35, ha='right', fontsize=9)
    ax3.set_ylabel('30s Drift (% distance)', fontsize=11, fontweight='bold')
    ax3.set_title('Panel 3: Suburban Drift Suppression via ZVD & Adaptive Damping', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', frameon=True)

    # Panel 4: Motorway (V-Vfa02) Drift Progression
    ax4 = axes[1, 1]
    vfa02_sub = per_trip_df[per_trip_df['trip'] == 'V-Vfa02']
    for m, c in zip(models, colors):
        sub_m = vfa02_sub[vfa02_sub['model'] == m]
        if len(sub_m) > 0:
            drift_pct = [float(sub_m[sub_m['horizon_sec'] == h]['drift_pct'].iloc[0]) for h in horizons]
            ax4.plot(horizons, drift_pct, 's--', linewidth=2.0, label=m, color=c)

    ax4.axhline(10.0, color='black', linestyle='--', linewidth=1.2, label='10% Target')
    ax4.set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Motorway (V-Vfa02) Drift (% distance)', fontsize=11, fontweight='bold')
    ax4.set_title('Panel 4: V-Vfa02 Motorway Drift across Horizons', fontsize=12, fontweight='bold')
    ax4.set_xticks(horizons)
    ax4.set_ylim(0, 30)
    ax4.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic plots to: {plot_path}")


def main():
    print("=" * 80)
    print("PHASE 5.6: ADAPTIVE REGIME-AWARE VELOCITY FUSION BENCHMARK")
    print("=" * 80)

    # 1. Discover data
    data_roots = [
        PROJECT_ROOT / 'data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset',
    ]
    builder = IOVNBDDatasetBuilder(data_roots=data_roots)
    all_trips = builder.discover_trips()

    test_trips = [t for t in all_trips if t['trip_name'] in TEST_TRIP_NAMES]
    print(f"Discovered {len(test_trips)} held-out test trips.")

    # 2. Load TCN model & feature scaler
    model, scaler_mean, scaler_scale = load_tcn_model_and_scaler()
    print("Loaded Expanded TCN model (65k params) and training feature scaler.")

    all_episode_records = []
    per_trip_summary_records = []

    for idx, t in enumerate(test_trips, 1):
        t_name = t['trip_name']
        t_cat = TRIP_CATEGORIES.get(t_name, 'Unknown')
        print(f"[{idx:02d}/19] Processing {t_name:<8s} ({t_cat})...")

        df = builder.load_clean_trip(t)
        feat, v_true, a_long, raw_dict = extract_features_and_raw(df)
        v_tcn = predict_tcn_rolling(model, feat, scaler_mean, scaler_scale, window_size=100)

        ep_records = evaluate_trip_adaptive(v_true, v_tcn, a_long, raw_dict, dt=0.1, stride_sec=15.0)

        if len(ep_records) == 0:
            print(f"  Notice: Trip {t_name} duration too short for blackout simulation ({len(df)*0.1:.1f}s).")
            continue

        ep_df = pd.DataFrame(ep_records)
        ep_df['trip'] = t_name
        ep_df['category'] = t_cat
        all_episode_records.append(ep_df)

        for (m, h), grp in ep_df.groupby(['model', 'horizon_sec']):
            mean_drift_m = float(grp['drift_m'].mean())
            mean_dist_m = float(grp['dist_true_m'].mean())
            mean_drift_pct = float(grp['drift_pct'].mean())
            pass_rate = float(grp['is_pass_10pct'].mean()) * 100.0
            is_trip_pass = 1 if mean_drift_pct < 10.0 else 0

            per_trip_summary_records.append({
                'trip': t_name,
                'category': t_cat,
                'model': m,
                'horizon_sec': h,
                'n_episodes': len(grp),
                'mean_dist_m': round(mean_dist_m, 2),
                'drift_m': round(mean_drift_m, 2),
                'drift_pct': round(mean_drift_pct, 2),
                'episode_pass_rate_pct': round(pass_rate, 1),
                'is_trip_pass': is_trip_pass
            })

    # Save per-trip results
    per_trip_df = pd.DataFrame(per_trip_summary_records)
    per_trip_df.to_csv(RESULTS_DIR / "adaptive_per_trip_results.csv", index=False)
    print("\nSaved per-trip results to: results/adaptive_fusion/adaptive_per_trip_results.csv")

    # Aggregate cross-trip summary
    agg_records = []
    horizons = [10, 20, 30, 60]
    models = ['Static_TCN', 'Pure_Kinematics', 'Fixed_Damped_Momentum', 'Adaptive_Regime_Fusion']

    for h in horizons:
        for m in models:
            sub = per_trip_df[(per_trip_df['horizon_sec'] == h) & (per_trip_df['model'] == m)]
            dyn_sub = sub[sub['category'] != 'Stationary']

            mean_drift_m = float(sub['drift_m'].mean()) if len(sub) > 0 else np.nan
            mean_drift_pct = float(dyn_sub['drift_pct'].mean()) if len(dyn_sub) > 0 else np.nan
            trips_passed = int(sub['is_trip_pass'].sum()) if len(sub) > 0 else 0
            total_trips = len(sub)

            agg_records.append({
                'horizon_sec': h,
                'model': m,
                'total_trips': total_trips,
                'mean_drift_m': round(mean_drift_m, 2),
                'mean_drift_pct': round(mean_drift_pct, 2),
                'trips_passed_10pct': trips_passed,
                'trip_pass_rate_pct': round(trips_passed / total_trips * 100.0, 1) if total_trips > 0 else 0.0
            })

    agg_summary_df = pd.DataFrame(agg_records)
    agg_summary_df.to_csv(RESULTS_DIR / "adaptive_aggregate_summary.csv", index=False)
    print("Saved aggregate summary to: results/adaptive_fusion/adaptive_aggregate_summary.csv")

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY ACROSS USABLE HELD-OUT TRIPS:")
    print("=" * 80)
    print(agg_summary_df[['horizon_sec', 'model', 'total_trips', 'mean_drift_m', 'mean_drift_pct', 'trips_passed_10pct', 'trip_pass_rate_pct']])

    # 3. Generate diagnostic figures
    plot_path = RESULTS_DIR / "adaptive_fusion_plots.png"
    generate_diagnostic_plots(per_trip_df, agg_summary_df, plot_path)

    # Copy plot to brain artifacts
    import shutil
    try:
        brain_plot_dir = PROJECT_ROOT / "results" / "figures"
        brain_plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, brain_plot_dir / "adaptive_fusion_plots.png")
    except Exception as e:
        print(f"Notice: Could not copy to brain dir: {e}")

    # 4. Summary JSON
    summary_data = {
        'aggregate_metrics': agg_summary_df.to_dict(orient='records'),
        'vfa02_metrics': per_trip_df[per_trip_df['trip'] == 'V-Vfa02'].to_dict(orient='records')
    }
    with open(RESULTS_DIR / "adaptive_fusion_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("\nPhase 5.6 Execution Finished Successfully.")


if __name__ == '__main__':
    main()
