"""
SIH26168 - Stage C5.2-B: Multi-Trip Training Evaluation
Script: experiments/validate_speed_c5_2b.py

Strictly controlled experiment isolating one variable:
  Single-Trip Training (Vta02 -> Vta04)
  vs.
  Multi-Trip Training (Vta02 + Vta03 -> Vta04)

Controlled boundaries:
  - Feature representation: Frozen 72 features (W=20, stride=1, causal)
  - Target: Scalar VBOX vehicle speed magnitude (veh_speed_ms)
  - Test set: Vta04 (1,770 windows)
  - Models: Baseline Mean, Random Forest (RF), HistGradientBoosting (GBDT)
  - Hyperparameters and random seeds strictly identical
  - Zero leakage: Vta04 is strictly out-of-sample
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.speed_features import load_and_extract_speed_dataset
from src.ml.speed_models import MeanPredictor
from src.data.loader import load_trip

MODEL_DIR = REPO_ROOT / "models" / "c5_2b"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    max_err = float(np.max(np.abs(y_pred - y_true)))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    return {
        'mae_ms': mae,
        'rmse_ms': rmse,
        'mae_kmh': mae * 3.6,
        'rmse_kmh': rmse * 3.6,
        'max_err_ms': max_err,
        'bias_ms': bias,
        'r2': r2
    }


def compute_trip_distribution_stats(trip_name: str) -> dict:
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    speed = df_v['veh_speed_ms'].to_numpy()
    ax = df_p['accel_x'].to_numpy()
    ay = df_p['accel_y'].to_numpy()
    az = df_p['accel_z'].to_numpy()
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    a_horiz = np.sqrt(ax**2 + ay**2)

    gx = df_p['gyro_x'].to_numpy()
    gy = df_p['gyro_y'].to_numpy()
    gz = df_p['gyro_z'].to_numpy()
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    return {
        'trip': trip_name,
        'sample_count': int(n),
        'duration_s': float(df_p['time_s'].iloc[-1] - df_p['time_s'].iloc[0]),
        'speed_ms': {
            'mean': float(np.mean(speed)),
            'std': float(np.std(speed)),
            'median': float(np.median(speed)),
            'p25': float(np.percentile(speed, 25)),
            'p75': float(np.percentile(speed, 75)),
            'min': float(np.min(speed)),
            'max': float(np.max(speed)),
            'variance': float(np.var(speed))
        },
        'accel_mag': {
            'mean': float(np.mean(a_mag)),
            'std': float(np.std(a_mag)),
            'min': float(np.min(a_mag)),
            'max': float(np.max(a_mag))
        },
        'accel_horiz': {
            'mean': float(np.mean(a_horiz)),
            'std': float(np.std(a_horiz)),
            'max': float(np.max(a_horiz))
        },
        'gyro_mag': {
            'mean': float(np.mean(g_mag)),
            'std': float(np.std(g_mag)),
            'max': float(np.max(g_mag))
        },
        'raw_speed': speed,
        'raw_a_mag': a_mag,
        'raw_a_horiz': a_horiz,
        'raw_g_mag': g_mag
    }


def main():
    print("=" * 85)
    print("STAGE C5.2-B: MULTI-TRIP TRAINING EVALUATION (Vta02 + Vta03 -> Vta04)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 1. Domain Distribution Diagnostic Audit
    # -------------------------------------------------------------------------
    print("\n[1/5] Auditing Domain Distributions P(Vta02), P(Vta03), P(Vta04)...")
    trips = ["Vta02", "Vta03", "Vta04"]
    dist_stats = {}
    for t in trips:
        dist_stats[t] = compute_trip_distribution_stats(t)
        sp = dist_stats[t]['speed_ms']
        print(f"  Trip {t} ({dist_stats[t]['sample_count']} pts, {dist_stats[t]['duration_s']:.1f} s):")
        print(f"    Speed: Mean={sp['mean']:.2f} m/s, Std={sp['std']:.2f}, Median={sp['median']:.2f}, Range=[{sp['min']:.2f}, {sp['max']:.2f}] m/s, Var={sp['variance']:.2f} (m/s)^2")
        print(f"    Accel Horiz Norm: Mean={dist_stats[t]['accel_horiz']['mean']:.2f} m/s^2, Std={dist_stats[t]['accel_horiz']['std']:.2f}")
        print(f"    Gyro Magnitude:   Mean={dist_stats[t]['gyro_mag']['mean']:.3f} rad/s, Std={dist_stats[t]['gyro_mag']['std']:.3f}")

    # Plot domain distribution diagnostics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Vta02': 'blue', 'Vta03': 'green', 'Vta04': 'red'}

    for t in trips:
        d = dist_stats[t]
        c = colors[t]
        lbl = f"{t} (mean={d['speed_ms']['mean']:.1f} m/s)"

        # Panel 1: Speed Distribution
        axs[0, 0].hist(d['raw_speed'], bins=30, density=True, alpha=0.35, color=c, label=lbl)
        axs[0, 0].set_title("Ground Truth Speed Distribution P(v)", fontweight='bold')
        axs[0, 0].set_xlabel("Speed (m/s)")
        axs[0, 0].set_ylabel("Density")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Panel 2: Horizontal Accel Distribution
        axs[0, 1].hist(d['raw_a_horiz'], bins=30, density=True, alpha=0.35, color=c, label=t)
        axs[0, 1].set_title("Horizontal Acceleration Norm P(||a_h||)", fontweight='bold')
        axs[0, 1].set_xlabel("Acceleration (m/s²)")
        axs[0, 1].set_ylabel("Density")
        axs[0, 1].set_xlim([0, 8])
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # Panel 3: Total Accel Norm Distribution
        axs[1, 0].hist(d['raw_a_mag'], bins=30, density=True, alpha=0.35, color=c, label=t)
        axs[1, 0].set_title("Total Acceleration Norm P(||a||)", fontweight='bold')
        axs[1, 0].set_xlabel("Acceleration (m/s²)")
        axs[1, 0].set_ylabel("Density")
        axs[1, 0].set_xlim([8, 14])
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # Panel 4: Gyro Magnitude Distribution
        axs[1, 1].hist(d['raw_g_mag'], bins=30, density=True, alpha=0.35, color=c, label=t)
        axs[1, 1].set_title("Gyroscope Magnitude P(||ω||)", fontweight='bold')
        axs[1, 1].set_xlabel("Angular Rate (rad/s)")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].set_xlim([0, 1.2])
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

    plt.tight_layout()
    dist_fig_path = FIG_DIR / "c5_2b_domain_distributions.png"
    plt.savefig(dist_fig_path, dpi=200)
    plt.close()
    print(f"  Saved domain distribution figure to: {dist_fig_path}")

    # Save distribution JSON (excluding numpy arrays)
    dist_json = {}
    for t in trips:
        d = dict(dist_stats[t])
        for k in ['raw_speed', 'raw_a_mag', 'raw_a_horiz', 'raw_g_mag']:
            del d[k]
        dist_json[t] = d
    dist_json_path = RES_DIR / "c5_2b_domain_distributions.json"
    with open(dist_json_path, "w") as f:
        json.dump(dist_json, f, indent=2)
    print(f"  Saved domain distribution metrics to: {dist_json_path}")

    # -------------------------------------------------------------------------
    # 2. Extract Features for All Trips (Frozen C5.1 Pipeline)
    # -------------------------------------------------------------------------
    print("\n[2/5] Extracting Frozen 72 Causal Features for Trips...")
    X_02, y_02, t_02, names_02, _ = load_and_extract_speed_dataset("Vta02", window_size=20, stride=1, frame="phone")
    X_03, y_03, t_03, names_03, _ = load_and_extract_speed_dataset("Vta03", window_size=20, stride=1, frame="phone")
    X_04, y_04, t_04, names_04, _ = load_and_extract_speed_dataset("Vta04", window_size=20, stride=1, frame="phone")

    print(f"  Vta02 (Train Trip 1): {X_02.shape} windows, mean speed = {np.mean(y_02):.2f} m/s")
    print(f"  Vta03 (Train Trip 2): {X_03.shape} windows, mean speed = {np.mean(y_03):.2f} m/s")
    print(f"  Vta04 (Frozen Test):  {X_04.shape} windows, mean speed = {np.mean(y_04):.2f} m/s")

    # Construct Training Sets
    X_train_single = X_02
    y_train_single = y_02

    X_train_multi = np.vstack([X_02, X_03])
    y_train_multi = np.concatenate([y_02, y_03])

    X_test = X_04
    y_test = y_04
    t_test = t_04

    print(f"\n  Single-Trip Training Pool: {X_train_single.shape} windows, Mean Speed = {np.mean(y_train_single):.2f} m/s")
    print(f"  Multi-Trip Training Pool:  {X_train_multi.shape} windows, Mean Speed = {np.mean(y_train_multi):.2f} m/s")
    print(f"  Frozen Test Set (Vta04):   {X_test.shape} windows, Mean Speed = {np.mean(y_test):.2f} m/s")

    # Anti-leakage checks
    assert len(X_train_multi) == len(X_02) + len(X_03)
    assert not np.any(np.isnan(X_train_multi))
    assert not np.any(np.isnan(X_test))

    # -------------------------------------------------------------------------
    # 3. Model Training & Evaluation
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Single-Trip vs Multi-Trip Models...")

    models = {
        'Baseline Mean': lambda: MeanPredictor(),
        'Random Forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
    }

    results = {
        'single_trip_vta02': {},
        'multi_trip_vta02_vta03': {},
        'deltas': {},
        'rough_road_diagnostic': {}
    }

    preds_single = {}
    preds_multi = {}

    rough_mask = (t_test >= 29.6) & (t_test <= 33.2)

    for m_key, model_fn in models.items():
        print(f"  Evaluating {m_key}...")

        # 1. Single-Trip
        m_single = model_fn()
        m_single.fit(X_train_single, y_train_single)
        y_pred_s = m_single.predict(X_test)
        preds_single[m_key] = y_pred_s
        met_s = compute_metrics(y_test, y_pred_s)
        results['single_trip_vta02'][m_key] = met_s

        # 2. Multi-Trip
        m_multi = model_fn()
        m_multi.fit(X_train_multi, y_train_multi)
        y_pred_m = m_multi.predict(X_test)
        preds_multi[m_key] = y_pred_m
        met_m = compute_metrics(y_test, y_pred_m)
        results['multi_trip_vta02_vta03'][m_key] = met_m

        # 3. Deltas (Multi - Single)
        delta_mae = met_m['mae_ms'] - met_s['mae_ms']
        delta_rmse = met_m['rmse_ms'] - met_s['rmse_ms']
        delta_r2 = met_m['r2'] - met_s['r2']
        delta_bias = met_m['bias_ms'] - met_s['bias_ms']
        results['deltas'][m_key] = {
            'delta_mae_ms': delta_mae,
            'delta_rmse_ms': delta_rmse,
            'delta_r2': delta_r2,
            'delta_bias_ms': delta_bias
        }

        # 4. Rough-Road diagnostic
        met_rough_s = compute_metrics(y_test[rough_mask], y_pred_s[rough_mask])
        met_rough_m = compute_metrics(y_test[rough_mask], y_pred_m[rough_mask])
        results['rough_road_diagnostic'][m_key] = {
            'single_trip': met_rough_s,
            'multi_trip': met_rough_m,
            'delta_mae_ms': met_rough_m['mae_ms'] - met_rough_s['mae_ms'],
            'delta_rmse_ms': met_rough_m['rmse_ms'] - met_rough_s['rmse_ms']
        }

        # Save models
        clean_name = m_key.lower().replace(" ", "_")
        joblib.dump(m_single, MODEL_DIR / f"single_{clean_name}.joblib")
        joblib.dump(m_multi, MODEL_DIR / f"multi_{clean_name}.joblib")

    # -------------------------------------------------------------------------
    # 4. Print Comparison Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("C5.2-B PRIMARY BENCHMARK RESULTS (TEST TRIP: VTA04, 1,770 WINDOWS)")
    print("=" * 90)
    print(f"{'Training Regime':<20} | {'Model':<18} | {'MAE (m/s)':<10} | {'RMSE (m/s)':<10} | {'R2':<10} | {'Bias (m/s)':<10}")
    print("-" * 90)

    for m_key in ['Baseline Mean', 'Random Forest', 'Gradient Boosting']:
        mS = results['single_trip_vta02'][m_key]
        mM = results['multi_trip_vta02_vta03'][m_key]
        d = results['deltas'][m_key]

        print(f"{'Vta02 (Single)':<20} | {m_key:<18} | {mS['mae_ms']:>9.2f}  | {mS['rmse_ms']:>9.2f}  | {mS['r2']:>9.3f} | {mS['bias_ms']:>+9.2f}")
        print(f"{'Vta02+03 (Multi)':<20} | {m_key:<18} | {mM['mae_ms']:>9.2f}  | {mM['rmse_ms']:>9.2f}  | {mM['r2']:>9.3f} | {mM['bias_ms']:>+9.2f}")
        print(f"{'  -> Delta (M - S)':<20} | {'':<18} | {d['delta_mae_ms']:>+9.2f}  | {d['delta_rmse_ms']:>+9.2f}  | {d['delta_r2']:>+9.3f} | {d['delta_bias_ms']:>+9.2f}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("ROUGH-ROAD DIAGNOSTIC RESULTS (29.6 - 33.2 s, 36 WINDOWS)")
    print("=" * 90)
    print(f"{'Model':<18} | {'Single MAE':<12} | {'Multi MAE':<12} | {'Delta MAE':<12} | {'Single RMSE':<12} | {'Multi RMSE':<12}")
    print("-" * 90)
    for m_key in ['Baseline Mean', 'Random Forest', 'Gradient Boosting']:
        diag = results['rough_road_diagnostic'][m_key]
        print(f"{m_key:<18} | {diag['single_trip']['mae_ms']:>10.2f}   | {diag['multi_trip']['mae_ms']:>10.2f}   | {diag['delta_mae_ms']:>+10.2f}   | {diag['single_trip']['rmse_ms']:>10.2f}   | {diag['multi_trip']['rmse_ms']:>10.2f}")
    print("-" * 90)

    # Save to JSON
    json_path = RES_DIR / "c5_2b_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {json_path}")

    # -------------------------------------------------------------------------
    # 5. Diagnostic Publication Plots
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Top: Speed predictions
    ax1.plot(t_test, y_test, 'k-', lw=2.0, label='Ground Truth Speed Magnitude')
    ax1.plot(t_test, preds_single['Gradient Boosting'], 'r--', lw=1.2, label=f"Single-Trip GBDT (MAE={results['single_trip_vta02']['Gradient Boosting']['mae_ms']:.2f} m/s)")
    ax1.plot(t_test, preds_multi['Gradient Boosting'], 'g-', lw=1.2, label=f"Multi-Trip GBDT (MAE={results['multi_trip_vta02_vta03']['Gradient Boosting']['mae_ms']:.2f} m/s)")
    ax1.axvspan(29.6, 33.2, color='orange', alpha=0.25, label='Rough-Road Segment (29.6-33.2 s)')
    ax1.set_title("C5.2-B: Single-Trip vs Multi-Trip Vehicle Speed Magnitude Predictions (Test Trip Vta04)", fontweight='bold')
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Bottom: Absolute errors
    err_s = np.abs(preds_single['Gradient Boosting'] - y_test)
    err_m = np.abs(preds_multi['Gradient Boosting'] - y_test)
    ax2.plot(t_test, err_s, 'r--', lw=1.0, alpha=0.7, label='Single-Trip GBDT Error')
    ax2.plot(t_test, err_m, 'g-', lw=1.0, alpha=0.7, label='Multi-Trip GBDT Error')
    ax2.axvspan(29.6, 33.2, color='orange', alpha=0.25)
    ax2.axhline(results['single_trip_vta02']['Baseline Mean']['mae_ms'], color='k', linestyle=':', label=f"Vta02 Mean Baseline ({results['single_trip_vta02']['Baseline Mean']['mae_ms']:.2f} m/s)")
    ax2.set_title("Instantaneous Absolute Error Comparison (Gradient Boosting)", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute Error (m/s)")
    ax2.set_ylim([0, 10])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig_path = FIG_DIR / "c5_2b_multitrip_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved multi-trip comparison figure to: {fig_path}")


if __name__ == "__main__":
    main()
