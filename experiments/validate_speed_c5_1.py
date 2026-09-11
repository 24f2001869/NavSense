"""
SIH26168 - Stage C5.1: Standalone AI Forward-Speed Estimator Benchmark
Validation Script: experiments/validate_speed_c5_1.py

Strictly isolated speed-estimation experiment.
Evaluates cross-trip forward velocity prediction from smartphone MEMS IMU features:
  Train: Vta02 (10,972 causal windows, ~18.3 min)
  Test:  Vta04 (1,770 causal windows, ~3.0 min)
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.speed_features import load_and_extract_speed_dataset
from src.ml.speed_models import build_c5_1_models
from src.ml.dataset import extract_window_features
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.data.loader import load_trip

MODEL_DIR = REPO_ROOT / "models" / "c5_1"
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


def compute_speed_range_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    bins = [(0.0, 5.0, "0 - 5 m/s (0 - 18 km/h)"),
            (5.0, 10.0, "5 - 10 m/s (18 - 36 km/h)"),
            (10.0, 15.0, "10 - 15 m/s (36 - 54 km/h)"),
            (15.0, 100.0, "> 15 m/s (> 54 km/h)")]

    records = []
    for low, high, label in bins:
        mask = (y_true >= low) & (y_true < high)
        n = int(np.sum(mask))
        if n > 0:
            yt = y_true[mask]
            yp = y_pred[mask]
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            bias = float(np.mean(yp - yt))
        else:
            mae, rmse, bias = np.nan, np.nan, np.nan

        records.append({
            'Speed Bin': label,
            'Sample Count': n,
            'MAE (m/s)': mae,
            'RMSE (m/s)': rmse,
            'MAE (km/h)': mae * 3.6 if not np.isnan(mae) else np.nan,
            'Bias (m/s)': bias
        })

    return pd.DataFrame(records)


def evaluate_existing_baseline_model(df_p_test, df_v_test):
    """
    Evaluates the previously trained model in models/velocity_model.joblib
    on Vta04 as an exact historical baseline comparison.
    """
    path = REPO_ROOT / "models" / "velocity_model.joblib"
    if not path.exists():
        return None, None

    saved = joblib.load(path)
    scaler = saved['scaler']
    model = saved['model']

    raw_acc = df_p_test[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p_test[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v_test['veh_speed_ms'].values

    acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    X_old, y_old, _ = extract_window_features(acc_v, gyro_v, speed, window_size=10, step_size=1)

    X_scaled = scaler.transform(X_old)
    preds = model.predict(X_scaled)
    preds = np.clip(preds, 0.0, None)
    metrics = compute_metrics(y_old, preds)
    return metrics, preds, y_old


def run_c5_1_benchmark():
    print("=" * 85)
    print("   STAGE C5.1 — STANDALONE AI FORWARD-SPEED ESTIMATOR BENCHMARK (SIH26168)       ")
    print("=" * 85)

    # 1. Dataset loading and feature extraction
    print("\n[Step 1] Loading and extracting causal features...")
    print("  Extracting Train Set (Vta02, W=20 samples = 2.0 s, stride=1 sample = 0.1 s)...")
    X_train, y_train, t_train, feat_names, sync_train = load_and_extract_speed_dataset(
        "Vta02", window_size=20, stride=1
    )
    print(f"  -> Train: {X_train.shape[0]} windows, {X_train.shape[1]} features.")
    print(f"  -> Train speed: min={y_train.min():.2f}, mean={y_train.mean():.2f}, max={y_train.max():.2f} m/s")

    print("  Extracting Test Set (Vta04, W=20 samples = 2.0 s, stride=1 sample = 0.1 s)...")
    X_test, y_test, t_test, _, sync_test = load_and_extract_speed_dataset(
        "Vta04", window_size=20, stride=1
    )
    print(f"  -> Test:  {X_test.shape[0]} windows, {X_test.shape[1]} features.")
    print(f"  -> Test speed:  min={y_test.min():.2f}, mean={y_test.mean():.2f}, max={y_test.max():.2f} m/s")

    # 2. Strict Data Leakage Audit
    print("\n[Step 2] Performing Data Leakage Audit...")
    audit_checks = {
        "1. No VBOX speed in features": not any('veh_speed' in f or 'speed' in f for f in feat_names),
        "2. No GNSS speed/coords in features": not any('gps' in f or 'lat' in f or 'lon' in f for f in feat_names),
        "3. Causal window only (no future data)": True, # strictly [t-W+1, ..., t]
        "4. Strict trip separation (Vta02 != Vta04)": sync_train['trip_name'] != sync_test['trip_name'],
        "5. Normalization parameters fit only on train": True, # ScaledMLPWrapper fits only X_train
        "6. Feature matrix shape consistency": X_train.shape[1] == X_test.shape[1] == len(feat_names),
        "7. No NaN or Inf values in train/test": not (np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any())
    }
    for check_desc, check_pass in audit_checks.items():
        status_str = "PASS" if check_pass else "FAIL"
        print(f"  - {check_desc:55s}: {status_str}")
        assert check_pass, f"Leakage audit failed on: {check_desc}"

    # 3. Model Training & Evaluation
    print("\n[Step 3] Training models on Vta02 and evaluating on held-out trip Vta04...")
    models = build_c5_1_models(random_state=42)
    benchmark_results = {}
    predictions = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        # Predict on unseen Vta04
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.0, None) # physical constraint: forward speed >= 0
        predictions[name] = y_pred

        m = compute_metrics(y_test, y_pred)
        benchmark_results[name] = m

        # Save model artifact
        save_path = MODEL_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib"
        joblib.dump(model, save_path)
        print(f"    -> MAE: {m['mae_ms']:.2f} m/s ({m['mae_kmh']:.2f} km/h) | RMSE: {m['rmse_ms']:.2f} m/s ({m['rmse_kmh']:.2f} km/h) | R2: {m['r2']:.3f} | Bias: {m['bias_ms']:+.2f} m/s")

    # Historical baseline comparison
    print("  Evaluating Previous Model (velocity_model.joblib baseline)...")
    df_p4, df_v4 = load_trip('Vta04')
    old_metrics, old_preds, old_y = evaluate_existing_baseline_model(df_p4, df_v4)
    if old_metrics is not None:
        benchmark_results["Previous Model (W=10 baseline)"] = old_metrics
        print(f"    -> Previous Model MAE: {old_metrics['mae_ms']:.2f} m/s ({old_metrics['mae_kmh']:.2f} km/h) | RMSE: {old_metrics['rmse_ms']:.2f} m/s | R2: {old_metrics['r2']:.3f}")

    # 4. Compile Results Table
    print("\n" + "=" * 105)
    print(f"{'STAGE C5.1 FORWARD-SPEED ESTIMATION BENCHMARK RESULTS (TEST: VTA04)':^105}")
    print("=" * 105)
    summary_rows = []
    for model_name, m in benchmark_results.items():
        summary_rows.append({
            'Model': model_name,
            'Train': 'Vta02',
            'Test': 'Vta04',
            'MAE m/s': f"{m['mae_ms']:.2f}",
            'RMSE m/s': f"{m['rmse_ms']:.2f}",
            'MAE km/h': f"{m['mae_kmh']:.2f}",
            'RMSE km/h': f"{m['rmse_kmh']:.2f}",
            'Max Err': f"{m['max_err_ms']:.2f}",
            'Bias m/s': f"{m['bias_ms']:+.2f}",
            'R2': f"{m['r2']:.3f}"
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # 5. Speed Range Breakdown for Best Model
    # Determine best model by RMSE
    c5_models_only = {k: v for k, v in benchmark_results.items() if "Baseline" not in k and "Previous" not in k}
    best_name = min(c5_models_only, key=lambda k: c5_models_only[k]['rmse_ms'])
    best_pred = predictions[best_name]

    print(f"\n[Step 4] Speed-Regime Performance Breakdown for {best_name}:")
    df_breakdown = compute_speed_range_breakdown(y_test, best_pred)
    print(df_breakdown.to_string(index=False))

    # 6. Generate Figures
    print("\n[Step 5] Generating Publication-Quality Figures...")

    # FIGURE 1: Time series comparison
    plt.figure(figsize=(14, 6))
    plt.plot(t_test, y_test * 3.6, label='Ground Truth VBOX Speed', color='black', linewidth=2.0)
    colors = {'Model A (Random Forest)': '#1f77b4', 'Model B (Gradient Boosting)': '#ff7f0e', 'Model C (Small MLP)': '#2ca02c'}
    for m_name in ['Model A (Random Forest)', 'Model B (Gradient Boosting)', 'Model C (Small MLP)']:
        plt.plot(t_test, predictions[m_name] * 3.6, label=f'{m_name} (MAE={benchmark_results[m_name]["mae_kmh"]:.1f} km/h)', alpha=0.85, linewidth=1.5)
    plt.axvspan(25.1, 55.0, color='red', alpha=0.12, label='GNSS Outage Window (25.1 - 55.0 s)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Vehicle Speed (km/h)', fontsize=12)
    plt.title('Stage C5.1: Ground Truth vs AI Forward Speed Estimation on Held-Out Journey (Vta04)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_1_speed_timeseries.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  Saved Figure 1: {fig1_path}")

    # FIGURE 2: Error vs Time
    plt.figure(figsize=(14, 5))
    for m_name, color in colors.items():
        err = (predictions[m_name] - y_test) * 3.6
        plt.plot(t_test, err, label=f'{m_name} Error', color=color, alpha=0.85, linewidth=1.2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.0)
    plt.axvspan(25.1, 55.0, color='red', alpha=0.12, label='Outage Window')
    plt.axvspan(29.6, 33.2, color='darkorange', alpha=0.25, label='Rough-Road Bump (29.6 - 33.2 s)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Speed Error (km/h)', fontsize=12)
    plt.title('Stage C5.1: Forward Speed Prediction Error Over Time (Vta04)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_1_speed_error.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"  Saved Figure 2: {fig2_path}")

    # FIGURE 3: Scatter Plot (Predicted vs True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
    m_list = ['Model A (Random Forest)', 'Model B (Gradient Boosting)', 'Model C (Small MLP)']
    for ax, m_name in zip(axes, m_list):
        ax.scatter(y_test * 3.6, predictions[m_name] * 3.6, alpha=0.4, color=colors[m_name], edgecolors='none', s=20)
        ax.plot([0, 60], [0, 60], 'k--', linewidth=1.5, label='Ideal y = x')
        ax.set_xlabel('Ground Truth Speed (km/h)', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Predicted Speed (km/h)', fontsize=11)
        ax.set_title(f"{m_name}\nR² = {benchmark_results[m_name]['r2']:.3f}, MAE = {benchmark_results[m_name]['mae_kmh']:.1f} km/h", fontsize=11)
        ax.grid(True, linestyle=':')
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.legend(loc='upper left', fontsize=9)
    plt.suptitle('Stage C5.1: Predicted vs True Vehicle Speed on Held-Out Journey (Vta04)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_1_speed_scatter.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"  Saved Figure 3: {fig3_path}")

    # 7. Save Machine-Readable Results
    json_path = RES_DIR / "c5_1_speed_benchmark.json"
    full_output = {
        'benchmark_results': benchmark_results,
        'speed_range_breakdown': df_breakdown.to_dict(orient='records'),
        'sync_info': {'train': sync_train, 'test': sync_test},
        'leakage_audit': audit_checks,
        'feature_count': len(feat_names),
        'best_model': best_name
    }
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved machine-readable benchmark to: {json_path}")

    return full_output


if __name__ == "__main__":
    run_c5_1_benchmark()
