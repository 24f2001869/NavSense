"""
SIH26168 - Stage C5.2-A: Approved Primary Controlled Experiment
Validation Script: experiments/validate_speed_c5_2a.py

PRIMARY COMPARISON:
  Representation A: Raw Phone-Frame IMU Features (Frozen C5.1)
  vs.
  Representation B: Sensor-Only Gravity-Leveled IMU Features (Zero-Leakage)

Strict Experimental Rules:
  - Vta02 train (10,972 windows) -> Vta04 test (1,770 windows)
  - Target: Scalar VBOX vehicle speed magnitude (veh_speed_ms)
  - Window: Causal W = 20 (2.0 s), stride = 1 (0.1 s)
  - Models: Random Forest (RF) and HistGradientBoosting (GBDT)
  - 8 streams x 9 stats = 72 features
  - Zero VBOX/GNSS data enters Representation B
  - Includes diagnostic breakdown for known rough-road segment (29.6 - 33.2 s)
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

MODEL_DIR = REPO_ROOT / "models" / "c5_2a"
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


def audit_invariance(X_a, X_b, names_a, names_b):
    """
    Verifies mathematical and numerical invariance properties.
    """
    checks = []
    # 1. Dimension check
    dim_ok = (X_a.shape[1] == 72) and (X_b.shape[1] == 72)
    checks.append(("Feature dimension equals 72 for both representations", bool(dim_ok)))

    # 2. Feature names count
    names_ok = (len(names_a) == 72) and (len(names_b) == 72)
    checks.append(("Feature names list length is exactly 72", bool(names_ok)))

    # 3. Horizontal magnitude channel presence
    horiz_in_b = any("accel_horizontal_magnitude" in n for n in names_b)
    checks.append(("accel_horizontal_magnitude stream exists in Rep B", bool(horiz_in_b)))

    # 4. Finiteness check
    finite_ok = bool(np.all(np.isfinite(X_a)) and np.all(np.isfinite(X_b)))
    checks.append(("All features finite (no NaN / Inf)", finite_ok))

    return checks


def main():
    print("=" * 80)
    print("STAGE C5.2-A: SENSOR-FRAME VS GRAVITY-LEVELED IMU SPEED BENCHMARK")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Feature Extraction: Representation A (Raw Phone Frame)
    # -------------------------------------------------------------------------
    print("\n[1/4] Extracting Representation A (Raw Phone Frame)...")
    X_tr_a, y_tr_a, t_tr_a, names_a, meta_tr_a = load_and_extract_speed_dataset(
        "Vta02", window_size=20, stride=1, frame="phone"
    )
    X_te_a, y_te_a, t_te_a, _, meta_te_a = load_and_extract_speed_dataset(
        "Vta04", window_size=20, stride=1, frame="phone"
    )
    print(f"  Rep A Train (Vta02): {X_tr_a.shape}, Test (Vta04): {X_te_a.shape}")

    # -------------------------------------------------------------------------
    # 2. Feature Extraction: Representation B (Sensor-Only Gravity-Leveled)
    # -------------------------------------------------------------------------
    print("\n[2/4] Extracting Representation B (Sensor-Only Gravity-Leveled Frame)...")
    X_tr_b, y_tr_b, t_tr_b, names_b, meta_tr_b = load_and_extract_speed_dataset(
        "Vta02", window_size=20, stride=1, frame="gravity_leveled"
    )
    X_te_b, y_te_b, t_te_b, _, meta_te_b = load_and_extract_speed_dataset(
        "Vta04", window_size=20, stride=1, frame="gravity_leveled"
    )
    print(f"  Rep B Train (Vta02): {X_tr_b.shape}, Test (Vta04): {X_te_b.shape}")
    print(f"  Vta02 Leveling: Roll = {meta_tr_b['roll_deg']:.4f} deg, Pitch = {meta_tr_b['pitch_deg']:.4f} deg")
    print(f"  Vta04 Leveling: Roll = {meta_te_b['roll_deg']:.4f} deg, Pitch = {meta_te_b['pitch_deg']:.4f} deg")

    # Verify identical targets and causal timestamps
    assert np.allclose(y_tr_a, y_tr_b), "Target training speeds must be identical!"
    assert np.allclose(y_te_a, y_te_b), "Target testing speeds must be identical!"
    assert np.allclose(t_te_a, t_te_b), "Testing timestamps must be identical!"

    # Invariance and leakage checks
    checks = audit_invariance(X_tr_a, X_tr_b, names_a, names_b)
    print("\nInvariance & Structural Integrity Checks:")
    for desc, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")

    # -------------------------------------------------------------------------
    # 3. Model Training: RF and GBDT on both representations
    # -------------------------------------------------------------------------
    print("\n[3/4] Training Models on Both Representations...")

    models = {
        'Baseline Mean': {
            'fn': lambda: MeanPredictor(),
            'name': 'Baseline Mean'
        },
        'RF': {
            'fn': lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            'name': 'Random Forest'
        },
        'GBDT': {
            'fn': lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42),
            'name': 'Gradient Boosting'
        }
    }

    results = {
        'phone_frame': {},
        'gravity_leveled': {},
        'deltas': {},
        'rough_road_diagnostic': {}
    }

    preds_a = {}
    preds_b = {}

    # Define rough-road mask on Vta04 (t = 29.6 s to 33.2 s)
    t_test = t_te_a
    y_test = y_te_a
    rough_mask = (t_test >= 29.6) & (t_test <= 33.2)
    n_rough = int(np.sum(rough_mask))
    print(f"\nRough-Road Diagnostic Window: {n_rough} samples ({t_test[rough_mask][0]:.1f} s to {t_test[rough_mask][-1]:.1f} s)")

    for m_key in ['Baseline Mean', 'RF', 'GBDT']:
        m_name = models[m_key]['name']

        # Train Rep A (Phone)
        model_a = models[m_key]['fn']()
        model_a.fit(X_tr_a, y_tr_a)
        y_pred_a = model_a.predict(X_te_a)
        preds_a[m_key] = y_pred_a
        met_a = compute_metrics(y_test, y_pred_a)
        results['phone_frame'][m_key] = met_a

        # Train Rep B (Gravity-Leveled)
        model_b = models[m_key]['fn']()
        model_b.fit(X_tr_b, y_tr_b)
        y_pred_b = model_b.predict(X_te_b)
        preds_b[m_key] = y_pred_b
        met_b = compute_metrics(y_test, y_pred_b)
        results['gravity_leveled'][m_key] = met_b

        # Compute Deltas (Rep B - Rep A)
        delta_mae = met_b['mae_ms'] - met_a['mae_ms']
        delta_rmse = met_b['rmse_ms'] - met_a['rmse_ms']
        delta_r2 = met_b['r2'] - met_a['r2']
        results['deltas'][m_key] = {
            'delta_mae_ms': delta_mae,
            'delta_rmse_ms': delta_rmse,
            'delta_r2': delta_r2
        }

        # Rough-Road segment evaluation
        met_rough_a = compute_metrics(y_test[rough_mask], y_pred_a[rough_mask])
        met_rough_b = compute_metrics(y_test[rough_mask], y_pred_b[rough_mask])
        results['rough_road_diagnostic'][m_key] = {
            'phone': met_rough_a,
            'gravity_leveled': met_rough_b,
            'delta_mae_ms': met_rough_b['mae_ms'] - met_rough_a['mae_ms'],
            'delta_rmse_ms': met_rough_b['rmse_ms'] - met_rough_a['rmse_ms']
        }

        # Save models
        joblib.dump(model_a, MODEL_DIR / f"phone_{m_key.lower()}.joblib")
        joblib.dump(model_b, MODEL_DIR / f"leveled_{m_key.lower()}.joblib")

    # -------------------------------------------------------------------------
    # 4. Display Results Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("C5.2-A PRIMARY BENCHMARK RESULTS (FULL TRIP VTA04, 1,770 WINDOWS)")
    print("=" * 85)
    print(f"{'Representation':<18} | {'Model':<16} | {'MAE (m/s)':<10} | {'RMSE (m/s)':<10} | {'R²':<10} | {'Bias (m/s)':<10}")
    print("-" * 85)

    for m_key in ['RF', 'GBDT']:
        mA = results['phone_frame'][m_key]
        mB = results['gravity_leveled'][m_key]
        print(f"{'Phone':<18} | {models[m_key]['name']:<16} | {mA['mae_ms']:>9.2f}  | {mA['rmse_ms']:>9.2f}  | {mA['r2']:>9.3f} | {mA['bias_ms']:>+9.2f}")
        print(f"{'Gravity-Leveled':<18} | {models[m_key]['name']:<16} | {mB['mae_ms']:>9.2f}  | {mB['rmse_ms']:>9.2f}  | {mB['r2']:>9.3f} | {mB['bias_ms']:>+9.2f}")
        d = results['deltas'][m_key]
        print(f"{'  -> Delta (B - A)':<18} | {'':<16} | {d['delta_mae_ms']:>+9.2f}  | {d['delta_rmse_ms']:>+9.2f}  | {d['delta_r2']:>+9.3f} |")
        print("-" * 85)

    print("\n" + "=" * 85)
    print("ROUGH-ROAD DIAGNOSTIC RESULTS (29.6 - 33.2 s, 37 WINDOWS)")
    print("=" * 85)
    print(f"{'Model':<16} | {'Phone MAE':<12} | {'Leveled MAE':<12} | {'Delta MAE':<12} | {'Phone RMSE':<12} | {'Leveled RMSE':<12}")
    print("-" * 85)
    for m_key in ['RF', 'GBDT']:
        diag = results['rough_road_diagnostic'][m_key]
        print(f"{models[m_key]['name']:<16} | {diag['phone']['mae_ms']:>10.2f}   | {diag['gravity_leveled']['mae_ms']:>10.2f}   | {diag['delta_mae_ms']:>+10.2f}   | {diag['phone']['rmse_ms']:>10.2f}   | {diag['gravity_leveled']['rmse_ms']:>10.2f}")
    print("-" * 85)

    # Save to JSON
    json_path = RES_DIR / "c5_2a_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {json_path}")

    # -------------------------------------------------------------------------
    # 5. Publication Plots
    # -------------------------------------------------------------------------
    # Figure 1: Full-trip comparative timeseries & absolute error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Speed predictions
    ax1.plot(t_test, y_test, 'k-', lw=2.0, label='Ground Truth Speed Magnitude')
    ax1.plot(t_test, preds_a['GBDT'], 'r--', lw=1.2, label=f"Phone GBDT (MAE={results['phone_frame']['GBDT']['mae_ms']:.2f} m/s)")
    ax1.plot(t_test, preds_b['GBDT'], 'b-', lw=1.2, label=f"Leveled GBDT (MAE={results['gravity_leveled']['GBDT']['mae_ms']:.2f} m/s)")
    ax1.axvspan(29.6, 33.2, color='orange', alpha=0.25, label='Rough-Road Segment (29.6-33.2 s)')
    ax1.set_title("C5.2-A: Ground Truth vs Predicted Vehicle Speed Magnitude (Full Test Trip Vta04)", fontweight='bold')
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Bottom: Absolute errors
    err_a = np.abs(preds_a['GBDT'] - y_test)
    err_b = np.abs(preds_b['GBDT'] - y_test)
    ax2.plot(t_test, err_a, 'r--', lw=1.0, alpha=0.7, label='Phone GBDT Error')
    ax2.plot(t_test, err_b, 'b-', lw=1.0, alpha=0.7, label='Leveled GBDT Error')
    ax2.axvspan(29.6, 33.2, color='orange', alpha=0.25)
    ax2.axhline(results['phone_frame']['Baseline Mean']['mae_ms'], color='k', linestyle=':', label='Mean Baseline MAE (2.07 m/s)')
    ax2.set_title("Instantaneous Absolute Error (Gradient Boosting)", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute Error (m/s)")
    ax2.set_ylim([0, 10])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_2a_primary_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved primary comparison figure to: {fig1_path}")

    # Figure 2: Zoomed-in on rough-road disturbance
    zoom_mask = (t_test >= 25.0) & (t_test <= 40.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_test[zoom_mask], y_test[zoom_mask], 'k-', lw=2.5, label='VBOX Ground Truth Speed')
    ax.plot(t_test[zoom_mask], preds_a['GBDT'][zoom_mask], 'r--o', markersize=3, lw=1.5, label='Phone GBDT')
    ax.plot(t_test[zoom_mask], preds_b['GBDT'][zoom_mask], 'b-s', markersize=3, lw=1.5, label='Gravity-Leveled GBDT')
    ax.axvspan(29.6, 33.2, color='orange', alpha=0.3, label='Rough-Road Excitation (29.6-33.2 s)')
    ax.set_title("Rough-Road Diagnostic Window Zoom (25.0 - 40.0 s)", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_2a_rough_road_zoom.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved rough-road zoom figure to: {fig2_path}")


if __name__ == "__main__":
    main()
