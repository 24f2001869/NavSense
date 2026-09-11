"""
SIH26168 - Step 6: AI Forward Velocity Model Training & Evaluation
Trains and compares regression models on IMU vibration features against vehicle ground truth speed.
Uses strict trip-level train/test separation to eliminate data leakage.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.ml.dataset import load_and_preprocess_dataset

MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(train_trips=["Vta02"], test_trips=["Vta04"]):
    print(f"Loading training data from: {train_trips}...")
    X_train, y_train, feature_names = load_and_preprocess_dataset(train_trips, window_size=10, step_size=1)
    print(f"Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    print(f"Loading test data from: {test_trips}...")
    X_test, y_test, _ = load_and_preprocess_dataset(test_trips, window_size=10, step_size=1)
    print(f"Test samples: {X_test.shape[0]}")

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=42)
    }

    results = {}
    best_model_name = None
    best_rmse = float("inf")
    best_preds = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        preds = np.clip(preds, 0.0, None) # Speeds cannot be negative

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE (m/s)": mae, "RMSE (m/s)": rmse, "R2 Score": r2}
        print(f"  Test MAE: {mae:.2f} m/s ({mae*3.6:.2f} km/h)")
        print(f"  Test RMSE: {rmse:.2f} m/s ({rmse*3.6:.2f} km/h)")
        print(f"  Test R2: {r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_preds = preds

    print(f"\n>>> Best Model: {best_model_name} (RMSE = {best_rmse:.2f} m/s) <<<")

    # Save pipeline (scaler + best model)
    pipeline = {
        "model_name": best_model_name,
        "scaler": scaler,
        "model": models[best_model_name],
        "feature_names": feature_names,
        "metrics": results[best_model_name]
    }
    model_path = MODEL_DIR / "velocity_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved best model pipeline to: {model_path}")

    # Plot test prediction timeseries
    t_test = np.arange(len(y_test)) * 0.1
    plt.figure(figsize=(14, 5))
    plt.plot(t_test, y_test * 3.6, label='Ground Truth True Speed (km/h)', color='#2ca02c', lw=2.0)
    plt.plot(t_test, best_preds * 3.6, label=f'AI Predicted Speed ({best_model_name})', color='#d62728', lw=1.5, ls='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title(f'AI Velocity Estimation on Unseen Journey ({test_trips[0]}) | R² = {results[best_model_name]["R2 Score"]:.3f}', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    out_fig = FIG_DIR / "ai_velocity_prediction.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved AI velocity plot: {out_fig}")

    return results

if __name__ == "__main__":
    train_and_evaluate()
