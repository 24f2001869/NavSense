"""
SIH26168 - Step 6: AI Velocity Predictor Inference & Low-Pass Filter
Loads the trained model pipeline and provides real-time single-step or batch
forward speed predictions from raw or vehicle-aligned IMU data.
"""

from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "velocity_model.joblib"

class AIVelocityPredictor:
    def __init__(self, model_path=MODEL_PATH, ema_alpha=0.08):
        self.pipeline = joblib.load(model_path)
        self.scaler = self.pipeline["scaler"]
        self.model = self.pipeline["model"]
        self.feature_names = self.pipeline["feature_names"]
        self.ema_alpha = ema_alpha
        self.prev_speed = 0.0

    def predict_batch(self, X):
        """
        Batch prediction with causal exponential moving average (EMA) smoothing.
        """
        X_scaled = self.scaler.transform(X)
        preds_raw = np.clip(self.model.predict(X_scaled), 0.0, None)

        smoothed = np.zeros_like(preds_raw)
        smoothed[0] = preds_raw[0]
        for i in range(1, len(preds_raw)):
            smoothed[i] = self.ema_alpha * preds_raw[i] + (1.0 - self.ema_alpha) * smoothed[i - 1]

        return smoothed

    def predict_step(self, features_1d):
        """
        Single-step prediction for real-time 10 Hz smartphone inference.
        """
        feat_scaled = self.scaler.transform(np.asarray(features_1d).reshape(1, -1))
        pred_raw = float(np.clip(self.model.predict(feat_scaled)[0], 0.0, None))
        self.prev_speed = self.ema_alpha * pred_raw + (1.0 - self.ema_alpha) * self.prev_speed
        return self.prev_speed
