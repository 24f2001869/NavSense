"""
Classical and Shallow ML Speed Estimation Baselines for IO-VNBD.

Implements:
1. Baseline 0: Raw Phone GPS Speed (baseline ceiling when GPS is active).
2. Baseline 1: Classical Acceleration Integration + ZUPT (physical floor).
3. Baseline 2: Ridge Regression on Branch A Features (linear benchmark).
4. Baseline 3: Random Forest Regressor on Branch A Features (non-linear reference).
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ClassicalIntegrationBaseline:
    """
    Naive double integration with stationary Zero Velocity Update (ZUPT).
    Floor baseline demonstrating physics drift without learned corrections.
    """
    def __init__(self, dt: float = 0.1, zupt_std_thresh: float = 0.25):
        self.dt = dt
        self.zupt_std_thresh = zupt_std_thresh
        
    def predict_trip(self, lin_acc_x: np.ndarray, lin_acc_mag: np.ndarray, initial_speed: float = 0.0) -> np.ndarray:
        n = len(lin_acc_x)
        v_est = np.zeros(n, dtype=np.float32)
        v_est[0] = initial_speed
        
        # Rolling variance over 0.5s (5 samples)
        window = 5
        rolling_std = np.zeros(n)
        for i in range(n):
            start = max(0, i - window)
            rolling_std[i] = np.std(lin_acc_mag[start:i+1])
            
        for i in range(1, n):
            # If acceleration variance is very low and raw acceleration near 0 -> ZUPT
            if rolling_std[i] < self.zupt_std_thresh and abs(lin_acc_x[i]) < 0.2:
                v_est[i] = 0.0
            else:
                v_est[i] = max(0.0, v_est[i-1] + lin_acc_x[i] * self.dt)
                
        return v_est


class RidgeFeatureBaseline:
    """L2 Regularized Linear Regression on Branch A Window Features."""
    def __init__(self, alpha: float = 1.0):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha, random_state=42)),
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 50.0)  # Vehicle speed non-negative


class RandomForestFeatureBaseline:
    """Non-linear Ensemble Random Forest on Branch A Window Features."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 14, n_jobs: int = -1):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=n_jobs,
                random_state=42,
            )),
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 50.0)
