"""
SIH26168 - Stage C5.1: Model Architectures for AI Forward-Speed Estimation
Module: src/ml/speed_models.py

Defines standard model architectures, wrappers, and baseline predictors
for forward speed estimation from IMU features.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MeanPredictor(BaseEstimator, RegressorMixin):
    """
    Trivial baseline predicting the global mean speed observed in the training set.
    """
    def __init__(self):
        self.mean_speed_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_speed_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_speed_, dtype=np.float64)


class ScaledMLPWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps an MLPRegressor with a dedicated StandardScaler that is fitted
    exclusively on the training set. Prevents data leakage.
    """
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        max_iter=250,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change

        self.scaler_ = StandardScaler()
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_.fit_transform(X)
        self.mlp_.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        preds = self.mlp_.predict(X_scaled)
        return np.clip(preds, 0.0, None)


def build_c5_1_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates the three model families along with the trivial mean baseline.
    """
    return {
        "Baseline Mean": MeanPredictor(),
        "Model A (Random Forest)": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        ),
        "Model B (Gradient Boosting)": HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=8,
            random_state=random_state
        ),
        "Model C (Small MLP)": ScaledMLPWrapper(
            hidden_layer_sizes=(64, 32),
            max_iter=250,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10
        )
    }
