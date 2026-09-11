#!/usr/bin/env python3
"""
Continuous Inverse-Density Loss Weighting for Speed Estimation
File: src/ml/loss_weighting.py

Computes empirical density p(v) of speed targets strictly on training data
and provides a PyTorch WeightedSmoothL1Loss / WeightedHuberLoss module.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gaussian_kde


class SpeedDensityEstimator:
    """
    Empirical continuous probability density estimator for vehicle speed targets.
    Fitted strictly on training data to compute inverse-density weights.
    """
    def __init__(self, clip_min: float = 0.5, clip_max: float = 4.0):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.kde = None
        self.mean_weight = 1.0

    def fit(self, speeds: np.ndarray):
        """Fit Gaussian KDE on 1D speed targets."""
        # Subsample to 50k points if massive for fast KDE fitting
        if len(speeds) > 50000:
            sample_idx = np.random.choice(len(speeds), 50000, replace=False)
            fit_data = speeds[sample_idx]
        else:
            fit_data = speeds

        # Add tiny jitter to pure zeros to prevent singular matrix in KDE
        jitter = np.random.normal(0, 1e-4, size=len(fit_data)).astype(np.float32)
        self.kde = gaussian_kde(fit_data + jitter, bw_method=0.15)
        
        # Compute mean weight on training sample for normalization
        eval_speeds = np.clip(fit_data, 0.0, 45.0)
        densities = self.kde.evaluate(eval_speeds)
        raw_weights = 1.0 / np.sqrt(np.maximum(densities, 1e-5))
        self.mean_weight = float(np.mean(raw_weights))
        print(f"SpeedDensityEstimator fitted: mean raw weight = {self.mean_weight:.4f}")

    def compute_weights(self, speeds: np.ndarray) -> np.ndarray:
        """Compute normalized inverse-density weights w(v) for given speeds."""
        if self.kde is None:
            return np.ones_like(speeds, dtype=np.float32)
        
        eval_speeds = np.clip(speeds, 0.0, 45.0)
        densities = self.kde.evaluate(eval_speeds)
        raw_weights = 1.0 / np.sqrt(np.maximum(densities, 1e-5))
        norm_weights = raw_weights / max(self.mean_weight, 1e-6)
        return np.clip(norm_weights, self.clip_min, self.clip_max).astype(np.float32)


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) Loss with sample-wise inverse-density speed weighting.
    L(y_pred, y_true) = mean( w(y_true) * SmoothL1(y_pred, y_true) )
    """
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        # Base SmoothL1 loss per element
        diff = torch.abs(pred - target)
        if self.beta < 1e-5:
            loss = diff
        else:
            loss = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        
        if weights is not None:
            loss = loss * weights
            
        return torch.mean(loss)
