#!/usr/bin/env python3
"""
Speed-Stratified Sampling for Imbalance Elimination
File: src/ml/speed_stratified_sampler.py

Computes sampling weights inversely proportional to speed regime frequency,
ensuring balanced representation across standstill, low, medium, and motorway regimes.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def create_speed_stratified_sampler(y_train: np.ndarray, num_samples: int = None) -> Tuple[WeightedRandomSampler, dict]:
    """
    Creates a PyTorch WeightedRandomSampler that equalizes sampling frequency
    across 4 distinct speed regimes:
      1. Standstill & Crawl (0 <= v < 5.0 m/s)
      2. City & Suburban    (5.0 <= v < 15.0 m/s)
      3. Fast Arterial      (15.0 <= v < 25.0 m/s)
      4. Motorway Extreme   (v >= 25.0 m/s)
    """
    bins = [0.0, 5.0, 15.0, 25.0, 100.0]
    bin_labels = [
        "Crawl/Standstill (0-5 m/s)",
        "Urban/Town (5-15 m/s)",
        "Arterial (15-25 m/s)",
        "Motorway (25+ m/s)"
    ]
    
    # Assign bin index to each sample
    digitized = np.digitize(y_train, bins[1:-1]) # returns 0, 1, 2, 3
    bin_counts = np.bincount(digitized, minlength=len(bin_labels))
    total_samples = len(y_train)
    
    # Target uniform weight per bin (each bin gets 25% of sampled probability)
    # Sample weight = 1.0 / (bin_count * 4)
    sample_weights = np.zeros(total_samples, dtype=np.float32)
    bin_stats = {}
    
    for b_idx in range(len(bin_labels)):
        count = bin_counts[b_idx]
        pct = (count / total_samples) * 100.0 if total_samples > 0 else 0.0
        weight_val = (1.0 / count) if count > 0 else 0.0
        mask = digitized == b_idx
        sample_weights[mask] = weight_val
        bin_stats[bin_labels[b_idx]] = {
            'count': int(count),
            'natural_pct': round(pct, 2),
            'effective_sampled_pct': 25.0 if count > 0 else 0.0
        }

    # Normalize weights so sum equals total samples
    sample_weights = sample_weights / np.sum(sample_weights) * total_samples
    
    n_samples = num_samples if num_samples is not None else total_samples
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=n_samples,
        replacement=True
    )
    
    return sampler, bin_stats
