"""
SIH26168 - Step 4: Navigation Performance Metrics
Computes standardized localization accuracy metrics:
  - 2D Position Error (meters)
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Final Position Error (FPE)
  - Maximum Drift Error
  - SIH Drift Percentage = (Position Error / Cumulative Distance Travelled) * 100
"""

import numpy as np

def compute_cumulative_distance(east, north):
    """Computes running cumulative distance along a 2D trajectory in meters."""
    dx = np.diff(east, prepend=east[0])
    dy = np.diff(north, prepend=north[0])
    step_dist = np.sqrt(dx**2 + dy**2)
    return np.cumsum(step_dist)

def calculate_navigation_metrics(est_east, est_north, gt_east, gt_north):
    """
    Evaluates estimated 2D trajectory against ground truth.
    Returns:
        metrics: dict of aggregate error statistics
        errors: array of point-by-point Euclidean errors (meters)
    """
    est_e = np.asarray(est_east)
    est_n = np.asarray(est_north)
    gt_e = np.asarray(gt_east)
    gt_n = np.asarray(gt_north)

    # 2D Euclidean position error at each time step
    errors = np.sqrt((est_e - gt_e)**2 + (est_n - gt_n)**2)

    total_distance = float(compute_cumulative_distance(gt_e, gt_n)[-1])
    fpe = float(errors[-1])
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_err = float(np.max(errors))

    drift_pct = (fpe / max(total_distance, 1e-6)) * 100.0

    metrics = {
        'total_distance_m': total_distance,
        'rmse_m': rmse,
        'mae_m': mae,
        'fpe_m': fpe,
        'max_error_m': max_err,
        'drift_percentage': drift_pct,
        'sih_target_pass': bool(drift_pct < 10.0)
    }

    return metrics, errors

if __name__ == "__main__":
    t = np.linspace(0, 100, 1000)
    gt_e = 10 * t
    gt_n = np.zeros_like(t)
    est_e = 10 * t + 0.05 * t**2 # drifted
    est_n = 0.02 * t**2

    m, err = calculate_navigation_metrics(est_e, est_n, gt_e, gt_n)
    print("Test Metrics:", m)
