"""
Offline Event & Driving Regime Segmentation Layer for IO-VNBD.

This module derives ground-truth motion regimes strictly from reference vehicle
channels (CAN bus velocity, wheel speeds, yaw rate, steering angle, longitudinal/lateral
accelerations, brake pressure, handbrake, engine RPM).

CRITICAL ARCHITECTURAL RULE:
These regime labels are STRICTLY OFFLINE evaluation labels. They are NEVER exposed
as inputs to ML models at training or inference time. They exist solely to diagnose
and report model performance across distinct driving conditions (e.g. Speed MAE by regime).
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


# Canonical Regime Identifiers
REGIMES = [
    'STOP',
    'START',
    'NORMAL_CRUISE',
    'ACCELERATION',
    'BRAKING',
    'TURN_LEFT',
    'TURN_RIGHT',
    'BUMP_TRANSIENT',
    'ROUGH_ROAD',
    'LOW_SPEED',
    'HIGH_SPEED',
]


def extract_regime_masks(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Extracts multi-label boolean masks and a primary categorical regime vector.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing vehicle reference channels:
        - can_speed_mps (float)
        - veh_lon_acc_g (float)
        - veh_lat_acc_g (float)
        - veh_yaw_rate (float, deg/s)
        - veh_steer_deg (float, deg)
        - brake_pressure_psi (float)
        - brake_pos (0 or 1)
        - handbrake (0 or 1)
        - wheel_rear_rads (float)
        - lin_acc_mag (float, m/s^2)
        
    Returns
    -------
    masks : Dict[str, np.ndarray]
        Dictionary of boolean arrays for each regime.
    primary_regime : np.ndarray
        Array of string labels representing the dominant mutually-exclusive regime per sample.
    """
    n_samples = len(df)
    v = df['can_speed_mps'].values
    lon_acc = df['veh_lon_acc_g'].values * 9.80665  # convert g to m/s^2
    lat_acc = df['veh_lat_acc_g'].values * 9.80665
    yaw_rate = df['veh_yaw_rate'].values  # deg/s
    steer = df['veh_steer_deg'].values  # deg
    brake_press = df['brake_pressure_psi'].values
    brake_pos = df['brake_pos'].values
    handbrake = df['handbrake'].values
    wheel_w = df['wheel_rear_rads'].values if 'wheel_rear_rads' in df else (v / 0.305)
    
    # 1. STOP: Vehicle stationary
    stop_mask = (v < 0.2) & ((brake_pos == 1.0) | (handbrake == 1.0) | (wheel_w < 0.2))
    
    # 2. START: Transitioning from stop into motion within 2.5s (25 frames @ 10Hz)
    start_mask = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if 0.2 <= v[i] <= 4.0 and lon_acc[i] > 0.3:
            # Check if recently stopped in previous 25 frames
            lookback = max(0, i - 25)
            if np.any(stop_mask[lookback:i]):
                start_mask[i] = True
                
    # 3. ACCELERATION: Forward longitudinal acceleration
    accel_mask = (v >= 0.5) & (lon_acc > 0.55) & (~start_mask)
    
    # 4. BRAKING: Deceleration or active brake engagement
    brake_mask = (v >= 0.5) & ((brake_press > 8.0) | (brake_pos == 1.0) | (lon_acc < -0.55))
    
    # 5. TURNING: Substantial yaw rate or steering deflection
    turn_left_mask = (v >= 1.0) & ((yaw_rate > 3.5) | (steer > 18.0) | (lat_acc > 0.8))
    turn_right_mask = (v >= 1.0) & ((yaw_rate < -3.5) | (steer < -18.0) | (lat_acc < -0.8))
    
    # 6. SPEED REGIMES
    low_speed_mask = (v >= 0.2) & (v <= 7.0) & (~stop_mask)  # ~1 - 25 km/h
    high_speed_mask = (v > 20.0)  # > 72 km/h
    
    # 7. NORMAL CRUISE: Stable motion without turning, braking, or rapid acceleration
    cruise_mask = (
        (v > 5.0) &
        (np.abs(lon_acc) <= 0.45) &
        (np.abs(yaw_rate) <= 2.5) &
        (np.abs(steer) <= 12.0) &
        (brake_pos == 0.0) &
        (brake_press <= 5.0) &
        (~stop_mask)
    )
    
    # 8. BUMP / TRANSIENT: Sharp acceleration spikes in IMU
    # Rolling standard deviation of linear acceleration magnitude over 0.5s window
    lin_acc = df['lin_acc_mag'].values if 'lin_acc_mag' in df else np.zeros(n_samples)
    bump_mask = np.zeros(n_samples, dtype=bool)
    if 'lin_acc_mag' in df:
        # High frequency deviation
        kernel = 5  # 0.5s
        rolling_std = pd.Series(lin_acc).rolling(kernel, center=True, min_periods=1).std().values
        bump_mask = (v > 2.0) & (lin_acc > 3.0) & (rolling_std > 1.2) & (~turn_left_mask) & (~turn_right_mask)
        
    # 9. ROUGH ROAD: Sustained vibration energy
    rough_road_mask = np.zeros(n_samples, dtype=bool)
    if 'lin_acc_mag' in df:
        long_kernel = 15  # 1.5s
        rolling_var = pd.Series(lin_acc).rolling(long_kernel, center=True, min_periods=1).var().values
        rough_road_mask = (v > 3.0) & (rolling_var > 1.5) & (~bump_mask)
        
    masks = {
        'STOP': stop_mask,
        'START': start_mask,
        'NORMAL_CRUISE': cruise_mask,
        'ACCELERATION': accel_mask,
        'BRAKING': brake_mask,
        'TURN_LEFT': turn_left_mask,
        'TURN_RIGHT': turn_right_mask,
        'BUMP_TRANSIENT': bump_mask,
        'ROUGH_ROAD': rough_road_mask,
        'LOW_SPEED': low_speed_mask,
        'HIGH_SPEED': high_speed_mask,
    }
    
    # Build mutually exclusive dominant regime for top-level slicing
    primary_regime = np.full(n_samples, 'CRUISE_OR_OTHER', dtype=object)
    
    # Priority order for primary classification
    priority = [
        ('STOP', stop_mask),
        ('START', start_mask),
        ('BRAKING', brake_mask),
        ('ACCELERATION', accel_mask),
        ('TURN_LEFT', turn_left_mask),
        ('TURN_RIGHT', turn_right_mask),
        ('BUMP_TRANSIENT', bump_mask),
        ('NORMAL_CRUISE', cruise_mask),
        ('LOW_SPEED', low_speed_mask),
        ('HIGH_SPEED', high_speed_mask),
    ]
    
    for name, mask in priority:
        # Assign only where not already assigned
        unassigned = (primary_regime == 'CRUISE_OR_OTHER')
        primary_regime[unassigned & mask] = name
        
    return masks, primary_regime


def compute_regime_distribution(primary_regimes: np.ndarray) -> pd.DataFrame:
    """Computes sample counts and percentage breakdown of driving regimes."""
    labels, counts = np.unique(primary_regimes, return_counts=True)
    total = len(primary_regimes)
    return pd.DataFrame({
        'Regime': labels,
        'Samples': counts,
        'Percentage': (counts / total) * 100.0,
        'Duration_sec': counts * 0.1,
    }).sort_values('Samples', ascending=False).reset_index(drop=True)


def evaluate_metrics_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: Dict[str, np.ndarray],
    primary_regimes: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes Velocity MAE, RMSE, Max Error, and sample count sliced across each regime.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth velocity (m/s).
    y_pred : np.ndarray
        Predicted velocity (m/s).
    masks : Dict[str, np.ndarray]
        Regime boolean masks from extract_regime_masks().
    primary_regimes : np.ndarray, optional
        Primary dominant regime vector.
        
    Returns
    -------
    pd.DataFrame
        Detailed benchmark comparison table by regime.
    """
    overall_err = np.abs(y_true - y_pred)
    results = [{
        'Regime': 'OVERALL (ALL FRAMES)',
        'Samples': len(y_true),
        'Duration_s': len(y_true) * 0.1,
        'MAE_mps': float(np.mean(overall_err)),
        'RMSE_mps': float(np.sqrt(np.mean((y_true - y_pred)**2))),
        'Max_Err_mps': float(np.max(overall_err)),
        'P80_Err_mps': float(np.percentile(overall_err, 80)),
    }]
    
    for regime_name, mask in masks.items():
        if np.sum(mask) == 0:
            continue
        err = np.abs(y_true[mask] - y_pred[mask])
        results.append({
            'Regime': regime_name,
            'Samples': int(np.sum(mask)),
            'Duration_s': float(np.sum(mask) * 0.1),
            'MAE_mps': float(np.mean(err)),
            'RMSE_mps': float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))),
            'Max_Err_mps': float(np.max(err)),
            'P80_Err_mps': float(np.percentile(err, 80)),
        })
        
    return pd.DataFrame(results)


def verify_regimes():
    """Unit test and validation function for regime segmentation."""
    print("Testing regime segmentation on synthetic test vector...")
    n = 1000
    dummy_df = pd.DataFrame({
        'can_speed_mps': np.concatenate([np.zeros(200), np.linspace(0, 15, 200), np.full(300, 15), np.linspace(15, 0, 300)]),
        'veh_lon_acc_g': np.concatenate([np.zeros(200), np.full(200, 0.15), np.zeros(300), np.full(300, -0.15)]),
        'veh_lat_acc_g': np.zeros(n),
        'veh_yaw_rate': np.concatenate([np.zeros(500), np.full(100, 6.0), np.full(100, -6.0), np.zeros(300)]),
        'veh_steer_deg': np.zeros(n),
        'brake_pressure_psi': np.concatenate([np.full(200, 20.0), np.zeros(500), np.full(300, 30.0)]),
        'brake_pos': np.concatenate([np.ones(200), np.zeros(500), np.ones(300)]),
        'handbrake': np.concatenate([np.ones(100), np.zeros(900)]),
        'wheel_rear_rads': np.zeros(n),
        'lin_acc_mag': np.full(n, 0.2),
    })
    
    masks, primary = extract_regime_masks(dummy_df)
    dist = compute_regime_distribution(primary)
    print("\nRegime Distribution on Synthetic Sequence:")
    print(dist.to_string())
    
    # Verification checks
    assert np.sum(masks['STOP']) >= 100, "STOP detection failed"
    assert np.sum(masks['ACCELERATION']) > 0, "ACCELERATION detection failed"
    assert np.sum(masks['BRAKING']) > 0, "BRAKING detection failed"
    assert np.sum(masks['TURN_LEFT']) > 0, "TURN_LEFT detection failed"
    assert np.sum(masks['TURN_RIGHT']) > 0, "TURN_RIGHT detection failed"
    
    coverage = np.sum(primary != 'CRUISE_OR_OTHER') / n * 100.0
    print(f"\n[PASS] Regime Segmentation verified successfully! Active coverage: {coverage:.1f}%")
    return True


if __name__ == '__main__':
    verify_regimes()
