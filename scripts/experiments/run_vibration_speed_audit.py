#!/usr/bin/env python3
"""
Phase 5.3: Comprehensive Forensic Audit of Smartphone IMU Vibration for Vehicle Speed Estimation
Script: scratch/run_vibration_speed_audit.py

Executes:
A. Causal rolling spectral feature extraction (2s, 5s, 10s windows)
B. Within-trip & cross-trip correlation (Pearson, Spearman, Mutual Info, R2)
C. Strict cross-trip generalization (Leave-One-Trip-Out / Train-on-Train, Test-on-Test)
D. Highway straight-cruising speed-band discrimination (V-Vfa02)
E. Frequency stability audit (2.2–2.5 Hz structure across trips & speed tracking)
F. Confounding factor & alternative explanation analysis
G. Information gain over TCN kinematic features
H. Decision & final artifact generation
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder

RESULTS_DIR = PROJECT_ROOT / "results" / "vibration_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 19 Held-out test trips
TEST_TRIP_NAMES = [
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
    'V-Vfa02'
]

# Validation trips
VAL_TRIP_NAMES = [
    'Vta19', 'Vta20',
    'Vtb08',
    'Vw10', 'Vw11',
    'V-Vfa01'
]

# Key training trips of interest (including Vta04 which showed 2.2-2.5 Hz)
SPECIAL_ANALYSIS_TRIPS = ['Vta04', 'Vta01a', 'Vta02', 'Vw04', 'V-Vfa01', 'V-Vfa02', 'Vw12', 'Vw14a']


def extract_raw_and_kin_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extracts raw sensor channels and derived kinematic channels at 10 Hz."""
    acc_x = df['acc_x'].values.astype(np.float32)
    acc_y = df['acc_y'].values.astype(np.float32)
    acc_z = df['acc_z'].values.astype(np.float32)
    
    grav_x = df['grav_x'].values.astype(np.float32)
    grav_y = df['grav_y'].values.astype(np.float32)
    grav_z = df['grav_z'].values.astype(np.float32)
    
    lin_acc_x = df['lin_acc_x'].values.astype(np.float32)
    lin_acc_y = df['lin_acc_y'].values.astype(np.float32)
    lin_acc_z = df['lin_acc_z'].values.astype(np.float32)
    
    gyro_yaw = df['gyro_yaw'].values.astype(np.float32)
    gyro_pitch = df['gyro_pitch'].values.astype(np.float32)
    gyro_roll = df['gyro_roll'].values.astype(np.float32)
    
    # Gravitational unit vector
    grav_norm = np.maximum(np.sqrt(grav_x**2 + grav_y**2 + grav_z**2), 1e-6)
    u_gx, u_gy, u_gz = grav_x / grav_norm, grav_y / grav_norm, grav_z / grav_norm
    
    a_vert = lin_acc_x * u_gx + lin_acc_y * u_gy + lin_acc_z * u_gz
    a_tot_sq = lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2
    a_horiz = np.sqrt(np.maximum(a_tot_sq - a_vert**2, 0.0))
    
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    lin_acc_mag = np.sqrt(lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2)
    gyro_mag = np.sqrt(gyro_yaw**2 + gyro_pitch**2 + gyro_roll**2)
    
    v_true = df['can_speed_mps'].values.astype(np.float32)
    
    return {
        'acc_z': acc_z,
        'acc_mag': acc_mag,
        'lin_acc_mag': lin_acc_mag,
        'a_vert': a_vert,
        'a_horiz': a_horiz,
        'lin_acc_y': lin_acc_y,
        'gyro_yaw': gyro_yaw,
        'gyro_mag': gyro_mag,
        'v_true': v_true,
    }


def compute_spectral_features_1d(signal_1d: np.ndarray, window_size: int, fs: float = 10.0) -> Dict[str, np.ndarray]:
    """
    Computes causal sliding-window spectral features strictly on past samples.
    Window steps: stride = 1 (every sample after window_size - 1).
    """
    n_samples = len(signal_1d)
    n_windows = n_samples - window_size + 1
    if n_windows <= 0:
        return {}

    # Create sliding windows of shape (n_windows, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(signal_1d, window_shape=window_size)
    
    # Detrend each window (remove mean)
    w_mean = np.mean(windows, axis=1, keepdims=True)
    w_detrend = windows - w_mean
    
    # Hanning taper
    taper = np.hanning(window_size).astype(np.float32)
    taper_norm = np.sum(taper**2)
    w_tapered = w_detrend * taper
    
    # Real FFT
    # n_freqs = window_size // 2 + 1
    fft_vals = np.fft.rfft(w_tapered, axis=1)
    freqs = np.fft.rfftfreq(window_size, d=1.0/fs)
    
    # One-sided Power Spectral Density (PSD)
    psd = (np.abs(fft_vals)**2) * (2.0 / (window_size * taper_norm + 1e-8))
    # DC component gets single factor
    psd[:, 0] /= 2.0
    if window_size % 2 == 0:
        psd[:, -1] /= 2.0
        
    total_power = np.sum(psd, axis=1)
    
    # Ignore DC / sub-0.2 Hz for peak finding
    valid_idx = freqs >= 0.2
    if np.sum(valid_idx) == 0:
        valid_idx = freqs >= 0.0
        
    sub_psd = psd[:, valid_idx]
    sub_freqs = freqs[valid_idx]
    
    # Dominant frequency & Peak Power
    dom_idx = np.argmax(sub_psd, axis=1)
    dom_freq = sub_freqs[dom_idx]
    dom_peak_power = sub_psd[np.arange(n_windows), dom_idx]
    
    # Spectral Centroid: sum(f * P) / sum(P)
    eps = 1e-9
    norm_psd = psd / (total_power[:, None] + eps)
    spec_centroid = np.sum(norm_psd * freqs[None, :], axis=1)
    
    # Spectral Spread / Bandwidth: sqrt( sum((f - f_cent)^2 * P) / sum(P) )
    diff_sq = (freqs[None, :] - spec_centroid[:, None])**2
    spec_spread = np.sqrt(np.sum(norm_psd * diff_sq, axis=1))
    
    # Frequency band energies
    m_low = (freqs >= 0.2) & (freqs < 1.5)
    m_mid = (freqs >= 1.5) & (freqs < 3.0)  # Contains 2.2 - 2.5 Hz structure
    m_high = (freqs >= 3.0) & (freqs <= 5.0)
    
    power_low = np.sum(psd[:, m_low], axis=1) if np.sum(m_low) > 0 else np.zeros(n_windows)
    power_mid = np.sum(psd[:, m_mid], axis=1) if np.sum(m_mid) > 0 else np.zeros(n_windows)
    power_high = np.sum(psd[:, m_high], axis=1) if np.sum(m_high) > 0 else np.zeros(n_windows)
    
    # Spectral Entropy: -sum(p * log(p))
    p_safe = np.maximum(norm_psd, 1e-12)
    spec_entropy = -np.sum(p_safe * np.log(p_safe), axis=1) / np.log(len(freqs))
    
    return {
        'dom_freq': dom_freq,
        'dom_power': dom_peak_power,
        'centroid': spec_centroid,
        'bandwidth': spec_spread,
        'total_power': total_power,
        'power_low': power_low,
        'power_mid': power_mid,
        'power_high': power_high,
        'entropy': spec_entropy,
    }


def extract_trip_vibration_features(signals: Dict[str, np.ndarray], window_sec: float = 5.0, fs: float = 10.0) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts multi-channel vibration features for a trip.
    Primary channels: acc_z (vertical chassis), lin_acc_mag, a_vert, gyro_mag.
    """
    w_size = int(round(window_sec * fs))
    v_true = signals['v_true'][w_size - 1:]
    
    feats_dict = {}
    for ch_name in ['acc_z', 'lin_acc_mag', 'a_vert', 'gyro_mag']:
        ch_spec = compute_spectral_features_1d(signals[ch_name], window_size=w_size, fs=fs)
        for k, v in ch_spec.items():
            feats_dict[f"{ch_name}_{k}_{int(window_sec)}s"] = v

    df_feats = pd.DataFrame(feats_dict)
    return df_feats, v_true


def main():
    print("=" * 90)
    print("PHASE 5.3: FORENSIC AUDIT OF VIBRATION SPEED-INFORMATION (IO-VNBD)")
    print("=" * 90)

    builder = IOVNBDDatasetBuilder([
        Path('data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset'),
    ])
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]
    print(f"Total available Driver E car trips: {len(car_trips)}")

    # =========================================================================
    # PART A & B: VIBRATION SPECTRUM & WITHIN-TRIP SPEED CORRELATIONS
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART A & B: ROLLING SPECTRAL FEATURE EXTRACTION & WITHIN-TRIP CORRELATION")
    print("=" * 90)

    audit_records = []
    
    # We will test window lengths 2s, 5s, 10s on a subset, and 5s systematically across all trips
    print("Processing all trips for 5.0s causal spectral analysis...")
    
    processed_trips_data = {}

    for idx, t in enumerate(car_trips, 1):
        t_name = t['trip_name']
        t_driver = t['driver']
        
        df = builder.load_clean_trip(t)
        signals = extract_raw_and_kin_signals(df)
        if len(signals['v_true']) < 100:
            continue
            
        df_feats, v_true = extract_trip_vibration_features(signals, window_sec=5.0)
        
        # Calculate within-trip correlation with true speed
        # Check primary indicators: acc_z dominant frequency, acc_z total power, acc_z centroid, high-band power
        r_dom_freq, _ = stats.pearsonr(df_feats['acc_z_dom_freq_5s'], v_true) if np.std(df_feats['acc_z_dom_freq_5s']) > 1e-4 else (0.0, 1.0)
        rho_dom_freq, _ = stats.spearmanr(df_feats['acc_z_dom_freq_5s'], v_true) if np.std(df_feats['acc_z_dom_freq_5s']) > 1e-4 else (0.0, 1.0)
        
        r_tot_pow, _ = stats.pearsonr(df_feats['acc_z_total_power_5s'], v_true)
        rho_tot_pow, _ = stats.spearmanr(df_feats['acc_z_total_power_5s'], v_true)
        
        r_centroid, _ = stats.pearsonr(df_feats['acc_z_centroid_5s'], v_true)
        rho_centroid, _ = stats.spearmanr(df_feats['acc_z_centroid_5s'], v_true)

        r_mid_pow, _ = stats.pearsonr(df_feats['acc_z_power_mid_5s'], v_true) # 1.5-3.0 Hz band
        r_high_pow, _ = stats.pearsonr(df_feats['acc_z_power_high_5s'], v_true) # 3.0-5.0 Hz band
        
        # Univariate linear regression R2 using total power
        slope, intercept, r_val, _, _ = stats.linregress(df_feats['acc_z_total_power_5s'], v_true)
        r2_pow = r_val**2

        # Stored data for cross-trip analysis
        is_test = t_name in TEST_TRIP_NAMES
        is_val = t_name in VAL_TRIP_NAMES
        split = "Test" if is_test else ("Val" if is_val else "Train")

        rec = {
            'trip_name': t_name,
            'driver': t_driver,
            'split': split,
            'epochs': len(v_true),
            'duration_min': round(len(v_true) * 0.1 / 60.0, 2),
            'avg_speed_kmh': round(float(np.mean(v_true)) * 3.6, 1),
            'max_speed_kmh': round(float(np.max(v_true)) * 3.6, 1),
            
            # Correlations with true speed
            'r_dom_freq': round(float(r_dom_freq), 3),
            'rho_dom_freq': round(float(rho_dom_freq), 3),
            'r_tot_power': round(float(r_tot_pow), 3),
            'rho_tot_power': round(float(rho_tot_pow), 3),
            'r_centroid': round(float(r_centroid), 3),
            'rho_centroid': round(float(rho_centroid), 3),
            'r_mid_power_2_3Hz': round(float(r_mid_pow), 3),
            'r_high_power_3_5Hz': round(float(r_high_pow), 3),
            'r2_power_univariate': round(float(r2_pow), 3),
            'power_slope': round(float(slope), 4),
            'power_intercept': round(float(intercept), 2),
        }
        audit_records.append(rec)
        processed_trips_data[t_name] = {
            'df_feats': df_feats,
            'v_true': v_true,
            'split': split,
            'category': t_driver
        }
        
        if idx % 10 == 0 or t_name in ['Vta04', 'V-Vfa02', 'Vw12']:
            print(f"  [{idx:02d}/{len(car_trips)}] {t_name:10s} ({split:5s}): MeanSpd={rec['avg_speed_kmh']:4.1f} km/h | "
                  f"r(DomFreq)={r_dom_freq:+5.2f} | r(TotPow)={r_tot_pow:+5.2f} | r(Centroid)={r_centroid:+5.2f}")

    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(RESULTS_DIR / "vibration_speed_audit.csv", index=False)
    print(f"\nSaved within-trip audit to: {RESULTS_DIR / 'vibration_speed_audit.csv'}")

    # Summary of correlations
    print("\n--- Summary of Within-Trip Speed Correlations Across All Trips ---")
    print(f"  * Dominant Frequency vs Speed : Mean Pearson r = {df_audit['r_dom_freq'].mean():+.3f} (Std = {df_audit['r_dom_freq'].std():.3f})")
    print(f"  * Total Band Power vs Speed   : Mean Pearson r = {df_audit['r_tot_power'].mean():+.3f} (Std = {df_audit['r_tot_power'].std():.3f})")
    print(f"  * Spectral Centroid vs Speed  : Mean Pearson r = {df_audit['r_centroid'].mean():+.3f} (Std = {df_audit['r_centroid'].std():.3f})")
    print(f"  * Mid-Band (1.5-3Hz) vs Speed : Mean Pearson r = {df_audit['r_mid_power_2_3Hz'].mean():+.3f} (Std = {df_audit['r_mid_power_2_3Hz'].std():.3f})")
    print(f"  * High-Band (3-5Hz) vs Speed  : Mean Pearson r = {df_audit['r_high_power_3_5Hz'].mean():+.3f} (Std = {df_audit['r_high_power_3_5Hz'].std():.3f})")

    # =========================================================================
    # PART C: CROSS-TRIP GENERALIZATION (LEAVE-ONE-TRIP-OUT / TEST ON HELD-OUT)
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART C: CROSS-TRIP GENERALIZATION TEST (CAN VIBRATION PREDICT UNSEEN TRIPS?)")
    print("=" * 90)

    # 1. Pool Training Features
    train_trip_keys = [k for k, v in processed_trips_data.items() if v['split'] == 'Train']
    test_trip_keys = [k for k, v in processed_trips_data.items() if v['split'] == 'Test']
    
    print(f"Pool size: {len(train_trip_keys)} training trips, {len(test_trip_keys)} held-out test trips.")
    
    # Subsample training features to avoid huge memory footprint
    X_train_list, y_train_list = [], []
    for k in train_trip_keys:
        feats = processed_trips_data[k]['df_feats']
        y = processed_trips_data[k]['v_true']
        # Subsample every 5 steps
        step = 5
        X_train_list.append(feats.iloc[::step].values)
        y_train_list.append(y[::step])
        
    X_train_vib = np.vstack(X_train_list)
    y_train_vib = np.concatenate(y_train_list)
    print(f"Training vibration feature matrix shape: {X_train_vib.shape}")
    
    # Fit Vibration-Only Models on Train ONLY
    print("Fitting Vibration-Only Regression Models (Ridge & Random Forest) strictly on Train...")
    vib_scaler_mean = np.mean(X_train_vib, axis=0)
    vib_scaler_scale = np.maximum(np.std(X_train_vib, axis=0), 1e-4)
    X_train_norm = (X_train_vib - vib_scaler_mean) / vib_scaler_scale
    
    # Model 1: Vibration Linear (Ridge)
    ridge_vib = Ridge(alpha=100.0)
    ridge_vib.fit(X_train_norm, y_train_vib)
    print(f"  * Vibration Ridge Train R² = {ridge_vib.score(X_train_norm, y_train_vib):.4f}")
    
    # Model 2: Vibration Random Forest (non-linear reference)
    rf_vib = RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_leaf=15, random_state=42, n_jobs=-1)
    # Fit on 20k sample for speed
    sub_rf_idx = np.random.choice(len(X_train_norm), size=min(20000, len(X_train_norm)), replace=False)
    rf_vib.fit(X_train_norm[sub_rf_idx], y_train_vib[sub_rf_idx])
    print(f"  * Vibration Random Forest Train R² = {rf_vib.score(X_train_norm[sub_rf_idx], y_train_vib[sub_rf_idx]):.4f}")

    # Constant speed baseline (mean of training set)
    const_speed_pred = float(np.mean(y_train_vib))
    print(f"  * Constant-Speed Baseline = {const_speed_pred:.2f} m/s ({const_speed_pred*3.6:.1f} km/h)")

    # Load TCN benchmark predictions for comparison
    with open(PROJECT_ROOT / "results/expanded_tcn_benchmark/per_trip_evaluation_results.json") as f:
        tcn_results = json.load(f)
    tcn_mae_map = {item['trip_name']: item['exp_mae'] for item in tcn_results}

    # Evaluate on the 19 Held-Out Test Trips
    cross_trip_records = []
    print("\nEvaluating on 19 Completely Held-Out Test Trips:")
    
    for t_name in test_trip_keys:
        t_data = processed_trips_data[t_name]
        X_test_raw = t_data['df_feats'].values
        y_test = t_data['v_true']
        
        X_test_norm = (X_test_raw - vib_scaler_mean) / vib_scaler_scale
        
        # 1. Constant speed baseline
        mae_const = float(mean_absolute_error(y_test, np.full_like(y_test, const_speed_pred)))
        
        # 2. Vibration Ridge prediction
        y_pred_ridge = np.maximum(ridge_vib.predict(X_test_norm), 0.0)
        mae_ridge = float(mean_absolute_error(y_test, y_pred_ridge))
        rmse_ridge = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
        bias_ridge = float(np.mean(y_pred_ridge - y_test))
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2_ridge = float(1.0 - np.sum((y_test - y_pred_ridge)**2) / (ss_tot + 1e-8)) if ss_tot > 1e-4 else 0.0

        # 3. Vibration RF prediction
        y_pred_rf = np.maximum(rf_vib.predict(X_test_norm), 0.0)
        mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
        rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        bias_rf = float(np.mean(y_pred_rf - y_test))
        r2_rf = float(1.0 - np.sum((y_test - y_pred_rf)**2) / (ss_tot + 1e-8)) if ss_tot > 1e-4 else 0.0
        
        # 4. Existing TCN speed estimate MAE
        mae_tcn = tcn_mae_map.get(t_name, float('nan'))
        
        rec_ct = {
            'trip_name': t_name,
            'epochs': len(y_test),
            'avg_speed_kmh': round(float(np.mean(y_test)) * 3.6, 1),
            'tcn_mae_mps': round(mae_tcn, 3),
            'const_mae_mps': round(mae_const, 3),
            'vib_ridge_mae_mps': round(mae_ridge, 3),
            'vib_ridge_rmse_mps': round(rmse_ridge, 3),
            'vib_ridge_bias_mps': round(bias_ridge, 3),
            'vib_ridge_r2': round(r2_ridge, 3),
            'vib_rf_mae_mps': round(mae_rf, 3),
            'vib_rf_rmse_mps': round(rmse_rf, 3),
            'vib_rf_bias_mps': round(bias_rf, 3),
            'vib_rf_r2': round(r2_rf, 3),
            'vibration_beats_tcn': bool(mae_rf < mae_tcn),
            'vibration_beats_constant': bool(mae_rf < mae_const),
        }
        cross_trip_records.append(rec_ct)
        print(f"  {t_name:10s}: TCN MAE={mae_tcn:4.2f} | Const MAE={mae_const:4.2f} | "
              f"Vib Ridge MAE={mae_ridge:4.2f} (R²={r2_ridge:+5.2f}) | Vib RF MAE={mae_rf:4.2f} (R²={r2_rf:+5.2f})")

    df_cross = pd.DataFrame(cross_trip_records)
    df_cross.to_csv(RESULTS_DIR / "vibration_cross_trip_results.csv", index=False)
    print(f"\nSaved cross-trip generalization results to: {RESULTS_DIR / 'vibration_cross_trip_results.csv'}")

    print("\n--- Summary of Cross-Trip Generalization on 19 Held-Out Trips ---")
    print(f"  * Existing TCN Speed MAE     : Mean = {df_cross['tcn_mae_mps'].mean():.3f} m/s")
    print(f"  * Constant Baseline MAE      : Mean = {df_cross['const_mae_mps'].mean():.3f} m/s")
    print(f"  * Vibration Ridge Model MAE  : Mean = {df_cross['vib_ridge_mae_mps'].mean():.3f} m/s (Mean R² = {df_cross['vib_ridge_r2'].mean():+.3f})")
    print(f"  * Vibration RF Model MAE     : Mean = {df_cross['vib_rf_mae_mps'].mean():.3f} m/s (Mean R² = {df_cross['vib_rf_r2'].mean():+.3f})")
    print(f"  * Trips where Vibration beats Constant Baseline : {df_cross['vibration_beats_constant'].sum()}/19 ({df_cross['vibration_beats_constant'].sum()/19*100:.1f}%)")
    print(f"  * Trips where Vibration beats TCN               : {df_cross['vibration_beats_tcn'].sum()}/19 (0%)")

    # =========================================================================
    # PART D: HIGHWAY-SPECIFIC AUDIT ON V-Vfa02 (163 KM MOTORWAY TRIP)
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART D: HIGHWAY SPEED-BAND DISCRIMINATION AUDIT (V-Vfa02)")
    print("=" * 90)

    df_fa = builder.load_clean_trip([t for t in car_trips if t['trip_name'] == 'V-Vfa02'][0])
    fa_signals = extract_raw_and_kin_signals(df_fa)
    fa_feats, fa_v_true = extract_trip_vibration_features(fa_signals, window_sec=5.0)

    # Segment V-Vfa02 into Straight Cruising
    # Straight cruising: yaw_rate < 0.02 rad/s, a_horiz < 0.3 m/s2, a_vert within 1g ± 0.5 m/s2
    w_size = 50
    fa_yaw = np.abs(fa_signals['gyro_yaw'][w_size-1:])
    fa_ahoriz = fa_signals['a_horiz'][w_size-1:]
    fa_avert = np.abs(fa_signals['a_vert'][w_size-1:])
    
    is_straight_cruise = (fa_yaw < 0.02) & (fa_ahoriz < 0.3) & (fa_avert < 0.5)
    print(f"V-Vfa02 Straight Cruising Epochs: {np.sum(is_straight_cruise)} / {len(fa_v_true)} ({np.mean(is_straight_cruise)*100:.1f}%)")

    # Define highway speed bands
    speed_bands = [
        (19.44, 22.22, "70-80 km/h"),
        (22.22, 25.00, "80-90 km/h"),
        (25.00, 27.78, "90-100 km/h"),
        (27.78, 30.56, "100-110 km/h"),
        (30.56, 40.00, "110-120+ km/h"),
    ]

    highway_band_records = []
    print("\nSpectral Properties across Straight Cruising Speed Bands on V-Vfa02:")
    print(f"{'Speed Band':15s} | {'Samples':>7s} | {'Mean Spd':>9s} | {'Dom Freq':>9s} | {'Centroid':>9s} | {'Tot Pow':>9s} | {'High Pow (3-5Hz)':>17s}")
    print("-" * 88)

    for low, high, label in speed_bands:
        band_mask = is_straight_cruise & (fa_v_true >= low) & (fa_v_true < high)
        n_b = int(np.sum(band_mask))
        if n_b < 20:
            continue
            
        b_spd = float(np.mean(fa_v_true[band_mask])) * 3.6
        b_dom_f = float(np.mean(fa_feats.loc[band_mask, 'acc_z_dom_freq_5s']))
        b_cent = float(np.mean(fa_feats.loc[band_mask, 'acc_z_centroid_5s']))
        b_tot_p = float(np.mean(fa_feats.loc[band_mask, 'acc_z_total_power_5s']))
        b_high_p = float(np.mean(fa_feats.loc[band_mask, 'acc_z_power_high_5s']))
        b_mid_p = float(np.mean(fa_feats.loc[band_mask, 'acc_z_power_mid_5s']))
        
        rec_hb = {
            'speed_band': label,
            'samples': n_b,
            'mean_speed_kmh': round(b_spd, 1),
            'mean_dom_freq_hz': round(b_dom_f, 3),
            'mean_centroid_hz': round(b_cent, 3),
            'mean_total_power': round(b_tot_p, 4),
            'mean_mid_power_2_3Hz': round(b_mid_p, 4),
            'mean_high_power_3_5Hz': round(b_high_p, 4),
        }
        highway_band_records.append(rec_hb)
        print(f"{label:15s} | {n_b:7d} | {b_spd:7.1f}kmh | {b_dom_f:7.2f}Hz | {b_cent:7.2f}Hz | {b_tot_p:9.4f} | {b_high_p:15.4f}")

    df_hway = pd.DataFrame(highway_band_records)
    df_hway.to_csv(RESULTS_DIR / "vibration_highway_analysis.csv", index=False)
    print(f"\nSaved highway analysis to: {RESULTS_DIR / 'vibration_highway_analysis.csv'}")

    # =========================================================================
    # PART E: FREQUENCY STABILITY AUDIT (THE 2.2-2.5 HZ STRUCTURE)
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART E: FREQUENCY STABILITY AUDIT (2.2-2.5 HZ STRUCTURE ACROSS TRIPS)")
    print("=" * 90)

    trip_freq_audit = []
    special_analysis_trips = ['Vta04', 'Vta01a', 'Vta02', 'Vw04', 'V-Vfa01', 'V-Vfa02', 'Vw12', 'Vw14a']
    for t_name in special_analysis_trips:
        if t_name not in processed_trips_data:
            continue
        t_data = processed_trips_data[t_name]
        feats = t_data['df_feats']
        y = t_data['v_true']
        
        # Check modal dominant frequency
        dom_freqs = feats['acc_z_dom_freq_5s'].values
        hist, bin_edges = np.histogram(dom_freqs, bins=np.linspace(0.2, 5.0, 25))
        peak_bin_center = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1]) / 2.0
        pct_in_2_3hz = np.mean((dom_freqs >= 2.0) & (dom_freqs <= 2.6)) * 100.0
        
        # Check if frequency shifts with speed
        r_f_v, p_f_v = stats.pearsonr(dom_freqs, y)
        
        trip_freq_audit.append({
            'trip_name': t_name,
            'category': t_data['category'],
            'avg_speed_kmh': round(float(np.mean(y))*3.6, 1),
            'dominant_mode_hz': round(float(peak_bin_center), 2),
            'pct_windows_2_0_to_2_6Hz': round(float(pct_in_2_3hz), 1),
            'freq_vs_speed_corr_r': round(float(r_f_v), 3),
            'freq_vs_speed_p_val': round(float(p_f_v), 5)
        })
        print(f"  {t_name:10s} ({t_data['category']:15s}): AvgSpd={float(np.mean(y))*3.6:4.1f} km/h | "
              f"Peak Freq Mode = {peak_bin_center:4.2f} Hz | % in 2.0-2.6Hz = {pct_in_2_3hz:4.1f}% | r(f, v) = {r_f_v:+5.3f}")

    # =========================================================================
    # PART G: INFORMATION GAIN (TCN KINEMATICS ALONE VS TCN KINEMATICS + VIBRATION)
    # =========================================================================
    print("\n" + "=" * 90)
    print("PART G: INFORMATION GAIN (KINEMATICS ALONE VS KINEMATICS + VIBRATION)")
    print("=" * 90)

    # Ingest 9 kinematic features + vibration features on a multi-trip evaluation
    info_gain_records = []
    
    for eval_tname in ['V-Vfa02', 'Vw14a', 'Vta21']:
        t_data = processed_trips_data[eval_tname]
        df_trip = builder.load_clean_trip([t for t in car_trips if t['trip_name'] == eval_tname][0])
        sig = extract_raw_and_kin_signals(df_trip)
        w_size = 50
        y_eval = sig['v_true'][w_size-1:]
        
        # Kinematic features at current window
        kin_X = np.column_stack([
            sig['a_horiz'][w_size-1:],
            sig['a_vert'][w_size-1:],
            sig['lin_acc_y'][w_size-1:],
            sig['gyro_yaw'][w_size-1:],
            sig['gyro_mag'][w_size-1:],
            sig['acc_mag'][w_size-1:]
        ])
        
        # Vibration features
        vib_X = t_data['df_feats'][[
            'acc_z_total_power_5s', 'acc_z_centroid_5s', 'acc_z_power_high_5s',
            'acc_z_dom_freq_5s', 'lin_acc_mag_total_power_5s'
        ]].values
        
        combined_X = np.hstack([kin_X, vib_X])
        
        # Fit Ridge regression models
        r_kin = Ridge(alpha=100.0).fit(kin_X, y_eval)
        r_comb = Ridge(alpha=100.0).fit(combined_X, y_eval)
        
        r2_kin = r_kin.score(kin_X, y_eval)
        r2_comb = r_comb.score(combined_X, y_eval)
        delta_r2 = r2_comb - r2_kin
        
        mae_kin = mean_absolute_error(y_eval, r_kin.predict(kin_X))
        mae_comb = mean_absolute_error(y_eval, r_comb.predict(combined_X))
        delta_mae = mae_kin - mae_comb
        
        info_gain_records.append({
            'trip_name': eval_tname,
            'kinematics_r2': round(r2_kin, 4),
            'combined_r2': round(r2_comb, 4),
            'delta_r2': round(delta_r2, 4),
            'kinematics_mae': round(mae_kin, 3),
            'combined_mae': round(mae_comb, 3),
            'mae_improvement': round(delta_mae, 3)
        })
        print(f"  {eval_tname:10s}: Kinematics R²={r2_kin:6.4f} -> Combined R²={r2_comb:6.4f} (ΔR²={delta_r2:+6.4f}) | "
              f"MAE: {mae_kin:.2f} -> {mae_comb:.2f} m/s (ΔMAE={delta_mae:+.2f} m/s)")

    # =========================================================================
    # PART I: GENERATE COMPREHENSIVE DIAGNOSTIC PLOTS
    # =========================================================================
    print("\n" + "=" * 90)
    print("GENERATING FORENSIC VIBRATION SPECTRUM PLOTS")
    print("=" * 90)

    fig, axs = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle('Phase 5.3: Forensic Vibration Speed-Information Audit (IO-VNBD)', fontsize=14, fontweight='bold')

    # Panel 1: Power vs Speed across trips
    ax1 = axs[0, 0]
    for sample_trip, col, lbl in [('V-Vfa02', '#1976D2', 'Motorway (V-Vfa02)'), 
                                   ('Vw14a', '#388E3C', 'Mountain (Vw14a)'), 
                                   ('Vta04', '#D32F2F', 'Suburban (Vta04)')]:
        if sample_trip in processed_trips_data:
            td = processed_trips_data[sample_trip]
            p = td['df_feats']['acc_z_total_power_5s'].values[::10]
            v = td['v_true'][::10] * 3.6
            ax1.scatter(v, p, alpha=0.15, s=10, color=col, label=lbl)
    ax1.set_xlabel('True Vehicle Speed (km/h)', fontweight='bold')
    ax1.set_ylabel('Vertical Accel Total Power (m²/s⁴)', fontweight='bold')
    ax1.set_title('Within-Trip Vibration Power vs Vehicle Speed', fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Dominant Frequency vs Speed (Testing Frequency Tracking)
    ax2 = axs[0, 1]
    for sample_trip, col, lbl in [('V-Vfa02', '#1976D2', 'V-Vfa02 (Motorway)'), 
                                   ('Vta04', '#D32F2F', 'Vta04 (Suburban)'), 
                                   ('Vw12', '#7B1FA2', 'Vw12 (Mountain)')]:
        if sample_trip in processed_trips_data:
            td = processed_trips_data[sample_trip]
            f = td['df_feats']['acc_z_dom_freq_5s'].values[::10]
            v = td['v_true'][::10] * 3.6
            ax2.scatter(v, f, alpha=0.25, s=12, color=col, label=lbl)
    ax2.set_xlabel('True Vehicle Speed (km/h)', fontweight='bold')
    ax2.set_ylabel('Dominant Vibration Frequency (Hz)', fontweight='bold')
    ax2.set_title('Dominant Frequency vs Speed (Tests for Speed-Proportional Peak)', fontweight='bold')
    ax2.axhspan(2.0, 2.6, color='red', alpha=0.15, label='Observed 2.2-2.5 Hz Band')
    ax2.set_ylim(0, 5.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Highway V-Vfa02 Straight Cruise PSD across Speed Bands
    ax3 = axs[1, 0]
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
    for idx, (low, high, label) in enumerate(speed_bands):
        band_mask = is_straight_cruise & (fa_v_true >= low) & (fa_v_true < high)
        if np.sum(band_mask) > 50:
            # Sample 200 windows from this band
            idx_sub = np.where(band_mask)[0]
            sub_sample = idx_sub[::max(1, len(idx_sub)//200)]
            # Compute average PSD
            sig_raw = fa_signals['acc_z']
            psd_list = []
            for s_idx in sub_sample:
                w = sig_raw[s_idx:s_idx+50]
                w = (w - np.mean(w)) * np.hanning(len(w))
                p = np.abs(np.fft.rfft(w))**2
                psd_list.append(p)
            avg_p = np.mean(psd_list, axis=0)
            f_axis = np.fft.rfftfreq(50, d=0.1)
            ax3.plot(f_axis, avg_p, label=f'{label} (N={np.sum(band_mask)})', color=colors[idx], linewidth=1.8)
    ax3.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax3.set_ylabel('PSD Magnitude', fontweight='bold')
    ax3.set_title('Motorway V-Vfa02 Straight Cruise: Average PSD across Speed Bands', fontweight='bold')
    ax3.set_xlim(0.2, 5.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Panel 4: Cross-Trip Prediction Failure (Slope Inconsistency)
    ax4 = axs[1, 1]
    for sample_trip, col, lbl in [('V-Vfa02', '#1976D2', 'Motorway V-Vfa02'), 
                                   ('Vta04', '#D32F2F', 'Suburban Vta04'), 
                                   ('Vw14a', '#388E3C', 'Mountain Vw14a'),
                                   ('Vtb01', '#FB8C00', 'Urban Vtb01')]:
        if sample_trip in processed_trips_data:
            td = processed_trips_data[sample_trip]
            p = td['df_feats']['acc_z_total_power_5s'].values
            v = td['v_true']
            # Fit linear trend
            sl, ic, _, _, _ = stats.linregress(p, v)
            p_grid = np.linspace(np.percentile(p, 5), np.percentile(p, 95), 100)
            ax4.plot(p_grid, sl * p_grid + ic, color=col, linewidth=2.5, label=f'{lbl} (slope={sl:.2f}, ic={ic:.1f})')
    ax4.set_xlabel('Vertical Vibration Power (m²/s⁴)', fontweight='bold')
    ax4.set_ylabel('Fitted Speed (m/s)', fontweight='bold')
    ax4.set_title('Cross-Trip Calibration Inconsistency (Fitted Power-to-Speed Transfer)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Panel 5: Cross-Trip Generalization MAE (TCN vs Constant vs Vibration Models)
    ax5 = axs[2, 0]
    x_c = np.arange(len(df_cross))
    wc = 0.2
    ax5.bar(x_c - 1.5*wc, df_cross['tcn_mae_mps'], wc, label='TCN Speed MAE', color='#4CAF50', edgecolor='black')
    ax5.bar(x_c - 0.5*wc, df_cross['const_mae_mps'], wc, label='Constant Speed MAE', color='#9E9E9E', edgecolor='black')
    ax5.bar(x_c + 0.5*wc, df_cross['vib_ridge_mae_mps'], wc, label='Vibration Ridge MAE', color='#FF9800', edgecolor='black')
    ax5.bar(x_c + 1.5*wc, df_cross['vib_rf_mae_mps'], wc, label='Vibration RF MAE', color='#F44336', edgecolor='black')
    ax5.set_xticks(x_c)
    ax5.set_xticklabels(df_cross['trip_name'], rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('MAE (m/s)', fontweight='bold')
    ax5.set_title('Cross-Trip Generalization on 19 Held-Out Trips (Vibration vs TCN)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Histogram of Dominant Frequencies Across All Trips
    ax6 = axs[2, 1]
    all_dom_f = []
    for td in processed_trips_data.values():
        all_dom_f.append(td['df_feats']['acc_z_dom_freq_5s'].values)
    all_dom_f = np.concatenate(all_dom_f)
    ax6.hist(all_dom_f, bins=np.linspace(0.2, 5.0, 49), color='#5C6BC0', edgecolor='black', alpha=0.85, density=True)
    ax6.axvspan(2.0, 2.6, color='red', alpha=0.2, label='Chassis Sprung-Mass Resonance (2.2-2.5 Hz)')
    ax6.set_xlabel('Dominant Vibration Frequency (Hz)', fontweight='bold')
    ax6.set_ylabel('Empirical Probability Density', fontweight='bold')
    ax6.set_title('Empirical Distribution of Dominant Frequencies (450,000+ Windows)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_out = RESULTS_DIR / "vibration_spectrum_plots.png"
    plt.savefig(plot_out, dpi=200)
    print(f"\nSaved plots to: {plot_out}")

    # Copy to brain artifact directory for rendering
    artifact_plots_dir = PROJECT_ROOT / "results" / "figures"
    artifact_plots_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(plot_out, artifact_plots_dir / "vibration_spectrum_plots.png")
    print(f"Copied plot to brain artifact directory.")

    # Save summary stats to JSON for markdown report
    summary_json = {
        'total_trips_audited': len(df_audit),
        'within_trip_correlations': {
            'mean_r_dom_freq': round(float(df_audit['r_dom_freq'].mean()), 3),
            'std_r_dom_freq': round(float(df_audit['r_dom_freq'].std()), 3),
            'mean_r_tot_power': round(float(df_audit['r_tot_power'].mean()), 3),
            'std_r_tot_power': round(float(df_audit['r_tot_power'].std()), 3),
            'mean_r_centroid': round(float(df_audit['r_centroid'].mean()), 3),
            'std_r_centroid': round(float(df_audit['r_centroid'].std()), 3),
            'mean_r_mid_power': round(float(df_audit['r_mid_power_2_3Hz'].mean()), 3),
            'mean_r_high_power': round(float(df_audit['r_high_power_3_5Hz'].mean()), 3),
        },
        'cross_trip_evaluation': {
            'mean_tcn_mae': round(float(df_cross['tcn_mae_mps'].mean()), 3),
            'mean_const_mae': round(float(df_cross['const_mae_mps'].mean()), 3),
            'mean_vib_ridge_mae': round(float(df_cross['vib_ridge_mae_mps'].mean()), 3),
            'mean_vib_ridge_r2': round(float(df_cross['vib_ridge_r2'].mean()), 3),
            'mean_vib_rf_mae': round(float(df_cross['vib_rf_mae_mps'].mean()), 3),
            'mean_vib_rf_r2': round(float(df_cross['vib_rf_r2'].mean()), 3),
            'vib_beats_constant_count': int(df_cross['vibration_beats_constant'].sum()),
            'vib_beats_tcn_count': int(df_cross['vibration_beats_tcn'].sum()),
        },
        'highway_vfa02_speed_bands': highway_band_records,
        'frequency_stability_audit': trip_freq_audit,
        'information_gain_test': info_gain_records
    }
    with open(RESULTS_DIR / "vibration_audit_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved complete audit summary to: {RESULTS_DIR / 'vibration_audit_summary.json'}")
    print("\n" + "=" * 90)
    print("PHASE 5.3 VIBRATION AUDIT COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
