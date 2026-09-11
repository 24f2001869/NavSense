"""
SIH26168 - Stage C5.3-B1: Linear Residual Estimator Baseline (Ridge Regression)

Script: experiments/run_ridge_baseline_c5_3b1.py

PURPOSE:
    Answer the two fundamental scientific questions for Stage C5.3-B1:
    
    Question A (Statistical Predictability):
        Is the time-varying dynamic acceleration residual r_a(k) predictable from
        causal smartphone IMU features using a regularized linear model (Ridge)?
        Does Ridge beat the constant-mean baseline r_hat = mean(r_train)?
        
    Question B (Navigation Usefulness):
        When the Ridge predicted residual is subtracted from leveled acceleration:
            a_corrected = a_x^level - r_hat_a
        does it reduce integrated velocity and position drift across outage horizons
        (5s, 10s, 20s, 30s, 60s) compared to raw baseline a_x^level and constant calibration?

STRICT PROTOCOL:
    - Train on Vta02 (Urban).
    - StandardScaler fit ONLY on Vta02 training features (zero val/test leakage).
    - Model selection: Tune alpha on Vta03 (Validation/Model-Selection).
    - Freeze alpha* and evaluate on Vta04 (Final Test, strictly untouched during tuning).
    - Zero RF, GBDT, MLP, ESKF, NHC, map matching, or heading modifications.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.verify_causal_pipeline_c5_3b0 import extract_causal_features, DEFAULT_WINDOW, DT

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"


# ===========================================================================
# 1. Statistical Predictability Evaluation (Question A)
# ===========================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes standard regression metrics and error distribution parameters."""
    err = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    err_mean = float(np.mean(err))
    err_std = float(np.std(err))
    
    # Pearson correlation between predicted and reference
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_pred > 1e-8 and std_true > 1e-8:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        corr = 0.0
        
    return {
        'mae_ms2': mae,
        'rmse_ms2': rmse,
        'r2': r2,
        'residual_error_mean_ms2': err_mean,
        'residual_error_std_ms2': err_std,
        'pred_true_correlation': corr,
        'pred_std_ms2': float(std_pred),
        'true_std_ms2': float(std_true),
    }


# ===========================================================================
# 2. Navigation Outage Simulation (Question B)
# ===========================================================================

def simulate_outage_navigation(v_true: np.ndarray, a_series_dict: dict, 
                               horizons: list = [5.0, 10.0, 20.0, 30.0, 60.0],
                               stride_s: float = 2.5) -> dict:
    """
    Simulates dead-reckoning outages over rolling windows.
    Compares integration of:
      - 'raw_baseline': a_x^level
      - 'const_cal': a_x^level - r_bar_train
      - 'ridge_corrected': a_x^level - r_hat_ridge
      - 'reference_oracle': a_ref (upper bound benchmark)
    """
    n = len(v_true)
    stride_k = max(1, int(round(stride_s / DT)))
    
    horizon_results = {}
    
    for h in horizons:
        w_k = int(round(h / DT))
        if n < w_k + 2:
            continue
            
        h_key = f"{int(h)}s"
        method_stats = {m: {'pos_errors': [], 'final_vel_errors': [], 'drift_pcts': []} 
                        for m in a_series_dict}
        
        n_evals = 0
        for k_start in range(0, n - w_k, stride_k):
            k_end = k_start + w_k
            v_win_true = v_true[k_start:k_end + 1]
            t_win = np.arange(len(v_win_true)) * DT
            dist_true = float(np.trapezoid(v_win_true, t_win))
            n_evals += 1
            
            for m_name, a_full in a_series_dict.items():
                a_win = a_full[k_start:k_end + 1]
                
                # Trapezoidal integration of acceleration -> velocity
                v_est = np.zeros(len(v_win_true))
                v_est[0] = v_win_true[0]  # Perfect initial velocity fix at outage start
                for j in range(1, len(v_win_true)):
                    v_est[j] = v_est[j-1] + 0.5 * (a_win[j] + a_win[j-1]) * DT
                    
                v_err = v_est - v_win_true
                final_v_err = abs(float(v_err[-1]))
                
                # Integration of velocity -> position
                p_err = 0.0
                for j in range(1, len(v_win_true)):
                    p_err += 0.5 * (v_err[j] + v_err[j-1]) * DT
                final_p_err = abs(float(p_err))
                
                method_stats[m_name]['pos_errors'].append(final_p_err)
                method_stats[m_name]['final_vel_errors'].append(final_v_err)
                if dist_true > 1.0:
                    method_stats[m_name]['drift_pcts'].append(final_p_err / dist_true * 100.0)
                    
        # Summary metrics for horizon h
        h_summary = {'horizon_s': h, 'n_windows': n_evals, 'methods': {}}
        for m_name in a_series_dict:
            p_errs = method_stats[m_name]['pos_errors']
            v_errs = method_stats[m_name]['final_vel_errors']
            d_pcts = method_stats[m_name]['drift_pcts']
            
            h_summary['methods'][m_name] = {
                'mean_pos_error_m': float(np.mean(p_errs)) if p_errs else 0.0,
                'median_pos_error_m': float(np.median(p_errs)) if p_errs else 0.0,
                'p90_pos_error_m': float(np.percentile(p_errs, 90)) if p_errs else 0.0,
                'mean_vel_error_ms': float(np.mean(v_errs)) if v_errs else 0.0,
                'mean_drift_pct': float(np.mean(d_pcts)) if d_pcts else 0.0,
            }
        horizon_results[h_key] = h_summary
        
    return horizon_results


# ===========================================================================
# 3. Main Stage C5.3-B1 Execution
# ===========================================================================

def main():
    print("===================================================================")
    print("Stage C5.3-B1: Linear Baseline (Ridge Regression)")
    print("===================================================================")
    
    # 1. Ingest clean label datasets
    csv_02 = PROC_DIR / "c5_3_labels_vta02.csv"
    csv_03 = PROC_DIR / "c5_3_labels_vta03.csv"
    csv_04 = PROC_DIR / "c5_3_labels_vta04.csv"
    
    df_02 = pd.read_csv(csv_02)
    df_03 = pd.read_csv(csv_03)
    df_04 = pd.read_csv(csv_04)
    
    # 2. Extract strictly causal features
    print("\n1. Extracting Causal Feature Matrices (W = 15 samples = 1.5s)...")
    X_train_raw, y_train, t_train, feat_names, meta_train = extract_causal_features(df_02, window_size=DEFAULT_WINDOW)
    X_val_raw, y_val, t_val, _, meta_val = extract_causal_features(df_03, window_size=DEFAULT_WINDOW)
    X_test_raw, y_test, t_test, _, meta_test = extract_causal_features(df_04, window_size=DEFAULT_WINDOW)
    
    print(f"  Train (Vta02):      X={X_train_raw.shape}, y={y_train.shape}")
    print(f"  Validation (Vta03): X={X_val_raw.shape}, y={y_val.shape} (Model-Selection Set)")
    print(f"  Test (Vta04):       X={X_test_raw.shape}, y={y_test.shape} (FINAL TEST - Untouched during tuning)")
    
    # 3. Standardize features (FIT ONLY ON TRAIN)
    print("\n2. Fitting StandardScaler on Vta02 Train Set ONLY...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Baseline constant-mean predictor
    r_bar_train = float(np.mean(y_train))
    print(f"  Constant Baseline (r_bar_train): {r_bar_train:+.4f} m/s²")
    
    # 4. Hyperparameter tuning on Vta03 (Model Selection)
    print("\n3. Tuning Ridge Regularization alpha on Vta03 Validation Set...")
    alphas = np.logspace(-4, 6, 25)
    tuning_records = []
    
    best_alpha = None
    best_val_mae = float('inf')
    best_model = None
    
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        m_train = compute_regression_metrics(y_train, y_train_pred)
        m_val = compute_regression_metrics(y_val, y_val_pred)
        
        tuning_records.append({
            'alpha': float(alpha),
            'train_mae': m_train['mae_ms2'],
            'train_rmse': m_train['rmse_ms2'],
            'train_r2': m_train['r2'],
            'val_mae': m_val['mae_ms2'],
            'val_rmse': m_val['rmse_ms2'],
            'val_r2': m_val['r2'],
            'val_corr': m_val['pred_true_correlation'],
        })
        
        if m_val['mae_ms2'] < best_val_mae:
            best_val_mae = m_val['mae_ms2']
            best_alpha = float(alpha)
            best_model = model
            
    print(f"  Optimal alpha selected: {best_alpha:.4e} (Validation MAE = {best_val_mae:.4f} m/s²)")
    
    # 5. Freeze Model & Compute Statistical Metrics (Question A)
    print("\n4. Question A: Statistical Predictability Analysis...")
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Constant baseline predictions
    y_train_const = np.full_like(y_train, r_bar_train)
    y_val_const = np.full_like(y_val, r_bar_train)
    y_test_const = np.full_like(y_test, r_bar_train)
    
    metrics_train_ridge = compute_regression_metrics(y_train, y_train_pred)
    metrics_train_const = compute_regression_metrics(y_train, y_train_const)
    
    metrics_val_ridge = compute_regression_metrics(y_val, y_val_pred)
    metrics_val_const = compute_regression_metrics(y_val, y_val_const)
    
    metrics_test_ridge = compute_regression_metrics(y_test, y_test_pred)
    metrics_test_const = compute_regression_metrics(y_test, y_test_const)
    
    print("\n--- Statistical Predictability Summary (Question A) ---")
    print(f"{'Partition':<25} | {'Model':<15} | {'MAE (m/s²)':<10} | {'RMSE (m/s²)':<11} | {'R²':<8} | {'Corr r':<8}")
    print("-" * 88)
    print(f"{'Train (Vta02)':<25} | {'Const Mean':<15} | {metrics_train_const['mae_ms2']:<10.4f} | {metrics_train_const['rmse_ms2']:<11.4f} | {metrics_train_const['r2']:<8.4f} | {metrics_train_const['pred_true_correlation']:<8.4f}")
    print(f"{'Train (Vta02)':<25} | {'Ridge (alpha*)':<15} | {metrics_train_ridge['mae_ms2']:<10.4f} | {metrics_train_ridge['rmse_ms2']:<11.4f} | {metrics_train_ridge['r2']:<8.4f} | {metrics_train_ridge['pred_true_correlation']:<8.4f}")
    print("-" * 88)
    print(f"{'Val/Select (Vta03)':<25} | {'Const Mean':<15} | {metrics_val_const['mae_ms2']:<10.4f} | {metrics_val_const['rmse_ms2']:<11.4f} | {metrics_val_const['r2']:<8.4f} | {metrics_val_const['pred_true_correlation']:<8.4f}")
    print(f"{'Val/Select (Vta03)':<25} | {'Ridge (alpha*)':<15} | {metrics_val_ridge['mae_ms2']:<10.4f} | {metrics_val_ridge['rmse_ms2']:<11.4f} | {metrics_val_ridge['r2']:<8.4f} | {metrics_val_ridge['pred_true_correlation']:<8.4f}")
    print("-" * 88)
    print(f"{'Final Test (Vta04)':<25} | {'Const Mean':<15} | {metrics_test_const['mae_ms2']:<10.4f} | {metrics_test_const['rmse_ms2']:<11.4f} | {metrics_test_const['r2']:<8.4f} | {metrics_test_const['pred_true_correlation']:<8.4f}")
    print(f"{'Final Test (Vta04)':<25} | {'Ridge (alpha*)':<15} | {metrics_test_ridge['mae_ms2']:<10.4f} | {metrics_test_ridge['rmse_ms2']:<11.4f} | {metrics_test_ridge['r2']:<8.4f} | {metrics_test_ridge['pred_true_correlation']:<8.4f}")
    print("-" * 88)
    
    # 6. Question B: Navigation Outage Simulation
    print("\n5. Question B: Navigation Outage Drift Simulation...")
    
    def prep_acceleration_dict(meta: dict, y_pred_arr: np.ndarray):
        ax_lvl_idx = feat_names.index('ax_level_k')
        # We need ax_level across valid target samples
        # In meta: 'v_vbox_targets', 'a_ref_targets'
        # ax_level is directly a_ref + y
        a_ref = meta['a_ref_targets']
        y_actual = meta['target_residual'] if 'target_residual' in meta else (meta['a_ref_targets'] * 0.0) # will retrieve from extraction
        return a_ref
        
    # Re-extract clean instantaneous arrays aligned with targets
    ax_level_train = X_train_raw[:, feat_names.index('ax_level_k')]
    ax_level_val = X_val_raw[:, feat_names.index('ax_level_k')]
    ax_level_test = X_test_raw[:, feat_names.index('ax_level_k')]
    
    # Build acceleration series for each trip
    trips_accel = {
        'Vta02': {
            'v_true': meta_train['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_train,
                'const_cal': ax_level_train - r_bar_train,
                'ridge_corrected': ax_level_train - y_train_pred,
                'oracle_ref': meta_train['a_ref_targets'],
            }
        },
        'Vta03': {
            'v_true': meta_val['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_val,
                'const_cal': ax_level_val - r_bar_train,
                'ridge_corrected': ax_level_val - y_val_pred,
                'oracle_ref': meta_val['a_ref_targets'],
            }
        },
        'Vta04': {
            'v_true': meta_test['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_test,
                'const_cal': ax_level_test - r_bar_train,
                'ridge_corrected': ax_level_test - y_test_pred,
                'oracle_ref': meta_test['a_ref_targets'],
            }
        }
    }
    
    nav_outage_results = {}
    for trip_name, t_info in trips_accel.items():
        print(f"\n  Simulating Outages on {trip_name}...")
        nav_res = simulate_outage_navigation(t_info['v_true'], t_info['series'], 
                                            horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
                                            stride_s=2.5)
        nav_outage_results[trip_name] = nav_res
        
        print(f"    Horizon | Raw Base Pos Err | Const Cal Pos Err | Ridge Corrected Pos Err | Oracle Ref Pos Err")
        for h_key, h_data in nav_res.items():
            m_data = h_data['methods']
            p_raw = m_data['raw_baseline']['mean_pos_error_m']
            p_const = m_data['const_cal']['mean_pos_error_m']
            p_ridge = m_data['ridge_corrected']['mean_pos_error_m']
            p_ref = m_data['oracle_ref']['mean_pos_error_m']
            print(f"    {h_key:<7} | {p_raw:<16.2f}m | {p_const:<17.2f}m | {p_ridge:<23.2f}m | {p_ref:<18.2f}m")
            
    # 7. Coefficient Inspection
    print("\n6. Inspecting Top Ridge Model Coefficients...")
    coefs = best_model.coef_
    intercept = float(best_model.intercept_)
    sorted_indices = np.argsort(np.abs(coefs))[::-1]
    
    top_coefs = []
    print(f"  Ridge Intercept: {intercept:+.4f} m/s²")
    print(f"  Top 10 Most Influential Features (|beta|):")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        feat_name = feat_names[idx]
        val_coef = float(coefs[idx])
        top_coefs.append({'rank': rank, 'feature': feat_name, 'coefficient': val_coef})
        print(f"    {rank:2d}. {feat_name:<30}: {val_coef:+.4f}")
        
    # 8. Diagnostic Figures
    print("\n7. Generating Diagnostic Figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Alpha Regularization Tuning Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    t_alphas = [r['alpha'] for r in tuning_records]
    train_maes = [r['train_mae'] for r in tuning_records]
    val_maes = [r['val_mae'] for r in tuning_records]
    val_r2s = [r['val_r2'] for r in tuning_records]
    
    ax.semilogx(t_alphas, train_maes, 'b-o', lw=2.0, label='Train MAE (Vta02)')
    ax.semilogx(t_alphas, val_maes, 'r-s', lw=2.0, label='Val MAE (Vta03)')
    ax.axvline(best_alpha, color='black', ls='--', lw=1.5, label=f'Best alpha* = {best_alpha:.2e}')
    ax.set_xlabel('Ridge Regularization Alpha (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (m/s²)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.3-B1: Ridge Regularization Alpha Tuning on Vta03', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_3b1_alpha_tuning_curve.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    
    # Figure 2: Scatter Plot of Predicted vs Actual Residual
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scatter_configs = [
        ('Train (Vta02)', y_train, y_train_pred, axes[0], 'royalblue'),
        ('Val/Select (Vta03)', y_val, y_val_pred, axes[1], 'crimson'),
        ('Final Test (Vta04)', y_test, y_test_pred, axes[2], 'forestgreen'),
    ]
    
    for title, y_act, y_pr, ax_sc, col in scatter_configs:
        corr = np.corrcoef(y_act, y_pr)[0, 1] if np.std(y_pr) > 1e-8 else 0.0
        ax_sc.scatter(y_act, y_pr, color=col, alpha=0.3, edgecolors='none', s=15)
        # Identity line
        lim_min = min(np.min(y_act), np.min(y_pr)) - 0.5
        lim_max = max(np.max(y_act), np.max(y_pr)) + 0.5
        ax_sc.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.5, label='Ideal 1:1')
        ax_sc.set_xlim(lim_min, lim_max)
        ax_sc.set_ylim(lim_min, lim_max)
        ax_sc.set_xlabel('Reference Residual r_a (m/s²)', fontweight='bold')
        ax_sc.set_ylabel('Predicted Residual r_hat_a (m/s²)', fontweight='bold')
        ax_sc.set_title(f"{title}\nr = {corr:.3f}, MAE = {mean_absolute_error(y_act, y_pr):.3f}", fontweight='bold')
        ax_sc.legend(loc='upper left')
        ax_sc.grid(True, alpha=0.5)
        
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_3b1_residual_scatter_audit.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    
    # Figure 3: Time Series Segment Analysis (Actual Residual vs Ridge Prediction vs Remaining Error)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    # Plot on Vta04 test segment: 60s window [30s, 90s]
    t_plot = t_test
    mask_win = (t_plot >= 30.0) & (t_plot <= 90.0)
    
    axes[0].plot(t_plot[mask_win], y_test[mask_win], 'black', lw=1.8, label='Reference Residual r_a(k)')
    axes[0].plot(t_plot[mask_win], y_test_pred[mask_win], 'royalblue', lw=2.0, label='Ridge Prediction r_hat_a(k)')
    axes[0].axhline(r_bar_train, color='gray', ls=':', lw=1.5, label=f'Const Mean ({r_bar_train:+.2f})')
    axes[0].set_ylabel('Residual (m/s²)', fontweight='bold')
    axes[0].set_title('Stage C5.3-B1: Causal Ridge Residual Prediction on Vta04 Test Segment', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.5)
    
    rem_err_raw = ax_level_test - meta_test['a_ref_targets']
    rem_err_ridge = (ax_level_test - y_test_pred) - meta_test['a_ref_targets']
    
    axes[1].plot(t_plot[mask_win], rem_err_raw[mask_win], 'gray', lw=1.5, alpha=0.7, label='Raw Error (ax_level - a_ref)')
    axes[1].plot(t_plot[mask_win], rem_err_ridge[mask_win], 'crimson', lw=1.8, label='Ridge Corrected Error (a_corrected - a_ref)')
    axes[1].axhline(0.0, color='black', ls='--', lw=1.0)
    axes[1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[1].set_ylabel('Acceleration Error (m/s²)', fontweight='bold')
    axes[1].set_title('Remaining Acceleration Error Relative to Offline VBOX Reference', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_3b1_timeseries_tracking.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    
    # Figure 4: Navigation Position Error Scaling across Horizons
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, (trip_name, col) in enumerate([('Vta02', 'navy'), ('Vta03', 'crimson'), ('Vta04', 'forestgreen')]):
        ax_nav = axes[idx]
        h_data = nav_outage_results[trip_name]
        h_vals = [h_data[k]['horizon_s'] for k in h_data]
        p_raw = [h_data[k]['methods']['raw_baseline']['mean_pos_error_m'] for k in h_data]
        p_const = [h_data[k]['methods']['const_cal']['mean_pos_error_m'] for k in h_data]
        p_ridge = [h_data[k]['methods']['ridge_corrected']['mean_pos_error_m'] for k in h_data]
        p_ref = [h_data[k]['methods']['oracle_ref']['mean_pos_error_m'] for k in h_data]
        
        ax_nav.plot(h_vals, p_raw, 'k--o', lw=1.8, label='Raw Baseline')
        ax_nav.plot(h_vals, p_const, 'gray', ls=':', marker='^', lw=1.8, label='Constant Cal')
        ax_nav.plot(h_vals, p_ridge, color=col, marker='s', lw=2.2, label='Ridge Corrected')
        ax_nav.plot(h_vals, p_ref, 'g-.d', lw=1.5, label='Oracle Reference Floor')
        
        ax_nav.set_xlabel('Outage Horizon (seconds)', fontweight='bold')
        if idx == 0:
            ax_nav.set_ylabel('Mean Final Position Error (meters)', fontweight='bold')
        ax_nav.set_title(f"{trip_name} Navigation Outage Drift", fontweight='bold')
        ax_nav.legend(fontsize=9)
        ax_nav.grid(True, alpha=0.5)
        
    plt.tight_layout()
    fig4_path = FIG_DIR / "c5_3b1_navigation_horizon_drift.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    
    # 9. Save Complete Results JSON
    b1_results = {
        'stage': 'C5.3-B1',
        'status': 'COMPLETE',
        'optimal_alpha': best_alpha,
        'scaler_fitted_on': 'Vta02_ONLY',
        'baseline_constant_mean_train_ms2': r_bar_train,
        'statistical_metrics': {
            'train_vta02': {'ridge': metrics_train_ridge, 'constant_mean': metrics_train_const},
            'val_model_selection_vta03': {'ridge': metrics_val_ridge, 'constant_mean': metrics_val_const},
            'final_test_vta04': {'ridge': metrics_test_ridge, 'constant_mean': metrics_test_const},
        },
        'navigation_outage_results': nav_outage_results,
        'top_coefficients': top_coefs,
        'alpha_tuning_records': tuning_records,
    }
    
    out_json = RES_DIR / "c5_3b1_ridge_results.json"
    with open(out_json, 'w') as f:
        json.dump(b1_results, f, indent=2)
        
    print(f"\nSaved complete Ridge baseline results to {out_json}")
    print("===================================================================")
    print("Stage C5.3-B1 execution finished successfully.")
    print("===================================================================")


if __name__ == "__main__":
    main()
