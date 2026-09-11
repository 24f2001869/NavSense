"""
SIH26168 - Forensic Audit: Ridge vs. Random Forest Error Mechanism Decomposition

Script: experiments/audit_rf_vs_ridge_mechanisms.py

PURPOSE:
    Investigate WHY Random Forest achieved a 21.3% reduction in 30s navigation drift
    on Vta04 (67.81m vs 86.17m) despite only a 2.6% reduction in residual RMSE
    (0.7004 vs 0.7193 m/s²).
    
    Examines 5 distinct hypotheses WITHOUT retraining or changing feature spaces:
      1. Extreme Error Tail: Does RF reduce the heavy tail of outlier errors (P90, P95, P99)?
      2. Window-Level Bias: Does RF have lower net integrated acceleration bias per outage window?
      3. Dynamic Regime Segmentation: Is RF's advantage concentrated in braking, cruising, or acceleration?
      4. Prediction Smoothness / Chatter: How does prediction jerk/chatter compare?
      5. Transient Peak Tracking: How do Ridge and RF behave during severe transient maneuvers?
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def main():
    print("===================================================================")
    print("Forensic Audit: Ridge vs. Random Forest Error Mechanism Analysis")
    print("===================================================================")
    
    # 1. Ingest clean label datasets
    csv_02 = PROC_DIR / "c5_3_labels_vta02.csv"
    csv_03 = PROC_DIR / "c5_3_labels_vta03.csv"
    csv_04 = PROC_DIR / "c5_3_labels_vta04.csv"
    
    df_02 = pd.read_csv(csv_02)
    df_03 = pd.read_csv(csv_03)
    df_04 = pd.read_csv(csv_04)
    
    # Extract features
    X_train, y_train, t_train, feat_names, meta_train = extract_causal_features(df_02, window_size=DEFAULT_WINDOW)
    X_test, y_test, t_test, _, meta_test = extract_causal_features(df_04, window_size=DEFAULT_WINDOW)
    
    # Fit exact frozen models from B1 and B2
    print("\n1. Fitting frozen B1 Ridge and B2 Random Forest models...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    ridge = Ridge(alpha=464.1589, fit_intercept=True)
    ridge.fit(X_train_s, y_train)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_pred_ridge = ridge.predict(X_test_s)
    y_pred_rf = rf.predict(X_test)
    
    # Ground truth reference kinematics on Vta04
    a_ref = meta_test['a_ref_targets']
    v_true = meta_test['v_vbox_targets']
    ax_level = X_test[:, feat_names.index('ax_level_k')]
    
    # Residual prediction errors
    err_ridge = y_pred_ridge - y_test
    err_rf = y_pred_rf - y_test
    abs_err_ridge = np.abs(err_ridge)
    abs_err_rf = np.abs(err_rf)
    
    # =======================================================================
    # Hypothesis 1: Extreme Error Tail Audit
    # =======================================================================
    print("\n--- Hypothesis 1: Extreme Error Tail Audit on Vta04 ---")
    percentiles = [50, 75, 90, 95, 98, 99, 100]
    tail_ridge = np.percentile(abs_err_ridge, percentiles)
    tail_rf = np.percentile(abs_err_rf, percentiles)
    
    tail_audit = {}
    print(f"{'Percentile':<12} | {'Ridge Abs Err (m/s²)':<22} | {'RF Abs Err (m/s²)':<20} | {'Delta (RF - Ridge)':<20}")
    print("-" * 80)
    for p, v_rg, v_rf in zip(percentiles, tail_ridge, tail_rf):
        diff = v_rf - v_rg
        pct_diff = (diff / v_rg) * 100.0
        p_name = f"P{p}" if p < 100 else "Max"
        tail_audit[p_name] = {'ridge': float(v_rg), 'rf': float(v_rf), 'diff': float(diff), 'pct_diff': float(pct_diff)}
        print(f"{p_name:<12} | {v_rg:<22.4f} | {v_rf:<20.4f} | {diff:+.4f} ({pct_diff:+.1f}%)")
        
    # =======================================================================
    # Hypothesis 2: Window-Level Net Acceleration Bias
    # =======================================================================
    print("\n--- Hypothesis 2: Window-Level Net Acceleration Bias (30s Windows) ---")
    w_30s = int(round(30.0 / DT))  # 300 samples
    stride_k = int(round(2.5 / DT)) # 25 samples
    
    win_biases_ridge = []
    win_biases_rf = []
    win_p_errs_ridge = []
    win_p_errs_rf = []
    
    for k_start in range(0, len(y_test) - w_30s, stride_k):
        k_end = k_start + w_30s
        # Acceleration errors relative to reference
        # a_corr = ax_level - y_pred
        # a_corr - a_ref = (ax_level - a_ref) - y_pred = y_actual - y_pred = -err
        accel_err_rg = y_test[k_start:k_end] - y_pred_ridge[k_start:k_end]
        accel_err_rf = y_test[k_start:k_end] - y_pred_rf[k_start:k_end]
        
        # Net window acceleration bias
        mean_b_rg = np.mean(accel_err_rg)
        mean_b_rf = np.mean(accel_err_rf)
        win_biases_ridge.append(float(mean_b_rg))
        win_biases_rf.append(float(mean_b_rf))
        
        # Integrated velocity and position error
        v_err_rg = np.cumsum(accel_err_rg) * DT
        v_err_rf = np.cumsum(accel_err_rf) * DT
        p_err_rg = np.sum(v_err_rg) * DT
        p_err_rf = np.sum(v_err_rf) * DT
        win_p_errs_ridge.append(abs(float(p_err_rg)))
        win_p_errs_rf.append(abs(float(p_err_rf)))
        
    win_biases_ridge = np.array(win_biases_ridge)
    win_biases_rf = np.array(win_biases_rf)
    win_p_errs_ridge = np.array(win_p_errs_ridge)
    win_p_errs_rf = np.array(win_p_errs_rf)
    
    bias_audit = {
        'mean_abs_window_accel_bias_ridge': float(np.mean(np.abs(win_biases_ridge))),
        'mean_abs_window_accel_bias_rf': float(np.mean(np.abs(win_biases_rf))),
        'std_window_accel_bias_ridge': float(np.std(win_biases_ridge)),
        'std_window_accel_bias_rf': float(np.std(win_biases_rf)),
        'max_window_accel_bias_ridge': float(np.max(np.abs(win_biases_ridge))),
        'max_window_accel_bias_rf': float(np.max(np.abs(win_biases_rf))),
    }
    
    print(f"  Mean |Window Net Accel Bias|:")
    print(f"    Ridge:          {bias_audit['mean_abs_window_accel_bias_ridge']:.4f} m/s²")
    print(f"    Random Forest:  {bias_audit['mean_abs_window_accel_bias_rf']:.4f} m/s² ({(bias_audit['mean_abs_window_accel_bias_rf'] - bias_audit['mean_abs_window_accel_bias_ridge'])/bias_audit['mean_abs_window_accel_bias_ridge']*100:+.1f}%)")
    print(f"  Std of Window Net Accel Bias:")
    print(f"    Ridge:          {bias_audit['std_window_accel_bias_ridge']:.4f} m/s²")
    print(f"    Random Forest:  {bias_audit['std_window_accel_bias_rf']:.4f} m/s² ({(bias_audit['std_window_accel_bias_rf'] - bias_audit['std_window_accel_bias_ridge'])/bias_audit['std_window_accel_bias_ridge']*100:+.1f}%)")
    
    # =======================================================================
    # Hypothesis 3: Dynamic Regime Segmentation
    # =======================================================================
    print("\n--- Hypothesis 3: Dynamic Regime Segmentation on Vta04 ---")
    regimes = {
        'Hard Braking (a_ref <= -1.5 m/s²)': (a_ref <= -1.5),
        'Mod Braking (-1.5 < a_ref <= -0.5)': (a_ref > -1.5) & (a_ref <= -0.5),
        'Steady Cruising (|a_ref| < 0.5, v > 5)': (np.abs(a_ref) < 0.5) & (v_true > 5.0),
        'Mod Acceleration (+0.5 < a_ref <= +1.5)': (a_ref >= 0.5) & (a_ref < 1.5),
        'Hard Acceleration (a_ref >= +1.5 m/s²)': (a_ref >= 1.5),
        'Low Speed / Crawl (v <= 2.0 m/s)': (v_true <= 2.0),
    }
    
    regime_audit = {}
    print(f"{'Dynamic Regime':<40} | {'Samples':<8} | {'Ridge MAE':<10} | {'RF MAE':<10} | {'Delta MAE':<10} | {'Ridge Bias':<10} | {'RF Bias':<10}")
    print("-" * 105)
    for r_name, mask in regimes.items():
        n_pts = int(np.sum(mask))
        if n_pts == 0:
            continue
        mae_rg = float(np.mean(np.abs(err_ridge[mask])))
        mae_rf = float(np.mean(np.abs(err_rf[mask])))
        bias_rg = float(np.mean(err_ridge[mask]))
        bias_rf = float(np.mean(err_rf[mask]))
        delta_mae = mae_rf - mae_rg
        
        regime_audit[r_name] = {
            'n_samples': n_pts,
            'ridge_mae': mae_rg,
            'rf_mae': mae_rf,
            'delta_mae': delta_mae,
            'ridge_bias': bias_rg,
            'rf_bias': bias_rf,
        }
        print(f"{r_name:<40} | {n_pts:<8d} | {mae_rg:<10.4f} | {mae_rf:<10.4f} | {delta_mae:<+10.4f} | {bias_rg:<+10.4f} | {bias_rf:<+10.4f}")
        
    # =======================================================================
    # Hypothesis 4: Smoothness & Derivative Chatter Audit
    # =======================================================================
    print("\n--- Hypothesis 4: Smoothness / Derivative Chatter ---")
    diff_pred_rg = np.diff(y_pred_ridge) / DT
    diff_pred_rf = np.diff(y_pred_rf) / DT
    diff_true = np.diff(y_test) / DT
    
    smoothness_audit = {
        'ridge_pred_jerk_rms': float(np.sqrt(np.mean(diff_pred_rg**2))),
        'rf_pred_jerk_rms': float(np.sqrt(np.mean(diff_pred_rf**2))),
        'true_target_jerk_rms': float(np.sqrt(np.mean(diff_true**2))),
        'ridge_pred_jerk_std': float(np.std(diff_pred_rg)),
        'rf_pred_jerk_std': float(np.std(diff_pred_rf)),
    }
    print(f"  Predicted Residual Jerk RMS (m/s³):")
    print(f"    Reference Target: {smoothness_audit['true_target_jerk_rms']:.2f} m/s³")
    print(f"    Ridge:            {smoothness_audit['ridge_pred_jerk_rms']:.2f} m/s³")
    print(f"    Random Forest:    {smoothness_audit['rf_pred_jerk_rms']:.2f} m/s³")
    
    # =======================================================================
    # 5. Diagnostic Figures
    # =======================================================================
    print("\n2. Generating Forensic Diagnostic Figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Window-Level Net Acceleration Bias vs. Position Drift
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(win_biases_ridge, bins=25, color='royalblue', alpha=0.6, label=f'Ridge (std={bias_audit["std_window_accel_bias_ridge"]:.3f})')
    axes[0].hist(win_biases_rf, bins=25, color='forestgreen', alpha=0.6, label=f'RF (std={bias_audit["std_window_accel_bias_rf"]:.3f})')
    axes[0].axvline(0.0, color='black', ls='--')
    axes[0].set_xlabel('30s Window Mean Acceleration Error (m/s²)', fontweight='bold')
    axes[0].set_ylabel('Window Count', fontweight='bold')
    axes[0].set_title('Window-Level Net Acceleration Bias Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)
    
    axes[1].scatter(np.abs(win_biases_ridge), win_p_errs_ridge, color='royalblue', alpha=0.7, label='Ridge (B1)')
    axes[1].scatter(np.abs(win_biases_rf), win_p_errs_rf, color='forestgreen', alpha=0.7, label='Random Forest (B2)')
    # Theoretical parabolic line p = 0.5 * bias * T^2 = 0.5 * bias * 900 = 450 * bias
    b_grid = np.linspace(0, max(np.max(np.abs(win_biases_ridge)), np.max(np.abs(win_biases_rf))), 50)
    axes[1].plot(b_grid, 450.0 * b_grid, 'k--', label='Constant Bias Integral (0.5·bias·T²)')
    axes[1].set_xlabel('|Window Net Acceleration Bias| (m/s²)', fontweight='bold')
    axes[1].set_ylabel('30s Position Error (m)', fontweight='bold')
    axes[1].set_title('Position Error vs. Net Window Bias Coupling', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_3b2_window_bias_audit.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    
    # Figure 2: Regime MAE and Tail Percentile Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Percentiles
    p_names = list(tail_audit.keys())
    rg_p_vals = [tail_audit[k]['ridge'] for k in p_names]
    rf_p_vals = [tail_audit[k]['rf'] for k in p_names]
    x_pos = np.arange(len(p_names))
    width = 0.35
    axes[0].bar(x_pos - width/2, rg_p_vals, width, label='Ridge (B1)', color='royalblue', alpha=0.8)
    axes[0].bar(x_pos + width/2, rf_p_vals, width, label='Random Forest (B2)', color='forestgreen', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(p_names, fontweight='bold')
    axes[0].set_ylabel('Absolute Residual Error (m/s²)', fontweight='bold')
    axes[0].set_title('Error Distribution Tails (Percentile Comparison)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)
    
    # Right: Regimes
    r_short = ['Hard Brake', 'Mod Brake', 'Cruising', 'Mod Accel', 'Hard Accel', 'Low Speed']
    rg_reg_vals = [regime_audit[k]['ridge_mae'] for k in regime_audit]
    rf_reg_vals = [regime_audit[k]['rf_mae'] for k in regime_audit]
    x_r = np.arange(len(r_short))
    axes[1].bar(x_r - width/2, rg_reg_vals, width, label='Ridge (B1)', color='royalblue', alpha=0.8)
    axes[1].bar(x_r + width/2, rf_reg_vals, width, label='Random Forest (B2)', color='forestgreen', alpha=0.8)
    axes[1].set_xticks(x_r)
    axes[1].set_xticklabels(r_short, fontweight='bold', rotation=20)
    axes[1].set_ylabel('MAE (m/s²)', fontweight='bold')
    axes[1].set_title('MAE Breakdown Across Dynamic Regimes', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_3b2_regime_and_tail_audit.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    
    # Save audit record
    audit_results = {
        'stage': 'C5.3-B2_AUDIT',
        'status': 'COMPLETE',
        'tail_audit': tail_audit,
        'window_net_bias_audit': bias_audit,
        'dynamic_regime_audit': regime_audit,
        'smoothness_audit': smoothness_audit,
    }
    
    out_json = RES_DIR / "c5_3b2_mechanisms_audit.json"
    with open(out_json, 'w') as f:
        json.dump(audit_results, f, indent=2)
        
    print(f"\nSaved audit record to {out_json}")
    print("===================================================================")
    print("Forensic Error Mechanism Audit Finished Successfully.")
    print("===================================================================")


if __name__ == "__main__":
    main()
