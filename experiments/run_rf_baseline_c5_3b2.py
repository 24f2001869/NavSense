"""
SIH26168 - Stage C5.3-B2: Non-Linear Residual Estimator (Random Forest Regressor)

Script: experiments/run_rf_baseline_c5_3b2.py

PURPOSE:
    Test the fundamental scientific hypothesis for Stage C5.3-B2:
    "Does non-linear model capacity improve generalization and integrated 
     navigation drift beyond the linear Ridge baseline?"
     
STRICT EXPERIMENTAL CONTROLS (ONE-VARIABLE EXPERIMENT):
    - Same 47 causal features as B0/B1 (X_B2 = X_B1).
    - Same causal trailing window: W = 15 samples (1.5 s).
    - Same target: r_a(k) = a_x^level(k) - a_ref(k) (SG9).
    - Same partitions: Train = Vta02, Model-Selection = Vta03, Final Test = Vta04.
    - Vta04 is STRICTLY UNTOUCHED during model tuning and hyperparameter selection.
    - Same navigation experiment: rolling windows, stride = 2.5 s, reset-at-window-start with v_0.
    - Zero ESKF, zero NHC, zero map matching, zero new features.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.verify_causal_pipeline_c5_3b0 import extract_causal_features, DEFAULT_WINDOW, DT
from experiments.run_ridge_baseline_c5_3b1 import compute_regression_metrics, simulate_outage_navigation

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"


def main():
    print("===================================================================")
    print("Stage C5.3-B2: Non-Linear Estimator (Random Forest Regressor)")
    print("===================================================================")
    
    # 1. Ingest clean label datasets
    csv_02 = PROC_DIR / "c5_3_labels_vta02.csv"
    csv_03 = PROC_DIR / "c5_3_labels_vta03.csv"
    csv_04 = PROC_DIR / "c5_3_labels_vta04.csv"
    
    df_02 = pd.read_csv(csv_02)
    df_03 = pd.read_csv(csv_03)
    df_04 = pd.read_csv(csv_04)
    
    # 2. Extract strictly causal features (identical 47 features)
    print("\n1. Extracting Causal Feature Matrices (W = 15 samples = 1.5s, 47 features)...")
    X_train, y_train, t_train, feat_names, meta_train = extract_causal_features(df_02, window_size=DEFAULT_WINDOW)
    X_val, y_val, t_val, _, meta_val = extract_causal_features(df_03, window_size=DEFAULT_WINDOW)
    X_test, y_test, t_test, _, meta_test = extract_causal_features(df_04, window_size=DEFAULT_WINDOW)
    
    print(f"  Train (Vta02):      X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation (Vta03): X={X_val.shape}, y={y_val.shape} (Model-Selection Set)")
    print(f"  Test (Vta04):       X={X_test.shape}, y={y_test.shape} (FINAL TEST - Untouched during tuning)")
    
    # Constant baseline
    r_bar_train = float(np.mean(y_train))
    
    # 3. Load Ridge Baseline from B1 for direct benchmarking
    ridge_json_path = RES_DIR / "c5_3b1_ridge_results.json"
    if ridge_json_path.exists():
        with open(ridge_json_path, 'r') as f:
            b1_data = json.load(f)
        optimal_alpha_ridge = b1_data['optimal_alpha']
    else:
        optimal_alpha_ridge = 464.1589
        
    print(f"\n2. Fitting Reference Ridge Model (alpha = {optimal_alpha_ridge:.2f}) for direct baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_model = Ridge(alpha=optimal_alpha_ridge, fit_intercept=True)
    ridge_model.fit(X_train_scaled, y_train)
    
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)
    y_test_pred_ridge = ridge_model.predict(X_test_scaled)
    
    # 4. Hyperparameter Tuning on Vta03 for Random Forest
    print("\n3. Tuning Random Forest Hyperparameters on Vta03 Validation Set...")
    # Define principled parameter grid balancing depth, regularization, and tree diversity
    param_grid = [
        {'n_estimators': 100, 'max_depth': 6,  'min_samples_leaf': 10, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 8,  'min_samples_leaf': 5,  'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5,  'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 2,  'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 2,  'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 8,  'min_samples_leaf': 5,  'max_features': 0.5},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5,  'max_features': 0.5},
        {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 2,  'max_features': 0.5},
        {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 5,  'max_features': 0.5},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 2,  'max_features': 0.5},
    ]
    
    rf_tuning_records = []
    best_rf_model = None
    best_rf_params = None
    best_rf_val_mae = float('inf')
    
    for idx, p in enumerate(param_grid, 1):
        rf = RandomForestRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_leaf=p['min_samples_leaf'],
            max_features=p['max_features'],
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        y_train_rf_cur = rf.predict(X_train)
        y_val_rf_cur = rf.predict(X_val)
        
        m_tr = compute_regression_metrics(y_train, y_train_rf_cur)
        m_v = compute_regression_metrics(y_val, y_val_rf_cur)
        
        rec = {
            'candidate_id': idx,
            'params': p,
            'train_mae': m_tr['mae_ms2'],
            'train_rmse': m_tr['rmse_ms2'],
            'train_r2': m_tr['r2'],
            'val_mae': m_v['mae_ms2'],
            'val_rmse': m_v['rmse_ms2'],
            'val_r2': m_v['r2'],
            'val_corr': m_v['pred_true_correlation'],
        }
        rf_tuning_records.append(rec)
        print(f"  Candidate {idx:2d} | params={p} -> Val MAE={m_v['mae_ms2']:.4f}, Val R2={m_v['r2']:+.4f}")
        
        if m_v['mae_ms2'] < best_rf_val_mae:
            best_rf_val_mae = m_v['mae_ms2']
            best_rf_params = p
            best_rf_model = rf
            
    print(f"\n  Optimal Random Forest Selected: {best_rf_params}")
    print(f"  Best Validation MAE on Vta03: {best_rf_val_mae:.4f} m/s² (Ridge was {compute_regression_metrics(y_val, y_val_pred_ridge)['mae_ms2']:.4f} m/s²)")
    
    # 5. Question A: Statistical Predictability Analysis
    print("\n4. Question A: Statistical Predictability Evaluation...")
    y_train_pred_rf = best_rf_model.predict(X_train)
    y_val_pred_rf = best_rf_model.predict(X_val)
    y_test_pred_rf = best_rf_model.predict(X_test)
    
    # Constant baseline
    y_train_const = np.full_like(y_train, r_bar_train)
    y_val_const = np.full_like(y_val, r_bar_train)
    y_test_const = np.full_like(y_test, r_bar_train)
    
    metrics_train = {
        'constant_mean': compute_regression_metrics(y_train, y_train_const),
        'ridge': compute_regression_metrics(y_train, y_train_pred_ridge),
        'random_forest': compute_regression_metrics(y_train, y_train_pred_rf),
    }
    metrics_val = {
        'constant_mean': compute_regression_metrics(y_val, y_val_const),
        'ridge': compute_regression_metrics(y_val, y_val_pred_ridge),
        'random_forest': compute_regression_metrics(y_val, y_val_pred_rf),
    }
    metrics_test = {
        'constant_mean': compute_regression_metrics(y_test, y_test_const),
        'ridge': compute_regression_metrics(y_test, y_test_pred_ridge),
        'random_forest': compute_regression_metrics(y_test, y_test_pred_rf),
    }
    
    print("\n--- Comparative Statistical Summary (Question A) ---")
    print(f"{'Partition':<25} | {'Model':<16} | {'MAE (m/s²)':<10} | {'RMSE (m/s²)':<11} | {'R²':<8} | {'Corr r':<8}")
    print("-" * 88)
    for part_name, m_dict in [('Train (Vta02)', metrics_train), 
                              ('Val/Select (Vta03)', metrics_val), 
                              ('Final Test (Vta04)', metrics_test)]:
        for m_name, m in m_dict.items():
            print(f"{part_name:<25} | {m_name:<16} | {m['mae_ms2']:<10.4f} | {m['rmse_ms2']:<11.4f} | {m['r2']:<8.4f} | {m['pred_true_correlation']:<8.4f}")
        print("-" * 88)
        
    # 6. Question B: Navigation Outage Simulation (Identical Reset-at-Window-Start Benchmark)
    print("\n5. Question B: Navigation Outage Drift Simulation...")
    ax_level_train = X_train[:, feat_names.index('ax_level_k')]
    ax_level_val = X_val[:, feat_names.index('ax_level_k')]
    ax_level_test = X_test[:, feat_names.index('ax_level_k')]
    
    trips_accel = {
        'Vta02': {
            'v_true': meta_train['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_train,
                'const_cal': ax_level_train - r_bar_train,
                'ridge_corrected': ax_level_train - y_train_pred_ridge,
                'rf_corrected': ax_level_train - y_train_pred_rf,
                'oracle_ref': meta_train['a_ref_targets'],
            }
        },
        'Vta03': {
            'v_true': meta_val['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_val,
                'const_cal': ax_level_val - r_bar_train,
                'ridge_corrected': ax_level_val - y_val_pred_ridge,
                'rf_corrected': ax_level_val - y_val_pred_rf,
                'oracle_ref': meta_val['a_ref_targets'],
            }
        },
        'Vta04': {
            'v_true': meta_test['v_vbox_targets'],
            'series': {
                'raw_baseline': ax_level_test,
                'const_cal': ax_level_test - r_bar_train,
                'ridge_corrected': ax_level_test - y_test_pred_ridge,
                'rf_corrected': ax_level_test - y_test_pred_rf,
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
        
        print(f"    Horizon | Raw Base | Const Cal | Ridge Corrected | RF Corrected | Oracle Floor")
        for h_key, h_data in nav_res.items():
            m_data = h_data['methods']
            p_raw = m_data['raw_baseline']['mean_pos_error_m']
            p_const = m_data['const_cal']['mean_pos_error_m']
            p_ridge = m_data['ridge_corrected']['mean_pos_error_m']
            p_rf = m_data['rf_corrected']['mean_pos_error_m']
            p_ref = m_data['oracle_ref']['mean_pos_error_m']
            print(f"    {h_key:<7} | {p_raw:6.2f}m   | {p_const:6.2f}m   | {p_ridge:6.2f}m         | {p_rf:6.2f}m       | {p_ref:5.2f}m")
            
    # 7. Feature Importances (MDI / Gini)
    print("\n6. Extracting Random Forest Feature Importances (MDI)...")
    importances = best_rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_rf_features = []
    print("  Top 10 Most Important Features in Random Forest:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        f_name = feat_names[idx]
        imp_val = float(importances[idx])
        top_rf_features.append({'rank': rank, 'feature': f_name, 'importance': imp_val})
        print(f"    {rank:2d}. {f_name:<28}: {imp_val * 100.0:5.2f}%")
        
    # 8. Diagnostic Visualizations
    print("\n7. Generating Diagnostic Figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Model Comparison Scatter on Final Test (Vta04)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Ridge
    corr_ridge = metrics_test['ridge']['pred_true_correlation']
    mae_ridge = metrics_test['ridge']['mae_ms2']
    axes[0].scatter(y_test, y_test_pred_ridge, color='royalblue', alpha=0.3, s=15, edgecolors='none')
    lims = [min(np.min(y_test), -5.0), max(np.max(y_test), 5.0)]
    axes[0].plot(lims, lims, 'k--', lw=1.5, label='Ideal 1:1')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel('Reference Residual r_a (m/s²)', fontweight='bold')
    axes[0].set_ylabel('Ridge Predicted Residual (m/s²)', fontweight='bold')
    axes[0].set_title(f"Stage C5.3-B1: Ridge on Vta04 (Test)\nMAE = {mae_ridge:.3f} m/s², R² = {metrics_test['ridge']['r2']:.3f}, r = {corr_ridge:.3f}", fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.5)
    
    # Panel 2: Random Forest
    corr_rf = metrics_test['random_forest']['pred_true_correlation']
    mae_rf = metrics_test['random_forest']['mae_ms2']
    axes[1].scatter(y_test, y_test_pred_rf, color='forestgreen', alpha=0.3, s=15, edgecolors='none')
    axes[1].plot(lims, lims, 'k--', lw=1.5, label='Ideal 1:1')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel('Reference Residual r_a (m/s²)', fontweight='bold')
    axes[1].set_ylabel('RF Predicted Residual (m/s²)', fontweight='bold')
    axes[1].set_title(f"Stage C5.3-B2: Random Forest on Vta04 (Test)\nMAE = {mae_rf:.3f} m/s², R² = {metrics_test['random_forest']['r2']:.3f}, r = {corr_rf:.3f}", fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_3b2_ridge_vs_rf_scatter.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    
    # Figure 2: Time Series Residual Tracking Comparison on Vta04
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    mask_plot = (t_test >= 40.0) & (t_test <= 100.0)
    
    axes[0].plot(t_test[mask_plot], y_test[mask_plot], 'black', lw=1.8, label='Reference Residual r_a(k)')
    axes[0].plot(t_test[mask_plot], y_test_pred_ridge[mask_plot], 'royalblue', lw=1.8, ls='--', label='Ridge Prediction')
    axes[0].plot(t_test[mask_plot], y_test_pred_rf[mask_plot], 'forestgreen', lw=2.0, label='Random Forest Prediction')
    axes[0].set_ylabel('Residual (m/s²)', fontweight='bold')
    axes[0].set_title('Stage C5.3-B2: Ridge vs. Random Forest Residual Tracking on Vta04 (Test Segment)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.5)
    
    err_ridge = (ax_level_test - y_test_pred_ridge) - meta_test['a_ref_targets']
    err_rf = (ax_level_test - y_test_pred_rf) - meta_test['a_ref_targets']
    axes[1].plot(t_test[mask_plot], err_ridge[mask_plot], 'royalblue', lw=1.5, alpha=0.7, label='Ridge Remaining Error')
    axes[1].plot(t_test[mask_plot], err_rf[mask_plot], 'crimson', lw=1.8, label='RF Remaining Error')
    axes[1].axhline(0.0, color='black', ls='--', lw=1.0)
    axes[1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[1].set_ylabel('Remaining Accel Error (m/s²)', fontweight='bold')
    axes[1].set_title('Remaining Acceleration Error: a_corrected - a_reference', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_3b2_timeseries_tracking.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    
    # Figure 3: Navigation Horizon Position Drift Comparison (Vta04 Test)
    fig, ax = plt.subplots(figsize=(9, 5))
    h_data_04 = nav_outage_results['Vta04']
    horizons_s = [h_data_04[k]['horizon_s'] for k in h_data_04]
    p_raw_04 = [h_data_04[k]['methods']['raw_baseline']['mean_pos_error_m'] for k in h_data_04]
    p_const_04 = [h_data_04[k]['methods']['const_cal']['mean_pos_error_m'] for k in h_data_04]
    p_ridge_04 = [h_data_04[k]['methods']['ridge_corrected']['mean_pos_error_m'] for k in h_data_04]
    p_rf_04 = [h_data_04[k]['methods']['rf_corrected']['mean_pos_error_m'] for k in h_data_04]
    p_ref_04 = [h_data_04[k]['methods']['oracle_ref']['mean_pos_error_m'] for k in h_data_04]
    
    ax.plot(horizons_s, p_raw_04, 'k--o', lw=1.8, label='Raw Baseline (No Correction)')
    ax.plot(horizons_s, p_const_04, 'gray', ls=':', marker='^', lw=1.8, label='Constant Cal Baseline')
    ax.plot(horizons_s, p_ridge_04, 'royalblue', marker='s', lw=2.0, label='Stage C5.3-B1 (Ridge Linear)')
    ax.plot(horizons_s, p_rf_04, 'forestgreen', marker='D', lw=2.5, label='Stage C5.3-B2 (Random Forest)')
    ax.plot(horizons_s, p_ref_04, 'darkgreen', ls='-.', marker='x', lw=1.5, label='Oracle Kinematic Floor')
    
    ax.set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Final Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.3-B2: Vta04 Navigation Drift Benchmark (Ridge vs. Random Forest)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_3b2_navigation_drift_benchmark.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    
    # Figure 4: Top 10 Feature Importances
    fig, ax = plt.subplots(figsize=(10, 5))
    top_names = [t['feature'] for t in top_rf_features][::-1]
    top_imps = [t['importance'] * 100.0 for t in top_rf_features][::-1]
    ax.barh(top_names, top_imps, color='forestgreen', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Mean Decrease in Impurity (Gini Importance, %)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.3-B2: Random Forest Top Feature Importances', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    fig4_path = FIG_DIR / "c5_3b2_feature_importances.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    
    # 9. Save Complete Results JSON
    b2_results = {
        'stage': 'C5.3-B2',
        'status': 'COMPLETE',
        'best_hyperparameters': best_rf_params,
        'baseline_constant_mean_train_ms2': r_bar_train,
        'statistical_metrics': {
            'train_vta02': metrics_train,
            'val_model_selection_vta03': metrics_val,
            'final_test_vta04': metrics_test,
        },
        'navigation_outage_results': nav_outage_results,
        'top_feature_importances': top_rf_features,
        'tuning_records': rf_tuning_records,
    }
    
    out_json = RES_DIR / "c5_3b2_rf_results.json"
    with open(out_json, 'w') as f:
        json.dump(b2_results, f, indent=2)
        
    print(f"\nSaved complete Random Forest results to {out_json}")
    print("===================================================================")
    print("Stage C5.3-B2 execution finished successfully.")
    print("===================================================================")


if __name__ == "__main__":
    main()
