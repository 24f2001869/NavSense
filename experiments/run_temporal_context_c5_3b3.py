"""
SIH26168 - Stage C5.3-B3: Causal Temporal-Context Experiment

Script: experiments/run_temporal_context_c5_3b3.py

PURPOSE:
    Execute a strictly controlled, single-variable experiment to test the hypothesis:
    "Does expanding causal temporal context allow the model to distinguish persistent 
     longitudinal vehicle acceleration from short-duration chassis/suspension pitch dynamics?"

STRICT EXPERIMENTAL CONTROLS (ONE-VARIABLE PROTOCOL):
    - Only variable altered: causal history length W in {5, 10, 15, 30, 50} samples (0.5s, 1.0s, 1.5s, 3.0s, 5.0s).
    - Fixed feature family: exactly 47 causal features (zero new features or extra statistics added for longer windows).
    - Fixed model: RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1).
    - Fixed target: r_a(k) = a_x^level(k) - a_ref(k) (SG9 reference).
    - Fixed partitions: Train on Vta02, validation monitor on Vta03, untouched final test on Vta04.
    - Fixed evaluation: rolling-window dead reckoning (horizons 5, 10, 20, 30, 60s, stride 2.5s, reset v_0).
    - Double-domain reporting: both per-window full valid domain and common intersection domain (k >= 49).
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.verify_causal_pipeline_c5_3b0 import extract_causal_features, DT
from experiments.run_ridge_baseline_c5_3b1 import compute_regression_metrics, simulate_outage_navigation

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"

# Window configurations to test
WINDOWS = [
    {'name': '0.5s (W=5)',  'samples': 5,  'duration_s': 0.5},
    {'name': '1.0s (W=10)', 'samples': 10, 'duration_s': 1.0},
    {'name': '1.5s (W=15)', 'samples': 15, 'duration_s': 1.5},  # B2 Baseline
    {'name': '3.0s (W=30)', 'samples': 30, 'duration_s': 3.0},
    {'name': '5.0s (W=50)', 'samples': 50, 'duration_s': 5.0},
]

FROZEN_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_leaf': 5,
    'max_features': 0.5,
    'random_state': 42,
    'n_jobs': -1
}


def evaluate_regimes(y_true: np.ndarray, y_pred: np.ndarray, a_ref: np.ndarray, v_vbox: np.ndarray) -> dict:
    """
    Decomposes error across kinematic regimes on test set.
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    
    regimes = {
        'severe_braking': (a_ref <= -1.5),
        'moderate_braking': (a_ref > -1.5) & (a_ref <= -0.5),
        'steady_cruising': (np.abs(a_ref) < 0.5) & (v_vbox > 5.0),
        'moderate_accel': (a_ref > 0.5) & (a_ref <= 1.5),
        'severe_accel': (a_ref >= 1.5),
    }
    
    out = {}
    for r_name, mask in regimes.items():
        n_m = int(np.sum(mask))
        if n_m > 0:
            out[r_name] = {
                'count': n_m,
                'pct_samples': float(n_m / len(y_true) * 100.0),
                'mae_ms2': float(np.mean(abs_err[mask])),
                'rmse_ms2': float(np.sqrt(np.mean(err[mask]**2))),
                'bias_ms2': float(np.mean(err[mask])),
            }
        else:
            out[r_name] = {'count': 0, 'pct_samples': 0.0, 'mae_ms2': 0.0, 'rmse_ms2': 0.0, 'bias_ms2': 0.0}
    return out


def compute_window_acceleration_biases(a_est_series: np.ndarray, a_ref_series: np.ndarray, 
                                      horizons: list = [5.0, 10.0, 20.0, 30.0, 60.0], 
                                      stride_s: float = 2.5) -> dict:
    """
    Computes mean absolute window acceleration bias |e_a_bar| across horizons.
    """
    n = len(a_est_series)
    stride_k = max(1, int(round(stride_s / DT)))
    bias_by_horizon = {}
    
    for h in horizons:
        w_k = int(round(h / DT))
        if n < w_k + 2:
            continue
        biases = []
        for k_start in range(0, n - w_k, stride_k):
            k_end = k_start + w_k
            a_est_win = a_est_series[k_start:k_end + 1]
            a_ref_win = a_ref_series[k_start:k_end + 1]
            e_a_win = a_est_win - a_ref_win
            biases.append(abs(float(np.mean(e_a_win))))
        bias_by_horizon[f"{int(h)}s"] = {
            'mean_abs_bias_ms2': float(np.mean(biases)),
            'median_abs_bias_ms2': float(np.median(biases)),
            'p90_abs_bias_ms2': float(np.percentile(biases, 90)),
        }
    return bias_by_horizon


def main():
    print("=" * 80)
    print("Stage C5.3-B3: Causal Temporal-Context Experiment")
    print("Controlled Single-Variable Sweep: W in {5, 10, 15, 30, 50} (0.5s to 5.0s)")
    print("=" * 80)
    
    # 1. Ingest clean label datasets
    csv_02 = PROC_DIR / "c5_3_labels_vta02.csv"
    csv_03 = PROC_DIR / "c5_3_labels_vta03.csv"
    csv_04 = PROC_DIR / "c5_3_labels_vta04.csv"
    
    df_02 = pd.read_csv(csv_02)
    df_03 = pd.read_csv(csv_03)
    df_04 = pd.read_csv(csv_04)
    
    print(f"Ingested Datasets: Vta02 ({len(df_02)} samples), Vta03 ({len(df_03)} samples), Vta04 ({len(df_04)} samples)")
    
    # Common intersection domain index (k >= 49, corresponding to W=50 start)
    max_w = max(w['samples'] for w in WINDOWS)  # 50
    k_common_start = max_w - 1  # 49 (at 4.9s)
    
    # Storage for all sweep results
    sweep_results = []
    
    # Also collect raw and oracle benchmarks on common domain for Vta04
    t_test_raw = df_04['time_s'].to_numpy()[k_common_start:]
    v_test_raw = df_04['LABEL_vbox_speed_ms'].to_numpy()[k_common_start:]
    a_ref_test_raw = df_04['LABEL_vbox_ref_accel_ms2'].to_numpy()[k_common_start:]
    ax_level_test_raw = df_04['ax_level'].to_numpy()[k_common_start:]
    
    nav_benchmarks_common = simulate_outage_navigation(
        v_test_raw,
        {
            'raw_baseline': ax_level_test_raw,
            'oracle_ref': a_ref_test_raw,
        },
        horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
        stride_s=2.5
    )
    
    print("\nStarting Model Training & Evaluation Loop across 5 Temporal Contexts...")
    print("-" * 80)
    
    for w_cfg in WINDOWS:
        w_name = w_cfg['name']
        w_size = w_cfg['samples']
        w_dur = w_cfg['duration_s']
        t0 = time.time()
        
        print(f"\n>>> Running Condition: {w_name} (Trailing Window = {w_dur:.1f} s, W = {w_size} samples) <<<")
        
        # 2. Extract strictly causal features for window size W
        X_train, y_train, t_train, feat_names, meta_train = extract_causal_features(df_02, window_size=w_size)
        X_val, y_val, t_val, _, meta_val = extract_causal_features(df_03, window_size=w_size)
        X_test, y_test, t_test, _, meta_test = extract_causal_features(df_04, window_size=w_size)
        
        assert X_train.shape[1] == 47, f"Feature count mismatch: {X_train.shape[1]} != 47"
        assert not np.isnan(X_train).any(), "NaN detected in training feature matrix"
        assert not np.isnan(X_test).any(), "NaN detected in test feature matrix"
        
        print(f"  Feature Matrix extracted: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape} (D=47 exactly)")
        
        # 3. Fit Frozen Random Forest on Vta02
        rf = RandomForestRegressor(**FROZEN_RF_PARAMS)
        rf.fit(X_train, y_train)
        fit_time = time.time() - t0
        print(f"  RF fitted in {fit_time:.2f} s on Vta02")
        
        # 4. Predict residuals
        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)
        y_test_pred = rf.predict(X_test)
        
        # 5. Compute standard regression metrics (Per-window full valid domain)
        m_train = compute_regression_metrics(y_train, y_train_pred)
        m_val = compute_regression_metrics(y_val, y_val_pred)
        m_test_full = compute_regression_metrics(y_test, y_test_pred)
        
        # Corrected forward acceleration series
        ax_level_test = X_test[:, feat_names.index('ax_level_k')]
        a_corr_test_full = ax_level_test - y_test_pred
        a_ref_test_full = meta_test['a_ref_targets']
        v_test_full = meta_test['v_vbox_targets']
        
        # Navigation simulation on full domain
        nav_res_full = simulate_outage_navigation(
            v_test_full,
            {'rf_corrected': a_corr_test_full},
            horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
            stride_s=2.5
        )
        
        # 6. Compute metrics on Common Intersection Domain (k >= 49, t >= 4.9s)
        # Offset from start of current window: offset = (max_w - 1) - (w_size - 1) = max_w - w_size
        offset = max_w - w_size
        y_test_com = y_test[offset:]
        y_test_pred_com = y_test_pred[offset:]
        a_corr_test_com = a_corr_test_full[offset:]
        a_ref_test_com = a_ref_test_full[offset:]
        v_test_com = v_test_full[offset:]
        
        assert len(y_test_com) == len(df_04) - max_w + 1, f"Common length mismatch: {len(y_test_com)}"
        
        m_test_com = compute_regression_metrics(y_test_com, y_test_pred_com)
        
        # Navigation simulation on common domain
        nav_res_com = simulate_outage_navigation(
            v_test_com,
            {'rf_corrected': a_corr_test_com},
            horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
            stride_s=2.5
        )
        
        # 7. Regime breakdown on common domain
        regimes_com = evaluate_regimes(y_test_com, y_test_pred_com, a_ref_test_com, v_test_com)
        
        # 8. Window acceleration biases on common domain
        window_biases_com = compute_window_acceleration_biases(a_corr_test_com, a_ref_test_com,
                                                              horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
                                                              stride_s=2.5)
        
        # 9. Extract top feature importances
        mdi = rf.feature_importances_
        sorted_idx = np.argsort(mdi)[::-1]
        top_features = [{'feature': feat_names[i], 'importance': float(mdi[i])} for i in sorted_idx[:8]]
        
        # Print summary for this condition
        print(f"  [Train Vta02] MAE: {m_train['mae_ms2']:.4f}, RMSE: {m_train['rmse_ms2']:.4f}, R2: {m_train['r2']:+.4f}")
        print(f"  [Val Vta03]   MAE: {m_val['mae_ms2']:.4f}, RMSE: {m_val['rmse_ms2']:.4f}, R2: {m_val['r2']:+.4f}")
        print(f"  [Test Vta04 Common] MAE: {m_test_com['mae_ms2']:.4f}, RMSE: {m_test_com['rmse_ms2']:.4f}, R2: {m_test_com['r2']:+.4f}, Corr: {m_test_com['pred_true_correlation']:.4f}")
        print(f"  [Nav 30s Drift Common] Mean: {nav_res_com['30s']['methods']['rf_corrected']['mean_pos_error_m']:.2f} m, Median: {nav_res_com['30s']['methods']['rf_corrected']['median_pos_error_m']:.2f} m")
        print(f"  [Nav 60s Drift Common] Mean: {nav_res_com['60s']['methods']['rf_corrected']['mean_pos_error_m']:.2f} m")
        print(f"  [Cruising MAE] {regimes_com['steady_cruising']['mae_ms2']:.4f} m/s² | [Severe Braking Bias] {regimes_com['severe_braking']['bias_ms2']:.4f} m/s²")
        
        rec = {
            'window_config': w_cfg,
            'fit_time_s': fit_time,
            'train_metrics': m_train,
            'val_metrics': m_val,
            'test_metrics_full_domain': m_test_full,
            'test_metrics_common_domain': m_test_com,
            'navigation_full_domain': nav_res_full,
            'navigation_common_domain': nav_res_com,
            'regimes_common_domain': regimes_com,
            'window_biases_common_domain': window_biases_com,
            'top_features': top_features,
        }
        sweep_results.append(rec)
        
    print("\n" + "=" * 80)
    print("STAGE C5.3-B3 TEMPORAL CONTEXT SWEEP COMPLETE")
    print("=" * 80)
    
    # -----------------------------------------------------------------------
    # Comparative Tables & Outcome Diagnostics
    # -----------------------------------------------------------------------
    print("\n--- Summary Table 1: Regression Metrics on Untouched Vta04 (Common Domain k>=49) ---")
    print(f"{'Condition':<14} | {'Window (s)':<10} | {'MAE (m/s²)':<10} | {'RMSE (m/s²)':<11} | {'R²':<8} | {'Corr r':<8} | {'30s Bias (m/s²)':<15}")
    print("-" * 88)
    for r in sweep_results:
        w_c = r['window_config']
        m = r['test_metrics_common_domain']
        b30 = r['window_biases_common_domain']['30s']['mean_abs_bias_ms2']
        print(f"{w_c['name']:<14} | {w_c['duration_s']:<10.1f} | {m['mae_ms2']:<10.4f} | {m['rmse_ms2']:<11.4f} | {m['r2']:<8.4f} | {m['pred_true_correlation']:<8.4f} | {b30:<15.4f}")
    print("-" * 88)
    
    print("\n--- Summary Table 2: Dead-Reckoning Position Drift Across Horizons (Untouched Vta04 Common Domain) ---")
    print(f"{'Condition':<14} | {'5 s Drift':<10} | {'10 s Drift':<11} | {'20 s Drift':<11} | {'30 s Drift':<11} | {'60 s Drift':<11} | {'30s Drift %':<11}")
    print("-" * 88)
    
    raw_p = {h: nav_benchmarks_common[h]['methods']['raw_baseline']['mean_pos_error_m'] for h in ['5s', '10s', '20s', '30s', '60s']}
    raw_d30_pct = nav_benchmarks_common['30s']['methods']['raw_baseline']['mean_drift_pct']
    print(f"{'Raw Baseline':<14} | {raw_p['5s']:<10.2f} | {raw_p['10s']:<11.2f} | {raw_p['20s']:<11.2f} | {raw_p['30s']:<11.2f} | {raw_p['60s']:<11.2f} | {raw_d30_pct:<11.2f}%")
    print("-" * 88)
    
    for r in sweep_results:
        w_c = r['window_config']
        nav = r['navigation_common_domain']
        p5 = nav['5s']['methods']['rf_corrected']['mean_pos_error_m']
        p10 = nav['10s']['methods']['rf_corrected']['mean_pos_error_m']
        p20 = nav['20s']['methods']['rf_corrected']['mean_pos_error_m']
        p30 = nav['30s']['methods']['rf_corrected']['mean_pos_error_m']
        p60 = nav['60s']['methods']['rf_corrected']['mean_pos_error_m']
        dp30 = nav['30s']['methods']['rf_corrected']['mean_drift_pct']
        print(f"{w_c['name']:<14} | {p5:<10.2f} | {p10:<11.2f} | {p20:<11.2f} | {p30:<11.2f} | {p60:<11.2f} | {dp30:<11.2f}%")
    print("-" * 88)
    
    oracle_p = {h: nav_benchmarks_common[h]['methods']['oracle_ref']['mean_pos_error_m'] for h in ['5s', '10s', '20s', '30s', '60s']}
    oracle_d30_pct = nav_benchmarks_common['30s']['methods']['oracle_ref']['mean_drift_pct']
    print(f"{'Oracle Floor':<14} | {oracle_p['5s']:<10.2f} | {oracle_p['10s']:<11.2f} | {oracle_p['20s']:<11.2f} | {oracle_p['30s']:<11.2f} | {oracle_p['60s']:<11.2f} | {oracle_d30_pct:<11.2f}%")
    print("-" * 88)
    
    print("\n--- Summary Table 3: Dynamic Regimes Breakdown (Common Domain Vta04) ---")
    print(f"{'Condition':<14} | {'Cruising MAE':<13} | {'Cruising Bias':<14} | {'Sev Brake MAE':<14} | {'Sev Brake Bias':<15} | {'Sev Accel MAE':<14}")
    print("-" * 88)
    for r in sweep_results:
        w_c = r['window_config']
        reg = r['regimes_common_domain']
        c_mae = reg['steady_cruising']['mae_ms2']
        c_bias = reg['steady_cruising']['bias_ms2']
        sb_mae = reg['severe_braking']['mae_ms2']
        sb_bias = reg['severe_braking']['bias_ms2']
        sa_mae = reg['severe_accel']['mae_ms2']
        print(f"{w_c['name']:<14} | {c_mae:<13.4f} | {c_bias:<14.4f} | {sb_mae:<14.4f} | {sb_bias:<15.4f} | {sa_mae:<14.4f}")
    print("-" * 88)
    
    # -----------------------------------------------------------------------
    # Hypothesis Outcome Assessment
    # -----------------------------------------------------------------------
    b2_baseline_rec = sweep_results[2]  # W=15 (1.5s)
    b2_p30 = b2_baseline_rec['navigation_common_domain']['30s']['methods']['rf_corrected']['mean_pos_error_m']
    best_p30 = min(r['navigation_common_domain']['30s']['methods']['rf_corrected']['mean_pos_error_m'] for r in sweep_results)
    best_rec = min(sweep_results, key=lambda r: r['navigation_common_domain']['30s']['methods']['rf_corrected']['mean_pos_error_m'])
    worst_p30 = max(r['navigation_common_domain']['30s']['methods']['rf_corrected']['mean_pos_error_m'] for r in sweep_results)
    
    p30_spread_pct = abs(worst_p30 - best_p30) / b2_p30 * 100.0
    p30_gain_over_b2 = (b2_p30 - best_p30) / b2_p30 * 100.0
    
    print("\n--- Diagnostic Outcome Assessment ---")
    print(f"B2 Baseline (1.5s) 30s Drift: {b2_p30:.2f} m")
    print(f"Best Window 30s Drift:        {best_p30:.2f} m ({best_rec['window_config']['name']}) -> Delta = {p30_gain_over_b2:+.2f}%")
    print(f"Spread across all 5 Windows:  {abs(worst_p30 - best_p30):.2f} m ({p30_spread_pct:.2f}% of baseline)")
    
    # Check criteria for Outcome A vs B vs C
    if p30_gain_over_b2 > 25.0 and best_p30 < 50.0:
        outcome = 'OUTCOME_A'
        outcome_desc = "Outcome A: Longer causal history substantially improves dead-reckoning navigation drift. Temporal context was a major bottleneck."
    elif p30_spread_pct < 15.0 or (p30_gain_over_b2 < 10.0 and best_rec['window_config']['samples'] in [30, 50]):
        outcome = 'OUTCOME_B'
        outcome_desc = "Outcome B: Longer causal history barely helps or plateaus. Temporal context is NOT the primary bottleneck; physical representation, frame dynamics, or sensor observability govern remaining drift."
    else:
        outcome = 'OUTCOME_C_OR_INTERMEDIATE'
        outcome_desc = "Intermediate / Mixed Outcome: Moderate sensitivity to window length observed."
        
    print(f"\nOfficial Verdict: {outcome}")
    print(f"Detail: {outcome_desc}")
    
    # -----------------------------------------------------------------------
    # Generate Publication-Quality Figures
    # -----------------------------------------------------------------------
    print("\nGenerating Diagnostic Figures...")
    durations = [r['window_config']['duration_s'] for r in sweep_results]
    
    # Figure 1: Statistical Metrics & Window Bias vs. Window Duration
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Subplot 1: MAE & RMSE
    maes = [r['test_metrics_common_domain']['mae_ms2'] for r in sweep_results]
    rmses = [r['test_metrics_common_domain']['rmse_ms2'] for r in sweep_results]
    axes[0].plot(durations, maes, 'o-', color='#1f77b4', lw=2, label='MAE (m/s²)')
    axes[0].plot(durations, rmses, 's--', color='#d62728', lw=2, label='RMSE (m/s²)')
    axes[0].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[0].set_xlabel('Causal Window Duration (s)')
    axes[0].set_ylabel('Residual Error (m/s²)')
    axes[0].set_title('Test Residual Error vs. Temporal Context')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: R² and Correlation
    r2s = [r['test_metrics_common_domain']['r2'] for r in sweep_results]
    corrs = [r['test_metrics_common_domain']['pred_true_correlation'] for r in sweep_results]
    axes[1].plot(durations, r2s, 'o-', color='#2ca02c', lw=2, label='R² Score')
    axes[1].plot(durations, corrs, '^--', color='#9467bd', lw=2, label='Correlation (r)')
    axes[1].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[1].set_xlabel('Causal Window Duration (s)')
    axes[1].set_ylabel('Score / Correlation')
    axes[1].set_title('Generalization Metrics vs. Temporal Context')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Subplot 3: 30s Window Acceleration Bias
    biases_30s = [r['window_biases_common_domain']['30s']['mean_abs_bias_ms2'] for r in sweep_results]
    axes[2].plot(durations, biases_30s, 'D-', color='#ff7f0e', lw=2.5, label='Mean |e_a| @ 30s')
    axes[2].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[2].set_xlabel('Causal Window Duration (s)')
    axes[2].set_ylabel('Mean Absolute Window Bias (m/s²)')
    axes[2].set_title('30s Net Acceleration Bias vs. Temporal Context')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_3b3_window_sweep_metrics.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig1_path.name}")
    
    # Figure 2: Navigation Drift Across Horizons vs. Window Duration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Position Drift vs Window Duration for 10s, 30s, 60s
    p10_list = [r['navigation_common_domain']['10s']['methods']['rf_corrected']['mean_pos_error_m'] for r in sweep_results]
    p30_list = [r['navigation_common_domain']['30s']['methods']['rf_corrected']['mean_pos_error_m'] for r in sweep_results]
    p60_list = [r['navigation_common_domain']['60s']['methods']['rf_corrected']['mean_pos_error_m'] for r in sweep_results]
    
    axes[0].plot(durations, p10_list, 'o-', color='#1f77b4', lw=2, label='10s Horizon')
    axes[0].plot(durations, p30_list, 's-', color='#ff7f0e', lw=2.5, label='30s Horizon')
    axes[0].plot(durations, p60_list, '^-', color='#d62728', lw=2, label='60s Horizon')
    axes[0].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[0].set_xlabel('Causal Window Duration (s)')
    axes[0].set_ylabel('Mean Position Drift (m)')
    axes[0].set_title('Position Drift vs. Causal Temporal History')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Subplot 2: Full Horizon Profile across all 5 Windows
    horizons_num = [5, 10, 20, 30, 60]
    colors = ['#8c564b', '#e377c2', '#1f77b4', '#2ca02c', '#9467bd']
    for idx, r in enumerate(sweep_results):
        w_c = r['window_config']
        nav = r['navigation_common_domain']
        p_curve = [nav[f"{h}s"]['methods']['rf_corrected']['mean_pos_error_m'] for h in horizons_num]
        axes[1].plot(horizons_num, p_curve, 'o-', color=colors[idx], lw=2, label=f"{w_c['name']}")
    
    # Add raw and oracle reference
    raw_curve = [raw_p[f"{h}s"] for h in horizons_num]
    oracle_curve = [oracle_p[f"{h}s"] for h in horizons_num]
    axes[1].plot(horizons_num, raw_curve, 'k--', lw=1.5, alpha=0.7, label='Raw Baseline')
    axes[1].plot(horizons_num, oracle_curve, 'g:', lw=1.5, alpha=0.7, label='Oracle Floor')
    
    axes[1].set_xlabel('Outage Horizon (s)')
    axes[1].set_ylabel('Mean Position Drift (m)')
    axes[1].set_title('Outage Horizon Progression across Window Lengths')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_3b3_window_sweep_drift.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig2_path.name}")
    
    # Figure 3: Dynamic Regimes Comparison (Cruising vs. Braking vs. Acceleration)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    c_maes = [r['regimes_common_domain']['steady_cruising']['mae_ms2'] for r in sweep_results]
    sb_maes = [r['regimes_common_domain']['severe_braking']['mae_ms2'] for r in sweep_results]
    sa_maes = [r['regimes_common_domain']['severe_accel']['mae_ms2'] for r in sweep_results]
    
    axes[0].plot(durations, c_maes, 'o-', color='#2ca02c', lw=2, label='Steady Cruising MAE (72.6% of trip)')
    axes[0].plot(durations, sb_maes, 's--', color='#d62728', lw=2, label='Severe Braking MAE (4.1% of trip)')
    axes[0].plot(durations, sa_maes, '^-.', color='#ff7f0e', lw=2, label='Severe Accel MAE (3.2% of trip)')
    axes[0].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[0].set_xlabel('Causal Window Duration (s)')
    axes[0].set_ylabel('Regime MAE (m/s²)')
    axes[0].set_title('Regime Residual MAE vs. Temporal Context')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    sb_biases = [r['regimes_common_domain']['severe_braking']['bias_ms2'] for r in sweep_results]
    sa_biases = [r['regimes_common_domain']['severe_accel']['bias_ms2'] for r in sweep_results]
    c_biases = [r['regimes_common_domain']['steady_cruising']['bias_ms2'] for r in sweep_results]
    
    axes[1].plot(durations, sb_biases, 's--', color='#d62728', lw=2, label='Severe Braking Bias')
    axes[1].plot(durations, sa_biases, '^-.', color='#ff7f0e', lw=2, label='Severe Accel Bias')
    axes[1].plot(durations, c_biases, 'o-', color='#2ca02c', lw=2, label='Steady Cruising Bias')
    axes[1].axhline(0.0, color='k', linestyle='-', alpha=0.5)
    axes[1].axvline(1.5, color='gray', linestyle=':', label='B2 Baseline (1.5s)')
    axes[1].set_xlabel('Causal Window Duration (s)')
    axes[1].set_ylabel('Signed Bias (m/s²)')
    axes[1].set_title('Transient Under-Prediction Bias vs. Temporal Context')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_3b3_regime_transient_breakdown.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig3_path.name}")
    
    # -----------------------------------------------------------------------
    # Export Structured JSON Deliverables
    # -----------------------------------------------------------------------
    out_json = {
        'metadata': {
            'stage': 'C5.3-B3',
            'description': 'Single-variable controlled causal temporal context experiment across window sizes W in {5, 10, 15, 30, 50}',
            'timestamp': '2026-09-05',
            'fixed_features': 47,
            'frozen_model': 'RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, max_features=0.5, random_state=42)',
            'sampling_rate_hz': 10.0,
            'dt_s': 0.1,
            'train_trip': 'Vta02',
            'val_trip': 'Vta03',
            'test_trip': 'Vta04',
            'common_intersection_domain_start_idx': k_common_start,
            'common_intersection_domain_time_s': k_common_start * DT,
            'common_samples_count': len(y_test_com),
        },
        'benchmarks_common_domain': {
            'raw_baseline': {h: nav_benchmarks_common[h]['methods']['raw_baseline'] for h in ['5s', '10s', '20s', '30s', '60s']},
            'oracle_floor': {h: nav_benchmarks_common[h]['methods']['oracle_ref'] for h in ['5s', '10s', '20s', '30s', '60s']},
        },
        'sweep_results': sweep_results,
        'diagnostic_assessment': {
            'outcome': outcome,
            'outcome_description': outcome_desc,
            'b2_baseline_30s_drift_m': b2_p30,
            'best_window_30s_drift_m': best_p30,
            'best_window_name': best_rec['window_config']['name'],
            'gain_over_b2_pct': p30_gain_over_b2,
            'spread_across_all_windows_m': abs(worst_p30 - best_p30),
            'spread_pct_of_baseline': p30_spread_pct,
        }
    }
    
    json_path = RES_DIR / "c5_3b3_temporal_context.json"
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\nStructured results exported to: {json_path}")


if __name__ == '__main__':
    main()
