"""
SIH26168 - Stage C5.3-B0: Causal Feature Construction, Timing Alignment & Strict Leakage Verification

Script: experiments/verify_causal_pipeline_c5_3b0.py

PURPOSE:
    Establish and verify the strictly causal feature extraction pipeline and
    data integrity harness before any model training in Stage C5.3-B1.

STRICT NON-TRAINING CONSTRAINTS:
    - Zero AI / Machine Learning (No Ridge, no RF, no GBDT, no MLP)
    - Zero ESKF / Kalman Filtering
    - Zero Non-Holonomic Constraints (NHC)
    - Pure pipeline verification, timestamp alignment, and machine-checkable leakage tests.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"

DT = 0.1  # 10 Hz sampling period
DEFAULT_WINDOW = 15  # 1.5 seconds causal trailing window


# ===========================================================================
# 1. Three-Tier Signal Categorization & Feature Extraction
# ===========================================================================

FEATURE_CATEGORIES = {
    'Category_A_Direct_Measured': [
        'ax_phone_k', 'ay_phone_k', 'az_phone_k',
        'gx_phone_k', 'gy_phone_k', 'gz_phone_k'
    ],
    'Category_B_Causal_Derived_Instantaneous': [
        'ax_level_k', 'accel_mag_k', 'gyro_norm_k',
        'jerk_ax_level_k', 'pitch_accel_gy_k'
    ],
    'Category_B_Causal_Derived_Windowed': [
        # Generated across primary streams over W_k: {mean, std, range, rms, slope, diff_var}
    ],
    'Category_C_Offline_Only_Supervision_FORBIDDEN': [
        'LABEL_vbox_speed_ms',
        'LABEL_vbox_ref_accel_ms2',
        'LABEL_target_residual_ms2',
        'LABEL_is_stopped',
        'veh_accel_long_ms2',
        'vbox_speed',
        'target_residual',
        'future_samples'
    ]
}


def extract_causal_features(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW) -> tuple:
    """
    Extracts strictly causal features over trailing temporal windows.
    For target step k (time t_k), the feature window spans [k - window_size + 1 : k + 1].
    ZERO future samples (j > k) are accessible.
    ZERO Category C supervision columns are included in feature matrix X.
    
    Returns:
        X: np.ndarray of shape (N - window_size + 1, n_features)
        y: np.ndarray of shape (N - window_size + 1,) containing target residual r(k)
        t_targets: np.ndarray of shape (N - window_size + 1,) containing t_k
        feature_names: list of str
        metadata: dict with auxiliary reference arrays for evaluation ONLY
    """
    n_total = len(df)
    n_samples = n_total - window_size + 1
    if n_samples <= 0:
        raise ValueError(f"Dataset length {n_total} smaller than window size {window_size}")

    # Ensure Category C columns are stripped from input data
    input_cols = [c for c in df.columns if not c.startswith('LABEL_') and c != 'time_s']
    
    # Pre-extract input arrays
    time_arr = df['time_s'].to_numpy()
    ax_p = df['ax_phone'].to_numpy()
    ay_p = df['ay_phone'].to_numpy()
    az_p = df['az_phone'].to_numpy()
    gx_p = df['gx_phone'].to_numpy()
    gy_p = df['gy_phone'].to_numpy()
    gz_p = df['gz_phone'].to_numpy()
    ax_l = df['ax_level'].to_numpy()
    
    # Supervision target arrays (strictly segregated)
    y_full = df['LABEL_target_residual_ms2'].to_numpy()
    v_vbox_full = df['LABEL_vbox_speed_ms'].to_numpy()
    a_ref_full = df['LABEL_vbox_ref_accel_ms2'].to_numpy()
    is_stop_full = df['LABEL_is_stopped'].to_numpy()
    
    # Derived instantaneous channels
    accel_mag = np.sqrt(ax_p**2 + ay_p**2 + az_p**2)
    gyro_norm = np.sqrt(gx_p**2 + gy_p**2 + gz_p**2)
    
    # Backward difference (causal jerk & pitch acceleration at step k)
    jerk_ax_level = np.zeros(n_total)
    pitch_accel_gy = np.zeros(n_total)
    jerk_ax_level[1:] = (ax_l[1:] - ax_l[:-1]) / DT
    pitch_accel_gy[1:] = (gy_p[1:] - gy_p[:-1]) / DT
    
    # Window statistics streams
    streams = {
        'ax_level': ax_l,
        'accel_mag': accel_mag,
        'gyro_norm': gyro_norm,
        'gy_phone': gy_p,
        'ax_phone': ax_p,
        'az_phone': az_p,
    }
    
    # Build feature names
    feature_names = []
    # 1. Category A: Instantaneous raw
    for col in ['ax_phone_k', 'ay_phone_k', 'az_phone_k', 'gx_phone_k', 'gy_phone_k', 'gz_phone_k']:
        feature_names.append(col)
    # 2. Category B: Instantaneous derived
    for col in ['ax_level_k', 'accel_mag_k', 'gyro_norm_k', 'jerk_ax_level_k', 'pitch_accel_gy_k']:
        feature_names.append(col)
    # 3. Category B: Windowed stats across streams
    for stream_name in streams:
        for stat in ['mean', 'std', 'range', 'rms', 'slope', 'diff_var']:
            feature_names.append(f"{stream_name}_{stat}")
            
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)
    t_targets = np.zeros(n_samples, dtype=np.float64)
    v_vbox_targets = np.zeros(n_samples, dtype=np.float64)
    a_ref_targets = np.zeros(n_samples, dtype=np.float64)
    is_stop_targets = np.zeros(n_samples, dtype=np.int32)
    
    # Normalized window time for linear slope computation
    w_indices = np.arange(window_size)
    w_time = w_indices * DT
    w_time_centered = w_time - np.mean(w_time)
    denom_slope = np.sum(w_time_centered**2)
    
    for row_idx in range(n_samples):
        k = row_idx + window_size - 1  # Target step k (current sample)
        w_slice = slice(row_idx, k + 1) # Samples [row_idx : k] inclusive (length window_size)
        
        feat_vals = []
        # Category A: Raw instantaneous at step k
        feat_vals.extend([ax_p[k], ay_p[k], az_p[k], gx_p[k], gy_p[k], gz_p[k]])
        # Category B: Derived instantaneous at step k
        feat_vals.extend([ax_l[k], accel_mag[k], gyro_norm[k], jerk_ax_level[k], pitch_accel_gy[k]])
        
        # Category B: Windowed statistics over W_k
        for stream_name, stream_arr in streams.items():
            w_data = stream_arr[w_slice]
            w_mean = np.mean(w_data)
            w_std = np.std(w_data)
            w_range = np.max(w_data) - np.min(w_data)
            w_rms = np.sqrt(np.mean(w_data**2))
            w_slope = np.sum(w_time_centered * (w_data - w_mean)) / denom_slope
            w_diff = np.diff(w_data)
            w_diff_var = np.var(w_diff) if len(w_diff) > 1 else 0.0
            
            feat_vals.extend([w_mean, w_std, w_range, w_rms, w_slope, w_diff_var])
            
        X[row_idx, :] = feat_vals
        y[row_idx] = y_full[k]
        t_targets[row_idx] = time_arr[k]
        v_vbox_targets[row_idx] = v_vbox_full[k]
        a_ref_targets[row_idx] = a_ref_full[k]
        is_stop_targets[row_idx] = is_stop_full[k]
        
    aux_metadata = {
        't_targets': t_targets,
        'v_vbox_targets': v_vbox_targets,
        'a_ref_targets': a_ref_targets,
        'is_stop_targets': is_stop_targets,
        'window_size': window_size,
        'window_duration_s': window_size * DT,
        'feature_names': feature_names,
    }
    
    return X, y, t_targets, feature_names, aux_metadata


# ===========================================================================
# 2. Timing Alignment & Cross-Correlation Lag Audit
# ===========================================================================

def audit_timestamp_lag(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW, max_lag_samples: int = 5) -> dict:
    """
    Computes cross-correlation between instantaneous feature a_x^level[k]
    and the target residual r[k + delta] for delta in [-max_lag, +max_lag].
    Verifies that the correlation peak occurs at delta = 0 (zero accidental shift).
    """
    X, y, t_targets, feat_names, _ = extract_causal_features(df, window_size=window_size)
    ax_level_idx = feat_names.index('ax_level_k')
    ax_feat = X[:, ax_level_idx]
    
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corrs = []
    
    n = len(y)
    for lag in lags:
        if lag < 0:
            # Feature leads target: compare ax_feat[:n+lag] with y[-lag:]
            r = np.corrcoef(ax_feat[:n + lag], y[-lag:])[0, 1]
        elif lag > 0:
            # Target leads feature: compare ax_feat[lag:] with y[:n-lag]
            r = np.corrcoef(ax_feat[lag:], y[:n - lag])[0, 1]
        else:
            r = np.corrcoef(ax_feat, y)[0, 1]
        corrs.append(float(r))
        
    corrs = np.array(corrs)
    best_idx = np.argmax(np.abs(corrs))
    best_lag = int(lags[best_idx])
    
    return {
        'lags_samples': lags.tolist(),
        'lags_seconds': (lags * DT).tolist(),
        'correlations': corrs.tolist(),
        'peak_lag_samples': best_lag,
        'peak_lag_seconds': float(best_lag * DT),
        'peak_abs_correlation': float(np.abs(corrs[best_idx])),
        'alignment_verified': (best_lag == 0),
    }


# ===========================================================================
# 3. Machine-Checkable Leakage Unit Test Suite
# ===========================================================================

def leakage_audit_detector(feature_names: list, feature_matrix: np.ndarray, reference_vector: np.ndarray) -> bool:
    """
    Strict audit detector.
    Raises ValueError if:
      1. Any Category C forbidden string appears in feature names.
      2. Any column in feature_matrix has correlation > 0.9999 with reference_vector.
      3. Any feature contains NaNs or Infs.
    """
    # 1. Column name check
    forbidden_tokens = ['vbox', 'ref', 'label', 'target', 'residual', 'ground', 'speed_ms', 'future']
    for name in feature_names:
        lower_name = name.lower()
        for tok in forbidden_tokens:
            if tok in lower_name and tok != 'diff_var':
                raise ValueError(f"CRITICAL LEAKAGE DETECTED: Forbidden token '{tok}' found in feature name '{name}'!")
                
    # 2. Reference correlation check
    ref_norm = (reference_vector - np.mean(reference_vector)) / (np.std(reference_vector) + 1e-12)
    for col_idx, name in enumerate(feature_names):
        col = feature_matrix[:, col_idx]
        col_std = np.std(col)
        if col_std > 1e-8:
            col_norm = (col - np.mean(col)) / col_std
            corr = np.abs(np.mean(col_norm * ref_norm))
            if corr > 0.9999:
                raise ValueError(f"CRITICAL LEAKAGE DETECTED: Feature '{name}' is mathematically identical to the target reference (corr={corr:.6f})!")
                
    # 3. Sanity check: no NaN, no Inf
    if np.any(np.isnan(feature_matrix)):
        raise ValueError("CRITICAL SANITY ERROR: Feature matrix contains NaNs!")
    if np.any(np.isinf(feature_matrix)):
        raise ValueError("CRITICAL SANITY ERROR: Feature matrix contains Infs!")
        
    return True


def run_machine_checkable_leakage_suite(clean_X: np.ndarray, clean_names: list, y: np.ndarray) -> dict:
    """
    Unit test suite verifying that:
      A. Clean feature matrix passes audit.
      B. Synthetic injection of a future sample is caught and fails.
      C. Synthetic injection of target residual is caught and fails.
      D. Synthetic injection of VBOX speed is caught and fails.
    """
    results = {}
    
    # Test 1: Clean matrix passes
    try:
        leakage_audit_detector(clean_names, clean_X, y)
        results['test_clean_matrix_passes'] = True
    except Exception as e:
        results['test_clean_matrix_passes'] = False
        results['clean_error'] = str(e)
        
    # Test 2: Injected target column MUST FAIL
    bad_names_target = clean_names + ['LABEL_target_residual_ms2']
    bad_X_target = np.column_stack([clean_X, y])
    try:
        leakage_audit_detector(bad_names_target, bad_X_target, y)
        results['test_target_injection_caught'] = False  # Should NOT reach here
    except ValueError:
        results['test_target_injection_caught'] = True   # Correctly raised error
        
    # Test 3: Injected future column MUST FAIL
    bad_names_future = clean_names + ['ax_phone_future_k1']
    bad_X_future = np.column_stack([clean_X, np.roll(clean_X[:, 0], -1)])
    try:
        leakage_audit_detector(bad_names_future, bad_X_future, y)
        results['test_future_injection_caught'] = False  # Should NOT reach here
    except ValueError:
        results['test_future_injection_caught'] = True   # Correctly raised error
        
    # Test 4: Injected duplicate of reference MUST FAIL
    bad_names_ref = clean_names + ['sneaky_feature_copy']
    bad_X_ref = np.column_stack([clean_X, y.copy()])
    try:
        leakage_audit_detector(bad_names_ref, bad_X_ref, y)
        results['test_reference_correlation_caught'] = False # Should NOT reach here
    except ValueError:
        results['test_reference_correlation_caught'] = True  # Correctly caught
        
    all_passed = (results['test_clean_matrix_passes'] and
                  results['test_target_injection_caught'] and
                  results['test_future_injection_caught'] and
                  results['test_reference_correlation_caught'])
    results['all_unit_tests_passed'] = all_passed
    
    return results


# ===========================================================================
# 4. Diagnostic Visualizations
# ===========================================================================

def generate_b0_figures(vta02_df: pd.DataFrame, X2: np.ndarray, y2: np.ndarray, names: list,
                        lag_audit_results: dict):
    """Generates 3 diagnostic figures for Stage C5.3-B0."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # FIGURE 1: Causal Window Schematic
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 5.0)
    ax.axis('off')
    
    # Window blocks
    # Past samples: 0 to 13
    for i in range(14):
        rect = patches.Rectangle((i, 2.5), 0.85, 1.2, linewidth=1.5, edgecolor='darkblue', facecolor='lightblue', alpha=0.8)
        ax.add_patch(rect)
        ax.text(i + 0.425, 3.1, f"k-{14-i}", ha='center', va='center', fontsize=9, fontweight='bold', color='navy')
        
    # Current sample: index 14 (step k)
    rect_k = patches.Rectangle((14, 2.5), 0.85, 1.2, linewidth=2.5, edgecolor='crimson', facecolor='salmon', alpha=0.9)
    ax.add_patch(rect_k)
    ax.text(14.425, 3.1, "k\n(Now)", ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')
    
    # Future samples (BLOCKED): index 15
    rect_fut = patches.Rectangle((15.3, 2.5), 0.85, 1.2, linewidth=2.0, edgecolor='darkred', facecolor='lightgray', alpha=0.5, ls='--')
    ax.add_patch(rect_fut)
    ax.text(15.725, 3.1, "k+1\n(Future)", ha='center', va='center', fontsize=9, color='gray')
    
    # Brick wall barrier between k and k+1
    ax.axvline(15.05, color='darkred', lw=4.0, ls='-')
    ax.text(15.05, 4.3, "STRICT CAUSAL BARRIER\nZero Future Access", ha='center', va='center', fontsize=11, fontweight='bold', color='darkred')
    
    # Bracket under causal window
    ax.annotate('', xy=(14.85, 2.1), xytext=(0.0, 2.1),
                arrowprops=dict(arrowstyle='<->', color='navy', lw=2.0))
    ax.text(7.425, 1.6, "Causal Trailing Feature Window W = 15 samples (1.5 s)\nSamples {k-14, k-13, ..., k}", ha='center', va='center', fontsize=12, fontweight='bold', color='navy')
    
    # Target Residual below current step k
    rect_target = patches.Rectangle((13.5, -0.8), 2.0, 1.3, linewidth=2.0, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.8)
    ax.add_patch(rect_target)
    ax.text(14.5, -0.15, "Target Residual\nr(k) = ax_level[k] - a_ref[k]", ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    # Downward arrow from step k to target
    ax.annotate('', xy=(14.5, 0.6), xytext=(14.5, 2.3),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))
    ax.text(14.5, 1.2, "Aligned Timestamp t_k", ha='right', va='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    # Offline supervision box
    rect_off = patches.Rectangle((0.0, -1.2), 12.0, 1.0, linewidth=1.5, edgecolor='gray', facecolor='whitesmoke', ls=':')
    ax.add_patch(rect_off)
    ax.text(6.0, -0.7, "Category C (Offline Supervision Only): v_VBOX, a_reference (SG9), veh_accel_long_ms2\n[STRICTLY EXCLUDED from Feature Matrix X]", ha='center', va='center', fontsize=10, color='dimgray')
    
    plt.title("Stage C5.3-B0: Causal Windowing Architecture & Zero-Leakage Boundary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_3b0_causal_window_schematic.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    
    # FIGURE 2: Timestamp Alignment Lag Cross-Correlation
    fig, ax = plt.subplots(figsize=(10, 5))
    for trip_name, col in [('Vta02', 'navy'), ('Vta03', 'crimson'), ('Vta04', 'forestgreen')]:
        res = lag_audit_results[trip_name]
        ax.plot(res['lags_seconds'], res['correlations'], marker='o', lw=2.0, label=f"{trip_name} (Peak at δ={res['peak_lag_seconds']:.1f}s, r={res['peak_abs_correlation']:.3f})", color=col)
        
    ax.axvline(0.0, color='black', ls='--', lw=1.5, label='Nominal Zero Lag (δ = 0.0 s)')
    ax.set_xlabel('Lag Shift δ (seconds, positive = target shifted forward)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Correlation: corr(ax_level[k], r[k+δ])', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.3-B0: Timestamp Alignment Audit Across All Trips', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_3b0_timestamp_alignment_lag.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    
    # FIGURE 3: Feature Distributions & Sanity
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    
    idx_ax_lvl = names.index('ax_level_k')
    idx_jerk = names.index('jerk_ax_level_k')
    idx_gyro_std = names.index('gyro_norm_std')
    idx_slope = names.index('ax_level_slope')
    
    axes[0, 0].hist(X2[:, idx_ax_lvl], bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('ax_level_k Distribution (Instantaneous Measured Force)', fontweight='bold')
    axes[0, 0].set_xlabel('m/s²', fontweight='bold')
    
    axes[0, 1].hist(X2[:, idx_jerk], bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('jerk_ax_level_k Distribution (Causal Backward 1st Difference)', fontweight='bold')
    axes[0, 1].set_xlabel('m/s³', fontweight='bold')
    
    axes[1, 0].hist(X2[:, idx_gyro_std], bins=50, color='crimson', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('gyro_norm_std Distribution (Causal Dynamic Rotation)', fontweight='bold')
    axes[1, 0].set_xlabel('rad/s', fontweight='bold')
    
    axes[1, 1].hist(X2[:, idx_slope], bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('ax_level_slope Distribution (Causal Trend over W=1.5s)', fontweight='bold')
    axes[1, 1].set_xlabel('m/s³', fontweight='bold')
    
    for ax_sub in axes.flat:
        ax_sub.grid(True, alpha=0.5)
        
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_3b0_feature_distributions.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    
    print(f"Generated 3 diagnostic figures in {FIG_DIR}")


# ===========================================================================
# 5. Main Execution Pipeline
# ===========================================================================

def main():
    print("===================================================================")
    print("Stage C5.3-B0: Causal Feature Construction & Leakage/Timing Audit")
    print("===================================================================")
    
    # 1. Ingest clean label datasets
    csv_02 = PROC_DIR / "c5_3_labels_vta02.csv"
    csv_03 = PROC_DIR / "c5_3_labels_vta03.csv"
    csv_04 = PROC_DIR / "c5_3_labels_vta04.csv"
    
    for p in [csv_02, csv_03, csv_04]:
        if not p.exists():
            raise FileNotFoundError(f"Missing canonical label dataset: {p}. Run Stage C5.3-A first.")
            
    df_02 = pd.read_csv(csv_02)
    df_03 = pd.read_csv(csv_03)
    df_04 = pd.read_csv(csv_04)
    print(f"Ingested canonical datasets: Vta02 ({len(df_02)}), Vta03 ({len(df_03)}), Vta04 ({len(df_04)})")
    
    # 2. Extract strictly causal features
    print("\n2. Extracting strictly causal features (W = 15 samples = 1.5 s, stride = 1)...")
    X_train, y_train, t_train, feat_names, meta_train = extract_causal_features(df_02, window_size=DEFAULT_WINDOW)
    X_val, y_val, t_val, _, meta_val = extract_causal_features(df_03, window_size=DEFAULT_WINDOW)
    X_test, y_test, t_test, _, meta_test = extract_causal_features(df_04, window_size=DEFAULT_WINDOW)
    
    print(f"Feature Matrix Shapes:")
    print(f"  Train (Vta02):      X={X_train.shape}, y={y_train.shape} (N={len(X_train)} samples)")
    print(f"  Validation (Vta03): X={X_val.shape}, y={y_val.shape} (N={len(X_val)} samples)")
    print(f"  Final Test (Vta04): X={X_test.shape}, y={y_test.shape} (N={len(X_test)} samples, UNTOUCHED)")
    print(f"Total Causal Features: {len(feat_names)}")
    
    # 3. Categorization inventory audit
    print("\n3. Verifying Feature Inventory Categorization...")
    cat_A = [f for f in feat_names if any(f.startswith(p) for p in ['ax_phone', 'ay_phone', 'az_phone', 'gx_phone', 'gy_phone', 'gz_phone']) and f.endswith('_k')]
    cat_B_inst = [f for f in feat_names if f in ['ax_level_k', 'accel_mag_k', 'gyro_norm_k', 'jerk_ax_level_k', 'pitch_accel_gy_k']]
    cat_B_win = [f for f in feat_names if f not in cat_A and f not in cat_B_inst]
    
    print(f"  Category A (Direct Measured Channels):        {len(cat_A)} features")
    print(f"  Category B (Causally Derived Instantaneous):  {len(cat_B_inst)} features")
    print(f"  Category B (Causally Derived Windowed Stats): {len(cat_B_win)} features")
    print(f"  Category C (Offline Supervision):             STRICTLY ZERO in X")
    
    # 4. Feature sanity checks (NaN, Inf, variance)
    print("\n4. Checking Feature Sanity (no NaN, no Inf, non-zero variance)...")
    for name_ds, X_cur in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        assert not np.any(np.isnan(X_cur)), f"NaN detected in {name_ds} feature matrix!"
        assert not np.any(np.isinf(X_cur)), f"Inf detected in {name_ds} feature matrix!"
        stds = np.std(X_cur, axis=0)
        zero_var = np.where(stds < 1e-7)[0]
        assert len(zero_var) == 0, f"Zero variance feature in {name_ds}: {[feat_names[i] for i in zero_var]}"
    print("  Feature Sanity PASSED: Zero NaNs, Zero Infs, all 47 features have non-zero variance.")
    
    # 5. Timestamp alignment & lag sweep
    print("\n5. Auditing Timestamp Alignment Across Trips (Lag Sweep)...")
    lag_audits = {}
    for trip_name, df_cur in [('Vta02', df_02), ('Vta03', df_03), ('Vta04', df_04)]:
        res = audit_timestamp_lag(df_cur, window_size=DEFAULT_WINDOW)
        lag_audits[trip_name] = res
        print(f"  {trip_name}: Peak correlation at lag delta = {res['peak_lag_samples']} samples ({res['peak_lag_seconds']:.1f} s), abs_corr = {res['peak_abs_correlation']:.4f} -> Aligned: {res['alignment_verified']}")
        assert res['alignment_verified'], f"Accidental lag shift detected in {trip_name}!"
    print("  Timestamp Alignment Audit PASSED: Maximum physical coupling occurs at delta = 0.0 s across all trips.")
    
    # 6. Machine-checkable leakage test suite
    print("\n6. Running Machine-Checkable Leakage Test Suite...")
    leakage_tests = run_machine_checkable_leakage_suite(X_train, feat_names, y_train)
    for test_name, passed in leakage_tests.items():
        print(f"  {test_name:<45}: {'PASSED' if passed else 'FAILED'}")
    assert leakage_tests['all_unit_tests_passed'], "Machine-checkable leakage suite FAILED!"
    print("  Leakage Test Suite PASSED: Synthetic future & reference injections were 100% caught.")
    
    # 7. Generate diagnostic figures
    print("\n7. Generating Diagnostic Figures...")
    generate_b0_figures(df_02, X_train, y_train, feat_names, lag_audits)
    
    # 8. Save audit JSON
    verification_record = {
        'stage': 'C5.3-B0',
        'status': 'VERIFIED_AND_CERTIFIED',
        'window_size_samples': DEFAULT_WINDOW,
        'window_duration_seconds': DEFAULT_WINDOW * DT,
        'n_features_total': len(feat_names),
        'feature_inventory': {
            'Category_A_Direct_Measured': cat_A,
            'Category_B_Causal_Derived_Instantaneous': cat_B_inst,
            'Category_B_Causal_Derived_Windowed': cat_B_win,
            'Category_C_Offline_Supervision_Excluded': FEATURE_CATEGORIES['Category_C_Offline_Only_Supervision_FORBIDDEN'],
        },
        'dataset_partitions': {
            'train_vta02': {'n_samples': len(X_train), 'duration_s': len(X_train) * DT},
            'val_vta03': {'n_samples': len(X_val), 'duration_s': len(X_val) * DT},
            'test_vta04_untouched': {'n_samples': len(X_test), 'duration_s': len(X_test) * DT},
        },
        'timestamp_alignment_audit': lag_audits,
        'machine_checkable_leakage_suite': leakage_tests,
    }
    
    out_json = RES_DIR / "c5_3b0_pipeline_verification.json"
    with open(out_json, 'w') as f:
        json.dump(verification_record, f, indent=2)
    print(f"\nSaved pipeline verification record to {out_json}")
    print("===================================================================")
    print("Stage C5.3-B0 execution completed successfully! Green light for C5.3-B1.")
    print("===================================================================")


if __name__ == "__main__":
    main()
