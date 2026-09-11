"""
SIH26168 - Stage C5.4: Dynamic Attitude & Gravity-Leakage Diagnostic Investigation

Script: experiments/run_attitude_gravity_audit_c5_4.py

PURPOSE:
    Execute a purely physical, diagnostic investigation into vehicle/phone attitude dynamics,
    mounting misalignment, and dynamic gravity projection to determine whether attitude-dependent
    gravity leakage accounts for the persistent ~2 m/s² dynamic transient error during braking.

STRICT NON-ML MANDATE:
    - Zero AI / Machine Learning (No Ridge, RF, GBDT, MLP, LSTM, Transformer).
    - Zero Navigation Filter Redesign (No ESKF, no new integration algorithms).
    - Pure kinematic analysis, physical rotation matrices, and statistical regression.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import compute_leveling_matrix
from experiments.run_ridge_baseline_c5_3b1 import simulate_outage_navigation

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"

DT = 0.1
G_CONST = 9.80665


def lowpass_filter(data: np.ndarray, cutoff_hz: float = 1.5, fs: float = 10.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter to isolate suspension band (< 1.5 Hz)."""
    nyq = 0.5 * fs
    wn = min(0.99, max(0.01, cutoff_hz / nyq))
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, data)


def compute_dynamic_leveling_matrix(gx: float, gy: float, gz: float) -> np.ndarray:
    """
    Computes instantaneous leveling rotation matrix R_dyn such that:
    R_dyn @ [gx, gy, gz]^T = [0, 0, ||g||]^T
    """
    roll = np.arctan2(gy, gz)
    pitch = np.arctan2(-gx, np.sqrt(gy**2 + gz**2))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return R_y @ R_x


def analyze_trip_attitude(trip_name: str) -> dict:
    """Performs full physical attitude & gravity leakage diagnostics on a single trip."""
    print(f"\n===================================================================")
    print(f"Analyzing Trip: {trip_name}")
    print(f"===================================================================")
    
    # 1. Ingest clean trip data and canonical labels
    df_p, df_v = load_trip(trip_name)
    labels = pd.read_csv(PROC_DIR / f"c5_3_labels_{trip_name.lower()}.csv")
    
    n = min(len(df_p), len(df_v), len(labels))
    df_p = df_p.iloc[:n].reset_index(drop=True)
    df_v = df_v.iloc[:n].reset_index(drop=True)
    labels = labels.iloc[:n].reset_index(drop=True)
    
    time_s = labels['time_s'].to_numpy()
    a_ref = labels['LABEL_vbox_ref_accel_ms2'].to_numpy()
    r_a = labels['LABEL_target_residual_ms2'].to_numpy()
    v_vbox = labels['LABEL_vbox_speed_ms'].to_numpy()
    ax_phone = labels['ax_phone'].to_numpy()
    ax_level = labels['ax_level'].to_numpy()
    
    # Raw phone channels
    ax_p = df_p['accel_x'].to_numpy()
    ay_p = df_p['accel_y'].to_numpy()
    az_p = df_p['accel_z'].to_numpy()
    
    # Orientation channels
    ori_pitch = df_p['ori_pitch_deg'].to_numpy() if 'ori_pitch_deg' in df_p.columns else np.zeros(n)
    ori_roll = df_p['ori_roll_deg'].to_numpy() if 'ori_roll_deg' in df_p.columns else np.zeros(n)
    ori_yaw = df_p['ori_yaw_deg'].to_numpy() if 'ori_yaw_deg' in df_p.columns else np.zeros(n)
    
    # Gyroscopes
    gx_p = df_p['gyro_x'].to_numpy() if 'gyro_x' in df_p.columns else np.zeros(n)
    gy_p = df_p['gyro_y'].to_numpy() if 'gyro_y' in df_p.columns else np.zeros(n)
    gz_p = df_p['gyro_z'].to_numpy() if 'gyro_z' in df_p.columns else np.zeros(n)
    
    # Phone gravity vector
    grav_x = df_p['grav_x'].to_numpy() if 'grav_x' in df_p.columns else np.zeros(n)
    grav_y = df_p['grav_y'].to_numpy() if 'grav_y' in df_p.columns else np.zeros(n)
    grav_z = df_p['grav_z'].to_numpy() if 'grav_z' in df_p.columns else np.full(n, G_CONST)
    
    # Vehicle CAN channels
    can_accel = df_v['veh_accel_long_ms2'].to_numpy() if 'veh_accel_long_ms2' in df_v.columns else np.zeros(n)
    v_vert = df_v['veh_vert_vel_kmh'].to_numpy() / 3.6 if 'veh_vert_vel_kmh' in df_v.columns else np.zeros(n)
    v_horiz = np.maximum(0.1, df_v['veh_speed_kmh'].to_numpy() / 3.6)
    theta_road_deg = np.degrees(np.arctan(v_vert / v_horiz))
    
    # -----------------------------------------------------------------------
    # Dimension 1: Static Mounting Analysis
    # -----------------------------------------------------------------------
    # Detect stationary samples
    is_stop = (v_vbox < 0.10)
    n_stop = int(np.sum(is_stop))
    if n_stop >= 15:
        ax_stat = float(np.mean(ax_p[is_stop]))
        ay_stat = float(np.mean(ay_p[is_stop]))
        az_stat = float(np.mean(az_p[is_stop]))
        stat_pitch_deg = float(np.degrees(np.arctan2(-ax_stat, np.sqrt(ay_stat**2 + az_stat**2))))
        stat_roll_deg = float(np.degrees(np.arctan2(ay_stat, az_stat)))
    else:
        # Fallback to trip median when no stops present (e.g. Vta04 highway)
        ax_stat = float(np.median(ax_p))
        ay_stat = float(np.median(ay_p))
        az_stat = float(np.median(az_p))
        stat_pitch_deg = float(np.degrees(np.arctan2(-ax_stat, np.sqrt(ay_stat**2 + az_stat**2))))
        stat_roll_deg = float(np.degrees(np.arctan2(ay_stat, az_stat)))
        
    print(f"1. Static Mounting Misalignment:")
    print(f"   Stationary Acceleration: ax={ax_stat:+.4f}, ay={ay_stat:+.4f}, az={az_stat:+.4f} m/s²")
    print(f"   Calculated Static Tilt: Pitch={stat_pitch_deg:+.2f}°, Roll={stat_roll_deg:+.2f}°")
    print(f"   Onboard Fused Orientation: Mean Pitch={np.mean(ori_pitch):.2f}°, Mean Roll={np.mean(ori_roll):.2f}°")
    
    # -----------------------------------------------------------------------
    # Dimension 2: Dynamic Attitude & Pitch Deflection during Braking
    # -----------------------------------------------------------------------
    brake_mask = a_ref < -0.5
    sev_brake_mask = a_ref <= -1.5
    cruise_mask = (np.abs(a_ref) < 0.5) & (v_vbox > 5.0)
    
    # Dynamic pitch deflection relative to cruise / stationary baseline
    pitch_baseline = np.mean(ori_pitch[cruise_mask]) if np.sum(cruise_mask) > 10 else np.mean(ori_pitch)
    delta_pitch_deg = ori_pitch - pitch_baseline
    
    # Gravity-vector pitch deflection
    grav_pitch_rad = np.arctan2(grav_x, grav_z)
    grav_pitch_deg = np.degrees(grav_pitch_rad)
    grav_pitch_baseline = np.mean(grav_pitch_deg[cruise_mask]) if np.sum(cruise_mask) > 10 else np.mean(grav_pitch_deg)
    delta_grav_pitch_deg = grav_pitch_deg - grav_pitch_baseline
    
    # Theoretical gravity projection: delta_a_grav = g * sin(delta_pitch)
    theoretical_gravity_ms2 = G_CONST * np.sin(np.radians(delta_pitch_deg))
    theoretical_grav_vector_ms2 = G_CONST * np.sin(np.radians(delta_grav_pitch_deg))
    
    # Pitch rates
    # Numerical derivative of pitch angle
    pitch_rate_num_degs = np.gradient(ori_pitch, DT)
    pitch_rate_num_rads = np.radians(pitch_rate_num_degs)
    # Gyro pitch rate (filtered)
    gyro_pitch_rads = gx_p # in loader gx is mapped to pitch gyro
    
    print(f"\n2. Dynamic Pitch Deflection during Braking:")
    print(f"   Braking samples (a_ref < -0.5): N={np.sum(brake_mask)}, Severe braking: N={np.sum(sev_brake_mask)}")
    if np.sum(sev_brake_mask) > 0:
        print(f"   Severe Braking Residual Mean: {np.mean(r_a[sev_brake_mask]):+.4f} m/s²")
        print(f"   Severe Braking Pitch Delta:   {np.mean(delta_pitch_deg[sev_brake_mask]):+.4f}°")
        print(f"   Theor Gravity Projection:     {np.mean(theoretical_gravity_ms2[sev_brake_mask]):+.4f} m/s²")
        print(f"   CAN Longitudinal Accel:       {np.mean(can_accel[sev_brake_mask]):+.4f} m/s²")
        print(f"   VBOX a_ref Mean:              {np.mean(a_ref[sev_brake_mask]):+.4f} m/s²")
    
    # -----------------------------------------------------------------------
    # Dimension 3: Correlation & Regression Analysis
    # -----------------------------------------------------------------------
    # Correlation during braking
    if np.sum(brake_mask) > 5:
        corr_fused = float(np.corrcoef(r_a[brake_mask], delta_pitch_deg[brake_mask])[0, 1])
        corr_grav = float(np.corrcoef(r_a[brake_mask], delta_grav_pitch_deg[brake_mask])[0, 1])
        corr_road = float(np.corrcoef(r_a[brake_mask], theta_road_deg[brake_mask])[0, 1])
        corr_gyro = float(np.corrcoef(r_a[brake_mask], gyro_pitch_rads[brake_mask])[0, 1])
        corr_theor = float(np.corrcoef(r_a[brake_mask], theoretical_gravity_ms2[brake_mask])[0, 1])
    else:
        corr_fused = corr_grav = corr_road = corr_gyro = corr_theor = 0.0
        
    # Correlation across entire trip
    corr_trip_fused = float(np.corrcoef(r_a, delta_pitch_deg)[0, 1])
    corr_trip_theor = float(np.corrcoef(r_a, theoretical_gravity_ms2)[0, 1])
    corr_can_ref = float(np.corrcoef(can_accel, a_ref)[0, 1])
    
    print(f"\n3. Statistical Correlations with Acceleration Residual r_a:")
    print(f"   corr(r_a, Delta theta_pitch) [Braking]:      {corr_fused:+.4f}")
    print(f"   corr(r_a, Delta theta_gravity) [Braking]:    {corr_grav:+.4f}")
    print(f"   corr(r_a, Road Grade Slope) [Braking]:       {corr_road:+.4f}")
    print(f"   corr(r_a, Gyro Pitch Rate) [Braking]:        {corr_gyro:+.4f}")
    print(f"   corr(CAN accel, VBOX a_ref) [Entire Trip]:   {corr_can_ref:+.4f}")
    
    # Diagnostic OLS Regression: r_a ~ beta_0 + beta_1 * delta_theta + beta_2 * theta_dot
    X_reg = np.column_stack([
        np.ones(len(r_a)),
        delta_pitch_deg,
        pitch_rate_num_rads,
        theta_road_deg
    ])
    w_ols, res, _, _ = np.linalg.lstsq(X_reg, r_a, rcond=None)
    pred_r_ols = X_reg @ w_ols
    r2_ols = 1.0 - float(np.var(r_a - pred_r_ols) / np.var(r_a))
    
    print(f"   Diagnostic OLS Regression R²: {r2_ols:+.4f}")
    print(f"   OLS Weights [const, delta_pitch(deg), pitch_rate(rad/s), road_grade(deg)]:")
    print(f"   {w_ols}")
    
    # -----------------------------------------------------------------------
    # Dimension 4: Three-Way Acceleration Profiles & Compensation
    # -----------------------------------------------------------------------
    # 1. Raw Phone Acceleration: ax_p
    err_raw = ax_p - a_ref
    mae_raw = float(np.mean(np.abs(err_raw)))
    rmse_raw = float(np.sqrt(np.mean(err_raw**2)))
    bias_raw_sev = float(np.mean(err_raw[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    # 2. Static Leveled Acceleration: ax_level
    err_static = ax_level - a_ref
    mae_static = float(np.mean(np.abs(err_static)))
    rmse_static = float(np.sqrt(np.mean(err_static**2)))
    bias_static_sev = float(np.mean(err_static[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    # 3. Dynamic Gravity-Compensated Acceleration:
    # Method A: Subtract theoretical gravity leakage based on fused pitch:
    ax_dyn_fused = ax_level - theoretical_gravity_ms2
    err_dyn_fused = ax_dyn_fused - a_ref
    mae_dyn_fused = float(np.mean(np.abs(err_dyn_fused)))
    rmse_dyn_fused = float(np.sqrt(np.mean(err_dyn_fused**2)))
    bias_dyn_fused_sev = float(np.mean(err_dyn_fused[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    # Method B: Dynamic instantaneous leveling matrix R_dyn(t) using real-time gravity vector
    ax_dyn_matrix = np.zeros(n)
    for k in range(n):
        R_k = compute_dynamic_leveling_matrix(grav_x[k], grav_y[k], grav_z[k])
        a_rot = R_k @ np.array([ax_p[k], ay_p[k], az_p[k]])
        ax_dyn_matrix[k] = a_rot[0]
        
    err_dyn_mat = ax_dyn_matrix - a_ref
    mae_dyn_mat = float(np.mean(np.abs(err_dyn_mat)))
    rmse_dyn_mat = float(np.sqrt(np.mean(err_dyn_mat**2)))
    bias_dyn_mat_sev = float(np.mean(err_dyn_mat[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    # Method C: Road-Grade Corrected Dynamic Leveling
    ax_dyn_road = ax_dyn_fused - G_CONST * np.sin(np.radians(theta_road_deg))
    err_dyn_road = ax_dyn_road - a_ref
    mae_dyn_road = float(np.mean(np.abs(err_dyn_road)))
    rmse_dyn_road = float(np.sqrt(np.mean(err_dyn_road**2)))
    bias_dyn_road_sev = float(np.mean(err_dyn_road[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    # Method D: CAN Acceleration Benchmark (Chassis Ground Truth Accelerometer)
    err_can = can_accel - a_ref
    mae_can = float(np.mean(np.abs(err_can)))
    rmse_can = float(np.sqrt(np.mean(err_can**2)))
    bias_can_sev = float(np.mean(err_can[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    print(f"\n4. Comparative Acceleration Error Profiles:")
    print(f"   {'Signal Mode':<30} | {'MAE (m/s²)':<10} | {'RMSE (m/s²)':<11} | {'Sev Brake Bias (m/s²)':<22}")
    print(f"   {'-'*78}")
    print(f"   {'1. Raw Phone Acceleration':<30} | {mae_raw:<10.4f} | {rmse_raw:<11.4f} | {bias_raw_sev:<22.4f}")
    print(f"   {'2. Static Leveled (Baseline)':<30} | {mae_static:<10.4f} | {rmse_static:<11.4f} | {bias_static_sev:<22.4f}")
    print(f"   {'3. Dyn Pitch Compensated':<30} | {mae_dyn_fused:<10.4f} | {rmse_dyn_fused:<11.4f} | {bias_dyn_fused_sev:<22.4f}")
    print(f"   {'4. Dyn Gravity Matrix':<30} | {mae_dyn_mat:<10.4f} | {rmse_dyn_mat:<11.4f} | {bias_dyn_mat_sev:<22.4f}")
    print(f"   {'5. Dyn + Road Grade Corrected':<30} | {mae_dyn_road:<10.4f} | {rmse_dyn_road:<11.4f} | {bias_dyn_road_sev:<22.4f}")
    print(f"   {'6. Chassis CAN (Ground Truth)':<30} | {mae_can:<10.4f} | {rmse_can:<11.4f} | {bias_can_sev:<22.4f}")
    
    # -----------------------------------------------------------------------
    # Dimension 5: Dead-Reckoning Navigation Drift Simulation
    # -----------------------------------------------------------------------
    nav_series = {
        'raw_phone': ax_p,
        'static_leveled': ax_level,
        'dyn_pitch_comp': ax_dyn_fused,
        'chassis_can': can_accel,
        'oracle_ref': a_ref
    }
    nav_res = simulate_outage_navigation(v_vbox, nav_series, horizons=[5.0, 10.0, 20.0, 30.0, 60.0], stride_s=2.5)
    
    print(f"\n5. Dead-Reckoning Navigation Position Drift (Meters):")
    print(f"   {'Horizon':<8} | {'Raw Phone':<10} | {'Static Level':<12} | {'Dyn Pitch Comp':<15} | {'Chassis CAN':<12} | {'Oracle':<8}")
    print(f"   {'-'*75}")
    for h in [5, 10, 20, 30, 60]:
        h_k = f"{h}s"
        if h_k in nav_res:
            m_data = nav_res[h_k]['methods']
            p_raw = m_data['raw_phone']['mean_pos_error_m']
            p_stat = m_data['static_leveled']['mean_pos_error_m']
            p_dyn = m_data['dyn_pitch_comp']['mean_pos_error_m']
            p_can = m_data['chassis_can']['mean_pos_error_m']
            p_orc = m_data['oracle_ref']['mean_pos_error_m']
            print(f"   {h_k:<8} | {p_raw:<10.2f} | {p_stat:<12.2f} | {p_dyn:<15.2f} | {p_can:<12.2f} | {p_orc:<8.2f}")
            
    return {
        'trip_name': trip_name,
        'samples': n,
        'static_misalignment': {
            'ax_stat_ms2': ax_stat,
            'ay_stat_ms2': ay_stat,
            'az_stat_ms2': az_stat,
            'stat_pitch_deg': stat_pitch_deg,
            'stat_roll_deg': stat_roll_deg,
            'mean_ori_pitch_deg': float(np.mean(ori_pitch)),
            'mean_ori_roll_deg': float(np.mean(ori_roll)),
        },
        'correlations': {
            'corr_r_a_delta_pitch_braking': corr_fused,
            'corr_r_a_delta_grav_braking': corr_grav,
            'corr_r_a_road_grade_braking': corr_road,
            'corr_r_a_gyro_pitch_braking': corr_gyro,
            'corr_can_ref_entire_trip': corr_can_ref,
        },
        'regression_ols': {
            'r2': r2_ols,
            'weights': {
                'intercept': float(w_ols[0]),
                'delta_pitch_deg': float(w_ols[1]),
                'pitch_rate_rads': float(w_ols[2]),
                'road_grade_deg': float(w_ols[3]),
            }
        },
        'acceleration_error_profiles': {
            'raw_phone': {'mae': mae_raw, 'rmse': rmse_raw, 'sev_brake_bias': bias_raw_sev},
            'static_leveled': {'mae': mae_static, 'rmse': rmse_static, 'sev_brake_bias': bias_static_sev},
            'dyn_pitch_comp': {'mae': mae_dyn_fused, 'rmse': rmse_dyn_fused, 'sev_brake_bias': bias_dyn_fused_sev},
            'dyn_grav_mat': {'mae': mae_dyn_mat, 'rmse': rmse_dyn_mat, 'sev_brake_bias': bias_dyn_mat_sev},
            'dyn_road_comp': {'mae': mae_dyn_road, 'rmse': rmse_dyn_road, 'sev_brake_bias': bias_dyn_road_sev},
            'chassis_can': {'mae': mae_can, 'rmse': rmse_can, 'sev_brake_bias': bias_can_sev},
        },
        'navigation_drift': {
            h_k: {m: nav_res[h_k]['methods'][m]['mean_pos_error_m'] for m in nav_res[h_k]['methods']}
            for h_k in nav_res
        },
        'time_s': time_s,
        'r_a': r_a,
        'a_ref': a_ref,
        'delta_pitch_deg': delta_pitch_deg,
        'theor_grav': theoretical_gravity_ms2,
        'ax_p': ax_p,
        'ax_level': ax_level,
        'ax_dyn_fused': ax_dyn_fused,
        'can_accel': can_accel,
        'brake_mask': brake_mask,
        'sev_brake_mask': sev_brake_mask,
    }


def generate_diagnostic_figures(results_02: dict, results_04: dict):
    """Generates 3 publication-grade diagnostic figures."""
    print("\nGenerating Publication-Grade Diagnostic Figures...")
    
    # -----------------------------------------------------------------------
    # Figure 1: Pitch vs Acceleration Residual Scatter & Correlation
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (res, title) in enumerate([(results_02, "Vta02 (Urban with Stops)"), 
                                        (results_04, "Vta04 (Untouched Highway)")]):
        r_a = res['r_a']
        dp = res['delta_pitch_deg']
        b_mask = res['brake_mask']
        sb_mask = res['sev_brake_mask']
        c_corr = res['correlations']['corr_r_a_delta_pitch_braking']
        
        # All samples in light gray
        axes[idx].scatter(dp, r_a, color='lightgray', alpha=0.3, s=10, label='All Samples')
        # Braking in orange
        axes[idx].scatter(dp[b_mask], r_a[b_mask], color='#ff7f0e', alpha=0.6, s=18, label='Braking (a_ref < -0.5)')
        # Severe braking in red
        if np.sum(sb_mask) > 0:
            axes[idx].scatter(dp[sb_mask], r_a[sb_mask], color='#d62728', alpha=0.9, s=28, label='Severe Braking (a_ref <= -1.5)')
            
        # Linear fit line during braking
        if np.sum(b_mask) > 5:
            poly = np.polyfit(dp[b_mask], r_a[b_mask], 1)
            x_line = np.linspace(np.min(dp[b_mask]), np.max(dp[b_mask]), 50)
            axes[idx].plot(x_line, np.polyval(poly, x_line), 'k--', lw=2, label=f'Fit: slope={poly[0]:.2f} (r={c_corr:.2f})')
            
        axes[idx].axhline(0, color='k', ls=':', alpha=0.5)
        axes[idx].axvline(0, color='k', ls=':', alpha=0.5)
        axes[idx].set_xlabel('Dynamic Pitch Deflection Δθ_pitch (degrees)')
        axes[idx].set_ylabel('Acceleration Residual r_a (m/s²)')
        axes[idx].set_title(f'{title}: Dynamic Pitch vs. Residual')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper right', fontsize=8.5)
        
    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_4_pitch_vs_residual_scatter.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig1_path.name}")
    
    # -----------------------------------------------------------------------
    # Figure 2: Gravity Leakage Decomposition & Zoomed Time Series
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Select a high-braking 40-second window on Vta02
    t02 = results_02['time_s']
    r02 = results_02['r_a']
    tg02 = results_02['theor_grav']
    can02 = results_02['can_accel']
    aref02 = results_02['a_ref']
    
    # Find window with most severe braking on Vta02
    window_mask = (t02 >= 180.0) & (t02 <= 240.0)
    
    axes[0].plot(t02[window_mask], aref02[window_mask], color='black', lw=2.5, label='VBOX a_ref (True Deceleration)')
    axes[0].plot(t02[window_mask], can02[window_mask], color='#2ca02c', lw=1.8, ls='--', label='Chassis CAN Accelerometer')
    axes[0].plot(t02[window_mask], results_02['ax_level'][window_mask], color='#1f77b4', lw=1.5, alpha=0.8, label='Static Leveled ax_level')
    axes[0].plot(t02[window_mask], results_02['ax_dyn_fused'][window_mask], color='#9467bd', lw=1.8, label='Dynamic Attitude Compensated')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Vta02 Maneuver Tracking: True Deceleration vs. Compensated Accelerations')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower left', fontsize=9)
    
    # Subplot 2: Acceleration Residual vs Theoretical Gravity Projection
    axes[1].plot(t02[window_mask], r02[window_mask], color='#d62728', lw=2, label='Observed Residual r_a = ax_level - a_ref')
    axes[1].plot(t02[window_mask], tg02[window_mask], color='#ff7f0e', lw=2, ls='-.', label='Theoretical Gravity Projection g*sin(Δθ)')
    axes[1].axhline(0, color='k', ls=':', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Residual / Gravity Error (m/s²)')
    axes[1].set_title('Observed Residual vs. Theoretical Gravity Leakage Component')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_4_gravity_leakage_decomposition.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig2_path.name}")
    
    # -----------------------------------------------------------------------
    # Figure 3: Three-Way Acceleration Comparison (Error Distributions)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Vta04 Error Distributions
    err_stat_04 = results_04['ax_level'] - results_04['a_ref']
    err_dyn_04 = results_04['ax_dyn_fused'] - results_04['a_ref']
    err_can_04 = results_04['can_accel'] - results_04['a_ref']
    
    bins = np.linspace(-6, 6, 61)
    axes[0].hist(err_stat_04, bins=bins, color='#1f77b4', alpha=0.5, density=True, label=f'Static Leveled (MAE={results_04["acceleration_error_profiles"]["static_leveled"]["mae"]:.2f})')
    axes[0].hist(err_dyn_04, bins=bins, color='#9467bd', alpha=0.5, density=True, label=f'Dyn Compensated (MAE={results_04["acceleration_error_profiles"]["dyn_pitch_comp"]["mae"]:.2f})')
    axes[0].hist(err_can_04, bins=bins, color='#2ca02c', alpha=0.5, density=True, label=f'Chassis CAN (MAE={results_04["acceleration_error_profiles"]["chassis_can"]["mae"]:.2f})')
    axes[0].axvline(0, color='k', ls='--', alpha=0.7)
    axes[0].set_xlabel('Acceleration Error vs. VBOX a_ref (m/s²)')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Vta04: Full Trip Error Distributions')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8.5)
    
    # Navigation Drift Benchmark at 30s across all modes
    horizons = [5, 10, 20, 30, 60]
    nav_04 = results_04['navigation_drift']
    p_stat = [nav_04[f"{h}s"]['static_leveled'] for h in horizons]
    p_dyn = [nav_04[f"{h}s"]['dyn_pitch_comp'] for h in horizons]
    p_can = [nav_04[f"{h}s"]['chassis_can'] for h in horizons]
    p_orc = [nav_04[f"{h}s"]['oracle_ref'] for h in horizons]
    
    axes[1].plot(horizons, p_stat, 'o-', color='#1f77b4', lw=2, label='Static Leveled Baseline')
    axes[1].plot(horizons, p_dyn, 's-', color='#9467bd', lw=2, label='Dyn Attitude Compensated')
    axes[1].plot(horizons, p_can, 'D-', color='#2ca02c', lw=2.5, label='Chassis CAN (Chassis Upper Bound)')
    axes[1].plot(horizons, p_orc, 'k:', lw=1.5, label='Oracle Floor')
    axes[1].set_xlabel('Outage Horizon (s)')
    axes[1].set_ylabel('Mean Position Drift (m)')
    axes[1].set_title('Vta04: Dead-Reckoning Position Drift Across Horizons')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=8.5)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_4_three_way_acceleration_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig3_path.name}")


def main():
    print("=" * 80)
    print("STAGE C5.4: DYNAMIC ATTITUDE & GRAVITY-LEAKAGE DIAGNOSTIC INVESTIGATION")
    print("=" * 80)
    
    # Analyze all 3 trips
    res_02 = analyze_trip_attitude("Vta02")
    res_03 = analyze_trip_attitude("Vta03")
    res_04 = analyze_trip_attitude("Vta04")
    
    # Generate diagnostic figures
    generate_diagnostic_figures(res_02, res_04)
    
    # Clean exportable dictionary
    export_dict = {
        'metadata': {
            'stage': 'C5.4',
            'description': 'Dynamic Attitude & Gravity-Leakage Diagnostic Investigation',
            'timestamp': '2026-09-05',
            'sampling_rate_hz': 10.0,
            'dt_s': 0.1,
            'g_const_ms2': G_CONST,
            'trips_evaluated': ['Vta02', 'Vta03', 'Vta04'],
        },
        'Vta02': {k: v for k, v in res_02.items() if not isinstance(v, (np.ndarray, list)) or k == 'navigation_drift'},
        'Vta03': {k: v for k, v in res_03.items() if not isinstance(v, (np.ndarray, list)) or k == 'navigation_drift'},
        'Vta04': {k: v for k, v in res_04.items() if not isinstance(v, (np.ndarray, list)) or k == 'navigation_drift'},
    }
    
    out_json_path = RES_DIR / "c5_4_attitude_gravity_audit.json"
    with open(out_json_path, 'w') as f:
        json.dump(export_dict, f, indent=2)
    print(f"\nDiagnostic audit results saved to: {out_json_path}")


if __name__ == '__main__':
    main()
