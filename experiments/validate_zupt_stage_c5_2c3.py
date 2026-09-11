"""
SIH26168 - Stage C5.2-C3: Stop Detection and Pure Kinematic ZUPT Benchmark
Script: experiments/validate_zupt_stage_c5_2c3.py

RESEARCH QUESTION:
    Can genuine vehicle stops reset accumulated velocity integration drift
    through an IMU-triggered Zero-Velocity Update (ZUPT)?

STRICT ONE-CHANGE-AT-A-TIME BOUNDARIES:
    - NO AI / Machine Learning
    - NO ESKF / Kalman Filtering
    - NO Non-Holonomic Constraints (NHC)
    - NO Map Matching / OSM / HMM
    - NO GNSS / GPS speed / GPS bearing fusion
    - NO wheel-speed fusion

EVALUATION DATASET:
    Trip Vta02 (1099.0 s, ~18.3 min), selected because it contains genuine
    stationary episodes (traffic lights, stops).

SUB-EXPERIMENTS:
    C3-A: Stationarity Detector Validation (IMU-only vs offline VBOX reference)
    C3-B: Pure Kinematic Integration + ZUPT (C3-B0 vs C3-B1)
    C3-C: ZUPT Event Validity Audit & Oracle Valid-ZUPT Decomposition (C3-B2)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import compute_leveling_matrix

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.1  # 10 Hz sample period


# ===========================================================================
# C3-A: Stationarity Detector (IMU-Only, Strictly Causal)
# ===========================================================================
def run_stationarity_detector(accel_x: np.ndarray, accel_y: np.ndarray,
                              accel_z: np.ndarray, gyro_x: np.ndarray,
                              gyro_y: np.ndarray, gyro_z: np.ndarray,
                              window_samples: int = 20,
                              accel_std_thresh: float = 0.15,
                              gyro_norm_thresh: float = 0.05) -> tuple:
    """
    Strictly causal IMU-only stationarity detector.
    
    At sample k, inspects trailing window [k - window_samples + 1 : k + 1].
    
    Criteria (inherited from C5.2-C2):
      1. std(||a||) < accel_std_thresh (0.15 m/s²)
      2. mean(||omega||) < gyro_norm_thresh (0.05 rad/s)
      
    Returns:
      is_stationary: boolean array of length n
      a_std_trace: rolling std of accel magnitude
      g_mean_trace: rolling mean of gyro norm
    """
    n = len(accel_x)
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    gyro_norm = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    is_stationary = np.zeros(n, dtype=bool)
    a_std_trace = np.zeros(n, dtype=np.float64)
    g_mean_trace = np.zeros(n, dtype=np.float64)
    
    w = window_samples
    for i in range(n):
        if i < w - 1:
            sl = slice(0, i + 1)
        else:
            sl = slice(i - w + 1, i + 1)
            
        a_std = float(np.std(accel_mag[sl]))
        g_mean = float(np.mean(gyro_norm[sl]))
        
        a_std_trace[i] = a_std
        g_mean_trace[i] = g_mean
        
        if i >= w - 1 and a_std < accel_std_thresh and g_mean < gyro_norm_thresh:
            is_stationary[i] = True
            
    return is_stationary, a_std_trace, g_mean_trace


def compute_vbox_stop_reference(v_vbox: np.ndarray, speed_thresh: float = 0.10,
                                min_duration_samples: int = 10) -> tuple:
    """
    Offline reference stop mask computed from VBOX ground truth speed.
    
    Criteria:
      - v_vbox < speed_thresh (0.10 m/s = 0.36 km/h)
      - Contiguous duration >= min_duration_samples (1.0 s)
      
    Returns:
      ref_mask: boolean array of length n
      episodes: list of dicts with episode metadata
    """
    raw_mask = v_vbox < speed_thresh
    labeled, num_features = label(raw_mask)
    
    ref_mask = np.zeros_like(raw_mask, dtype=bool)
    episodes = []
    
    for ep_id in range(1, num_features + 1):
        idx = np.where(labeled == ep_id)[0]
        dur_samples = len(idx)
        if dur_samples >= min_duration_samples:
            ref_mask[idx] = True
            episodes.append({
                'episode_id': ep_id,
                'start_idx': int(idx[0]),
                'end_idx': int(idx[-1]),
                'duration_samples': int(dur_samples),
                'duration_s': float(dur_samples * DT),
                'mean_speed_ms': float(np.mean(v_vbox[idx])),
                'max_speed_ms': float(np.max(v_vbox[idx])),
            })
            
    return ref_mask, episodes


# ===========================================================================
# C3-B: Kinematic Integration Engine with Optional ZUPT
# ===========================================================================
def run_kinematic_integration(time_s: np.ndarray, a_long: np.ndarray,
                              v_true: np.ndarray, is_stationary: np.ndarray,
                              t_start: float, duration_s: float,
                              use_zupt: bool = False) -> dict:
    """
    Trapezoidal integration of longitudinal acceleration with optional ZUPT.
    
    When use_zupt is True:
      If is_stationary[k] is True: v_est[k] = 0.0
    """
    k_start = int(round(t_start / DT))
    k_end = min(int(round((t_start + duration_s) / DT)), len(time_s) - 1)
    
    t_win = time_s[k_start:k_end + 1]
    v_win_true = v_true[k_start:k_end + 1]
    a_win = a_long[k_start:k_end + 1]
    stat_win = is_stationary[k_start:k_end + 1]
    
    n_win = len(t_win)
    v_est = np.zeros(n_win, dtype=np.float64)
    v_est[0] = v_win_true[0]
    
    zupt_applied_count = 0
    pre_zupt_errors = []
    post_zupt_errors = []
    zupt_indices = []
    
    for k in range(1, n_win):
        if use_zupt and stat_win[k]:
            v_integrated = v_est[k - 1] + 0.5 * (a_win[k] + a_win[k - 1]) * DT
            pre_err = v_integrated - v_win_true[k]
            
            v_est[k] = 0.0
            post_err = 0.0 - v_win_true[k]
            
            zupt_applied_count += 1
            zupt_indices.append(k)
            pre_zupt_errors.append(float(pre_err))
            post_zupt_errors.append(float(post_err))
        else:
            v_est[k] = v_est[k - 1] + 0.5 * (a_win[k] + a_win[k - 1]) * DT
            
    err = v_est - v_win_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    final_err = float(err[-1])
    
    pos_err = np.zeros(n_win, dtype=np.float64)
    for k in range(1, n_win):
        pos_err[k] = pos_err[k - 1] + 0.5 * (err[k] + err[k - 1]) * DT
        
    true_dist = float(np.trapezoid(v_win_true, t_win))
    drift_pct = float(np.abs(pos_err[-1]) / max(true_dist, 1.0) * 100.0)
    
    return {
        't_start_s': float(t_start),
        'duration_s': float(duration_s),
        'v0_ms': float(v_win_true[0]),
        'v_final_true_ms': float(v_win_true[-1]),
        'v_final_est_ms': float(v_est[-1]),
        'mae_ms': mae,
        'rmse_ms': rmse,
        'final_err_ms': final_err,
        'final_pos_err_m': float(pos_err[-1]),
        'true_dist_m': true_dist,
        'drift_pct': drift_pct,
        'use_zupt': use_zupt,
        'zupt_applied_count': zupt_applied_count,
        'zupt_fraction': float(zupt_applied_count / max(n_win, 1)),
        'pre_zupt_errors': pre_zupt_errors,
        'post_zupt_errors': post_zupt_errors,
        't_win': t_win,
        'v_est': v_est,
        'v_true': v_win_true,
        'err': err,
        'pos_err': pos_err,
    }


# ===========================================================================
# Main Execution
# ===========================================================================
def main():
    print("=" * 90)
    print("STAGE C5.2-C3: STOP DETECTION AND PURE KINEMATIC ZUPT BENCHMARK")
    print("=" * 90)
    
    # Load Vta02
    df_p, df_v = load_trip("Vta02")
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()
    
    time_s = df_p['time_s'].to_numpy()
    v_true = df_v['veh_speed_ms'].to_numpy()
    
    ax_p = df_p['accel_x'].to_numpy()
    ay_p = df_p['accel_y'].to_numpy()
    az_p = df_p['accel_z'].to_numpy()
    
    gx = df_p['gyro_x'].to_numpy()
    gy = df_p['gyro_y'].to_numpy()
    gz = df_p['gyro_z'].to_numpy()
    
    # Gravity-leveled longitudinal acceleration
    if 'grav_x' in df_p.columns:
        g_vec = df_p[['grav_x', 'grav_y', 'grav_z']].values.mean(axis=0)
    else:
        g_vec = np.array([ax_p.mean(), ay_p.mean(), az_p.mean()])
    R_level, roll, pitch = compute_leveling_matrix(g_vec)
    acc_l_stack = R_level @ np.vstack([ax_p, ay_p, az_p])
    ax_level = acc_l_stack[0]
    
    print(f"\nTrip: Vta02 ({n} samples, {n * DT:.1f} s, {n * DT / 60:.1f} min)")
    print(f"Leveling: roll = {np.degrees(roll):.2f}°, pitch = {np.degrees(pitch):.2f}°")
    print(f"Longitudinal a_x_level: mean = {ax_level.mean():+.4f} m/s², std = {ax_level.std():.4f} m/s²")
    print(f"VBOX speed: mean = {v_true.mean():.2f} m/s, max = {v_true.max():.2f} m/s")
    
    # =======================================================================
    # SUB-EXPERIMENT C3-A: Stationarity Detector Validation
    # =======================================================================
    print("\n" + "=" * 90)
    print("SUB-EXPERIMENT C3-A: STATIONARITY DETECTOR VALIDATION")
    print("=" * 90)
    
    W_SAMPLES = 20  # 2.0 s causal trailing window
    ACCEL_STD_THRESH = 0.15  # m/s²
    GYRO_NORM_THRESH = 0.05  # rad/s (~2.86 deg/s)
    
    is_stat_imu, a_std_trace, g_mean_trace = run_stationarity_detector(
        ax_p, ay_p, az_p, gx, gy, gz,
        window_samples=W_SAMPLES,
        accel_std_thresh=ACCEL_STD_THRESH,
        gyro_norm_thresh=GYRO_NORM_THRESH
    )
    
    # Offline ground-truth stop reference from VBOX speed
    VBOX_STOP_THRESH = 0.10  # m/s (0.36 km/h)
    MIN_STOP_DUR_SAMPLES = 10  # 1.0 s
    
    ref_mask, ref_episodes = compute_vbox_stop_reference(
        v_true, speed_thresh=VBOX_STOP_THRESH,
        min_duration_samples=MIN_STOP_DUR_SAMPLES
    )
    
    # Sample-level confusion matrix
    tp = int(np.sum(is_stat_imu & ref_mask))
    fp = int(np.sum(is_stat_imu & (~ref_mask)))
    fn = int(np.sum((~is_stat_imu) & ref_mask))
    tn = int(np.sum((~is_stat_imu) & (~ref_mask)))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / n)
    
    print(f"\n[C3-A Sample-Level Confusion Matrix]:")
    print(f"  True Positives  (TP): {tp:5d} samples ({tp * DT:.1f} s)")
    print(f"  False Positives (FP): {fp:5d} samples ({fp * DT:.1f} s)  <-- CRITICAL INSPECTION")
    print(f"  False Negatives (FN): {fn:5d} samples ({fn * DT:.1f} s)")
    print(f"  True Negatives  (TN): {tn:5d} samples ({tn * DT:.1f} s)")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1 Score:  {f1 * 100:.2f}%")
    print(f"  IoU:       {iou * 100:.2f}%")
    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    
    # Event-level stop analysis with correct premature lead / lag
    print(f"\n[C3-A Reference Stops from VBOX (Total {len(ref_episodes)} stops, {np.sum(ref_mask)*DT:.1f} s)]:")
    
    # Extract continuous IMU ZUPT episodes
    labeled_zupt, num_zupt = label(is_stat_imu)
    
    for ep in ref_episodes:
        t_ref_start = time_s[ep['start_idx']]
        t_ref_end = time_s[ep['end_idx']]
        
        # Find which IMU episodes overlap with this reference stop
        ref_set = set(range(ep['start_idx'], ep['end_idx'] + 1))
        matching_ep_indices = []
        for i_ep in range(1, num_zupt + 1):
            zupt_idx = np.where(labeled_zupt == i_ep)[0]
            if len(ref_set.intersection(set(zupt_idx))) > 0:
                matching_ep_indices.append(zupt_idx)
                
        if len(matching_ep_indices) > 0:
            first_idx = matching_ep_indices[0][0]
            last_idx = matching_ep_indices[-1][-1]
            t_first_det = float(time_s[first_idx])
            t_last_det = float(time_s[last_idx])
            
            lead_s = float(t_ref_start - t_first_det) # positive = premature trigger
            lag_s = float(t_last_det - t_ref_end)     # positive = delayed release
            
            sub_det = is_stat_imu[ep['start_idx']:ep['end_idx'] + 1]
            det_pct = float(np.sum(sub_det) / len(sub_det) * 100)
            
            ep['first_det_t_s'] = t_first_det
            ep['lead_time_s'] = lead_s
            ep['lag_time_s'] = lag_s
            ep['imu_coverage_pct'] = det_pct
            
            if lead_s > 0.05:
                status = f"DETECTED (PREMATURE TRIGGER: +{lead_s:.1f} s early, coverage: {det_pct:.1f}%)"
            elif lead_s < -0.05:
                status = f"DETECTED (DELAYED TRIGGER: {abs(lead_s):.1f} s latency, coverage: {det_pct:.1f}%)"
            else:
                status = f"DETECTED (EXACT TRIGGER, coverage: {det_pct:.1f}%)"
        else:
            status = "MISSED"
            ep['first_det_t_s'] = None
            ep['lead_time_s'] = None
            ep['lag_time_s'] = None
            ep['imu_coverage_pct'] = 0.0
            
        print(f"  Ref Stop {ep['episode_id']}: t = {t_ref_start:6.1f} to {t_ref_end:6.1f} s "
              f"(dur = {ep['duration_s']:4.1f} s, mean v = {ep['mean_speed_ms']:.3f} m/s) -> {status}")

    # =======================================================================
    # SUB-EXPERIMENT C3-C: ZUPT Event Validity Audit
    # =======================================================================
    print("\n" + "=" * 90)
    print("SUB-EXPERIMENT C3-C: ZUPT EVENT VALIDITY AUDIT (11 CONTIGUOUS EPISODES)")
    print("=" * 90)
    
    episodes_c3c = []
    for i in range(1, num_zupt + 1):
        idx = np.where(labeled_zupt == i)[0]
        t_start = float(time_s[idx[0]])
        t_end = float(time_s[idx[-1]])
        dur = float(len(idx) * DT)
        v_sub = v_true[idx]
        v_start = float(v_sub[0])
        v_end = float(v_sub[-1])
        v_min = float(np.min(v_sub))
        v_max = float(np.max(v_sub))
        
        # Check overlap with any reference stop
        overlaps = []
        for ref in ref_episodes:
            ref_idx_range = set(range(ref['start_idx'], ref['end_idx'] + 1))
            common = ref_idx_range.intersection(set(idx))
            if len(common) > 0:
                overlaps.append(ref)
                
        if len(overlaps) > 0:
            ref = overlaps[0]
            t_ref_start = float(time_s[ref['start_idx']])
            t_ref_end = float(time_s[ref['end_idx']])
            lead_time = float(t_ref_start - t_start)
            lag_time = float(t_end - t_ref_end)
            
            if lead_time > 0.05 and v_max >= 0.10:
                category = "EARLY TRIGGER (Pre-stop motion)"
            elif lag_time > 0.05 and v_max >= 0.10:
                category = "LATE RELEASE (Post-stop creep)"
            elif v_max < 0.10:
                category = "VALID ZUPT (True Stop)"
            else:
                category = "PARTIALLY VALID"
            ref_id = ref['episode_id']
        else:
            lead_time = None
            lag_time = None
            ref_id = None
            category = "FALSE ZUPT (No stop overlap)"
            
        episodes_c3c.append({
            'episode_id': i,
            't_start_s': t_start,
            't_end_s': t_end,
            'duration_s': dur,
            'v_start_ms': v_start,
            'v_end_ms': v_end,
            'v_min_ms': v_min,
            'v_max_ms': v_max,
            'category': category,
            'ref_stop_id': ref_id,
            'lead_time_s': lead_time,
            'lag_time_s': lag_time
        })
        print(f"  ZUPT Ep {i:2d}: t = {t_start:6.1f} to {t_end:6.1f} s (dur = {dur:4.1f} s) | "
              f"VBOX [{v_min:4.2f}..{v_max:4.2f}] m/s | RefStop: {str(ref_id):4s} | {category}")

    # =======================================================================
    # SUB-EXPERIMENT C3-B & ORACLE C3-B2: Kinematic Integration Benchmark
    # =======================================================================
    print("\n" + "=" * 90)
    print("SUB-EXPERIMENT C3-B & ORACLE DECOMPOSITION: INTEGRATION BENCHMARK")
    print("=" * 90)
    
    segments = [
        {
            'name': 'Full Trip Vta02',
            't_start': 0.0,
            'duration': float(n * DT - 0.1),
            'type': 'Complete Multi-Stop Mission (18.3 min)'
        },
        {
            'name': 'Segment 1: Early Stops',
            't_start': 0.0,
            'duration': 40.0,
            'type': 'Stop-Containing (Stops 1 & 2 at 14.4-23.1 s)'
        },
        {
            'name': 'Segment 2: Major Intersection Stop',
            't_start': 680.0,
            'duration': 100.0,
            'type': 'Stop-Containing (Major 45.9 s Stop 3 at 707.3-753.1 s)'
        },
        {
            'name': 'Segment 3: Late Stops',
            't_start': 970.0,
            'duration': 60.0,
            'type': 'Stop-Containing (Stops 4 & 5 at 994.7-1004.8 s)'
        },
        {
            'name': 'Segment 4: Highway Cruise 1',
            't_start': 40.0,
            'duration': 320.0,
            'type': 'Continuous Driving (Zero Stops for 320 s)'
        },
        {
            'name': 'Segment 5: Highway Cruise 2',
            't_start': 380.0,
            'duration': 220.0,
            'type': 'Continuous Driving (Zero Stops for 220 s)'
        },
    ]
    
    benchmark_results = {}
    
    for seg in segments:
        name = seg['name']
        t0 = seg['t_start']
        dur = seg['duration']
        
        # C3-B0: No ZUPT
        res_b0 = run_kinematic_integration(
            time_s, ax_level, v_true, is_stat_imu,
            t_start=t0, duration_s=dur, use_zupt=False
        )
        
        # C3-B1: Raw IMU-triggered ZUPT
        res_b1 = run_kinematic_integration(
            time_s, ax_level, v_true, is_stat_imu,
            t_start=t0, duration_s=dur, use_zupt=True
        )
        
        # C3-B2: Oracle Valid-ZUPT-Only (strictly during true VBOX stops)
        res_b2 = run_kinematic_integration(
            time_s, ax_level, v_true, ref_mask,
            t_start=t0, duration_s=dur, use_zupt=True
        )
        
        benchmark_results[name] = {
            'type': seg['type'],
            't_start_s': t0,
            'duration_s': dur,
            'true_dist_m': res_b0['true_dist_m'],
            'C3-B0_no_zupt': {
                'mae_ms': res_b0['mae_ms'],
                'rmse_ms': res_b0['rmse_ms'],
                'final_err_ms': res_b0['final_err_ms'],
                'final_pos_err_m': res_b0['final_pos_err_m'],
                'drift_pct': res_b0['drift_pct'],
            },
            'C3-B1_raw_imu_zupt': {
                'mae_ms': res_b1['mae_ms'],
                'rmse_ms': res_b1['rmse_ms'],
                'final_err_ms': res_b1['final_err_ms'],
                'final_pos_err_m': res_b1['final_pos_err_m'],
                'drift_pct': res_b1['drift_pct'],
                'zupt_applied_count': res_b1['zupt_applied_count'],
                'zupt_fraction': res_b1['zupt_fraction'],
            },
            'C3-B2_oracle_valid_zupt': {
                'mae_ms': res_b2['mae_ms'],
                'rmse_ms': res_b2['rmse_ms'],
                'final_err_ms': res_b2['final_err_ms'],
                'final_pos_err_m': res_b2['final_pos_err_m'],
                'drift_pct': res_b2['drift_pct'],
                'zupt_applied_count': res_b2['zupt_applied_count'],
                'zupt_fraction': res_b2['zupt_fraction'],
            },
            'comparisons': {
                'raw_vs_none_mae_pct': float((res_b1['mae_ms'] - res_b0['mae_ms']) / res_b0['mae_ms'] * 100),
                'oracle_vs_none_mae_pct': float((res_b2['mae_ms'] - res_b0['mae_ms']) / res_b0['mae_ms'] * 100),
                'raw_vs_oracle_mae_diff': float(res_b1['mae_ms'] - res_b2['mae_ms']),
            }
        }
        
        print(f"\n--- {name} ({dur:.1f} s, Dist: {res_b0['true_dist_m']:.1f} m) ---")
        print(f"  C3-B0 (No ZUPT):        MAE = {res_b0['mae_ms']:6.2f} m/s, PosErr = {res_b0['final_pos_err_m']:9.1f} m (Drift: {res_b0['drift_pct']:6.1f}%)")
        print(f"  C3-B1 (Raw IMU ZUPT):   MAE = {res_b1['mae_ms']:6.2f} m/s, PosErr = {res_b1['final_pos_err_m']:9.1f} m (Drift: {res_b1['drift_pct']:6.1f}%)")
        print(f"  C3-B2 (Oracle Valid):   MAE = {res_b2['mae_ms']:6.2f} m/s, PosErr = {res_b2['final_pos_err_m']:9.1f} m (Drift: {res_b2['drift_pct']:6.1f}%)")

    # Get full trip runs for plotting
    full_b0 = run_kinematic_integration(time_s, ax_level, v_true, is_stat_imu, t_start=0.0, duration_s=float(n*DT-0.1), use_zupt=False)
    full_b1 = run_kinematic_integration(time_s, ax_level, v_true, is_stat_imu, t_start=0.0, duration_s=float(n*DT-0.1), use_zupt=True)
    full_b2 = run_kinematic_integration(time_s, ax_level, v_true, ref_mask, t_start=0.0, duration_s=float(n*DT-0.1), use_zupt=True)
    
    # =======================================================================
    # PLOTTING: 5 REQUIRED FIGURES (UPDATED WITH ORACLE COMPARISON)
    # =======================================================================
    print("\n" + "=" * 90)
    print("GENERATING 5 REQUIRED DIAGNOSTIC FIGURES (WITH ORACLE C3-B2)")
    print("=" * 90)
    
    plt.style.use('default')
    
    # -----------------------------------------------------------------------
    # FIGURE 1: Speed Comparison + Detections
    # -----------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(full_b0['t_win'], full_b0['v_true'], label='VBOX Ground Truth Speed', color='black', lw=1.5, zorder=5)
    ax.plot(full_b0['t_win'], full_b0['v_est'], label='C3-B0: No ZUPT (Integrated)', color='#e74c3c', lw=1.2, alpha=0.8, zorder=2)
    ax.plot(full_b1['t_win'], full_b1['v_est'], label='C3-B1: Raw IMU-Triggered ZUPT', color='#2ecc71', lw=1.3, zorder=4)
    ax.plot(full_b2['t_win'], full_b2['v_est'], label='C3-B2: Oracle Valid ZUPT (True Stops Only)', color='#3498db', lw=1.3, linestyle='--', zorder=3)
    
    ax.fill_between(time_s, 0, np.max(v_true) * 1.1, where=is_stat_imu,
                    color='#2ecc71', alpha=0.15, label='IMU Detected ZUPT (Active)', zorder=1)
    ax.fill_between(time_s, 0, np.max(v_true) * 1.1, where=ref_mask,
                    color='blue', alpha=0.15, label='VBOX True Stop ($v < 0.10$ m/s)', zorder=1)
    
    ax.set_title('Stage C5.2-C3: VBOX Speed vs Kinematic Integration (No ZUPT vs Raw ZUPT vs Oracle Valid ZUPT)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Speed (m/s)', fontsize=10)
    ax.set_ylim(-2, max(v_true.max() * 1.1, 35))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    fig1.tight_layout()
    fig1_path = FIG_DIR / "c5_2c3_speed_and_detections.png"
    fig1.savefig(fig1_path)
    plt.close(fig1)
    print(f"  Saved Figure 1: {fig1_path.name}")
    
    # -----------------------------------------------------------------------
    # FIGURE 2: IMU Signals with Thresholds
    # -----------------------------------------------------------------------
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=300)
    
    ax_top.plot(time_s, a_std_trace, color='#8e44ad', lw=1.0, label=r'Rolling $\mathrm{std}(\|\mathbf{a}\|)$ ($W = 2.0\ \mathrm{s}$)')
    ax_top.axhline(ACCEL_STD_THRESH, color='red', linestyle='--', lw=1.5, label=f'Threshold = {ACCEL_STD_THRESH} m/s²')
    ax_top.fill_between(time_s, 0, 1.5, where=ref_mask, color='blue', alpha=0.15, label='VBOX True Stop ($v < 0.10$ m/s)')
    ax_top.fill_between(time_s, 0, 1.5, where=(is_stat_imu & ~ref_mask), color='red', alpha=0.15, label='False Positive / Premature Detection')
    ax_top.set_ylabel(r'$\mathrm{std}(\|\mathbf{a}\|)$ (m/s²)', fontsize=10)
    ax_top.set_ylim(0, 1.2)
    ax_top.set_title('Stage C5.2-C3: IMU Stationarity Signals, Thresholds, and False-Positive Regions', fontsize=11, fontweight='bold')
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(loc='upper right', fontsize=8)
    
    ax_bot.plot(time_s, g_mean_trace, color='#e67e22', lw=1.0, label=r'Rolling $\mathrm{mean}(\|\boldsymbol{\omega}\|)$ ($W = 2.0\ \mathrm{s}$)')
    ax_bot.axhline(GYRO_NORM_THRESH, color='red', linestyle='--', lw=1.5, label=f'Threshold = {GYRO_NORM_THRESH} rad/s (2.86°/s)')
    ax_bot.fill_between(time_s, 0, 0.5, where=ref_mask, color='blue', alpha=0.15, label='VBOX True Stop ($v < 0.10$ m/s)')
    ax_bot.fill_between(time_s, 0, 0.5, where=(is_stat_imu & ~ref_mask), color='red', alpha=0.15, label='False Positive / Premature Detection')
    ax_bot.set_ylabel(r'$\mathrm{mean}(\|\boldsymbol{\omega}\|)$ (rad/s)', fontsize=10)
    ax_bot.set_xlabel('Time (s)', fontsize=10)
    ax_bot.set_ylim(0, 0.35)
    ax_bot.grid(True, linestyle='--', alpha=0.5)
    ax_bot.legend(loc='upper right', fontsize=8)
    
    fig2.tight_layout()
    fig2_path = FIG_DIR / "c5_2c3_imu_thresholds.png"
    fig2.savefig(fig2_path)
    plt.close(fig2)
    print(f"  Saved Figure 2: {fig2_path.name}")
    
    # -----------------------------------------------------------------------
    # FIGURE 3: Velocity Error Comparison
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 5), dpi=300)
    ax3.plot(full_b0['t_win'], full_b0['err'], label='C3-B0: No ZUPT Velocity Error', color='#e74c3c', lw=1.2)
    ax3.plot(full_b1['t_win'], full_b1['err'], label='C3-B1: Raw IMU ZUPT Velocity Error', color='#2ecc71', lw=1.2)
    ax3.plot(full_b2['t_win'], full_b2['err'], label='C3-B2: Oracle Valid ZUPT Velocity Error', color='#3498db', lw=1.2, linestyle='--')
    ax3.axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.5)
    
    for ep in ref_episodes:
        t_mid = 0.5 * (time_s[ep['start_idx']] + time_s[ep['end_idx']])
        ax3.axvline(t_mid, color='blue', linestyle=':', alpha=0.4)
        
    ax3.set_title('Stage C5.2-C3: Velocity Error vs Time (Decomposition of Raw vs Oracle ZUPT)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Velocity Error (m/s)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', frameon=True, fontsize=8)
    fig3.tight_layout()
    fig3_path = FIG_DIR / "c5_2c3_velocity_error_comparison.png"
    fig3.savefig(fig3_path)
    plt.close(fig3)
    print(f"  Saved Figure 3: {fig3_path.name}")
    
    # -----------------------------------------------------------------------
    # FIGURE 4: Position Error Comparison
    # -----------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 5), dpi=300)
    ax4.plot(full_b0['t_win'], full_b0['pos_err'] / 1000.0, label='C3-B0: No ZUPT (km)', color='#e74c3c', lw=1.4)
    ax4.plot(full_b1['t_win'], full_b1['pos_err'] / 1000.0, label='C3-B1: Raw IMU ZUPT (km)', color='#2ecc71', lw=1.4)
    ax4.plot(full_b2['t_win'], full_b2['pos_err'] / 1000.0, label='C3-B2: Oracle Valid ZUPT (km)', color='#3498db', lw=1.4, linestyle='--')
    ax4.axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.5)
    
    ax4.set_title('Stage C5.2-C3: Longitudinal Position Error vs Time (km)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Position Error (km)', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left', frameon=True, fontsize=8)
    fig4.tight_layout()
    fig4_path = FIG_DIR / "c5_2c3_position_error_comparison.png"
    fig4.savefig(fig4_path)
    plt.close(fig4)
    print(f"  Saved Figure 4: {fig4_path.name}")
    
    # -----------------------------------------------------------------------
    # FIGURE 5: Stop Event Zooms (Early Stop & Major Intersection Stop)
    # -----------------------------------------------------------------------
    fig5, (ax_z1, ax_z2) = plt.subplots(2, 1, figsize=(12, 7), dpi=300)
    
    # Zoom 1: Early Stops (t = 10 to 30 s)
    mask_z1 = (time_s >= 10.0) & (time_s <= 30.0)
    ax_z1.plot(time_s[mask_z1], v_true[mask_z1], label='VBOX Ground Truth', color='black', lw=1.8, zorder=3)
    ax_z1.plot(time_s[mask_z1], full_b0['v_est'][mask_z1], label='C3-B0: No ZUPT', color='#e74c3c', linestyle='--', lw=1.5)
    ax_z1.plot(time_s[mask_z1], full_b1['v_est'][mask_z1], label='C3-B1: Raw IMU ZUPT', color='#2ecc71', lw=1.8, zorder=4)
    ax_z1.plot(time_s[mask_z1], full_b2['v_est'][mask_z1], label='C3-B2: Oracle Valid ZUPT', color='#3498db', linestyle=':', lw=1.8, zorder=4)
    ax_z1.fill_between(time_s[mask_z1], 0, 5, where=is_stat_imu[mask_z1], color='#2ecc71', alpha=0.2, label='IMU ZUPT Active')
    ax_z1.fill_between(time_s[mask_z1], 0, 5, where=ref_mask[mask_z1], color='blue', alpha=0.15, label='VBOX True Stop')
    ax_z1.set_title('Zoom Event 1: Early Stops (t = 10 to 30 s) — Note Premature Trigger at t = 13.1 s vs True Stop at 14.4 s', fontsize=10, fontweight='bold')
    ax_z1.set_ylabel('Speed (m/s)', fontsize=9)
    ax_z1.set_ylim(-0.5, 4.0)
    ax_z1.grid(True, linestyle='--', alpha=0.5)
    ax_z1.legend(loc='upper right', fontsize=8)
    
    # Zoom 2: Major Intersection Stop (t = 700 to 765 s)
    mask_z2 = (time_s >= 700.0) & (time_s <= 765.0)
    ax_z2.plot(time_s[mask_z2], v_true[mask_z2], label='VBOX Ground Truth', color='black', lw=1.8, zorder=3)
    ax_z2.plot(time_s[mask_z2], full_b0['v_est'][mask_z2], label='C3-B0: No ZUPT', color='#e74c3c', linestyle='--', lw=1.5)
    ax_z2.plot(time_s[mask_z2], full_b1['v_est'][mask_z2], label='C3-B1: Raw IMU ZUPT', color='#2ecc71', lw=1.8, zorder=4)
    ax_z2.plot(time_s[mask_z2], full_b2['v_est'][mask_z2], label='C3-B2: Oracle Valid ZUPT', color='#3498db', linestyle=':', lw=1.8, zorder=4)
    ax_z2.fill_between(time_s[mask_z2], -2, 16, where=is_stat_imu[mask_z2], color='#2ecc71', alpha=0.2, label='IMU ZUPT Active')
    ax_z2.fill_between(time_s[mask_z2], -2, 16, where=ref_mask[mask_z2], color='blue', alpha=0.15, label='VBOX True Stop')
    ax_z2.set_title('Zoom Event 2: Major Stop 3 (t = 700 to 765 s) — Note Premature Reset at t = 704.6 s (+2.7 s early) at v = 1.50 m/s', fontsize=10, fontweight='bold')
    ax_z2.set_ylabel('Speed (m/s)', fontsize=9)
    ax_z2.set_xlabel('Time (s)', fontsize=10)
    ax_z2.set_ylim(-2.0, 16.0)
    ax_z2.grid(True, linestyle='--', alpha=0.5)
    ax_z2.legend(loc='upper right', fontsize=8)
    
    fig5.tight_layout()
    fig5_path = FIG_DIR / "c5_2c3_stop_event_zooms.png"
    fig5.savefig(fig5_path)
    plt.close(fig5)
    print(f"  Saved Figure 5: {fig5_path.name}")
    
    # =======================================================================
    # SAVE JSON RESULTS
    # =======================================================================
    output_data = {
        'trip': 'Vta02',
        'sample_count': n,
        'duration_s': float(n * DT),
        'C3_A_detector_validation': {
            'thresholds': {
                'window_samples': W_SAMPLES,
                'window_s': float(W_SAMPLES * DT),
                'accel_std_thresh_ms2': ACCEL_STD_THRESH,
                'gyro_norm_thresh_rads': GYRO_NORM_THRESH,
                'causal': True,
            },
            'reference_criteria': {
                'vbox_stop_thresh_ms': VBOX_STOP_THRESH,
                'min_duration_samples': MIN_STOP_DUR_SAMPLES,
                'min_duration_s': float(MIN_STOP_DUR_SAMPLES * DT),
            },
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'iou': iou,
                'accuracy': accuracy,
            },
            'reference_episodes': ref_episodes,
            'zupt_episodes_audit': episodes_c3c,
        },
        'C3_B_and_C_benchmarks': benchmark_results
    }
    
    json_path = RES_DIR / "c5_2c3_zupt_benchmark.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved benchmark metrics to {json_path}")
    print("\n[STAGE C5.2-C3 BENCHMARK & AUDIT COMPLETE]")


if __name__ == "__main__":
    main()
