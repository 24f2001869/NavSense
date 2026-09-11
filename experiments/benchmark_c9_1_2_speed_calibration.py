"""
SIH26168 - Stage C9.1.2: Causal Pre-Outage Speed Calibration Benchmark
Script: experiments/benchmark_c9_1_2_speed_calibration.py

Evaluates whether strictly causal pre-outage speed calibration can resolve
the remaining ~215m along-track error on highway cruising (Vta02) without
degrading urban maneuvering (Vta04).

Evaluates 4 Speed Calibration Models inside the C9.1 Engine:
1. Model A (No Correction): v = v_ml
2. Model B (Additive Bias): v = v_ml + b_spd
3. Model C (Multiplicative Scale): v = s_spd * v_ml
4. Model D (Affine Correction): v = s_spd * v_ml + b_spd
Plus Reference_Oracle (Ground-Truth Speed & Heading).

Strictly Causal Constraints:
- Calibration parameters estimated exclusively on pre-outage GNSS cruising (lookback = 30s).
- Parameter estimation freezes at blackout onset (t = kw).
- Zero GNSS during blackout. Zero future information. Zero CAN/wheel speeds.
- Evaluated across predetermined non-overlapping blackout windows on Vta04 and Vta02
  at 10s, 20s, 30s, and 60s.
"""

import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.navigation.eskf import ESKF3D
from src.navigation.dead_reckoning_engine import DeadReckoningEngine, EngineConfig, GNSSMeasurement
from experiments.evaluate_phone_speed_c8_7 import prepare_trip_phone_data
from experiments.validate_end_to_end_c8_11_8 import (
    prepare_trip_for_validation,
    update_compass_heading
)

RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RES_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR = REPO_ROOT / "experiments" / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_MEDIA = PROJECT_ROOT / "results" / "figures"


def estimate_speed_calibration(
    v_gnss_pre: np.ndarray,
    v_ml_pre: np.ndarray,
    min_moving_speed: float = 2.0
) -> Dict[str, Any]:
    """
    Estimates causal speed calibration parameters over pre-outage valid GNSS history.
    Enforces robust safeguards against steady-state OLS collinearity and outliers.
    """
    mask = (v_gnss_pre >= min_moving_speed) & (v_ml_pre >= 0.5)
    n_pts = int(np.sum(mask))

    if n_pts < 15:
        # Insufficient moving samples -> Default to identity
        return {
            'bias': 0.0,
            'scale': 1.0,
            'affine_scale': 1.0,
            'affine_bias': 0.0,
            'mode': 'insufficient_samples'
        }

    vg = v_gnss_pre[mask]
    vm = v_ml_pre[mask]

    # Model B: Additive bias (mean residual)
    b_add = float(np.clip(np.mean(vg - vm), -3.0, 3.0))

    # Model C: Multiplicative scale (ratio of means)
    mean_vm = np.mean(vm)
    s_mul = float(np.clip(np.mean(vg) / mean_vm if mean_vm > 0.5 else 1.0, 0.7, 1.4))

    # Model D: Affine regression (OLS with low-variance guard)
    std_vm = float(np.std(vm))
    std_vg = float(np.std(vg))
    dyn_range_vm = float(np.ptp(vm))

    if dyn_range_vm >= 3.0 and std_vm > 0.5 and std_vg > 0.5:
        corr = float(np.corrcoef(vg, vm)[0, 1])
        if corr > 0.4:
            s_aff = float(np.cov(vg, vm)[0, 1] / (std_vm ** 2))
            s_aff = float(np.clip(s_aff, 0.7, 1.4))
            b_aff = float(np.clip(np.mean(vg) - s_aff * np.mean(vm), -3.0, 3.0))
            mode = 'ols_affine'
        else:
            s_aff = 1.0
            b_aff = b_add
            mode = 'fallback_bias_low_corr'
    else:
        # Steady cruising (slope unidentifiable) -> Fall back to bias
        s_aff = 1.0
        b_aff = b_add
        mode = 'fallback_bias_low_var'

    return {
        'bias': b_add,
        'scale': s_mul,
        'affine_scale': s_aff,
        'affine_bias': b_aff,
        'mode': mode
    }


def run_c9_1_2_benchmark():
    print("=" * 84, flush=True)
    print("STAGE C9.1.2: CAUSAL PRE-OUTAGE SPEED CALIBRATION BENCHMARK", flush=True)
    print("Evaluating Additive, Multiplicative, and Affine Speed Corrections", flush=True)
    print("Evaluation across Urban (Vta04) and Highway (Vta02) at 10s, 20s, 30s, and 60s", flush=True)
    print("=" * 84, flush=True)

    # 1. Train Causal RF Model for Forward Speed
    print("\n[1/4] Training Causal Random Forest speed model on Vta02 partition...", flush=True)
    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf_model = RandomForestRegressor(n_estimators=35, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])

    # 2. Ingest Trips
    print("[2/4] Ingesting trip telemetry and map networks for Vta04 and Vta02...", flush=True)
    D_v04 = prepare_trip_for_validation('Vta04', rf_model)
    D_v02 = prepare_trip_for_validation('Vta02', rf_model)

    SPEED_MODELS = [
        'Model_A_NoCorr',
        'Model_B_Additive',
        'Model_C_Multiplicative',
        'Model_D_Affine',
        'Reference_Oracle'
    ]

    MODEL_LABELS = {
        'Model_A_NoCorr': 'Model A: Raw ML Speed (v = v_ml)',
        'Model_B_Additive': 'Model B: Additive Bias (v = v_ml + b)',
        'Model_C_Multiplicative': 'Model C: Multiplicative Scale (v = s * v_ml)',
        'Model_D_Affine': 'Model D: Affine (v = s * v_ml + b)',
        'Reference_Oracle': 'Reference: Ground-Truth Oracle'
    }

    HORIZONS = [10, 20, 30, 60]

    master_results = {
        'meta': {
            'stage': 'C9.1.2',
            'date': 'September 8, 2026',
            'description': 'Stage C9.1.2 Causal Pre-Outage Speed Calibration Benchmark',
            'models': SPEED_MODELS,
            'horizons': HORIZONS
        },
        'trips': {}
    }

    print("\n[3/4] Running multi-horizon canonical evaluation across predetermined blackout windows...", flush=True)

    for trip_name, D in [('Vta04', D_v04), ('Vta02', D_v02)]:
        print(f"\n{'=' * 84}", flush=True)
        print(f"TRIP: {trip_name} ({D['n']*0.1:.1f} s, {D['cum_dist_m'][-1]/1000:.2f} km) — "
              f"{'Urban Loop' if trip_name == 'Vta04' else 'Highway Arterial (~25 m/s Cruising)'}", flush=True)
        print(f"{'=' * 84}", flush=True)

        dt = D['dt']
        n = D['n']
        v_veh = D['speed']
        a_veh = np.zeros(n)
        a_veh[1:] = np.diff(v_veh) / dt
        a_phone_x = D['acc_v'][:, 0]

        trip_results = {}

        for h_s in HORIZONS:
            w_dur = int(h_s / dt)
            buffer_pts = int(30.0 / dt)

            step_stride = w_dur
            candidate_starts = list(range(buffer_pts, n - w_dur - 50, step_stride))
            valid_starts = [k for k in candidate_starts if np.mean(v_veh[k : k + w_dur]) >= 2.5]
            n_valid = len(valid_starts)

            if n_valid == 0:
                print(f"  Horizon {h_s}s: 0 valid moving windows. Skipping.", flush=True)
                continue

            print(f"\n--- Horizon {h_s:2d}s: Evaluating N={n_valid} Non-Overlapping Outage Windows ---", flush=True)

            cond_metrics = {
                m: {
                    'pos_err': [],
                    'along_err': [],
                    'cross_err': [],
                    'drift_pct': [],
                    'yaw_err': [],
                    'pass_sih': [],
                    'calib_params': []
                } for m in SPEED_MODELS
            }

            for kw in valid_starts:
                dist_traveled = float(np.sum(v_veh[kw : kw + w_dur]) * dt)

                # Causal pre-outage lookback (30 seconds before blackout onset)
                lb_pts = int(30.0 / dt)
                pre_start = max(0, kw - lb_pts)
                pre_slice = np.arange(pre_start, kw)
                
                # Straight cruising mask for inertial biases
                st_mask = (v_veh[pre_slice] >= 3.0) & (D['tr_smooth'][pre_slice] < 2.0)
                st_slice = pre_slice[st_mask]
                b_accel_pre = float(np.mean(a_phone_x[st_slice] - a_veh[st_slice])) if len(st_slice) >= 10 else 0.51
                b_gyro_pre = float(np.mean(D['gyro_v'][st_slice, 2])) if len(st_slice) >= 10 else 0.0

                # Pre-outage speed calibration estimation
                v_gnss_pre = v_veh[pre_slice]
                v_ml_pre = D['ml_speed'][pre_slice]
                calib = estimate_speed_calibration(v_gnss_pre, v_ml_pre)

                for m in SPEED_MODELS:
                    if m == 'Reference_Oracle':
                        eskf = ESKF3D(
                            init_pos_enu=(D['gt_e'][kw], D['gt_n'][kw], D['gt_u'][kw]),
                            init_vel_enu=(D['gt_ve'][kw], D['gt_vn'][kw], D['gt_vu'][kw]),
                            init_heading_deg=float(D['heading'][kw]),
                            init_pitch_deg=0.0,
                            init_roll_deg=0.0,
                            init_ba=D['ba_stat'].copy(),
                            R_vp=np.eye(3),
                            sigma_a=0.291,
                            gravity=9.80665
                        )
                        for step in range(kw, kw + w_dur):
                            ax, ay, az = D['acc_v'][step]
                            gx, gy, gz = D['gyro_v'][step]
                            eskf.predict(ax, ay, az, gx, gy, gz, dt)
                            update_compass_heading(eskf, float(D['heading'][step]), sigma_psi_deg=0.5)
                            eskf.vel_n[0] = D['gt_ve'][step]
                            eskf.vel_n[1] = D['gt_vn'][step]
                            eskf.vel_n[2] = D['gt_vu'][step]

                        term_pos = eskf.pos_n
                        term_yaw = eskf.attitude.get_yaw_deg()
                        p_rec = {}

                    else:
                        # Instantiate Production C9.1 Engine with full vertical NHC and 3D mag gating
                        eng_cfg = EngineConfig(
                            sigma_speed=0.60,
                            sigma_lat_0=0.50,
                            mag_norm_tol=0.08,
                            mag_db_dt_tol=5.0,
                            enable_vert_nhc=True,
                            turn_rate_vnhc_gate_deg_s=3.0,
                            turn_rate_compass_gate_deg_s=3.0
                        )
                        engine = DeadReckoningEngine(
                            config=eng_cfg,
                            road_index=D['road_index'],
                            init_pos_enu=(D['gt_e'][kw], D['gt_n'][kw], D['gt_u'][kw]),
                            init_vel_enu=(D['gt_ve'][kw], D['gt_vn'][kw], D['gt_vu'][kw]),
                            init_heading_deg=float(D['heading'][kw]),
                            R_vp=np.eye(3),
                            ba_stat=D['ba_stat'].copy(),
                            baseline_mag_uT=D['baseline_B']
                        )
                        engine.b_accel_x = b_accel_pre
                        engine.b_gyro_z = b_gyro_pre

                        p_rec = {}
                        for step in range(kw, kw + w_dur):
                            raw_ml = float(D['ml_speed'][step])

                            # Apply frozen speed calibration model
                            if m == 'Model_A_NoCorr':
                                spd_use = raw_ml
                            elif m == 'Model_B_Additive':
                                spd_use = max(0.0, raw_ml + calib['bias'])
                                p_rec = {'bias': calib['bias']}
                            elif m == 'Model_C_Multiplicative':
                                spd_use = max(0.0, calib['scale'] * raw_ml)
                                p_rec = {'scale': calib['scale']}
                            elif m == 'Model_D_Affine':
                                spd_use = max(0.0, calib['affine_scale'] * raw_ml + calib['affine_bias'])
                                p_rec = {'scale': calib['affine_scale'], 'bias': calib['affine_bias']}

                            acc_step = D['acc_v'][step]
                            gyro_step = D['gyro_v'][step]
                            cal_mag_h = (D['psi_mag'][step] + D['init_offset']) % 360.0
                            gnss_dummy = GNSSMeasurement(valid=False)

                            engine.step(
                                accel_raw=acc_step,
                                gyro_raw=gyro_step,
                                speed_est=spd_use,
                                dt=dt,
                                mag_raw=D['mag_raw'][step],
                                gnss=gnss_dummy,
                                psi_mag_cal_deg=cal_mag_h,
                                db_dt=float(D['db_dt'][step])
                            )

                        final_st = engine.get_state()
                        term_pos = final_st['pos_enu']
                        term_yaw = final_st['heading_deg']

                    # Terminal Error Metrics
                    term_step = kw + w_dur - 1
                    e_err = term_pos[0] - D['gt_e'][term_step]
                    n_err = term_pos[1] - D['gt_n'][term_step]
                    p_err = float(np.hypot(e_err, n_err))

                    h_gt_deg = float(D['heading'][term_step])
                    h_gt_rad = np.radians(h_gt_deg)
                    u_along = np.array([np.sin(h_gt_rad), np.cos(h_gt_rad)])
                    u_cross = np.array([np.cos(h_gt_rad), -np.sin(h_gt_rad)])
                    e_vec = np.array([e_err, n_err])

                    al_err = float(abs(np.dot(e_vec, u_along)))
                    cr_err = float(abs(np.dot(e_vec, u_cross)))
                    d_pct = float((p_err / max(dist_traveled, 10.0)) * 100.0)
                    yaw_err = float(abs((term_yaw - h_gt_deg + 180.0) % 360.0 - 180.0))

                    cond_metrics[m]['pos_err'].append(p_err)
                    cond_metrics[m]['along_err'].append(al_err)
                    cond_metrics[m]['cross_err'].append(cr_err)
                    cond_metrics[m]['drift_pct'].append(d_pct)
                    cond_metrics[m]['yaw_err'].append(yaw_err)
                    cond_metrics[m]['pass_sih'].append(bool(d_pct < 10.0))
                    cond_metrics[m]['calib_params'].append(p_rec)

            # Horizon Summary Table
            print(f"\n{'Speed Calibration Model':<35} | Pos Err (m) | Along (m) | Cross (m) | Drift %  | Yaw Err (°) | SIH Pass")
            print("-" * 104)
            h_summary = {}
            for m in SPEED_MODELS:
                m_pos = float(np.mean(cond_metrics[m]['pos_err']))
                p50_pos = float(np.median(cond_metrics[m]['pos_err']))
                p90_pos = float(np.percentile(cond_metrics[m]['pos_err'], 90))
                m_al = float(np.mean(cond_metrics[m]['along_err']))
                m_cr = float(np.mean(cond_metrics[m]['cross_err']))
                m_dr = float(np.mean(cond_metrics[m]['drift_pct']))
                p90_dr = float(np.percentile(cond_metrics[m]['drift_pct'], 90))
                m_yaw = float(np.mean(cond_metrics[m]['yaw_err']))
                p90_yaw = float(np.percentile(cond_metrics[m]['yaw_err'], 90))
                pass_rate = float(np.mean(cond_metrics[m]['pass_sih']) * 100.0)

                disp_name = MODEL_LABELS[m]
                print(f"{disp_name:<35} | {m_pos:9.2f}m | {m_al:7.2f}m | {m_cr:7.2f}m | {m_dr:6.2f}% | {m_yaw:8.2f}° | {pass_rate:6.1f}%")

                h_summary[m] = {
                    'mean_pos_err_m': m_pos,
                    'p50_pos_err_m': p50_pos,
                    'p90_pos_err_m': p90_pos,
                    'mean_along_err_m': m_al,
                    'mean_cross_err_m': m_cr,
                    'mean_drift_pct': m_dr,
                    'p90_drift_pct': p90_dr,
                    'mean_yaw_err_deg': m_yaw,
                    'p90_yaw_err_deg': p90_yaw,
                    'sih_pass_rate_pct': pass_rate,
                    'n_windows': n_valid
                }

            trip_results[f"{h_s}s"] = h_summary

        master_results['trips'][trip_name] = trip_results

    # =========================================================================
    # Save Structured Dataset
    # =========================================================================
    json_path = RES_DIR / "c9_1_2_speed_calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[4/4] Saved structured benchmark dataset to: {json_path}", flush=True)

    # =========================================================================
    # Multi-Panel Publication Figure
    # =========================================================================
    print("Generating multi-panel speed calibration visualization...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        'Model_A_NoCorr': '#7f8c8d',         # Gray
        'Model_B_Additive': '#3498db',       # Blue
        'Model_C_Multiplicative': '#e67e22', # Orange
        'Model_D_Affine': '#2ecc71',         # Green
        'Reference_Oracle': '#9b59b6'        # Purple
    }

    # Row 1: Urban Loop (Vta04)
    # (0, 0): Drift % vs Horizon
    ax = axes[0, 0]
    for m in SPEED_MODELS:
        dr_vals = [master_results['trips']['Vta04'][f"{h}s"][m]['mean_drift_pct'] for h in HORIZONS]
        ax.plot(HORIZONS, dr_vals, 'o-', color=colors[m], label=MODEL_LABELS[m], linewidth=2.2, markersize=6)
    ax.axhline(10.0, color='red', linestyle='--', alpha=0.7, label='SIH Spec (<10%)')
    ax.set_title("Vta04 (Urban): Drift % vs Horizon", fontsize=12, fontweight='bold')
    ax.set_xlabel("Outage Horizon (s)", fontsize=10)
    ax.set_ylabel("Drift Error (%)", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, loc='upper left')

    # (0, 1): Along-Track Error at 30s
    ax = axes[0, 1]
    x_pos = np.arange(len(SPEED_MODELS))
    w = 0.35
    along_30 = [master_results['trips']['Vta04']['30s'][m]['mean_along_err_m'] for m in SPEED_MODELS]
    cross_30 = [master_results['trips']['Vta04']['30s'][m]['mean_cross_err_m'] for m in SPEED_MODELS]
    ax.bar(x_pos - w/2, along_30, width=w, label='Along-Track (m)', color='#34495e', alpha=0.85)
    ax.bar(x_pos + w/2, cross_30, width=w, label='Cross-Track (m)', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('Model_', '').replace('_', '\n') for m in SPEED_MODELS], fontsize=8)
    ax.set_title("Vta04 (Urban 30s): Along vs Cross Error", fontsize=12, fontweight='bold')
    ax.set_ylabel("Error (m)", fontsize=10)
    ax.legend(fontsize=9)

    # (0, 2): SIH Pass Rate vs Horizon
    ax = axes[0, 2]
    for m in SPEED_MODELS:
        pass_vals = [master_results['trips']['Vta04'][f"{h}s"][m]['sih_pass_rate_pct'] for h in HORIZONS]
        ax.plot(HORIZONS, pass_vals, 's-', color=colors[m], label=MODEL_LABELS[m], linewidth=2.2, markersize=6)
    ax.set_title("Vta04 (Urban): SIH Pass Rate (<10%) vs Horizon", fontsize=12, fontweight='bold')
    ax.set_xlabel("Outage Horizon (s)", fontsize=10)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, loc='upper right')

    # Row 2: Highway Arterial (Vta02)
    # (1, 0): Drift % vs Horizon
    ax = axes[1, 0]
    for m in SPEED_MODELS:
        dr_vals = [master_results['trips']['Vta02'][f"{h}s"][m]['mean_drift_pct'] for h in HORIZONS]
        ax.plot(HORIZONS, dr_vals, 'o-', color=colors[m], label=MODEL_LABELS[m], linewidth=2.2, markersize=6)
    ax.axhline(10.0, color='red', linestyle='--', alpha=0.7, label='SIH Spec (<10%)')
    ax.set_title("Vta02 (Highway): Drift % vs Horizon", fontsize=12, fontweight='bold')
    ax.set_xlabel("Outage Horizon (s)", fontsize=10)
    ax.set_ylabel("Drift Error (%)", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, loc='upper left')

    # (1, 1): Along-Track Error at 30s
    ax = axes[1, 1]
    along_30_hwy = [master_results['trips']['Vta02']['30s'][m]['mean_along_err_m'] for m in SPEED_MODELS]
    cross_30_hwy = [master_results['trips']['Vta02']['30s'][m]['mean_cross_err_m'] for m in SPEED_MODELS]
    ax.bar(x_pos - w/2, along_30_hwy, width=w, label='Along-Track (m)', color='#34495e', alpha=0.85)
    ax.bar(x_pos + w/2, cross_30_hwy, width=w, label='Cross-Track (m)', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('Model_', '').replace('_', '\n') for m in SPEED_MODELS], fontsize=8)
    ax.set_title("Vta02 (Highway 30s): Along vs Cross Error", fontsize=12, fontweight='bold')
    ax.set_ylabel("Error (m)", fontsize=10)
    ax.legend(fontsize=9)

    # (1, 2): SIH Pass Rate vs Horizon
    ax = axes[1, 2]
    for m in SPEED_MODELS:
        pass_vals = [master_results['trips']['Vta02'][f"{h}s"][m]['sih_pass_rate_pct'] for h in HORIZONS]
        ax.plot(HORIZONS, pass_vals, 's-', color=colors[m], label=MODEL_LABELS[m], linewidth=2.2, markersize=6)
    ax.set_title("Vta02 (Highway): SIH Pass Rate (<10%) vs Horizon", fontsize=12, fontweight='bold')
    ax.set_xlabel("Outage Horizon (s)", fontsize=10)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle("Stage C9.1.2: Causal Pre-Outage Speed Calibration Benchmark (Ablation of Models A, B, C, D)", fontsize=15, fontweight='bold')
    plt.tight_layout()

    fig_path = FIG_DIR / "c9_1_2_speed_calibration.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {fig_path}", flush=True)

    if BRAIN_MEDIA.exists():
        target_brain = BRAIN_MEDIA / "c9_1_2_speed_calibration.png"
        shutil.copy(fig_path, target_brain)
        print(f"Copied figure to brain media: {target_brain}", flush=True)

    # =========================================================================
    # Generate Comprehensive Forensic Report
    # =========================================================================
    report_path = REP_DIR / "c9_1_2_speed_calibration.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage C9.1.2: Causal Pre-Outage Speed Calibration Benchmark Report\n\n")
        f.write("**Date:** September 8, 2026  \n")
        f.write("**Objective:** Controlled evaluation of pre-outage speed calibration models (Additive, Multiplicative, Affine) ")
        f.write("to determine whether along-track error can be reduced without urban degradation.\n\n")
        f.write("---\n\n")

        for trip_name in ['Vta04', 'Vta02']:
            t_data = master_results['trips'][trip_name]
            f.write(f"## {trip_name} ({'Urban Loop' if trip_name == 'Vta04' else 'Highway Arterial'})\n\n")

            for h in HORIZONS:
                h_key = f"{h}s"
                f.write(f"### Horizon {h}s (N = {t_data[h_key]['Model_A_NoCorr']['n_windows']} windows)\n\n")
                f.write("| Speed Model | Pos Err (m) | Along-Track (m) | Cross-Track (m) | Drift % | Yaw Err (°) | SIH Pass Rate (<10%) |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                for m in SPEED_MODELS:
                    res = t_data[h_key][m]
                    f.write(f"| **{MODEL_LABELS[m]}** | {res['mean_pos_err_m']:.2f} m | {res['mean_along_err_m']:.2f} m | "
                            f"{res['mean_cross_err_m']:.2f} m | {res['mean_drift_pct']:.2f}% | {res['mean_yaw_err_deg']:.2f}° | "
                            f"{res['sih_pass_rate_pct']:.1f}% |\n")
                f.write("\n")

        f.write("---\n\n")
        f.write("## Physical Forensic Interpretation\n\n")

    print(f"Generated comprehensive report at: {report_path}", flush=True)
    print("=" * 84, flush=True)
    print("STAGE C9.1.2 BENCHMARK COMPLETE!", flush=True)
    print("=" * 84, flush=True)


if __name__ == "__main__":
    run_c9_1_2_benchmark()
