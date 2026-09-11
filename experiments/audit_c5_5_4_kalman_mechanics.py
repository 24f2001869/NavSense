"""
SIH26168 - Stage C5.5.4 Audit: Kalman Mechanics & Shuffled-Confidence Control
Script: experiments/audit_c5_5_4_kalman_mechanics.py

PURPOSE:
    1. Implementation & Mathematical Audit (WHY does C1 work?):
       Numerically trace the exact chain:
           q(k) -> Q_k -> P_{k|k-1} -> K_NHC -> delta_x (NHC correction)
       across 4 representative vehicle dynamic events on Vta02:
           - Event 1: Normal Cruising
           - Event 2: Vibration Spike
           - Event 3: Severe Braking
           - Event 4: Rough Road
       comparing Baseline A (constant Q) vs Adaptive C1 (dynamic Q) side-by-side.

    2. Shuffled-Confidence Control (Is the timing of q(k) doing the work?):
       Benchmark 4 conditions across 30 s rolling GNSS outage windows on Vta02:
           - A. Raw Baseline: Constant nominal sigma_a = 0.15 m/s^2
           - C0. Constant Inflated: Constant sigma_a with equivalent average noise:
                 sigma_a = sigma_{a,0} * sqrt(1 + 0.1 * mean(q))
           - C1-Shuffled: Randomly permuted q(k) sequence (exact same distribution,
                 mean, and variance, but temporally decorrelated from vehicle dynamics)
           - C1. True Causal Adaptive: Vibration-driven q(k)

DELIVERABLES:
    - results/c5_5_4b_kalman_audit_and_shuffled_control.json
    - results/figures/c5_5_4b_kalman_gain_and_correction_trace.png
    - results/figures/c5_5_4b_shuffled_confidence_ablation.png
    - results/c5_5_4b_kalman_mechanics_report.md
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D
from src.navigation.nhc import NonHolonomicConstraint, compute_nhc_residual_and_jacobian

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80, flush=True)
    print("STAGE C5.5.4 AUDIT: KALMAN MECHANICS & SHUFFLED-CONFIDENCE CONTROL", flush=True)
    print("=" * 80, flush=True)

    # 1. Ingest Data & Setup Ground Truth
    print("\n[1] Ingesting Vta02 Dataset...", flush=True)
    df_p, df_v = load_trip("Vta02")
    n = len(df_p)
    dt = 0.1
    time_s = np.arange(n) * dt

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    can_acc = df_v['veh_accel_long_ms2'].values
    can_lat = df_v['veh_accel_lat_ms2'].values
    veh_heading_arr = df_v['veh_heading_deg'].values

    # Vehicle Alignment
    acc_v, gyro_v, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.mean(acc_v[142:191], axis=0) - np.array([0.0, 0.0, 9.80665])

    # Ground Truth ENU
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    heading_rad = np.radians(veh_heading_arr)
    gt_ve = speed * np.sin(heading_rad)
    gt_vn = speed * np.cos(heading_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6

    # Causal Trailing Vibration Metric q(k)
    print("    Computing causal trailing vibration index q(k)...", flush=True)
    acc_mag = np.linalg.norm(acc_v, axis=1)
    rolling_std = np.zeros(n)
    w_k = 10  # 1.0 s trailing window
    for i in range(n):
        sl = acc_mag[max(0, i - w_k + 1) : i + 1]
        rolling_std[i] = np.std(sl) if len(sl) > 1 else 0.15

    sigma_nom = 0.15
    q_k = np.maximum(0.0, (rolling_std**2 - sigma_nom**2) / (sigma_nom**2))
    mean_q = float(np.mean(q_k))
    print(f"    q(k) summary: Mean={mean_q:.2f}, Median={np.median(q_k):.2f}, Max={np.max(q_k):.2f}", flush=True)

    # 2. Trace Kalman Mechanics across 4 Representative Events
    print("\n[2] Performing Kalman Mechanics Numerical Audit across 4 Dynamic Events...", flush=True)

    # Compute vertical variance for rough road identification
    vert_var = np.zeros(n)
    for i in range(n):
        vert_var[i] = np.var(acc_v[max(0, i - w_k + 1) : i + 1, 2])

    events = {
        'Event 1 (Normal Cruising)': {
            'idx': int(np.where((speed > 8.0) & (np.abs(can_acc) < 0.2) & (q_k < 8.0))[0][100]),
            'desc': 'Steady speed, low acceleration, smooth road'
        },
        'Event 2 (Vibration Spike)': {
            'idx': int(np.where((speed > 5.0) & (q_k > 60.0))[0][10]),
            'desc': 'High vibration burst (q > 60) during forward motion'
        },
        'Event 3 (Severe Braking)': {
            'idx': int(np.argmin(can_acc)),
            'desc': 'Peak longitudinal deceleration (a_CAN = -5.78 m/s²)'
        },
        'Event 4 (Rough Road)': {
            'idx': int(np.where((speed > 5.0) & (vert_var > 0.08))[0][20]),
            'desc': 'High vertical road excitation (vert_var > 0.08 m²/s⁴)'
        }
    }

    kalman_audit_data = {}

    for ev_name, ev_info in events.items():
        k_ev = ev_info['idx']
        t_ev = time_s[k_ev]
        print(f"\n  Tracing {ev_name} at t = {t_ev:.1f} s (Epoch {k_ev}):", flush=True)
        print(f"    Context: Speed={speed[k_ev]:.1f} m/s | a_CAN={can_acc[k_ev]:.2f} m/s² | q(k)={q_k[k_ev]:.2f}", flush=True)

        # To capture steady-state filter convergence, simulate a 5.0 s runup prior to the event
        k_start = max(0, k_ev - 50)

        results_by_cond = {}
        for cond_name, is_adapt in [('A_Raw', False), ('C1_Adaptive', True)]:
            eskf = ESKF3D(
                init_pos_enu=(gt_e[k_start], gt_n[k_start], gt_u[k_start]),
                init_vel_enu=(gt_ve[k_start], gt_vn[k_start], gt_vu[k_start]),
                init_heading_deg=float(veh_heading_arr[k_start]),
                init_ba=ba_stat,
                R_vp=np.eye(3),
                sigma_a=0.15,
                gravity=9.80665
            )
            nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)

            # Step up to event step
            for step_k in range(k_start, k_ev):
                if is_adapt:
                    eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_k[step_k])
                else:
                    eskf.sigma_a = 0.15
                eskf.predict(acc_v[step_k, 0], acc_v[step_k, 1], acc_v[step_k, 2],
                             gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)
                nhc.update_eskf(eskf)

            # At the exact event step k_ev:
            if is_adapt:
                sigma_a_cur = 0.15 * np.sqrt(1.0 + 0.1 * q_k[k_ev])
                eskf.sigma_a = sigma_a_cur
            else:
                sigma_a_cur = 0.15
                eskf.sigma_a = sigma_a_cur

            # Execute PREDICT step
            eskf.predict(acc_v[k_ev, 0], acc_v[k_ev, 1], acc_v[k_ev, 2],
                         gyro_v[k_ev, 0], gyro_v[k_ev, 1], gyro_v[k_ev, 2], dt)

            # Extract Predicted Prior Covariance P_{k|k-1}
            P_prior = eskf.P.copy()
            Q_diag_v = (sigma_a_cur**2) * dt
            Q_diag_p = (1.0 / 3.0) * (sigma_a_cur**2) * (dt**3)

            # Extract NHC Update Quantities
            C_v_n = eskf.attitude.get_dcm()
            vel_n = eskf.vel_n.copy()
            r_nhc, H_nhc, vel_v = compute_nhc_residual_and_jacobian(C_v_n, vel_n)
            R_mat = nhc.R_nhc
            S_mat = H_nhc @ P_prior @ H_nhc.T + R_mat
            K_gain = P_prior @ H_nhc.T @ np.linalg.inv(S_mat)
            delta_x = K_gain @ r_nhc

            # Execute NHC update
            nhc.update_eskf(eskf)
            P_post = eskf.P.copy()

            results_by_cond[cond_name] = {
                'sigma_a': float(sigma_a_cur),
                'sigma_a_sq': float(sigma_a_cur**2),
                'Q_vel_diag': float(Q_diag_v),
                'Q_pos_diag': float(Q_diag_p),
                'P_prior_diag_vel': [float(P_prior[3, 3]), float(P_prior[4, 4]), float(P_prior[5, 5])],
                'P_prior_diag_att': [float(P_prior[6, 6]), float(P_prior[7, 7]), float(P_prior[8, 8])],
                'P_prior_trace_vel': float(np.trace(P_prior[3:6, 3:6])),
                'P_prior_trace_att': float(np.trace(P_prior[6:9, 6:9])),
                'nhc_innovation': [float(r_nhc[0]), float(r_nhc[1])],
                'nhc_innovation_norm': float(np.linalg.norm(r_nhc)),
                'S_det': float(np.linalg.det(S_mat)),
                'K_vel_norm': float(np.linalg.norm(K_gain[3:6, :])),
                'K_att_norm': float(np.linalg.norm(K_gain[6:9, :])),
                'K_total_norm': float(np.linalg.norm(K_gain)),
                'delta_pos_norm': float(np.linalg.norm(delta_x[0:3])),
                'delta_vel_norm': float(np.linalg.norm(delta_x[3:6])),
                'delta_att_norm': float(np.linalg.norm(delta_x[6:9])),
                'delta_x_norm': float(np.linalg.norm(delta_x)),
                'P_post_trace_vel': float(np.trace(P_post[3:6, 3:6]))
            }

        # Side-by-side comparison print
        res_a = results_by_cond['A_Raw']
        res_c = results_by_cond['C1_Adaptive']

        print(f"      {'Parameter':<28} | {'Baseline A (Const Q)':>20} | {'Adaptive C1 (Dyn Q)':>20} | {'Ratio (C1/A)':>12}", flush=True)
        print("      " + "-" * 88, flush=True)
        print(f"      {'sigma_a (m/s²)':<28} | {res_a['sigma_a']:20.4f} | {res_c['sigma_a']:20.4f} | {res_c['sigma_a']/res_a['sigma_a']:12.2f}x", flush=True)
        print(f"      {'Q_vel diagonal (m²/s²)':<28} | {res_a['Q_vel_diag']:20.6f} | {res_c['Q_vel_diag']:20.6f} | {res_c['Q_vel_diag']/res_a['Q_vel_diag']:12.2f}x", flush=True)
        print(f"      {'Prior Tr(P_vel) (m²/s²)':<28} | {res_a['P_prior_trace_vel']:20.6f} | {res_c['P_prior_trace_vel']:20.6f} | {res_c['P_prior_trace_vel']/res_a['P_prior_trace_vel']:12.2f}x", flush=True)
        print(f"      {'NHC Residual ||r|| (m/s)':<28} | {res_a['nhc_innovation_norm']:20.4f} | {res_c['nhc_innovation_norm']:20.4f} | {res_c['nhc_innovation_norm']/(res_a['nhc_innovation_norm']+1e-9):12.2f}x", flush=True)
        print(f"      {'NHC Kalman Gain ||K_vel||':<28} | {res_a['K_vel_norm']:20.4f} | {res_c['K_vel_norm']:20.4f} | {res_c['K_vel_norm']/res_a['K_vel_norm']:12.2f}x", flush=True)
        print(f"      {'NHC Kalman Gain ||K_att||':<28} | {res_a['K_att_norm']:20.4f} | {res_c['K_att_norm']:20.4f} | {res_c['K_att_norm']/(res_a['K_att_norm']+1e-9):12.2f}x", flush=True)
        print(f"      {'Velocity Correction ||dv||':<28} | {res_a['delta_vel_norm']:20.4f} | {res_c['delta_vel_norm']:20.4f} | {res_c['delta_vel_norm']/(res_a['delta_vel_norm']+1e-9):12.2f}x", flush=True)
        print(f"      {'Attitude Correction ||dth||':<28} | {res_a['delta_att_norm']:20.6f} | {res_c['delta_att_norm']:20.6f} | {res_c['delta_att_norm']/(res_a['delta_att_norm']+1e-9):12.2f}x", flush=True)

        kalman_audit_data[ev_name] = {
            'epoch': k_ev,
            'time_s': float(t_ev),
            'speed_ms': float(speed[k_ev]),
            'can_acc_ms2': float(can_acc[k_ev]),
            'q_k': float(q_k[k_ev]),
            'conditions': results_by_cond
        }

    # 3. Shuffled-Confidence & Constant-Inflated Control Experiment
    print("\n[3] Running Shuffled-Confidence & Constant-Inflated Control Benchmark (30s Outages)...", flush=True)
    print("    Hypothesis Test: Does C1 help because of TEMPORALLY LOCALIZED vibration detection,", flush=True)
    print("    or merely because it increases process noise on average?", flush=True)

    # Create shuffled q(k) with fixed random seed
    rng = np.random.default_rng(seed=42)
    q_k_shuffled = rng.permutation(q_k)

    # Constant inflated sigma_a matching average variance
    # sigma_a_const = sigma_nom * sqrt(1 + 0.1 * mean(q))
    sigma_a_const_inflated = float(sigma_nom * np.sqrt(1.0 + 0.1 * mean_q))
    print(f"    Baseline nominal sigma_a:  {sigma_nom:.4f} m/s²", flush=True)
    print(f"    Constant inflated sigma_a: {sigma_a_const_inflated:.4f} m/s² (mean q = {mean_q:.2f})", flush=True)

    h = 30.0
    w_dur = int(round(h / dt))
    stride_s = 20.0
    stride_k = int(round(stride_s / dt))
    window_starts = list(range(100, n - w_dur - 1, stride_k))

    control_conditions = {
        'A_Raw': {
            'label': 'A. Constant Nominal Q (sigma_a = 0.15)',
            'mode': 'constant',
            'sigma_val': 0.15,
            'drifts': [], 'verrs': [], 'regimes': {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []}
        },
        'C0_Constant_Inflated': {
            'label': f'C0. Constant Inflated Q (sigma_a = {sigma_a_const_inflated:.3f})',
            'mode': 'constant',
            'sigma_val': sigma_a_const_inflated,
            'drifts': [], 'verrs': [], 'regimes': {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []}
        },
        'C1_Shuffled': {
            'label': 'C1-Shuffled. Permuted q(k) (Same Distribution, Broken Timing)',
            'mode': 'dynamic_shuffled',
            'drifts': [], 'verrs': [], 'regimes': {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []}
        },
        'C1_Adaptive': {
            'label': 'C1. True Causal Adaptive q(k) (Vibration-Driven)',
            'mode': 'dynamic_true',
            'drifts': [], 'verrs': [], 'regimes': {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []}
        }
    }

    t0_sim = time.time()
    for idx, kw in enumerate(window_starts):
        win_speed = speed[kw : kw + w_dur + 1]
        win_can_a = can_acc[kw : kw + w_dur + 1]
        win_pvert = acc_v[kw : kw + w_dur + 1, 2]

        is_standstill = float(np.mean(win_speed)) < 0.15
        is_braking = float(np.min(win_can_a)) < -1.5
        is_rough = float(np.var(win_pvert)) > 0.05
        is_cruising = (float(np.mean(win_speed)) > 5.0) and (not is_braking) and (not is_rough)

        if is_standstill:
            reg_tag = 'standstill'
        elif is_braking:
            reg_tag = 'braking'
        elif is_rough:
            reg_tag = 'rough_road'
        else:
            reg_tag = 'cruising'

        for ckey, ccfg in control_conditions.items():
            eskf = ESKF3D(
                init_pos_enu=(gt_e[kw], gt_n[kw], gt_u[kw]),
                init_vel_enu=(gt_ve[kw], gt_vn[kw], gt_vu[kw]),
                init_heading_deg=float(veh_heading_arr[kw]),
                init_ba=ba_stat,
                R_vp=np.eye(3),
                sigma_a=0.15,
                gravity=9.80665
            )
            nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)

            for step_k in range(kw, kw + w_dur):
                if ccfg['mode'] == 'constant':
                    eskf.sigma_a = ccfg['sigma_val']
                elif ccfg['mode'] == 'dynamic_shuffled':
                    eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_k_shuffled[step_k])
                elif ccfg['mode'] == 'dynamic_true':
                    eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_k[step_k])

                eskf.predict(acc_v[step_k, 0], acc_v[step_k, 1], acc_v[step_k, 2],
                             gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)
                nhc.update_eskf(eskf)

            st = eskf.get_state()
            gt_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur], gt_u[kw + w_dur]])
            pos_err = float(np.linalg.norm(st['pos_n'][:2] - gt_end[:2]))
            gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur], gt_vu[kw + w_dur]])
            vel_err = float(np.linalg.norm(st['vel_n'][:2] - gt_v_end[:2]))

            ccfg['drifts'].append(pos_err)
            ccfg['verrs'].append(vel_err)
            ccfg['regimes'][reg_tag].append(pos_err)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(window_starts):
            print(f"    Completed {idx+1}/{len(window_starts)} windows ({((idx+1)/len(window_starts))*100:.0f}%) in {time.time()-t0_sim:.1f}s", flush=True)

    # Print Summary Table
    print(f"\n  --- 30s Outage Control Ablation Results (N={len(window_starts)} windows) ---", flush=True)
    print(f"    {'Condition':<35} | {'Mean Drift':>11} | {'Median Drift':>13} | {'P90 Drift':>11} | {'Vel Err':>8}", flush=True)
    print("    " + "-" * 88, flush=True)

    summary_control_export = {}
    for ckey, ccfg in control_conditions.items():
        drifts = np.array(ccfg['drifts'])
        verrs = np.array(ccfg['verrs'])
        m_drift = float(np.mean(drifts))
        med_drift = float(np.median(drifts))
        p90_drift = float(np.percentile(drifts, 90))
        m_verr = float(np.mean(verrs))

        print(f"    {ccfg['label']:<35} | {m_drift:9.2f} m | {med_drift:11.2f} m | {p90_drift:9.2f} m | {m_verr:6.2f} m/s", flush=True)

        summary_control_export[ckey] = {
            'label': ccfg['label'],
            'mean_drift_m': m_drift,
            'median_drift_m': med_drift,
            'std_drift_m': float(np.std(drifts)),
            'p75_drift_m': float(np.percentile(drifts, 75)),
            'p90_drift_m': float(p90_drift),
            'mean_vel_err_ms': m_verr,
            'regime_means_m': {
                'cruising': float(np.mean(ccfg['regimes']['cruising'])) if len(ccfg['regimes']['cruising']) > 0 else 0.0,
                'braking': float(np.mean(ccfg['regimes']['braking'])) if len(ccfg['regimes']['braking']) > 0 else 0.0,
                'rough_road': float(np.mean(ccfg['regimes']['rough_road'])) if len(ccfg['regimes']['rough_road']) > 0 else 0.0,
                'standstill': float(np.mean(ccfg['regimes']['standstill'])) if len(ccfg['regimes']['standstill']) > 0 else 0.0
            }
        }

    # 4. Save Structured JSON Deliverable
    print("\n[4] Saving Structured Deliverable: results/c5_5_4b_kalman_audit_and_shuffled_control.json...", flush=True)
    full_deliverable = {
        'metadata': {
            'stage': 'C5.5.4b',
            'title': 'Kalman Mechanics Audit & Shuffled-Confidence Control',
            'trip': 'Vta02',
            'sampling_rate_hz': 10.0,
            'outage_duration_s': 30.0,
            'stride_s': stride_s,
            'mean_q': mean_q,
            'sigma_nom': sigma_nom,
            'sigma_a_const_inflated': sigma_a_const_inflated
        },
        'kalman_mechanics_audit': kalman_audit_data,
        'shuffled_confidence_control': summary_control_export
    }

    json_path = RES_DIR / "c5_5_4b_kalman_audit_and_shuffled_control.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_deliverable, f, indent=2)
    print(f"    Saved structured JSON ({json_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # 5. Generate Diagnostic Visualizations
    print("\n[5] Rendering Diagnostic Visualizations...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Shuffled Confidence Control Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    cond_keys = ['A_Raw', 'C0_Constant_Inflated', 'C1_Shuffled', 'C1_Adaptive']
    c_labels = [
        'A. Constant Nominal Q\n(sigma_a = 0.15)',
        'C0. Constant Inflated Q\n(sigma_a = 0.291)',
        'C1-Shuffled\n(Permuted q(k) Timing)',
        'C1. True Adaptive\n(Vibration-Driven)'
    ]
    means = [summary_control_export[k]['mean_drift_m'] for k in cond_keys]
    meds = [summary_control_export[k]['median_drift_m'] for k in cond_keys]
    bar_colors = ['black', 'dimgray', 'crimson', 'forestgreen']

    x = np.arange(len(cond_keys))
    w = 0.35
    b1 = ax.bar(x - w/2, means, w, label='Mean 30 s Drift', color=bar_colors, alpha=0.85, edgecolor='black')
    b2 = ax.bar(x + w/2, meds, w, label='Median 30 s Drift', color=bar_colors, alpha=0.45, hatch='//', edgecolor='black')

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 6, f'{yval:.1f}m', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    for bar in b2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 6, f'{yval:.1f}m', ha='center', va='bottom', fontsize=9.0)

    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.4b: Shuffled-Confidence Control Experiment (Vta02, 30s Outages)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10.5)
    ax.grid(True, alpha=0.5, axis='y')

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_5_4b_shuffled_confidence_ablation.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: Kalman Gain & Innovation Trace during Vibration Spike
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    k_spike = events['Event 2 (Vibration Spike)']['idx']
    t_span = 3.0  # +/- 3 seconds around spike
    k_range = range(int(k_spike - t_span / dt), int(k_spike + t_span / dt))
    t_plot = [k * dt for k in k_range]

    # Pre-run trace for this window
    q_trace = [q_k[k] for k in k_range]
    p_trace_a = []
    p_trace_c = []
    k_trace_a = []
    k_trace_c = []

    eskf_a = ESKF3D(init_pos_enu=(gt_e[k_range[0]], gt_n[k_range[0]], gt_u[k_range[0]]),
                    init_vel_enu=(gt_ve[k_range[0]], gt_vn[k_range[0]], gt_vu[k_range[0]]),
                    init_heading_deg=float(veh_heading_arr[k_range[0]]),
                    init_ba=ba_stat, R_vp=np.eye(3), sigma_a=0.15)
    eskf_c = ESKF3D(init_pos_enu=(gt_e[k_range[0]], gt_n[k_range[0]], gt_u[k_range[0]]),
                    init_vel_enu=(gt_ve[k_range[0]], gt_vn[k_range[0]], gt_vu[k_range[0]]),
                    init_heading_deg=float(veh_heading_arr[k_range[0]]),
                    init_ba=ba_stat, R_vp=np.eye(3), sigma_a=0.15)
    nhc_inst = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)

    for step_k in k_range:
        # A
        eskf_a.sigma_a = 0.15
        eskf_a.predict(acc_v[step_k, 0], acc_v[step_k, 1], acc_v[step_k, 2],
                       gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)
        C_v_n_a = eskf_a.attitude.get_dcm()
        r_a, H_a, _ = compute_nhc_residual_and_jacobian(C_v_n_a, eskf_a.vel_n)
        S_a = H_a @ eskf_a.P @ H_a.T + nhc_inst.R_nhc
        K_a = eskf_a.P @ H_a.T @ np.linalg.inv(S_a)
        p_trace_a.append(np.trace(eskf_a.P[3:6, 3:6]))
        k_trace_a.append(np.linalg.norm(K_a[3:6, :]))
        nhc_inst.update_eskf(eskf_a)

        # C
        eskf_c.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_k[step_k])
        eskf_c.predict(acc_v[step_k, 0], acc_v[step_k, 1], acc_v[step_k, 2],
                       gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)
        C_v_n_c = eskf_c.attitude.get_dcm()
        r_c, H_c, _ = compute_nhc_residual_and_jacobian(C_v_n_c, eskf_c.vel_n)
        S_c = H_c @ eskf_c.P @ H_c.T + nhc_inst.R_nhc
        K_c = eskf_c.P @ H_c.T @ np.linalg.inv(S_c)
        p_trace_c.append(np.trace(eskf_c.P[3:6, 3:6]))
        k_trace_c.append(np.linalg.norm(K_c[3:6, :]))
        nhc_inst.update_eskf(eskf_c)

    # Subplot 1: Vibration Metric q(k)
    axes[0].plot(t_plot, q_trace, color='darkorange', lw=2.0, label='Causal Vibration Metric q(k)')
    axes[0].axvline(k_spike * dt, color='gray', ls=':', label=f'Spike Peak (t={k_spike*dt:.1f}s)')
    axes[0].set_ylabel('Vibration Index q(k)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('Stage C5.5.4b: Kalman Mechanics Dynamic Trace across Vibration Spike', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9.5)
    axes[0].grid(True, alpha=0.5)

    # Subplot 2: Prior Velocity Covariance Trace
    axes[1].plot(t_plot, p_trace_a, color='black', lw=1.8, ls='--', label='Baseline A: Tr(P_vel) [Const Q]')
    axes[1].plot(t_plot, p_trace_c, color='crimson', lw=2.0, label='Adaptive C1: Tr(P_vel) [Dyn Q]')
    axes[1].axvline(k_spike * dt, color='gray', ls=':')
    axes[1].set_ylabel('Velocity Covariance\nTr(P_vel) (m²/s²)', fontsize=10.5, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9.5)
    axes[1].grid(True, alpha=0.5)

    # Subplot 3: NHC Kalman Gain Norm
    axes[2].plot(t_plot, k_trace_a, color='black', lw=1.8, ls='--', label='Baseline A: ||K_vel|| [Const Gain]')
    axes[2].plot(t_plot, k_trace_c, color='forestgreen', lw=2.0, label='Adaptive C1: ||K_vel|| [Dynamic NHC Trust]')
    axes[2].axvline(k_spike * dt, color='gray', ls=':')
    axes[2].set_xlabel('Time (seconds)', fontsize=10.5, fontweight='bold')
    axes[2].set_ylabel('NHC Velocity Gain\n||K_vel||', fontsize=10.5, fontweight='bold')
    axes[2].legend(loc='upper left', fontsize=9.5)
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_5_4b_kalman_gain_and_correction_trace.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    print("\nAll figures and results generated successfully.", flush=True)
    print("Stage C5.5.4b audit execution finished.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
