"""
SIH26168 - Stage C5.5.6: Frozen Cross-Trip Validation on Untouched Trip Vta04
Script: experiments/run_cross_trip_validation_c5_5_6.py

PURPOSE:
    Untouched cross-trip evaluation of Candidate 1 (Jerk-Gated Adaptive Covariance)
    and C0 (Constant Inflated Process Noise) on held-out trip Vta04.

STRICT PROTOCOL & FROZEN PARAMETERS (ZERO RECALIBRATION FROM Vta02):
    - Nominal sigma_a,0 = 0.15 m/s^2
    - C0 constant noise: sigma_a = 0.291 m/s^2 (inherited from Vta02)
    - Jerk normalization: mu_jerk = 27.5726 m/s^3 (Vta02 median)
    - Candidate 1 Formula:
          q_cand1(k) = q_old(k) * [0.5 + 0.5 * (sigma_jerk(k) / 27.5726)]
          where q_old(k) = max(0, (std(mag)^2 - 0.15^2) / 0.15^2)
          sigma_a(k) = 0.15 * sqrt(1 + 0.1 * q(k))
    - Static bias ba_stat: inherited from Vta02 ([-0.1471, -0.0124, 0.0031])
    - ESKF and NHC parameters (sigma_lat=0.5, sigma_vert=0.5): identical
    - Horizons: 5.0, 10.0, 20.0, 30.0, 60.0 seconds
    - Stride: 5.0 s (50 epochs) across Vta04
    - Strict zero reference leakage: No CAN, VBOX, GPS speed during outages

CONDITIONS BENCHMARKED:
    1. Condition A: Raw Nominal Baseline (sigma_a = 0.15 m/s^2)
    2. Condition C0: Constant Inflated Baseline (sigma_a = 0.291 m/s^2) [THE HONEST BENCHMARK]
    3. Condition C1: Old Adaptive (q_old)
    4. Condition Cand 1: Frozen Jerk-Gated Quality (q_cand1)
    5. Condition Cand 1 Shuffled: Permuted sequence on Vta04 (seed=42)

DELIVERABLES:
    - results/c5_5_6_cross_trip_validation.json
    - results/figures/c5_5_6_vta04_horizon_drift_comparison.png
    - results/figures/c5_5_6_vta04_c0_vs_cand1_distribution.png
    - results/figures/c5_5_6_vta04_regime_breakdown.png
    - results/c5_5_6_cross_trip_report.md
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
from src.navigation.nhc import NonHolonomicConstraint

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80, flush=True)
    print("STAGE C5.5.6: FROZEN CROSS-TRIP VALIDATION ON UNTOUCHED TRIP VTA04", flush=True)
    print("=" * 80, flush=True)

    # 1. Ingest Vta04 Dataset
    print("\n[1] Ingesting Held-Out Trip Vta04...", flush=True)
    df_p, df_v = load_trip("Vta04")
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
    print(f"    Loaded Vta04: {n} epochs ({time_s[-1]:.1f} s). Alignment angles: {angles}", flush=True)

    # Frozen Static Bias inherited from Vta02 (zero recalibration)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124])
    print(f"    Inherited Frozen Static Bias from Vta02: {ba_stat}", flush=True)

    # Ground Truth ENU
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    heading_rad = np.radians(veh_heading_arr)
    gt_ve = speed * np.sin(heading_rad)
    gt_vn = speed * np.cos(heading_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6

    # 2. Extract Frozen Candidate 1 Features on Vta04
    print("\n[2] Extracting Causal IMU Features on Vta04 with Frozen Vta02 Parameters...", flush=True)
    acc_mag = np.linalg.norm(acc_v, axis=1)
    w_k = 10  # 1.0 s trailing window
    sigma_nom = 0.15
    mu_jerk_vta02 = 27.5726  # Vta02 median jerk RMS frozen

    q_old = np.zeros(n)
    jerk_rms = np.zeros(n)
    vert_var = np.zeros(n)

    for i in range(n):
        i_start = max(0, i - w_k + 1)
        win_acc = acc_v[i_start : i + 1]
        win_mag = acc_mag[i_start : i + 1]

        # q_old
        std_mag = np.std(win_mag) if len(win_mag) > 1 else sigma_nom
        q_old[i] = max(0.0, (std_mag**2 - sigma_nom**2) / (sigma_nom**2))

        # Jerk RMS
        if len(win_acc) > 1:
            jerk = np.diff(win_acc, axis=0) / dt
            jerk_rms[i] = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        else:
            jerk_rms[i] = 0.0

        # Vertical variance for rough road tagging
        vert_var[i] = np.var(win_acc[:, 2]) if len(win_acc) > 1 else 0.0

    # Frozen Candidate 1 formula:
    q_cand1 = q_old * (0.5 + 0.5 * (jerk_rms / mu_jerk_vta02))

    # Shuffled control of Candidate 1 on Vta04 (seed=42)
    rng = np.random.default_rng(seed=42)
    q_cand1_shuffled = rng.permutation(q_cand1)

    print(f"    Vta04 Feature Stats:", flush=True)
    print(f"      q_old:    Median={np.median(q_old):.2f}, Mean={np.mean(q_old):.2f}, Max={np.max(q_old):.2f}", flush=True)
    print(f"      Jerk RMS: Median={np.median(jerk_rms):.2f} m/s³ (vs Vta02 {mu_jerk_vta02:.2f}), Mean={np.mean(jerk_rms):.2f}", flush=True)
    print(f"      q_cand1:  Median={np.median(q_cand1):.2f}, Mean={np.mean(q_cand1):.2f}, Max={np.max(q_cand1):.2f}", flush=True)

    # 3. Setup Multi-Condition Navigation Evaluation Across Horizons
    print("\n[3] Simulating Rolling GNSS Outages on Vta04 (3D ESKF + NHC Pipeline)...", flush=True)
    horizons = [5.0, 10.0, 20.0, 30.0, 60.0]
    stride_s = 5.0
    stride_k = int(round(stride_s / dt))  # 5.0 s stride on Vta04

    # Frozen Noise Levels
    sigma_c0 = 0.291  # Inherited from Vta02

    conditions_to_run = {
        'A_Raw_Nominal': {
            'label': 'A. Raw Nominal (sigma_a = 0.15)',
            'type': 'constant', 'sigma_val': 0.15
        },
        'C0_Honest_Baseline': {
            'label': 'C0. Constant Inflated (sigma_a = 0.291) [BENCHMARK]',
            'type': 'constant', 'sigma_val': sigma_c0
        },
        'C1_Old_Adaptive': {
            'label': 'C1. Old Adaptive q_old(k)',
            'type': 'dynamic', 'q_series': q_old
        },
        'Cand1_Jerk_Gated': {
            'label': 'Cand 1. Frozen Jerk-Gated Quality',
            'type': 'dynamic', 'q_series': q_cand1
        },
        'Cand1_Shuffled_Ctrl': {
            'label': 'Cand 1 Shuffled Control (Broken Timing)',
            'type': 'dynamic', 'q_series': q_cand1_shuffled
        }
    }

    outage_results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, n - w_dur - 1, stride_k))

        outage_results[h_key] = {
            'horizon_s': h,
            'n_windows': len(window_starts),
            'conditions': {c: {'drift_m': [], 'vel_err_ms': [], 'drift_pct': []} for c in conditions_to_run},
            'regimes': {c: {'cruising': [], 'acceleration': [], 'braking': [], 'rough_road': [], 'standstill': []} for c in conditions_to_run}
        }

        t_h = time.time()
        print(f"\n  --- Simulating Horizon {h_key} (N={len(window_starts)} windows) ---", flush=True)

        for idx, kw in enumerate(window_starts):
            win_speed = speed[kw : kw + w_dur + 1]
            win_can_a = can_acc[kw : kw + w_dur + 1]
            win_pvert = acc_v[kw : kw + w_dur + 1, 2]

            t_axis = np.arange(len(win_speed)) * dt
            dist_traveled = float(np.trapezoid(win_speed, t_axis))

            # Regime identification
            is_standstill = float(np.mean(win_speed)) < 0.15
            is_braking = float(np.min(win_can_a)) < -1.5
            is_accel = (float(np.max(win_can_a)) > 1.0) and (not is_braking)
            is_cruising = (not is_standstill) and (not is_braking) and (not is_accel)
            is_rough = float(np.var(win_pvert)) > 0.514  # median roughness on Vta04

            for ckey, ccfg in conditions_to_run.items():
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
                    if ccfg['type'] == 'constant':
                        eskf.sigma_a = ccfg['sigma_val']
                    else:
                        q_val = ccfg['q_series'][step_k]
                        eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_val)

                    eskf.predict(acc_v[step_k, 0], acc_v[step_k, 1], acc_v[step_k, 2],
                                 gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)
                    nhc.update_eskf(eskf)

                st = eskf.get_state()
                gt_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur], gt_u[kw + w_dur]])
                pos_err = float(np.linalg.norm(st['pos_n'][:2] - gt_end[:2]))
                gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur], gt_vu[kw + w_dur]])
                vel_err = float(np.linalg.norm(st['vel_n'][:2] - gt_v_end[:2]))
                drift_pct = (pos_err / max(dist_traveled, 1.0)) * 100.0 if dist_traveled > 5.0 else 0.0

                outage_results[h_key]['conditions'][ckey]['drift_m'].append(pos_err)
                outage_results[h_key]['conditions'][ckey]['vel_err_ms'].append(vel_err)
                outage_results[h_key]['conditions'][ckey]['drift_pct'].append(drift_pct)

                if is_standstill:
                    outage_results[h_key]['regimes'][ckey]['standstill'].append(pos_err)
                if is_braking:
                    outage_results[h_key]['regimes'][ckey]['braking'].append(pos_err)
                if is_accel:
                    outage_results[h_key]['regimes'][ckey]['acceleration'].append(pos_err)
                if is_cruising:
                    outage_results[h_key]['regimes'][ckey]['cruising'].append(pos_err)
                if is_rough:
                    outage_results[h_key]['regimes'][ckey]['rough_road'].append(pos_err)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(window_starts):
                print(f"    Horizon {h_key}: completed {idx+1}/{len(window_starts)} windows in {time.time()-t_h:.1f}s", flush=True)

        print(f"\n  --- Horizon {h_key} Results on Vta04 (N={len(window_starts)} windows) ---", flush=True)
        print(f"    {'Condition':<35} | {'Mean Drift':>11} | {'Median Drift':>13} | {'P90 Drift':>11} | {'Vel Err (Mean/Med)':>18}", flush=True)
        print("    " + "-" * 98, flush=True)
        for ckey in conditions_to_run:
            drifts = outage_results[h_key]['conditions'][ckey]['drift_m']
            verrs = outage_results[h_key]['conditions'][ckey]['vel_err_ms']
            m_drift = np.mean(drifts)
            med_drift = np.median(drifts)
            p90_drift = np.percentile(drifts, 90)
            m_verr = np.mean(verrs)
            med_verr = np.median(verrs)
            print(f"    {conditions_to_run[ckey]['label']:<35} | {m_drift:9.2f} m | {med_drift:11.2f} m | {p90_drift:9.2f} m | {m_verr:6.2f} / {med_verr:5.2f} m/s", flush=True)

    # 4. Export Structured JSON
    print("\n[4] Saving Structured Deliverable: results/c5_5_6_cross_trip_validation.json...", flush=True)
    summary_export = {
        'metadata': {
            'stage': 'C5.5.6',
            'title': 'Frozen Cross-Trip Validation on Untouched Trip Vta04',
            'test_trip': 'Vta04',
            'calibration_source': 'Vta02 (Frozen)',
            'sigma_nom': 0.15,
            'sigma_c0': sigma_c0,
            'mu_jerk_frozen': mu_jerk_vta02,
            'horizons_s': horizons,
            'stride_s': stride_s
        },
        'outage_horizon_summary': {}
    }

    for h_key in outage_results:
        summary_export['outage_horizon_summary'][h_key] = {
            'horizon_s': outage_results[h_key]['horizon_s'],
            'n_windows': outage_results[h_key]['n_windows'],
            'conditions': {}
        }
        for ckey in conditions_to_run:
            drifts = outage_results[h_key]['conditions'][ckey]['drift_m']
            verrs = outage_results[h_key]['conditions'][ckey]['vel_err_ms']
            dpcts = outage_results[h_key]['conditions'][ckey]['drift_pct']
            reg_dict = outage_results[h_key]['regimes'][ckey]

            summary_export['outage_horizon_summary'][h_key]['conditions'][ckey] = {
                'label': conditions_to_run[ckey]['label'],
                'mean_drift_m': float(np.mean(drifts)),
                'median_drift_m': float(np.median(drifts)),
                'std_drift_m': float(np.std(drifts)),
                'p25_drift_m': float(np.percentile(drifts, 25)),
                'p75_drift_m': float(np.percentile(drifts, 75)),
                'p90_drift_m': float(np.percentile(drifts, 90)),
                'p95_drift_m': float(np.percentile(drifts, 95)),
                'mean_vel_err_ms': float(np.mean(verrs)),
                'median_vel_err_ms': float(np.median(verrs)),
                'p90_vel_err_ms': float(np.percentile(verrs, 90)),
                'mean_drift_pct': float(np.mean(dpcts)),
                'raw_drifts_m': [float(x) for x in drifts],
                'regime_means_m': {
                    r: float(np.mean(reg_dict[r])) if len(reg_dict[r]) > 0 else 0.0
                    for r in ['cruising', 'acceleration', 'braking', 'rough_road', 'standstill']
                },
                'regime_medians_m': {
                    r: float(np.median(reg_dict[r])) if len(reg_dict[r]) > 0 else 0.0
                    for r in ['cruising', 'acceleration', 'braking', 'rough_road', 'standstill']
                },
                'regime_counts': {
                    r: len(reg_dict[r])
                    for r in ['cruising', 'acceleration', 'braking', 'rough_road', 'standstill']
                }
            }

    json_path = RES_DIR / "c5_5_6_cross_trip_validation.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, indent=2)
    print(f"    Saved structured JSON ({json_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # 5. Diagnostic Figures
    print("\n[5] Rendering Diagnostic Visualizations...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Horizon Drift Comparison on Vta04
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h_vals = horizons

    # Panel A: Mean Position Drift
    for ckey, col, ls, m in [
        ('A_Raw_Nominal', 'black', '--', 'o'),
        ('C0_Honest_Baseline', 'dimgray', '-', 's'),
        ('C1_Old_Adaptive', 'purple', ':', '^'),
        ('Cand1_Jerk_Gated', 'forestgreen', '-', 'D'),
        ('Cand1_Shuffled_Ctrl', 'crimson', '-.', 'x')
    ]:
        means = [summary_export['outage_horizon_summary'][f"{int(h)}s"]['conditions'][ckey]['mean_drift_m'] for h in h_vals]
        axes[0].plot(h_vals, means, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=conditions_to_run[ckey]['label'])

    axes[0].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[0].set_title('Stage C5.5.6: Mean Position Drift vs. Horizon (Untouched Vta04)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(fontsize=9.5)

    # Panel B: Median Position Drift
    for ckey, col, ls, m in [
        ('A_Raw_Nominal', 'black', '--', 'o'),
        ('C0_Honest_Baseline', 'dimgray', '-', 's'),
        ('C1_Old_Adaptive', 'purple', ':', '^'),
        ('Cand1_Jerk_Gated', 'forestgreen', '-', 'D'),
        ('Cand1_Shuffled_Ctrl', 'crimson', '-.', 'x')
    ]:
        meds = [summary_export['outage_horizon_summary'][f"{int(h)}s"]['conditions'][ckey]['median_drift_m'] for h in h_vals]
        axes[1].plot(h_vals, meds, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=conditions_to_run[ckey]['label'])

    axes[1].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Median Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[1].set_title('Stage C5.5.6: Median Position Drift vs. Horizon (Untouched Vta04)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_5_6_vta04_horizon_drift_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: C0 vs Candidate 1 Drift Distribution (30s Outage on Vta04)
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(15, 6))
    h30 = summary_export['outage_horizon_summary']['30s']['conditions']
    c_keys = ['A_Raw_Nominal', 'C0_Honest_Baseline', 'C1_Old_Adaptive', 'Cand1_Jerk_Gated', 'Cand1_Shuffled_Ctrl']
    c_short_labels = [
        'A. Raw\n(0.15)',
        'C0. Constant\n(0.291)',
        'C1. Old\nAdaptive',
        'Cand 1.\nJerk-Gated',
        'Cand 1\nShuffled'
    ]
    means_30 = [h30[k]['mean_drift_m'] for k in c_keys]
    meds_30 = [h30[k]['median_drift_m'] for k in c_keys]
    p90s_30 = [h30[k]['p90_drift_m'] for k in c_keys]
    colors = ['#333333', '#555555', '#7b2cbf', '#2d6a4f', '#c1121f']

    # Panel A: Summary Metrics (Mean, Median, P90)
    x = np.arange(len(c_keys))
    w = 0.26
    b1 = ax_bar.bar(x - w, means_30, w, label='Mean Drift', color=colors, alpha=0.9, edgecolor='black')
    b2 = ax_bar.bar(x, meds_30, w, label='Median Drift', color=colors, alpha=0.5, hatch='//', edgecolor='black')
    b3 = ax_bar.bar(x + w, p90s_30, w, label='P90 Drift', color=colors, alpha=0.3, hatch='xx', edgecolor='black')

    c0_mean_val = h30['C0_Honest_Baseline']['mean_drift_m']
    ax_bar.axhline(c0_mean_val, color='black', ls='--', lw=1.2, label=f'C0 Mean ({c0_mean_val:.1f}m)')

    for bar in b1:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.0f}m', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(c_short_labels, fontsize=9.5, fontweight='bold')
    ax_bar.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax_bar.set_title('(A) 30 s Summary Metrics on Vta04 (N=29)', fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=9, loc='upper left')
    ax_bar.grid(True, alpha=0.4, axis='y')

    # Panel B: Boxplot showing exact distributions across all 29 windows
    data_for_box = [h30[k]['raw_drifts_m'] for k in c_keys]
    bp = ax_box.boxplot(data_for_box, tick_labels=c_short_labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        medianprops=dict(color='darkred', lw=2),
                        whiskerprops=dict(color='black', lw=1.2),
                        capprops=dict(color='black', lw=1.2),
                        flierprops=dict(marker='o', color='red', markersize=6, alpha=0.7))

    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)

    ax_box.axhline(c0_mean_val, color='black', ls='--', lw=1.2, label=f'C0 Mean ({c0_mean_val:.1f}m)')
    ax_box.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax_box.set_title('(B) 30 s Outage Drift Distributions (Boxplots)', fontsize=12, fontweight='bold')
    ax_box.grid(True, alpha=0.4, axis='y')
    ax_box.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_5_6_vta04_c0_vs_cand1_distribution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Figure 3: Regime Breakdown on Vta04 (30s Outages)
    fig, ax = plt.subplots(figsize=(12, 6))
    regimes = ['cruising', 'acceleration', 'braking', 'rough_road']
    counts_30 = h30['Cand1_Jerk_Gated']['regime_counts']
    reg_labels = [
        f"Steady Cruising\n(N={counts_30['cruising']})",
        f"Acceleration (a>1.0)\n(N={counts_30['acceleration']})",
        f"Severe Braking (a<-1.5)\n(N={counts_30['braking']})",
        f"Rough Road (var>0.51)\n(N={counts_30['rough_road']})"
    ]

    x_r = np.arange(len(regimes))
    w_r = 0.22

    plot_conds = [
        ('A_Raw_Nominal', 'A. Raw Nominal (0.15)', 'black'),
        ('C0_Honest_Baseline', 'C0. Constant Inflated (0.291) [BENCHMARK]', 'dimgray'),
        ('Cand1_Jerk_Gated', 'Cand 1. Jerk-Gated (Candidate)', 'forestgreen')
    ]

    for idx, (ckey, clbl, col) in enumerate(plot_conds):
        vals = [h30[ckey]['regime_means_m'][r] for r in regimes]
        bars = ax.bar(x_r + (idx - 1) * w_r, vals, w_r, label=clbl, color=col, alpha=0.85, edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.0f}m', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_r)
    ax.set_xticklabels(reg_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean 30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.6: Vta04 Regime Drift Breakdown (C0 vs. Frozen Candidate 1)\nNote: Standstill N=0 (Continuous Dynamic Driving on Vta04)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_5_6_vta04_regime_breakdown.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    print("\nAll figures and results generated successfully.", flush=True)
    print("Stage C5.5.6 execution completed successfully.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
