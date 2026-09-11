"""
SIH26168 - Stage C5.5.7: Causal Ambient-Baseline Normalization Audit
Script: experiments/run_adaptive_baseline_audit_c5_5_7.py

PURPOSE:
    Test whether decoupling the fast acute disturbance J(k) from the slow
    ambient road-roughness baseline B(k) produces a trip-invariant confidence
    signal that resolves the steady-cruising over-inflation on held-out trip Vta04,
    without sacrificing gains during braking, acceleration, or rough-road events.

STRICT PROTOCOL & GUARDRAILS:
    1. Zero Hyperparameter Tuning on Vta04:
       Baseline window W_b in {10s, 30s, 60s} is selected strictly on Vta02.
       The winning window W_b* is frozen 100% before evaluating Vta04.
    2. Causal Implementation:
       B(k) uses only causal trailing samples median(J(k - W_b + 1), ..., J(k)).
       Zero future samples, zero GPS, zero CAN, zero VBOX velocity during outages.
    3. Benchmark against C0 (sigma_a = 0.291 m/s^2) as the honest success standard.
    4. Bounded C0 Multiplier Control included to test simple bounded adaptation.
    5. Shuffled-confidence control included to verify temporal alignment.

DELIVERABLES:
    - results/c5_5_7_adaptive_baseline_audit.json
    - results/figures/c5_5_7_baseline_window_selection_vta02.png
    - results/figures/c5_5_7_vta04_normalization_distribution.png
    - results/figures/c5_5_7_vta04_horizon_drift_comparison.png
    - results/figures/c5_5_7_vta04_regime_breakdown.png
    - results/c5_5_7_adaptive_baseline_report.md
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


def compute_causal_features(acc_v, dt=0.1, w_k=10, sigma_nom=0.15):
    """
    Extract causal q_old and trailing jerk RMS J(k).
    """
    n = len(acc_v)
    acc_mag = np.linalg.norm(acc_v, axis=1)
    q_old = np.zeros(n)
    jerk_rms = np.zeros(n)

    for i in range(n):
        i_start = max(0, i - w_k + 1)
        win_acc = acc_v[i_start : i + 1]
        win_mag = acc_mag[i_start : i + 1]

        std_mag = np.std(win_mag) if len(win_mag) > 1 else sigma_nom
        q_old[i] = max(0.0, (std_mag**2 - sigma_nom**2) / (sigma_nom**2))

        if len(win_acc) > 1:
            jerk = np.diff(win_acc, axis=0) / dt
            jerk_rms[i] = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        else:
            jerk_rms[i] = 0.0

    return q_old, jerk_rms


def compute_causal_ambient_baseline(jerk_rms, window_epochs):
    """
    Compute slow ambient baseline B(k) using causal trailing median.
    For i < window_epochs, uses all available history from 0 to i.
    """
    n = len(jerk_rms)
    baseline = np.zeros(n)
    for i in range(n):
        i_start = max(0, i - window_epochs + 1)
        baseline[i] = np.median(jerk_rms[i_start : i + 1])
    return baseline


def evaluate_outages(acc_v, gyro_v, speed, veh_heading_arr, gt_e, gt_n, gt_u,
                     gt_ve, gt_vn, gt_vu, ba_stat, can_acc,
                     conditions, horizons, stride_s=5.0, dt=0.1):
    """
    Run rolling outage evaluation across given horizons and conditions.
    """
    n = len(acc_v)
    stride_k = int(round(stride_s / dt))
    outage_results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, n - w_dur - 1, stride_k))

        outage_results[h_key] = {
            'horizon_s': h,
            'n_windows': len(window_starts),
            'conditions': {c: {'drift_m': [], 'vel_err_ms': [], 'drift_pct': []} for c in conditions},
            'regimes': {c: {'cruising': [], 'acceleration': [], 'braking': [], 'rough_road': [], 'standstill': []} for c in conditions}
        }

        for kw in window_starts:
            win_speed = speed[kw : kw + w_dur + 1]
            win_can_a = can_acc[kw : kw + w_dur + 1]
            win_pvert = acc_v[kw : kw + w_dur + 1, 2]

            t_axis = np.arange(len(win_speed)) * dt
            dist_traveled = float(np.trapezoid(win_speed, t_axis))

            is_standstill = float(np.mean(win_speed)) < 0.15
            is_braking = float(np.min(win_can_a)) < -1.5
            is_accel = (float(np.max(win_can_a)) > 1.0) and (not is_braking)
            is_cruising = (not is_standstill) and (not is_braking) and (not is_accel)
            is_rough = float(np.var(win_pvert)) > 0.514

            for ckey, ccfg in conditions.items():
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
                    elif ccfg['mode'] == 'q_dynamic':
                        q_val = ccfg['q_series'][step_k]
                        eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_val)
                    elif ccfg['mode'] == 'sigma_direct':
                        eskf.sigma_a = ccfg['sigma_series'][step_k]

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

    return outage_results


def main():
    print("=" * 80, flush=True)
    print("STAGE C5.5.7: CAUSAL AMBIENT-BASELINE NORMALIZATION AUDIT", flush=True)
    print("=" * 80, flush=True)

    dt = 0.1
    sigma_nom = 0.15
    sigma_c0 = 0.291
    mu_jerk_vta02 = 27.5726  # Static scalar from Vta02
    ba_stat = np.array([-0.147147, -0.012351, 0.003124])

    # -------------------------------------------------------------------------
    # PHASE 1: HYPERPARAMETER WINDOW SELECTION ON VTA02
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Ingesting Calibration Trip Vta02 for Ambient Window Selection...", flush=True)
    df_p2, df_v2 = load_trip("Vta02")
    n2 = len(df_p2)

    raw_acc2 = df_p2[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro2 = df_p2[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed2 = df_v2['veh_speed_ms'].values
    can_acc2 = df_v2['veh_accel_long_ms2'].values
    heading2 = df_v2['veh_heading_deg'].values

    acc_v2, gyro_v2, _, _ = align_phone_to_vehicle(raw_acc2, raw_gyro2, speed2)
    lat0_2, lon0_2 = df_v2['veh_lat'].iloc[0], df_v2['veh_lon'].iloc[0]
    gt_e2, gt_n2, gt_u2 = geodetic_to_enu(df_v2['veh_lat'].values, df_v2['veh_lon'].values, lat0_2, lon0_2)
    h_rad2 = np.radians(heading2)
    gt_ve2 = speed2 * np.sin(h_rad2)
    gt_vn2 = speed2 * np.cos(h_rad2)
    gt_vu2 = df_v2['veh_vert_vel_kmh'].values / 3.6

    q_old2, jerk_rms2 = compute_causal_features(acc_v2, dt=dt, w_k=10, sigma_nom=sigma_nom)

    # Static Candidate 1 on Vta02
    q_cand1_vta02 = q_old2 * (0.5 + 0.5 * (jerk_rms2 / mu_jerk_vta02))

    # Evaluate Baseline Windows W_b in {10s, 30s, 60s}
    candidate_windows_s = [10.0, 30.0, 60.0]
    vta02_conditions = {
        'C0_Baseline': {'mode': 'constant', 'sigma_val': sigma_c0, 'label': 'C0 (0.291)'},
        'Cand1_Static': {'mode': 'q_dynamic', 'q_series': q_cand1_vta02, 'label': 'Cand 1 (Static 27.57)'}
    }

    vta02_norm_series = {}
    for wb_s in candidate_windows_s:
        wb_epochs = int(round(wb_s / dt))
        b2 = compute_causal_ambient_baseline(jerk_rms2, wb_epochs)
        j_norm2 = jerk_rms2 / (b2 + 1e-4)
        vta02_norm_series[wb_s] = j_norm2
        q_cand_b2 = q_old2 * (0.5 + 0.5 * j_norm2)
        vta02_conditions[f'Cand1B_{int(wb_s)}s'] = {
            'mode': 'q_dynamic', 'q_series': q_cand_b2,
            'label': f'Cand 1B (Adaptive W_b={int(wb_s)}s)'
        }
        # Bounded C0 control
        sig_bounded2 = sigma_c0 * np.clip(np.sqrt(j_norm2), 0.75, 1.35)
        vta02_conditions[f'Bounded_C0_{int(wb_s)}s'] = {
            'mode': 'sigma_direct', 'sigma_series': sig_bounded2,
            'label': f'Bounded C0 (W_b={int(wb_s)}s)'
        }

    print("  Evaluating 30s Outages across Window Candidates on Vta02...", flush=True)
    res_vta02 = evaluate_outages(acc_v2, gyro_v2, speed2, heading2, gt_e2, gt_n2, gt_u2,
                                 gt_ve2, gt_vn2, gt_vu2, ba_stat, can_acc2,
                                 vta02_conditions, horizons=[30.0], stride_s=5.0, dt=dt)

    print("\n  --- Phase 1: Vta02 30s Outage Results ---", flush=True)
    print(f"  {'Condition':<35} | {'Mean Drift':>11} | {'Median Drift':>13} | {'P90 Drift':>11}", flush=True)
    print("  " + "-" * 78, flush=True)

    vta02_30s = res_vta02['30s']['conditions']
    for ckey, ccfg in vta02_conditions.items():
        drifts = vta02_30s[ckey]['drift_m']
        print(f"  {ccfg['label']:<35} | {np.mean(drifts):9.2f} m | {np.median(drifts):11.2f} m | {np.percentile(drifts, 90):9.2f} m", flush=True)

    # Determine Best Window W_b* based strictly on Vta02 Cand 1B mean drift
    window_scores = {
        wb_s: np.mean(vta02_30s[f'Cand1B_{int(wb_s)}s']['drift_m'])
        for wb_s in candidate_windows_s
    }
    w_b_star = min(window_scores, key=window_scores.get)
    print(f"\n  >>> Winning Causal Baseline Window Selected from Vta02: W_b* = {int(w_b_star)} s (Mean Drift = {window_scores[w_b_star]:.2f} m) <<<", flush=True)
    print(f"  >>> Freezing W_b* = {int(w_b_star)} s with ZERO recalibration for Vta04 test <<<", flush=True)

    # Render Phase 1 Figure: Baseline Window Selection on Vta02
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_keys = ['C0_Baseline', 'Cand1_Static', 'Cand1B_10s', 'Cand1B_30s', 'Cand1B_60s', f'Bounded_C0_{int(w_b_star)}s']
    plot_labels = [
        'C0 Benchmark\n(0.291)',
        'Cand 1 Static\n(Frozen 27.57)',
        'Cand 1B\n(W_b = 10s)',
        'Cand 1B\n(W_b = 30s)',
        'Cand 1B\n(W_b = 60s)',
        f'Bounded C0\n(W_b = {int(w_b_star)}s)'
    ]
    means_v2 = [np.mean(vta02_30s[k]['drift_m']) for k in plot_keys]
    meds_v2 = [np.median(vta02_30s[k]['drift_m']) for k in plot_keys]

    x_v2 = np.arange(len(plot_keys))
    w_v2 = 0.35
    b1_v2 = ax.bar(x_v2 - w_v2/2, means_v2, w_v2, label='Mean 30s Drift (m)', color='#2b2d42', alpha=0.85, edgecolor='black')
    b2_v2 = ax.bar(x_v2 + w_v2/2, meds_v2, w_v2, label='Median 30s Drift (m)', color='#8d99ae', alpha=0.6, hatch='//', edgecolor='black')

    c0_v2_val = np.mean(vta02_30s['C0_Baseline']['drift_m'])
    ax.axhline(c0_v2_val, color='red', ls='--', lw=1.2, label=f'C0 Baseline ({c0_v2_val:.1f}m)')

    for bar in b1_v2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f}m', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x_v2)
    ax.set_xticklabels(plot_labels, fontsize=9.5, fontweight='bold')
    ax.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.7 Phase 1: Baseline Window Selection on Calibration Journey Vta02', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_5_7_baseline_window_selection_vta02.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE 2: UNTOUCHED FROZEN EVALUATION ON HELD-OUT TRIP VTA04
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print(f"[PHASE 2] Ingesting Held-Out Trip Vta04 (Frozen W_b* = {int(w_b_star)}s)...", flush=True)
    print("=" * 80, flush=True)

    df_p4, df_v4 = load_trip("Vta04")
    n4 = len(df_p4)

    raw_acc4 = df_p4[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro4 = df_p4[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed4 = df_v4['veh_speed_ms'].values
    can_acc4 = df_v4['veh_accel_long_ms2'].values
    heading4 = df_v4['veh_heading_deg'].values

    acc_v4, gyro_v4, _, _ = align_phone_to_vehicle(raw_acc4, raw_gyro4, speed4)
    lat0_4, lon0_4 = df_v4['veh_lat'].iloc[0], df_v4['veh_lon'].iloc[0]
    gt_e4, gt_n4, gt_u4 = geodetic_to_enu(df_v4['veh_lat'].values, df_v4['veh_lon'].values, lat0_4, lon0_4)
    h_rad4 = np.radians(heading4)
    gt_ve4 = speed4 * np.sin(h_rad4)
    gt_vn4 = speed4 * np.cos(h_rad4)
    gt_vu4 = df_v4['veh_vert_vel_kmh'].values / 3.6

    q_old4, jerk_rms4 = compute_causal_features(acc_v4, dt=dt, w_k=10, sigma_nom=sigma_nom)

    # 1. Condition C0: Constant 0.291
    # 2. Condition Cand 1 Static: Frozen 27.5726
    q_cand1_vta04 = q_old4 * (0.5 + 0.5 * (jerk_rms4 / mu_jerk_vta02))

    # 3. Condition Cand 1B: Adaptive Causal Ambient Baseline (W_b*)
    wb_star_epochs = int(round(w_b_star / dt))
    b4 = compute_causal_ambient_baseline(jerk_rms4, wb_star_epochs)
    j_norm4 = jerk_rms4 / (b4 + 1e-4)
    q_cand_b4 = q_old4 * (0.5 + 0.5 * j_norm4)

    # 4. Condition Cand 1B Shuffled Control (seed=42)
    rng = np.random.default_rng(seed=42)
    q_cand_b4_shuffled = rng.permutation(q_cand_b4)

    # 5. Condition Bounded C0 Control
    sig_bounded4 = sigma_c0 * np.clip(np.sqrt(j_norm4), 0.75, 1.35)

    vta04_conditions = {
        'C0_Honest_Baseline': {
            'mode': 'constant', 'sigma_val': sigma_c0,
            'label': 'C0. Constant Inflated (0.291) [BENCHMARK]'
        },
        'Cand1_Static_Frozen': {
            'mode': 'q_dynamic', 'q_series': q_cand1_vta04,
            'label': 'Cand 1. Static Frozen Normalization'
        },
        'Cand1B_Adaptive_Baseline': {
            'mode': 'q_dynamic', 'q_series': q_cand_b4,
            'label': f'Cand 1B. Adaptive Baseline (W_b={int(w_b_star)}s)'
        },
        'Cand1B_Shuffled_Ctrl': {
            'mode': 'q_dynamic', 'q_series': q_cand_b4_shuffled,
            'label': 'Cand 1B Shuffled Control (Broken Timing)'
        },
        'Bounded_C0_Control': {
            'mode': 'sigma_direct', 'sigma_series': sig_bounded4,
            'label': f'Bounded C0 Multiplier (W_b={int(w_b_star)}s)'
        }
    }

    # Statistical Distribution Audit
    print("\n[PHASE 3] Auditing Normalization Distributions on Vta02 vs Vta04...", flush=True)
    b2_star = compute_causal_ambient_baseline(jerk_rms2, wb_star_epochs)
    j_norm2_star = jerk_rms2 / (b2_star + 1e-4)

    elev_dist_pct_v2 = float(np.mean(j_norm2_star > 1.25) * 100.0)
    elev_dist_pct_v4 = float(np.mean(j_norm4 > 1.25) * 100.0)

    print(f"  Vta02: Jerk RMS Median = {np.median(jerk_rms2):.2f} m/s³, Baseline B(k) Median = {np.median(b2_star):.2f} m/s³", flush=True)
    print(f"         J_norm Median = {np.median(j_norm2_star):.2f}, Mean = {np.mean(j_norm2_star):.2f}, Elevated Disturbance (>1.25) = {elev_dist_pct_v2:.1f}%", flush=True)
    print(f"  Vta04: Jerk RMS Median = {np.median(jerk_rms4):.2f} m/s³, Baseline B(k) Median = {np.median(b4):.2f} m/s³", flush=True)
    print(f"         J_norm Median = {np.median(j_norm4):.2f}, Mean = {np.mean(j_norm4):.2f}, Elevated Disturbance (>1.25) = {elev_dist_pct_v4:.1f}%", flush=True)

    # Render Figure 2: Normalization Distributions Comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: Raw Jerk RMS J(k)
    axes[0, 0].hist(jerk_rms2, bins=50, density=True, alpha=0.6, color='blue', label=f'Vta02 (Med={np.median(jerk_rms2):.1f})')
    axes[0, 0].hist(jerk_rms4, bins=50, density=True, alpha=0.6, color='red', label=f'Vta04 (Med={np.median(jerk_rms4):.1f})')
    axes[0, 0].set_title('(A) Fast Jerk RMS J(k) Distributions (+61% shift)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Jerk RMS (m/s³)', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].legend(fontsize=9.5)
    axes[0, 0].grid(True, alpha=0.4)

    # Panel B: Ambient Baseline B(k)
    axes[0, 1].hist(b2_star, bins=50, density=True, alpha=0.6, color='blue', label=f'Vta02 Baseline (Med={np.median(b2_star):.1f})')
    axes[0, 1].hist(b4, bins=50, density=True, alpha=0.6, color='red', label=f'Vta04 Baseline (Med={np.median(b4):.1f})')
    axes[0, 1].set_title(f'(B) Slow Ambient Baseline B(k) (W_b = {int(w_b_star)}s)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Ambient Baseline B(k) (m/s³)', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].legend(fontsize=9.5)
    axes[0, 1].grid(True, alpha=0.4)

    # Panel C: Normalized Disturbance J_norm(k)
    axes[1, 0].hist(j_norm2_star, bins=50, range=(0, 3), density=True, alpha=0.6, color='blue', label=f'Vta02 J_norm (Med={np.median(j_norm2_star):.2f})')
    axes[1, 0].hist(j_norm4, bins=50, range=(0, 3), density=True, alpha=0.6, color='red', label=f'Vta04 J_norm (Med={np.median(j_norm4):.2f})')
    axes[1, 0].axvline(1.0, color='black', ls='--', lw=1.2, label='Neutral (J=B)')
    axes[1, 0].set_title('(C) Normalized Disturbance J_norm = J(k) / B(k) (Re-centered!)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('J_norm(k)', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    axes[1, 0].legend(fontsize=9.5)
    axes[1, 0].grid(True, alpha=0.4)

    # Panel D: Multiplier [0.5 + 0.5 * J_norm]
    mult2 = 0.5 + 0.5 * j_norm2_star
    mult4 = 0.5 + 0.5 * j_norm4
    axes[1, 1].hist(mult2, bins=50, range=(0.5, 2.5), density=True, alpha=0.6, color='blue', label=f'Vta02 Mult (Med={np.median(mult2):.2f})')
    axes[1, 1].hist(mult4, bins=50, range=(0.5, 2.5), density=True, alpha=0.6, color='red', label=f'Vta04 Mult (Med={np.median(mult4):.2f})')
    axes[1, 1].axvline(1.0, color='black', ls='--', lw=1.2, label='Nominal 1.0')
    axes[1, 1].set_title('(D) Gating Multiplier [0.5 + 0.5 * J_norm]', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Multiplier Value', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].legend(fontsize=9.5)
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_5_7_vta04_normalization_distribution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Multi-Horizon Simulation on Vta04
    horizons = [5.0, 10.0, 20.0, 30.0, 60.0]
    print(f"\n[PHASE 4] Simulating Outages on Untouched Trip Vta04 across {horizons}s...", flush=True)
    res_vta04 = evaluate_outages(acc_v4, gyro_v4, speed4, heading4, gt_e4, gt_n4, gt_u4,
                                 gt_ve4, gt_vn4, gt_vu4, ba_stat, can_acc4,
                                 vta04_conditions, horizons=horizons, stride_s=5.0, dt=dt)

    for h in horizons:
        h_key = f"{int(h)}s"
        h_data = res_vta04[h_key]
        print(f"\n  --- Horizon {h_key} Results on Vta04 (N={h_data['n_windows']} windows) ---", flush=True)
        print(f"    {'Condition':<40} | {'Mean Drift':>11} | {'Median Drift':>13} | {'P90 Drift':>11} | {'Vel Err (Mean/Med)':>18}", flush=True)
        print("    " + "-" * 103, flush=True)
        for ckey in vta04_conditions:
            drifts = h_data['conditions'][ckey]['drift_m']
            verrs = h_data['conditions'][ckey]['vel_err_ms']
            m_drift = np.mean(drifts)
            med_drift = np.median(drifts)
            p90_drift = np.percentile(drifts, 90)
            m_verr = np.mean(verrs)
            med_verr = np.median(verrs)
            print(f"    {vta04_conditions[ckey]['label']:<40} | {m_drift:9.2f} m | {med_drift:11.2f} m | {p90_drift:9.2f} m | {m_verr:6.2f} / {med_verr:5.2f} m/s", flush=True)

    # -------------------------------------------------------------------------
    # EXPORT STRUCTURED JSON DELIVERABLE
    # -------------------------------------------------------------------------
    print("\n[PHASE 5] Exporting Deliverable: results/c5_5_7_adaptive_baseline_audit.json...", flush=True)
    summary_export = {
        'metadata': {
            'stage': 'C5.5.7',
            'title': 'Causal Ambient-Baseline Normalization Audit',
            'selected_w_b_star_s': w_b_star,
            'sigma_nom': sigma_nom,
            'sigma_c0': sigma_c0,
            'mu_jerk_static_vta02': mu_jerk_vta02,
            'horizons_s': horizons,
            'stride_s': 5.0,
            'elevated_disturbance_pct': {
                'vta02': elev_dist_pct_v2,
                'vta04': elev_dist_pct_v4
            }
        },
        'phase1_vta02_window_sweep': {
            f'{int(wb_s)}s': {
                'mean_drift_30s_m': float(np.mean(res_vta02['30s']['conditions'][f'Cand1B_{int(wb_s)}s']['drift_m'])),
                'median_drift_30s_m': float(np.median(res_vta02['30s']['conditions'][f'Cand1B_{int(wb_s)}s']['drift_m'])),
                'p90_drift_30s_m': float(np.percentile(res_vta02['30s']['conditions'][f'Cand1B_{int(wb_s)}s']['drift_m'], 90))
            } for wb_s in candidate_windows_s
        },
        'phase2_vta04_outages': {}
    }

    for h_key in res_vta04:
        summary_export['phase2_vta04_outages'][h_key] = {
            'horizon_s': res_vta04[h_key]['horizon_s'],
            'n_windows': res_vta04[h_key]['n_windows'],
            'conditions': {}
        }
        for ckey in vta04_conditions:
            drifts = res_vta04[h_key]['conditions'][ckey]['drift_m']
            verrs = res_vta04[h_key]['conditions'][ckey]['vel_err_ms']
            dpcts = res_vta04[h_key]['conditions'][ckey]['drift_pct']
            reg_dict = res_vta04[h_key]['regimes'][ckey]

            summary_export['phase2_vta04_outages'][h_key]['conditions'][ckey] = {
                'label': vta04_conditions[ckey]['label'],
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

    json_path = RES_DIR / "c5_5_7_adaptive_baseline_audit.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, indent=2)
    print(f"  Saved structured JSON ({json_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # -------------------------------------------------------------------------
    # DIAGNOSTIC FIGURES FOR VTA04
    # -------------------------------------------------------------------------
    print("\n[PHASE 6] Rendering Diagnostic Visualizations for Vta04...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 3: Horizon Drift Comparison on Vta04
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h_vals = horizons

    cond_styles = [
        ('C0_Honest_Baseline', 'dimgray', '-', 's'),
        ('Cand1_Static_Frozen', 'darkorange', '--', 'o'),
        ('Cand1B_Adaptive_Baseline', 'forestgreen', '-', 'D'),
        ('Cand1B_Shuffled_Ctrl', 'crimson', '-.', 'x'),
        ('Bounded_C0_Control', 'teal', ':', '^')
    ]

    for ckey, col, ls, m in cond_styles:
        means = [summary_export['phase2_vta04_outages'][f"{int(h)}s"]['conditions'][ckey]['mean_drift_m'] for h in h_vals]
        axes[0].plot(h_vals, means, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=vta04_conditions[ckey]['label'])

    axes[0].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[0].set_title('Mean Drift vs. Horizon on Held-Out Vta04', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(fontsize=8.5)

    for ckey, col, ls, m in cond_styles:
        meds = [summary_export['phase2_vta04_outages'][f"{int(h)}s"]['conditions'][ckey]['median_drift_m'] for h in h_vals]
        axes[1].plot(h_vals, meds, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=vta04_conditions[ckey]['label'])

    axes[1].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Median Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[1].set_title('Median Drift vs. Horizon on Held-Out Vta04', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_5_7_vta04_horizon_drift_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # Figure 4: Regime Breakdown on Vta04 (30s Outages)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    h30_v4 = summary_export['phase2_vta04_outages']['30s']['conditions']
    regimes = ['cruising', 'acceleration', 'braking', 'rough_road']
    counts_30 = h30_v4['Cand1B_Adaptive_Baseline']['regime_counts']
    reg_labels = [
        f"Steady Cruising\n(N={counts_30['cruising']})",
        f"Acceleration (a>1.0)\n(N={counts_30['acceleration']})",
        f"Severe Braking (a<-1.5)\n(N={counts_30['braking']})",
        f"Rough Road (var>0.51)\n(N={counts_30['rough_road']})"
    ]

    x_r = np.arange(len(regimes))
    w_r = 0.18

    plot_reg_conds = [
        ('C0_Honest_Baseline', 'C0. Constant (0.291) [BENCHMARK]', 'dimgray'),
        ('Cand1_Static_Frozen', 'Cand 1. Static Frozen', 'darkorange'),
        ('Cand1B_Adaptive_Baseline', f'Cand 1B. Adaptive Baseline (W_b={int(w_b_star)}s)', 'forestgreen'),
        ('Bounded_C0_Control', 'Bounded C0 Control', 'teal')
    ]

    for idx, (ckey, clbl, col) in enumerate(plot_reg_conds):
        vals = [h30_v4[ckey]['regime_means_m'][r] for r in regimes]
        bars = ax.bar(x_r + (idx - 1.5) * w_r, vals, w_r, label=clbl, color=col, alpha=0.85, edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f'{yval:.0f}m', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_r)
    ax.set_xticklabels(reg_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean 30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title(f'Stage C5.5.7: Vta04 Regime Drift Breakdown (Cruising Over-Inflation Audit)\nNote: Standstill N=0 (Continuous Driving on Vta04)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig4_path = FIG_DIR / "c5_5_7_vta04_regime_breakdown.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()

    print("\nStage C5.5.7 execution completed successfully.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
