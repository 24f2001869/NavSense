"""
SIH26168 - Stage C5.5.4: End-to-End Signal Conditioning Ablation
Script: experiments/run_signal_conditioning_ablation_c5_5_4.py

PURPOSE:
    Closed-loop navigation ablation on clean control trip Vta02 to determine
    whether recognizing and dynamically down-weighting contaminated smartphone
    IMU measurements improves dead-reckoning performance compared to raw IMU
    integration and fixed low-pass filtering.

EXPERIMENTAL CONDITIONS:
    - Condition A: Raw Smartphone IMU (unconditioned baseline)
    - Condition B: Fixed Causal Low-Pass Filtering (fc in {0.5, 1.0, 1.5, 2.0} Hz)
    - Condition C: Vibration-Based Adaptive Weighting (q(k) from causal IMU features):
        * Sub-variant C1: Covariance-Only Adaptive Scaling (dynamic sigma_a)
        * Sub-variant C2: Dynamic Signal Blending (gamma_k blending with smoothed signal)
        * Sub-variant C3: Combined (Signal Blending + Dynamic Covariance)
    - Condition D: Oracle Reference Diagnostic (true vehicle reference acceleration)

CONTROLS & GUARDRAILS:
    - Clean control trip Vta02 only (in-trip diagnostic, not cross-trip generalization)
    - Zero neural networks (transparent, physics-grounded baseline)
    - Zero reference (CAN/VBOX) or future sample leakage for deployable methods (A, B, C)
    - Identical 3D ESKF + NHC navigation pipeline and identical initial-state resets
    - Multi-horizon evaluation: 5, 10, 20, 30, 60 seconds (stride = 2.5 s)
    - Stratified regime analysis: Cruising, Severe Braking, Rough Road, Standstill
"""

import sys
import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
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


def butter_lowpass_causal(sig: np.ndarray, fc: float, fs: float = 10.0, order: int = 2) -> np.ndarray:
    """Applies strictly causal 2nd-order digital Butterworth lowpass filter."""
    wn = fc / (fs / 2.0)
    b, a = butter(order, wn, btype='low')
    if sig.ndim == 1:
        zi = lfilter_zi(b, a) * sig[0]
        out, _ = lfilter(b, a, sig, zi=zi)
        return out
    else:
        out = np.zeros_like(sig)
        for ax in range(sig.shape[1]):
            zi = lfilter_zi(b, a) * sig[0, ax]
            out[:, ax], _ = lfilter(b, a, sig[:, ax], zi=zi)
        return out


def main():
    print("=" * 80)
    print("STAGE C5.5.4: END-TO-END SIGNAL CONDITIONING ABLATION (Vta02 Control)")
    print("=" * 80)

    # 1. Ingest Data & Setup Ground Truth
    print("\n[1] Ingesting Vta02 Dataset...")
    df_p, df_v = load_trip("Vta02")
    n = len(df_p)
    dt = 0.1
    time_s = np.arange(n) * dt

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values

    # Vehicle Alignment
    acc_v, gyro_v, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    print(f"    Aligned Vta02: {n} epochs ({time_s[-1]:.1f} s). Alignment angles: {angles}")

    # Ground Truth ENU
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    heading_rad = np.radians(df_v['veh_heading_deg'].values)
    gt_ve = speed * np.sin(heading_rad)
    gt_vn = speed * np.cos(heading_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6

    # Reference Accelerations
    can_acc = df_v['veh_accel_long_ms2'].values
    can_lat = df_v['veh_accel_lat_ms2'].values
    dv_dt = np.gradient(speed, dt)

    # Static Calibration Bias from Standstill (t=14.2s to 19.1s)
    ba_stat = np.mean(acc_v[142:191], axis=0) - np.array([0.0, 0.0, 9.80665])
    print(f"    Calibrated Static Accelerometer Bias ba_stat: {ba_stat}")

    # 2. Signal Conditioning Implementations
    print("\n[2] Generating Experimental Conditions...")

    # Condition A: Raw IMU
    acc_raw = acc_v.copy()

    # Condition B: Fixed Causal Low-Pass Sweep (0.5, 1.0, 1.5, 2.0 Hz)
    cutoffs = [0.5, 1.0, 1.5, 2.0]
    acc_lp = {}
    for fc in cutoffs:
        acc_lp[fc] = butter_lowpass_causal(acc_v, fc=fc)
        print(f"    Generated Fixed Low-Pass Filter at fc = {fc:.1f} Hz")

    # Quantify Filter Latency & Phase Delay on Severe Braking Transients
    # Find severe braking event (e.g. around t=200s to 215s where vehicle decelerates sharply)
    brake_mask = (can_acc < -2.0)
    brake_indices = np.where(brake_mask)[0]
    peak_brake_idx = brake_indices[np.argmin(can_acc[brake_indices])]
    t_peak_brake = time_s[peak_brake_idx]
    print(f"    Diagnostic Severe Braking Event at t = {t_peak_brake:.1f} s (a_CAN = {can_acc[peak_brake_idx]:.2f} m/s²)")

    filter_latencies = {}
    for fc in cutoffs:
        # Measure time of minimum forward acceleration in local window [peak - 2s, peak + 2s]
        win_sl = slice(max(0, peak_brake_idx - 20), min(n, peak_brake_idx + 21))
        local_min_idx = (peak_brake_idx - 20) + np.argmin(acc_lp[fc][win_sl, 0])
        delay_ms = (time_s[local_min_idx] - t_peak_brake) * 1000.0
        attenuation = float(np.min(acc_lp[fc][win_sl, 0]) - np.min(acc_v[win_sl, 0]))
        filter_latencies[f"{fc}Hz"] = {
            'fc_hz': fc,
            'peak_delay_ms': float(delay_ms),
            'attenuation_ms2': attenuation
        }
        print(f"      fc = {fc:.1f} Hz: Peak Delay = {delay_ms:+.1f} ms | Transient Attenuation = {attenuation:+.2f} m/s²")

    # Condition C: Adaptive Weighting (q(k))
    print("    Computing Causal Adaptive Vibration Metric q(k)...")
    acc_mag = np.linalg.norm(acc_v, axis=1)
    rolling_std = np.zeros(n)
    w_k = 10  # 1.0 s trailing window
    for i in range(n):
        sl = acc_mag[max(0, i - w_k + 1) : i + 1]
        rolling_std[i] = np.std(sl) if len(sl) > 1 else 0.15

    sigma_nom = 0.15
    q_k = np.maximum(0.0, (rolling_std**2 - sigma_nom**2) / (sigma_nom**2))
    print(f"    q(k) Statistics: Min={np.min(q_k):.2f}, Median={np.median(q_k):.2f}, Mean={np.mean(q_k):.2f}, Max={np.max(q_k):.2f}")

    # Sub-variant C2: Adaptive Signal Blending
    lam_blend = 0.05
    gamma_k = (lam_blend * q_k) / (1.0 + lam_blend * q_k)
    acc_adapt_blend = (1.0 - gamma_k[:, None]) * acc_v + gamma_k[:, None] * acc_lp[1.0]

    # Condition D: Oracle Reference Diagnostic
    # In vehicle frame: forward = CAN long accel, lateral = CAN lat accel, vertical = 9.80665 m/s^2
    acc_oracle = np.column_stack([can_acc, can_lat, 9.80665 * np.ones(n)])

    # 3. Multi-Condition Navigation Evaluation Across Rolling Outage Windows
    print("\n[3] Simulating Rolling GNSS Outages (3D ESKF + NHC Pipeline)...", flush=True)
    horizons = [5.0, 10.0, 20.0, 30.0, 60.0]
    stride_s = 20.0
    stride_k = int(round(stride_s / dt))  # 20.0 s stride (N ~ 52 windows per horizon across the 1,104 s trip)
    nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)
    veh_heading_arr = df_v['veh_heading_deg'].values

    def run_outage_window(acc_series, kw, w_dur, dynamic_sigma=False):
        eskf = ESKF3D(
            init_pos_enu=(gt_e[kw], gt_n[kw], gt_u[kw]),
            init_vel_enu=(gt_ve[kw], gt_vn[kw], gt_vu[kw]),
            init_heading_deg=float(veh_heading_arr[kw]),
            init_ba=ba_stat,
            R_vp=np.eye(3),
            sigma_a=0.15,
            gravity=9.80665
        )
        for k in range(kw, kw + w_dur):
            if dynamic_sigma:
                eskf.sigma_a = 0.15 * np.sqrt(1.0 + 0.1 * q_k[k])
            eskf.predict(acc_series[k, 0], acc_series[k, 1], acc_series[k, 2],
                         gyro_v[k, 0], gyro_v[k, 1], gyro_v[k, 2], dt)
            nhc.update_eskf(eskf)

        st = eskf.get_state()
        gt_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur], gt_u[kw + w_dur]])
        pos_err_2d = float(np.linalg.norm(st['pos_n'][:2] - gt_end[:2]))
        vel_end = st['vel_n']
        gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur], gt_vu[kw + w_dur]])
        vel_err_2d = float(np.linalg.norm(vel_end[:2] - gt_v_end[:2]))
        return pos_err_2d, vel_err_2d

    # Outage results storage
    outage_results = {}
    
    # Conditions to benchmark
    conditions_to_run = {
        'A_Raw': {'acc': acc_raw, 'dyn_sigma': False, 'label': 'A. Raw IMU'},
        'B_LP_0.5Hz': {'acc': acc_lp[0.5], 'dyn_sigma': False, 'label': 'B. Fixed LP (0.5 Hz)'},
        'B_LP_1.0Hz': {'acc': acc_lp[1.0], 'dyn_sigma': False, 'label': 'B. Fixed LP (1.0 Hz)'},
        'B_LP_2.0Hz': {'acc': acc_lp[2.0], 'dyn_sigma': False, 'label': 'B. Fixed LP (2.0 Hz)'},
        'C1_Covariance_Only': {'acc': acc_raw, 'dyn_sigma': True, 'label': 'C1. Covariance Only'},
        'C2_Adaptive_Blend': {'acc': acc_adapt_blend, 'dyn_sigma': False, 'label': 'C2. Adaptive Blend'},
        'C3_Combined': {'acc': acc_adapt_blend, 'dyn_sigma': True, 'label': 'C3. Combined (Blend+Cov)'},
        'D_Oracle_CAN': {'acc': acc_oracle, 'dyn_sigma': False, 'label': 'D. Oracle Reference'}
    }

    # Regime masks for window classification
    # Calculate regime tags for each window start
    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        outage_results[h_key] = {
            'horizon_s': h,
            'n_windows': 0,
            'conditions': {c: {'drift_m': [], 'vel_err_ms': [], 'drift_pct': []} for c in conditions_to_run},
            'regimes': {c: {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []} for c in conditions_to_run}
        }

        window_starts = list(range(100, n - w_dur - 1, stride_k))
        outage_results[h_key]['n_windows'] = len(window_starts)
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
            is_rough = float(np.var(win_pvert)) > 0.05
            is_cruising = (float(np.mean(win_speed)) > 5.0) and (not is_braking) and (not is_rough)

            if is_standstill:
                reg_tag = 'standstill'
            elif is_braking:
                reg_tag = 'braking'
            elif is_rough:
                reg_tag = 'rough_road'
            elif is_cruising:
                reg_tag = 'cruising'
            else:
                reg_tag = 'cruising'

            for cname, ccfg in conditions_to_run.items():
                p_err, v_err = run_outage_window(ccfg['acc'], kw, w_dur, dynamic_sigma=ccfg['dyn_sigma'])
                drift_pct = (p_err / max(dist_traveled, 1.0)) * 100.0 if dist_traveled > 5.0 else 0.0

                outage_results[h_key]['conditions'][cname]['drift_m'].append(p_err)
                outage_results[h_key]['conditions'][cname]['vel_err_ms'].append(v_err)
                outage_results[h_key]['conditions'][cname]['drift_pct'].append(drift_pct)
                outage_results[h_key]['regimes'][cname][reg_tag].append(p_err)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(window_starts):
                print(f"    Horizon {h_key}: completed {idx+1}/{len(window_starts)} windows ({((idx+1)/len(window_starts))*100:.0f}%) in {time.time()-t_h:.1f}s", flush=True)

        print(f"\n  --- Horizon {h_key} Results (N={len(window_starts)} windows, time={time.time()-t_h:.1f}s) ---", flush=True)
        print(f"    {'Condition':<25} | {'Mean Drift':>10} | {'Median Drift':>12} | {'P90 Drift':>10} | {'Vel Err':>8}", flush=True)
        print("    " + "-" * 75, flush=True)
        for cname in conditions_to_run:
            drifts = outage_results[h_key]['conditions'][cname]['drift_m']
            verrs = outage_results[h_key]['conditions'][cname]['vel_err_ms']
            m_drift = np.mean(drifts)
            med_drift = np.median(drifts)
            p90_drift = np.percentile(drifts, 90)
            m_verr = np.mean(verrs)
            print(f"    {conditions_to_run[cname]['label']:<25} | {m_drift:8.2f} m | {med_drift:10.2f} m | {p90_drift:8.2f} m | {m_verr:6.2f} m/s", flush=True)

    # 4. Acceleration Signal Metrics
    print("\n[4] Computing Acceleration Signal Errors vs Reference...", flush=True)
    accel_metrics = {}
    for cname, ccfg in conditions_to_run.items():
        if cname == 'D_Oracle_CAN':
            continue
        asig = ccfg['acc'][:, 0] - ba_stat[0]
        ref = can_acc
        err = asig - ref
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))
        corr = float(np.corrcoef(asig, ref)[0, 1])
        accel_metrics[cname] = {
            'mae_ms2': mae,
            'rmse_ms2': rmse,
            'bias_ms2': bias,
            'correlation': corr
        }
        print(f"    {conditions_to_run[cname]['label']:<25}: MAE={mae:.4f} m/s², RMSE={rmse:.4f} m/s², Bias={bias:+.4f} m/s², r={corr:.4f}", flush=True)

    # 5. Export Structured JSON
    print("\n[5] Saving Deliverable: results/c5_5_4_signal_conditioning_ablation.json...", flush=True)
    export_summary = {
        'metadata': {
            'stage': 'C5.5.4',
            'title': 'End-to-End Signal Conditioning Ablation',
            'control_trip': 'Vta02',
            'pipeline': '3D ESKF + NHC',
            'sampling_rate_hz': 10.0,
            'stride_s': stride_s,
            'horizons_s': horizons
        },
        'filter_latencies': filter_latencies,
        'acceleration_metrics': accel_metrics,
        'outage_horizon_summary': {}
    }

    for h_key in outage_results:
        export_summary['outage_horizon_summary'][h_key] = {
            'horizon_s': outage_results[h_key]['horizon_s'],
            'n_windows': outage_results[h_key]['n_windows'],
            'conditions': {}
        }
        for cname in conditions_to_run:
            drifts = outage_results[h_key]['conditions'][cname]['drift_m']
            verrs = outage_results[h_key]['conditions'][cname]['vel_err_ms']
            dpcts = outage_results[h_key]['conditions'][cname]['drift_pct']
            reg_dict = outage_results[h_key]['regimes'][cname]

            export_summary['outage_horizon_summary'][h_key]['conditions'][cname] = {
                'label': conditions_to_run[cname]['label'],
                'mean_drift_m': float(np.mean(drifts)),
                'median_drift_m': float(np.median(drifts)),
                'std_drift_m': float(np.std(drifts)),
                'p75_drift_m': float(np.percentile(drifts, 75)),
                'p90_drift_m': float(np.percentile(drifts, 90)),
                'p95_drift_m': float(np.percentile(drifts, 95)),
                'mean_vel_err_ms': float(np.mean(verrs)),
                'median_vel_err_ms': float(np.median(verrs)),
                'mean_drift_pct': float(np.mean(dpcts)),
                'regime_means_m': {
                    'cruising': float(np.mean(reg_dict['cruising'])) if len(reg_dict['cruising']) > 0 else 0.0,
                    'braking': float(np.mean(reg_dict['braking'])) if len(reg_dict['braking']) > 0 else 0.0,
                    'rough_road': float(np.mean(reg_dict['rough_road'])) if len(reg_dict['rough_road']) > 0 else 0.0,
                    'standstill': float(np.mean(reg_dict['standstill'])) if len(reg_dict['standstill']) > 0 else 0.0
                }
            }

    json_path = RES_DIR / "c5_5_4_signal_conditioning_ablation.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_summary, f, indent=2)
    print(f"    Saved structured JSON ({json_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # 6. Generate Diagnostic Figures
    print("\n[6] Rendering Diagnostic Figures...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: c5_5_4_horizon_drift_comparison.png
    print("    Plot 1: Horizon Drift Comparison (Raw vs LP vs Adaptive vs Oracle)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h_vals = horizons

    # Panel A: Mean Position Drift
    for cname, col, ls, m in [
        ('A_Raw', 'black', '--', 'o'),
        ('B_LP_1.0Hz', 'dodgerblue', '-', 's'),
        ('C2_Adaptive_Blend', 'darkorange', '-', '^'),
        ('C3_Combined', 'crimson', '-', 'D'),
        ('D_Oracle_CAN', 'forestgreen', '-.', 'x')
    ]:
        means = [export_summary['outage_horizon_summary'][f"{int(h)}s"]['conditions'][cname]['mean_drift_m'] for h in h_vals]
        axes[0].plot(h_vals, means, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=conditions_to_run[cname]['label'])

    axes[0].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[0].set_title('Stage C5.5.4: Mean Position Drift vs. Outage Horizon (Vta02)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(fontsize=10)

    # Panel B: Median Position Drift
    for cname, col, ls, m in [
        ('A_Raw', 'black', '--', 'o'),
        ('B_LP_1.0Hz', 'dodgerblue', '-', 's'),
        ('C2_Adaptive_Blend', 'darkorange', '-', '^'),
        ('C3_Combined', 'crimson', '-', 'D'),
        ('D_Oracle_CAN', 'forestgreen', '-.', 'x')
    ]:
        meds = [export_summary['outage_horizon_summary'][f"{int(h)}s"]['conditions'][cname]['median_drift_m'] for h in h_vals]
        axes[1].plot(h_vals, meds, color=col, ls=ls, marker=m, lw=2.0, ms=7, label=conditions_to_run[cname]['label'])

    axes[1].set_xlabel('Outage Horizon (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Median Position Drift (meters)', fontsize=11, fontweight='bold')
    axes[1].set_title('Stage C5.5.4: Median Position Drift vs. Outage Horizon (Vta02)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_5_4_horizon_drift_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: c5_5_4_regime_drift_breakdown.png
    print("    Plot 2: Regime Drift Breakdown at 30s Outage...")
    fig, ax = plt.subplots(figsize=(12, 6))
    h30 = export_summary['outage_horizon_summary']['30s']['conditions']
    regimes = ['cruising', 'braking', 'rough_road', 'standstill']
    reg_labels = ['Steady Cruising\n(v > 5 m/s)', 'Severe Braking\n(a < -1.5 m/s²)', 'Rough Road\n(high vert var)', 'Standstill\n(v < 0.15 m/s)']

    x = np.arange(len(regimes))
    width = 0.16

    plot_conds = [
        ('A_Raw', 'Raw IMU', 'black'),
        ('B_LP_1.0Hz', 'LP 1.0Hz', 'dodgerblue'),
        ('C1_Covariance_Only', 'C1 Cov-Only', 'purple'),
        ('C2_Adaptive_Blend', 'C2 Blend', 'darkorange'),
        ('C3_Combined', 'C3 Combined', 'crimson')
    ]

    for idx, (cname, clbl, col) in enumerate(plot_conds):
        vals = [h30[cname]['regime_means_m'][r] for r in regimes]
        ax.bar(x + (idx - 2) * width, vals, width, label=clbl, color=col, alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean 30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.4: 30 s Outage Drift Stratified by Vehicle Regime (Vta02)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5, axis='y')

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_5_4_regime_drift_breakdown.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Figure 3: c5_5_4_filter_latency_and_transient_distortion.png
    print("    Plot 3: Filter Latency & Transient Distortion during Severe Braking...")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    t_start_zoom = t_peak_brake - 4.0
    t_end_zoom = t_peak_brake + 4.0
    mask_zoom = (time_s >= t_start_zoom) & (time_s <= t_end_zoom)

    axes[0].plot(time_s[mask_zoom], can_acc[mask_zoom], 'forestgreen', lw=2.5, label='Chassis CAN Reference')
    axes[0].plot(time_s[mask_zoom], acc_v[mask_zoom, 0] - ba_stat[0], 'black', alpha=0.5, lw=1.2, label='Raw Leveled Phone')
    axes[0].plot(time_s[mask_zoom], acc_lp[0.5][mask_zoom, 0] - ba_stat[0], 'dodgerblue', lw=1.8, ls='--', label='Fixed LP (0.5 Hz) - Heavy Lag')
    axes[0].plot(time_s[mask_zoom], acc_lp[1.0][mask_zoom, 0] - ba_stat[0], 'blue', lw=1.8, label='Fixed LP (1.0 Hz)')
    axes[0].plot(time_s[mask_zoom], acc_adapt_blend[mask_zoom, 0] - ba_stat[0], 'darkorange', lw=2.0, label='Adaptive Blend C2')

    axes[0].axvline(t_peak_brake, color='gray', ls=':', alpha=0.7, label=f'True Decel Peak (t={t_peak_brake:.1f}s)')
    axes[0].set_ylabel('Forward Acceleration (m/s²)', fontsize=11, fontweight='bold')
    axes[0].set_title('Stage C5.5.4: Signal Conditioning Waveforms during Severe Braking Event', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower left', fontsize=9.5)
    axes[0].grid(True, alpha=0.5)

    # Panel B: Difference from CAN Reference (Instantaneous Error)
    axes[1].plot(time_s[mask_zoom], (acc_v[mask_zoom, 0] - ba_stat[0]) - can_acc[mask_zoom], 'black', alpha=0.6, lw=1.2, label='Raw Error')
    axes[1].plot(time_s[mask_zoom], (acc_lp[0.5][mask_zoom, 0] - ba_stat[0]) - can_acc[mask_zoom], 'dodgerblue', lw=1.8, ls='--', label='LP 0.5 Hz Error (Phase Lag Distortion)')
    axes[1].plot(time_s[mask_zoom], (acc_lp[1.0][mask_zoom, 0] - ba_stat[0]) - can_acc[mask_zoom], 'blue', lw=1.8, label='LP 1.0 Hz Error')
    axes[1].plot(time_s[mask_zoom], (acc_adapt_blend[mask_zoom, 0] - ba_stat[0]) - can_acc[mask_zoom], 'darkorange', lw=2.0, label='Adaptive C2 Error')
    axes[1].axhline(0.0, color='gray', ls='--', lw=1.0)
    axes[1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Error vs. Reference (m/s²)', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9.5)
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_5_4_filter_latency_and_transient_distortion.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # Figure 4: c5_5_4_adaptive_ablation_breakdown.png
    print("    Plot 4: Ablation Breakdown (Smoothing vs. Adaptive Confidence Weighting)...")
    fig, ax = plt.subplots(figsize=(11, 6))
    ablation_conds = [
        ('A_Raw', 'A. Raw IMU\n(Baseline)', 'black'),
        ('B_LP_0.5Hz', 'B. Fixed LP\n(0.5 Hz)', 'lightsteelblue'),
        ('B_LP_1.0Hz', 'B. Fixed LP\n(1.0 Hz)', 'cornflowerblue'),
        ('B_LP_2.0Hz', 'B. Fixed LP\n(2.0 Hz)', 'royalblue'),
        ('C1_Covariance_Only', 'C1. Covariance\nOnly (Dyn Sigma)', 'purple'),
        ('C2_Adaptive_Blend', 'C2. Adaptive\nSignal Blend', 'darkorange'),
        ('C3_Combined', 'C3. Combined\n(Blend + Cov)', 'crimson')
    ]

    labels = [a[1] for a in ablation_conds]
    means_30s = [h30[a[0]]['mean_drift_m'] for a in ablation_conds]
    meds_30s = [h30[a[0]]['median_drift_m'] for a in ablation_conds]
    colors = [a[2] for a in ablation_conds]

    x = np.arange(len(ablation_conds))
    w = 0.35
    b1 = ax.bar(x - w/2, means_30s, w, label='Mean 30 s Drift', color=colors, alpha=0.85, edgecolor='black')
    b2 = ax.bar(x + w/2, meds_30s, w, label='Median 30 s Drift', color=colors, alpha=0.45, hatch='//', edgecolor='black')

    # Value annotations on top of bars
    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f}m', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.4: Ablation — Smoothing vs. Adaptive Confidence Weighting (Vta02)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5, axis='y')

    plt.tight_layout()
    fig4_path = FIG_DIR / "c5_5_4_adaptive_ablation_breakdown.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()

    print("\nAll 4 figures saved successfully to results/figures/.", flush=True)
    print("Stage C5.5.4 ablation execution completed successfully.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
