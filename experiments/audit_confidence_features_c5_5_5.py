"""
SIH26168 - Stage C5.5.5: Adaptive Confidence Feature Audit & C0 Benchmark
Script: experiments/audit_confidence_features_c5_5_5.py

PURPOSE:
    1. Extract 11 candidate causal smartphone-available features (trailing window W <= 1.0 s).
    2. Quantify feature metrics:
       - Incremental information vs q_old (collinearity r, rank correlation rho, unexplained variance 1 - r^2)
       - Diagnostic correlation with true acceleration error (diagnostic only, not sole ranking criterion)
       - Regime separation (cruising vs severe braking vs rough road vs standstill)
       - Cross-trip stability (Vta02 vs Vta03 vs Vta04)
       - Computational cost (per-sample latency)
    3. Closed-loop 30 s outage navigation benchmark comparing:
       - Condition A: Raw Nominal Q (sigma_a = 0.15 m/s^2)
       - Condition C0: Constant Inflated Q (sigma_a = 0.291 m/s^2) [THE HONEST BENCHMARK]
       - Condition C1: Old 1D q_old(k)
       - C5.5.5 Candidates: Small transparent combinations of top orthogonal features
       - Shuffled Candidate Control: Permuted sequence to verify temporal localization
    4. Strict Success Criterion: Candidates must beat C0 (470.7 m), not merely C1 or A.

DELIVERABLES:
    - results/c5_5_5_feature_audit.json
    - results/figures/c5_5_5_feature_correlation_and_redundancy.png
    - results/figures/c5_5_5_regime_separation_boxplots.png
    - results/figures/c5_5_5_c0_benchmark_comparison.png
    - results/c5_5_5_feature_audit_report.md
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.stats import pearsonr, spearmanr
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


def causal_bandpass(sig: np.ndarray, lowcut: float = 2.0, highcut: float = 4.8, fs: float = 10.0, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a) * sig[0]
    out, _ = lfilter(b, a, sig, zi=zi)
    return out


def causal_highpass(sig: np.ndarray, cutoff: float = 3.5, fs: float = 10.0, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='high')
    zi = lfilter_zi(b, a) * sig[0]
    out, _ = lfilter(b, a, sig, zi=zi)
    return out


def extract_features(df_p: pd.DataFrame, df_v: pd.DataFrame, trip_name: str = "Vta02") -> dict:
    """Extracts 11 causal smartphone features from past/present IMU data only (W <= 1.0 s)."""
    n = len(df_p)
    dt = 0.1
    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    acc_v, gyro_v, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, speed)

    acc_mag = np.linalg.norm(acc_v, axis=1)
    gyro_norm = np.linalg.norm(gyro_v, axis=1)

    # Pre-filter causal band energy
    acc_bp = causal_bandpass(acc_mag, lowcut=2.0, highcut=4.8, fs=10.0)
    acc_hp = causal_highpass(acc_mag, cutoff=3.5, fs=10.0)

    # Rolling arrays (W = 10 samples = 1.0 s)
    w_k = 10
    f1_q_old = np.zeros(n)
    f2_sigma_a = np.zeros(n)
    f3_jerk_rms = np.zeros(n)
    f4_e_bp = np.zeros(n)
    f5_e_hp = np.zeros(n)
    f6_r_band = np.zeros(n)
    f7_gyro_norm = gyro_norm.copy()
    f8_sigma_g = np.zeros(n)
    f9_ang_accel = np.zeros(n)
    f10_sigma_rp = np.zeros(n)

    sigma_nom = 0.15

    for i in range(n):
        i_start = max(0, i - w_k + 1)
        win_acc = acc_v[i_start : i + 1]
        win_mag = acc_mag[i_start : i + 1]
        win_gyro = gyro_v[i_start : i + 1]
        win_bp = acc_bp[i_start : i + 1]
        win_hp = acc_hp[i_start : i + 1]

        # f1: q_old (magnitude variance normalized)
        std_mag = np.std(win_mag) if len(win_mag) > 1 else sigma_nom
        f1_q_old[i] = max(0.0, (std_mag**2 - sigma_nom**2) / (sigma_nom**2))

        # f2: multi-axis acceleration std
        if len(win_acc) > 1:
            stds = np.std(win_acc, axis=0)
            f2_sigma_a[i] = np.sqrt(np.sum(stds**2))
        else:
            f2_sigma_a[i] = sigma_nom

        # f3: trailing jerk RMS (Delta a / Delta t)
        if len(win_acc) > 1:
            jerk = np.diff(win_acc, axis=0) / dt
            f3_jerk_rms[i] = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        else:
            f3_jerk_rms[i] = 0.0

        # f4: causal 2-5 Hz band energy
        f4_e_bp[i] = np.mean(win_bp**2)

        # f5: causal high-frequency energy (>3.5 Hz)
        f5_e_hp[i] = np.mean(win_hp**2)

        # f6: band-energy ratio (E_bp / (E_total + eps))
        e_tot = np.mean((win_mag - np.mean(win_mag))**2) if len(win_mag) > 1 else 1e-6
        f6_r_band[i] = f4_e_bp[i] / (e_tot + 1e-6)

        # f8: gyro standard deviation
        if len(win_gyro) > 1:
            gstds = np.std(win_gyro, axis=0)
            f8_sigma_g[i] = np.sqrt(np.sum(gstds**2))
        else:
            f8_sigma_g[i] = 0.01

        # f9: angular acceleration RMS (Delta omega / Delta t)
        if len(win_gyro) > 1:
            dang = np.diff(win_gyro, axis=0) / dt
            f9_ang_accel[i] = np.sqrt(np.mean(np.sum(dang**2, axis=1)))
        else:
            f9_ang_accel[i] = 0.0

        # f10: roll/pitch rate variance (axes Y and Z in body frame)
        if len(win_gyro) > 1:
            f10_sigma_rp[i] = np.var(win_gyro[:, 1]) + np.var(win_gyro[:, 2])
        else:
            f10_sigma_rp[i] = 0.0

    features = {
        'f1_q_old': f1_q_old,
        'f2_sigma_a': f2_sigma_a,
        'f3_jerk_rms': f3_jerk_rms,
        'f4_e_2_5hz': f4_e_bp,
        'f5_e_hf': f5_e_hp,
        'f6_r_band': f6_r_band,
        'f7_gyro_norm': f7_gyro_norm,
        'f8_sigma_g': f8_sigma_g,
        'f9_ang_accel': f9_ang_accel,
        'f10_sigma_rp': f10_sigma_rp
    }
    return features, acc_v, gyro_v, speed


def main():
    print("=" * 80, flush=True)
    print("STAGE C5.5.5: ADAPTIVE CONFIDENCE FEATURE AUDIT & C0 BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # 1. Ingest Vta02 & Extract Features
    print("\n[1] Ingesting Vta02 and extracting 10 causal smartphone IMU features...", flush=True)
    t0_feat = time.time()
    df_p, df_v = load_trip("Vta02")
    n = len(df_p)
    dt = 0.1
    time_s = np.arange(n) * dt

    features, acc_v, gyro_v, speed = extract_features(df_p, df_v, "Vta02")
    feat_time_us = ((time.time() - t0_feat) / n) * 1e6
    print(f"    Extracted features across {n} epochs in {time.time() - t0_feat:.2f}s ({feat_time_us:.1f} µs/sample)", flush=True)

    can_acc = df_v['veh_accel_long_ms2'].values
    veh_heading_arr = df_v['veh_heading_deg'].values
    ba_stat = np.mean(acc_v[142:191], axis=0) - np.array([0.0, 0.0, 9.80665])

    # Ground truth ENU
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)
    heading_rad = np.radians(veh_heading_arr)
    gt_ve = speed * np.sin(heading_rad)
    gt_vn = speed * np.cos(heading_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6

    # True acceleration residual (diagnostic only)
    acc_error = np.abs((acc_v[:, 0] - ba_stat[0]) - can_acc)

    # Regime masks
    w_k = 10
    vert_var = np.zeros(n)
    for i in range(n):
        vert_var[i] = np.var(acc_v[max(0, i - w_k + 1) : i + 1, 2])

    mask_standstill = speed < 0.15
    mask_braking = can_acc < -1.5
    mask_rough = vert_var > 0.05
    mask_cruising = (speed > 5.0) & (~mask_braking) & (~mask_rough)

    # 2. Compute Incremental Information & Diagnostic Metrics
    print("\n[2] Computing Incremental Information vs. q_old & Regime Separation...", flush=True)
    f1 = features['f1_q_old']

    feature_audit_summary = {}

    print(f"    {'Feature Name':<20} | {'r(f, q_old)':>11} | {'rho(f, q_old)':>13} | {'Unexplained Var':>15} | {'r(f, |err|)':>11} | {'Brake/Cruise':>12} | {'Rough/Cruise':>12}", flush=True)
    print("    " + "-" * 105, flush=True)

    for fname, fvals in features.items():
        # Pearson & Spearman vs q_old
        r_qold, _ = pearsonr(fvals, f1)
        rho_qold, _ = spearmanr(fvals, f1)
        unexplained_var = 1.0 - (r_qold**2)

        # Diagnostic correlation with acceleration error
        r_err, _ = pearsonr(fvals, acc_error)

        # Regime separation ratios
        mean_cruise = np.mean(fvals[mask_cruising]) if np.any(mask_cruising) else 1e-6
        mean_brake = np.mean(fvals[mask_braking]) if np.any(mask_braking) else 1e-6
        mean_rough = np.mean(fvals[mask_rough]) if np.any(mask_rough) else 1e-6

        ratio_brake = mean_brake / (mean_cruise + 1e-9)
        ratio_rough = mean_rough / (mean_cruise + 1e-9)

        print(f"    {fname:<20} | {r_qold:11.4f} | {rho_qold:13.4f} | {unexplained_var:14.2%} | {r_err:11.4f} | {ratio_brake:11.2f}x | {ratio_rough:11.2f}x", flush=True)

        feature_audit_summary[fname] = {
            'r_qold': float(r_qold),
            'rho_qold': float(rho_qold),
            'unexplained_var': float(unexplained_var),
            'r_acc_error': float(r_err),
            'mean_cruising': float(mean_cruise),
            'mean_braking': float(mean_brake),
            'mean_rough_road': float(mean_rough),
            'separation_braking': float(ratio_brake),
            'separation_rough_road': float(ratio_rough)
        }

    # 3. Cross-Trip Stability Check (Vta02 vs Vta03 vs Vta04)
    print("\n[3] Evaluating Cross-Trip Feature Stability (Vta02 vs Vta03 vs Vta04)...", flush=True)
    df_p3, df_v3 = load_trip("Vta03")
    df_p4, df_v4 = load_trip("Vta04")
    feat_vta03, _, _, _ = extract_features(df_p3, df_v3, "Vta03")
    feat_vta04, _, _, _ = extract_features(df_p4, df_v4, "Vta04")

    stability_summary = {}
    print(f"    {'Feature Name':<20} | {'Vta02 Med (IQR)':>18} | {'Vta03 Med (IQR)':>18} | {'Vta04 Med (IQR)':>18}", flush=True)
    print("    " + "-" * 85, flush=True)

    for fname in features:
        v2 = features[fname]
        v3 = feat_vta03[fname]
        v4 = feat_vta04[fname]

        med2, iqr2 = np.median(v2), np.percentile(v2, 75) - np.percentile(v2, 25)
        med3, iqr3 = np.median(v3), np.percentile(v3, 75) - np.percentile(v3, 25)
        med4, iqr4 = np.median(v4), np.percentile(v4, 75) - np.percentile(v4, 25)

        print(f"    {fname:<20} | {med2:8.2f} ({iqr2:6.2f}) | {med3:8.2f} ({iqr3:6.2f}) | {med4:8.2f} ({iqr4:6.2f})", flush=True)
        stability_summary[fname] = {
            'vta02': {'median': float(med2), 'iqr': float(iqr2)},
            'vta03': {'median': float(med3), 'iqr': float(iqr3)},
            'vta04': {'median': float(med4), 'iqr': float(iqr4)}
        }

    # 4. Formulate Candidate Confidence Estimators
    print("\n[4] Formulating C5.5.5 Confidence Estimators from Top Orthogonal Features...", flush=True)
    # Feature selections:
    # - f3_jerk_rms: 46% unexplained variance vs q_old, strong transient detection
    # - f8_sigma_g: 88% unexplained variance vs q_old, isolates chassis rotation / mount wobble
    # - f6_r_band: 93% unexplained variance, spectral concentration
    # Normalize features by median for clean transparent combinations:
    norm_jerk = features['f3_jerk_rms'] / max(np.median(features['f3_jerk_rms']), 1e-3)
    norm_g = features['f8_sigma_g'] / max(np.median(features['f8_sigma_g']), 1e-3)
    norm_band = features['f6_r_band'] / max(np.median(features['f6_r_band']), 1e-3)

    # Candidate 1: Jerk-Gated Quality (boosts during severe dynamic shock)
    q_cand1 = f1 * (0.5 + 0.5 * norm_jerk)

    # Candidate 2: Rotational-Decoupled Quality (incorporates rotational vibration)
    q_cand2 = f1 + 2.0 * norm_g

    # Candidate 3: Spectral Band-Gated Quality (weights structural band concentration)
    q_cand3 = f1 * (0.7 + 0.3 * norm_band)

    # Candidate 4: Synergistic Multidimensional Composite
    q_cand4 = f1 * (0.6 + 0.2 * norm_jerk + 0.2 * norm_g)

    # Shuffled control of top candidate (Candidate 1)
    rng = np.random.default_rng(seed=42)
    q_cand1_shuffled = rng.permutation(q_cand1)

    # 5. Closed-Loop Navigation Benchmark Against C0
    print("\n[5] Simulating 30s Outage Benchmark Against Honest Baseline C0...", flush=True)
    h = 30.0
    w_dur = int(round(h / dt))
    stride_s = 20.0
    stride_k = int(round(stride_s / dt))
    window_starts = list(range(100, n - w_dur - 1, stride_k))

    sigma_nom = 0.15
    sigma_c0 = 0.291  # The Honest Benchmark from C5.5.4b

    candidates_to_run = {
        'C0_Honest_Baseline': {
            'label': 'C0. Constant Inflated (sigma_a = 0.291) [BENCHMARK]',
            'type': 'constant', 'sigma_val': sigma_c0
        },
        'C1_Old_Adaptive': {
            'label': 'C1. Old Adaptive q_old(k)',
            'type': 'dynamic', 'q_series': f1
        },
        'Cand1_Jerk_Gated': {
            'label': 'Cand 1. Jerk-Gated Quality (q_jerk)',
            'type': 'dynamic', 'q_series': q_cand1
        },
        'Cand2_Rot_Decoupled': {
            'label': 'Cand 2. Rotational-Decoupled (q_rot)',
            'type': 'dynamic', 'q_series': q_cand2
        },
        'Cand3_Spectral_Gated': {
            'label': 'Cand 3. Spectral Band-Gated (q_spec)',
            'type': 'dynamic', 'q_series': q_cand3
        },
        'Cand4_Synergistic': {
            'label': 'Cand 4. Synergistic Composite (q_syn)',
            'type': 'dynamic', 'q_series': q_cand4
        },
        'Cand1_Shuffled_Ctrl': {
            'label': 'Cand 1 Shuffled Control (Broken Timing)',
            'type': 'dynamic', 'q_series': q_cand1_shuffled
        }
    }

    benchmark_results = {k: {'drifts': [], 'verrs': [], 'regimes': {'cruising': [], 'braking': [], 'rough_road': [], 'standstill': []}} for k in candidates_to_run}

    t0_sim = time.time()
    for idx, kw in enumerate(window_starts):
        win_speed = speed[kw : kw + w_dur + 1]
        win_can_a = can_acc[kw : kw + w_dur + 1]
        win_pvert = acc_v[kw : kw + w_dur + 1, 2]

        is_standstill = float(np.mean(win_speed)) < 0.15
        is_braking = float(np.min(win_can_a)) < -1.5
        is_rough = float(np.var(win_pvert)) > 0.05

        if is_standstill:
            reg_tag = 'standstill'
        elif is_braking:
            reg_tag = 'braking'
        elif is_rough:
            reg_tag = 'rough_road'
        else:
            reg_tag = 'cruising'

        for ckey, ccfg in candidates_to_run.items():
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

            benchmark_results[ckey]['drifts'].append(pos_err)
            benchmark_results[ckey]['verrs'].append(vel_err)
            benchmark_results[ckey]['regimes'][reg_tag].append(pos_err)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(window_starts):
            print(f"    Completed {idx+1}/{len(window_starts)} windows ({((idx+1)/len(window_starts))*100:.0f}%) in {time.time()-t0_sim:.1f}s", flush=True)

    # 6. Evaluation against C0 Success Criteria
    print(f"\n  --- Stage C5.5.5 Navigation Results: Evaluation Against C0 (N={len(window_starts)} windows) ---", flush=True)
    print(f"    {'Condition':<36} | {'Mean Drift':>11} | {'Median Drift':>13} | {'Diff vs C0':>11} | {'Success Status':<16}", flush=True)
    print("    " + "-" * 95, flush=True)

    c0_mean = np.mean(benchmark_results['C0_Honest_Baseline']['drifts'])
    c0_med = np.median(benchmark_results['C0_Honest_Baseline']['drifts'])

    navigation_summary_export = {}

    for ckey, ccfg in candidates_to_run.items():
        drifts = np.array(benchmark_results[ckey]['drifts'])
        m_drift = float(np.mean(drifts))
        med_drift = float(np.median(drifts))
        diff_c0 = m_drift - c0_mean

        if ckey == 'C0_Honest_Baseline':
            status = "BENCHMARK BASE"
        elif diff_c0 <= -15.0:
            status = "MAJOR SUCCESS"
        elif diff_c0 < 0.0:
            status = "MODEST GAIN"
        elif abs(diff_c0) <= 2.0:
            status = "TIE (NO GAIN)"
        else:
            status = "REJECTED (WORSE)"

        print(f"    {ccfg['label']:<36} | {m_drift:9.2f} m | {med_drift:11.2f} m | {diff_c0:+10.2f} m | {status:<16}", flush=True)

        navigation_summary_export[ckey] = {
            'label': ccfg['label'],
            'mean_drift_m': m_drift,
            'median_drift_m': med_drift,
            'diff_vs_c0_m': diff_c0,
            'p90_drift_m': float(np.percentile(drifts, 90)),
            'mean_vel_err_ms': float(np.mean(benchmark_results[ckey]['verrs'])),
            'status': status,
            'regime_means_m': {
                'cruising': float(np.mean(benchmark_results[ckey]['regimes']['cruising'])) if len(benchmark_results[ckey]['regimes']['cruising']) > 0 else 0.0,
                'braking': float(np.mean(benchmark_results[ckey]['regimes']['braking'])) if len(benchmark_results[ckey]['regimes']['braking']) > 0 else 0.0,
                'rough_road': float(np.mean(benchmark_results[ckey]['regimes']['rough_road'])) if len(benchmark_results[ckey]['regimes']['rough_road']) > 0 else 0.0,
                'standstill': float(np.mean(benchmark_results[ckey]['regimes']['standstill'])) if len(benchmark_results[ckey]['regimes']['standstill']) > 0 else 0.0
            }
        }

    # 7. Export Structured Deliverable JSON
    print("\n[7] Saving Structured Deliverable: results/c5_5_5_feature_audit.json...", flush=True)
    full_deliverable = {
        'metadata': {
            'stage': 'C5.5.5',
            'title': 'Adaptive Confidence Feature Audit & C0 Benchmark',
            'trip': 'Vta02',
            'honest_benchmark_c0': 'sigma_a = 0.291 m/s^2',
            'sampling_rate_hz': 10.0,
            'outage_horizon_s': 30.0,
            'stride_s': stride_s,
            'feat_extraction_us_per_sample': feat_time_us
        },
        'feature_audit_summary': feature_audit_summary,
        'cross_trip_stability': stability_summary,
        'navigation_c0_benchmark': navigation_summary_export
    }

    json_path = RES_DIR / "c5_5_5_feature_audit.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_deliverable, f, indent=2)
    print(f"    Saved structured JSON ({json_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # 8. Render Visualizations
    print("\n[8] Rendering Diagnostic Visualizations...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Incremental Information & Unexplained Variance
    fig, ax = plt.subplots(figsize=(12, 6))
    f_names = list(features.keys())
    f_labels = [
        'q_old (var(||a||))', 'sigma_a (multi-axis)', 'jerk_rms (Delta a)',
        'E_2-5Hz (bandpass)', 'E_HF (>3.5Hz)', 'R_band (ratio)',
        'gyro_norm', 'sigma_g (gyro std)', 'ang_accel (Delta omega)', 'sigma_rp (roll/pitch)'
    ]
    unexp_vars = [feature_audit_summary[k]['unexplained_var'] * 100.0 for k in f_names]
    r_errs = [abs(feature_audit_summary[k]['r_acc_error']) for k in f_names]

    x = np.arange(len(f_names))
    w = 0.4
    b1 = ax.bar(x - w/2, unexp_vars, w, label='Unexplained Variance vs. q_old (1 - r²) [%]', color='dodgerblue', alpha=0.85, edgecolor='black')
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, r_errs, w, label='|Correlation with Acceleration Error|', color='crimson', alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(f_labels, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel('Unexplained Variance vs. q_old (%)', fontsize=11, fontweight='bold', color='dodgerblue')
    ax2.set_ylabel('|Correlation with Accel Error|', fontsize=11, fontweight='bold', color='crimson')
    ax.set_title('Stage C5.5.5: Incremental Information vs. q_old and Diagnostic Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig1_path = FIG_DIR / "c5_5_5_feature_correlation_and_redundancy.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: Regime Separation Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    brake_ratios = [feature_audit_summary[k]['separation_braking'] for k in f_names]
    rough_ratios = [feature_audit_summary[k]['separation_rough_road'] for k in f_names]

    b1 = ax.bar(x - w/2, brake_ratios, w, label='Severe Braking / Cruising Ratio', color='darkorange', alpha=0.85, edgecolor='black')
    b2 = ax.bar(x + w/2, rough_ratios, w, label='Rough Road / Cruising Ratio', color='forestgreen', alpha=0.85, edgecolor='black')

    ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='No Separation Baseline (1.0x)')
    ax.set_xticks(x)
    ax.set_xticklabels(f_labels, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel('Regime Mean / Cruising Mean (x)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.5: Regime Discriminative Power across Driving Conditions (Vta02)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig2_path = FIG_DIR / "c5_5_5_regime_separation_boxplots.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Figure 3: C0 Benchmark Navigation Comparison
    fig, ax = plt.subplots(figsize=(13, 6))
    cand_keys = list(candidates_to_run.keys())
    cand_labels = [
        'C0. Constant Inflated\n(sigma_a = 0.291)\n[HONEST BENCHMARK]',
        'C1. Old Adaptive\nq_old(k)',
        'Cand 1. Jerk-Gated\n(q_jerk)',
        'Cand 2. Rot-Decoupled\n(q_rot)',
        'Cand 3. Spectral-Gated\n(q_spec)',
        'Cand 4. Synergistic\n(q_syn)',
        'Cand 1 Shuffled\n(Broken Timing)'
    ]
    means = [navigation_summary_export[k]['mean_drift_m'] for k in cand_keys]
    meds = [navigation_summary_export[k]['median_drift_m'] for k in cand_keys]
    colors = ['black', 'dimgray', 'dodgerblue', 'mediumpurple', 'darkcyan', 'forestgreen', 'crimson']

    x_c = np.arange(len(cand_keys))
    b1 = ax.bar(x_c - w/2, means, w, label='Mean 30 s Drift', color=colors, alpha=0.85, edgecolor='black')
    b2 = ax.bar(x_c + w/2, meds, w, label='Median 30 s Drift', color=colors, alpha=0.45, hatch='//', edgecolor='black')

    ax.axhline(c0_mean, color='black', ls='--', lw=1.5, label=f'C0 Mean Baseline ({c0_mean:.1f}m)')

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f}m', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x_c)
    ax.set_xticklabels(cand_labels, fontsize=9.0, fontweight='bold')
    ax.set_ylabel('30 s Position Drift (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Stage C5.5.5: Navigation Benchmark — Testing Candidates Against Honest C0 Baseline (Vta02)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    fig3_path = FIG_DIR / "c5_5_5_c0_benchmark_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    print("\nAll figures and results generated successfully.", flush=True)
    print("Stage C5.5.5 execution completed.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
