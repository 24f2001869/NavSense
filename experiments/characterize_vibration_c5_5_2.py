"""
C5.5.2 — Stationary & Regime Vibration Characterization on Vta02
================================================================
Diagnostic investigation to separate:
  A. Vehicle motion (chassis CAN / VBOX ground truth)
  B. Phone mount motion (elastic flexure / vibration)
  C. Sensor noise (MEMS accelerometer noise floor & aliasing)

Regimes analyzed on clean control trip Vta02:
  1. Engine Idling (Stationary): v < 0.15 m/s, RPM ~ 800
     - Examines Block 5 (46s), Block 7 (10.3s), Block 0 (4.9s), Block 1 (3.2s), and aggregate.
  2. Smooth Cruising: v > 10 m/s, |a_CAN| < 0.5 m/s²
  3. Hard Braking: a_CAN < -1.5 m/s²
  4. Rough Road: v > 5 m/s, high vertical accel variance (top 5th percentile)

Frequency Bands evaluated (sampling fs = 10 Hz, Nyquist = 5 Hz):
  - Band 1: 0.0 – 0.5 Hz (quasi-static vehicle dynamics)
  - Band 2: 0.5 – 2.0 Hz (suspension heave/pitch / body bounce)
  - Band 3: 2.0 – 5.0 Hz (mount flutter / engine vibration up to Nyquist)
  - Band 4: > 5.0 Hz (Nyquist limit analysis & aliased engine harmonic folding)

Outputs:
  - results/c5_5_2_vibration_characterization.json
  - results/figures/c5_5_2_stationary_timeseries_and_psd.png
  - results/figures/c5_5_2_regime_comparison_psd.png
  - results/figures/c5_5_2_energy_band_breakdown.png
  - results/figures/c5_5_2_correlation_and_residuals_by_regime.png
  - results/figures/c5_5_2_cross_trip_stationary_benchmark.png
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Spectral & Statistical Helper Functions
# ──────────────────────────────────────────────────────────────────────

def compute_stats(sig):
    """Compute summary statistics for a 1D signal."""
    sig = sig[~np.isnan(sig)]
    if len(sig) == 0:
        return {'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'p2p': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    return {
        'mean': float(np.mean(sig)),
        'std': float(np.std(sig)),
        'rms': float(np.sqrt(np.mean(sig**2))),
        'p2p': float(np.ptp(sig)),
        'min': float(np.min(sig)),
        'max': float(np.max(sig)),
        'count': int(len(sig))
    }

def compute_psd_and_bands(sig, fs=10.0, nperseg=None):
    """Compute Welch PSD and integrate power across defined frequency bands."""
    sig = sig[~np.isnan(sig)]
    n = len(sig)
    if n < 8:
        return {
            'freqs': [], 'psd': [], 'dominant_freq': 0.0,
            'total_power': 0.0,
            'band_0_0_5': 0.0, 'band_0_5_2': 0.0, 'band_2_5': 0.0,
            'pct_0_0_5': 0.0, 'pct_0_5_2': 0.0, 'pct_2_5': 0.0
        }
    
    # Choose nperseg
    if nperseg is None:
        if n >= 256:
            nperseg = 128
        elif n >= 64:
            nperseg = 64
        elif n >= 32:
            nperseg = 32
        else:
            nperseg = n

    # Detrend to remove DC for spectral distribution
    sig_detrended = sig - np.mean(sig)
    freqs, psd = welch(sig_detrended, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')

    # Total power (variance)
    total_power = float(np.var(sig_detrended))
    if total_power <= 1e-12:
        total_power = 1e-12

    # Band masks
    mask_0_0_5 = (freqs >= 0.0) & (freqs < 0.5)
    mask_0_5_2 = (freqs >= 0.5) & (freqs < 2.0)
    mask_2_5 = (freqs >= 2.0) & (freqs <= 5.0)

    # Trapezoidal integration of PSD over bands
    power_0_0_5 = float(np.trapezoid(psd[mask_0_0_5], freqs[mask_0_0_5])) if np.sum(mask_0_0_5) > 1 else 0.0
    power_0_5_2 = float(np.trapezoid(psd[mask_0_5_2], freqs[mask_0_5_2])) if np.sum(mask_0_5_2) > 1 else 0.0
    power_2_5 = float(np.trapezoid(psd[mask_2_5], freqs[mask_2_5])) if np.sum(mask_2_5) > 1 else 0.0

    sum_bands = power_0_0_5 + power_0_5_2 + power_2_5
    if sum_bands > 1e-12:
        pct_0_0_5 = (power_0_0_5 / sum_bands) * 100.0
        pct_0_5_2 = (power_0_5_2 / sum_bands) * 100.0
        pct_2_5 = (power_2_5 / sum_bands) * 100.0
    else:
        pct_0_0_5, pct_0_5_2, pct_2_5 = 0.0, 0.0, 0.0

    # Dominant non-DC frequency peak
    non_dc_mask = freqs > 0.05
    if np.sum(non_dc_mask) > 0:
        dominant_freq = float(freqs[non_dc_mask][np.argmax(psd[non_dc_mask])])
    else:
        dominant_freq = 0.0

    return {
        'freqs': freqs.tolist(),
        'psd': psd.tolist(),
        'dominant_freq': dominant_freq,
        'total_power': total_power,
        'band_0_0_5': power_0_0_5,
        'band_0_5_2': power_0_5_2,
        'band_2_5': power_2_5,
        'pct_0_0_5': pct_0_0_5,
        'pct_0_5_2': pct_0_5_2,
        'pct_2_5': pct_2_5
    }

def analyze_regime_signals(regime_name, mask, data_dict):
    """Compute statistics, correlations, and PSD for all signals in a regime."""
    sub = {k: v[mask] for k, v in data_dict.items()}
    duration_s = float(np.sum(mask) * 0.1)
    n_samples = int(np.sum(mask))

    metrics = {
        'regime': regime_name,
        'duration_s': duration_s,
        'n_samples': n_samples
    }

    # Statistical moments
    metrics['stats'] = {}
    for sig_name in ['can_accel', 'vbox_accel', 'phone_fwd', 'phone_lat', 'phone_vert',
                     'raw_ax', 'raw_ay', 'raw_az', 'residual_can']:
        if sig_name in sub:
            metrics['stats'][sig_name] = compute_stats(sub[sig_name])

    # Correlations
    can = sub['can_accel']
    pfwd = sub['phone_fwd']
    vbox = sub['vbox_accel']

    valid_can = (~np.isnan(can)) & (~np.isnan(pfwd))
    if np.sum(valid_can) > 10 and np.std(can[valid_can]) > 1e-4 and np.std(pfwd[valid_can]) > 1e-4:
        r_can = float(np.corrcoef(pfwd[valid_can], can[valid_can])[0, 1])
    else:
        r_can = 0.0

    valid_vbox = (~np.isnan(vbox)) & (~np.isnan(pfwd))
    if np.sum(valid_vbox) > 10 and np.std(vbox[valid_vbox]) > 1e-4 and np.std(pfwd[valid_vbox]) > 1e-4:
        r_vbox = float(np.corrcoef(pfwd[valid_vbox], vbox[valid_vbox])[0, 1])
    else:
        r_vbox = 0.0

    metrics['correlations'] = {
        'corr_phone_fwd_vs_can': r_can,
        'corr_phone_fwd_vs_vbox': r_vbox
    }

    # Spectral analysis
    metrics['spectral'] = {
        'phone_fwd': compute_psd_and_bands(sub['phone_fwd']),
        'can_accel': compute_psd_and_bands(sub['can_accel']),
        'phone_lat': compute_psd_and_bands(sub['phone_lat']),
        'phone_vert': compute_psd_and_bands(sub['phone_vert']),
        'residual_can': compute_psd_and_bands(sub['residual_can'])
    }

    return metrics

# ──────────────────────────────────────────────────────────────────────
# Main Execution Flow
# ──────────────────────────────────────────────────────────────────────

def run_vibration_characterization():
    print("=" * 75)
    print("  STAGE C5.5.2: STATIONARY & REGIME VIBRATION CHARACTERIZATION")
    print("=" * 75)

    # ── 1. Load Vta02 (Clean Control) ──
    print("\n[1] Loading Vta02 (Clean Control Trip)...")
    df_p, df_v = load_trip("Vta02")

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    veh_speed = df_v['veh_speed_ms'].values
    can_acc = df_v['veh_accel_long_ms2'].values
    engine_rpm = df_v['engine_rpm'].values if 'engine_rpm' in df_v.columns else np.zeros(len(veh_speed))
    time_s = np.arange(len(veh_speed)) * 0.1

    # VBOX derived reference acceleration
    vbox_acc = np.gradient(veh_speed, 0.1)

    # Align phone to vehicle coordinates
    accel_veh, gyro_veh, R_pv, angles = align_phone_to_vehicle(raw_acc, raw_gyro, veh_speed)
    phone_fwd = accel_veh[:, 0]
    phone_lat = accel_veh[:, 1]
    phone_vert = accel_veh[:, 2]

    residual_can = phone_fwd - can_acc

    data_dict = {
        'can_accel': can_acc,
        'vbox_accel': vbox_acc,
        'phone_fwd': phone_fwd,
        'phone_lat': phone_lat,
        'phone_vert': phone_vert,
        'raw_ax': raw_acc[:, 0],
        'raw_ay': raw_acc[:, 1],
        'raw_az': raw_acc[:, 2],
        'residual_can': residual_can,
        'engine_rpm': engine_rpm,
        'veh_speed': veh_speed
    }

    n_total = len(veh_speed)
    print(f"    Total duration: {n_total*0.1:.1f} s ({n_total} samples)")
    print(f"    Phone leveling angles: roll={angles['roll_deg']:.2f}°, pitch={angles['pitch_deg']:.2f}°, yaw={angles['yaw_deg']:.2f}°")

    # ── 2. Identify Regimes on Vta02 ──
    print("\n[2] Segmenting Vta02 into Physical Motion Regimes...")

    # Regime A: Stationary / Idle
    stat_mask = veh_speed < 0.15
    stat_indices = np.where(stat_mask)[0]
    diffs = np.diff(stat_indices)
    splits = np.where(diffs > 1)[0] + 1
    stat_blocks = np.split(stat_indices, splits)

    block_info = []
    for i, b in enumerate(stat_blocks):
        dur = len(b) * 0.1
        mean_rpm = float(np.mean(engine_rpm[b])) if len(b) > 0 else 0.0
        block_info.append({
            'block_id': i,
            'start_idx': int(b[0]),
            'end_idx': int(b[-1]),
            'n_samples': int(len(b)),
            'duration_s': dur,
            'mean_rpm': mean_rpm
        })
    
    # Sort blocks by length
    long_blocks = [b for b in block_info if b['duration_s'] >= 3.0]
    print(f"    Total Stationary: {np.sum(stat_mask)*0.1:.1f} s across {len(stat_blocks)} blocks")
    for b in long_blocks:
        print(f"      - Block {b['block_id']}: idx {b['start_idx']}..{b['end_idx']} ({b['duration_s']:.1f} s), Engine RPM: {b['mean_rpm']:.1f}")

    # Prominent individual blocks
    # Block 5: idx 7072..7531 (46.0 s)
    # Block 7: idx 9946..10048 (10.3 s)
    # Block 0: idx 142..190 (4.9 s)
    # Block 1: idx 200..231 (3.2 s)
    b5_mask = np.zeros(n_total, dtype=bool)
    b5_mask[7072:7532] = True

    b7_mask = np.zeros(n_total, dtype=bool)
    b7_mask[9946:10049] = True

    b0_mask = np.zeros(n_total, dtype=bool)
    b0_mask[142:191] = True

    # Regime B: Smooth Cruising (v > 10 m/s, |a_CAN| < 0.5 m/s²)
    cruise_mask = (veh_speed > 10.0) & (np.abs(can_acc) < 0.5)
    print(f"    Smooth Cruising: {np.sum(cruise_mask)*0.1:.1f} s ({np.sum(cruise_mask)} samples)")

    # Regime C: Hard Braking (a_CAN < -1.5 m/s²)
    brake_mask = can_acc < -1.5
    print(f"    Hard Braking: {np.sum(brake_mask)*0.1:.1f} s ({np.sum(brake_mask)} samples)")

    # Regime D: Rough-Road Events (moving > 5 m/s with high vertical variance)
    moving_mask = veh_speed > 5.0
    vert_series = pd.Series(phone_vert)
    rolling_vert_var = vert_series.rolling(20, center=True).var().to_numpy()
    valid_moving_vars = rolling_vert_var[moving_mask & (~np.isnan(rolling_vert_var))]
    p95_vert_var = float(np.percentile(valid_moving_vars, 95))
    rough_mask = moving_mask & (rolling_vert_var > p95_vert_var)
    print(f"    Rough Road (P95 vert var > {p95_vert_var:.3f}): {np.sum(rough_mask)*0.1:.1f} s ({np.sum(rough_mask)} samples)")

    # ── 3. Quantitative Analysis across Regimes ──
    print("\n[3] Computing Statistical Moments, Correlations, and Spectral PSDs...")

    regimes_to_eval = [
        ('stationary_idle_all', stat_mask, 'Stationary (Engine Idling, All Blocks, 67.5s)'),
        ('stationary_block_5_46s', b5_mask, 'Stationary Block 5 (Continuous 46.0s Standstill)'),
        ('stationary_block_7_10s', b7_mask, 'Stationary Block 7 (10.3s Standstill)'),
        ('stationary_block_0_5s', b0_mask, 'Stationary Block 0 (4.9s Standstill)'),
        ('smooth_cruising', cruise_mask, 'Smooth Cruising (v > 10 m/s, |a| < 0.5 m/s², 434.9s)'),
        ('hard_braking', brake_mask, 'Hard Braking (a_CAN < -1.5 m/s², 36.7s)'),
        ('rough_road', rough_mask, 'Rough Road (Moving v > 5 m/s, High Var, 46.3s)')
    ]

    vta02_results = {}
    for code, mask, label in regimes_to_eval:
        res = analyze_regime_signals(code, mask, data_dict)
        res['label'] = label
        vta02_results[code] = res
        st_p = res['stats']['phone_fwd']
        st_c = res['stats']['can_accel']
        sp = res['spectral']['phone_fwd']
        corr = res['correlations']['corr_phone_fwd_vs_can']
        print(f"\n  --- {label} ---")
        print(f"    Duration: {res['duration_s']:.1f} s | Correlation r(Phone, CAN): {corr:.4f}")
        print(f"    CAN accel:   mean={st_c['mean']:+.3f}, std={st_c['std']:.3f}, RMS={st_c['rms']:.3f}, p2p={st_c['p2p']:.3f} m/s²")
        print(f"    Phone fwd:   mean={st_p['mean']:+.3f}, std={st_p['std']:.3f}, RMS={st_p['rms']:.3f}, p2p={st_p['p2p']:.3f} m/s²")
        print(f"    Phone PSD:   Total Power={sp['total_power']:.4f} (m/s²)² | Dominant peak: {sp['dominant_freq']:.2f} Hz")
        print(f"    Band Energy: 0-0.5Hz: {sp['pct_0_0_5']:.1f}% | 0.5-2Hz: {sp['pct_0_5_2']:.1f}% | 2-5Hz: {sp['pct_2_5']:.1f}%")

    # ── 4. Cross-Trip Standstill Comparison (Vta02, Vta03, Vta04) ──
    print("\n[4] Comparing Standstill Noise Floor Across Trips...")
    cross_trip_standstill = {}

    trip_status = {
        'Vta02': '[CONTROL] Clean synchronization control',
        'Vta03': '[CORRUPTED] Raw synchronization corrupted (excluded from dynamic eval)',
        'Vta04': '[AMBIGUOUS] Synchronization ambiguous (speed +4.9s vs accel -2.1s)'
    }

    for trip_name in ['Vta02', 'Vta03', 'Vta04']:
        df_pt, df_vt = load_trip(trip_name)
        sp_t = df_vt['veh_speed_ms'].values
        can_t = df_vt['veh_accel_long_ms2'].values
        raw_at = df_pt[['accel_x', 'accel_y', 'accel_z']].values
        raw_gt = df_pt[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_vt, _, _, _ = align_phone_to_vehicle(raw_at, raw_gt, sp_t)

        st_mask_t = sp_t < 0.15
        n_stat = int(np.sum(st_mask_t))
        if n_stat > 10:
            c_stat = can_t[st_mask_t]
            p_fwd_stat = acc_vt[st_mask_t, 0]
            p_lat_stat = acc_vt[st_mask_t, 1]
            p_vert_stat = acc_vt[st_mask_t, 2]

            sp_fwd = compute_psd_and_bands(p_fwd_stat)

            cross_trip_standstill[trip_name] = {
                'status': trip_status[trip_name],
                'stationary_duration_s': float(n_stat * 0.1),
                'n_samples': n_stat,
                'can_stats': compute_stats(c_stat),
                'phone_fwd_stats': compute_stats(p_fwd_stat),
                'phone_lat_stats': compute_stats(p_lat_stat),
                'phone_vert_stats': compute_stats(p_vert_stat),
                'spectral_phone_fwd': sp_fwd
            }
            print(f"  {trip_name} ({trip_status[trip_name]}):")
            print(f"    Standstill dur: {n_stat*0.1:.1f} s | CAN std={np.std(c_stat):.3f} | Phone fwd std={np.std(p_fwd_stat):.3f}, p2p={np.ptp(p_fwd_stat):.3f}")
            print(f"    Band Power: 0-0.5Hz: {sp_fwd['pct_0_0_5']:.1f}% | 0.5-2Hz: {sp_fwd['pct_0_5_2']:.1f}% | 2-5Hz: {sp_fwd['pct_2_5']:.1f}%")
        else:
            print(f"  {trip_name}: No significant stationary period found (< 10 samples)")

    # ── 5. Nyquist & Engine RPM Aliasing Analysis ──
    print("\n[5] Formulating Nyquist Limit & Engine Harmonic Folding Analysis...")
    # Idle RPM ~ 800 RPM
    # Fundamental shaft rotation frequency: f_0 = 800 / 60 = 13.33 Hz
    # 4-cylinder 4-stroke firing frequency: f_fire = 2 * f_0 = 26.67 Hz
    # Sampling frequency fs = 10.0 Hz, Nyquist f_nyq = 5.0 Hz
    # Aliasing calculation: f_alias = |f - k * fs| <= f_nyq
    # For f_0 = 13.33 Hz:
    #   k = 1: |13.33 - 10| = 3.33 Hz (falls inside 2–5 Hz band!)
    # For f_fire = 26.67 Hz:
    #   k = 3: |26.67 - 30| = 3.33 Hz (also folds to 3.33 Hz!)
    nyquist_analysis = {
        'sampling_frequency_hz': 10.0,
        'nyquist_frequency_hz': 5.0,
        'engine_idle_rpm': 800.0,
        'engine_shaft_frequency_hz': 800.0 / 60.0,
        'engine_shaft_aliased_frequency_hz': abs((800.0 / 60.0) - 10.0),
        'engine_firing_frequency_4cyl_hz': 2.0 * (800.0 / 60.0),
        'engine_firing_aliased_frequency_hz': abs((2.0 * (800.0 / 60.0)) - 30.0),
        'aliasing_finding': (
            "Because phone accelerometer is sampled at only 10 Hz without verifiable analog anti-aliasing "
            "prior to digitization, the primary engine mechanical rotation (13.33 Hz at 800 RPM) and 4-cylinder "
            "firing pulse (26.67 Hz) both mathematically alias directly into the observable band at 3.33 Hz. "
            "This explains why a sharp spectral energy concentration is observed in Band 3 (2–5 Hz) during standstill."
        )
    }

    # ── 6. Export Full Structured JSON Deliverable ──
    print("\n[6] Exporting Results to results/c5_5_2_vibration_characterization.json...")
    full_output = {
        'metadata': {
            'stage': 'C5.5.2',
            'title': 'Stationary and Motion Regime Vibration Characterization',
            'primary_control_trip': 'Vta02',
            'trip_classifications': trip_status,
            'sampling_rate_hz': 10.0,
            'nyquist_hz': 5.0
        },
        'nyquist_and_aliasing': nyquist_analysis,
        'vta02_stationary_blocks': block_info,
        'vta02_regimes': vta02_results,
        'cross_trip_standstill': cross_trip_standstill
    }

    json_path = RESULTS_DIR / "c5_5_2_vibration_characterization.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2)
    print(f"    Saved JSON ({json_path.stat().st_size / 1024:.1f} KB)")

    # ── 7. Generate Diagnostic Figures ──
    print("\n[7] Generating Diagnostic Figures...")

    # Plot 1: c5_5_2_stationary_timeseries_and_psd.png
    print("    Plot 1: Stationary timeseries, standstill zoom, and PSD...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Block 5 (46s standstill) overview
    t_b5 = time_s[7072:7532] - time_s[7072]
    axes[0, 0].plot(t_b5, can_acc[7072:7532], 'k-', label='CAN Accel (Chassis)', lw=1.5)
    axes[0, 0].plot(t_b5, phone_fwd[7072:7532], 'b-', label='Phone Fwd Accel', alpha=0.7, lw=1.0)
    axes[0, 0].plot(t_b5, phone_lat[7072:7532], 'g-', label='Phone Lat Accel', alpha=0.6, lw=0.9)
    axes[0, 0].axhline(0, color='gray', ls='--', lw=0.8)
    axes[0, 0].set_title("Vta02 Block 5: 46-Second Engine Idle Standstill (v = 0.0 m/s)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Time within Standstill (s)")
    axes[0, 0].set_ylabel("Acceleration (m/s²)")
    axes[0, 0].set_ylim([-2.5, 2.5])
    axes[0, 0].legend(loc='upper right', framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: 5-second zoom showing persistent vibration oscillations
    zoom_slice = slice(7100, 7150) # 5 seconds
    t_zoom = (time_s[zoom_slice] - time_s[7100])
    axes[0, 1].plot(t_zoom, can_acc[zoom_slice], 'k.-', label='CAN Accel', lw=2.0, ms=5)
    axes[0, 1].plot(t_zoom, phone_fwd[zoom_slice], 'b.-', label='Phone Fwd Accel', lw=1.5, ms=4)
    axes[0, 1].plot(t_zoom, phone_lat[zoom_slice], 'g.-', label='Phone Lat Accel', lw=1.2, ms=4)
    axes[0, 1].set_title("5-Second Zoom: Standstill Vibration Detail (10 Hz Samples)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Acceleration (m/s²)")
    axes[0, 1].set_ylim([-2.0, 2.0])
    axes[0, 1].legend(loc='upper right', framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: PSD during Standstill
    sp_b5_fwd = vta02_results['stationary_block_5_46s']['spectral']['phone_fwd']
    sp_b5_can = vta02_results['stationary_block_5_46s']['spectral']['can_accel']
    sp_b5_lat = vta02_results['stationary_block_5_46s']['spectral']['phone_lat']
    axes[1, 0].semilogy(sp_b5_can['freqs'], sp_b5_can['psd'], 'k-', lw=2.0, label=f"CAN Accel (Total: {sp_b5_can['total_power']:.4f})")
    axes[1, 0].semilogy(sp_b5_fwd['freqs'], sp_b5_fwd['psd'], 'b-', lw=2.0, label=f"Phone Fwd (Total: {sp_b5_fwd['total_power']:.4f}, Peak: {sp_b5_fwd['dominant_freq']:.2f}Hz)")
    axes[1, 0].semilogy(sp_b5_lat['freqs'], sp_b5_lat['psd'], 'g--', lw=1.5, label=f"Phone Lat (Total: {sp_b5_lat['total_power']:.4f}, Peak: {sp_b5_lat['dominant_freq']:.2f}Hz)")
    axes[1, 0].axvspan(0.0, 0.5, color='green', alpha=0.1, label='Band 1 (0-0.5 Hz)')
    axes[1, 0].axvspan(0.5, 2.0, color='orange', alpha=0.1, label='Band 2 (0.5-2 Hz)')
    axes[1, 0].axvspan(2.0, 5.0, color='red', alpha=0.1, label='Band 3 (2-5 Hz)')
    axes[1, 0].set_title("Standstill Power Spectral Density (Welch PSD)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("PSD ((m/s²)² / Hz)")
    axes[1, 0].set_xlim([0, 5.0])
    axes[1, 0].legend(loc='lower left', framealpha=0.9, fontsize=9)
    axes[1, 0].grid(True, which='both', alpha=0.3)

    # Panel 4: Distribution histograms at Standstill
    axes[1, 1].hist(can_acc[7072:7532], bins=30, alpha=0.6, color='black', label=f"CAN (std={np.std(can_acc[7072:7532]):.3f})", density=True)
    axes[1, 1].hist(phone_fwd[7072:7532], bins=30, alpha=0.5, color='blue', label=f"Phone Fwd (std={np.std(phone_fwd[7072:7532]):.3f})", density=True)
    axes[1, 1].hist(phone_lat[7072:7532], bins=30, alpha=0.4, color='green', label=f"Phone Lat (std={np.std(phone_lat[7072:7532]):.3f})", density=True)
    axes[1, 1].set_title("Standstill Acceleration Distributions (Density)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Acceleration (m/s²)")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].legend(loc='upper right', framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c5_5_2_stationary_timeseries_and_psd.png", dpi=300)
    plt.close()

    # Plot 2: c5_5_2_regime_comparison_psd.png
    print("    Plot 2: PSD across 4 regimes (Idle, Cruising, Braking, Rough Road)...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    regime_plot_configs = [
        ('stationary_idle_all', axes[0, 0], 'A. Engine Idling (Standstill, 67.5s)'),
        ('smooth_cruising', axes[0, 1], 'B. Smooth Cruising (v > 10 m/s, 434.9s)'),
        ('hard_braking', axes[1, 0], 'C. Hard Braking (a < -1.5 m/s², 36.7s)'),
        ('rough_road', axes[1, 1], 'D. Rough Road (High Vert Var, 46.3s)')
    ]

    for code, ax, title in regime_plot_configs:
        sp_f = vta02_results[code]['spectral']['phone_fwd']
        sp_c = vta02_results[code]['spectral']['can_accel']
        ax.semilogy(sp_c['freqs'], sp_c['psd'], 'k-', lw=2.0, label=f"CAN Accel (Var={sp_c['total_power']:.3f})")
        ax.semilogy(sp_f['freqs'], sp_f['psd'], 'b-', lw=2.0, label=f"Phone Fwd (Var={sp_f['total_power']:.3f})")
        ax.axvspan(0.0, 0.5, color='green', alpha=0.08)
        ax.axvspan(0.5, 2.0, color='orange', alpha=0.08)
        ax.axvspan(2.0, 5.0, color='red', alpha=0.08)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD ((m/s²)² / Hz)")
        ax.set_xlim([0, 5.0])
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c5_5_2_regime_comparison_psd.png", dpi=300)
    plt.close()

    # Plot 3: c5_5_2_energy_band_breakdown.png
    print("    Plot 3: Energy band breakdown by regime...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    regimes_for_bar = ['stationary_idle_all', 'smooth_cruising', 'hard_braking', 'rough_road']
    labels_bar = ['Idle Standstill', 'Smooth Cruising', 'Hard Braking', 'Rough Road']
    x = np.arange(len(regimes_for_bar))
    width = 0.25

    # Phone Fwd Band Percentages
    pct_b1_p = [vta02_results[r]['spectral']['phone_fwd']['pct_0_0_5'] for r in regimes_for_bar]
    pct_b2_p = [vta02_results[r]['spectral']['phone_fwd']['pct_0_5_2'] for r in regimes_for_bar]
    pct_b3_p = [vta02_results[r]['spectral']['phone_fwd']['pct_2_5'] for r in regimes_for_bar]

    # CAN Band Percentages
    pct_b1_c = [vta02_results[r]['spectral']['can_accel']['pct_0_0_5'] for r in regimes_for_bar]
    pct_b2_c = [vta02_results[r]['spectral']['can_accel']['pct_0_5_2'] for r in regimes_for_bar]
    pct_b3_c = [vta02_results[r]['spectral']['can_accel']['pct_2_5'] for r in regimes_for_bar]

    # Panel 1: Phone Band Distribution
    ax1.bar(x - width, pct_b1_p, width, label='0–0.5 Hz (Quasi-Static)', color='#2ca02c')
    ax1.bar(x, pct_b2_p, width, label='0.5–2.0 Hz (Chassis Heave/Pitch)', color='#ff7f0e')
    ax1.bar(x + width, pct_b3_p, width, label='2.0–5.0 Hz (Mount/Aliased Idle)', color='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_bar, rotation=15, fontweight='bold')
    ax1.set_ylabel("Spectral Energy Fraction (%)")
    ax1.set_title("Smartphone Forward Accelerometer: Energy by Band", fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.3)

    # Panel 2: CAN Band Distribution
    ax2.bar(x - width, pct_b1_c, width, label='0–0.5 Hz (Quasi-Static)', color='#2ca02c')
    ax2.bar(x, pct_b2_c, width, label='0.5–2.0 Hz (Chassis Heave/Pitch)', color='#ff7f0e')
    ax2.bar(x + width, pct_b3_c, width, label='2.0–5.0 Hz (High Frequency)', color='#d62728')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_bar, rotation=15, fontweight='bold')
    ax2.set_ylabel("Spectral Energy Fraction (%)")
    ax2.set_title("Vehicle Chassis CAN Accelerometer: Energy by Band", fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c5_5_2_energy_band_breakdown.png", dpi=300)
    plt.close()

    # Plot 4: c5_5_2_correlation_and_residuals_by_regime.png
    print("    Plot 4: Correlation and residual distributions by regime...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for idx, (code, label_s) in enumerate([
        ('stationary_idle_all', 'Engine Idle (Standstill)'),
        ('smooth_cruising', 'Smooth Cruising'),
        ('hard_braking', 'Hard Braking'),
        ('rough_road', 'Rough Road')
    ]):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        mask = [m for c, m, l in regimes_to_eval if c == code][0]
        c_sub = can_acc[mask]
        p_sub = phone_fwd[mask]
        corr_val = vta02_results[code]['correlations']['corr_phone_fwd_vs_can']
        mae_val = np.mean(np.abs(p_sub - c_sub))

        # Sample for scatter if large
        if len(c_sub) > 1000:
            np.random.seed(42)
            idx_sample = np.random.choice(len(c_sub), 1000, replace=False)
            c_plot, p_plot = c_sub[idx_sample], p_sub[idx_sample]
        else:
            c_plot, p_plot = c_sub, p_sub

        ax.scatter(c_plot, p_plot, alpha=0.4, s=16, color='#1f77b4', edgecolors='none')
        # Identity line
        lims = [min(np.min(c_plot), np.min(p_plot)) - 0.5, max(np.max(c_plot), np.max(p_plot)) + 0.5]
        ax.plot(lims, lims, 'r--', lw=1.5, label='1:1 Ideal Reference')
        ax.set_title(f"{label_s}\nr = {corr_val:.3f} | MAE = {mae_val:.3f} m/s²", fontsize=11, fontweight='bold')
        ax.set_xlabel("Chassis CAN Acceleration (m/s²)")
        ax.set_ylabel("Phone Fwd Acceleration (m/s²)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c5_5_2_correlation_and_residuals_by_regime.png", dpi=300)
    plt.close()

    # Plot 5: c5_5_2_cross_trip_stationary_benchmark.png
    print("    Plot 5: Cross-trip standstill benchmark...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {'Vta02': 'blue', 'Vta03': 'red', 'Vta04': 'orange'}
    for trip_name, res_t in cross_trip_standstill.items():
        sp = res_t['spectral_phone_fwd']
        st = res_t['phone_fwd_stats']
        c = colors.get(trip_name, 'purple')
        ax1.semilogy(sp['freqs'], sp['psd'], color=c, lw=2.0,
                     label=f"{trip_name} (Std: {st['std']:.3f}, P2P: {st['p2p']:.3f} m/s²)")
        
    ax1.set_title("Cross-Trip Standstill PSD Comparison (Phone Fwd Accel)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("PSD ((m/s²)² / Hz)")
    ax1.set_xlim([0, 5.0])
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3)

    # Boxplot of stationary phone fwd accel distributions
    box_data = []
    box_labels = []
    for trip_name in cross_trip_standstill.keys():
        df_pt, df_vt = load_trip(trip_name)
        sp_t = df_vt['veh_speed_ms'].values
        raw_at = df_pt[['accel_x', 'accel_y', 'accel_z']].values
        raw_gt = df_pt[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_vt, _, _, _ = align_phone_to_vehicle(raw_at, raw_gt, sp_t)
        st_mask_t = sp_t < 0.15
        box_data.append(acc_vt[st_mask_t, 0])
        box_labels.append(f"{trip_name}\n({cross_trip_standstill[trip_name]['status'].split()[0]})")

    ax2.boxplot(box_data, tick_labels=box_labels, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.axhline(0, color='red', ls='--', lw=1.2, label='Zero (Ideal Standstill)')
    ax2.set_title("Standstill Acceleration Distribution (P25–P75 & Medians)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Forward Acceleration (m/s²)")
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c5_5_2_cross_trip_stationary_benchmark.png", dpi=300)
    plt.close()

    print("\n[SUCCESS] C5.5.2 Execution complete. All deliverables generated.")
    return full_output

if __name__ == "__main__":
    run_vibration_characterization()
