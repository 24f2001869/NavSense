"""
SIH26168 - Stage C5.5: Smartphone Mount-Dynamics Identification

Script: experiments/run_mount_dynamics_identification_c5_5.py

PURPOSE:
    Execute a purely physical, diagnostic investigation to identify the dynamic transfer
    relationship between vehicle chassis acceleration (a_CAN) and smartphone IMU acceleration (a_phone):
        a_phone(s) ~ H_m(s) * a_CAN(s) + n(s)
    
    Determine whether a reproducible, invariant transfer function H_m(f) exists across trips,
    or whether mount compliance causes non-stationary, trip-varying dynamic distortion.

STRICT METHODOLOGICAL MANDATE:
    - Zero Machine Learning (No AI, no Ridge, no RF, no GBDT, no neural nets).
    - No Premature (k, c) Parametric Curve Fitting.
    - No Navigation Filter Redesign.
    - CAN Bus used strictly as a ground-truth scientific reference instrument.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch, csd, coherence, butter, filtfilt
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from experiments.run_ridge_baseline_c5_3b1 import simulate_outage_navigation

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"

FS = 10.0  # 10 Hz sampling
DT = 0.1   # 0.1 s step


def lowpass_filter(data: np.ndarray, cutoff_hz: float = 0.5, fs: float = FS, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    wn = min(0.99, max(0.01, cutoff_hz / nyq))
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, data)


def highpass_filter(data: np.ndarray, cutoff_hz: float = 0.5, fs: float = FS, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter."""
    nyq = 0.5 * fs
    wn = min(0.99, max(0.01, cutoff_hz / nyq))
    b, a = butter(order, wn, btype='high')
    return filtfilt(b, a, data)


def compute_spectral_transfer_function(x: np.ndarray, y: np.ndarray, fs: float = FS, nperseg: int = 128):
    """
    Computes empirical transfer function H(f) = S_xy(f) / S_xx(f)
    where x = chassis CAN acceleration (input)
          y = smartphone leveled acceleration (output)
    """
    # Welch auto-spectral densities
    f, S_xx = welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    _, S_yy = welch(y, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    
    # Cross-spectral density
    _, S_xy = csd(x, y, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    
    # Magnitude-squared coherence
    _, coh = coherence(x, y, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    
    # Empirical transfer function H(f) = S_xy / S_xx
    H = S_xy / np.maximum(1e-12, S_xx)
    mag_linear = np.abs(H)
    mag_db = 20.0 * np.log10(np.maximum(1e-6, mag_linear))
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    
    return {
        'freqs': f,
        'S_xx': S_xx,
        'S_yy': S_yy,
        'S_xy': S_xy,
        'coherence': coh,
        'H': H,
        'mag_linear': mag_linear,
        'mag_db': mag_db,
        'phase_deg': phase_deg
    }


def extract_braking_events(time_s: np.ndarray, a_ref: np.ndarray, a_can: np.ndarray, a_phone: np.ndarray,
                           min_peak_decel: float = -1.5, pre_s: float = 2.0, post_s: float = 5.0):
    """
    Extracts isolated braking epochs aligned at onset t_0 = 0 (where a_ref crosses -0.8 m/s²).
    Window length: pre_s + post_s seconds.
    """
    n_pre = int(pre_s * FS)
    n_post = int(post_s * FS)
    window_len = n_pre + n_post
    
    # Find candidate braking intervals
    is_brake = a_ref <= -0.8
    # Identify continuous segments of braking
    diff = np.diff(is_brake.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    
    events = []
    t_epoch = np.linspace(-pre_s, post_s, window_len)
    
    for idx_onset in onsets:
        if idx_onset < n_pre or idx_onset + n_post >= len(a_ref):
            continue
        
        # Check that this event contains a significant deceleration peak
        event_ref = a_ref[idx_onset - n_pre : idx_onset + n_post]
        min_acc = np.min(event_ref)
        if min_acc > min_peak_decel:
            continue
        
        event_can = a_can[idx_onset - n_pre : idx_onset + n_post]
        event_phone = a_phone[idx_onset - n_pre : idx_onset + n_post]
        
        # Find peak timing
        idx_peak_can = np.argmin(event_can)
        idx_peak_phone = np.argmin(event_phone)
        t_peak_can = t_epoch[idx_peak_can]
        t_peak_phone = t_epoch[idx_peak_phone]
        peak_delay_s = t_peak_phone - t_peak_can
        
        peak_can_val = event_can[idx_peak_can]
        peak_phone_val = event_phone[idx_peak_phone]
        attenuation_ratio = peak_phone_val / (peak_can_val if abs(peak_can_val) > 1e-3 else -1.0)
        
        # Post-braking rebound (positive overshoot in window [1.5, 4.0]s)
        rebound_mask = (t_epoch >= 1.5) & (t_epoch <= 4.0)
        rebound_phone = float(np.max(event_phone[rebound_mask])) if np.sum(rebound_mask) > 0 else 0.0
        rebound_can = float(np.max(event_can[rebound_mask])) if np.sum(rebound_mask) > 0 else 0.0
        
        events.append({
            'onset_time_s': float(time_s[idx_onset]),
            'can_series': event_can,
            'phone_series': event_phone,
            'ref_series': event_ref,
            'residual_series': event_phone - event_can,
            'peak_delay_s': float(peak_delay_s),
            'attenuation_ratio': float(attenuation_ratio),
            'peak_can_ms2': float(peak_can_val),
            'peak_phone_ms2': float(peak_phone_val),
            'rebound_phone_ms2': rebound_phone,
            'rebound_can_ms2': rebound_can
        })
        
    return events, t_epoch


def analyze_trip_mount_dynamics(trip_name: str) -> dict:
    """Performs full mount dynamics identification on a single trip."""
    print(f"\n===================================================================")
    print(f"Identifying Mount Dynamics: {trip_name}")
    print(f"===================================================================")
    
    df_p, df_v = load_trip(trip_name)
    labels = pd.read_csv(PROC_DIR / f"c5_3_labels_{trip_name.lower()}.csv")
    
    n = min(len(df_p), len(df_v), len(labels))
    df_p = df_p.iloc[:n].reset_index(drop=True)
    df_v = df_v.iloc[:n].reset_index(drop=True)
    labels = labels.iloc[:n].reset_index(drop=True)
    
    time_s = labels['time_s'].to_numpy()
    a_ref = labels['LABEL_vbox_ref_accel_ms2'].to_numpy()
    v_vbox = labels['LABEL_vbox_speed_ms'].to_numpy()
    ax_level = labels['ax_level'].to_numpy()
    ax_phone = labels['ax_phone'].to_numpy()
    can_accel = df_v['veh_accel_long_ms2'].to_numpy() if 'veh_accel_long_ms2' in df_v.columns else np.zeros(n)
    
    # -----------------------------------------------------------------------
    # Test 1: Frequency-Domain Transfer Function
    # -----------------------------------------------------------------------
    # Use nperseg=128 (12.8s) for Vta02/Vta04, or 64 (6.4s) if len < 1000
    nperseg = 128 if n >= 1000 else 64
    spec_data = compute_spectral_transfer_function(can_accel, ax_level, fs=FS, nperseg=nperseg)
    
    freqs = spec_data['freqs']
    coh = spec_data['coherence']
    mag_lin = spec_data['mag_linear']
    mag_db = spec_data['mag_db']
    phase_deg = spec_data['phase_deg']
    
    # Identify bandwidth where coherence > 0.6
    coherent_mask = coh >= 0.6
    coherent_freqs = freqs[coherent_mask]
    coh_cutoff_hz = float(np.max(coherent_freqs)) if len(coherent_freqs) > 0 else 0.0
    
    # Low frequency gain and phase (f <= 0.5 Hz)
    lf_mask = (freqs <= 0.5) & (freqs > 0.0)
    mean_lf_gain = float(np.mean(mag_lin[lf_mask])) if np.sum(lf_mask) > 0 else 1.0
    mean_lf_phase = float(np.mean(phase_deg[lf_mask])) if np.sum(lf_mask) > 0 else 0.0
    mean_lf_coh = float(np.mean(coh[lf_mask])) if np.sum(lf_mask) > 0 else 0.0
    
    # High frequency gain and phase (f > 0.5 Hz)
    hf_mask = freqs > 0.5
    mean_hf_gain = float(np.mean(mag_lin[hf_mask])) if np.sum(hf_mask) > 0 else 1.0
    mean_hf_phase = float(np.mean(phase_deg[hf_mask])) if np.sum(hf_mask) > 0 else 0.0
    mean_hf_coh = float(np.mean(coh[hf_mask])) if np.sum(hf_mask) > 0 else 0.0
    
    print(f"1. Frequency-Domain Characterization:")
    print(f"   Coherence Cutoff (gamma² >= 0.6): {coh_cutoff_hz:.2f} Hz")
    print(f"   Low-Freq Band (<= 0.5 Hz): Mean Gain={mean_lf_gain:.3f}, Phase={mean_lf_phase:+.2f}°, Coherence={mean_lf_coh:.3f}")
    print(f"   High-Freq Band (> 0.5 Hz): Mean Gain={mean_hf_gain:.3f}, Phase={mean_hf_phase:+.2f}°, Coherence={mean_hf_coh:.3f}")
    
    # -----------------------------------------------------------------------
    # Test 2: Time-Domain Superposed Epoch Braking Transients
    # -----------------------------------------------------------------------
    events, t_epoch = extract_braking_events(time_s, a_ref, can_accel, ax_level, min_peak_decel=-1.5)
    print(f"\n2. Time-Domain Braking Epochs Extracted: N={len(events)} events")
    
    if len(events) > 0:
        can_matrix = np.array([e['can_series'] for e in events])
        phone_matrix = np.array([e['phone_series'] for e in events])
        res_matrix = np.array([e['residual_series'] for e in events])
        
        mean_can_epoch = np.mean(can_matrix, axis=0)
        std_can_epoch = np.std(can_matrix, axis=0)
        mean_phone_epoch = np.mean(phone_matrix, axis=0)
        std_phone_epoch = np.std(phone_matrix, axis=0)
        mean_res_epoch = np.mean(res_matrix, axis=0)
        
        delays = [e['peak_delay_s'] for e in events]
        attenuations = [e['attenuation_ratio'] for e in events]
        rebounds = [e['rebound_phone_ms2'] for e in events]
        
        mean_delay_s = float(np.mean(delays))
        mean_attenuation = float(np.mean(attenuations))
        mean_rebound_ms2 = float(np.mean(rebounds))
        
        print(f"   Mean Peak Deceleration Delay (Phone - CAN): {mean_delay_s:+.3f} s ({mean_delay_s * FS:+.1f} samples)")
        print(f"   Mean Deceleration Attenuation Ratio:       {mean_attenuation:.3f} (phone/CAN peak)")
        print(f"   Mean Post-Braking Phone Rebound:           {mean_rebound_ms2:+.3f} m/s²")
    else:
        mean_can_epoch = np.zeros_like(t_epoch)
        std_can_epoch = np.zeros_like(t_epoch)
        mean_phone_epoch = np.zeros_like(t_epoch)
        std_phone_epoch = np.zeros_like(t_epoch)
        mean_res_epoch = np.zeros_like(t_epoch)
        mean_delay_s = 0.0
        mean_attenuation = 1.0
        mean_rebound_ms2 = 0.0
        
    # -----------------------------------------------------------------------
    # Test 4: Low-Frequency vs. Transient Error Decomposition
    # -----------------------------------------------------------------------
    # Filter both signals at 0.5 Hz
    ax_level_lf = lowpass_filter(ax_level, cutoff_hz=0.5)
    ax_level_hf = highpass_filter(ax_level, cutoff_hz=0.5)
    can_accel_lf = lowpass_filter(can_accel, cutoff_hz=0.5)
    can_accel_hf = highpass_filter(can_accel, cutoff_hz=0.5)
    
    # Error metrics in Low-Freq Band vs High-Freq Band
    err_lf = ax_level_lf - can_accel_lf
    err_hf = ax_level_hf - can_accel_hf
    err_total = ax_level - can_accel
    
    mae_lf = float(np.mean(np.abs(err_lf)))
    rmse_lf = float(np.sqrt(np.mean(err_lf**2)))
    r2_lf = 1.0 - float(np.var(err_lf) / np.maximum(1e-6, np.var(can_accel_lf)))
    corr_lf = float(np.corrcoef(ax_level_lf, can_accel_lf)[0, 1])
    
    mae_hf = float(np.mean(np.abs(err_hf)))
    rmse_hf = float(np.sqrt(np.mean(err_hf**2)))
    r2_hf = 1.0 - float(np.var(err_hf) / np.maximum(1e-6, np.var(can_accel_hf)))
    corr_hf = float(np.corrcoef(ax_level_hf, can_accel_hf)[0, 1])
    
    mae_tot = float(np.mean(np.abs(err_total)))
    rmse_tot = float(np.sqrt(np.mean(err_total**2)))
    
    # Severe braking subset for LF vs HF
    sev_brake_mask = a_ref <= -1.5
    bias_sev_tot = float(np.mean(err_total[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    bias_sev_lf = float(np.mean(err_lf[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    bias_sev_hf = float(np.mean(err_hf[sev_brake_mask])) if np.sum(sev_brake_mask) > 0 else 0.0
    
    print(f"\n4. Bandwidth Error Decomposition (Cutoff = 0.5 Hz):")
    print(f"   {'Band':<16} | {'MAE (m/s²)':<10} | {'RMSE (m/s²)':<11} | {'Corr (r)':<10} | {'Sev Brake Bias':<15}")
    print(f"   {'-'*68}")
    print(f"   {'Total Band':<16} | {mae_tot:<10.4f} | {rmse_tot:<11.4f} | {float(np.corrcoef(ax_level, can_accel)[0,1]):<10.4f} | {bias_sev_tot:<15.4f}")
    print(f"   {'Low-Pass (<0.5Hz)':<16} | {mae_lf:<10.4f} | {rmse_lf:<11.4f} | {corr_lf:<10.4f} | {bias_sev_lf:<15.4f}")
    print(f"   {'High-Pass (>=0.5Hz)':<16} | {mae_hf:<10.4f} | {rmse_hf:<11.4f} | {corr_hf:<10.4f} | {bias_sev_hf:<15.4f}")
    
    # -----------------------------------------------------------------------
    # Dead-Reckoning Navigation Benchmark with Low-Pass Filtering
    # -----------------------------------------------------------------------
    nav_series = {
        'raw_phone': ax_phone,
        'static_leveled': ax_level,
        'lowpass_phone': ax_level_lf,
        'chassis_can': can_accel,
        'oracle_ref': a_ref
    }
    nav_res = simulate_outage_navigation(v_vbox, nav_series, horizons=[5.0, 10.0, 20.0, 30.0, 60.0], stride_s=2.5)
    
    print(f"\n5. Dead-Reckoning Navigation Impact of Low-Pass Filtering:")
    print(f"   {'Horizon':<8} | {'Raw Phone':<10} | {'Static Level':<12} | {'Low-Pass Phone':<14} | {'Chassis CAN':<12}")
    print(f"   {'-'*64}")
    for h in [5, 10, 20, 30, 60]:
        h_k = f"{h}s"
        if h_k in nav_res:
            m_data = nav_res[h_k]['methods']
            p_raw = m_data['raw_phone']['mean_pos_error_m']
            p_stat = m_data['static_leveled']['mean_pos_error_m']
            p_lf = m_data['lowpass_phone']['mean_pos_error_m']
            p_can = m_data['chassis_can']['mean_pos_error_m']
            print(f"   {h_k:<8} | {p_raw:<10.2f} | {p_stat:<12.2f} | {p_lf:<14.2f} | {p_can:<12.2f}")
            
    return {
        'trip_name': trip_name,
        'samples': n,
        'spectral': {
            'freqs': freqs.tolist(),
            'coherence': coh.tolist(),
            'mag_linear': mag_lin.tolist(),
            'mag_db': mag_db.tolist(),
            'phase_deg': phase_deg.tolist(),
            'coh_cutoff_hz': coh_cutoff_hz,
            'mean_lf_gain': mean_lf_gain,
            'mean_lf_phase': mean_lf_phase,
            'mean_lf_coh': mean_lf_coh,
            'mean_hf_gain': mean_hf_gain,
            'mean_hf_phase': mean_hf_phase,
            'mean_hf_coh': mean_hf_coh
        },
        'transient': {
            'n_events': len(events),
            't_epoch': t_epoch.tolist(),
            'mean_can_epoch': mean_can_epoch.tolist(),
            'std_can_epoch': std_can_epoch.tolist(),
            'mean_phone_epoch': mean_phone_epoch.tolist(),
            'std_phone_epoch': std_phone_epoch.tolist(),
            'mean_res_epoch': mean_res_epoch.tolist(),
            'mean_delay_s': mean_delay_s,
            'mean_attenuation': mean_attenuation,
            'mean_rebound_ms2': mean_rebound_ms2
        },
        'band_decomposition': {
            'total': {'mae': mae_tot, 'rmse': rmse_tot, 'sev_brake_bias': bias_sev_tot},
            'low_pass_0_5hz': {'mae': mae_lf, 'rmse': rmse_lf, 'corr': corr_lf, 'r2': r2_lf, 'sev_brake_bias': bias_sev_lf},
            'high_pass_0_5hz': {'mae': mae_hf, 'rmse': rmse_hf, 'corr': corr_hf, 'r2': r2_hf, 'sev_brake_bias': bias_sev_hf}
        },
        'navigation_drift': {
            h_k: {m: nav_res[h_k]['methods'][m]['mean_pos_error_m'] for m in nav_res[h_k]['methods']}
            for h_k in nav_res
        }
    }


def main():
    print("\n===================================================================")
    print("STARTING STAGE C5.5: MOUNT-DYNAMICS IDENTIFICATION")
    print("===================================================================")
    
    trips = ["Vta02", "Vta03", "Vta04"]
    results = {}
    
    for t in trips:
        results[t] = analyze_trip_mount_dynamics(t)
        
    # -----------------------------------------------------------------------
    # Test 3: Cross-Trip Consistency Analysis
    # -----------------------------------------------------------------------
    print("\n===================================================================")
    print("Test 3: Cross-Trip Transfer Function Consistency Evaluation")
    print("===================================================================")
    
    # Resample all frequency responses onto a standard frequency grid (0.05 to 5.0 Hz)
    f_grid = np.linspace(0.1, 4.9, 50)
    mag_grid = {}
    phase_grid = {}
    coh_grid = {}
    
    for t in trips:
        f_orig = np.array(results[t]['spectral']['freqs'])
        m_orig = np.array(results[t]['spectral']['mag_linear'])
        p_orig = np.array(results[t]['spectral']['phase_deg'])
        c_orig = np.array(results[t]['spectral']['coherence'])
        
        mag_grid[t] = np.interp(f_grid, f_orig, m_orig)
        phase_grid[t] = np.interp(f_grid, f_orig, p_orig)
        coh_grid[t] = np.interp(f_grid, f_orig, c_orig)
        
    # Calculate cross-trip root-mean-square differences
    diff_mag_02_04 = float(np.sqrt(np.mean((mag_grid['Vta02'] - mag_grid['Vta04'])**2)))
    diff_mag_03_04 = float(np.sqrt(np.mean((mag_grid['Vta03'] - mag_grid['Vta04'])**2)))
    diff_phase_02_04 = float(np.sqrt(np.mean((phase_grid['Vta02'] - phase_grid['Vta04'])**2)))
    diff_phase_03_04 = float(np.sqrt(np.mean((phase_grid['Vta03'] - phase_grid['Vta04'])**2)))
    
    print(f"Cross-Trip Magnitude RMS Difference (Vta02 vs Vta04): {diff_mag_02_04:.4f}")
    print(f"Cross-Trip Magnitude RMS Difference (Vta03 vs Vta04): {diff_mag_03_04:.4f}")
    print(f"Cross-Trip Phase RMS Difference (Vta02 vs Vta04):     {diff_phase_02_04:.2f}°")
    print(f"Cross-Trip Phase RMS Difference (Vta03 vs Vta04):     {diff_phase_03_04:.2f}°")
    
    results['cross_trip_consistency'] = {
        'f_grid': f_grid.tolist(),
        'rms_mag_diff_02_04': diff_mag_02_04,
        'rms_mag_diff_03_04': diff_mag_03_04,
        'rms_phase_diff_02_04': diff_phase_02_04,
        'rms_phase_diff_03_04': diff_phase_03_04
    }
    
    # -----------------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE FIGURES
    # -----------------------------------------------------------------------
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.4})
    
    # Figure 1: Bode Gain, Phase, and Coherence
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = {'Vta02': '#1f77b4', 'Vta03': '#ff7f0e', 'Vta04': '#2ca02c'}
    
    # Subplot 1: Magnitude Ratio (dB)
    for t in trips:
        f = results[t]['spectral']['freqs']
        mag_db = results[t]['spectral']['mag_db']
        axes[0].plot(f, mag_db, label=f"{t} Gain (dB)", color=colors[t], lw=2.0)
    axes[0].axhline(0.0, color='black', linestyle='--', alpha=0.7, label='0 dB Ideal (Phone = CAN)')
    axes[0].set_ylabel("Gain |H(f)| (dB)")
    axes[0].set_title("C5.5: Mount Frequency Response (Bode Representation & Coherence)")
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([-25, 10])
    
    # Subplot 2: Phase Lag (deg)
    for t in trips:
        f = results[t]['spectral']['freqs']
        phase = results[t]['spectral']['phase_deg']
        axes[1].plot(f, phase, label=f"{t} Phase", color=colors[t], lw=2.0)
    axes[1].axhline(0.0, color='black', linestyle='--', alpha=0.7, label='0° Ideal')
    axes[1].set_ylabel("Phase Lag ∠H(f) (°)")
    axes[1].legend(loc='lower left')
    axes[1].set_ylim([-180, 45])
    
    # Subplot 3: Magnitude-Squared Coherence
    for t in trips:
        f = results[t]['spectral']['freqs']
        coh = results[t]['spectral']['coherence']
        axes[2].plot(f, coh, label=f"{t} Coherence γ²", color=colors[t], lw=2.0)
    axes[2].axhline(0.6, color='red', linestyle=':', lw=1.5, label='γ² = 0.6 Coherence Threshold')
    axes[2].set_ylabel("Coherence γ²(f)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylim([0, 1.05])
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    fig_path_1 = FIG_DIR / "c5_5_frequency_response_bode_coherence.png"
    plt.savefig(fig_path_1, dpi=300)
    plt.close()
    print(f"\nSaved Figure 1: {fig_path_1}")
    
    # Figure 2: Superposed Epoch Braking Transients
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    
    for idx, t in enumerate(trips):
        t_ep = np.array(results[t]['transient']['t_epoch'])
        m_can = np.array(results[t]['transient']['mean_can_epoch'])
        s_can = np.array(results[t]['transient']['std_can_epoch'])
        m_phone = np.array(results[t]['transient']['mean_phone_epoch'])
        s_phone = np.array(results[t]['transient']['std_phone_epoch'])
        m_res = np.array(results[t]['transient']['mean_res_epoch'])
        n_ev = results[t]['transient']['n_events']
        
        ax = axes[idx]
        ax.plot(t_ep, m_can, color='black', lw=2.5, label='Chassis CAN (Mean)')
        ax.fill_between(t_ep, m_can - s_can, m_can + s_can, color='black', alpha=0.15)
        
        ax.plot(t_ep, m_phone, color='crimson', lw=2.0, linestyle='-', label='Smartphone Leveled (Mean)')
        ax.fill_between(t_ep, m_phone - s_phone, m_phone + s_phone, color='crimson', alpha=0.15)
        
        ax.plot(t_ep, m_res, color='blue', lw=1.5, linestyle='--', label='Mount Residual (Phone - CAN)')
        
        ax.axvline(0.0, color='gray', linestyle=':', label='Brake Onset (t=0)')
        ax.axhline(0.0, color='gray', alpha=0.5)
        ax.set_ylabel("Accel (m/s²)")
        ax.set_title(f"Trip {t} (N={n_ev} Braking Epochs) — Mean Delay: {results[t]['transient']['mean_delay_s']:+.2f} s")
        if idx == 0:
            ax.legend(loc='lower left', ncol=2)
            
    axes[-1].set_xlabel("Time Relative to Braking Onset (s)")
    plt.tight_layout()
    fig_path_2 = FIG_DIR / "c5_5_superposed_epoch_braking_transients.png"
    plt.savefig(fig_path_2, dpi=300)
    plt.close()
    print(f"Saved Figure 2: {fig_path_2}")
    
    # Figure 3: Cross-Trip Transfer Function Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for t in trips:
        f = f_grid
        ax1.plot(f, mag_grid[t], label=f"{t} Magnitude Ratio", color=colors[t], lw=2.2)
        ax2.plot(f, phase_grid[t], label=f"{t} Phase Lag", color=colors[t], lw=2.2)
        
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Ideal Ratio = 1.0')
    ax1.set_ylabel("Linear Gain |H_m(f)|")
    ax1.set_title("C5.5: Cross-Trip Transfer Function Consistency (Interpolated Grid)")
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.0, 2.0])
    
    ax2.axhline(0.0, color='black', linestyle='--', alpha=0.7, label='Ideal Phase = 0°')
    ax2.set_ylabel("Phase Lag (Degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.legend(loc='lower left')
    ax2.set_ylim([-180, 45])
    
    plt.tight_layout()
    fig_path_3 = FIG_DIR / "c5_5_cross_trip_transfer_function_comparison.png"
    plt.savefig(fig_path_3, dpi=300)
    plt.close()
    print(f"Saved Figure 3: {fig_path_3}")
    
    # Figure 4: Frequency Band Decomposition (Low-Pass vs High-Pass Drift)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of MAE in low-pass vs high-pass
    bar_width = 0.25
    x_indices = np.arange(len(trips))
    
    mae_tot = [results[t]['band_decomposition']['total']['mae'] for t in trips]
    mae_lf = [results[t]['band_decomposition']['low_pass_0_5hz']['mae'] for t in trips]
    mae_hf = [results[t]['band_decomposition']['high_pass_0_5hz']['mae'] for t in trips]
    
    axes[0].bar(x_indices - bar_width, mae_tot, width=bar_width, label='Total Band', color='#7f7f7f')
    axes[0].bar(x_indices, mae_lf, width=bar_width, label='Low-Pass (<0.5 Hz)', color='#2ca02c')
    axes[0].bar(x_indices + bar_width, mae_hf, width=bar_width, label='High-Pass (>=0.5 Hz)', color='#d62728')
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(trips)
    axes[0].set_ylabel("Acceleration MAE (m/s²)")
    axes[0].set_title("Acceleration Error by Frequency Band")
    axes[0].legend()
    
    # 30s Dead-Reckoning Position Drift comparison
    drift_stat = [results[t]['navigation_drift']['30s']['static_leveled'] for t in trips]
    drift_lf = [results[t]['navigation_drift']['30s']['lowpass_phone'] for t in trips]
    drift_can = [results[t]['navigation_drift']['30s']['chassis_can'] for t in trips]
    
    axes[1].bar(x_indices - bar_width, drift_stat, width=bar_width, label='Static Leveled Phone', color='#1f77b4')
    axes[1].bar(x_indices, drift_lf, width=bar_width, label='Low-Pass Filtered Phone (<0.5Hz)', color='#9467bd')
    axes[1].bar(x_indices + bar_width, drift_can, width=bar_width, label='Chassis CAN (Rigid Mount)', color='#2ca02c')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(trips)
    axes[1].set_ylabel("30s Dead-Reckoning Drift (m)")
    axes[1].set_title("30s Dead-Reckoning Drift Comparison")
    axes[1].legend()
    
    plt.tight_layout()
    fig_path_4 = FIG_DIR / "c5_5_frequency_band_decomposition.png"
    plt.savefig(fig_path_4, dpi=300)
    plt.close()
    print(f"Saved Figure 4: {fig_path_4}")
    
    # Export JSON
    json_path = RES_DIR / "c5_5_mount_dynamics_audit.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON Results: {json_path}")
    print("\n===================================================================")
    print("STAGE C5.5 DIAGNOSTIC SUITE COMPLETE")
    print("===================================================================")


if __name__ == "__main__":
    main()
