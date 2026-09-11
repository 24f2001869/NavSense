"""
SIH26168 - Stage C5.2-A: Offline Kinematic Acceleration Audit
Script: experiments/audit_kinematics_c5_2.py

Offline scientific analysis of the relationship between smartphone IMU
acceleration and vehicle ground-truth longitudinal acceleration (dv/dt).
STRICT RULE: dv/dt is used solely for diagnostic audit and never for ML training.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float, fs: float = 10.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def compute_lagged_cross_correlation(x: np.ndarray, y: np.ndarray, max_lags: int = 20):
    """
    Computes normalized cross-correlation over lag range [-max_lags, +max_lags].
    Lag tau > 0 means x is shifted forward (y lags x).
    """
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2)) + 1e-12
    lags = np.arange(-max_lags, max_lags + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            c = np.sum(x_c[:lag] * y_c[-lag:])
        elif lag > 0:
            c = np.sum(x_c[lag:] * y_c[:-lag])
        else:
            c = np.sum(x_c * y_c)
        corrs.append(float(c / denom))
    return lags, np.array(corrs)


def analyze_trip_kinematics(trip_name: str) -> dict:
    df_phone, df_veh = load_trip(trip_name)
    n = min(len(df_phone), len(df_veh))
    df_phone = df_phone.iloc[:n].copy()
    df_veh = df_veh.iloc[:n].copy()

    dt = 0.1  # 10 Hz nominal
    time_s = df_phone['time_s'].to_numpy()

    # Ground truth speed and acceleration
    v_true = df_veh['veh_speed_ms'].to_numpy()
    dv_dt_raw = np.gradient(v_true, dt)
    # Low-pass ground truth acceleration (0.8 Hz) to remove 10 Hz discrete differentiation quantization
    dv_dt_smooth = butter_lowpass_filter(dv_dt_raw, cutoff_hz=0.8, fs=10.0)

    # Phone IMU acceleration
    acc_xp = df_phone['accel_x'].to_numpy()
    acc_yp = df_phone['accel_y'].to_numpy()
    acc_zp = df_phone['accel_z'].to_numpy()

    # Vehicle frame using validated mounting R_vp = diag(1, -1, -1)
    # a_x^v = acc_xp (forward)
    # a_y^v = -acc_yp (left)
    # a_z^v = -acc_zp (up)
    a_long_raw = acc_xp.copy()
    a_long_1hz = butter_lowpass_filter(a_long_raw, cutoff_hz=1.0, fs=10.0)
    a_long_05hz = butter_lowpass_filter(a_long_raw, cutoff_hz=0.5, fs=10.0)

    # Stationary segment or steady cruising for mounting bias estimation
    stationary_mask = v_true < 0.2
    if np.sum(stationary_mask) >= 10:
        bias_mounting = float(np.mean(a_long_raw[stationary_mask]))
        bias_source = "stationary (<0.2 m/s)"
    else:
        # If trip is continuously moving (e.g. Vta04), estimate mounting pitch bias during steady cruising (|dv/dt| < 0.1 m/s^2)
        cruise_steady_mask = (np.abs(dv_dt_smooth) < 0.08) & (v_true > 5.0)
        if np.sum(cruise_steady_mask) >= 20:
            bias_mounting = float(np.mean(a_long_raw[cruise_steady_mask]))
            bias_source = "steady cruising (|dv/dt|<0.08 m/s^2)"
        else:
            bias_mounting = float(np.mean(a_long_raw))
            bias_source = "trip mean"

    a_long_demeaned = a_long_raw - bias_mounting
    a_long_05hz_demeaned = a_long_05hz - bias_mounting

    # 1. Pearson and Spearman correlations
    from scipy.stats import spearmanr
    corr_raw_p = float(np.corrcoef(a_long_raw, dv_dt_raw)[0, 1])
    corr_raw_s = float(spearmanr(a_long_raw, dv_dt_raw)[0])

    corr_filt_p = float(np.corrcoef(a_long_05hz_demeaned, dv_dt_smooth)[0, 1])
    corr_filt_s = float(spearmanr(a_long_05hz_demeaned, dv_dt_smooth)[0])

    # Check other axes correlation to verify mounting alignment
    corr_yp = float(np.corrcoef(acc_yp, dv_dt_smooth)[0, 1])
    corr_zp = float(np.corrcoef(acc_zp, dv_dt_smooth)[0, 1])

    # 2. Lagged cross-correlation
    lags, corrs_lagged = compute_lagged_cross_correlation(a_long_05hz_demeaned, dv_dt_smooth, max_lags=20)
    best_lag_idx = np.argmax(np.abs(corrs_lagged))
    best_lag_s = float(lags[best_lag_idx] * dt)
    max_lag_corr = float(corrs_lagged[best_lag_idx])

    # 3. Dynamic regimes: acceleration, braking, cruising
    accel_mask = dv_dt_smooth > 0.3
    brake_mask = dv_dt_smooth < -0.3
    cruise_mask = (np.abs(dv_dt_smooth) <= 0.3) & (v_true > 2.0)

    # Sign consistency during dynamic maneuvers
    sign_accel_match = float(np.mean(np.sign(a_long_05hz_demeaned[accel_mask]) == np.sign(dv_dt_smooth[accel_mask]))) if np.sum(accel_mask) > 0 else 0.0
    sign_brake_match = float(np.mean(np.sign(a_long_05hz_demeaned[brake_mask]) == np.sign(dv_dt_smooth[brake_mask]))) if np.sum(brake_mask) > 0 else 0.0
    sign_overall_match = float(np.mean(np.sign(a_long_05hz_demeaned[accel_mask | brake_mask]) == np.sign(dv_dt_smooth[accel_mask | brake_mask])))

    # 4. Variance breakdown / disturbance noise floor
    var_dv_dt = float(np.var(dv_dt_smooth))
    var_a_long_raw = float(np.var(a_long_raw))
    var_vibration = float(np.var(a_long_raw - a_long_05hz))
    var_kinematic = float(np.var(a_long_05hz))
    snr_db = float(10.0 * np.log10(var_kinematic / (var_vibration + 1e-12)))

    return {
        'trip': trip_name,
        'duration_s': float(time_s[-1] - time_s[0]),
        'sample_count': int(n),
        'mean_speed_ms': float(np.mean(v_true)),
        'var_speed': float(np.var(v_true)),
        'bias_mounting_ms2': bias_mounting,
        'bias_source': bias_source,
        'var_raw_imu_ms4': var_a_long_raw,
        'var_kinematic_signal_ms4': var_kinematic,
        'var_vibration_noise_ms4': var_vibration,
        'snr_db': snr_db,
        'corr_raw_pearson': corr_raw_p,
        'corr_raw_spearman': corr_raw_s,
        'corr_filtered_pearson': corr_filt_p,
        'corr_filtered_spearman': corr_filt_s,
        'corr_other_axes_with_dvdt': {
            'phone_x': float(np.corrcoef(acc_xp, dv_dt_smooth)[0, 1]),
            'phone_y': corr_yp,
            'phone_z': corr_zp
        },
        'best_lag_s': best_lag_s,
        'max_lag_correlation': max_lag_corr,
        'sign_consistency': {
            'accel_regime_fraction': sign_accel_match,
            'brake_regime_fraction': sign_brake_match,
            'dynamic_overall_fraction': sign_overall_match,
            'accel_samples': int(np.sum(accel_mask)),
            'brake_samples': int(np.sum(brake_mask)),
            'cruise_samples': int(np.sum(cruise_mask))
        },
        'time_s': time_s,
        'v_true': v_true,
        'dv_dt_smooth': dv_dt_smooth,
        'a_long_raw': a_long_raw,
        'a_long_filt': a_long_05hz_demeaned,
        'lags': lags * dt,
        'lag_corrs': corrs_lagged
    }


def main():
    print("=" * 70)
    print("STAGE C5.2-A: OFFLINE KINEMATIC ACCELERATION AUDIT")
    print("=" * 70)

    trips = ["Vta02", "Vta04"]
    results = {}

    for t in trips:
        print(f"\nAnalyzing kinematics for trip: {t}...")
        res = analyze_trip_kinematics(t)
        results[t] = res

        print(f"  Duration: {res['duration_s']:.1f} s ({res['sample_count']} samples)")
        print(f"  Mounting/pitch bias: {res['bias_mounting_ms2']:+.3f} m/s² (Source: {res['bias_source']})")
        print(f"  Kinematic SNR: {res['snr_db']:.2f} dB (Kinematic Var={res['var_kinematic_signal_ms4']:.3f}, Noise Var={res['var_vibration_noise_ms4']:.3f})")
        print(f"  Raw Correlation r(a_long_raw, dv/dt): {res['corr_raw_pearson']:+.4f} (Spearman: {res['corr_raw_spearman']:+.4f})")
        print(f"  Filtered Correlation r(a_long_filt, dv/dt): {res['corr_filtered_pearson']:+.4f} (Spearman: {res['corr_filtered_spearman']:+.4f})")
        print(f"  Axes correlations with dv/dt: Xp={res['corr_other_axes_with_dvdt']['phone_x']:+.3f}, Yp={res['corr_other_axes_with_dvdt']['phone_y']:+.3f}, Zp={res['corr_other_axes_with_dvdt']['phone_z']:+.3f}")
        print(f"  Optimal Lag: {res['best_lag_s']:+.2f} s (Peak r = {res['max_lag_correlation']:+.4f})")
        print(f"  Sign Consistency during dynamic events: {res['sign_consistency']['dynamic_overall_fraction']*100:.1f}%")
        print(f"    - Acceleration: {res['sign_consistency']['accel_regime_fraction']*100:.1f}% ({res['sign_consistency']['accel_samples']} pts)")
        print(f"    - Braking:      {res['sign_consistency']['brake_regime_fraction']*100:.1f}% ({res['sign_consistency']['brake_samples']} pts)")

    # Save summary JSON (excluding numpy arrays)
    summary_json = {}
    for t in trips:
        d = dict(results[t])
        for k in ['time_s', 'v_true', 'dv_dt_smooth', 'a_long_raw', 'a_long_filt', 'lags', 'lag_corrs']:
            del d[k]
        summary_json[t] = d

    json_path = RES_DIR / "c5_2a_kinematics_audit.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved kinematics audit metrics to: {json_path}")

    # Generate Publication Diagnostic Plot
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    for idx, t in enumerate(trips):
        res = results[t]
        t_arr = res['time_s']
        v_arr = res['v_true']
        dv_arr = res['dv_dt_smooth']
        a_filt = res['a_long_filt']
        a_raw = res['a_long_raw']

        # Subplot 1: Speed profile
        axs[0, idx].plot(t_arr, v_arr, 'b-', lw=1.5, label='VBOX Ground Truth Speed')
        axs[0, idx].set_title(f"{t}: Ground Truth Forward Speed Profile", fontsize=11, fontweight='bold')
        axs[0, idx].set_xlabel("Time (s)")
        axs[0, idx].set_ylabel("Speed (m/s)")
        axs[0, idx].grid(True, alpha=0.3)
        axs[0, idx].legend(loc='upper right')

        # Subplot 2: Acceleration Comparison (dv/dt vs a_long)
        axs[1, idx].plot(t_arr, a_raw - res['bias_mounting_ms2'], color='lightgray', lw=0.6, alpha=0.6, label='Raw a_x^v (IMU)')
        axs[1, idx].plot(t_arr, a_filt, 'r-', lw=1.5, label='Filtered a_x^v (0.5 Hz)')
        axs[1, idx].plot(t_arr, dv_arr, 'k--', lw=1.5, label='Ground Truth dv/dt')
        axs[1, idx].set_title(f"{t}: a_long vs dv/dt (r_filt = {res['corr_filtered_pearson']:+.3f}, SNR = {res['snr_db']:.1f} dB)", fontsize=11, fontweight='bold')
        axs[1, idx].set_xlabel("Time (s)")
        axs[1, idx].set_ylabel("Acceleration (m/s²)")
        axs[1, idx].set_ylim([-3.5, 3.5])
        axs[1, idx].grid(True, alpha=0.3)
        axs[1, idx].legend(loc='upper right')

        # Subplot 3: Lagged Cross-Correlation
        axs[2, idx].plot(res['lags'], res['lag_corrs'], 'm-o', markersize=3, lw=1.5)
        axs[2, idx].axvline(res['best_lag_s'], color='r', linestyle='--', label=f"Peak: {res['max_lag_correlation']:+.3f} at {res['best_lag_s']:+.2f} s")
        axs[2, idx].axhline(0, color='k', linestyle='-', alpha=0.3)
        axs[2, idx].set_title(f"{t}: Lagged Cross-Correlation (IMU vs dv/dt)", fontsize=11, fontweight='bold')
        axs[2, idx].set_xlabel("Lag τ (s) [τ > 0: IMU leads dv/dt]")
        axs[2, idx].set_ylabel("Normalized Correlation")
        axs[2, idx].set_ylim([-0.2, 1.0])
        axs[2, idx].grid(True, alpha=0.3)
        axs[2, idx].legend(loc='upper right')

    plt.tight_layout()
    plot_path = FIG_DIR / "c5_2a_kinematics_correlation.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to: {plot_path}")


if __name__ == "__main__":
    main()
