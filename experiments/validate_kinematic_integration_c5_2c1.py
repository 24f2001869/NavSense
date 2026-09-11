"""
SIH26168 - Stage C5.2-C1: Pure Kinematic Velocity Integration Diagnostic
Script: experiments/validate_kinematic_integration_c5_2c1.py

Strictly isolated kinematic velocity integration experiment.
Evaluates pure numerical integration of smartphone longitudinal acceleration:
    v(t) = v(t0) + \int_{t0}^t a_long(\tau) d\tau

STRICT RULES:
  - ❌ No ZUPT
  - ❌ No AI / learned correction
  - ❌ No ESKF / Kalman filter
  - ❌ No map matching
  - ❌ No dynamic bounds or velocity clamping
  - Evaluates velocity MAE, RMSE, final error, and drift rate over 5s, 10s, 20s, 30s, 60s windows
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
from src.preprocessing.gravity_alignment import compute_leveling_matrix

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float, fs: float = 10.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def run_pure_integration(time_s: np.ndarray, a_long: np.ndarray, v_true: np.ndarray, t_start: float, duration_s: float):
    """
    Performs pure trapezoidal integration of a_long from t_start for duration_s,
    initialized with ground truth speed v_true(t_start).
    """
    dt = 0.1
    k_start = int(round(t_start / dt))
    k_end = min(int(round((t_start + duration_s) / dt)), len(time_s) - 1)

    t_win = time_s[k_start:k_end + 1]
    v_win_true = v_true[k_start:k_end + 1]
    a_win = a_long[k_start:k_end + 1]

    # Trapezoidal integration: v[k] = v[k-1] + 0.5 * (a[k] + a[k-1]) * dt
    v_est = np.zeros(len(t_win), dtype=np.float64)
    v_est[0] = v_win_true[0]

    for k in range(1, len(t_win)):
        v_est[k] = v_est[k - 1] + 0.5 * (a_win[k] + a_win[k - 1]) * dt

    err = v_est - v_win_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    final_err = float(err[-1])
    max_err = float(np.max(np.abs(err)))
    drift_rate_ms_per_s = float(np.abs(final_err) / duration_s)

    # Position error resulting from velocity integration: e_p = \int (v_est - v_true) dt
    pos_err = np.zeros(len(t_win))
    for k in range(1, len(t_win)):
        pos_err[k] = pos_err[k - 1] + 0.5 * (err[k] + err[k - 1]) * dt
    final_pos_err = float(pos_err[-1])

    return {
        't_start_s': float(t_start),
        'duration_s': float(duration_s),
        'v0_ms': float(v_win_true[0]),
        'v_final_true_ms': float(v_win_true[-1]),
        'v_final_est_ms': float(v_est[-1]),
        'mae_ms': mae,
        'rmse_ms': rmse,
        'final_err_ms': final_err,
        'max_err_ms': max_err,
        'drift_rate_ms_per_s': drift_rate_ms_per_s,
        'final_pos_err_m': final_pos_err,
        't_win': t_win,
        'v_est': v_est,
        'v_true': v_win_true,
        'err': err,
        'pos_err': pos_err
    }


def main():
    print("=" * 85)
    print("STAGE C5.2-C1: PURE KINEMATIC VELOCITY INTEGRATION DIAGNOSTIC")
    print("=" * 85)

    df_p, df_v = load_trip("Vta04")
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    dt = 0.1
    time_s = df_p['time_s'].to_numpy()
    v_true = df_v['veh_speed_ms'].to_numpy()

    # Accelerations
    ax_p = df_p['accel_x'].to_numpy()
    ay_p = df_p['accel_y'].to_numpy()
    az_p = df_p['accel_z'].to_numpy()

    # Leveled frame (removing static pitch/roll)
    g_vec = df_p[['grav_x', 'grav_y', 'grav_z']].values.mean(axis=0)
    R_level, _, _ = compute_leveling_matrix(g_vec)
    acc_l_stack = R_level @ np.vstack([ax_p, ay_p, az_p])
    ax_level = acc_l_stack[0]

    # Bias estimation during pre-outage window (t = 0.0 s to 25.0 s)
    # The standardized outage in C2/C3/C4 begins at t = 25.1 s.
    # Prior to t = 25.0 s, GNSS/velocity reference is available to estimate accelerometer bias.
    k_pre = int(round(25.0 / dt))
    dv_dt_pre = np.gradient(v_true[:k_pre], dt)
    acc_bias_pre_raw = float(np.mean(ax_p[:k_pre] - dv_dt_pre))
    acc_bias_pre_level = float(np.mean(ax_level[:k_pre] - dv_dt_pre))

    print(f"Pre-Outage Calibration (t = 0.0 to 25.0 s):")
    print(f"  Raw ax Pre-Outage Bias:    {acc_bias_pre_raw:+.4f} m/s^2")
    print(f"  Leveled ax Pre-Outage Bias:{acc_bias_pre_level:+.4f} m/s^2")

    # Signal variants to test
    signals = {
        'Raw Phone ax (Uncalibrated)': ax_p,
        'Raw Phone ax (Pre-Calibrated Bias)': ax_p - acc_bias_pre_raw,
        'Leveled ax (Uncalibrated)': ax_level,
        'Leveled ax (Pre-Calibrated Bias)': ax_level - acc_bias_pre_level,
        'Leveled ax + 0.5Hz Filter (Pre-Calibrated)': butter_lowpass_filter(ax_level - acc_bias_pre_level, 0.5)
    }

    # Evaluation durations from t = 25.1 s
    durations = [5.0, 10.0, 20.0, 30.0, 60.0]
    t_outage_start = 25.1

    all_results = {}

    print("\n" + "=" * 95)
    print("PURE INTEGRATION OVER INCREASING OUTAGE DURATIONS (STARTING AT t = 25.1 s)")
    print("=" * 95)
    print(f"{'Signal Variant':<38} | {'Dur (s)':<7} | {'MAE (m/s)':<10} | {'RMSE (m/s)':<10} | {'Final Err':<10} | {'Pos Err (m)':<11}")
    print("-" * 95)

    for sig_name, sig_data in signals.items():
        all_results[sig_name] = {}
        for dur in durations:
            res = run_pure_integration(time_s, sig_data, v_true, t_start=t_outage_start, duration_s=dur)
            all_results[sig_name][f"{int(dur)}s"] = res

            print(f"{sig_name:<38} | {dur:>5.1f}s | {res['mae_ms']:>8.2f}   | {res['rmse_ms']:>8.2f}   | {res['final_err_ms']:>+8.2f} m/s | {res['final_pos_err_m']:>+9.2f} m")
        print("-" * 95)

    # 30-Second Standardized C2/C3/C4 Outage Deep-Dive Comparison
    res_30s = {}
    for sig_name in signals.keys():
        res_30s[sig_name] = all_results[sig_name]['30s']

    print("\n" + "=" * 85)
    print("STANDARDIZED 30-SECOND OUTAGE DEEP DIVE (t = 25.1 s to 55.0 s, Distance = 344.5 m)")
    print("=" * 85)
    print(f"{'Configuration':<40} | {'Vel MAE':<10} | {'Vel RMSE':<10} | {'Final Vel Err':<14} | {'Pos Error':<10}")
    print("-" * 85)
    for sig_name, r in res_30s.items():
        print(f"{sig_name:<40} | {r['mae_ms']:>6.2f} m/s | {r['rmse_ms']:>6.2f} m/s | {r['final_err_ms']:>+8.2f} m/s     | {r['final_pos_err_m']:>+7.2f} m")
    print("-" * 85)

    # Save summary JSON (excluding numpy arrays)
    summary_json = {}
    for sig_name in signals.keys():
        summary_json[sig_name] = {}
        for dur_key, r in all_results[sig_name].items():
            summary_json[sig_name][dur_key] = {
                k: v for k, v in r.items() if not isinstance(v, np.ndarray)
            }

    json_path = RES_DIR / "c5_2c1_kinematic_integration.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved kinematic integration metrics to: {json_path}")

    # Publication-Grade Diagnostic Figure
    fig, axs = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    t_plot_end = 85.1  # covering 60s outage
    mask_plot = (time_s >= 20.0) & (time_s <= t_plot_end)
    t_sub = time_s[mask_plot]
    v_sub = v_true[mask_plot]

    # Panel 1: Velocity Comparison during 30s outage
    r_uncal = res_30s['Leveled ax (Uncalibrated)']
    r_calib = res_30s['Leveled ax (Pre-Calibrated Bias)']
    r_filt = res_30s['Leveled ax + 0.5Hz Filter (Pre-Calibrated)']

    axs[0].plot(time_s[mask_plot], v_true[mask_plot], 'k-', lw=2.0, label='Ground Truth Speed (VBOX)')
    axs[0].plot(r_uncal['t_win'], r_uncal['v_est'], 'r--', lw=1.2, label=f"Uncalibrated Leveled (MAE={r_uncal['mae_ms']:.2f} m/s)")
    axs[0].plot(r_calib['t_win'], r_calib['v_est'], 'b-', lw=1.5, label=f"Bias-Calibrated Leveled (MAE={r_calib['mae_ms']:.2f} m/s)")
    axs[0].axvspan(25.1, 55.0, color='gray', alpha=0.15, label='30s GNSS Outage Window (25.1-55.0 s)')
    axs[0].set_title("C5.2-C1: Pure Longitudinal Kinematic Velocity Integration", fontweight='bold')
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='lower left')

    # Panel 2: Instantaneous Velocity Error Growth
    axs[1].plot(r_uncal['t_win'], r_uncal['err'], 'r--', lw=1.2, label=f"Uncalibrated Error (Final={r_uncal['final_err_ms']:+.2f} m/s)")
    axs[1].plot(r_calib['t_win'], r_calib['err'], 'b-', lw=1.5, label=f"Bias-Calibrated Error (Final={r_calib['final_err_ms']:+.2f} m/s)")
    axs[1].axhline(0, color='k', linestyle='-', alpha=0.3)
    axs[1].axvspan(25.1, 55.0, color='gray', alpha=0.15)
    axs[1].set_title("Velocity Error Growth Over Time: e_v(t) = v_est(t) - v_true(t)", fontweight='bold')
    axs[1].set_ylabel("Velocity Error (m/s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper left')

    # Panel 3: Resulting Longitudinal Position Error Growth
    axs[2].plot(r_uncal['t_win'], r_uncal['pos_err'], 'r--', lw=1.2, label=f"Uncalibrated Pos Error (Final={r_uncal['final_pos_err_m']:+.1f} m)")
    axs[2].plot(r_calib['t_win'], r_calib['pos_err'], 'b-', lw=1.5, label=f"Bias-Calibrated Pos Error (Final={r_calib['final_pos_err_m']:+.1f} m)")
    axs[2].axhline(0, color='k', linestyle='-', alpha=0.3)
    axs[2].axvspan(25.1, 55.0, color='gray', alpha=0.15)
    axs[2].set_title("Resulting Longitudinal Position Error: e_p(t) = \\int e_v(\\tau) d\\tau", fontweight='bold')
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Position Error (m)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plot_path = FIG_DIR / "c5_2c1_kinematic_integration.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to: {plot_path}")


if __name__ == "__main__":
    main()
