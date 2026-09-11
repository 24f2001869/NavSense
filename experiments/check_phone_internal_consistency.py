"""
C5.5.1b — Phone Internal Timing Consistency Check
===================================================
Key question: Does the phone's OWN GPS speed derivative match its OWN
accelerometer? If yes, the phone is internally consistent and the
speed/accel lag disagreement on Vta04 is purely a V-file alignment issue.
If no, the phone has internal sensor timing issues.

For each trip:
  1. Compute phone-GPS-derived acceleration (finite diff of GPS speed)
  2. Cross-correlate with raw phone accel_x
  3. Compare with the external (VBOX/CAN) cross-correlations from C5.5.1

This is a self-consistency check using ONLY the S-*.csv file.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import find_trip_dir

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def lag_correlation_sweep(sig_a, sig_b, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag < 0:
            a = sig_a[:lag]
            b = sig_b[-lag:]
        elif lag > 0:
            a = sig_a[lag:]
            b = sig_b[:-lag]
        else:
            a = sig_a.copy()
            b = sig_b.copy()
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) > 100:
            corrs[i] = np.corrcoef(a[mask], b[mask])[0, 1]
    best_idx = np.nanargmax(corrs)
    return lags, corrs, lags[best_idx], corrs[best_idx]

results = {}
trips = ["Vta02", "Vta03", "Vta04"]

fig, axes = plt.subplots(len(trips), 2, figsize=(16, 4*len(trips)))

for row, trip in enumerate(trips):
    print(f"\n{'='*70}")
    print(f"  PHONE INTERNAL CONSISTENCY: {trip}")
    print(f"{'='*70}")

    trip_dir = find_trip_dir(trip)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')

    # Phone signals (all from S-*.csv)
    col_s_speed = [c for c in df_s.columns if 'GPS SPEED' in c.upper()][0]
    col_s_ax = [c for c in df_s.columns if 'ACCELEROMETER X' in c.upper()][0]
    col_s_time = [c for c in df_s.columns if 'TIME SINCE START' in c.upper()]

    s_speed_ms = df_s[col_s_speed].to_numpy()  # Actually in m/s
    s_ax = df_s[col_s_ax].to_numpy()

    # Phone time
    if col_s_time:
        s_time_ms = df_s[col_s_time[0]].to_numpy()
        s_time_s = (s_time_ms - s_time_ms[0]) / 1000.0
        dt = np.median(np.diff(s_time_s))
        print(f"  Phone sample rate: {1/dt:.1f} Hz (dt={dt:.4f}s)")
    else:
        dt = 0.1
        s_time_s = np.arange(len(s_speed_ms)) * dt
        print(f"  Phone sample rate: assumed 10 Hz")

    # Derive acceleration from phone GPS speed
    gps_accel = np.gradient(s_speed_ms, dt)

    # Smooth GPS accel slightly (GPS speed is 1Hz staircase, gradient is spiky)
    # Use a 5-sample running mean to match the ~1Hz GPS update
    kernel = np.ones(5) / 5
    gps_accel_smooth = np.convolve(gps_accel, kernel, mode='same')

    # Also get VBOX data for comparison
    col_v_speed = [c for c in df_v.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    col_v_accel = [c for c in df_v.columns if 'LONGITUDINAL' in c.upper()][0]
    v_speed_kmh = df_v[col_v_speed].to_numpy()
    can_accel_ms2 = df_v[col_v_accel].to_numpy() * 9.80665

    n = min(len(s_ax), len(gps_accel))
    s_ax_n = s_ax[:n]
    gps_accel_n = gps_accel_smooth[:n]

    # 1. Internal cross-correlation: phone GPS-derived accel vs phone IMU accel_x
    print(f"\n  [1] Phone-internal: GPS-derived accel vs accel_x")
    lags_int, corrs_int, best_lag_int, best_corr_int = lag_correlation_sweep(
        s_ax_n, gps_accel_n, 100  # +/- 10 seconds
    )
    corr_int_zero = corrs_int[100]
    print(f"      Best lag: {best_lag_int} samples ({best_lag_int*dt:+.2f} s)")
    print(f"      Max corr: {best_corr_int:.4f}")
    print(f"      Corr at 0 lag: {corr_int_zero:.4f}")

    # 2. External: phone accel_x vs VBOX-derived accel (from C5.5.1)
    n_ext = min(len(s_ax), len(v_speed_kmh))
    vbox_accel = np.gradient(v_speed_kmh[:n_ext] / 3.6, dt)
    print(f"\n  [2] External: phone accel_x vs VBOX-derived accel")
    lags_ext, corrs_ext, best_lag_ext, best_corr_ext = lag_correlation_sweep(
        s_ax[:n_ext], vbox_accel, 100
    )
    print(f"      Best lag: {best_lag_ext} samples ({best_lag_ext*dt:+.2f} s)")
    print(f"      Max corr: {best_corr_ext:.4f}")

    # 3. External: phone GPS speed vs VBOX speed (sanity)
    n_spd = min(len(s_speed_ms), len(v_speed_kmh))
    print(f"\n  [3] Phone GPS speed vs VBOX speed")
    lags_spd, corrs_spd, best_lag_spd, best_corr_spd = lag_correlation_sweep(
        s_speed_ms[:n_spd] * 3.6, v_speed_kmh[:n_spd], 500
    )
    print(f"      Best lag: {best_lag_spd} samples ({best_lag_spd*dt:+.2f} s)")
    print(f"      Max corr: {best_corr_spd:.4f}")

    # 4. Phone GPS speed vs phone accel_x (internal, via speed)
    # This checks if GPS speed and IMU are on the same clock
    # If phone GPS and accel are aligned, GPS-derived accel should peak at lag=0 vs accel_x
    print(f"\n  [4] Summary:")
    print(f"      Phone GPS speed vs VBOX speed:   lag = {best_lag_spd*dt:+.1f}s")
    print(f"      Phone accel_x vs phone GPS accel: lag = {best_lag_int*dt:+.1f}s")
    print(f"      Phone accel_x vs VBOX accel:      lag = {best_lag_ext*dt:+.1f}s")

    # If internal lag ~ 0 and external lag ~ X, then phone is self-consistent
    # and the offset is purely S-vs-V file alignment
    internal_consistent = abs(best_lag_int) <= 5  # within 0.5s
    print(f"\n      Phone internally consistent: {'YES' if internal_consistent else 'NO'} "
          f"(internal lag = {best_lag_int*dt:+.2f}s)")

    results[trip] = {
        "phone_internal_lag_samples": int(best_lag_int),
        "phone_internal_lag_seconds": round(float(best_lag_int * dt), 2),
        "phone_internal_corr": round(float(best_corr_int), 4),
        "phone_internal_corr_at_zero": round(float(corr_int_zero), 4),
        "phone_vs_vbox_accel_lag_samples": int(best_lag_ext),
        "phone_vs_vbox_accel_lag_seconds": round(float(best_lag_ext * dt), 2),
        "phone_vs_vbox_accel_corr": round(float(best_corr_ext), 4),
        "phone_vs_vbox_speed_lag_samples": int(best_lag_spd),
        "phone_vs_vbox_speed_lag_seconds": round(float(best_lag_spd * dt), 2),
        "phone_vs_vbox_speed_corr": round(float(best_corr_spd), 4),
        "phone_internally_consistent": internal_consistent,
    }

    # Plots
    ax1 = axes[row, 0]
    lag_sec_int = lags_int * dt
    ax1.plot(lag_sec_int, corrs_int, 'b-', linewidth=1, label='GPS-accel vs accel_x (internal)')
    lag_sec_ext = lags_ext * dt
    ax1.plot(lag_sec_ext, corrs_ext, 'r-', linewidth=1, label='accel_x vs VBOX-accel (external)')
    ax1.axvline(best_lag_int * dt, color='b', linestyle='--', alpha=0.7,
                label=f'Int lag={best_lag_int*dt:+.1f}s (r={best_corr_int:.3f})')
    ax1.axvline(best_lag_ext * dt, color='r', linestyle='--', alpha=0.7,
                label=f'Ext lag={best_lag_ext*dt:+.1f}s (r={best_corr_ext:.3f})')
    ax1.axvline(0, color='gray', linestyle=':', linewidth=0.5)
    ax1.set_title(f'{trip} — Acceleration Cross-Correlations')
    ax1.set_ylabel('Pearson r')
    ax1.set_xlabel('Lag (s)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Overlay: first 30 seconds of aligned accel
    ax2 = axes[row, 1]
    t_plot = np.arange(min(300, n)) * dt
    np_plot = len(t_plot)
    ax2.plot(t_plot, gps_accel_n[:np_plot], 'g-', linewidth=0.8, alpha=0.7, label='Phone GPS-derived accel')
    ax2.plot(t_plot, s_ax_n[:np_plot], 'r-', linewidth=0.5, alpha=0.5, label='Phone accel_x (raw)')
    if n_ext >= np_plot:
        ax2.plot(t_plot, vbox_accel[:np_plot], 'k-', linewidth=1, alpha=0.8, label='VBOX-derived accel')
    ax2.set_title(f'{trip} — Accel Overlay (first 30s)')
    ax2.set_ylabel('m/s^2')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-8, 8)

plt.tight_layout()
fig_path = FIG_DIR / "c5_5_1b_internal_consistency.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig_path}")

# Save JSON
json_path = REPO_ROOT / "results" / "c5_5_1b_internal_consistency.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {json_path}")

# Summary
print("\n" + "="*70)
print("  SUMMARY: Phone Internal Timing Consistency")
print("="*70)
for trip in trips:
    r = results[trip]
    print(f"\n  {trip}:")
    print(f"    Internal (GPS-accel vs IMU): lag={r['phone_internal_lag_seconds']:+.2f}s, "
          f"r={r['phone_internal_corr']:.4f}")
    print(f"    External (IMU vs VBOX):      lag={r['phone_vs_vbox_accel_lag_seconds']:+.2f}s, "
          f"r={r['phone_vs_vbox_accel_corr']:.4f}")
    print(f"    Speed   (GPS vs VBOX):       lag={r['phone_vs_vbox_speed_lag_seconds']:+.2f}s, "
          f"r={r['phone_vs_vbox_speed_corr']:.4f}")
    print(f"    Internally consistent: {r['phone_internally_consistent']}")
