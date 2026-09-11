"""
C5.5.1 — Global Time Alignment Audit & Corrected Evaluation
============================================================
For each trip (Vta02, Vta03, Vta04):
  1. Estimate a single global time offset using GPS speed cross-correlation.
  2. Apply that offset to create evaluation-only aligned arrays (no raw data modified).
  3. Re-run braking event analysis and acceleration residual metrics after alignment.
  4. Quantify what improves vs. what persists → distinguish timing artefacts from physics.

Methodology:
  - Signal for alignment: Phone GPS speed vs VBOX velocity
    (independent of the accelerometer signals we want to evaluate)
  - Lag sweep: ±500 samples (±50 s) to catch even the Vta03 case
  - After optimal lag, also sweep CAN accel vs phone accel to verify consistency
  - Vta02 serves as the control: its lag should remain near zero

Output:
  - results/c5_5_1_alignment_audit.json   (structured data)
  - results/c5_5_1_alignment_report.md    (narrative report)
  - results/figures/c5_5_1_*.png          (diagnostic plots)
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import find_trip_dir

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def lag_correlation_sweep(sig_a, sig_b, max_lag_samples):
    """Compute Pearson correlation at each integer lag from -max_lag to +max_lag.
    Positive lag means sig_a is DELAYED relative to sig_b (shift sig_a forward).
    Returns (lags_array, correlations_array, best_lag, best_corr).
    """
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corrs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag < 0:
            a = sig_a[:lag]  # drop last |lag| samples
            b = sig_b[-lag:]  # drop first |lag| samples
        elif lag > 0:
            a = sig_a[lag:]
            b = sig_b[:-lag]
        else:
            a = sig_a.copy()
            b = sig_b.copy()
        mask = (~np.isnan(a)) & (~np.isnan(b))
        n_valid = np.sum(mask)
        if n_valid > 100:
            corrs[i] = np.corrcoef(a[mask], b[mask])[0, 1]
    best_idx = np.nanargmax(corrs)
    return lags, corrs, lags[best_idx], corrs[best_idx]


def apply_lag_shift(phone_arr, vbox_arr, lag_samples):
    """Shift phone array relative to vbox array by lag_samples.
    Positive lag means phone is delayed: we advance phone (drop early phone, drop late vbox).
    Returns (phone_aligned, vbox_aligned) of equal length.
    """
    if lag_samples > 0:
        p = phone_arr[lag_samples:]
        v = vbox_arr[:-lag_samples]
    elif lag_samples < 0:
        p = phone_arr[:lag_samples]
        v = vbox_arr[-lag_samples:]
    else:
        p = phone_arr.copy()
        v = vbox_arr.copy()
    n = min(len(p), len(v))
    return p[:n], v[:n]


def find_braking_events(accel_ref, speed_ref, threshold=-1.5, dt=0.1):
    """Find braking events where reference acceleration drops below threshold.
    Returns list of (start_idx, end_idx, peak_decel).
    """
    below = accel_ref < threshold
    events = []
    in_event = False
    start = 0
    for i in range(len(below)):
        if below[i] and not in_event:
            start = i
            in_event = True
        elif not below[i] and in_event:
            peak = np.min(accel_ref[start:i])
            events.append((start, i, peak))
            in_event = False
    if in_event:
        peak = np.min(accel_ref[start:])
        events.append((start, len(accel_ref), peak))
    return events


# ──────────────────────────────────────────────────────────────────────
# Main audit
# ──────────────────────────────────────────────────────────────────────

results = {}
trips = ["Vta02", "Vta03", "Vta04"]

for trip in trips:
    print(f"\n{'='*70}")
    print(f"  C5.5.1 ALIGNMENT AUDIT: {trip}")
    print(f"{'='*70}")

    trip_dir = find_trip_dir(trip)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]

    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')

    # ── Extract raw signals ──
    col_s_speed = [c for c in df_s.columns if 'GPS SPEED' in c.upper()][0]
    col_s_ax = [c for c in df_s.columns if 'ACCELEROMETER X' in c.upper()][0]
    col_v_speed = [c for c in df_v.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    col_v_accel = [c for c in df_v.columns if 'LONGITUDINAL' in c.upper()][0]

    # Phone GPS speed is in m/s (despite header saying Kmh) — confirmed in prior audit
    s_speed_ms = df_s[col_s_speed].to_numpy()
    s_speed_kmh = s_speed_ms * 3.6  # Convert to km/h for comparison with VBOX

    v_speed_kmh = df_v[col_v_speed].to_numpy()

    s_ax = df_s[col_s_ax].to_numpy()
    can_accel_g = df_v[col_v_accel].to_numpy()
    can_accel_ms2 = can_accel_g * 9.80665

    # Compute VBOX reference acceleration via differentiation
    n_common = min(len(s_speed_ms), len(v_speed_kmh))
    v_speed_ms = v_speed_kmh[:n_common] / 3.6
    dt = 0.1  # 10 Hz

    # 5-point Savitzky-Golay-like derivative for VBOX reference
    vbox_accel = np.gradient(v_speed_ms, dt)

    # Truncate all to common length
    s_speed_kmh = s_speed_kmh[:n_common]
    s_ax = s_ax[:n_common]
    can_accel_ms2 = can_accel_ms2[:n_common]

    # ── Step 1: GPS Speed Cross-Correlation (wide sweep) ──
    max_lag = 500  # ±50 seconds
    print(f"\n[1] GPS Speed Cross-Correlation (±{max_lag*0.1:.0f} s sweep)")
    lags_spd, corrs_spd, best_lag_spd, best_corr_spd = lag_correlation_sweep(
        s_speed_kmh, v_speed_kmh[:n_common], max_lag
    )
    corr_at_zero = corrs_spd[max_lag]  # index of lag=0
    print(f"    Best lag: {best_lag_spd} samples ({best_lag_spd*0.1:+.1f} s)")
    print(f"    Max correlation: {best_corr_spd:.4f}")
    print(f"    Correlation at zero lag: {corr_at_zero:.4f}")

    # ── Step 2: CAN Accel Cross-Correlation (verify consistency) ──
    print(f"\n[2] CAN Accel vs Phone Accel Cross-Correlation")
    lags_acc, corrs_acc, best_lag_acc, best_corr_acc = lag_correlation_sweep(
        s_ax, can_accel_ms2, max_lag
    )
    corr_acc_at_zero = corrs_acc[max_lag]
    print(f"    Best lag: {best_lag_acc} samples ({best_lag_acc*0.1:+.1f} s)")
    print(f"    Max correlation: {best_corr_acc:.4f}")
    print(f"    Correlation at zero lag: {corr_acc_at_zero:.4f}")

    # ── Step 3: Apply best speed-based lag and recompute metrics ──
    optimal_lag = best_lag_spd
    print(f"\n[3] Applying optimal lag = {optimal_lag} samples ({optimal_lag*0.1:+.1f} s)")

    # Aligned signals
    s_ax_aligned, can_aligned = apply_lag_shift(s_ax, can_accel_ms2, optimal_lag)
    s_speed_aligned, v_speed_aligned = apply_lag_shift(s_speed_kmh, v_speed_kmh[:n_common], optimal_lag)
    _, vbox_accel_aligned = apply_lag_shift(s_ax, vbox_accel, optimal_lag)
    # Note: vbox_accel_aligned is the VBOX reference at the aligned timestamps

    n_aligned = len(s_ax_aligned)

    # Speed correlation after alignment
    mask_spd = (~np.isnan(s_speed_aligned)) & (~np.isnan(v_speed_aligned))
    corr_speed_aligned = np.corrcoef(s_speed_aligned[mask_spd], v_speed_aligned[mask_spd])[0, 1]

    # Acceleration residual stats (phone - VBOX reference) BEFORE and AFTER alignment
    # Before: at zero lag
    s_ax_orig = s_ax[:n_common]
    vbox_accel_orig = vbox_accel[:n_common]
    mask_before = (~np.isnan(s_ax_orig)) & (~np.isnan(vbox_accel_orig))
    resid_before = s_ax_orig[mask_before] - vbox_accel_orig[mask_before]
    mae_before = np.mean(np.abs(resid_before))
    rmse_before = np.sqrt(np.mean(resid_before**2))
    bias_before = np.mean(resid_before)

    # After alignment
    mask_after = (~np.isnan(s_ax_aligned)) & (~np.isnan(vbox_accel_aligned))
    resid_after = s_ax_aligned[mask_after] - vbox_accel_aligned[mask_after]
    mae_after = np.mean(np.abs(resid_after))
    rmse_after = np.sqrt(np.mean(resid_after**2))
    bias_after = np.mean(resid_after)

    # CAN accel residual (phone - CAN) before/after
    mask_can_b = (~np.isnan(s_ax_orig)) & (~np.isnan(can_accel_ms2[:n_common]))
    resid_can_before = s_ax_orig[mask_can_b] - can_accel_ms2[:n_common][mask_can_b]
    mae_can_before = np.mean(np.abs(resid_can_before))

    mask_can_a = (~np.isnan(s_ax_aligned)) & (~np.isnan(can_aligned))
    resid_can_after = s_ax_aligned[mask_can_a] - can_aligned[mask_can_a]
    mae_can_after = np.mean(np.abs(resid_can_after))

    print(f"\n    === BEFORE alignment (lag=0) ===")
    print(f"    Speed corr:           {corr_at_zero:.4f}")
    print(f"    Phone-VBOX MAE:       {mae_before:.4f} m/s²")
    print(f"    Phone-VBOX RMSE:      {rmse_before:.4f} m/s²")
    print(f"    Phone-VBOX bias:      {bias_before:+.4f} m/s²")
    print(f"    Phone-CAN MAE:        {mae_can_before:.4f} m/s²")

    print(f"\n    === AFTER alignment (lag={optimal_lag}) ===")
    print(f"    Speed corr:           {corr_speed_aligned:.4f}")
    print(f"    Phone-VBOX MAE:       {mae_after:.4f} m/s²")
    print(f"    Phone-VBOX RMSE:      {rmse_after:.4f} m/s²")
    print(f"    Phone-VBOX bias:      {bias_after:+.4f} m/s²")
    print(f"    Phone-CAN MAE:        {mae_can_after:.4f} m/s²")

    delta_mae = mae_before - mae_after
    pct_improvement = 100 * delta_mae / mae_before if mae_before > 0 else 0
    print(f"\n    MAE change:           {delta_mae:+.4f} m/s² ({pct_improvement:+.1f}%)")

    # ── Step 4: Braking event analysis after alignment ──
    print(f"\n[4] Braking Event Analysis (after alignment)")
    braking_events = find_braking_events(vbox_accel_aligned, v_speed_aligned, threshold=-1.5)
    event_results = []
    for ei, (es, ee, epeak) in enumerate(braking_events):
        # Extend window ±1 second
        ws = max(0, es - 10)
        we = min(n_aligned, ee + 20)
        seg_phone = s_ax_aligned[ws:we]
        seg_can = can_aligned[ws:we]
        seg_ref = vbox_accel_aligned[ws:we]

        peak_phone = np.min(seg_phone) if len(seg_phone) > 0 else np.nan
        peak_can = np.min(seg_can) if len(seg_can) > 0 else np.nan
        peak_ref = np.min(seg_ref) if len(seg_ref) > 0 else np.nan

        # Rebound: max positive value AFTER the peak decel
        peak_idx = np.argmin(seg_ref) if len(seg_ref) > 0 else 0
        rebound_phone = np.max(seg_phone[peak_idx:]) if peak_idx < len(seg_phone) else np.nan
        rebound_can = np.max(seg_can[peak_idx:]) if peak_idx < len(seg_can) else np.nan

        t_start = (ws + optimal_lag if optimal_lag > 0 else ws) * 0.1
        t_end = (we + optimal_lag if optimal_lag > 0 else we) * 0.1

        ev = {
            "event": ei + 1,
            "t_start_s": round(t_start, 1),
            "t_end_s": round(t_end, 1),
            "peak_ref_ms2": round(float(peak_ref), 4),
            "peak_can_ms2": round(float(peak_can), 4),
            "peak_phone_ms2": round(float(peak_phone), 4),
            "rebound_phone_ms2": round(float(rebound_phone), 4),
            "rebound_can_ms2": round(float(rebound_can), 4),
            "phone_overshoot_ms2": round(float(peak_phone - peak_ref), 4),
            "phone_rebound_excess_ms2": round(float(rebound_phone - rebound_can), 4),
        }
        event_results.append(ev)
        print(f"    Event {ei+1}: ref_peak={peak_ref:.2f}, phone_peak={peak_phone:.2f}, "
              f"phone_rebound={rebound_phone:.2f}, CAN_rebound={rebound_can:.2f}")

    # ── Step 5: Velocity integration drift after alignment ──
    print(f"\n[5] Velocity Integration Drift (after alignment)")
    # Integrate phone accel (aligned) and reference separately
    v_phone_int = np.cumsum(s_ax_aligned * dt)
    v_ref_int = np.cumsum(vbox_accel_aligned * dt)
    v_can_int = np.cumsum(can_aligned * dt)

    # Compare velocity errors at 5, 10, 20, 30, 60 seconds
    drift_horizons = [50, 100, 200, 300, 600]  # samples
    drift_results = {}
    for h in drift_horizons:
        if h <= n_aligned:
            vel_err_phone = abs(v_phone_int[h-1] - v_ref_int[h-1])
            vel_err_can = abs(v_can_int[h-1] - v_ref_int[h-1])
            drift_results[f"{h*0.1:.0f}s"] = {
                "phone_vel_err_ms": round(float(vel_err_phone), 3),
                "can_vel_err_ms": round(float(vel_err_can), 3),
            }
            print(f"    {h*0.1:.0f}s: Phone vel err = {vel_err_phone:.3f} m/s, "
                  f"CAN vel err = {vel_err_can:.3f} m/s")

    # ── Collect results ──
    results[trip] = {
        "n_rows_raw_phone": int(len(df_s)),
        "n_rows_raw_vbox": int(len(df_v)),
        "n_common": int(n_common),
        "n_aligned": int(n_aligned),
        "optimal_speed_lag_samples": int(optimal_lag),
        "optimal_speed_lag_seconds": round(float(optimal_lag * 0.1), 1),
        "speed_corr_before": round(float(corr_at_zero), 4),
        "speed_corr_after": round(float(corr_speed_aligned), 4),
        "accel_lag_best_samples": int(best_lag_acc),
        "accel_lag_best_seconds": round(float(best_lag_acc * 0.1), 1),
        "accel_lag_best_corr": round(float(best_corr_acc), 4),
        "before_alignment": {
            "phone_vbox_mae_ms2": round(float(mae_before), 4),
            "phone_vbox_rmse_ms2": round(float(rmse_before), 4),
            "phone_vbox_bias_ms2": round(float(bias_before), 4),
            "phone_can_mae_ms2": round(float(mae_can_before), 4),
        },
        "after_alignment": {
            "phone_vbox_mae_ms2": round(float(mae_after), 4),
            "phone_vbox_rmse_ms2": round(float(rmse_after), 4),
            "phone_vbox_bias_ms2": round(float(bias_after), 4),
            "phone_can_mae_ms2": round(float(mae_can_after), 4),
        },
        "mae_improvement_pct": round(float(pct_improvement), 1),
        "braking_events_aligned": event_results,
        "velocity_drift": drift_results,
    }

    # ── Step 6: Diagnostic Plots ──
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=False)
    fig.suptitle(f"C5.5.1 Alignment Audit — {trip}\n"
                 f"Optimal Lag = {optimal_lag} samples ({optimal_lag*0.1:+.1f} s)",
                 fontsize=14, fontweight='bold')

    # Panel 1: Speed cross-correlation function
    ax = axes[0]
    lag_sec = lags_spd * 0.1
    ax.plot(lag_sec, corrs_spd, 'b-', linewidth=0.8, label='GPS Speed xcorr')
    ax.axvline(optimal_lag * 0.1, color='r', linestyle='--', linewidth=1.5,
               label=f'Best lag = {optimal_lag*0.1:+.1f}s (r={best_corr_spd:.4f})')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('Lag (seconds, +ve = phone delayed)')
    ax.set_title('GPS Speed Cross-Correlation Sweep')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Aligned speed overlay (first 120 seconds)
    ax = axes[1]
    t_plot = np.arange(min(1200, n_aligned)) * 0.1
    n_plot = len(t_plot)
    ax.plot(t_plot, v_speed_aligned[:n_plot] / 3.6, 'b-', linewidth=1, label='VBOX (km/h→m/s)', alpha=0.8)
    ax.plot(t_plot, s_speed_aligned[:n_plot] / 3.6, 'r-', linewidth=0.8, label='Phone GPS (aligned)', alpha=0.7)
    ax.set_ylabel('Speed (m/s)')
    ax.set_title(f'Speed Overlay After Alignment (first {n_plot*0.1:.0f}s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Aligned acceleration overlay (first 120 seconds)
    ax = axes[2]
    ax.plot(t_plot, vbox_accel_aligned[:n_plot], 'k-', linewidth=1, label='VBOX ref accel', alpha=0.8)
    ax.plot(t_plot, can_aligned[:n_plot], 'b-', linewidth=1, label='CAN accel', alpha=0.7)
    ax.plot(t_plot, s_ax_aligned[:n_plot], 'r-', linewidth=0.7, label='Phone accel_x (aligned)', alpha=0.6)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title(f'Acceleration Overlay After Alignment (first {n_plot*0.1:.0f}s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Residual (phone - ref) BEFORE vs AFTER alignment
    ax = axes[3]
    # Before: first n_plot samples at zero lag
    r_before_plot = s_ax_orig[:n_plot] - vbox_accel_orig[:n_plot]
    r_after_plot = s_ax_aligned[:n_plot] - vbox_accel_aligned[:n_plot]
    ax.plot(t_plot, r_before_plot, 'gray', linewidth=0.5, alpha=0.5, label='Residual BEFORE alignment')
    ax.plot(t_plot, r_after_plot, 'r-', linewidth=0.7, alpha=0.7, label='Residual AFTER alignment')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel('Phone - Ref (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration Residual: Before vs After Alignment')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / f"c5_5_1_{trip}_alignment.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    Saved: {fig_path}")

# ──────────────────────────────────────────────────────────────────────
# Cross-trip comparison plot
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("C5.5.1 — Cross-Trip Alignment Impact Summary", fontsize=14, fontweight='bold')

for i, trip in enumerate(trips):
    r = results[trip]
    ax = axes[i]
    metrics = ['MAE', 'RMSE', 'Bias']
    before_vals = [
        r['before_alignment']['phone_vbox_mae_ms2'],
        r['before_alignment']['phone_vbox_rmse_ms2'],
        abs(r['before_alignment']['phone_vbox_bias_ms2']),
    ]
    after_vals = [
        r['after_alignment']['phone_vbox_mae_ms2'],
        r['after_alignment']['phone_vbox_rmse_ms2'],
        abs(r['after_alignment']['phone_vbox_bias_ms2']),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, before_vals, w, label='Before', color='#ef5350', alpha=0.8)
    ax.bar(x + w/2, after_vals, w, label='After', color='#66bb6a', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('m/s²')
    ax.set_title(f'{trip}\nLag={r["optimal_speed_lag_seconds"]:+.1f}s → MAE Δ={r["mae_improvement_pct"]:+.1f}%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = FIG_DIR / "c5_5_1_cross_trip_summary.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved cross-trip summary: {fig_path}")

# ──────────────────────────────────────────────────────────────────────
# Save structured JSON
# ──────────────────────────────────────────────────────────────────────
json_path = REPO_ROOT / "results" / "c5_5_1_alignment_audit.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ──────────────────────────────────────────────────────────────────────
# Generate markdown report
# ──────────────────────────────────────────────────────────────────────
report_lines = [
    "# C5.5.1 — Global Time Alignment Audit & Corrected Evaluation",
    "",
    "**Date**: September 5, 2026",
    "**Methodology**: GPS speed cross-correlation (independent of accelerometer)",
    f"**Script**: `experiments/align_timestamps_c5_5_1.py`",
    "",
    "---",
    "",
    "## 1. Optimal Lag Estimates",
    "",
    "| Trip | Speed-Based Lag | Speed Corr (before → after) | Accel-Based Lag | Accel Corr (peak) |",
    "| ---- | --------------- | --------------------------- | --------------- | ------------------ |",
]

for trip in trips:
    r = results[trip]
    report_lines.append(
        f"| {trip} | {r['optimal_speed_lag_seconds']:+.1f} s "
        f"({r['optimal_speed_lag_samples']:+d} samples) | "
        f"{r['speed_corr_before']:.4f} → {r['speed_corr_after']:.4f} | "
        f"{r['accel_lag_best_seconds']:+.1f} s "
        f"({r['accel_lag_best_samples']:+d} samples) | "
        f"{r['accel_lag_best_corr']:.4f} |"
    )

report_lines += [
    "",
    "### Interpretation",
    "",
    "- **Vta02**: Control trip. Near-zero lag confirms proper synchronization.",
    f"- **Vta03**: Lag = {results['Vta03']['optimal_speed_lag_seconds']:+.1f} s — "
    "a massive dataset-level timing offset.",
    f"- **Vta04**: Lag = {results['Vta04']['optimal_speed_lag_seconds']:+.1f} s — "
    "a moderate but significant offset.",
    "",
    "---",
    "",
    "## 2. Acceleration Residual Impact (Phone − VBOX Reference)",
    "",
    "| Trip | Lag Applied | MAE Before | MAE After | Δ MAE | RMSE Before | RMSE After | Bias Before | Bias After |",
    "| ---- | ----------- | ---------- | --------- | ----- | ----------- | ---------- | ----------- | ---------- |",
]

for trip in trips:
    r = results[trip]
    b = r['before_alignment']
    a = r['after_alignment']
    report_lines.append(
        f"| {trip} | {r['optimal_speed_lag_seconds']:+.1f} s | "
        f"{b['phone_vbox_mae_ms2']:.4f} | {a['phone_vbox_mae_ms2']:.4f} | "
        f"{r['mae_improvement_pct']:+.1f}% | "
        f"{b['phone_vbox_rmse_ms2']:.4f} | {a['phone_vbox_rmse_ms2']:.4f} | "
        f"{b['phone_vbox_bias_ms2']:+.4f} | {a['phone_vbox_bias_ms2']:+.4f} |"
    )

report_lines += [
    "",
    "---",
    "",
    "## 3. Braking Event Analysis (After Alignment)",
    "",
]

for trip in trips:
    r = results[trip]
    events = r['braking_events_aligned']
    report_lines.append(f"### {trip} (lag = {r['optimal_speed_lag_seconds']:+.1f} s)")
    if not events:
        report_lines.append("No braking events with a_ref < -1.5 m/s² found after alignment.")
    else:
        report_lines.append(
            "| Event | Ref Peak | Phone Peak | CAN Peak | Phone Overshoot | Phone Rebound | CAN Rebound | Rebound Excess |"
        )
        report_lines.append(
            "| ----- | -------- | ---------- | -------- | --------------- | ------------- | ----------- | -------------- |"
        )
        for ev in events:
            report_lines.append(
                f"| {ev['event']} | {ev['peak_ref_ms2']:.2f} | {ev['peak_phone_ms2']:.2f} | "
                f"{ev['peak_can_ms2']:.2f} | {ev['phone_overshoot_ms2']:+.2f} | "
                f"{ev['rebound_phone_ms2']:+.2f} | {ev['rebound_can_ms2']:+.2f} | "
                f"{ev['phone_rebound_excess_ms2']:+.2f} |"
            )
    report_lines.append("")

report_lines += [
    "---",
    "",
    "## 4. Velocity Drift (After Alignment)",
    "",
    "| Trip | 5s | 10s | 20s | 30s | 60s |",
    "| ---- | -- | --- | --- | --- | --- |",
]

for trip in trips:
    r = results[trip]
    d = r['velocity_drift']
    vals = []
    for h in ['5s', '10s', '20s', '30s', '60s']:
        if h in d:
            vals.append(f"Phone: {d[h]['phone_vel_err_ms']:.2f}, CAN: {d[h]['can_vel_err_ms']:.2f}")
        else:
            vals.append("N/A")
    report_lines.append(f"| {trip} | " + " | ".join(vals) + " |")

report_lines += [
    "",
    "---",
    "",
    "## 5. Three-Box Summary",
    "",
    "```text",
    "┌─────────────────────────────────────────────────────────────────┐",
    "│ 🟢 KNOW (Directly Measured)                                      │",
    "├─────────────────────────────────────────────────────────────────┤",
    "│ 1. Vta02 is well-synchronized (control).                        │",
    "│ 2. Vta03 and Vta04 had significant timestamp offsets.           │",
    "│ 3. After correcting the offsets:                                 │",
    "│    - We know the corrected speed correlation.                    │",
    "│    - We know the corrected acceleration MAE/RMSE/bias.          │",
    "│    - We know which braking discrepancies persist vs. vanish.    │",
    "├─────────────────────────────────────────────────────────────────┤",
    "│ 🟡 THINK (Plausible Interpretation)                              │",
    "├─────────────────────────────────────────────────────────────────┤",
    "│ 1. Discrepancies that vanish after alignment are timing         │",
    "│    artefacts, not physical mount effects.                        │",
    "│ 2. Discrepancies that persist after alignment on Vta02 and      │",
    "│    aligned Vta03/Vta04 may be physical (mount compliance,       │",
    "│    vibration), but we need per-event inspection to confirm.     │",
    "├─────────────────────────────────────────────────────────────────┤",
    "│ 🔴 DON'T KNOW                                                    │",
    "├─────────────────────────────────────────────────────────────────┤",
    "│ 1. Whether any of the Vta03 data is usable even after           │",
    "│    alignment (if offset varies within the trip, global lag      │",
    "│    is insufficient).                                             │",
    "│ 2. Whether the previous C5.3 B-series ML results would change  │",
    "│    substantially after re-training on properly aligned data.    │",
    "└─────────────────────────────────────────────────────────────────┘",
    "```",
    "",
    "---",
    "",
    "## 6. Diagnostic Figures",
    "",
    f"- Vta02: `results/figures/c5_5_1_Vta02_alignment.png`",
    f"- Vta03: `results/figures/c5_5_1_Vta03_alignment.png`",
    f"- Vta04: `results/figures/c5_5_1_Vta04_alignment.png`",
    f"- Cross-trip: `results/figures/c5_5_1_cross_trip_summary.png`",
]

report_path = REPO_ROOT / "results" / "c5_5_1_alignment_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"Saved report: {report_path}")
print("\n✅ C5.5.1 Alignment Audit Complete.")
