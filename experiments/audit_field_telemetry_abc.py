#!/usr/bin/env python3
"""
Field Telemetry Audit Script: Protocols A, B, and C
Author: DeepMind / IDR Engineering Team
Purpose: Comprehensive 10-point sensor, ML, and navigation state diagnostic for:
  - Test A: Stationary Desk (5 min) - Baseline sensor noise, static drift, TCN speed at true zero.
  - Test B: Vehicle Stationary Engine Idling (5 min) - Engine vibration signature, mount stability, TCN robustness to idle.
  - Test C: Driving Test (10-15 min) - Dynamic TCN vs GNSS correlation, heading tracking, outage drift performance.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def parse_telemetry(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]
    
    # Flexible alias mapping
    col_map = {
        'ax': 'accel_x', 'ay': 'accel_y', 'az': 'accel_z',
        'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z',
        'mx': 'mag_x', 'my': 'mag_y', 'mz': 'mag_z',
        'ref_heading': 'heading_deg',
        'ref_pos_e': 'pos_e', 'ref_pos_n': 'pos_n', 'ref_pos_u': 'pos_u',
        'time_s': 'time_s'
    }
    for old_c, new_c in col_map.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]
            
    if 'time_s' in df.columns and 'timestamp_ms' not in df.columns:
        df['timestamp_ms'] = (df['time_s'] * 1000.0).astype(np.int64)
        
    return df

def audit_file(df, label="Test"):
    """Audits a single telemetry dataframe across all available channels."""
    res = {"label": label, "num_epochs": len(df)}
    if len(df) == 0:
        return res
    
    # Check timestamps and sampling rate (Point 7)
    if 'timestamp_ms' in df.columns:
        dt_ms = np.diff(df['timestamp_ms'].values)
        res['dt_mean_ms'] = float(np.mean(dt_ms)) if len(dt_ms) > 0 else 100.0
        res['dt_std_ms'] = float(np.std(dt_ms)) if len(dt_ms) > 0 else 0.0
        res['dt_min_ms'] = float(np.min(dt_ms)) if len(dt_ms) > 0 else 100.0
        res['dt_max_ms'] = float(np.max(dt_ms)) if len(dt_ms) > 0 else 100.0
        res['jitter_rate_pct'] = float(np.mean((dt_ms < 60) | (dt_ms > 140)) * 100.0) if len(dt_ms) > 0 else 0.0
        duration_s = (df['timestamp_ms'].iloc[-1] - df['timestamp_ms'].iloc[0]) / 1000.0
        res['duration_s'] = float(duration_s)
    else:
        res['duration_s'] = len(df) * 0.1

    # Accelerometer Noise & Bias (Point 1)
    acc_cols = [c for c in ['accel_x', 'accel_y', 'accel_z'] if c in df.columns]
    if len(acc_cols) == 3:
        ax, ay, az = df['accel_x'].values, df['accel_y'].values, df['accel_z'].values
        anorm = np.sqrt(ax**2 + ay**2 + az**2)
        res['acc_norm_mean'] = float(np.mean(anorm))
        res['acc_norm_std'] = float(np.std(anorm))
        res['acc_norm_err_vs_g'] = float(np.mean(anorm) - 9.80665)
        res['acc_std_x'] = float(np.std(ax))
        res['acc_std_y'] = float(np.std(ay))
        res['acc_std_z'] = float(np.std(az))
    
    # Gyroscope Bias & Noise (Point 2)
    gyr_cols = [c for c in ['gyro_x', 'gyro_y', 'gyro_z'] if c in df.columns]
    if len(gyr_cols) == 3:
        gx, gy, gz = df['gyro_x'].values, df['gyro_y'].values, df['gyro_z'].values
        # rad/s to deg/s
        gx_deg, gy_deg, gz_deg = np.rad2deg(gx), np.rad2deg(gy), np.rad2deg(gz)
        res['gyro_bias_deg_x'] = float(np.mean(gx_deg))
        res['gyro_bias_deg_y'] = float(np.mean(gy_deg))
        res['gyro_bias_deg_z'] = float(np.mean(gz_deg))
        res['gyro_std_deg_x'] = float(np.std(gx_deg))
        res['gyro_std_deg_y'] = float(np.std(gy_deg))
        res['gyro_std_deg_z'] = float(np.std(gz_deg))
        res['gyro_norm_mean_deg'] = float(np.mean(np.sqrt(gx_deg**2 + gy_deg**2 + gz_deg**2)))

    # Gravity Vector Stability & Mount Movement (Point 3 & 9)
    grav_cols = [c for c in ['grav_x', 'grav_y', 'grav_z'] if c in df.columns]
    if len(grav_cols) == 3:
        gvec = df[['grav_x', 'grav_y', 'grav_z']].values
        gnorm = np.linalg.norm(gvec, axis=1)
        res['grav_norm_mean'] = float(np.mean(gnorm))
        res['grav_norm_std'] = float(np.std(gnorm))
        # Angle from initial orientation
        g0 = gvec[0] / (np.linalg.norm(g0_raw := gvec[0]) + 1e-9)
        g_unit = gvec / (gnorm[:, None] + 1e-9)
        dot_prod = np.clip(np.sum(g_unit * g0, axis=1), -1.0, 1.0)
        tilt_angles_deg = np.rad2deg(np.arccos(dot_prod))
        res['tilt_drift_max_deg'] = float(np.max(tilt_angles_deg))
        res['tilt_drift_std_deg'] = float(np.std(tilt_angles_deg))
        res['mount_slip_detected'] = bool(np.max(tilt_angles_deg) > 5.0)

    # TCN Speed Inference (Point 4)
    if 'ml_speed' in df.columns:
        mls = df['ml_speed'].values
        res['ml_speed_mean'] = float(np.mean(mls))
        res['ml_speed_median'] = float(np.median(mls))
        res['ml_speed_p90'] = float(np.percentile(mls, 90))
        res['ml_speed_p95'] = float(np.percentile(mls, 95))
        res['ml_speed_max'] = float(np.max(mls))
        res['ml_speed_pct_under_01'] = float(np.mean(mls < 0.1) * 100.0)
        res['ml_speed_pct_under_02'] = float(np.mean(mls < 0.2) * 100.0)
        res['ml_speed_pct_under_05'] = float(np.mean(mls < 0.5) * 100.0)

    # GNSS Speed vs TCN Speed (Point 5)
    if 'ml_speed' in df.columns and 'gnss_speed' in df.columns:
        mls = df['ml_speed'].values
        gns = df['gnss_speed'].values
        valid_mask = np.isfinite(mls) & np.isfinite(gns)
        # Check dynamic driving subset (> 0.5 m/s)
        dyn_mask = valid_mask & (gns > 0.5)
        if np.sum(dyn_mask) > 10:
            v_tcn = mls[dyn_mask]
            v_gps = gns[dyn_mask]
            r_val, _ = pearsonr(v_tcn, v_gps)
            res['speed_corr_r'] = float(r_val)
            res['speed_r2'] = float(r_val**2)
            res['speed_mae'] = float(np.mean(np.abs(v_tcn - v_gps)))
            res['speed_rmse'] = float(np.sqrt(np.mean((v_tcn - v_gps)**2)))
            # Scale factor (slope without intercept)
            slope = float(np.sum(v_tcn * v_gps) / (np.sum(v_gps**2) + 1e-9))
            res['speed_scale_factor'] = slope
        else:
            res['speed_corr_r'] = 0.0
            res['speed_r2'] = 0.0
            res['speed_mae'] = 0.0
            res['speed_rmse'] = 0.0
            res['speed_scale_factor'] = 1.0

    # Heading Stability & Tracking (Point 6)
    if 'heading_deg' in df.columns:
        hdg = df['heading_deg'].values
        # Unwrapped heading for drift calculation
        hdg_unwrapped = np.unwrap(np.deg2rad(hdg))
        drift_rate_deg_min = (np.rad2deg(hdg_unwrapped[-1] - hdg_unwrapped[0])) / (max(res['duration_s'], 1.0) / 60.0)
        res['heading_drift_deg_min'] = float(drift_rate_deg_min)
        res['heading_std_deg'] = float(np.std(hdg))

        if 'gnss_bearing' in df.columns and 'gnss_speed' in df.columns:
            gns_mask = (df['gnss_speed'] > 2.0) & (df['gnss_accuracy'] < 10.0) if 'gnss_accuracy' in df.columns else (df['gnss_speed'] > 2.0)
            if np.sum(gns_mask) > 10:
                h_est = hdg[gns_mask]
                h_gps = df['gnss_bearing'].values[gns_mask]
                diff = (h_est - h_gps + 180.0) % 360.0 - 180.0
                res['heading_err_vs_gnss_mean'] = float(np.mean(np.abs(diff)))
                res['heading_err_vs_gnss_median'] = float(np.median(np.abs(diff)))

    # Navigation State Integrity (Point 10)
    pos_cols = [c for c in ['pos_e', 'pos_n', 'pos_u'] if c in df.columns]
    if len(pos_cols) == 3:
        pe, pn, pu = df['pos_e'].values, df['pos_n'].values, df['pos_u'].values
        res['pos_u_min'] = float(np.min(pu))
        res['pos_u_max'] = float(np.max(pu))
        res['pos_u_std'] = float(np.std(pu))
        res['pos_u_final'] = float(pu[-1])
        res['vertical_diverged'] = bool(np.abs(pu[-1]) > 50.0 or np.std(pu) > 20.0)
        res['final_pos_drift_norm'] = float(np.sqrt(pe[-1]**2 + pn[-1]**2))

    return res

def analyze_vibration_psd(df_a, df_b, df_c, output_fig):
    """Point 8: Spectral density comparison across desk, idle, and driving."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    dfs = [("Test A (Desk)", df_a, '#2b5c8f'), 
           ("Test B (Idle)", df_b, '#d95f02'), 
           ("Test C (Driving)", df_c, '#1b9e77')]
    
    fs = 10.0 # 10 Hz sampling rate
    peak_info = {}

    for ax_idx, axis_name in enumerate(['accel_x', 'accel_y', 'accel_z']):
        ax = axes[ax_idx]
        for name, d, col in dfs:
            if d is not None and axis_name in d.columns and len(d) > 30:
                vals = d[axis_name].values - np.mean(d[axis_name].values)
                nperseg = min(len(vals), 128)
                freqs, psd = welch(vals, fs=fs, nperseg=nperseg)
                ax.semilogy(freqs, psd, label=f"{name} ({axis_name})", color=col, lw=2.0)
                # Find dominant frequency
                dom_f = freqs[np.argmax(psd)]
                dom_p = np.max(psd)
                if name not in peak_info:
                    peak_info[name] = {}
                peak_info[name][axis_name] = (float(dom_f), float(dom_p))
        ax.set_ylabel(f"{axis_name}\nPSD [(m/s²)²/Hz]", fontsize=11)
        ax.set_ylim(1e-6, 1e2)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    axes[-1].set_xlabel("Frequency (Hz) [Nyquist = 5.0 Hz at 10 Hz ODR]", fontsize=11)
    fig.suptitle("Field Telemetry Vibration Signature & PSD Audit (Welch Periodogram)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()
    return peak_info

def generate_multi_test_dashboard(df_a, df_b, df_c, output_fig):
    """Generates a master 6-panel audit figure across all tests."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Panel 1: Accelerometer Norm Comparison
    ax = axes[0, 0]
    for name, d, col in [("Test A (Desk)", df_a, '#2b5c8f'), ("Test B (Idle)", df_b, '#d95f02'), ("Test C (Drive)", df_c, '#1b9e77')]:
        if d is not None and 'accel_x' in d.columns:
            t = np.arange(len(d)) * 0.1
            anorm = np.sqrt(d['accel_x']**2 + d['accel_y']**2 + d['accel_z']**2)
            ax.plot(t[:1000], anorm[:1000], label=name, color=col, alpha=0.75, lw=1.2)
    ax.axhline(9.80665, color='black', ls='--', lw=1.0, label='g (9.81 m/s²)')
    ax.set_title("1. Accelerometer Magnitude Norm Stability", fontweight='bold')
    ax.set_ylabel("||a|| (m/s²)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right')

    # Panel 2: Gyroscope Norm Comparison
    ax = axes[0, 1]
    for name, d, col in [("Test A (Desk)", df_a, '#2b5c8f'), ("Test B (Idle)", df_b, '#d95f02'), ("Test C (Drive)", df_c, '#1b9e77')]:
        if d is not None and 'gyro_x' in d.columns:
            t = np.arange(len(d)) * 0.1
            gnorm = np.rad2deg(np.sqrt(d['gyro_x']**2 + d['gyro_y']**2 + d['gyro_z']**2))
            ax.plot(t[:1000], gnorm[:1000], label=name, color=col, alpha=0.75, lw=1.2)
    ax.set_title("2. Gyroscope Angular Rate Norm (deg/s)", fontweight='bold')
    ax.set_ylabel("||ω|| (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right')

    # Panel 3: TCN Speed at Rest (A vs B)
    ax = axes[1, 0]
    for name, d, col in [("Test A (Desk)", df_a, '#2b5c8f'), ("Test B (Idle)", df_b, '#d95f02')]:
        if d is not None and 'ml_speed' in d.columns:
            ax.hist(d['ml_speed'], bins=30, alpha=0.6, label=f"{name} (mean={np.mean(d['ml_speed']):.2f})", color=col, density=True)
    ax.axvline(0.2, color='red', ls=':', lw=1.5, label='Standstill Threshold (0.2 m/s)')
    ax.set_title("4. TCN Inference Speed Distribution at True Rest", fontweight='bold')
    ax.set_xlabel("ML Speed (m/s)")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')

    # Panel 4: TCN Speed vs GNSS Speed (Test C)
    ax = axes[1, 1]
    if df_c is not None and 'ml_speed' in df_c.columns and 'gnss_speed' in df_c.columns:
        t = np.arange(len(df_c)) * 0.1
        ax.plot(t, df_c['gnss_speed'], label='GNSS Speed', color='#2b5c8f', lw=1.8)
        ax.plot(t, df_c['ml_speed'], label='TCN-kin Speed', color='#d95f02', lw=1.5, alpha=0.85)
        ax.set_title("5. Dynamic Test C: TCN-kin vs GNSS Ground Truth", fontweight='bold')
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, "Test C Speed Data Not Available", ha='center', va='center')

    # Panel 5: Sampling Jitter & Time Delta Distribution
    ax = axes[2, 0]
    for name, d, col in [("Test A", df_a, '#2b5c8f'), ("Test B", df_b, '#d95f02'), ("Test C", df_c, '#1b9e77')]:
        if d is not None and 'timestamp_ms' in d.columns:
            dts = np.diff(d['timestamp_ms'].values)
            ax.hist(dts, bins=np.arange(50, 150, 2), alpha=0.5, label=f"{name} (μ={np.mean(dts):.1f}ms)", color=col, density=True)
    ax.axvline(100.0, color='black', ls='--', lw=1.2, label='Nominal 100ms (10Hz)')
    ax.set_title("7. Sensor Sampling Delta (dt) Distribution", fontweight='bold')
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')

    # Panel 6: Navigation Altitude (Pos U) Stability
    ax = axes[2, 1]
    for name, d, col in [("Test A", df_a, '#2b5c8f'), ("Test B", df_b, '#d95f02'), ("Test C", df_c, '#1b9e77')]:
        if d is not None and 'pos_u' in d.columns:
            t = np.arange(len(d)) * 0.1
            ax.plot(t, d['pos_u'], label=f"{name} (Δ={d['pos_u'].iloc[-1]:.1f}m)", color=col, lw=1.5)
    ax.axhline(0.0, color='black', ls='--', lw=1.0)
    ax.set_title("10. Vertical Position (pos_u) Drift Integrity", fontweight='bold')
    ax.set_ylabel("pos_u (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Audit Field Telemetry Protocols A, B, and C.")
    parser.add_argument("--test-a", type=str, default=None, help="Path to Test A CSV (Stationary Desk)")
    parser.add_argument("--test-b", type=str, default=None, help="Path to Test B CSV (Vehicle Stationary Idle)")
    parser.add_argument("--test-c", type=str, default=None, help="Path to Test C CSV (Driving Test)")
    parser.add_argument("--file", type=str, default=None, help="Single file to audit")
    parser.add_argument("--out-dir", type=str, default="results/field_audit_abc", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    df_a = parse_telemetry(args.test_a) if args.test_a else None
    df_b = parse_telemetry(args.test_b) if args.test_b else None
    df_c = parse_telemetry(args.test_c) if args.test_c else None

    if args.file and not (df_a or df_b or df_c):
        df_single = parse_telemetry(args.file)
        audit_res = audit_file(df_single, label=os.path.basename(args.file))
        print(f"\nAudit completed for single file: {args.file}")
        for k, v in audit_res.items():
            print(f"  {k}: {v}")
        return

    results = {}
    if df_a is not None:
        results['Test A (Desk)'] = audit_file(df_a, "Test A (Desk)")
    if df_b is not None:
        results['Test B (Idle)'] = audit_file(df_b, "Test B (Idle)")
    if df_c is not None:
        results['Test C (Driving)'] = audit_file(df_c, "Test C (Driving)")

    # Produce figures
    fig_dash = os.path.join(args.out_dir, "master_field_audit_dashboard.png")
    generate_multi_test_dashboard(df_a, df_b, df_c, fig_dash)

    fig_psd = os.path.join(args.out_dir, "vibration_signature_psd_comparison.png")
    peak_info = analyze_vibration_psd(df_a, df_b, df_c, fig_psd)

    # Write Markdown Report
    report_path = os.path.join(args.out_dir, "field_telemetry_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Field Telemetry Audit Report: Protocols A, B, and C\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        f.write("## 1. Multi-Test Executive Summary Table\n\n")
        f.write("| Metric / Check | Test A (Desk) | Test B (Idle) | Test C (Driving) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        keys_to_show = [
            ("Epochs Recorded", "num_epochs", "{:d}"),
            ("Duration (s)", "duration_s", "{:.1f} s"),
            ("Mean Accel Norm (m/s²)", "acc_norm_mean", "{:.3f}"),
            ("Accel Norm Noise Std (m/s²)", "acc_norm_std", "{:.4f}"),
            ("Accel Error vs 1g (m/s²)", "acc_norm_err_vs_g", "{:+.4f}"),
            ("Gyro Bias X / Y / Z (deg/s)", "gyro_bias_xyz", "{:.2f}, {:.2f}, {:.2f}"),
            ("Gyro Noise Std (deg/s)", "gyro_std_xyz", "{:.3f}, {:.3f}, {:.3f}"),
            ("Mount Max Tilt Drift (deg)", "tilt_drift_max_deg", "{:.2f}°"),
            ("TCN Speed Mean (m/s)", "ml_speed_mean", "{:.3f}"),
            ("TCN Speed 95th Pct (m/s)", "ml_speed_p95", "{:.3f}"),
            ("TCN Speed % < 0.2 m/s", "ml_speed_pct_under_02", "{:.1f}%"),
            ("TCN vs GNSS Pearson r", "speed_corr_r", "{:.4f}"),
            ("TCN vs GNSS MAE (m/s)", "speed_mae", "{:.3f}"),
            ("Sampling dt Mean (ms)", "dt_mean_ms", "{:.1f} ms"),
            ("Sampling Jitter (>140ms, <60ms)", "jitter_rate_pct", "{:.2f}%"),
            ("Pos U Final Drift (m)", "pos_u_final", "{:+.2f} m"),
        ]
        
        for name, key, fmt in keys_to_show:
            row = [name]
            for test_name in ['Test A (Desk)', 'Test B (Idle)', 'Test C (Driving)']:
                t_res = results.get(test_name, {})
                if key == "gyro_bias_xyz":
                    if 'gyro_bias_deg_x' in t_res:
                        row.append(f"{t_res['gyro_bias_deg_x']:.2f}, {t_res['gyro_bias_deg_y']:.2f}, {t_res['gyro_bias_deg_z']:.2f}")
                    else:
                        row.append("—")
                elif key == "gyro_std_xyz":
                    if 'gyro_std_deg_x' in t_res:
                        row.append(f"{t_res['gyro_std_deg_x']:.2f}, {t_res['gyro_std_deg_y']:.2f}, {t_res['gyro_std_deg_z']:.2f}")
                    else:
                        row.append("—")
                elif key in t_res:
                    row.append(fmt.format(t_res[key]))
                else:
                    row.append("—")
            f.write(f"| {' | '.join(row)} |\n")
        
        f.write("\n## 2. Vibration & Spectral Peak Findings\n\n")
        f.write("```json\n" + str(peak_info) + "\n```\n\n")

    print(f"\nAudit complete! Report generated at: {report_path}")
    print(f"Dashboard saved to: {fig_dash}")
    print(f"PSD plot saved to: {fig_psd}")

if __name__ == "__main__":
    main()
