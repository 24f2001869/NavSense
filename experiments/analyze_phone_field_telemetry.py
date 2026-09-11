"""
Script: experiments/analyze_phone_field_telemetry.py
Analyzes the live smartphone telemetry log recorded on real hardware (Stage C10.2).
Audits:
- Sample rate stability & dt distribution
- Sensor health (accel norm vs 9.81, gyro dynamics, mag norm)
- Causal ML speed behavior
- Physical constraints activation (NHC, VNHC, Compass, Map)
- Filter state-machine and trajectory evolution
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_CSV = REPO_ROOT / "files" / "idr_telemetry_20260909_090715.csv"
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def analyze():
    print(f"1. Ingesting live telemetry from: {TELEMETRY_CSV}")
    df = pd.read_csv(TELEMETRY_CSV)
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # 1. Timing analysis
    t_ms = df['timestamp_ms'].values
    t_s = (t_ms - t_ms[0]) / 1000.0
    dt = np.diff(t_s)
    
    print("\n--- TIMING & SAMPLING ANALYSIS ---")
    print(f"Total duration:     {t_s[-1]:.2f} s ({t_s[-1]/60.0:.2f} min)")
    print(f"Mean dt:            {np.mean(dt)*1000:.2f} ms (target: 100.0 ms)")
    print(f"Median dt:          {np.median(dt)*1000:.2f} ms")
    print(f"Std dt:             {np.std(dt)*1000:.2f} ms")
    print(f"Min dt / Max dt:    {np.min(dt)*1000:.2f} ms / {np.max(dt)*1000:.2f} ms")
    effective_rate = len(df) / t_s[-1]
    print(f"Effective Rate:     {effective_rate:.2f} Hz")

    # 2. Sensor health
    acc = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = df[['mag_x', 'mag_y', 'mag_z']].values

    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    mag_norm = np.linalg.norm(mag, axis=1)

    print("\n--- SENSOR HEALTH & DYNAMICS ---")
    print(f"Accel Norm:         Mean = {np.mean(acc_norm):.3f} m/s², Std = {np.std(acc_norm):.3f} m/s² (1g ref = 9.81)")
    print(f"Accel Min / Max:    {np.min(acc_norm):.2f} / {np.max(acc_norm):.2f} m/s²")
    print(f"Gyro Rate Norm:     Mean = {np.degrees(np.mean(gyro_norm)):.2f}°/s, Max = {np.degrees(np.max(gyro_norm)):.2f}°/s")
    print(f"Mag Field Norm:     Mean = {np.mean(mag_norm):.2f} µT, Std = {np.std(mag_norm):.2f} µT")

    # 3. Causal ML Speed
    ml_speed = df['ml_speed'].values
    print("\n--- CAUSAL ML SPEED MODEL ---")
    print(f"ML Speed:           Mean = {np.mean(ml_speed):.2f} m/s ({np.mean(ml_speed)*3.6:.1f} km/h)")
    print(f"ML Speed Max:       {np.max(ml_speed):.2f} m/s ({np.max(ml_speed)*3.6:.1f} km/h)")
    print(f"ML Speed Min:       {np.min(ml_speed):.2f} m/s")

    # 4. Filter Constraints & State Machine
    states = df['engine_state'].value_counts().to_dict()
    nhc_active = df['nhc_enabled'].mean() * 100.0
    vnhc_active = df['vnhc_enabled'].mean() * 100.0
    mag_active = df['compass_accepted'].mean() * 100.0
    map_active = df['map_accepted'].mean() * 100.0

    print("\n--- ESKF FILTER & CONSTRAINTS ---")
    print(f"Filter States:      {states}")
    print(f"Lateral NHC Active: {nhc_active:.1f}% of epochs")
    print(f"Vertical NHC Active:{vnhc_active:.1f}% of epochs")
    print(f"Compass Accepted:   {mag_active:.1f}% of epochs")
    print(f"Map Accepted:       {map_active:.1f}% of epochs")

    # 5. Visual Diagnostics Plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"Real Smartphone Field Telemetry Audit (Stage C10.2)\nDuration: {t_s[-1]:.1f} s ({len(df)} epochs @ {effective_rate:.1f} Hz)", fontsize=14, fontweight='bold')

    # (0, 0) Accelerometer
    ax = axes[0, 0]
    ax.plot(t_s, acc[:, 0], label='a_x', alpha=0.7)
    ax.plot(t_s, acc[:, 1], label='a_y', alpha=0.7)
    ax.plot(t_s, acc[:, 2], label='a_z', alpha=0.7)
    ax.plot(t_s, acc_norm, 'k--', label='||a||', alpha=0.5)
    ax.set_ylabel('Accel (m/s²)')
    ax.set_title('Triaxial Accelerometer')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0, 1) Gyroscope
    ax = axes[0, 1]
    ax.plot(t_s, np.degrees(gyro[:, 0]), label='ω_x', alpha=0.7)
    ax.plot(t_s, np.degrees(gyro[:, 1]), label='ω_y', alpha=0.7)
    ax.plot(t_s, np.degrees(gyro[:, 2]), label='ω_z (yaw)', alpha=0.8, color='crimson')
    ax.set_ylabel('Gyro (°/s)')
    ax.set_title('Triaxial Gyroscope')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1, 0) Magnetometer
    ax = axes[1, 0]
    ax.plot(t_s, mag[:, 0], label='B_x', alpha=0.7)
    ax.plot(t_s, mag[:, 1], label='B_y', alpha=0.7)
    ax.plot(t_s, mag[:, 2], label='B_z', alpha=0.7)
    ax.plot(t_s, mag_norm, 'k--', label='||B||', alpha=0.5)
    ax.set_ylabel('Mag (µT)')
    ax.set_title('Triaxial Magnetometer')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1, 1) Causal ML Speed
    ax = axes[1, 1]
    ax.plot(t_s, ml_speed * 3.6, color='royalblue', lw=1.5, label='ML Speed (km/h)')
    ax.plot(t_s, ml_speed, color='deepskyblue', ls='--', lw=1.0, label='ML Speed (m/s)')
    ax.set_ylabel('Speed')
    ax.set_title('Causal Random Forest Speed')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2, 0) Filter Heading
    ax = axes[2, 0]
    ax.plot(t_s, df['heading_deg'].values, color='forestgreen', lw=1.5, label='Heading (°)')
    ax.set_ylabel('Heading (°)')
    ax.set_title('ESKF Estimated Heading')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2, 1) Constraints Timeline
    ax = axes[2, 1]
    ax.plot(t_s, df['nhc_enabled'].astype(int), label='Lateral NHC', lw=1.2, color='teal')
    ax.plot(t_s, df['vnhc_enabled'].astype(int) * 0.8, label='Vertical NHC', lw=1.2, color='purple')
    ax.plot(t_s, df['compass_accepted'].astype(int) * 0.6, label='Compass Gate', lw=1.2, color='coral')
    ax.set_ylabel('Active (binary)')
    ax.set_yticks([0, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['Off', 'Compass', 'VNHC', 'NHC'])
    ax.set_title('Physical Constraints Activation')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3, 0) dt distribution
    ax = axes[3, 0]
    ax.plot(t_s[1:], dt * 1000.0, color='darkorange', lw=1.0)
    ax.axhline(100.0, color='red', ls='--', alpha=0.7, label='100 ms target (10 Hz)')
    ax.set_ylabel('Interval dt (ms)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Loop Sampling Interval Stability')
    ax.set_ylim([50, 200])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3, 1) 2D Estimated Trajectory (ENU)
    ax = axes[3, 1]
    e = df['pos_e'].values
    n = df['pos_n'].values
    ax.plot(e - e[0], n - n[0], color='crimson', lw=1.5, label='Estimated Track')
    ax.scatter(0, 0, color='green', marker='o', s=80, label='Start')
    ax.scatter(e[-1] - e[0], n[-1] - n[0], color='red', marker='x', s=80, label='End')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Estimated Relative Trajectory (ENU)')
    ax.axis('equal')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RES_DIR / "c10_2_field_telemetry_audit.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"\nSaved diagnostic figure to: {out_png}")


if __name__ == '__main__':
    analyze()
