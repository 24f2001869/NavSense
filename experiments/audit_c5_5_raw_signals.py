"""
SIH26168 - Forensic Audit: Raw Signal Sanity Check for C5.5

Script: experiments/audit_c5_5_raw_signals.py

PURPOSE:
    Conduct a rigorous sanity check on the raw, unprocessed data to answer the user's 8 questions:
    1. Show individual braking events for Vta02, Vta03, and Vta04.
    2. Plot phone acceleration and CAN acceleration on the same time axis.
    3. Verify the timestamp synchronization used (check author's synchronization vs loader vs raw stamps).
    4. Verify the phone-to-vehicle axis/sign transformation (how ax_level is computed from raw accel_x, accel_y, accel_z).
    5. Verify whether the claimed 'rebound' exists in original raw phone accelerometer data or is an artifact of transformation/filtering.
    6. Compare raw phone acceleration against CAN and VBOX without frequency-domain processing.
    7. Report exactly what is directly observed vs what is inferred.
    8. Check vehicle CAN brake pressure/pedal channels during these events.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip, find_trip_dir
from src.preprocessing.gravity_alignment import compute_leveling_matrix

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = REPO_ROOT / "data" / "processed"


def inspect_raw_csv_headers_and_timestamps(trip_name: str):
    """Examines raw S-*.csv and V-*.csv directly from disk."""
    trip_dir = find_trip_dir(trip_name)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    
    df_s_raw = pd.read_csv(s_path, encoding='latin1', nrows=10)
    df_v_raw = pd.read_csv(v_path, encoding='latin1', nrows=10)
    
    print(f"\n--- RAW FILE AUDIT: {trip_name} ---")
    print(f"S-File: {s_path.name} ({s_path.stat().st_size} bytes)")
    print(f"V-File: {v_path.name} ({v_path.stat().st_size} bytes)")
    
    print("\nRaw Phone Columns:", list(df_s_raw.columns[:8]))
    print("Raw Vehicle Columns:", list(df_v_raw.columns[:8]))
    
    # Check time columns
    print("\nFirst 5 Phone Time entries:")
    for col in df_s_raw.columns:
        if any(t in col.upper() for t in ['TIME', 'DATE', 'STAMP']):
            print(f"  {col}: {df_s_raw[col].iloc[:5].tolist()}")
            
    print("\nFirst 5 Vehicle Time entries:")
    for col in df_v_raw.columns:
        if any(t in col.upper() for t in ['TIME', 'DATE', 'SAMPLE', 'STAMP']):
            print(f"  {col}: {df_v_raw[col].iloc[:5].tolist()}")


def audit_trip_braking_events(trip_name: str):
    """Audits raw signals during all braking events for a trip."""
    trip_dir = find_trip_dir(trip_name)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    
    df_s_raw = pd.read_csv(s_path, encoding='latin1')
    df_v_raw = pd.read_csv(v_path, encoding='latin1')
    df_p, df_v = load_trip(trip_name)
    labels = pd.read_csv(PROC_DIR / f"c5_3_labels_{trip_name.lower()}.csv")
    
    n = min(len(df_p), len(df_v), len(labels))
    df_p = df_p.iloc[:n].reset_index(drop=True)
    df_v = df_v.iloc[:n].reset_index(drop=True)
    labels = labels.iloc[:n].reset_index(drop=True)
    df_s_raw = df_s_raw.iloc[:n].reset_index(drop=True)
    df_v_raw = df_v_raw.iloc[:n].reset_index(drop=True)
    
    time_s = labels['time_s'].to_numpy()
    a_ref = labels['LABEL_vbox_ref_accel_ms2'].to_numpy()
    v_vbox = labels['LABEL_vbox_speed_ms'].to_numpy()
    ax_level = labels['ax_level'].to_numpy()
    
    # Raw phone accelerations in sensor body frame
    raw_ax = df_p['accel_x'].to_numpy()
    raw_ay = df_p['accel_y'].to_numpy()
    raw_az = df_p['accel_z'].to_numpy()
    
    # Vehicle CAN channels
    can_accel = df_v['veh_accel_long_ms2'].to_numpy() if 'veh_accel_long_ms2' in df_v.columns else np.zeros(n)
    brake_press = df_v['brake_pressure_psi'].to_numpy() if 'brake_pressure_psi' in df_v.columns else np.zeros(n)
    brake_pos = df_v['brake_pos'].to_numpy() if 'brake_pos' in df_v.columns else np.zeros(n)
    throttle = df_v['throttle_pos'].to_numpy() if 'throttle_pos' in df_v.columns else np.zeros(n)
    
    # Identify braking events: where a_ref <= -1.2 m/s² for at least 3 consecutive samples (0.3s)
    is_brake = a_ref <= -1.2
    diff = np.diff(is_brake.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    
    # Group into contiguous events
    events = []
    for ons in onsets:
        # scan forward until brake ends
        offs = ons
        while offs < n and a_ref[offs] <= -0.5:
            offs += 1
        duration_s = (offs - ons) * 0.1
        min_decel = np.min(a_ref[ons:offs])
        
        # Check if already covered
        if events and ons < events[-1]['end_idx'] + 10:
            continue
            
        if duration_s >= 0.5 and min_decel <= -1.5:
            events.append({
                'start_idx': ons,
                'end_idx': offs,
                'start_t': time_s[ons],
                'end_t': time_s[min(offs, n-1)],
                'duration_s': duration_s,
                'min_decel_ref': min_decel
            })
            
    print(f"\n===================================================================")
    print(f"TRIP {trip_name}: Identified {len(events)} Significant Braking Events (a_ref <= -1.5 m/s²)")
    print(f"===================================================================")
    for i, ev in enumerate(events):
        print(f"Event {i+1}: t = [{ev['start_t']:.1f}s - {ev['end_t']:.1f}s] (dur: {ev['duration_s']:.1f}s, min_a_ref: {ev['min_decel_ref']:.2f} m/s²)")
        
    return {
        'trip_name': trip_name,
        'time_s': time_s,
        'a_ref': a_ref,
        'v_vbox': v_vbox,
        'raw_ax': raw_ax,
        'raw_ay': raw_ay,
        'raw_az': raw_az,
        'ax_level': ax_level,
        'can_accel': can_accel,
        'brake_press': brake_press,
        'brake_pos': brake_pos,
        'throttle': throttle,
        'events': events,
        'df_s_raw': df_s_raw,
        'df_v_raw': df_v_raw
    }


def main():
    print("STARTING RAW SIGNAL FORENSIC SANITY CHECK")
    
    for t in ["Vta02", "Vta03", "Vta04"]:
        inspect_raw_csv_headers_and_timestamps(t)
        
    data = {}
    for t in ["Vta02", "Vta03", "Vta04"]:
        data[t] = audit_trip_braking_events(t)
        
    # -----------------------------------------------------------------------
    # PLOT DETAILED INDIVIDUAL BRAKING EVENTS
    # -----------------------------------------------------------------------
    # We want to plot 4 prominent events:
    # Event 1 & 2 from Vta02, Event 1 from Vta03, Event 1 & 2 from Vta04
    selected_plots = [
        ("Vta02", 0, "Vta02 Event 1 (Stop 1)"),
        ("Vta02", 1, "Vta02 Event 2 (Stop 2)"),
        ("Vta03", 0, "Vta03 Event 1 (Only Major Stop)"),
        ("Vta04", 0, "Vta04 Event 1 (Highway Decel 1)"),
        ("Vta04", 1, "Vta04 Event 2 (Highway Decel 2)"),
    ]
    
    fig, axes = plt.subplots(len(selected_plots), 1, figsize=(12, 16), sharex=False)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.4})
    
    audit_summary = []
    
    for idx, (trip_name, ev_idx, title) in enumerate(selected_plots):
        td = data[trip_name]
        ax = axes[idx]
        
        if ev_idx >= len(td['events']):
            ax.text(0.5, 0.5, f"No event {ev_idx} found in {trip_name}", ha='center')
            continue
            
        ev = td['events'][ev_idx]
        # Window: 3s before start to 5s after end
        t_start = max(0, ev['start_t'] - 3.0)
        t_end = min(td['time_s'][-1], ev['end_t'] + 5.0)
        mask = (td['time_s'] >= t_start) & (td['time_s'] <= t_end)
        
        t_sub = td['time_s'][mask]
        a_ref_sub = td['a_ref'][mask]
        can_sub = td['can_accel'][mask]
        ax_level_sub = td['ax_level'][mask]
        raw_ax_sub = td['raw_ax'][mask]
        raw_ay_sub = td['raw_ay'][mask]
        raw_az_sub = td['raw_az'][mask]
        v_sub = td['v_vbox'][mask]
        press_sub = td['brake_press'][mask]
        
        # Plot acceleration channels
        ax.plot(t_sub, a_ref_sub, color='black', lw=2.5, label='VBOX Ref a_ref')
        ax.plot(t_sub, can_sub, color='forestgreen', lw=2.0, linestyle='-', label='Chassis CAN Accel')
        ax.plot(t_sub, ax_level_sub, color='crimson', lw=2.0, label='Phone Leveled ax_level')
        ax.plot(t_sub, raw_ax_sub, color='orange', lw=1.2, linestyle=':', label='Raw Phone accel_x')
        
        # Add secondary axis for speed and brake pressure
        ax2 = ax.twinx()
        ax2.plot(t_sub, v_sub, color='blue', lw=1.2, linestyle='--', alpha=0.6, label='VBOX Speed (m/s)')
        if np.max(press_sub) > 0:
            ax2.plot(t_sub, press_sub / 100.0, color='purple', lw=1.0, linestyle='-.', alpha=0.5, label='Brake Press (x100 psi)')
        ax2.set_ylabel("Speed (m/s)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(False)
        
        ax.axvspan(ev['start_t'], ev['end_t'], color='yellow', alpha=0.2, label='Braking Window')
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel("Accel (m/s²)")
        ax.set_title(f"{title}: t=[{t_start:.1f}s, {t_end:.1f}s] | Duration={ev['duration_s']:.1f}s | Min a_ref={ev['min_decel_ref']:.2f} m/s²")
        
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=3)
            
        # Inspect the post-braking rebound in this exact window
        post_mask = (t_sub >= ev['end_t']) & (t_sub <= ev['end_t'] + 2.5)
        rebound_phone = float(np.max(ax_level_sub[post_mask])) if np.sum(post_mask) > 0 else 0.0
        rebound_raw_ax = float(np.max(raw_ax_sub[post_mask])) if np.sum(post_mask) > 0 else 0.0
        rebound_can = float(np.max(can_sub[post_mask])) if np.sum(post_mask) > 0 else 0.0
        rebound_ref = float(np.max(a_ref_sub[post_mask])) if np.sum(post_mask) > 0 else 0.0
        
        # Deceleration peak in braking window
        brake_sub_mask = (t_sub >= ev['start_t']) & (t_sub <= ev['end_t'])
        peak_ref = float(np.min(a_ref_sub[brake_sub_mask]))
        peak_can = float(np.min(can_sub[brake_sub_mask]))
        peak_phone = float(np.min(ax_level_sub[brake_sub_mask]))
        peak_raw_ax = float(np.min(raw_ax_sub[brake_sub_mask]))
        
        audit_summary.append({
            'event_name': title,
            't_start': ev['start_t'],
            't_end': ev['end_t'],
            'peak_ref_ms2': peak_ref,
            'peak_can_ms2': peak_can,
            'peak_phone_level_ms2': peak_phone,
            'peak_phone_raw_ax_ms2': peak_raw_ax,
            'rebound_phone_level_ms2': rebound_phone,
            'rebound_phone_raw_ax_ms2': rebound_raw_ax,
            'rebound_can_ms2': rebound_can,
            'rebound_ref_ms2': rebound_ref
        })
        
    axes[-1].set_xlabel("Elapsed Time (s)")
    plt.tight_layout()
    fig_path = FIG_DIR / "c5_5_raw_signal_sanity_check.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved Comprehensive Raw Signal Plot: {fig_path}")
    
    # Save json summary
    json_path = RES_DIR / "c5_5_raw_signal_audit.json"
    with open(json_path, 'w') as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved JSON Audit: {json_path}")


if __name__ == "__main__":
    main()
