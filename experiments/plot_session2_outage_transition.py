"""
Script: experiments/plot_session2_outage_transition.py
Visualizes the complete real-device GNSS -> Dead Reckoning -> Reacquiring -> GNSS transition (Test 2).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_CSV = REPO_ROOT / "files" / "idr_telemetry_20260909_091947.csv"
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

def plot_session2():
    df = pd.read_csv(TELEMETRY_CSV)
    t0 = df['timestamp_ms'].iloc[0]
    t_s = (df['timestamp_ms'] - t0) / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Real Smartphone Live GNSS Outage & Reacquisition Transition (Stage C10.2 Test 2)\nDevice: CPH2381 | Total Duration: {t_s.iloc[-1]:.1f} s ({len(df)} epochs)", fontsize=13, fontweight='bold')

    # State regions
    states = df['engine_state'].values
    dr_mask = (states == 'DEAD_RECKONING')
    reacq_mask = (states == 'REACQUIRING')
    lock_mask = (states == 'GNSS_LOCKED')

    outage_t_start = t_s[dr_mask].iloc[0]
    outage_t_end = t_s[dr_mask].iloc[-1]

    # Subplot 1: Engine State
    ax = axes[0]
    state_numeric = np.zeros(len(df))
    state_numeric[lock_mask] = 2.0
    state_numeric[reacq_mask] = 1.0
    state_numeric[dr_mask] = 0.0

    ax.plot(t_s, state_numeric, color='crimson', lw=2.0, drawstyle='steps-post')
    ax.axvspan(outage_t_start, outage_t_end, color='red', alpha=0.15, label=f'44.1s Outage (Epoch 275-655)')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['DEAD_RECKONING', 'REACQUIRING', 'GNSS_LOCKED'], fontweight='bold')
    ax.set_ylabel('Engine State')
    ax.set_title('1. ESKF Navigation State Machine Transitions')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Subplot 2: Causal ML Speed & GNSS Speed
    ax = axes[1]
    ax.plot(t_s, df['ml_speed'] * 3.6, color='royalblue', lw=1.5, label='Causal ML Speed (km/h)')
    ax.plot(t_s, df['gnss_speed'] * 3.6, color='seagreen', ls='--', lw=1.2, label='GNSS Speed (km/h)')
    ax.axvspan(outage_t_start, outage_t_end, color='red', alpha=0.15)
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('2. Causal ML Longitudinal Speed vs Live GNSS Speed')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Subplot 3: Heading Evolution & Compass
    ax = axes[2]
    ax.plot(t_s, df['heading_deg'], color='darkorange', lw=1.5, label='Filter Heading (deg)')
    ax.axvspan(outage_t_start, outage_t_end, color='red', alpha=0.15)
    ax.set_ylabel('Heading (°)')
    ax.set_title('3. Estimated Vehicle Heading Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Subplot 4: Relative ENU Position Evolution
    ax = axes[3]
    ax.plot(t_s, df['pos_e'], label='East (m)', color='blue', alpha=0.8)
    ax.plot(t_s, df['pos_n'], label='North (m)', color='green', alpha=0.8)
    ax.plot(t_s, df['pos_u'], label='Up (m)', color='purple', alpha=0.8)
    ax.axvspan(outage_t_start, outage_t_end, color='red', alpha=0.15)
    ax.set_ylabel('Local ENU (m)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('4. Local Cartesian Coordinates (ENU)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out_png = RES_DIR / "c10_2_outage_transition_audit.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved transition diagnostic figure to: {out_png}")

if __name__ == '__main__':
    plot_session2()
