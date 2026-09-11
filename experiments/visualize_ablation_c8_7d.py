"""
SIH26168 - Stage C8-7D: Component Attribution & Ablation Visualization Dashboard
Module: experiments/visualize_ablation_c8_7d.py

Generates a publication-grade 9-panel diagnostic dashboard analyzing the exact
attribution and ablation findings from Stage C8-7D.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_c8_7d_dashboard():
    json_path = RES_DIR / "c8_7d_ablation_audit.json"
    if not json_path.exists():
        print(f"Results file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    trips = data['trips']
    v4 = trips['Vta04']
    v2 = trips['Vta02']

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.patch.set_facecolor('#0f172a')

    for row in axes:
        for ax in row:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8', labelsize=9)
            ax.xaxis.label.set_color('#cbd5e1')
            ax.yaxis.label.set_color('#cbd5e1')
            ax.title.set_color('#f8fafc')
            for spine in ax.spines.values():
                spine.set_color('#334155')

    colors_map = {
        'A0_Baseline_PureIMU': '#ef4444',
        'B1_Speed_Only': '#f59e0b',
        'B2_Compass_Only': '#8b5cf6',
        'B3_NHC_Only': '#ec4899',
        'B4_ZUPT_Only': '#64748b',
        'C1_Speed_Compass': '#3b82f6',
        'C2_Speed_Compass_NHC': '#06b6d4',
        'C3_Speed_Compass_NHC_ZUPT': '#10b981',
        'C4_Full_Smartphone_Map': '#22c55e',
        'REF_CAN_Wheel_Full': '#a855f7'
    }

    # =========================================================================
    # Panel 1: Step-Up Waterfall at 10s (Vta04)
    # =========================================================================
    ax = axes[0, 0]
    step_keys = ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'C1_Speed_Compass', 'C2_Speed_Compass_NHC', 'C3_Speed_Compass_NHC_ZUPT', 'C4_Full_Smartphone_Map', 'REF_CAN_Wheel_Full']
    step_labels = ['Pure IMU', '+Speed (1D)', '+Compass', '+NHC', '+ZUPT', '+Map (Full)', 'CAN Ref']
    errs_10s = [v4['10s'][k]['mean_pos_err_m'] for k in step_keys]
    c_list = [colors_map.get(k, '#38bdf8') for k in step_keys]

    bars = ax.bar(step_labels, errs_10s, color=c_list, edgecolor='#f8fafc', alpha=0.85, width=0.6)
    ax.axhline(10.4, color='#f43f5e', linestyle='--', linewidth=1.5, label='SIH 10% Target (~10.4m)')
    ax.set_title('1. Cumulative Step-Up: 10s Blackout (Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.tick_params(axis='x', rotation=25)
    for b, v in zip(bars, errs_10s):
        ax.text(b.get_x() + b.get_width()/2, v + 1.0, f'{v:.1f}m', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 2: Step-Up Waterfall at 20s (Vta04)
    # =========================================================================
    ax = axes[0, 1]
    errs_20s = [v4['20s'][k]['mean_pos_err_m'] for k in step_keys]
    bars = ax.bar(step_labels, errs_20s, color=c_list, edgecolor='#f8fafc', alpha=0.85, width=0.6)
    ax.axhline(21.0, color='#f43f5e', linestyle='--', linewidth=1.5, label='SIH 10% Target (~21m)')
    ax.set_title('2. Cumulative Step-Up: 20s Blackout (Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.tick_params(axis='x', rotation=25)
    for b, v in zip(bars, errs_20s):
        ax.text(b.get_x() + b.get_width()/2, v + 3.0, f'{v:.1f}m', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 3: Step-Up Waterfall at 30s (Vta04)
    # =========================================================================
    ax = axes[0, 2]
    errs_30s = [v4['30s'][k]['mean_pos_err_m'] for k in step_keys]
    bars = ax.bar(step_labels, errs_30s, color=c_list, edgecolor='#f8fafc', alpha=0.85, width=0.6)
    ax.axhline(31.5, color='#f43f5e', linestyle='--', linewidth=1.5, label='SIH 10% Target (~31.5m)')
    ax.set_title('3. Cumulative Step-Up: 30s Blackout (Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.tick_params(axis='x', rotation=25)
    for b, v in zip(bars, errs_30s):
        ax.text(b.get_x() + b.get_width()/2, v + 5.0, f'{v:.1f}m', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 4: Single Component Isolation from Baseline (20s Vta04)
    # =========================================================================
    ax = axes[1, 0]
    single_keys = ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'B2_Compass_Only', 'B3_NHC_Only', 'B4_ZUPT_Only']
    single_labels = ['Pure IMU', 'Speed Alone', 'Compass Alone', 'NHC Alone', 'ZUPT Alone']
    s_errs = [v4['20s'][k]['mean_pos_err_m'] for k in single_keys]
    s_colors = [colors_map.get(k, '#94a3b8') for k in single_keys]

    bars = ax.bar(single_labels, s_errs, color=s_colors, edgecolor='#f8fafc', alpha=0.85, width=0.55)
    ax.set_title('4. Isolated Component Value from Pure IMU (20s Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.tick_params(axis='x', rotation=20)
    for b, v in zip(bars, s_errs):
        ax.text(b.get_x() + b.get_width()/2, v + 3.0, f'{v:.1f}m', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')

    # =========================================================================
    # Panel 5: Leave-One-Out Sensitivity Analysis from Full Stack (20s Vta04)
    # =========================================================================
    ax = axes[1, 1]
    loo_keys = ['C4_Full_Smartphone_Map', 'D1_Ablate_Minus_Speed', 'D2_Ablate_Minus_Compass', 'D3_Ablate_Minus_NHC', 'D4_Ablate_Minus_ZUPT', 'D5_Ablate_Minus_Map']
    loo_labels = ['Full Stack', '- Speed', '- Compass', '- NHC', '- ZUPT', '- Map']
    loo_errs = [v4['20s'][k]['mean_pos_err_m'] for k in loo_keys]
    loo_colors = ['#22c55e', '#ef4444', '#f59e0b', '#ec4899', '#64748b', '#06b6d4']

    bars = ax.bar(loo_labels, loo_errs, color=loo_colors, edgecolor='#f8fafc', alpha=0.85, width=0.55)
    base_full = v4['20s']['C4_Full_Smartphone_Map']['mean_pos_err_m']
    ax.axhline(base_full, color='#22c55e', linestyle=':', linewidth=1.5, label=f'Full Stack Baseline ({base_full:.1f}m)')
    ax.set_title('5. Leave-One-Out Ablation (Damage when Removed at 20s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.tick_params(axis='x', rotation=20)
    for b, v in zip(bars, loo_errs):
        diff = v - base_full
        sign = f'+{diff:.1f}m' if diff > 0 else f'{diff:.1f}m'
        ax.text(b.get_x() + b.get_width()/2, v + 2.0, f'{v:.1f}m\n({sign})', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 6: Along-Track vs Cross-Track Decomposition (20s Vta04)
    # =========================================================================
    ax = axes[1, 2]
    decomp_keys = ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'B2_Compass_Only', 'B3_NHC_Only', 'C4_Full_Smartphone_Map']
    decomp_labels = ['Pure IMU', '+Speed', '+Compass', '+NHC', 'Full Stack']
    along_vals = [v4['20s'][k]['mean_along_err_m'] for k in decomp_keys]
    cross_vals = [v4['20s'][k]['mean_cross_err_m'] for k in decomp_keys]

    x = np.arange(len(decomp_labels))
    w = 0.35
    b1 = ax.bar(x - w/2, along_vals, width=w, label='Along-Track Error (m)', color='#38bdf8', edgecolor='#f8fafc')
    b2 = ax.bar(x + w/2, cross_vals, width=w, label='Cross-Track Error (m)', color='#f43f5e', edgecolor='#f8fafc')
    ax.set_xticks(x)
    ax.set_xticklabels(decomp_labels, rotation=20)
    ax.set_title('6. Error Orthogonal Decomposition (20s Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error Magnitude (m)')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 7: Drift % Scaling vs Blackout Horizon (Vta04)
    # =========================================================================
    ax = axes[2, 0]
    horizons = [5, 10, 20, 30]
    h_labels = ['5s', '10s', '20s', '30s']

    track_conds = [
        ('A0_Baseline_PureIMU', 'Pure IMU', '#ef4444', 'o--'),
        ('B1_Speed_Only', '+Speed Only', '#f59e0b', 's--'),
        ('C1_Speed_Compass', '+Speed +Compass', '#3b82f6', '^--'),
        ('C4_Full_Smartphone_Map', 'Full Stack', '#22c55e', 'D-'),
        ('REF_CAN_Wheel_Full', 'CAN Ref', '#a855f7', '*:')
    ]

    for c_key, c_lbl, c_col, c_mk in track_conds:
        d_vals = [v4[f'{h}s'][c_key]['mean_drift_pct'] for h in horizons]
        ax.plot(h_labels, d_vals, c_mk, color=c_col, label=c_lbl, linewidth=2.0, markersize=7)

    ax.axhline(10.0, color='#f43f5e', linestyle='--', linewidth=2.0, label='SIH Target (<10%)')
    ax.set_title('7. Drift % vs Blackout Duration (Vta04)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drift Percentage (%)')
    ax.set_xlabel('Blackout Horizon')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 8: Cross-Trip Comparison at 20s (Suburban Vta02 vs Highway Vta04)
    # =========================================================================
    ax = axes[2, 1]
    comp_keys = ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'C4_Full_Smartphone_Map', 'REF_CAN_Wheel_Full']
    comp_labels = ['Pure IMU', '+Speed', 'Full Stack', 'CAN Ref']
    v4_comp = [v4['20s'][k]['mean_pos_err_m'] for k in comp_keys]
    v2_comp = [v2['20s'][k]['mean_pos_err_m'] for k in comp_keys]

    x = np.arange(len(comp_labels))
    w = 0.35
    ax.bar(x - w/2, v4_comp, width=w, label='Vta04 (Highway)', color='#38bdf8', edgecolor='#f8fafc')
    ax.bar(x + w/2, v2_comp, width=w, label='Vta02 (Suburban)', color='#fbbf24', edgecolor='#f8fafc')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, rotation=20)
    ax.set_title('8. Cross-Trip Comparison at 20s Horizon', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

    # =========================================================================
    # Panel 9: 60s Horizon Forensic Breakdown (Vta04 N=1 vs Vta02 N=17)
    # =========================================================================
    ax = axes[2, 2]
    # Show Vta02 at 60s
    v2_60_keys = ['A0_Baseline_PureIMU', 'B1_Speed_Only', 'C4_Full_Smartphone_Map', 'REF_CAN_Wheel_Full']
    v2_60_labels = ['Pure IMU\n(Vta02)', '+Speed\n(Vta02)', 'Full Stack\n(Vta02)', 'CAN Ref\n(Vta02)']
    v2_60_errs = [v2['60s'][k]['mean_pos_err_m'] for k in v2_60_keys]
    v2_60_colors = ['#ef4444', '#f59e0b', '#22c55e', '#a855f7']

    bars = ax.bar(v2_60_labels, v2_60_errs, color=v2_60_colors, edgecolor='#f8fafc', alpha=0.85, width=0.55)
    ax.set_title('9. 60s Horizon on Suburban Trip (Vta02, N=17 Windows)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean 2D Position Error (m)')
    for b, v in zip(bars, v2_60_errs):
        ax.text(b.get_x() + b.get_width()/2, v + 25.0, f'{v:.0f}m', ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_img = FIG_DIR / "c8_7d_ablation_audit.png"
    plt.savefig(out_img, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved 9-panel ablation dashboard to: {out_img}")


if __name__ == '__main__':
    plot_c8_7d_dashboard()
