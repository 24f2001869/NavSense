"""
SIH26168 - Stage C8-11.8: Plot End-to-End Validation Figures
Generates publication-quality diagnostic visualizations for C8-11.8 report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_PATH = REPO_ROOT / "results" / "c8_11_8_end_to_end_validation.json"
FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_MEDIA = PROJECT_ROOT / "results" / "figures"

with open(RES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13
})

COLORS = {
    'A0_Pure_IMU': '#d62728',
    'A1_IMU_NHC': '#ff7f0e',
    'A2_IMU_ML_Speed_NHC': '#2ca02c',
    'A3_Calibrated_Compass_Static': '#9467bd',
    'A4_Fallback_Quality_Gated': '#8c564b',
    'A4_Frozen_Quality_Gated': '#1f77b4',
    'A5_Full_Stack': '#17becf',
    'Oracle_Heading': '#bcbd22',
    'Oracle_Speed': '#e377c2',
    'Oracle_Heading_Speed': '#7f7f7f',
    'REF_CAN_Wheel_Full': '#111111'
}

LABELS = {
    'A0_Pure_IMU': 'A0: Pure IMU',
    'A1_IMU_NHC': 'A1: IMU + NHC',
    'A2_IMU_ML_Speed_NHC': 'A2: IMU + ML Speed + NHC',
    'A3_Calibrated_Compass_Static': 'A3: + Static Compass',
    'A4_Fallback_Quality_Gated': 'A4: Quality-Gated Fallback',
    'A4_Frozen_Quality_Gated': 'A4: Quality-Gated Rolling',
    'A5_Full_Stack': 'A5: Full Stack + ZUPT + Map',
    'Oracle_Heading': 'Oracle Heading',
    'Oracle_Speed': 'Oracle Speed',
    'Oracle_Heading_Speed': 'Oracle Heading + Speed',
    'REF_CAN_Wheel_Full': 'CAN Wheel Speed Ref'
}

# -----------------------------------------------------------------------------
# Figure 1: Horizon vs Drift % Comparison (Vta04 vs Vta02)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Vta04
ax = axes[0]
horizons_v04 = [5, 10, 20, 30, 60]
for c in ['A0_Pure_IMU', 'A1_IMU_NHC', 'A2_IMU_ML_Speed_NHC', 'A4_Frozen_Quality_Gated', 'A5_Full_Stack', 'Oracle_Heading', 'REF_CAN_Wheel_Full']:
    drifts = []
    for h in horizons_v04:
        h_str = f"{h}s"
        m = data['trips']['Vta04'][h_str]['conditions'].get(c, {})
        drifts.append(m.get('mean_drift_pct', np.nan))
    ls = '--' if 'Oracle' in c else ('-.' if 'REF' in c else '-')
    lw = 2.5 if c in ['A4_Frozen_Quality_Gated', 'Oracle_Heading'] else 1.5
    ax.plot(horizons_v04, drifts, label=LABELS[c], color=COLORS[c], linestyle=ls, linewidth=lw, marker='o', markersize=4)

ax.axhline(10.0, color='red', linestyle=':', linewidth=2.0, label='SIH Benchmark (<10%)')
ax.set_title("Vta04 (Urban Loop, 1.99 km)")
ax.set_xlabel("Outage Horizon (seconds)")
ax.set_ylabel("Mean Positional Drift (% of dist traveled)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

# Vta02
ax = axes[1]
horizons_v02 = [5, 10, 20, 30, 60, 90, 120]
for c in ['A0_Pure_IMU', 'A1_IMU_NHC', 'A2_IMU_ML_Speed_NHC', 'A4_Frozen_Quality_Gated', 'A5_Full_Stack', 'Oracle_Heading', 'REF_CAN_Wheel_Full']:
    drifts = []
    for h in horizons_v02:
        h_str = f"{h}s"
        m = data['trips']['Vta02'][h_str]['conditions'].get(c, {})
        drifts.append(m.get('mean_drift_pct', np.nan))
    ls = '--' if 'Oracle' in c else ('-.' if 'REF' in c else '-')
    lw = 2.5 if c in ['A4_Frozen_Quality_Gated', 'Oracle_Heading'] else 1.5
    ax.plot(horizons_v02, drifts, label=LABELS[c], color=COLORS[c], linestyle=ls, linewidth=lw, marker='s', markersize=4)

ax.axhline(10.0, color='red', linestyle=':', linewidth=2.0, label='SIH Benchmark (<10%)')
ax.set_title("Vta02 (Highway/Arterial, 11.05 km)")
ax.set_xlabel("Outage Horizon (seconds)")
ax.set_ylabel("Mean Positional Drift (% of dist traveled)")
ax.set_ylim(0, 200)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

plt.suptitle("Stage C8-11.8: End-to-End Dead-Reckoning Drift % vs Outage Horizon", fontsize=14, y=0.98)
plt.tight_layout()
fig1_path = FIG_DIR / "c8_11_8_horizon_drift_comparison.png"
fig.savefig(fig1_path, dpi=200)
fig.savefig(BRAIN_MEDIA / "c8_11_8_horizon_drift_comparison.png", dpi=200)
plt.close(fig)
print(f"Saved: {fig1_path}")

# -----------------------------------------------------------------------------
# Figure 2: Along-Track vs Cross-Track Error Decomposition (30s Outage)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

conds_to_show = [
    'A0_Pure_IMU', 'A1_IMU_NHC', 'A2_IMU_ML_Speed_NHC',
    'A4_Frozen_Quality_Gated', 'A5_Full_Stack',
    'Oracle_Heading', 'REF_CAN_Wheel_Full'
]

# Vta04 30s
ax = axes[0]
x = np.arange(len(conds_to_show))
along_v04 = [data['trips']['Vta04']['30s']['conditions'][c]['mean_along_err_m'] for c in conds_to_show]
cross_v04 = [data['trips']['Vta04']['30s']['conditions'][c]['mean_cross_err_m'] for c in conds_to_show]
width = 0.38
ax.bar(x - width/2, along_v04, width, label='Along-Track Error (m)', color='#3498db', alpha=0.85)
ax.bar(x + width/2, cross_v04, width, label='Cross-Track Error (m)', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in conds_to_show], rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Error Magnitude (meters)")
ax.set_title("Vta04 (Urban 30s Outage, N=3)")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend()

# Vta02 30s
ax = axes[1]
x = np.arange(len(conds_to_show))
along_v02 = [data['trips']['Vta02']['30s']['conditions'][c]['mean_along_err_m'] for c in conds_to_show]
cross_v02 = [data['trips']['Vta02']['30s']['conditions'][c]['mean_cross_err_m'] for c in conds_to_show]
ax.bar(x - width/2, along_v02, width, label='Along-Track Error (m)', color='#3498db', alpha=0.85)
ax.bar(x + width/2, cross_v02, width, label='Cross-Track Error (m)', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in conds_to_show], rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Error Magnitude (meters)")
ax.set_title("Vta02 (Highway 30s Outage, N=33)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle("Stage C8-11.8: Along-Track vs Cross-Track Error Decomposition (30s Outage)", fontsize=14, y=0.98)
plt.tight_layout()
fig2_path = FIG_DIR / "c8_11_8_along_vs_cross_decomposition.png"
fig.savefig(fig2_path, dpi=200)
fig.savefig(BRAIN_MEDIA / "c8_11_8_along_vs_cross_decomposition.png", dpi=200)
plt.close(fig)
print(f"Saved: {fig2_path}")

# -----------------------------------------------------------------------------
# Figure 3: Urban Loop 30s Outage Waterfall (Vta04)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
waterfall_conds = [
    'A0_Pure_IMU', 'A1_IMU_NHC', 'A2_IMU_ML_Speed_NHC',
    'A4_Frozen_Quality_Gated', 'Oracle_Heading'
]
waterfall_labels = [
    'Pure IMU\n(A0)',
    '+ NHC Constraints\n(A1)',
    '+ ML Forward Speed\n(A2)',
    '+ Quality-Gated Rolling Compass\n(A4)',
    '+ Oracle Heading Ceiling\n(Counterfactual)'
]
waterfall_vals = [data['trips']['Vta04']['30s']['conditions'][c]['mean_pos_err_m'] for c in waterfall_conds]
colors = ['#c0392b', '#d35400', '#27ae60', '#2980b9', '#8e44ad']

bars = ax.bar(range(len(waterfall_vals)), waterfall_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.axhline(32.0, color='red', linestyle='--', linewidth=1.5, label='SIH <10% Pass Threshold (~32m)')
for bar, val in zip(bars, waterfall_vals):
    y_pos = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y_pos + (15 if y_pos < 1000 else 40),
            f"{val:.1f} m\n({(val/323.0)*100:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(waterfall_vals)))
ax.set_xticklabels(waterfall_labels, fontsize=9.5)
ax.set_ylabel("30s Final Position Error (meters)")
ax.set_title("Vta04 Urban Loop (30s Outage, Dist Traveled ~323m): Stack Error Reduction & Oracle Bound", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="upper right")

plt.tight_layout()
fig3_path = FIG_DIR / "c8_11_8_ablation_waterfall_vta04_30s.png"
fig.savefig(fig3_path, dpi=200)
fig.savefig(BRAIN_MEDIA / "c8_11_8_ablation_waterfall_vta04_30s.png", dpi=200)
plt.close(fig)
print(f"Saved: {fig3_path}")

print("All diagnostic plots generated successfully.")
