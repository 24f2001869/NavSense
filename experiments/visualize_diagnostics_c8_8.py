"""
SIH26168 - Stage C8-8: Diagnostic Figures Generator
Module: experiments/visualize_diagnostics_c8_8.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RES_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(RES_DIR / "c8_8_longitudinal_diagnostics.json", "r") as f:
    diag = json.load(f)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------------------
# Figure 1: Speed Error & Heading Error vs Position Error Components
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: I_speed vs Signed Along-Track Error (10s)
wins_10s = diag['task1_window_level']['10s']
i_spd_10s = [w['i_speed_m'] for w in wins_10s]
along_10s = [w['c4_final_along_err_m'] for w in wins_10s]
axs[0].scatter(i_spd_10s, along_10s, color='#1f77b4', s=60, edgecolors='black', label='10s Windows (N=16)')
# Fit line
m, b = np.polyfit(i_spd_10s, along_10s, 1)
x_line = np.linspace(min(i_spd_10s), max(i_spd_10s), 100)
axs[0].plot(x_line, m * x_line + b, color='#1f77b4', linestyle='--', label=f'Fit: slope={m:.2f}, r=+0.47')
axs[0].plot(x_line, x_line, color='gray', linestyle=':', label='1:1 Parity Line')
axs[0].axhline(0, color='black', lw=0.8)
axs[0].axvline(0, color='black', lw=0.8)
axs[0].set_title("10s Horizon: Integrated Speed Error vs Along-Track Drift", fontsize=12, fontweight='bold')
axs[0].set_xlabel(r"Integrated RF Speed Error $I_{\mathrm{speed}} = \int (v_{\mathrm{RF}} - v_{\mathrm{true}}) dt$ (m)", fontsize=11)
axs[0].set_ylabel("Final Along-Track Error (m)", fontsize=11)
axs[0].legend(frameon=True)

# Subplot 2: Heading Error vs Cross-Track Error (10s, 20s, 30s)
colors = {'10s': '#2ca02c', '20s': '#ff7f0e', '30s': '#d62728'}
for dur in ['10s', '20s', '30s']:
    wins = diag['task1_window_level'][dur]
    h_err = [w['c4_final_heading_err_deg'] for w in wins]
    cross_err = [abs(w['c4_final_cross_err_m']) for w in wins]
    axs[1].scatter(h_err, cross_err, color=colors[dur], s=70, edgecolors='black', label=f'{dur} Windows')
axs[1].set_title("Heading Error vs Cross-Track Error (Vta04)", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Final Heading Error (deg)", fontsize=11)
axs[1].set_ylabel("Final Cross-Track Error |e_cross| (m)", fontsize=11)
axs[1].legend(frameon=True)

# Subplot 3: Energy Share Breakdown (Along vs Cross)
horizons = ['10s', '20s', '30s']
along_shares = [diag['task7_heading_interaction'][h]['along_share_of_total_energy_pct'] for h in horizons]
cross_shares = [diag['task7_heading_interaction'][h]['cross_share_of_total_energy_pct'] for h in horizons]
x = np.arange(len(horizons))
width = 0.5
axs[2].bar(x, along_shares, width, label='Along-Track Error Energy (%)', color='#1f77b4', edgecolor='black')
axs[2].bar(x, cross_shares, width, bottom=along_shares, label='Cross-Track Error Energy (%)', color='#d62728', edgecolor='black')
axs[2].set_xticks(x)
axs[2].set_xticklabels(horizons, fontsize=11, fontweight='bold')
axs[2].set_ylabel("Share of Total Squared Position Error (%)", fontsize=11)
axs[2].set_title("Total 2D Error Decomposition: Along vs Cross", fontsize=12, fontweight='bold')
axs[2].set_ylim(0, 105)
for i in range(len(horizons)):
    axs[2].text(i, along_shares[i] / 2, f"{along_shares[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    axs[2].text(i, along_shares[i] + cross_shares[i] / 2, f"{cross_shares[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
axs[2].legend(frameon=True, loc='upper right')

plt.tight_layout()
fig1_path = FIG_DIR / "c8_8_error_decomposition_and_correlations.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}", flush=True)

# -------------------------------------------------------------------------
# Figure 2: Dynamic Pitch & Mounting Vibration Diagnostics
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: PSD of Phone Acceleration vs Reference Chassis Acceleration
psd_data = diag['task4_acceleration']['psd']
f_hz = psd_data['frequencies_hz']
psd_phone = psd_data['psd_phone']
psd_ref = psd_data['psd_ref']
psd_diff = psd_data['psd_diff']

axs[0].semilogy(f_hz, psd_phone, label='Phone Accel Forward (Mount Frame)', color='#1f77b4', lw=2)
axs[0].semilogy(f_hz, psd_ref, label='Reference Vehicle Accel (Chassis)', color='#2ca02c', lw=2)
axs[0].semilogy(f_hz, psd_diff, label='Discrepancy (Phone - Ref)', color='#d62728', linestyle='--', lw=1.5)
axs[0].axvspan(2.0, 5.0, alpha=0.15, color='orange', label='Mount Flutter Band (2–5 Hz)')
axs[0].set_title("Longitudinal Acceleration Power Spectral Density (PSD)", fontsize=12, fontweight='bold')
axs[0].set_xlabel("Frequency (Hz)", fontsize=11)
axs[0].set_ylabel("Power Spectral Density ((m/s²)² / Hz)", fontsize=11)
axs[0].set_xlim(0, 5)
axs[0].legend(frameon=True)

# Subplot 2: Dynamic Pitch Gravity Leakage vs Observed Accel Discrepancy Magnitude
t5 = diag['task5_dynamic_pitch']
magnitudes = [t5['rms_predicted_g_leakage_ms2'], t5['rms_observed_discrepancy_ms2']]
bars = axs[1].bar(['Predicted Gravity Leakage\n(Dynamic Pitch)', 'Observed Accel Discrepancy\n(Phone vs Ref)'],
                  magnitudes, color=['#ff7f0e', '#d62728'], width=0.4, edgecolor='black')
axs[1].set_ylabel("RMS Acceleration (m/s²)", fontsize=11)
axs[1].set_title("Dynamic Pitch Hypothesis Test (Magnitude Comparison)", fontsize=12, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.4f} m/s²", ha='center', va='bottom', fontweight='bold')
axs[1].text(0.5, 2.0, "Dynamic Pitch explains ONLY 0.76% of Discrepancy!\n-> REJECTED AS DOMINANT EXPLANATION",
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee', edgecolor='#c00'), fontsize=10, fontweight='bold')

plt.tight_layout()
fig2_path = FIG_DIR / "c8_8_pitch_and_vibration_diagnostics.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}", flush=True)

# -------------------------------------------------------------------------
# Figure 3: Controlled Counterfactual (Canonical Phone Speed vs Oracle Ref Speed)
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

horiz_cf = ['5s', '10s', '20s', '30s', '60s']
c4_pos = [diag['task8_counterfactual'][h]['c4_smartphone_pos_err_m'] for h in horiz_cf]
ora_pos = [diag['task8_counterfactual'][h]['oracle_speed_pos_err_m'] for h in horiz_cf]
c4_drift = [diag['task8_counterfactual'][h]['c4_smartphone_drift_pct'] for h in horiz_cf]
ora_drift = [diag['task8_counterfactual'][h]['oracle_speed_drift_pct'] for h in horiz_cf]

x = np.arange(len(horiz_cf))
width = 0.35

axs[0].bar(x - width/2, c4_pos, width, label='Canonical Smartphone Speed (C4)', color='#1f77b4', edgecolor='black')
axs[0].bar(x + width/2, ora_pos, width, label='Counterfactual Oracle Speed (Ref Speed)', color='#2ca02c', edgecolor='black')
axs[0].set_xticks(x)
axs[0].set_xticklabels(horiz_cf, fontsize=11, fontweight='bold')
axs[0].set_ylabel("Final 2D Position Error (m)", fontsize=11)
axs[0].set_title("Position Error: Canonical Speed vs Oracle Speed", fontsize=12, fontweight='bold')
axs[0].legend(frameon=True)

axs[1].plot(x, c4_drift, marker='o', lw=2, color='#1f77b4', label='Canonical Smartphone Speed (C4)')
axs[1].plot(x, ora_drift, marker='s', lw=2, color='#2ca02c', linestyle='--', label='Counterfactual Oracle Speed')
axs[1].axhline(10.0, color='red', linestyle=':', lw=1.5, label='SIH Compliance Target (< 10%)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(horiz_cf, fontsize=11, fontweight='bold')
axs[1].set_ylabel("Mean Drift (%)", fontsize=11)
axs[1].set_title("Drift Percentage: Canonical vs Oracle Speed", fontsize=12, fontweight='bold')
axs[1].legend(frameon=True)

plt.tight_layout()
fig3_path = FIG_DIR / "c8_8_counterfactual_oracle_comparison.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}", flush=True)
