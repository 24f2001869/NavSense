"""
generate_figure_03_clean.py
---------------------------
Regenerates Figure 3 for The Journal of Navigation:
Controlled empirical scaling of Causal TCN training (6-trip vs. 39-trip).
Excludes Random Forest and all exploratory baselines.
Uses strict 2-model comparison (Current 6-trip TCN vs Expanded 39-trip TCN).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def generate_figure_03():
    with open('results/expanded_tcn_benchmark/overall_benchmark_summary.json', 'r') as f:
        summary_data = json.load(f)
    with open('results/expanded_tcn_benchmark/category_breakdown_summary.json', 'r') as f:
        cat_data = json.load(f)
    with open('results/expanded_tcn_benchmark/per_trip_evaluation_results.json', 'r') as f:
        trip_data = json.load(f)

    df_summary = pd.DataFrame(summary_data)
    df_cat = pd.DataFrame(cat_data)
    df_trip = pd.DataFrame(trip_data)

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'

    c_6trip = '#e11d48'   # Rose / Coral red
    c_39trip = '#059669'  # Emerald green

    # Panel 1: Vehicle Speed MAE
    ax1 = axs[0, 0]
    models = ['6-Trip TCN\n(1.1 h training)', '39-Trip TCN\n(12.5 h training)']
    maes = [df_summary.loc[0, 'Vehicle MAE (m/s)'], df_summary.loc[1, 'Vehicle MAE (m/s)']]
    bars1 = ax1.bar([0, 1], maes, width=0.45, color=[c_6trip, c_39trip], edgecolor='#0f172a', linewidth=1.2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(models, fontweight='bold', fontsize=9.5)
    ax1.set_ylabel('Mean Absolute Error (m/s)', fontweight='bold', fontsize=10.5)
    ax1.set_title('(a) Forward Velocity MAE (19 Held-Out Routes, 115,420 Epochs)', fontweight='bold', fontsize=10.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f'{h:.2f} m/s\n({h*3.6:.1f} km/h)', 
                 ha='center', va='bottom', fontweight='bold', fontsize=9.0)
    ax1.text(0.5, 4.8, '-55.27% MAE Reduction\n(6.17 → 2.76 m/s)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.9),
             fontsize=9.0, fontweight='bold', color='#0f172a')
    ax1.set_ylim(0, 7.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: 30s and 60s Dead-Reckoning Drift
    ax2 = axs[0, 1]
    x_pos = np.arange(2)
    w_bar = 0.30
    d30s = [df_summary.loc[0, '30s Drift (m)'], df_summary.loc[1, '30s Drift (m)']]
    d60s = [df_summary.loc[0, '60s Drift (m)'], df_summary.loc[1, '60s Drift (m)']]
    r1 = ax2.bar(x_pos - w_bar/2, d30s, w_bar, label='30-second Outage Horizon', color='#f59e0b', edgecolor='#0f172a', linewidth=1.1)
    r2 = ax2.bar(x_pos + w_bar/2, d60s, w_bar, label='60-second Outage Horizon', color='#8b5cf6', edgecolor='#0f172a', linewidth=1.1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontweight='bold', fontsize=9.5)
    ax2.set_ylabel('Along-Track Drift Error (m)', fontweight='bold', fontsize=10.5)
    ax2.set_title('(b) Integrated Position Drift (30 s & 60 s Blackout Horizons)', fontweight='bold', fontsize=10.5)
    for bar in r1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 3.0, f'{h:.1f} m', ha='center', fontsize=8.5, fontweight='bold')
    for bar in r2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 3.0, f'{h:.1f} m', ha='center', fontsize=8.5, fontweight='bold')
    ax2.text(0.5, 210, '-64.72% 60s Drift Reduction\n(263.6 → 93.0 m)', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.9),
             fontsize=9.0, fontweight='bold', color='#0f172a')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax2.set_ylim(0, 310)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Road Category Breakdown
    ax3 = axs[1, 0]
    cats = ['Suburban Towns (Vta)', 'Dense Urban (Vtb)', 'Mountain Routes (Vw)', 'Motorways (V-Vfa)']
    x_cat = np.arange(len(cats))
    w_c = 0.32
    mae_6 = df_cat['Current TCN MAE'].values
    mae_39 = df_cat['Expanded TCN MAE'].values
    ax3.bar(x_cat - w_c/2, mae_6, w_c, label='6-Trip TCN', color=c_6trip, edgecolor='#0f172a', linewidth=1.1)
    ax3.bar(x_cat + w_c/2, mae_39, w_c, label='39-Trip TCN', color=c_39trip, edgecolor='#0f172a', linewidth=1.1)
    ax3.set_xticks(x_cat)
    ax3.set_xticklabels(cats, fontweight='bold', fontsize=8.8, rotation=10, ha='right')
    ax3.set_ylabel('MAE (m/s)', fontweight='bold', fontsize=10.5)
    ax3.set_title('(c) Cross-Route Generalisation Across Road Environments', fontweight='bold', fontsize=10.5)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 9.0)

    # Panel 4: Per-Trip MAE Distribution
    ax4 = axs[1, 1]
    box_data = [df_trip['curr_mae'].values, df_trip['exp_mae'].values]
    bp = ax4.boxplot(box_data, tick_labels=['6-Trip TCN', '39-Trip TCN'], patch_artist=True,
                     boxprops=dict(linewidth=1.2), medianprops=dict(color='#0f172a', linewidth=2.0))
    box_colors = ['#fecdd3', '#a7f3d0']
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col)
    ax4.set_ylabel('Per-Trip MAE (m/s)', fontweight='bold', fontsize=10.5)
    ax4.set_title('(d) Per-Trip MAE Across 19 Held-Out Test Routes', fontweight='bold', fontsize=10.5)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png = 'publication/journal_of_navigation_submission/figures/Figure_03.png'
    out_tif = 'publication/journal_of_navigation_submission/figures/Figure_03.tif'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    im = Image.open(out_png)
    im.save(out_tif, dpi=(600, 600), compression="tiff_lzw")
    print(f'Created clean Figure 3: {out_png} and {out_tif}')

if __name__ == '__main__':
    generate_figure_03()
