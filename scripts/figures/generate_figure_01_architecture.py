"""
generate_figure_01_architecture.py
----------------------------------
Generates publication-grade Figure 1 for The Journal of Navigation:
End-to-end NavSense modular navigation architecture.
Clean, modern, crisp typography (Times New Roman), zero text overlap,
accurate parameter counts (4 residual blocks, 61-step receptive field, ~60k weights).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_figure_01():
    fig, ax = plt.subplots(figsize=(11, 7.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis('off')

    # Styling constants
    box_edge = '#1e293b'
    text_color = '#0f172a'
    sub_color = '#334155'
    accent_blue = '#0284c7'
    accent_purple = '#7c3aed'
    accent_amber = '#d97706'
    accent_emerald = '#059669'

    # Title block
    ax.text(50, 72.8, 'NavSense: Failure-Aware Smartphone Inertial Dead Reckoning Architecture', 
            ha='center', va='center', fontsize=12.2, fontweight='bold', color=text_color, fontname='Times New Roman')
    ax.text(50, 70.4, 'Causal TCN Speed Estimation, Decoupled Kalman Damping, and Adaptive Regime Fusion', 
            ha='center', va='center', fontsize=9.5, fontstyle='italic', color=sub_color, fontname='Times New Roman')

    # Helper function for drawing boxes
    def draw_box(x, y, w, h, title, desc_lines, tag=None, tag_color=accent_blue, bg='#ffffff'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                     linewidth=1.2, edgecolor=box_edge, facecolor=bg)
        ax.add_patch(box)
        ax.text(x + 2, y + h - 2.2, title, fontsize=9.2, fontweight='bold', color=text_color, fontname='Times New Roman')
        
        # Draw description lines
        line_y = y + h - 4.4
        for line in desc_lines:
            ax.text(x + 2, line_y, line, fontsize=7.8, color=sub_color, fontname='Times New Roman')
            line_y -= 1.95

        if tag:
            tbox = patches.FancyBboxPatch((x + w - 21, y + h - 3.2), 19, 2.3, boxstyle="round,pad=0.2,rounding_size=0.5",
                                          linewidth=0.8, edgecolor=tag_color, facecolor=tag_color)
            ax.add_patch(tbox)
            ax.text(x + w - 11.5, y + h - 2.0, tag, fontsize=6.2, fontweight='bold', color='#ffffff', ha='center', va='center', fontname='Times New Roman')

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, color=box_edge, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=1.2, color=color, shrinkA=2, shrinkB=2))

    # SECTION 1: SENSOR INGESTION (Y: 57.5 to 66.5)
    ax.text(6, 62.0, 'LAYER 1:\nSENSORS', fontsize=8.2, fontweight='bold', color=sub_color, ha='center', va='center', fontname='Times New Roman')
    draw_box(16, 57.5, 39, 9.0, 'Consumer Smartphone IMU', 
             ['Tri-axial specific force & angular rate', 
              'HAL ~400 Hz, Android callback ~8.63 Hz avg (mean dt = 115.8 ms)',
              'Benchmark: synchronized 10 Hz telemetry; CAN velocity reference'], 
             'CONTINUOUS', accent_emerald)
    draw_box(57, 57.5, 39, 9.0, 'Consumer Smartphone GNSS', 
             ['Open-sky PVT positioning (1.0 Hz fix)', 
              'Triggers zero-satellite blackout state upon signal loss',
              'Provides pre-outage calibration & initial kinematic state'], 
             'INTERMITTENT', accent_amber)

    # Arrows 1 -> 2
    draw_arrow(35.5, 57.5, 35.5, 52.5)
    draw_arrow(76.5, 57.5, 76.5, 52.5)

    # SECTION 2: FEATURE PREPROCESSING (Y: 44.5 to 52.5)
    ax.text(6, 48.5, 'LAYER 2:\nPRE-PROCESS', fontsize=8.2, fontweight='bold', color=sub_color, ha='center', va='center', fontname='Times New Roman')
    draw_box(16, 44.5, 80, 8.0, 'Kinematic Preprocessing & Causal Sliding Buffer', 
             ['Pre-outage gravity leveling via tilt quaternion; 4.5 Hz low-pass filter decimation',
              'Rolling 100-sample input buffer (10.0 s): 9 channels [a_lin(3), ω(3), a_h, a_v, κ = ω_mag · a_h]'], 
             '9 CHANNELS', accent_blue)

    # Arrow 2 -> 3
    draw_arrow(35.5, 44.5, 35.5, 39.5)
    draw_arrow(76.5, 44.5, 76.5, 39.5)

    # SECTION 3: ESTIMATORS (Y: 28.0 to 39.5)
    ax.text(6, 33.5, 'LAYER 3:\nESTIMATION', fontsize=8.2, fontweight='bold', color=sub_color, ha='center', va='center', fontname='Times New Roman')
    draw_box(16, 28.0, 39, 11.5, 'Causal Dilated TCN', 
             ['4 causal dilated residual blocks (d = 1, 2, 4, 8; k = 3)',
              'Receptive field reach: 61 samples (6.0 s causal look-back)',
              '60,225 weights; Huber loss; emits forward speed v_TCN',
              'On-device execution: 4.2–7.8 ms latency on Google Pixel 7a'], 
             'ONNX (5.8 ms)', accent_purple)
    draw_box(57, 28.0, 39, 11.5, 'Adaptive Regime Fusion & ZVD', 
             ['Causal Zero-Velocity Detector (ZVD) on accel/gyro variance',
              'Dynamic momentum blending weight β_k across regimes',
              'Smooth cruise state preservation & standstill clamp',
              'Clamps velocity during detected stationary intervals'], 
             'REGIME GATE', accent_amber)

    # Arrows 3 -> 4
    draw_arrow(35.5, 28.0, 35.5, 23.5)
    draw_arrow(76.5, 28.0, 76.5, 23.5)

    # SECTION 4: STATE ESTIMATION & FUSION (Y: 11.5 to 23.5)
    ax.text(6, 17.5, 'LAYER 4:\nKALMAN FUSION', fontsize=8.2, fontweight='bold', color=sub_color, ha='center', va='center', fontname='Times New Roman')
    draw_box(16, 11.5, 80, 12.0, '15-State Error-State Kalman Filter (ESKF) with Decoupled Velocity Damping', 
             ['Error states: δx = [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]^T; Quaternion attitude integration',
              'Decoupled Lateral Velocity Damping (K_y[6:15] = 0) isolates lateral innovations strictly to velocity',
              'Suppresses false attitude & gyroscope bias corruption caused by vehicle sideslip during turns',
              'On-device execution latency: 0.8–1.4 ms per epoch on Google Pixel 7a (Total loop mean: 9.15 ms)'], 
             'ESKF (1.0 ms)', accent_blue)

    # Arrow 4 -> 5
    draw_arrow(56.0, 11.5, 56.0, 7.5)

    # SECTION 5: DOWNSTREAM SERVICES (Y: 1.0 to 7.5)
    ax.text(6, 4.2, 'LAYER 5:\nOUTPUT', fontsize=8.2, fontweight='bold', color=sub_color, ha='center', va='center', fontname='Times New Roman')
    draw_box(16, 1.0, 80, 6.5, 'Failure-Aware Dead Reckoning Output & Vector Shadow Map', 
             ['Propagated 3D position, velocity, and orientation; dynamic covariance confidence bounds',
              'Open-loop OSM road network visualization; closed-loop heading feedback disabled to prevent turn latching'], 
             '100 ms EPOCH BUDGET', accent_emerald)

    plt.tight_layout()
    out_png = 'publication/journal_of_navigation_submission/figures/Figure_01.png'
    out_tif = 'publication/journal_of_navigation_submission/figures/Figure_01.tif'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    im = Image.open(out_png)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(out_tif, dpi=(600, 600), compression="tiff_lzw")
    print(f'Created clean Figure 1: {out_png} and {out_tif}')

if __name__ == '__main__':
    generate_figure_01()
