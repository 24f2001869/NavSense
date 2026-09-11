"""
SIH26168 - Stage C8-5.2: Wheel-Speed Measurement-Model & Heading-Observability Audit
Module: experiments/audit_heading_observability_c8_5_2.py

Evaluates:
- Part 1: Wheel-Speed Measurement-Model Perturbation (scale s in [0.98, 1.02], bias b in [-0.2, 0.2] m/s).
- Part 2: Window-by-window Heading Observability and Error Tracking (10s, 20s, 30s, 60s).
- Part 3: Diagnostic Geometric Heading Attribution (comparing integrated chord e_geom = int v sin(e_psi) dt
          against actual cross-track error to quantify how much drift variance is attributable to heading).
All evaluated under the provisional No-Huber baseline with Strict Attitude & Bias Freeze.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D
from src.navigation.nhc import NonHolonomicConstraint
from src.navigation.zupt import ZeroVelocityUpdate, CausalStationaryDetector
from src.navigation.kinematic_constraints import SoftLongitudinalAccelerationConstraint
from src.navigation.wheel_odometry import ChassisWheelSpeedFusion
from experiments.run_wheel_speed_fusion_c8_5 import compute_bcac_series
from experiments.audit_wheel_speed_integrity_c8_5_1 import prepare_trip_data_c8_5_1

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def wrap_angle_deg(deg: float) -> float:
    """Wraps angle to [-180, 180) degrees."""
    return float((deg + 180.0) % 360.0 - 180.0)


# =============================================================================
# 1. WINDOW SIMULATION WITH FULL HEADING AND GEOMETRIC TRACKING
# =============================================================================

def simulate_window_with_heading_tracking(
    trip_data: Dict[str, Any],
    kw: int,
    w_dur: int,
    scale_s: float = 1.00,
    bias_b: float = 0.00,
    sigma_wheel: float = 0.20
) -> Dict[str, Any]:
    """
    Simulates a window under provisional No-Huber Condition C2 (Velocity + NHC)
    while recording full trajectory heading errors and geometric integrals.
    """
    dt = trip_data['dt']
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    wheel_rl = trip_data['wheel_rl']
    wheel_rr = trip_data['wheel_rr']
    gt_e = trip_data['gt_e']
    gt_n = trip_data['gt_n']
    gt_u = trip_data['gt_u']
    gt_ve = trip_data['gt_ve']
    gt_vn = trip_data['gt_vn']
    gt_vu = trip_data['gt_vu']
    heading = trip_data['heading']
    sigma_series = trip_data['sigma_series']
    ba_stat = trip_data['ba_stat']

    init_h = float(heading[kw])

    eskf = ESKF3D(
        init_pos_enu=(gt_e[kw], gt_n[kw], gt_u[kw]),
        init_vel_enu=(gt_ve[kw], gt_vn[kw], gt_vu[kw]),
        init_heading_deg=init_h,
        init_pitch_deg=0.0,
        init_roll_deg=0.0,
        init_ba=ba_stat.copy(),
        R_vp=np.eye(3),
        sigma_a=0.291,
        gravity=9.80665
    )

    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )
    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    b1_constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )
    wheel_fusion = ChassisWheelSpeedFusion(
        sigma_wheel=sigma_wheel,
        sigma_lat=0.50,
        sigma_vert=0.50,
        tire_radius=0.2766
    )

    # Base rear wheel speed with scale and bias perturbation
    raw_wheel_speed = 0.5 * (wheel_rl + wheel_rr) * 0.2766
    modeled_wheel_speed = scale_s * raw_wheel_speed + bias_b

    # Continuous trajectory logs
    heading_errors_deg = []
    heading_cov_deg = []
    v_est_history = []
    accum_turn_rad = 0.0

    geom_lat_integral = 0.0
    geom_long_integral = 0.0

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]
        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]
        cur_wheel_spd = modeled_wheel_speed[step_k]

        accum_turn_rad += abs(gz) * dt

        # 1. ESKF strapdown propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Coupled 3D Body Velocity update (NO HUBER: apply_nis_gate=False)
        wheel_fusion.update_eskf_3d_velocity(eskf, cur_wheel_spd, apply_nis_gate=False)

        # 3. ZUPT update
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 4. B1 Strict Attitude Freeze
        b1_constraint.update_eskf(eskf)

        # Telemetry tracking
        yaw_est_deg = float(eskf.attitude.get_yaw_deg())
        yaw_est_rad = np.radians(yaw_est_deg)
        yaw_true_deg = float(heading[step_k])
        e_psi_deg = wrap_angle_deg(yaw_est_deg - yaw_true_deg)
        e_psi_rad = np.radians(e_psi_deg)

        heading_errors_deg.append(e_psi_deg)
        sigma_psi_deg = float(np.degrees(np.sqrt(max(eskf.P[8, 8], 1e-12))))
        heading_cov_deg.append(sigma_psi_deg)

        v_speed = float(np.linalg.norm(eskf.vel_n[0:2]))
        v_est_history.append(v_speed)

        # Continuous geometric chord integration
        geom_lat_integral += v_speed * np.sin(e_psi_rad) * dt
        geom_long_integral += v_speed * (1.0 - np.cos(e_psi_rad)) * dt

    # Terminal errors
    k_end = kw + w_dur - 1
    p_est = eskf.pos_n[0:2]
    p_true = np.array([gt_e[k_end], gt_n[k_end]])
    err_vec = p_est - p_true
    total_drift = float(np.linalg.norm(err_vec))

    # Along-track / Cross-track decomposition
    h_end = np.radians(heading[k_end])
    u_fwd = np.array([np.sin(h_end), np.cos(h_end)])
    u_lat = np.array([np.cos(h_end), -np.sin(h_end)])

    along_track = float(abs(np.dot(err_vec, u_fwd)))
    cross_track = float(abs(np.dot(err_vec, u_lat)))

    final_h_err = heading_errors_deg[-1]
    mean_abs_h_err = float(np.mean(np.abs(heading_errors_deg)))
    max_abs_h_err = float(np.max(np.abs(heading_errors_deg)))

    P_2d = eskf.P[0:2, 0:2]
    cov_trace = float(np.trace(P_2d))

    return {
        'total_drift_m': total_drift,
        'along_track_m': along_track,
        'cross_track_m': cross_track,
        'cov_trace_m2': cov_trace,
        'sigma_p_m': float(np.sqrt(cov_trace)),
        'final_heading_err_deg': final_h_err,
        'mean_abs_heading_err_deg': mean_abs_h_err,
        'max_abs_heading_err_deg': max_abs_h_err,
        'final_sigma_psi_deg': heading_cov_deg[-1],
        'accum_turn_deg': float(np.degrees(accum_turn_rad)),
        'geom_lat_m': float(abs(geom_lat_integral)),
        'geom_long_m': float(abs(geom_long_integral)),
        'geom_total_m': float(np.sqrt(geom_lat_integral**2 + geom_long_integral**2))
    }


# =============================================================================
# 2. PART 1: WHEEL-SPEED MEASUREMENT-MODEL PERTURBATION GRID
# =============================================================================

def audit_measurement_model_grid(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Sweeps scale factor s in [0.98, 1.02] and bias b in [-0.2, +0.2] m/s."""
    scales = [0.98, 0.99, 1.00, 1.01, 1.02]
    biases = [-0.20, -0.10, 0.00, +0.10, +0.20]
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}
        for dur_s in [30, 60]:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"
            trip_res[dur_key] = {}

            grid_matrix = np.zeros((len(scales), len(biases)))

            for i, s in enumerate(scales):
                for j, b in enumerate(biases):
                    drifts = []
                    alongs = []
                    crosses = []

                    for kw in w_indices:
                        res = simulate_window_with_heading_tracking(
                            trip_data, kw, w_dur, scale_s=s, bias_b=b
                        )
                        drifts.append(res['total_drift_m'])
                        alongs.append(res['along_track_m'])
                        crosses.append(res['cross_track_m'])

                    mean_d = float(np.mean(drifts))
                    grid_matrix[i, j] = mean_d
                    trip_res[dur_key][f"s_{s:.2f}_b_{b:+.2f}"] = {
                        'scale_s': s,
                        'bias_b': b,
                        'mean_drift_m': mean_d,
                        'mean_along_m': float(np.mean(alongs)),
                        'mean_cross_m': float(np.mean(crosses)),
                        'std_drift_m': float(np.std(drifts))
                    }

            # Summary sensitivity analysis
            nom_drift = trip_res[dur_key]["s_1.00_b_+0.00"]['mean_drift_m']
            min_drift = float(np.min(grid_matrix))
            max_drift = float(np.max(grid_matrix))
            drift_range = max_drift - min_drift

            trip_res[dur_key]['summary'] = {
                'nominal_drift_m': nom_drift,
                'min_drift_m': min_drift,
                'max_drift_m': max_drift,
                'drift_range_m': drift_range,
                'relative_spread_pct': float(drift_range / max(nom_drift, 1e-3) * 100.0)
            }

        results[t_name] = trip_res

    return results


# =============================================================================
# 3. PART 2 & 3: HEADING OBSERVABILITY & DIAGNOSTIC GEOMETRIC ATTRIBUTION
# =============================================================================

def audit_heading_observability_and_attribution(
    trip_data_vta02: Dict[str, Any],
    trip_data_vta04: Dict[str, Any]
) -> Dict[str, Any]:
    """Audits heading errors, covariance, and geometric correlation to position drift."""
    horizons = [10, 20, 30, 60]
    results = {}

    for trip_data, t_name in [(trip_data_vta02, 'Vta02_suburban'), (trip_data_vta04, 'Vta04_highway')]:
        trip_res = {}

        for dur_s in horizons:
            w_dur = int(round(dur_s / trip_data['dt']))
            w_indices = trip_data['windows'].get(dur_s, [])
            dur_key = f"{dur_s}s"

            window_runs = []
            for kw in w_indices:
                res = simulate_window_with_heading_tracking(
                    trip_data, kw, w_dur, scale_s=1.00, bias_b=0.00
                )
                res['window_idx'] = kw
                window_runs.append(res)

            # Extract series for statistical correlation
            drifts = np.array([r['total_drift_m'] for r in window_runs])
            alongs = np.array([r['along_track_m'] for r in window_runs])
            crosses = np.array([r['cross_track_m'] for r in window_runs])
            final_h_errs = np.array([abs(r['final_heading_err_deg']) for r in window_runs])
            mean_abs_h_errs = np.array([r['mean_abs_heading_err_deg'] for r in window_runs])
            turns = np.array([r['accum_turn_deg'] for r in window_runs])
            geom_lats = np.array([r['geom_lat_m'] for r in window_runs])
            geom_totals = np.array([r['geom_total_m'] for r in window_runs])
            sigma_psis = np.array([r['final_sigma_psi_deg'] for r in window_runs])

            # Correlation coefficients
            def safe_corr(x, y):
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    return 0.0
                return float(np.corrcoef(x, y)[0, 1])

            r_h_drift = safe_corr(final_h_errs, drifts)
            r_mean_h_drift = safe_corr(mean_abs_h_errs, drifts)
            r_geom_cross = safe_corr(geom_lats, crosses)
            r_turn_drift = safe_corr(turns, drifts)

            # R^2 variance explained
            r2_geom_cross = r_geom_cross**2
            r2_h_drift = r_mean_h_drift**2

            # Residual unmodelled error: actual drift minus geometric heading chord
            geom_residual = np.abs(drifts - geom_totals)
            mean_geom_ratio = float(np.mean(geom_totals / np.maximum(drifts, 1e-3)))

            trip_res[dur_key] = {
                'window_count': len(window_runs),
                'mean_total_drift_m': float(np.mean(drifts)),
                'mean_along_track_m': float(np.mean(alongs)),
                'mean_cross_track_m': float(np.mean(crosses)),
                'heading_statistics': {
                    'mean_final_heading_err_deg': float(np.mean(final_h_errs)),
                    'median_final_heading_err_deg': float(np.median(final_h_errs)),
                    'p90_final_heading_err_deg': float(np.percentile(final_h_errs, 90)),
                    'mean_trajectory_heading_err_deg': float(np.mean(mean_abs_h_errs)),
                    'mean_filter_sigma_psi_deg': float(np.mean(sigma_psis)),
                    'mean_accumulated_turn_deg': float(np.mean(turns))
                },
                'diagnostic_geometric_attribution': {
                    'mean_geom_lat_chord_m': float(np.mean(geom_lats)),
                    'mean_geom_total_chord_m': float(np.mean(geom_totals)),
                    'mean_geom_ratio_to_drift': mean_geom_ratio,
                    'correlation_geom_lat_vs_cross_track': r_geom_cross,
                    'r2_variance_explained_cross_track': float(r2_geom_cross),
                    'correlation_heading_err_vs_total_drift': r_mean_h_drift,
                    'r2_variance_explained_total_drift': float(r2_h_drift),
                    'correlation_turn_angle_vs_drift': r_turn_drift,
                    'mean_unmodelled_inertial_residual_m': float(np.mean(geom_residual))
                },
                'window_data': window_runs
            }

        results[t_name] = trip_res

    return results


# =============================================================================
# 4. PUBLICATION DIAGNOSTIC DASHBOARD (6 PANELS)
# =============================================================================

def generate_c8_5_2_dashboard(
    grid_res: Dict[str, Any],
    observ_res: Dict[str, Any],
    out_path: Path
):
    """Generates 6-panel publication diagnostic figure."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Stage C8-5.2: Wheel-Speed Measurement Model & Heading Observability Diagnostic", fontsize=15, fontweight='bold')

    scales = [0.98, 0.99, 1.00, 1.01, 1.02]
    biases = [-0.20, -0.10, 0.00, +0.10, +0.20]

    # Panel 1: Measurement Model Heatmap (Suburban Vta02 at 60s)
    ax1 = axes[0, 0]
    mat_60_02 = np.zeros((len(scales), len(biases)))
    for i, s in enumerate(scales):
        for j, b in enumerate(biases):
            mat_60_02[i, j] = grid_res['Vta02_suburban']['60s'][f"s_{s:.2f}_b_{b:+.2f}"]['mean_drift_m']

    im = ax1.imshow(mat_60_02, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_xticks(np.arange(len(biases)))
    ax1.set_yticks(np.arange(len(scales)))
    ax1.set_xticklabels([f"{b:+.2f}" for b in biases], fontsize=9)
    ax1.set_yticklabels([f"{s:.2f}" for s in scales], fontsize=9)
    ax1.set_xlabel("Wheel Speed Bias b (m/s)")
    ax1.set_ylabel("Wheel Speed Scale s")
    ax1.set_title("1. Suburban 60s Drift: Scale vs Bias Sensitivity", fontweight='bold')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Drift (m)")

    # Annotate values in heatmap
    for i in range(len(scales)):
        for j in range(len(biases)):
            val = mat_60_02[i, j]
            color = "white" if val < np.mean(mat_60_02) else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    # Panel 2: Heading Error Distribution across Horizons
    ax2 = axes[0, 1]
    horiz_labels = ['10s', '20s', '30s', '60s']
    h_err_02 = [observ_res['Vta02_suburban'][h]['heading_statistics']['mean_final_heading_err_deg'] for h in horiz_labels]
    h_p90_02 = [observ_res['Vta02_suburban'][h]['heading_statistics']['p90_final_heading_err_deg'] for h in horiz_labels]
    h_err_04 = [observ_res['Vta04_highway'][h]['heading_statistics']['mean_final_heading_err_deg'] for h in horiz_labels]
    h_p90_04 = [observ_res['Vta04_highway'][h]['heading_statistics']['p90_final_heading_err_deg'] for h in horiz_labels]

    x = np.arange(len(horiz_labels))
    width = 0.35
    ax2.bar(x - width/2, h_err_02, width, label='Vta02 Mean Err', color='#2b5c8f', alpha=0.85)
    ax2.bar(x + width/2, h_err_04, width, label='Vta04 Mean Err', color='#d95f02', alpha=0.85)
    ax2.plot(x - width/2, h_p90_02, 'o--', color='#1b9e77', label='Vta02 P90')
    ax2.plot(x + width/2, h_p90_04, 's--', color='#e7298a', label='Vta04 P90')
    ax2.set_xticks(x)
    ax2.set_xticklabels(horiz_labels)
    ax2.set_ylabel("Heading Error (°)")
    ax2.set_title("2. Heading Error vs Outage Horizon", fontweight='bold')
    ax2.legend(fontsize=8)

    # Panel 3: Scatter Plot: Heading Error vs Position Drift at 60s
    ax3 = axes[0, 2]
    w_data_02_60 = observ_res['Vta02_suburban']['60s']['window_data']
    w_data_04_60 = observ_res['Vta04_highway']['60s']['window_data']

    h_errs_02 = [r['mean_abs_heading_err_deg'] for r in w_data_02_60]
    drifts_02 = [r['total_drift_m'] for r in w_data_02_60]
    h_errs_04 = [r['mean_abs_heading_err_deg'] for r in w_data_04_60]
    drifts_04 = [r['total_drift_m'] for r in w_data_04_60]

    ax3.scatter(h_errs_02, drifts_02, color='#2b5c8f', alpha=0.75, label=f"Vta02 60s (r={observ_res['Vta02_suburban']['60s']['diagnostic_geometric_attribution']['correlation_heading_err_vs_total_drift']:.2f})")
    ax3.scatter(h_errs_04, drifts_04, color='#d95f02', marker='^', s=60, label=f"Vta04 60s (r={observ_res['Vta04_highway']['60s']['diagnostic_geometric_attribution']['correlation_heading_err_vs_total_drift']:.2f})")

    # Trendline for Vta02
    if len(h_errs_02) > 2:
        p = np.polyfit(h_errs_02, drifts_02, 1)
        x_line = np.linspace(min(h_errs_02), max(h_errs_02), 50)
        ax3.plot(x_line, np.polyval(p, x_line), color='#2b5c8f', linestyle=':')

    ax3.set_xlabel("Mean Trajectory Heading Error (°)")
    ax3.set_ylabel("Total Position Drift (m)")
    ax3.set_title("3. Heading Error vs Total Position Drift (60s)", fontweight='bold')
    ax3.legend(fontsize=8)

    # Panel 4: Scatter Plot: Geometric Lateral Chord vs Actual Cross-Track Error
    ax4 = axes[1, 0]
    geom_lats_02 = [r['geom_lat_m'] for r in w_data_02_60]
    crosses_02 = [r['cross_track_m'] for r in w_data_02_60]
    r_lat = observ_res['Vta02_suburban']['60s']['diagnostic_geometric_attribution']['correlation_geom_lat_vs_cross_track']
    r2_lat = observ_res['Vta02_suburban']['60s']['diagnostic_geometric_attribution']['r2_variance_explained_cross_track']

    ax4.scatter(geom_lats_02, crosses_02, color='#7570b3', alpha=0.75, label=f"Suburban 60s (r={r_lat:.2f}, R²={r2_lat*100:.1f}%)")
    max_val = max(max(geom_lats_02), max(crosses_02))
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label='Ideal 1:1 Identity')
    ax4.set_xlabel("Integrated Geometric Chord: ∫ v sin(e_ψ) dt (m)")
    ax4.set_ylabel("Actual Cross-Track Drift (m)")
    ax4.set_title("4. Geometric Lateral Chord vs Cross-Track Drift", fontweight='bold')
    ax4.legend(fontsize=8)

    # Panel 5: Accumulated Turn Excursion vs Drift
    ax5 = axes[1, 1]
    turns_02 = [r['accum_turn_deg'] for r in w_data_02_60]
    turns_04 = [r['accum_turn_deg'] for r in w_data_04_60]
    r_turn_02 = observ_res['Vta02_suburban']['60s']['diagnostic_geometric_attribution']['correlation_turn_angle_vs_drift']

    ax5.scatter(turns_02, drifts_02, color='#4575b4', alpha=0.75, label=f"Vta02 (r={r_turn_02:.2f})")
    ax5.scatter(turns_04, drifts_04, color='#fc8d59', marker='^', s=60, label="Vta04 Highway")
    ax5.set_xlabel("Accumulated Angular Excursion: ∫ |ω_z| dt (°)")
    ax5.set_ylabel("Total Position Drift (m)")
    ax5.set_title("5. Turn Excursion vs Position Drift (60s)", fontweight='bold')
    ax5.legend(fontsize=8)

    # Panel 6: Variance Attribution & Error Breakdown (10s to 60s)
    ax6 = axes[1, 2]
    r2_drift_02 = [observ_res['Vta02_suburban'][h]['diagnostic_geometric_attribution']['r2_variance_explained_total_drift'] * 100 for h in horiz_labels]
    r2_cross_02 = [observ_res['Vta02_suburban'][h]['diagnostic_geometric_attribution']['r2_variance_explained_cross_track'] * 100 for h in horiz_labels]

    x = np.arange(len(horiz_labels))
    ax6.bar(x - width/2, r2_cross_02, width, label='Cross-Track Var Explained by Heading Chord', color='#1b9e77', alpha=0.85)
    ax6.bar(x + width/2, r2_drift_02, width, label='Total Drift Var Explained by Heading', color='#7570b3', alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(horiz_labels)
    ax6.set_ylabel("Proportion of Variance Explained R² (%)")
    ax6.set_ylim(0, 100)
    ax6.set_title("6. Heading Error Explanatory Power (R²)", fontweight='bold')
    ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[C8-5.2] Diagnostic figure saved to: {out_path}")


# =============================================================================
# 5. MASTER ORCHESTRATION
# =============================================================================

def main():
    json_path = RES_DIR / "c8_5_2_heading_observability.json"
    fig_path = FIG_DIR / "c8_5_2_heading_observability.png"

    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only" and json_path.exists():
        print(f"[C8-5.2] Loading existing JSON from {json_path} for plotting...")
        with open(json_path, 'r') as f:
            res = json.load(f)
        generate_c8_5_2_dashboard(
            res['part_1_measurement_model_grid'],
            res['part_2_and_3_heading_observability'],
            fig_path
        )
        return

    print("=" * 80)
    print("SIH26168 - Stage C8-5.2: Wheel-Speed Measurement-Model & Heading Observability Audit")
    print("=" * 80)
    t0 = time.time()

    print("[1/4] Ingesting telemetry and IMU data for Vta02 and Vta04...")
    data_vta02 = prepare_trip_data_c8_5_1("Vta02", dt=0.1)
    data_vta04 = prepare_trip_data_c8_5_1("Vta04", dt=0.1)

    print("[2/4] Executing Part 1: Wheel-Speed Measurement-Model Perturbation Grid (25 pairs)...")
    grid_res = audit_measurement_model_grid(data_vta02, data_vta04)

    print("[3/4] Executing Part 2 & 3: Heading Observability Tracking & Geometric Attribution...")
    observ_res = audit_heading_observability_and_attribution(data_vta02, data_vta04)

    # Master consolidation (strip heavy window_data from JSON summary)
    master_results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'stage': 'C8-5.2',
            'configuration': 'Provisional No-Huber with Strict Attitude & Bias Freeze',
            'trips_evaluated': ['Vta02', 'Vta04'],
            'total_windows': len(data_vta02['windows'].get(30, [])) + len(data_vta04['windows'].get(30, [])),
            'execution_time_sec': float(time.time() - t0)
        },
        'part_1_measurement_model_grid': grid_res,
        'part_2_and_3_heading_observability': observ_res
    }

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"[C8-5.2] Master results saved to: {json_path}")

    # Generate Figure
    print("[4/4] Generating 6-panel publication diagnostic figure...")
    generate_c8_5_2_dashboard(grid_res, observ_res, fig_path)

    print("=" * 80)
    print(f"Stage C8-5.2 Audit completed in {time.time() - t0:.2f} seconds.")
    print("=" * 80)


if __name__ == '__main__':
    main()
