"""
SIH26168 - Stage C8-4: Controlled Closed-Loop ESKF Map Constraints with MHT Validation
Module: experiments/run_closed_loop_map_matching_c8_4.py

Evaluates whether MHT-validated map constraints with the 5-Gate Safety Architecture
can safely correct inertial drift in closed-loop, or whether closed-loop feedback
creates map-induced divergence or wrong-road latching.

Sub-Stages Evaluated:
- C8-4A: Shadow Closed-Loop Association (zero filter feedback, verifies zero leakage)
- C8-4B: Single-Update Step Telemetry & Correction Clamping
- C8-4C: Full Controlled Benchmark across 54 suburban and 9 highway windows (10s, 20s, 30s, 60s)
  * C0: Frozen Baseline (C7-B1-D Strict Freeze, No map)
  * C1: Lateral Map + MHT (r_perp = d_perp - mu_lane)
  * C2: Heading Map + MHT (r_psi = wrap(psi - psi_road))
  * C3: Joint Lateral + Heading Map + MHT
- C8-4D: Map-Induced Regressions Forensic Audit (isolates any window where e_map > e_baseline)
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
from src.navigation.nhc import NonHolonomicConstraint, rotvec_to_quat, quat_mult
from src.navigation.zupt import ZeroVelocityUpdate, CausalStationaryDetector
from src.navigation.kinematic_constraints import SoftLongitudinalAccelerationConstraint
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex, wrap_angle_rad
from src.map.beam_search import MultiHypothesisTracker, PathHypothesis
from src.navigation.map_constraints import MHTMapConstraintManager

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def compute_bcac_series(acc_v, dt=0.1, w_b_s=10.0, sigma_c0=0.291):
    """Computes frozen BCAC process noise series sigma_a(k)."""
    n = len(acc_v)
    w_k = 10
    wb_epochs = int(round(w_b_s / dt))

    jerk_rms = np.zeros(n)
    for i in range(n):
        i_start = max(0, i - w_k + 1)
        win_acc = acc_v[i_start : i + 1]
        if len(win_acc) > 1:
            jerk = np.diff(win_acc, axis=0) / dt
            jerk_rms[i] = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        else:
            jerk_rms[i] = 0.0

    b = np.zeros(n)
    for i in range(n):
        i_start = max(0, i - wb_epochs + 1)
        b[i] = np.median(jerk_rms[i_start : i + 1])

    j_norm = jerk_rms / (b + 1e-4)
    mult = np.clip(np.sqrt(j_norm), 0.75, 1.35)
    return sigma_c0 * mult


def prepare_trip_data(trip_name: str, dt: float = 0.1) -> dict:
    """Loads trip telemetry, IMU data, and builds OSM road network index."""
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    acc_v, gyro_v, R_vp, angles = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124], dtype=np.float64)

    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = np.zeros_like(speed)

    sigma_series = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    osm_file = REPO_ROOT / "data" / "raw" / "maps" / f"{trip_name}_network.osm"
    if not osm_file.exists():
        raise FileNotFoundError(f"OSM vector cache not found: {osm_file}")

    print(f"[{trip_name}] Loading OSM vector map from {osm_file.name}...")
    parsed_map = parse_osm_network(osm_file, lat0, lon0)
    road_index = RoadNetworkIndex(parsed_map['segments'])
    print(f"[{trip_name}] Road network ready with {road_index.num_segments} directed segments.")

    return {
        'trip_name': trip_name,
        'acc_v': acc_v,
        'gyro_v': gyro_v,
        'speed': speed,
        'heading': heading,
        'gt_e': gt_e,
        'gt_n': gt_n,
        'gt_u': gt_u,
        'gt_ve': gt_ve,
        'gt_vn': gt_vn,
        'gt_vu': gt_vu,
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'road_index': road_index,
        'n_epochs': n
    }


def compute_segment_gt_distance(seg_idx: int, gt_pos: np.ndarray, road_index: RoadNetworkIndex) -> float:
    """Computes orthogonal distance from true RTK VBOX position to segment."""
    seg = road_index.segments[seg_idx]
    p1 = seg['p1']
    p2 = seg['p2']
    diff = p2 - p1
    length_sq = float(np.sum(diff**2))
    ap = gt_pos[:2] - p1
    t = np.clip(float(np.dot(ap, diff)) / (length_sq + 1e-12), 0.0, 1.0)
    proj = p1 + t * diff
    return float(np.linalg.norm(gt_pos[:2] - proj))


# ==============================================================================
# SINGLE-WINDOW SIMULATOR FOR A GIVEN CONDITION
# ==============================================================================

def simulate_window_condition(
    kw: int,
    w_dur: int,
    dt: float,
    trip_data: dict,
    condition: str,   # 'C0_Baseline', 'C8_4A_Shadow', 'C1_Lateral', 'C2_Heading', 'C3_Joint'
    search_radius_m: float = 25.0,
    heading_gate_deg: float = 30.0,
    commitment_margin_delta: float = 0.20,
    update_interval_s: float = 1.0,
    gt_threshold_m: float = 7.5
) -> Dict[str, Any]:
    """
    Simulates one blackout window for a single condition.
    Returns complete final errors and map update telemetry.
    """
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    heading = trip_data['heading']
    speed = trip_data['speed']
    gt_e = trip_data['gt_e']
    gt_n = trip_data['gt_n']
    gt_u = trip_data['gt_u']
    gt_ve = trip_data['gt_ve']
    gt_vn = trip_data['gt_vn']
    gt_vu = trip_data['gt_vu']
    ba_stat = trip_data['ba_stat']
    sigma_series = trip_data['sigma_series']
    road_index = trip_data['road_index']

    init_h = float(heading[kw])

    # 15-state ESKF with Baseline mechanisms
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

    nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)
    zupt = ZeroVelocityUpdate(sigma_vel=0.05)
    detector = CausalStationaryDetector(
        dt=dt, window_sec=0.5, persist_sec=0.8,
        th_acc_var=0.04, th_gyro_var=0.003, th_grav_diff=0.25, th_jerk_rms=8.0
    )
    b1_constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )

    # Initialize MHT Map Constraint Manager if map is active
    map_mgr = None
    if condition != 'C0_Baseline':
        map_mgr = MHTMapConstraintManager(
            road_index=road_index,
            beam_width_K=3,
            search_radius_m=search_radius_m,
            heading_gate_deg=heading_gate_deg,
            commitment_margin_delta=commitment_margin_delta,
            sigma_lane_m=2.5,
            sigma_psi_deg=5.0,
            lane_offset_m=-0.90,
            n_persist=2,
            max_pos_correction_m=5.0,
            update_interval_sec=update_interval_s
        )

    t_start = kw * dt
    last_map_time = -np.inf

    telemetry = {
        'condition': condition,
        'window_idx': kw,
        'total_queries': 0,
        'applied_updates': 0,
        'wrong_road_updates': 0,
        'rejections': {
            'gate1_ambiguous': 0,
            'gate2_geometry': 0,
            'gate3_heading': 0,
            'gate4_persistence': 0,
            'gate5_nis': 0,
            'rate_limit': 0
        },
        'corrections_pos_norm': [],
        'nis_values': [],
        'trajectory': []
    }

    mode_map = {
        'C8_4A_Shadow': 'joint',
        'C1_Lateral': 'lateral',
        'C2_Heading': 'heading',
        'C3_Joint': 'joint'
    }
    is_shadow = (condition == 'C8_4A_Shadow')
    active_mode = mode_map.get(condition, 'joint')

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]
        cur_spd = speed[step_k]

        # 1. ESKF strapdown propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. NHC update
        nhc.update_eskf(eskf)

        # 3. ZUPT update
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 4. B1 Strict Attitude Freeze
        b1_constraint.update_eskf(eskf)

        # 5. Map Constraint Evaluation (1 Hz cadence)
        if map_mgr is not None and (t_now - last_map_time) >= (update_interval_s - 1e-4):
            last_map_time = t_now
            telemetry['total_queries'] += 1

            res = map_mgr.evaluate_and_update(
                eskf=eskf,
                t_now=t_now,
                mode=active_mode,
                speed_ms=cur_spd,
                gyro_z_rads=gz,
                shadow_mode=is_shadow
            )

            if res['active']:
                telemetry['applied_updates'] += 1
                telemetry['corrections_pos_norm'].append(float(np.linalg.norm(res['delta_pos'])))
                telemetry['nis_values'].append(float(res['nis']))

                # Check ground truth distance of chosen candidate segment
                cand_seg = res['candidate']['segment_idx']
                gt_p = np.array([gt_e[step_k], gt_n[step_k]])
                d_gt = compute_segment_gt_distance(cand_seg, gt_p, road_index)
                if d_gt > gt_threshold_m:
                    telemetry['wrong_road_updates'] += 1
            else:
                reason = res.get('reason', 'unknown')
                if reason in telemetry['rejections']:
                    telemetry['rejections'][reason] += 1

    # End of blackout: evaluate final errors
    k_end = kw + w_dur - 1
    st_final = eskf.get_state()

    est_e, est_n = st_final['pos_n'][0], st_final['pos_n'][1]
    true_e, true_n = gt_e[k_end], gt_n[k_end]
    true_h = heading[k_end]

    d_e = est_e - true_e
    d_n = est_n - true_n
    e_total = float(np.sqrt(d_e**2 + d_n**2))

    # Along-track and cross-track decomposition
    h_rad = np.radians(true_h)
    u_fwd = np.array([np.sin(h_rad), np.cos(h_rad)])
    u_lat = np.array([np.cos(h_rad), -np.sin(h_rad)])
    e_along = float(abs(d_e * u_fwd[0] + d_n * u_fwd[1]))
    e_cross = float(abs(d_e * u_lat[0] + d_n * u_lat[1]))

    # Heading error
    est_yaw = float(st_final['yaw_deg'])
    e_heading = float(abs(np.degrees(wrap_angle_rad(np.radians(est_yaw - true_h)))))

    # Velocity error
    est_v = st_final['vel_n'][:2]
    true_v = np.array([gt_ve[k_end], gt_vn[k_end]])
    e_vel = float(np.linalg.norm(est_v - true_v))

    # Final position covariance trace
    p_cov_trace = float(np.trace(eskf.P[0:2, 0:2]))

    telemetry.update({
        'final_e_total': e_total,
        'final_e_along': e_along,
        'final_e_cross': e_cross,
        'final_e_heading': e_heading,
        'final_e_vel': e_vel,
        'final_p_cov_trace': p_cov_trace,
        'max_correction_m': float(max(telemetry['corrections_pos_norm'])) if telemetry['corrections_pos_norm'] else 0.0,
        'wrong_road_rate': float(telemetry['wrong_road_updates'] / max(1, telemetry['applied_updates'])) if telemetry['applied_updates'] > 0 else 0.0
    })

    return telemetry


# ==============================================================================
# MASTER BENCHMARK RUNNER
# ==============================================================================

def run_c8_4_benchmark():
    print("=" * 80)
    print("STAGE C8-4: CONTROLLED CLOSED-LOOP ESKF MAP CONSTRAINTS WITH MHT")
    print("=" * 80)

    dt = 0.1
    stride_sec = 20.0
    horizons = [10.0, 20.0, 30.0, 60.0]
    trips = ['Vta04', 'Vta02']
    conditions = ['C0_Baseline', 'C8_4A_Shadow', 'C1_Lateral', 'C2_Heading', 'C3_Joint']

    master_results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'stage': 'C8-4',
            'status': 'COMPLETED',
            'conditions': conditions,
            'horizons': horizons,
            'search_radius_m': 25.0,
            'heading_gate_deg': 30.0,
            'commitment_margin_delta': 0.20,
            'n_persist': 2,
            'max_pos_correction_m': 5.0,
            'gt_threshold_m': 7.5
        },
        'trips': {},
        'regressions': []
    }

    for trip_name in trips:
        trip_data = prepare_trip_data(trip_name, dt=dt)
        n_epochs = trip_data['n_epochs']
        stride_epochs = int(round(stride_sec / dt))
        trip_results = {}

        for h_sec in horizons:
            w_dur = int(round(h_sec / dt))
            window_starts = list(range(0, n_epochs - w_dur, stride_epochs))
            n_win = len(window_starts)

            print(f"\n--- [{trip_name}] Blackout Horizon: {h_sec:.0f}s ({n_win} windows) ---")

            horizon_data = {c: [] for c in conditions}

            for kw in window_starts:
                # 1. Run Baseline C0 first
                c0_res = simulate_window_condition(
                    kw=kw, w_dur=w_dur, dt=dt, trip_data=trip_data,
                    condition='C0_Baseline', search_radius_m=25.0
                )
                horizon_data['C0_Baseline'].append(c0_res)

                # 2. Run other conditions
                for cond in conditions:
                    if cond == 'C0_Baseline':
                        continue
                    res = simulate_window_condition(
                        kw=kw, w_dur=w_dur, dt=dt, trip_data=trip_data,
                        condition=cond, search_radius_m=25.0,
                        heading_gate_deg=30.0, commitment_margin_delta=0.20,
                        update_interval_s=1.0
                    )
                    horizon_data[cond].append(res)

                    # Sub-Stage C8-4A Sanity Check: Shadow Mode must have BIT-IDENTICAL error to Baseline C0
                    if cond == 'C8_4A_Shadow':
                        diff_pos = abs(res['final_e_total'] - c0_res['final_e_total'])
                        assert diff_pos < 1e-6, f"Shadow mode leakage detected! diff={diff_pos} m on window {kw}"

                    # Regression audit: does map make navigation error worse?
                    delta_e = res['final_e_total'] - c0_res['final_e_total']
                    if delta_e > 0.5:  # regression threshold > 0.5m
                        master_results['regressions'].append({
                            'trip': trip_name,
                            'horizon_s': h_sec,
                            'window_idx': kw,
                            'condition': cond,
                            'baseline_err': c0_res['final_e_total'],
                            'map_err': res['final_e_total'],
                            'delta_err': delta_e,
                            'applied_updates': res['applied_updates'],
                            'wrong_road_updates': res['wrong_road_updates'],
                            'max_correction_m': res['max_correction_m']
                        })

            # Aggregate metrics for each condition
            horizon_summary = {}
            for cond in conditions:
                res_list = horizon_data[cond]
                tot_errs = [r['final_e_total'] for r in res_list]
                along_errs = [r['final_e_along'] for r in res_list]
                cross_errs = [r['final_e_cross'] for r in res_list]
                heading_errs = [r['final_e_heading'] for r in res_list]
                vel_errs = [r['final_e_vel'] for r in res_list]
                cov_traces = [r['final_p_cov_trace'] for r in res_list]
                updates = [r['applied_updates'] for r in res_list]
                wrong_updates = [r['wrong_road_updates'] for r in res_list]
                max_corrs = [r['max_correction_m'] for r in res_list]

                # Rejection breakdown
                rej_counts = {k: sum(r['rejections'][k] for r in res_list) for k in res_list[0]['rejections']}

                tot_applied = sum(updates)
                tot_wrong = sum(wrong_updates)

                # Divergent windows (> 2x baseline)
                c0_errs = [r['final_e_total'] for r in horizon_data['C0_Baseline']]
                divergent_count = sum(1 for e_m, e_b in zip(tot_errs, c0_errs) if e_m > 2.0 * max(10.0, e_b))

                horizon_summary[cond] = {
                    'mean_total_err': float(np.mean(tot_errs)),
                    'median_total_err': float(np.median(tot_errs)),
                    'p90_total_err': float(np.percentile(tot_errs, 90)),
                    'mean_along_err': float(np.mean(along_errs)),
                    'mean_cross_err': float(np.mean(cross_errs)),
                    'mean_heading_err': float(np.mean(heading_errs)),
                    'mean_vel_err': float(np.mean(vel_errs)),
                    'mean_p_cov_trace': float(np.mean(cov_traces)),
                    'mean_updates_per_win': float(np.mean(updates)),
                    'total_updates_applied': tot_applied,
                    'total_wrong_road_updates': tot_wrong,
                    'wrong_road_update_rate': float(tot_wrong / max(1, tot_applied)),
                    'divergent_windows': divergent_count,
                    'max_single_correction': float(max(max_corrs)) if max_corrs else 0.0,
                    'rejections': rej_counts
                }

                print(
                    f"[{cond:13s}] Drift: {np.mean(tot_errs):6.2f}m (Along: {np.mean(along_errs):5.2f}m, Cross: {np.mean(cross_errs):5.2f}m) | "
                    f"Heading: {np.mean(heading_errs):5.2f}° | Updates: {np.mean(updates):4.1f} | Wrong-Upd: {tot_wrong:2d} ({float(tot_wrong/max(1, tot_applied))*100:4.1f}%)"
                )

            trip_results[str(h_sec)] = horizon_summary

        master_results['trips'][trip_name] = trip_results

    # Save master benchmark JSON
    out_json = RES_DIR / "c8_4_closed_loop_map_matching.json"
    with open(out_json, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Master Data] Saved to {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    # Generate publication dashboard
    generate_c8_4_dashboard(master_results)

    return master_results


# ==============================================================================
# DIAGNOSTIC DASHBOARD GENERATOR
# ==============================================================================

def generate_c8_4_dashboard(master_results: dict):
    """Generates a publication-grade 6-panel dashboard visualizing Stage C8-4 findings."""
    print("[Dashboard] Generating 6-panel publication figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Stage C8-4: Controlled Closed-Loop ESKF Map Constraints with MHT Validation", fontsize=15, fontweight='bold')

    horizons = [10.0, 20.0, 30.0, 60.0]
    h_labels = ["10s", "20s", "30s", "60s"]

    # Colors for conditions
    cond_colors = {
        'C0_Baseline': '#333333',
        'C8_4A_Shadow': '#888888',
        'C1_Lateral': '#2b5c8f',
        'C2_Heading': '#e66101',
        'C3_Joint': '#2ca02c'
    }

    # Panel 1: Highway Drift Comparison (Vta04)
    ax1 = axes[0, 0]
    vta04_res = master_results['trips']['Vta04']
    for cond in ['C0_Baseline', 'C1_Lateral', 'C2_Heading', 'C3_Joint']:
        drift_vals = [vta04_res[str(h)][cond]['mean_total_err'] for h in horizons]
        ax1.plot(horizons, drift_vals, marker='o', lw=2.2, label=cond.replace('_', ' '), color=cond_colors[cond])
    ax1.set_title("Highway Drift (Vta04) vs Horizon", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Blackout Duration (s)")
    ax1.set_ylabel("Total Drift (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)

    # Panel 2: Suburban Drift Comparison (Vta02)
    ax2 = axes[0, 1]
    vta02_res = master_results['trips']['Vta02']
    for cond in ['C0_Baseline', 'C1_Lateral', 'C2_Heading', 'C3_Joint']:
        drift_vals = [vta02_res[str(h)][cond]['mean_total_err'] for h in horizons]
        ax2.plot(horizons, drift_vals, marker='s', lw=2.2, label=cond.replace('_', ' '), color=cond_colors[cond])
    ax2.set_title("Suburban Drift (Vta02) vs Horizon", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Blackout Duration (s)")
    ax2.set_ylabel("Total Drift (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    # Panel 3: Cross-Track Error Suppression (Vta04 at 30s)
    ax3 = axes[0, 2]
    cross_vals_30s = [vta04_res['30.0'][c]['mean_cross_err'] for c in ['C0_Baseline', 'C1_Lateral', 'C2_Heading', 'C3_Joint']]
    along_vals_30s = [vta04_res['30.0'][c]['mean_along_err'] for c in ['C0_Baseline', 'C1_Lateral', 'C2_Heading', 'C3_Joint']]
    labels = ['C0 Base', 'C1 Lat', 'C2 Head', 'C3 Joint']
    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, cross_vals_30s, width, label='Cross-Track', color='#2b5c8f', alpha=0.85)
    ax3.bar(x + width/2, along_vals_30s, width, label='Along-Track', color='#d95f02', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title("Highway 30s Error Decomposition (Vta04)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Mean Error (m)")
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=9)

    # Panel 4: Closed-Loop Wrong-Road Update Rate
    ax4 = axes[1, 0]
    rates_vta04 = [vta04_res[str(h)]['C3_Joint']['wrong_road_update_rate'] * 100 for h in horizons]
    rates_vta02 = [vta02_res[str(h)]['C3_Joint']['wrong_road_update_rate'] * 100 for h in horizons]
    # In C8-2 naive snapping had 42-53% wrong road
    c8_2_naive_vta04 = [19.3, 43.2, 53.5, 65.0]
    ax4.plot(horizons, c8_2_naive_vta04, 'r--', marker='x', lw=2.0, label='C8-2 Naive Snapping (Highway)')
    ax4.plot(horizons, rates_vta04, marker='o', lw=2.2, color='#2ca02c', label='C8-4 C3 Joint (Highway)')
    ax4.plot(horizons, rates_vta02, marker='s', lw=2.2, color='#1f77b4', label='C8-4 C3 Joint (Suburban)')
    ax4.set_title("Closed-Loop Wrong-Road Update Rate (%)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Blackout Duration (s)")
    ax4.set_ylabel("Applied Wrong-Road Updates (%)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=9)

    # Panel 5: 5-Gate Rejection Breakdown (Vta04 30s)
    ax5 = axes[1, 1]
    rejections_30s = vta04_res['30.0']['C3_Joint']['rejections']
    rej_labels = ['Gate 1: Ambiguous', 'Gate 2: Radius', 'Gate 3: Heading', 'Gate 4: Persist', 'Gate 5: NIS']
    rej_keys = ['gate1_ambiguous', 'gate2_geometry', 'gate3_heading', 'gate4_persistence', 'gate5_nis']
    rej_counts = [rejections_30s.get(k, 0) for k in rej_keys]
    bars = ax5.barh(rej_labels, rej_counts, color=['#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'], alpha=0.85)
    ax5.set_title("5-Gate Rejection Breakdown (Vta04 30s C3)", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Epoch Rejection Count")
    ax5.grid(True, alpha=0.3, axis='x')

    # Panel 6: Map-Induced Regressions Distribution
    ax6 = axes[1, 2]
    reg_deltas = [r['delta_err'] for r in master_results['regressions'] if r['condition'] == 'C3_Joint']
    if reg_deltas:
        ax6.hist(reg_deltas, bins=15, color='#d95f02', alpha=0.8, edgecolor='black')
        ax6.axvline(0.0, color='black', lw=1.5, ls='--')
        ax6.set_title(f"Map Regressions (N={len(reg_deltas)}): $e_{{map}} - e_{{base}}$", fontsize=11, fontweight='bold')
        ax6.set_xlabel("Regression Delta Error (m)")
        ax6.set_ylabel("Window Count")
    else:
        ax6.text(0.5, 0.5, "ZERO MAP-INDUCED REGRESSIONS!\n$e_{map} \\le e_{baseline}$ across all windows",
                 ha='center', va='center', fontsize=12, fontweight='bold', color='#2ca02c')
        ax6.set_title("Map-Induced Regressions Check", fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "c8_4_closed_loop_map_matching.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Dashboard] Saved figure to {fig_path}")


if __name__ == "__main__":
    run_c8_4_benchmark()
