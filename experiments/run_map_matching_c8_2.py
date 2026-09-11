"""
SIH26168 - Stage C8-2: Controlled ESKF Map Constraints Benchmark
Module: experiments/run_map_matching_c8_2.py

Evaluates the navigation performance of incorporating OpenStreetMap (OSM) vector road
constraints into the 15-state 3D Error-State Kalman Filter (ESKF3D) under controlled
blackout conditions.

Four Controlled Conditions:
1. Baseline: Frozen C7-B1-D Strict Freeze (No map)
   - 15-state 3D ESKF
   - Non-Holonomic Constraints (NHC, sigma=0.5 m/s)
   - Bounded Causal Adaptive Covariance (BCAC, w_b=10s)
   - Deployable Standstill ZUPT (0.8s persistence)
   - Soft Longitudinal Acceleration Bounds with Strict Attitude Freeze (K_theta = 0)
2. Condition A: Map Lateral Only (r_perp = d_perp - mu_lane, H_perp, R_perp = (2.5 m)^2)
3. Condition B: Map Heading Only (r_psi = wrap(psi_road - psi_hat), H_psi, R_psi = (0.087 rad)^2)
4. Condition C: Joint Lateral + Heading (H_joint, R_joint)

Evaluation Protocol:
- Trips: Suburban control Vta02 (54 windows) and continuous highway Vta04 (9 windows).
- Horizons: 10s, 20s, 30s, 60s.
- Stride: 20.0s.
- Multi-layer gating: R <= 25.0m, |Delta psi| <= 30.0 deg, NIS <= 9.210 (99%).
- Telemetry tracked: total drift, along-track, cross-track, heading error, velocity error,
  and wrong-road association rate (distance to ground truth > 7.5m).
"""

import sys
import time
import json
from pathlib import Path
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
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex
from src.navigation.map_constraints import MapConstraintManager

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


def prepare_trip_with_map(trip_name: str, dt: float = 0.1) -> dict:
    """Loads trip telemetry, IMU data, and builds the OSM road network index."""
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

    # Load and build OSM road network index
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


def run_single_window_map(
    kw: int,
    w_dur: int,
    dt: float,
    trip_data: dict,
    condition: str = 'Baseline',
    record_timeseries: bool = False
) -> dict:
    """
    Executes a single blackout window under one of the 4 controlled conditions.
    """
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    heading = trip_data['heading']
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

    # B1 Strict Freeze constraint (identical across all conditions)
    b1_constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )

    # Map Constraint Manager
    map_mgr = MapConstraintManager(
        road_index=road_index,
        search_radius_m=25.0,
        heading_gate_deg=30.0,
        sigma_lane_m=2.5,
        sigma_psi_deg=5.0,
        lane_offset_m=-0.90,
        nis_gate_2dof=9.210,
        nis_gate_1dof=6.635,
        update_interval_sec=1.0,
        ambiguity_margin_m=3.0
    )

    # Telemetry
    map_updates_count = 0
    map_gated_count = 0
    map_wrong_road_count = 0
    lateral_innovations = []
    heading_innovations = []

    ts_time = []
    ts_total_drift = []
    ts_along_err = []
    ts_cross_err = []
    ts_pos_e = []
    ts_pos_n = []

    t_start = kw * dt

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax = acc_v[step_k, 0]
        ay = acc_v[step_k, 1]
        az = acc_v[step_k, 2]
        gx = gyro_v[step_k, 0]
        gy = gyro_v[step_k, 1]
        gz = gyro_v[step_k, 2]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 1. NHC update
        nhc.update_eskf(eskf)

        # 2. Deployable ZUPT update
        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        # 3. Soft Acceleration Bounds update
        b1_constraint.update_eskf(eskf, dt=dt)

        # 4. Map Constraint update (Conditions A, B, C)
        if condition != 'Baseline':
            res_map = None
            if condition == 'Condition_A':  # Lateral only
                res_map = map_mgr.update_lateral(eskf, t_now=t_now)
            elif condition == 'Condition_B':  # Heading only
                res_map = map_mgr.update_heading(eskf, t_now=t_now)
            elif condition == 'Condition_C':  # Joint lateral + heading
                res_map = map_mgr.update_joint(eskf, t_now=t_now)

            if res_map is not None:
                if res_map['active']:
                    map_updates_count += 1
                    if 'innovation' in res_map:
                        if res_map['type'] == 'lateral':
                            lateral_innovations.append(abs(float(res_map['innovation'])))
                        elif res_map['type'] == 'heading':
                            heading_innovations.append(abs(float(np.degrees(res_map['innovation']))))
                    elif 'innovation_perp' in res_map:
                        lateral_innovations.append(abs(float(res_map['innovation_perp'])))
                        heading_innovations.append(abs(float(np.degrees(res_map['innovation_psi']))))

                    # Safety check: Verify matched candidate against VBOX ground truth
                    cand = res_map['candidate']
                    seg_idx = cand['segment_idx']
                    seg = road_index.segments[seg_idx]
                    p1 = seg['p1']
                    p2 = seg['p2']
                    gt_pos = np.array([gt_e[step_k], gt_n[step_k]])

                    # Distance from true vehicle position to the matched road segment
                    seg_diff = p2 - p1
                    seg_len_sq = float(np.sum(seg_diff**2))
                    ap = gt_pos - p1
                    t_proj = np.clip(float(np.dot(ap, seg_diff)) / (seg_len_sq + 1e-12), 0.0, 1.0)
                    proj_gt = p1 + t_proj * seg_diff
                    dist_gt_to_matched = float(np.linalg.norm(gt_pos - proj_gt))

                    # Flag as wrong-road association if matched segment is > 7.5 m away from true path
                    if dist_gt_to_matched > 7.5:
                        map_wrong_road_count += 1
                else:
                    if res_map.get('reason') in ('nis_gated', 'no_valid_candidate'):
                        map_gated_count += 1

        if record_timeseries:
            st = eskf.get_state()
            p_curr = st['pos_n'][:2]
            gt_p_curr = np.array([gt_e[step_k + 1], gt_n[step_k + 1]])
            curr_h_rad = np.radians(float(heading[step_k + 1]))
            e_vec = p_curr - gt_p_curr
            u_fwd = np.array([np.sin(curr_h_rad), np.cos(curr_h_rad)])
            u_lat = np.array([-np.cos(curr_h_rad), np.sin(curr_h_rad)])

            ts_time.append(t_now)
            ts_total_drift.append(float(np.linalg.norm(e_vec)))
            ts_along_err.append(float(abs(np.dot(e_vec, u_fwd))))
            ts_cross_err.append(float(abs(np.dot(e_vec, u_lat))))
            ts_pos_e.append(float(p_curr[0]))
            ts_pos_n.append(float(p_curr[1]))

    # Final State & Metrics
    st_end = eskf.get_state()
    gt_p_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur]])
    gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur]])
    end_h_deg = float(heading[kw + w_dur])
    end_h_rad = np.radians(end_h_deg)

    e_vec_end = st_end['pos_n'][:2] - gt_p_end
    u_fwd_end = np.array([np.sin(end_h_rad), np.cos(end_h_rad)])
    u_lat_end = np.array([-np.cos(end_h_rad), np.sin(end_h_rad)])

    final_yaw = st_end['yaw_deg']
    final_h_err = abs(float((final_yaw - end_h_deg + 180.0) % 360.0 - 180.0))

    # Travel distance over window
    p_start_gt = np.array([gt_e[kw], gt_n[kw]])
    dist_traveled = float(np.linalg.norm(gt_p_end - p_start_gt))
    total_drift = float(np.linalg.norm(e_vec_end))
    drift_pct = (total_drift / max(dist_traveled, 1.0)) * 100.0

    res = {
        'total_drift_m': total_drift,
        'along_track_m': float(abs(np.dot(e_vec_end, u_fwd_end))),
        'cross_track_m': float(abs(np.dot(e_vec_end, u_lat_end))),
        'along_track_signed_m': float(np.dot(e_vec_end, u_fwd_end)),
        'cross_track_signed_m': float(np.dot(e_vec_end, u_lat_end)),
        'vel_err_ms': float(np.linalg.norm(st_end['vel_n'][:2] - gt_v_end)),
        'final_heading_err_deg': final_h_err,
        'dist_traveled_m': dist_traveled,
        'drift_pct': drift_pct,
        'si_target_met': bool(drift_pct < 10.0),
        'map_updates_count': map_updates_count,
        'map_gated_count': map_gated_count,
        'map_wrong_road_count': map_wrong_road_count,
        'lateral_innov_mean': float(np.mean(lateral_innovations)) if lateral_innovations else 0.0,
        'heading_innov_mean': float(np.mean(heading_innovations)) if heading_innovations else 0.0
    }

    if record_timeseries:
        res['timeseries'] = {
            'time': ts_time,
            'total_drift': ts_total_drift,
            'along_err': ts_along_err,
            'cross_err': ts_cross_err,
            'pos_e': ts_pos_e,
            'pos_n': ts_pos_n
        }

    return res


def evaluate_trip_map_matching(
    trip_data: dict,
    conditions: list[str],
    horizons: list[float] = [10.0, 20.0, 30.0, 60.0],
    stride_s: float = 20.0,
    dt: float = 0.1
) -> dict:
    """
    Sweeps all conditions and horizons across a trip.
    """
    stride_k = int(round(stride_s / dt))
    results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, trip_data['n_epochs'] - w_dur - 1, stride_k))
        n_win = len(window_starts)

        print(f"\n[{trip_data['trip_name']}] Evaluating horizon {h_key} ({n_win} windows)...")

        results[h_key] = {
            'n_windows': n_win,
            'conditions': {}
        }

        for c in conditions:
            win_outputs = []
            for kw in window_starts:
                out = run_single_window_map(kw, w_dur, dt, trip_data, condition=c, record_timeseries=False)
                win_outputs.append(out)

            drifts = [w['total_drift_m'] for w in win_outputs]
            long_errs = [w['along_track_m'] for w in win_outputs]
            lat_errs = [w['cross_track_m'] for w in win_outputs]
            vel_errs = [w['vel_err_ms'] for w in win_outputs]
            h_errs = [w['final_heading_err_deg'] for w in win_outputs]

            total_map_updates = sum(w['map_updates_count'] for w in win_outputs)
            total_wrong_roads = sum(w['map_wrong_road_count'] for w in win_outputs)
            wrong_road_rate = (total_wrong_roads / max(total_map_updates, 1)) * 100.0 if total_map_updates > 0 else 0.0

            si_met_count = sum(1 for w in win_outputs if w['si_target_met'])
            si_met_pct = (si_met_count / max(n_win, 1)) * 100.0

            results[h_key]['conditions'][c] = {
                'drift_mean': float(np.mean(drifts)),
                'drift_median': float(np.median(drifts)),
                'drift_p90': float(np.percentile(drifts, 90)),
                'along_track_mean': float(np.mean(long_errs)),
                'along_track_median': float(np.median(long_errs)),
                'cross_track_mean': float(np.mean(lat_errs)),
                'cross_track_median': float(np.median(lat_errs)),
                'vel_err_mean': float(np.mean(vel_errs)),
                'vel_err_median': float(np.median(vel_errs)),
                'heading_err_mean': float(np.mean(h_errs)),
                'heading_err_median': float(np.median(h_errs)),
                'total_map_updates': int(total_map_updates),
                'total_wrong_road_updates': int(total_wrong_roads),
                'wrong_road_rate_pct': float(wrong_road_rate),
                'si_met_pct': float(si_met_pct)
            }

            print(f"  {c:<15}: Drift={np.mean(drifts):6.2f}m (Med={np.median(drifts):6.2f}m) | "
                  f"Along={np.mean(long_errs):6.2f}m | Cross={np.mean(lat_errs):6.2f}m | "
                  f"HeadErr={np.mean(h_errs):5.2f}° | WrongRoad={wrong_road_rate:4.1f}%")

    return results


def run_benchmark():
    """Main execution script for Stage C8-2."""
    print("=" * 80)
    print("STAGE C8-2: CONTROLLED ESKF MAP CONSTRAINTS BENCHMARK")
    print("Conditions: Baseline | Condition A (Lateral) | Condition B (Heading) | Condition C (Joint)")
    print("=" * 80)

    conditions = ['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']
    horizons = [10.0, 20.0, 30.0, 60.0]

    # Ingest data and maps
    vta04_data = prepare_trip_with_map('Vta04')
    vta02_data = prepare_trip_with_map('Vta02')

    t0 = time.time()
    vta04_results = evaluate_trip_map_matching(vta04_data, conditions, horizons=horizons)
    vta02_results = evaluate_trip_map_matching(vta02_data, conditions, horizons=horizons)
    elapsed = time.time() - t0
    print(f"\nExecution completed in {elapsed:.1f} seconds.")

    # Record representative timeseries for 30s outage on Vta04 and Vta02
    print("\nRecording representative 30s trajectory timeseries...")
    ts_vta04 = {}
    for c in conditions:
        ts_vta04[c] = run_single_window_map(50, int(round(30.0 / 0.1)), 0.1, vta04_data, condition=c, record_timeseries=True)

    ts_vta02 = {}
    for c in conditions:
        ts_vta02[c] = run_single_window_map(50, int(round(30.0 / 0.1)), 0.1, vta02_data, condition=c, record_timeseries=True)

    # Master results dict
    master_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'stage': 'C8-2',
            'status': 'COMPLETED',
            'conditions': conditions,
            'horizons': horizons,
            'parameters': {
                'search_radius_m': 25.0,
                'heading_gate_deg': 30.0,
                'sigma_lane_m': 2.5,
                'sigma_psi_deg': 5.0,
                'lane_offset_m': -0.90,
                'nis_gate_2dof': 9.210,
                'update_interval_sec': 1.0
            }
        },
        'Vta04': vta04_results,
        'Vta02': vta02_results
    }

    # Save JSON deliverable
    json_path = RES_DIR / "c8_2_map_matching_benchmark.json"
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\nSaved structured benchmark results to: {json_path}")

    # Generate 6-Panel Diagnostic Figure
    generate_figures(master_results, ts_vta04, ts_vta02, vta04_data, vta02_data)

    return master_results


def generate_figures(results: dict, ts_vta04: dict, ts_vta02: dict, vta04_data: dict, vta02_data: dict):
    """Generates the 6-panel publication diagnostic figure."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Stage C8-2: Controlled ESKF Map Constraints Benchmark\n(Suburban Vta02 & Continuous Highway Vta04)", fontsize=16, fontweight='bold')

    horizons = ['10s', '20s', '30s', '60s']
    cond_labels = {
        'Baseline': ('Baseline (No Map)', '#7f7f7f', '--'),
        'Condition_A': ('Cond A (Lateral Only)', '#1f77b4', '-'),
        'Condition_B': ('Cond B (Heading Only)', '#ff7f0e', '-.'),
        'Condition_C': ('Cond C (Joint Lat+Head)', '#2ca02c', '-')
    }

    # 1. Panel 1: Total Drift Comparison across Horizons (Highway Vta04)
    ax1 = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.2
    for i, c in enumerate(['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']):
        drifts = [results['Vta04'][h]['conditions'][c]['drift_mean'] for h in horizons]
        ax1.bar(x + (i - 1.5) * width, drifts, width, label=cond_labels[c][0], color=cond_labels[c][1], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_ylabel("Mean Total Drift (m)", fontweight='bold')
    ax1.set_title("Highway Vta04: Total Drift vs. Horizon", fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Panel 2: Cross-Track Error Suppression (Highway Vta04)
    ax2 = axes[0, 1]
    for c in ['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']:
        cross_errs = [results['Vta04'][h]['conditions'][c]['cross_track_mean'] for h in horizons]
        ax2.plot(horizons, cross_errs, marker='o', linewidth=2.0, label=cond_labels[c][0], color=cond_labels[c][1], linestyle=cond_labels[c][2])
    ax2.set_ylabel("Mean Cross-Track Error (m)", fontweight='bold')
    ax2.set_title("Highway Vta04: Cross-Track Suppression", fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Panel 3: Along-Track Error Comparison (Highway Vta04)
    ax3 = axes[0, 2]
    for c in ['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']:
        along_errs = [results['Vta04'][h]['conditions'][c]['along_track_mean'] for h in horizons]
        ax3.plot(horizons, along_errs, marker='s', linewidth=2.0, label=cond_labels[c][0], color=cond_labels[c][1], linestyle=cond_labels[c][2])
    ax3.set_ylabel("Mean Along-Track Error (m)", fontweight='bold')
    ax3.set_title("Highway Vta04: Along-Track Error (Strict Null Space)", fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Panel 4: Heading Error Comparison (Highway Vta04)
    ax4 = axes[1, 0]
    for c in ['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']:
        h_errs = [results['Vta04'][h]['conditions'][c]['heading_err_mean'] for h in horizons]
        ax4.plot(horizons, h_errs, marker='^', linewidth=2.0, label=cond_labels[c][0], color=cond_labels[c][1], linestyle=cond_labels[c][2])
    ax4.set_ylabel("Mean Heading Error (deg)", fontweight='bold')
    ax4.set_title("Highway Vta04: Heading Error vs. Horizon", fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Panel 5: Suburban Vta02 Total Drift Comparison
    ax5 = axes[1, 1]
    for i, c in enumerate(['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']):
        drifts_v2 = [results['Vta02'][h]['conditions'][c]['drift_mean'] for h in horizons]
        ax5.bar(x + (i - 1.5) * width, drifts_v2, width, label=cond_labels[c][0], color=cond_labels[c][1], alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(horizons)
    ax5.set_ylabel("Mean Total Drift (m)", fontweight='bold')
    ax5.set_title("Suburban Vta02: Total Drift vs. Horizon", fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Panel 6: 30s Outage Trajectory Comparison (Highway Vta04 Window 50)
    ax6 = axes[1, 2]
    # Ground truth trajectory
    kw = 50
    w_dur = int(round(30.0 / 0.1))
    gt_e = vta04_data['gt_e'][kw : kw + w_dur + 1]
    gt_n = vta04_data['gt_n'][kw : kw + w_dur + 1]
    ax6.plot(gt_e - gt_e[0], gt_n - gt_n[0], 'k-', linewidth=2.5, label='VBOX Ground Truth')

    for c in ['Baseline', 'Condition_A', 'Condition_B', 'Condition_C']:
        ts = ts_vta04[c]['timeseries']
        pe = np.array(ts['pos_e']) - gt_e[0]
        pn = np.array(ts['pos_n']) - gt_n[0]
        ax6.plot(pe, pn, color=cond_labels[c][1], linestyle=cond_labels[c][2], linewidth=1.8, label=cond_labels[c][0])

    ax6.set_xlabel("Relative East (m)", fontweight='bold')
    ax6.set_ylabel("Relative North (m)", fontweight='bold')
    ax6.set_title("Vta04: 30s Trajectory Overlay (Window 50)", fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = FIG_DIR / "c8_2_map_matching_benchmark.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication diagnostic dashboard to: {fig_path}")


if __name__ == "__main__":
    run_benchmark()
