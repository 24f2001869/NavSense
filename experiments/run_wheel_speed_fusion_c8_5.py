"""
SIH26168 - Stage C8-5: Controlled Chassis Wheel-Speed Odometry Fusion
Module: experiments/run_wheel_speed_fusion_c8_5.py

Evaluates whether integrating chassis wheel-speed odometry (CAN-bus rear non-driven
wheel speeds) resolves along-track dead reckoning drift, bounds position covariance
growth, preserves NIS discrimination at long blackout horizons, and eliminates
map-induced regressions in closed-loop map matching.

Evaluated Conditions:
- C0: Frozen Baseline (C7-B1-D Strict Freeze: ESKF + NHC + BCAC + ZUPT + Strict Attitude Freeze, No wheel, No map)
- C1: Velocity-Only Fusion (ESKF + Forward Wheel Speed v_x^v = v_wheel, NHC disabled)
- C2: Velocity + NHC Fusion (Full 3D Body Velocity [v_wheel, 0, 0]^T)
- C3: Velocity + NHC + MHT Map (Full 3D Body Velocity + C8-4 5-Gate MHT Map Constraints)
- C_MapOnly: Counterfactual comparison with C8-4 Joint Map (No wheel speed)

Evaluated across 54 suburban windows (Vta02) and 9 highway windows (Vta04)
across 10s, 20s, 30s, and 60s blackout horizons.
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
from src.navigation.wheel_odometry import ChassisWheelSpeedFusion, compute_wheel_speed_from_can
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex, wrap_angle_rad
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


def prepare_trip_data_c8_5(trip_name: str, dt: float = 0.1) -> dict:
    """Loads trip telemetry, IMU, CAN wheel speeds, and builds OSM road network index."""
    df_p, df_v = load_trip(trip_name)
    n = min(len(df_p), len(df_v))
    df_p = df_p.iloc[:n].copy()
    df_v = df_v.iloc[:n].copy()

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    heading = df_v['veh_heading_deg'].values

    # Extract non-driven rear wheel speeds from CAN
    wheel_rl = df_v['wheel_rl_rads'].values
    wheel_rr = df_v['wheel_rr_rads'].values
    wheel_speed = 0.5 * (wheel_rl + wheel_rr) * 0.2766

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
        'wheel_speed': wheel_speed,
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


def simulate_window_c8_5(
    kw: int,
    w_dur: int,
    dt: float,
    trip_data: dict,
    condition: str,
    search_radius_m: float = 25.0,
    heading_gate_deg: float = 30.0,
    commitment_margin_delta: float = 0.20,
    update_interval_s: float = 1.0,
    gt_threshold_m: float = 7.5
) -> Dict[str, Any]:
    """
    Simulates a blackout window under a specified C8-5 condition.
    """
    acc_v = trip_data['acc_v']
    gyro_v = trip_data['gyro_v']
    heading = trip_data['heading']
    speed = trip_data['speed']
    wheel_speed = trip_data['wheel_speed']
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
    b1_constraint = SoftLongitudinalAccelerationConstraint(
        a_soft=3.0, sigma_a_meas=0.50, mode="coupled",
        decouple_attitude=True, strict_attitude_freeze=True
    )
    wheel_fusion = ChassisWheelSpeedFusion(sigma_wheel=0.20, sigma_lat=0.50, sigma_vert=0.50)

    # Initialize MHT Map Manager if map condition active
    map_mgr = None
    has_map = condition in ['C3_Velocity_NHC_Map', 'C_MapOnly']
    if has_map:
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
        'wheel_slip_events': 0,
        'pos_cov_trace_history': []
    }

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]
        cur_spd = speed[step_k]
        cur_wheel_spd = wheel_speed[step_k]

        # 1. ESKF strapdown propagation
        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # 2. Velocity / NHC update depending on condition
        if condition == 'C0_Baseline' or condition == 'C_MapOnly':
            # Pure NHC (No wheel speed)
            nhc.update_eskf(eskf)

        elif condition == 'C1_Velocity_Only':
            # 1-DOF Forward Wheel Speed Only (NHC disabled)
            res_w = wheel_fusion.update_eskf_forward_velocity(eskf, cur_wheel_spd, apply_nis_gate=True)
            if res_w.get('gated_out', False):
                telemetry['wheel_slip_events'] += 1

        elif condition in ['C2_Velocity_NHC', 'C3_Velocity_NHC_Map']:
            # Coupled 3-DOF Body Velocity (Forward Wheel Speed + Lateral/Vertical NHC)
            res_w = wheel_fusion.update_eskf_3d_velocity(eskf, cur_wheel_spd, apply_nis_gate=True)
            if res_w.get('gated_out', False):
                telemetry['wheel_slip_events'] += 1

        # 3. ZUPT update
        k_start = max(0, step_k - 10)
        det_res = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_res['is_stationary']:
            zupt.update_eskf(eskf)

        # 4. B1 Strict Attitude Freeze
        b1_constraint.update_eskf(eskf)

        # Track position covariance trace
        tr_pos = float(np.trace(eskf.P[0:2, 0:2]))
        telemetry['pos_cov_trace_history'].append(tr_pos)

        # 5. Map Constraint Evaluation (1 Hz cadence)
        if map_mgr is not None and (t_now - last_map_time) >= (update_interval_s - 1e-4):
            last_map_time = t_now
            telemetry['total_queries'] += 1

            res = map_mgr.evaluate_and_update(
                eskf=eskf,
                t_now=t_now,
                mode='joint',
                speed_ms=cur_spd,
                gyro_z_rads=gz,
                shadow_mode=False
            )

            if res['active']:
                telemetry['applied_updates'] += 1
                telemetry['corrections_pos_norm'].append(float(np.linalg.norm(res['delta_pos'])))
                telemetry['nis_values'].append(float(res['nis']))

                cand_seg = res['candidate']['segment_idx']
                gt_p = np.array([gt_e[step_k], gt_n[step_k]])
                d_gt = compute_segment_gt_distance(cand_seg, gt_p, road_index)
                if d_gt > gt_threshold_m:
                    telemetry['wrong_road_updates'] += 1
            else:
                reason = res.get('reason', 'unknown')
                if reason in telemetry['rejections']:
                    telemetry['rejections'][reason] += 1

    # End of blackout: evaluate final navigation errors
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

    # Distance travelled during outage
    dist_travelled = float(np.sum(speed[kw : kw + w_dur] * dt))
    drift_pct = float((e_total / (dist_travelled + 1e-6)) * 100.0)

    final_pos_cov_trace = float(np.sum(st_final['cov_diag'][0:2]))
    max_corr = float(max(telemetry['corrections_pos_norm'])) if telemetry['corrections_pos_norm'] else 0.0

    return {
        'condition': condition,
        'window_idx': kw,
        'e_total': e_total,
        'e_along': e_along,
        'e_cross': e_cross,
        'e_heading': e_heading,
        'e_vel': e_vel,
        'dist_travelled': dist_travelled,
        'drift_pct': drift_pct,
        'final_pos_cov_trace': final_pos_cov_trace,
        'max_corr': max_corr,
        'telemetry': telemetry
    }


def run_c8_5_benchmark() -> Dict[str, Any]:
    """
    Executes the master Stage C8-5 benchmark across all trips, windows, and conditions.
    """
    print("=" * 80)
    print("SIH26168 - Stage C8-5: Controlled Chassis Wheel-Speed Odometry Fusion")
    print("=" * 80)

    trips = ['Vta04', 'Vta02']
    horizons = [10, 20, 30, 60]
    dt = 0.1

    conditions = [
        'C0_Baseline',
        'C1_Velocity_Only',
        'C2_Velocity_NHC',
        'C3_Velocity_NHC_Map',
        'C_MapOnly'
    ]

    all_results = {}

    for trip_name in trips:
        trip_data = prepare_trip_data_c8_5(trip_name, dt=dt)
        n_epochs = trip_data['n_epochs']

        # Match exact window stride from C8-4
        if trip_name == 'Vta04':
            window_stride_epochs = 150   # 15s stride -> 9 windows
        else:
            window_stride_epochs = 200   # 20s stride -> ~54 windows

        trip_results = {}

        for h in horizons:
            w_dur = int(round(h / dt))
            max_start = n_epochs - w_dur
            window_starts = list(range(0, max_start, window_stride_epochs))

            print(f"\n[{trip_name}] Running Horizon {h}s ({len(window_starts)} windows)...")

            h_res = {cond: [] for cond in conditions}

            for kw in window_starts:
                cond_runs = {}
                for cond in conditions:
                    res = simulate_window_c8_5(
                        kw=kw,
                        w_dur=w_dur,
                        dt=dt,
                        trip_data=trip_data,
                        condition=cond
                    )
                    cond_runs[cond] = res
                    h_res[cond].append(res)

            # Summarize metrics for horizon
            h_summary = {}
            for cond in conditions:
                runs = h_res[cond]
                e_totals = [r['e_total'] for r in runs]
                e_alongs = [r['e_along'] for r in runs]
                e_crosses = [r['e_cross'] for r in runs]
                e_headings = [r['e_heading'] for r in runs]
                e_vels = [r['e_vel'] for r in runs]
                drift_pcts = [r['drift_pct'] for r in runs]
                cov_traces = [r['final_pos_cov_trace'] for r in runs]
                max_corrs = [r['max_corr'] for r in runs]

                app_upds = [r['telemetry']['applied_updates'] for r in runs]
                wrong_upds = [r['telemetry']['wrong_road_updates'] for r in runs]
                tot_app = sum(app_upds)
                tot_wrong = sum(wrong_upds)
                wrong_pct = float((tot_wrong / tot_app * 100.0)) if tot_app > 0 else 0.0

                slip_evts = sum([r['telemetry']['wheel_slip_events'] for r in runs])

                # Check regressions vs C0 baseline
                regressions = []
                c0_runs = h_res['C0_Baseline']
                for idx_w in range(len(runs)):
                    e_c0 = c0_runs[idx_w]['e_total']
                    e_curr = runs[idx_w]['e_total']
                    if e_curr > (e_c0 + 0.5):
                        regressions.append({
                            'window_idx': window_starts[idx_w],
                            'e_baseline': e_c0,
                            'e_cond': e_curr,
                            'diff': e_curr - e_c0
                        })

                h_summary[cond] = {
                    'mean_drift_m': float(np.mean(e_totals)),
                    'median_drift_m': float(np.median(e_totals)),
                    'mean_along_m': float(np.mean(e_alongs)),
                    'mean_cross_m': float(np.mean(e_crosses)),
                    'mean_heading_deg': float(np.mean(e_headings)),
                    'mean_vel_err_ms': float(np.mean(e_vels)),
                    'mean_drift_pct': float(np.mean(drift_pcts)),
                    'mean_cov_trace': float(np.mean(cov_traces)),
                    'mean_applied_upd': float(np.mean(app_upds)),
                    'wrong_road_upd_pct': wrong_pct,
                    'max_single_corr_m': float(max(max_corrs)) if max_corrs else 0.0,
                    'num_regressions': len(regressions),
                    'regression_rate_pct': float(len(regressions) / len(runs) * 100.0),
                    'regressions': regressions,
                    'total_slip_events': slip_evts
                }

                print(f"  {cond:22s}: Drift={h_summary[cond]['mean_drift_m']:7.2f}m "
                      f"(Along={h_summary[cond]['mean_along_m']:6.2f}m, Cross={h_summary[cond]['mean_cross_m']:6.2f}m) "
                      f"Head={h_summary[cond]['mean_heading_deg']:5.1f}° "
                      f"CovTr={h_summary[cond]['mean_cov_trace']:7.1f}m² "
                      f"Regress={len(regressions)}/{len(runs)}")

            trip_results[str(h)] = {
                'summary': h_summary,
                'raw_runs': {c: [
                    {k: v for k, v in r.items() if k != 'telemetry' or k == 'telemetry' and False}
                    for r in h_res[c]
                ] for c in conditions}
            }

        all_results[trip_name] = trip_results

    # Save master benchmark JSON
    json_path = RES_DIR / "c8_5_wheel_speed_fusion.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] Structured deliverable saved to {json_path}")

    # Generate Publication Diagnostic Dashboard
    generate_c8_5_dashboard(all_results)

    return all_results


def generate_c8_5_dashboard(results: dict):
    """Generates a 6-panel publication-grade diagnostic dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stage C8-5: Chassis Wheel-Speed Odometry Fusion Diagnostic Dashboard", fontsize=16, fontweight='bold', y=0.98)

    horizons = [10, 20, 30, 60]
    palette = {
        'C0_Baseline': '#7f7f7f',
        'C1_Velocity_Only': '#1f77b4',
        'C2_Velocity_NHC': '#2ca02c',
        'C3_Velocity_NHC_Map': '#d62728',
        'C_MapOnly': '#ff7f0e'
    }
    labels = {
        'C0_Baseline': 'C0 Baseline (No Wheel, No Map)',
        'C1_Velocity_Only': 'C1 Fwd Wheel Speed Only',
        'C2_Velocity_NHC': 'C2 3D Wheel + NHC',
        'C3_Velocity_NHC_Map': 'C3 3D Wheel + NHC + Map',
        'C_MapOnly': 'C_Map C8-4 Map Only'
    }

    # Panel 1: Highway Vta04 Total Drift vs Horizon
    ax1 = axes[0, 0]
    for cond in palette:
        drifts = [results['Vta04'][str(h)]['summary'][cond]['mean_drift_m'] for h in horizons]
        ax1.plot(horizons, drifts, marker='o', lw=2.2, color=palette[cond], label=labels[cond])
    ax1.set_title("1. Highway Vta04: 2D Drift vs Outage Horizon", fontweight='bold')
    ax1.set_xlabel("Outage Duration (s)")
    ax1.set_ylabel("Mean Position Drift (m)")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper left')

    # Panel 2: Suburban Vta02 Total Drift vs Horizon
    ax2 = axes[0, 1]
    for cond in palette:
        drifts = [results['Vta02'][str(h)]['summary'][cond]['mean_drift_m'] for h in horizons]
        ax2.plot(horizons, drifts, marker='s', lw=2.2, color=palette[cond], label=labels[cond])
    ax2.set_title("2. Suburban Vta02: 2D Drift vs Outage Horizon", fontweight='bold')
    ax2.set_xlabel("Outage Duration (s)")
    ax2.set_ylabel("Mean Position Drift (m)")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Along-Track vs Cross-Track Error Breakdown at 30s
    ax3 = axes[0, 2]
    cond_keys = list(palette.keys())
    x = np.arange(len(cond_keys))
    width = 0.35

    along_vta04 = [results['Vta04']['30']['summary'][c]['mean_along_m'] for c in cond_keys]
    cross_vta04 = [results['Vta04']['30']['summary'][c]['mean_cross_m'] for c in cond_keys]

    ax3.bar(x - width/2, along_vta04, width, label='Along-Track Error (m)', color='#3498db', alpha=0.85)
    ax3.bar(x + width/2, cross_vta04, width, label='Cross-Track Error (m)', color='#e74c3c', alpha=0.85)
    ax3.set_title("3. Vta04 30s Error Decomposition (Along vs Cross)", fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace('_', '\n') for c in cond_keys], fontsize=8)
    ax3.set_ylabel("Error (m)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Panel 4: Position Covariance Trace Tr(P_pp) Evolution
    ax4 = axes[1, 0]
    for cond in palette:
        covs = [results['Vta02'][str(h)]['summary'][cond]['mean_cov_trace'] for h in horizons]
        ax4.plot(horizons, covs, marker='^', lw=2.0, color=palette[cond], label=labels[cond])
    ax4.axhline(400.0, color='red', ls='--', lw=1.5, label='400 m² (20m σp Envelope)')
    ax4.set_title("4. Filter Position Covariance Tr(P_pp) Growth", fontweight='bold')
    ax4.set_xlabel("Outage Duration (s)")
    ax4.set_ylabel("Covariance Trace Tr(P_pp) [m²]")
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, loc='upper left')

    # Panel 5: Regressions Count at 60s
    ax5 = axes[1, 1]
    regr_vta02 = [results['Vta02']['60']['summary'][c]['num_regressions'] for c in cond_keys]
    regr_vta04 = [results['Vta04']['60']['summary'][c]['num_regressions'] for c in cond_keys]
    total_regr = [regr_vta02[i] + regr_vta04[i] for i in range(len(cond_keys))]

    bars = ax5.bar([c.replace('_', '\n') for c in cond_keys], total_regr, color='#e67e22', alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}", ha='center', va='bottom', fontweight='bold')
    ax5.set_title("5. Map-Induced Regressions at 60s (Total across trips)", fontweight='bold')
    ax5.set_ylabel("Number of Regressive Windows (e_map > e_base)")
    ax5.grid(True, alpha=0.3)

    # Panel 6: Wheel Speed vs VBOX Doppler Error Distribution
    ax6 = axes[1, 2]
    dp, dv = load_trip('Vta04')
    v_vbox = dv['veh_speed_ms'].values
    v_wheel = compute_wheel_speed_from_can(dv['wheel_rl_rads'].values, dv['wheel_rr_rads'].values)
    err = v_wheel - v_vbox

    ax6.hist(err, bins=60, range=(-2.0, 2.0), color='#27ae60', alpha=0.75, edgecolor='black', density=True)
    ax6.axvline(0.0, color='black', ls='--', lw=1.5)
    ax6.set_title(f"6. CAN Wheel Speed vs VBOX Error (Vta04)\nMean={np.mean(err):.3f} m/s, Std={np.std(err):.3f} m/s", fontweight='bold')
    ax6.set_xlabel("v_wheel - v_vbox (m/s)")
    ax6.set_ylabel("Probability Density")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "c8_5_wheel_speed_fusion.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Publication dashboard saved to {fig_path}")


if __name__ == "__main__":
    t0 = time.time()
    res = run_c8_5_benchmark()
    print(f"\nCompleted Stage C8-5 benchmark in {time.time() - t0:.2f} seconds.")
