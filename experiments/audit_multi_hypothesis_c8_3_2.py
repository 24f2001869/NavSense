"""
SIH26168 - Stage C8-3.2: Multi-Hypothesis Beam Search (MHT) Association Experiment
Module: experiments/audit_multi_hypothesis_c8_3_2.py

Objective:
Evaluates whether maintaining multiple candidate paths (K in {1, 2, 3, 5}) through forks
with delayed commitment can reduce wrong-road associations and recover from fork ambiguity,
without modifying the ESKF state.

Strict Experimental Rules:
1. Zero ESKF modifications (quarantined on Baseline C7-B1-D Strict Freeze).
2. Beam width sweep: K in {1, 2, 3, 5}.
3. Commitment margin sweep: Delta in {0.00 (immediate), 0.20 (delayed)}.
4. Focus on Fork-Specific Recovery, Wrong-Road Rate, Switching, and Diversity.
5. Benchmark evaluated across Suburban Vta02 (54 windows) and Highway Vta04 (9 windows) at 10s, 20s, 30s, 60s.
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
from src.map.osm_parser import parse_osm_network
from src.map.geometry import RoadNetworkIndex, wrap_angle_rad
from src.map.beam_search import MultiHypothesisTracker, PathHypothesis

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
# WINDOW BENCHMARK FOR MHT
# ==============================================================================

def run_window_mht_audit(
    kw: int,
    w_dur: int,
    dt: float,
    trip_data: dict,
    tracker_configs: List[Tuple[str, int, float]],
    search_radius_m: float = 25.0,
    heading_gate_deg: float = 30.0,
    update_interval_s: float = 1.0
) -> Dict[str, Any]:
    """
    Executes Baseline ESKF and evaluates multiple MHT tracker configurations.
    Tracks overall wrong-road rate, fork wrong-road rate, fork recovery, switches, and beam diversity.
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

    # 15-state ESKF with Baseline mechanisms (NO MAP UPDATES TOUCH FILTER)
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

    # Initialize MHT trackers
    trackers = {}
    tracker_telemetry = {}

    for name, k_val, delta_val in tracker_configs:
        trackers[name] = MultiHypothesisTracker(
            road_index=road_index,
            beam_width_K=k_val,
            search_radius_m=search_radius_m,
            heading_gate_deg=heading_gate_deg,
            commitment_margin_delta=delta_val,
            decay_gamma=0.85
        )
        tracker_telemetry[name] = {
            'total_queries': 0,
            'associated_count': 0,
            'wrong_road_count': 0,
            'correct_road_count': 0,
            'ambiguous_count': 0,
            'no_cand_count': 0,
            'initial_seed_wrong': False,
            'fork_queries': 0,
            'fork_wrong_count': 0,
            'fork_events': 0,
            'fork_recoveries': 0,
            'beam_sizes': [],
            'switches': 0,
            'last_committed_way': None,
            'streaks': [],
            'curr_streak': 0,
            'exec_times_us': []
        }

    t_start = kw * dt
    last_eval_time = -np.inf

    # Pending fork recovery tracking: {name: list of {'fork_start_t': float, 'true_way_id': int, 'resolved': bool}}
    active_fork_trackers = {name: [] for name, _, _ in tracker_configs}

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]
        sp_k = speed[step_k]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)
        nhc.update_eskf(eskf)

        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        b1_constraint.update_eskf(eskf, dt=dt)

        # 1 Hz Offline MHT Evaluation
        if (t_now - last_eval_time) >= (update_interval_s - 1e-4):
            last_eval_time = t_now

            st = eskf.get_state()
            est_pos = st['pos_n'][:2]
            est_heading = float(st['yaw_deg'])
            gt_pos = np.array([gt_e[step_k], gt_n[step_k]])

            # Check true road segment for ground truth
            gt_nearest = road_index.query_point(gt_pos)
            true_seg_idx = gt_nearest['nearest_idx']
            true_way_id = int(road_index.way_ids[true_seg_idx]) if true_seg_idx >= 0 else -1

            # Detect if current GT position is in proximity of a topological branch / fork
            # A fork exists if the true segment's end node connects to >1 outgoing way,
            # or if multiple candidates in R <= 25m have distinct ways and |delta_psi| <= 30 deg
            gt_cands = road_index.query_candidates(
                p=gt_pos, radius_m=20.0,
                veh_heading_rad=np.radians(heading[step_k]),
                heading_gate_rad=np.radians(30.0)
            )
            distinct_gt_ways = len({c['way_id'] for c in gt_cands})
            is_ground_truth_fork = (distinct_gt_ways > 1)

            for name, k_val, delta_val in tracker_configs:
                tr = trackers[name]
                tel = tracker_telemetry[name]
                tel['total_queries'] += 1

                t_exec_start = time.perf_counter()
                out = tr.update(
                    est_pos=est_pos,
                    est_heading_deg=est_heading,
                    speed_ms=sp_k,
                    gyro_z_rads=gz,
                    t_now=t_now
                )
                t_exec_elapsed_us = (time.perf_counter() - t_exec_start) * 1e6
                tel['exec_times_us'].append(t_exec_elapsed_us)

                beam = out['beam_hypotheses']
                tel['beam_sizes'].append(len(beam))

                # Fork tracking
                if is_ground_truth_fork:
                    tel['fork_queries'] += 1

                # Check which candidate to evaluate:
                # If delta > 0 and ambiguous, committed_candidate is None
                chosen_cand = out['committed_candidate'] if delta_val > 0 else out['top_candidate']

                if out['status'] == 'ambiguous':
                    tel['ambiguous_count'] += 1
                elif out['status'] == 'no_candidates':
                    tel['no_cand_count'] += 1

                if chosen_cand is not None:
                    tel['associated_count'] += 1
                    dist_gt = compute_segment_gt_distance(chosen_cand['segment_idx'], gt_pos, road_index)
                    chosen_way = chosen_cand['way_id']

                    # Switches
                    if tel['last_committed_way'] is not None and chosen_way != tel['last_committed_way']:
                        tel['switches'] += 1
                    tel['last_committed_way'] = chosen_way

                    # Wrong-road check
                    is_wrong = (dist_gt > 7.5)
                    if is_wrong:
                        tel['wrong_road_count'] += 1
                        if is_ground_truth_fork:
                            tel['fork_wrong_count'] += 1
                        if tel['curr_streak'] > 0:
                            tel['streaks'].append(tel['curr_streak'])
                            tel['curr_streak'] = 0
                    else:
                        tel['correct_road_count'] += 1
                        tel['curr_streak'] += 1

                    # Check epoch 0 initial seed
                    if tel['total_queries'] == 1 and is_wrong:
                        tel['initial_seed_wrong'] = True
                else:
                    if tel['curr_streak'] > 0:
                        tel['streaks'].append(tel['curr_streak'])
                        tel['curr_streak'] = 0

                # Fork Recovery Evaluation:
                # If entering a fork, track whether true road is in beam and if it recovers to rank 1
                if is_ground_truth_fork and len(beam) > 1:
                    tel['fork_events'] += 1
                    # Check rank of true road in active beam
                    true_in_beam = False
                    true_rank = -1
                    for rank_i, h in enumerate(beam):
                        h_dist = compute_segment_gt_distance(h.seg_idx, gt_pos, road_index)
                        if h_dist <= 7.5:
                            true_in_beam = True
                            true_rank = rank_i
                            break

                    if true_in_beam:
                        if true_rank == 0:
                            # Already correctly rank 1
                            tel['fork_recoveries'] += 1
                        else:
                            # In beam as alternative: register for deferred recovery check
                            active_fork_trackers[name].append({
                                't_fork': t_now,
                                'true_way': true_way_id,
                                'checked_until': t_now + 10.0,
                                'recovered': False
                            })

                # Check pending fork recoveries
                for pending in active_fork_trackers[name]:
                    if not pending['recovered'] and t_now <= pending['checked_until']:
                        if beam and compute_segment_gt_distance(beam[0].seg_idx, gt_pos, road_index) <= 7.5:
                            pending['recovered'] = True
                            tel['fork_recoveries'] += 1

    for name, _, _ in tracker_configs:
        tel = tracker_telemetry[name]
        if tel['curr_streak'] > 0:
            tel['streaks'].append(tel['curr_streak'])

    return tracker_telemetry


# ==============================================================================
# MASTER BENCHMARK RUNNER
# ==============================================================================

def run_c8_3_2_benchmark():
    print("=" * 80)
    print("STAGE C8-3.2: MULTI-HYPOTHESIS BEAM SEARCH (MHT) ASSOCIATION AUDIT")
    print("=" * 80)

    dt = 0.1
    stride_sec = 20.0
    horizons = [10.0, 20.0, 30.0, 60.0]
    trips = ['Vta04', 'Vta02']

    # 6 Configurations tested:
    # 1. K1_Imm: K=1, Delta=0.0 (Single-thread baseline, C8-3.1 S4)
    # 2. K2_Imm: K=2, Delta=0.0 (Minimal branching)
    # 3. K3_Imm: K=3, Delta=0.0 (Nominal beam, immediate commit)
    # 4. K5_Imm: K=5, Delta=0.0 (Wide beam, immediate commit)
    # 5. K3_Delayed: K=3, Delta=0.20 (Nominal beam with delayed commitment)
    # 6. K5_Delayed: K=5, Delta=0.20 (Wide beam with delayed commitment)
    tracker_configs = [
        ('K1_Imm', 1, 0.00),
        ('K2_Imm', 2, 0.00),
        ('K3_Imm', 3, 0.00),
        ('K5_Imm', 5, 0.00),
        ('K3_Delayed', 3, 0.20),
        ('K5_Delayed', 5, 0.20)
    ]

    master_results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'stage': 'C8-3.2',
            'status': 'COMPLETED',
            'configs': [c[0] for c in tracker_configs],
            'horizons': horizons,
            'search_radius_m': 25.0,
            'heading_gate_deg': 30.0,
            'gt_threshold_m': 7.5
        }
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

            cfg_aggregates = {
                c[0]: {
                    'total_queries': 0,
                    'total_associated': 0,
                    'total_wrong_road': 0,
                    'total_correct_road': 0,
                    'total_ambiguous': 0,
                    'total_no_cand': 0,
                    'initial_seed_wrong_count': 0,
                    'fork_queries': 0,
                    'fork_wrong_count': 0,
                    'fork_events': 0,
                    'fork_recoveries': 0,
                    'all_beam_sizes': [],
                    'total_switches': 0,
                    'all_streaks': [],
                    'all_exec_times_us': []
                }
                for c in tracker_configs
            }

            for kw in window_starts:
                win_tel = run_window_mht_audit(
                    kw=kw, w_dur=w_dur, dt=dt,
                    trip_data=trip_data, tracker_configs=tracker_configs,
                    search_radius_m=25.0, heading_gate_deg=30.0
                )

                for cfg_name, tel in win_tel.items():
                    agg = cfg_aggregates[cfg_name]
                    agg['total_queries'] += tel['total_queries']
                    agg['total_associated'] += tel['associated_count']
                    agg['total_wrong_road'] += tel['wrong_road_count']
                    agg['total_correct_road'] += tel['correct_road_count']
                    agg['total_ambiguous'] += tel['ambiguous_count']
                    agg['total_no_cand'] += tel['no_cand_count']
                    if tel['initial_seed_wrong']:
                        agg['initial_seed_wrong_count'] += 1
                    agg['fork_queries'] += tel['fork_queries']
                    agg['fork_wrong_count'] += tel['fork_wrong_count']
                    agg['fork_events'] += tel['fork_events']
                    agg['fork_recoveries'] += tel['fork_recoveries']
                    agg['all_beam_sizes'].extend(tel['beam_sizes'])
                    agg['total_switches'] += tel['switches']
                    agg['all_streaks'].extend(tel['streaks'])
                    agg['all_exec_times_us'].extend(tel['exec_times_us'])

            duration_minutes = (n_win * h_sec) / 60.0
            h_summary = {}

            for cfg_name, agg in cfg_aggregates.items():
                n_assoc = agg['total_associated']
                n_wrong = agg['total_wrong_road']
                n_queries = agg['total_queries']

                wrong_road_rate_pct = (n_wrong / n_assoc * 100.0) if n_assoc > 0 else 0.0
                coverage_pct = (n_assoc / n_queries * 100.0) if n_queries > 0 else 0.0
                fork_wrong_rate_pct = (agg['fork_wrong_count'] / agg['fork_queries'] * 100.0) if agg['fork_queries'] > 0 else 0.0
                fork_recovery_rate_pct = (agg['fork_recoveries'] / agg['fork_events'] * 100.0) if agg['fork_events'] > 0 else 0.0
                initial_seed_wrong_pct = (agg['initial_seed_wrong_count'] / n_win * 100.0) if n_win > 0 else 0.0
                switches_per_min = (agg['total_switches'] / duration_minutes) if duration_minutes > 0 else 0.0
                mean_persistence_s = float(np.mean(agg['all_streaks'])) if agg['all_streaks'] else 0.0
                mean_beam_size = float(np.mean(agg['all_beam_sizes'])) if agg['all_beam_sizes'] else 0.0
                mean_exec_time_us = float(np.mean(agg['all_exec_times_us'])) if agg['all_exec_times_us'] else 0.0

                h_summary[cfg_name] = {
                    'total_queries': n_queries,
                    'associated_count': n_assoc,
                    'wrong_road_count': n_wrong,
                    'wrong_road_rate_pct': wrong_road_rate_pct,
                    'coverage_pct': coverage_pct,
                    'fork_wrong_rate_pct': fork_wrong_rate_pct,
                    'fork_recovery_rate_pct': fork_recovery_rate_pct,
                    'initial_seed_wrong_pct': initial_seed_wrong_pct,
                    'switches_per_min': switches_per_min,
                    'mean_persistence_s': mean_persistence_s,
                    'mean_beam_size': mean_beam_size,
                    'mean_exec_time_us': mean_exec_time_us
                }

            trip_results[f"{int(h_sec)}s"] = {
                'n_windows': n_win,
                'configs': h_summary
            }

            print(f"[{trip_name}] Horizon {int(h_sec)}s ({n_win} windows):")
            for cfg_name in [c[0] for c in tracker_configs]:
                m = h_summary[cfg_name]
                print(f"  {cfg_name:<11}: Wrong: {m['wrong_road_rate_pct']:5.1f}% | ForkWrong: {m['fork_wrong_rate_pct']:5.1f}% | ForkRecov: {m['fork_recovery_rate_pct']:5.1f}% | Switches: {m['switches_per_min']:4.1f}/min | Latency: {m['mean_exec_time_us']:4.0f}us")

        master_results[trip_name] = trip_results

    # Save JSON
    json_path = RES_DIR / "c8_3_2_multi_hypothesis_audit.json"
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Master Data] Saved to {json_path}")

    # Plot publication dashboard
    plot_c8_3_2_dashboard(master_results)

    return master_results


def plot_c8_3_2_dashboard(data: dict):
    """Generates 6-panel diagnostic dashboard for Stage C8-3.2."""
    horizons = [10, 20, 30, 60]
    configs = [
        ('K1_Imm', 'K=1 Single (S4)', '#e74c3c', 'x', '--'),
        ('K2_Imm', 'K=2 Branch', '#e67e22', 's', '-.'),
        ('K3_Imm', 'K=3 Beam', '#3498db', '^', '-'),
        ('K5_Imm', 'K=5 Wide', '#9b59b6', 'v', '-'),
        ('K3_Delayed', 'K=3 Delayed (Δ=0.2)', '#2ecc71', 'o', '-'),
        ('K5_Delayed', 'K=5 Delayed (Δ=0.2)', '#1abc9c', 'D', ':')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Wrong-Road Rate on Highway (Vta04)
    ax = axes[0, 0]
    for cfg_key, label, color, marker, ls in configs:
        rates = [data['Vta04'][f"{h}s"]['configs'][cfg_key]['wrong_road_rate_pct'] for h in horizons]
        ax.plot(horizons, rates, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=7, label=label)
    ax.axhline(5.0, color='gray', linestyle=':', label='Target Threshold (<5%)')
    ax.set_title("A. Highway (Vta04) — Wrong-Road Association Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Wrong-Road Rate (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')

    # Panel 2: Wrong-Road Rate on Suburban (Vta02)
    ax = axes[0, 1]
    for cfg_key, label, color, marker, ls in configs:
        rates = [data['Vta02'][f"{h}s"]['configs'][cfg_key]['wrong_road_rate_pct'] for h in horizons]
        ax.plot(horizons, rates, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=7, label=label)
    ax.axhline(5.0, color='gray', linestyle=':', label='Target Threshold (<5%)')
    ax.set_title("B. Suburban (Vta02) — Wrong-Road Association Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Wrong-Road Rate (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')

    # Panel 3: Fork Recovery Rate on Highway (Vta04)
    ax = axes[1, 0]
    for cfg_key, label, color, marker, ls in configs:
        recov = [data['Vta04'][f"{h}s"]['configs'][cfg_key]['fork_recovery_rate_pct'] for h in horizons]
        ax.plot(horizons, recov, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=7, label=label)
    ax.set_title("C. Highway (Vta04) — Fork Recovery Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Fork Recovery (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')

    # Panel 4: Fork Recovery Rate on Suburban (Vta02)
    ax = axes[1, 1]
    for cfg_key, label, color, marker, ls in configs:
        recov = [data['Vta02'][f"{h}s"]['configs'][cfg_key]['fork_recovery_rate_pct'] for h in horizons]
        ax.plot(horizons, recov, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=7, label=label)
    ax.set_title("D. Suburban (Vta02) — Fork Recovery Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Fork Recovery (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')

    # Panel 5: Fork Wrong-Road Rate at 30s Horizon
    ax = axes[2, 0]
    x = np.arange(len(configs))
    width = 0.35
    fwr_v04 = [data['Vta04']['30s']['configs'][c[0]]['fork_wrong_rate_pct'] for c in configs]
    fwr_v02 = [data['Vta02']['30s']['configs'][c[0]]['fork_wrong_rate_pct'] for c in configs]
    ax.bar(x - width/2, fwr_v04, width, label='Highway (Vta04)', color='#34495e')
    ax.bar(x + width/2, fwr_v02, width, label='Suburban (Vta02)', color='#1abc9c')
    ax.set_title("E. Fork Wrong-Road Rate at 30 s Horizon (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fork Wrong-Road (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in configs], rotation=25, ha='right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=9)

    # Panel 6: Execution Latency per Epoch (Microseconds)
    ax = axes[2, 1]
    lat_v04 = [data['Vta04']['30s']['configs'][c[0]]['mean_exec_time_us'] for c in configs]
    lat_v02 = [data['Vta02']['30s']['configs'][c[0]]['mean_exec_time_us'] for c in configs]
    ax.bar(x - width/2, lat_v04, width, label='Highway (Vta04)', color='#2980b9')
    ax.bar(x + width/2, lat_v02, width, label='Suburban (Vta02)', color='#8e44ad')
    ax.set_title("F. Computational Latency per Epoch (μs)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Execution Time (μs)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in configs], rotation=25, ha='right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=9)

    plt.suptitle("Stage C8-3.2: Multi-Hypothesis Beam Search (MHT) Diagnostic Dashboard", fontsize=15, fontweight='bold', y=0.98)
    fig_path = FIG_DIR / "c8_3_2_multi_hypothesis_audit.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Dashboard] Figure saved to {fig_path}")


if __name__ == '__main__':
    run_c8_3_2_benchmark()
