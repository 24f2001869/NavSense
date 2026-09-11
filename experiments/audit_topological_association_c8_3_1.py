"""
SIH26168 - Stage C8-3.1: Offline Temporal & Topological Map Association Experiment
Module: experiments/audit_topological_association_c8_3_1.py

Objective:
Evaluates whether temporal consistency (persistence/hysteresis) and topological graph
connectivity can eliminate wrong-road associations without modifying the ESKF state.

Strict Experimental Rules:
1. Zero ESKF modifications (quarantined on Baseline C7-B1-D Strict Freeze).
2. Five candidate selection strategies evaluated:
   - S1: Instantaneous Nearest Candidate (Naive baseline)
   - S2: Instantaneous Heading-Gated Candidate (C8-2 baseline)
   - S3: Continuity-Aware Candidate (Way persistence & hysteresis)
   - S4: Topology-Aware Candidate (OSM node graph connectivity)
   - S5: Multi-Factor Sequence Scorer (Distance + Heading + Continuity + Transition)
3. Evaluated across Suburban Vta02 (54 windows) and Highway Vta04 (9 windows) at 10s, 20s, 30s, 60s.
4. Acceptance hierarchy:
   - Primary: Wrong-road association rate (dist_gt > 7.5 m)
   - Secondary: Correct-road persistence (consecutive seconds)
   - Tertiary: Candidate-switch frequency (switches/min)
   - Coverage: Association recall / rejection rate
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


def compute_gt_distance(cand: dict, gt_pos: np.ndarray, road_index: RoadNetworkIndex) -> float:
    """Computes orthogonal distance from true RTK VBOX position to candidate segment."""
    seg_idx = cand['segment_idx']
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
# 5 CANDIDATE SELECTION STRATEGIES
# ==============================================================================

class Strategy1_InstantaneousNearest:
    """Strategy 1: Naive nearest candidate snapping without heading or topology gating."""
    def __init__(self, road_index: RoadNetworkIndex):
        self.index = road_index
        self.name = "S1_InstantaneousNearest"

    def select(self, candidates: List[dict], est_pos: np.ndarray, est_heading_deg: float) -> Optional[dict]:
        if not candidates:
            return None
        return candidates[0]  # Already sorted by distance in query_candidates


class Strategy2_HeadingGated:
    """Strategy 2: Instantaneous heading-gated snapping with 3m ambiguity margin (C8-2)."""
    def __init__(self, road_index: RoadNetworkIndex, ambiguity_margin_m: float = 3.0):
        self.index = road_index
        self.ambiguity_margin_m = ambiguity_margin_m
        self.name = "S2_HeadingGated"

    def select(self, candidates: List[dict], est_pos: np.ndarray, est_heading_deg: float) -> Optional[dict]:
        # Candidates passed to this method already have |delta_heading| <= 30 deg
        if not candidates:
            return None
        if len(candidates) > 1:
            c0, c1 = candidates[0], candidates[1]
            if c0['way_id'] != c1['way_id']:
                if abs(c0['distance_m'] - c1['distance_m']) < self.ambiguity_margin_m:
                    return None
        return candidates[0]


class Strategy3_ContinuityAware:
    """Strategy 3: Way-persistence tracker with hysteresis against switching ways."""
    def __init__(self, road_index: RoadNetworkIndex, hysteresis_m: float = 8.0):
        self.index = road_index
        self.hysteresis_m = hysteresis_m
        self.active_way_id: Optional[int] = None
        self.active_seg_idx: Optional[int] = None
        self.name = "S3_ContinuityAware"

    def reset(self):
        self.active_way_id = None
        self.active_seg_idx = None

    def select(self, candidates: List[dict], est_pos: np.ndarray, est_heading_deg: float) -> Optional[dict]:
        if not candidates:
            return None

        # First association: pick nearest heading-gated candidate
        if self.active_way_id is None:
            chosen = candidates[0]
            self.active_way_id = chosen['way_id']
            self.active_seg_idx = chosen['segment_idx']
            return chosen

        # Separate candidates into active way vs competing ways
        on_way = [c for c in candidates if c['way_id'] == self.active_way_id]
        off_way = [c for c in candidates if c['way_id'] != self.active_way_id]

        if on_way:
            best_on = on_way[0]  # Sorted by distance
            if not off_way:
                self.active_seg_idx = best_on['segment_idx']
                return best_on

            best_off = off_way[0]
            # Only switch away if the competing road is significantly closer than current road
            if best_off['distance_m'] < (best_on['distance_m'] - self.hysteresis_m):
                self.active_way_id = best_off['way_id']
                self.active_seg_idx = best_off['segment_idx']
                return best_off
            else:
                self.active_seg_idx = best_on['segment_idx']
                return best_on
        else:
            # Active road is no longer within search radius: switch to best candidate
            best_off = off_way[0]
            self.active_way_id = best_off['way_id']
            self.active_seg_idx = best_off['segment_idx']
            return best_off


class Strategy4_TopologyAware:
    """Strategy 4: Strict graph connectivity constraint (rejects disconnected candidates)."""
    def __init__(self, road_index: RoadNetworkIndex):
        self.index = road_index
        self.active_seg_idx: Optional[int] = None
        self.name = "S4_TopologyAware"

    def reset(self):
        self.active_seg_idx = None

    def select(self, candidates: List[dict], est_pos: np.ndarray, est_heading_deg: float) -> Optional[dict]:
        if not candidates:
            return None

        # First association: pick nearest heading-gated candidate
        if self.active_seg_idx is None:
            chosen = candidates[0]
            self.active_seg_idx = chosen['segment_idx']
            return chosen

        # Filter candidates that are topologically connected to previous segment
        connected_cands = [
            c for c in candidates
            if self.index.are_topologically_connected(self.active_seg_idx, c['segment_idx'])
        ]

        if connected_cands:
            # Pick closest connected candidate
            chosen = connected_cands[0]
            self.active_seg_idx = chosen['segment_idx']
            return chosen
        else:
            # Disconnected: all candidates are disjoint roads (parallel roads/alleys). Reject update!
            return None


class Strategy5_MultiFactorScorer:
    """Strategy 5: Multi-factor sequence score combining distance, heading, continuity, and topology."""
    def __init__(
        self,
        road_index: RoadNetworkIndex,
        w_d: float = 0.35,
        w_psi: float = 0.25,
        w_c: float = 0.20,
        w_t: float = 0.20,
        sigma_d: float = 5.0,
        sigma_psi_deg: float = 15.0,
        min_score: float = 0.35
    ):
        self.index = road_index
        self.w_d = w_d
        self.w_psi = w_psi
        self.w_c = w_c
        self.w_t = w_t
        self.sigma_d = sigma_d
        self.sigma_psi_rad = np.radians(sigma_psi_deg)
        self.min_score = min_score

        self.prev_seg_idx: Optional[int] = None
        self.prev_way_id: Optional[int] = None
        self.name = "S5_MultiFactorScorer"

    def reset(self):
        self.prev_seg_idx = None
        self.prev_way_id = None

    def select(self, candidates: List[dict], est_pos: np.ndarray, est_heading_deg: float) -> Optional[dict]:
        if not candidates:
            return None

        best_score = -1.0
        best_cand = None

        for c in candidates:
            d = c['distance_m']
            delta_psi = c['delta_heading_rad'] if c['delta_heading_rad'] is not None else 0.0

            # 1. Distance likelihood
            s_d = float(np.exp(-0.5 * (d / self.sigma_d) ** 2))

            # 2. Heading alignment likelihood
            s_psi = float(np.exp(-0.5 * (delta_psi / self.sigma_psi_rad) ** 2))

            # 3. Continuity score (way persistence)
            s_c = 1.0 if (self.prev_way_id is not None and c['way_id'] == self.prev_way_id) else (0.5 if self.prev_way_id is None else 0.0)

            # 4. Topological transition score (graph connectivity)
            s_t = self.index.topological_transition_score(self.prev_seg_idx, c['segment_idx'])

            total_score = (
                self.w_d * s_d +
                self.w_psi * s_psi +
                self.w_c * s_c +
                self.w_t * s_t
            )

            if total_score > best_score:
                best_score = total_score
                best_cand = c

        if best_score < self.min_score or best_cand is None:
            return None

        self.prev_seg_idx = best_cand['segment_idx']
        self.prev_way_id = best_cand['way_id']
        return best_cand


# ==============================================================================
# WINDOW EVALUATION
# ==============================================================================

def run_window_association_audit(
    kw: int,
    w_dur: int,
    dt: float,
    trip_data: dict,
    strategies: List[Any],
    search_radius_m: float = 25.0,
    heading_gate_deg: float = 30.0,
    update_interval_s: float = 1.0
) -> Dict[str, Any]:
    """
    Runs the Baseline ESKF (strictly without map feedback) through a blackout window,
    and evaluates each of the 5 association strategies at 1 Hz intervals against RTK ground truth.
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

    # Reset stateful strategies
    for s in strategies:
        if hasattr(s, 'reset'):
            s.reset()

    # Telemetry storage per strategy
    strat_telemetry = {
        s.name: {
            'total_queries': 0,
            'associated_count': 0,
            'wrong_road_count': 0,
            'correct_road_count': 0,
            'rejected_count': 0,
            'persistence_streaks': [],
            'current_streak': 0,
            'way_switches': 0,
            'last_way_id': None,
            'gt_distances': []
        }
        for s in strategies
    }

    t_start = kw * dt
    last_map_eval_time = -np.inf
    h_gate_rad = np.radians(heading_gate_deg)

    for step_k in range(kw, kw + w_dur):
        t_now = step_k * dt - t_start
        eskf.sigma_a = sigma_series[step_k]

        ax, ay, az = acc_v[step_k]
        gx, gy, gz = gyro_v[step_k]

        eskf.predict(ax, ay, az, gx, gy, gz, dt)

        # Baseline updates (NO MAP UPDATES TOUCH ESKF)
        nhc.update_eskf(eskf)

        k_start = max(0, step_k - 10)
        det_out = detector.update(acc_v[k_start : step_k + 1], gyro_v[k_start : step_k + 1])
        if det_out['is_stationary']:
            zupt.update_eskf(eskf)

        b1_constraint.update_eskf(eskf, dt=dt)

        # 1 Hz Offline Association Evaluation
        if (t_now - last_map_eval_time) >= (update_interval_s - 1e-4):
            last_map_eval_time = t_now

            st = eskf.get_state()
            est_pos = st['pos_n'][:2]
            est_heading = float(st['yaw_deg'])
            est_h_rad = np.radians(est_heading)

            gt_pos = np.array([gt_e[step_k], gt_n[step_k]])

            # Query all candidates in radius R
            # For S1: unfiltered candidates
            raw_cands = road_index.query_candidates(
                p=est_pos, radius_m=search_radius_m
            )
            # For S2-S5: heading-gated candidates (|Delta psi| <= 30 deg)
            gated_cands = road_index.query_candidates(
                p=est_pos, radius_m=search_radius_m,
                veh_heading_rad=est_h_rad, heading_gate_rad=h_gate_rad
            )

            # Evaluate each strategy
            for s in strategies:
                tel = strat_telemetry[s.name]
                tel['total_queries'] += 1

                cands_to_use = raw_cands if s.name == "S1_InstantaneousNearest" else gated_cands
                chosen = s.select(cands_to_use, est_pos, est_heading)

                if chosen is not None:
                    tel['associated_count'] += 1
                    dist_gt = compute_gt_distance(chosen, gt_pos, road_index)
                    tel['gt_distances'].append(dist_gt)

                    # Track way switches
                    if tel['last_way_id'] is not None and chosen['way_id'] != tel['last_way_id']:
                        tel['way_switches'] += 1
                    tel['last_way_id'] = chosen['way_id']

                    # Check against ground truth threshold (7.5 m)
                    if dist_gt > 7.5:
                        tel['wrong_road_count'] += 1
                        # Streak broken
                        if tel['current_streak'] > 0:
                            tel['persistence_streaks'].append(tel['current_streak'])
                            tel['current_streak'] = 0
                    else:
                        tel['correct_road_count'] += 1
                        tel['current_streak'] += 1
                else:
                    tel['rejected_count'] += 1
                    if tel['current_streak'] > 0:
                        tel['persistence_streaks'].append(tel['current_streak'])
                        tel['current_streak'] = 0

    # Finalize streaks
    for s in strategies:
        tel = strat_telemetry[s.name]
        if tel['current_streak'] > 0:
            tel['persistence_streaks'].append(tel['current_streak'])

    return strat_telemetry


# ==============================================================================
# MASTER BENCHMARK RUNNER
# ==============================================================================

def run_c8_3_1_benchmark():
    print("=" * 80)
    print("STAGE C8-3.1: OFFLINE TEMPORAL & TOPOLOGICAL MAP ASSOCIATION AUDIT")
    print("=" * 80)

    dt = 0.1
    stride_sec = 20.0
    horizons = [10.0, 20.0, 30.0, 60.0]
    trips = ['Vta04', 'Vta02']

    master_results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'stage': 'C8-3.1',
            'status': 'COMPLETED',
            'strategies': [
                'S1_InstantaneousNearest',
                'S2_HeadingGated',
                'S3_ContinuityAware',
                'S4_TopologyAware',
                'S5_MultiFactorScorer'
            ],
            'horizons': horizons,
            'search_radius_m': 25.0,
            'heading_gate_deg': 30.0,
            'gt_threshold_m': 7.5
        }
    }

    for trip_name in trips:
        trip_data = prepare_trip_data(trip_name, dt=dt)
        road_index = trip_data['road_index']

        strategies = [
            Strategy1_InstantaneousNearest(road_index),
            Strategy2_HeadingGated(road_index, ambiguity_margin_m=3.0),
            Strategy3_ContinuityAware(road_index, hysteresis_m=8.0),
            Strategy4_TopologyAware(road_index),
            Strategy5_MultiFactorScorer(road_index)
        ]

        n_epochs = trip_data['n_epochs']
        stride_epochs = int(round(stride_sec / dt))
        trip_results = {}

        for h_sec in horizons:
            w_dur = int(round(h_sec / dt))
            window_starts = list(range(0, n_epochs - w_dur, stride_epochs))
            n_win = len(window_starts)

            strat_aggregates = {
                s.name: {
                    'total_queries': 0,
                    'total_associated': 0,
                    'total_wrong_road': 0,
                    'total_correct_road': 0,
                    'total_rejected': 0,
                    'all_streaks': [],
                    'total_switches': 0,
                    'all_gt_distances': []
                }
                for s in strategies
            }

            for kw in window_starts:
                win_tel = run_window_association_audit(
                    kw=kw, w_dur=w_dur, dt=dt,
                    trip_data=trip_data, strategies=strategies,
                    search_radius_m=25.0, heading_gate_deg=30.0
                )

                for s_name, tel in win_tel.items():
                    agg = strat_aggregates[s_name]
                    agg['total_queries'] += tel['total_queries']
                    agg['total_associated'] += tel['associated_count']
                    agg['total_wrong_road'] += tel['wrong_road_count']
                    agg['total_correct_road'] += tel['correct_road_count']
                    agg['total_rejected'] += tel['rejected_count']
                    agg['all_streaks'].extend(tel['persistence_streaks'])
                    agg['total_switches'] += tel['way_switches']
                    agg['all_gt_distances'].extend(tel['gt_distances'])

            # Summary metrics per strategy
            duration_minutes = (n_win * h_sec) / 60.0
            h_summary = {}

            for s_name, agg in strat_aggregates.items():
                n_assoc = agg['total_associated']
                n_wrong = agg['total_wrong_road']
                n_queries = agg['total_queries']

                wrong_road_rate_pct = (n_wrong / n_assoc * 100.0) if n_assoc > 0 else 0.0
                coverage_pct = (n_assoc / n_queries * 100.0) if n_queries > 0 else 0.0
                mean_persistence_s = float(np.mean(agg['all_streaks'])) if agg['all_streaks'] else 0.0
                switches_per_min = (agg['total_switches'] / duration_minutes) if duration_minutes > 0 else 0.0
                mean_gt_dist_m = float(np.mean(agg['all_gt_distances'])) if agg['all_gt_distances'] else 0.0

                h_summary[s_name] = {
                    'total_queries': n_queries,
                    'associated_count': n_assoc,
                    'wrong_road_count': n_wrong,
                    'wrong_road_rate_pct': wrong_road_rate_pct,
                    'coverage_pct': coverage_pct,
                    'mean_persistence_s': mean_persistence_s,
                    'switches_per_min': switches_per_min,
                    'mean_gt_dist_m': mean_gt_dist_m
                }

            trip_results[f"{int(h_sec)}s"] = {
                'n_windows': n_win,
                'strategies': h_summary
            }

            print(f"[{trip_name}] Horizon {int(h_sec)}s ({n_win} windows):")
            for s_name in ['S1_InstantaneousNearest', 'S2_HeadingGated', 'S3_ContinuityAware', 'S4_TopologyAware', 'S5_MultiFactorScorer']:
                m = h_summary[s_name]
                print(f"  {s_name:<26}: Wrong-Road: {m['wrong_road_rate_pct']:5.1f}% | Coverage: {m['coverage_pct']:5.1f}% | Persistence: {m['mean_persistence_s']:4.1f}s | Switches: {m['switches_per_min']:4.1f}/min")

        master_results[trip_name] = trip_results

    # Save master JSON
    json_path = RES_DIR / "c8_3_1_topological_association_audit.json"
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Master Data] Saved to {json_path}")

    # Generate Publication Figure
    plot_c8_3_1_dashboard(master_results)

    return master_results


def plot_c8_3_1_dashboard(data: dict):
    """Plots 6-panel diagnostic dashboard for C8-3.1 association audit."""
    horizons = [10, 20, 30, 60]
    strats = [
        ('S1_InstantaneousNearest', 'S1: Nearest', '#e74c3c', 'x', '--'),
        ('S2_HeadingGated', 'S2: Heading Gate (C8-2)', '#e67e22', 's', '-.'),
        ('S3_ContinuityAware', 'S3: Way Persistence', '#3498db', '^', '-'),
        ('S4_TopologyAware', 'S4: Topology Graph', '#9b59b6', 'v', '-'),
        ('S5_MultiFactorScorer', 'S5: Multi-Factor', '#2ecc71', 'o', '-')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Wrong-Road Rate on Highway (Vta04)
    ax = axes[0, 0]
    for s_key, label, color, marker, ls in strats:
        rates = [data['Vta04'][f"{h}s"]['strategies'][s_key]['wrong_road_rate_pct'] for h in horizons]
        ax.plot(horizons, rates, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=label)
    ax.axhline(5.0, color='gray', linestyle=':', label='Target Threshold (<5%)')
    ax.set_title("A. Highway (Vta04) — Wrong-Road Association Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Wrong-Road Rate (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    # Panel 2: Wrong-Road Rate on Suburban (Vta02)
    ax = axes[0, 1]
    for s_key, label, color, marker, ls in strats:
        rates = [data['Vta02'][f"{h}s"]['strategies'][s_key]['wrong_road_rate_pct'] for h in horizons]
        ax.plot(horizons, rates, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=label)
    ax.axhline(5.0, color='gray', linestyle=':', label='Target Threshold (<5%)')
    ax.set_title("B. Suburban (Vta02) — Wrong-Road Association Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Wrong-Road Rate (%)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    # Panel 3: Correct-Road Persistence on Highway (Vta04)
    ax = axes[1, 0]
    for s_key, label, color, marker, ls in strats:
        pers = [data['Vta04'][f"{h}s"]['strategies'][s_key]['mean_persistence_s'] for h in horizons]
        ax.plot(horizons, pers, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=label)
    ax.set_title("C. Highway (Vta04) — Correct-Road Persistence (Streak Sec)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Mean Persistence Streak (s)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    # Panel 4: Correct-Road Persistence on Suburban (Vta02)
    ax = axes[1, 1]
    for s_key, label, color, marker, ls in strats:
        pers = [data['Vta02'][f"{h}s"]['strategies'][s_key]['mean_persistence_s'] for h in horizons]
        ax.plot(horizons, pers, marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=label)
    ax.set_title("D. Suburban (Vta02) — Correct-Road Persistence (Streak Sec)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Blackout Horizon (s)", fontsize=10)
    ax.set_ylabel("Mean Persistence Streak (s)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    # Panel 5: Candidate-Switch Frequency (Vta04 & Vta02 at 30s)
    ax = axes[2, 0]
    x = np.arange(len(strats))
    width = 0.35
    switches_v04 = [data['Vta04']['30s']['strategies'][s[0]]['switches_per_min'] for s in strats]
    switches_v02 = [data['Vta02']['30s']['strategies'][s[0]]['switches_per_min'] for s in strats]
    rects1 = ax.bar(x - width/2, switches_v04, width, label='Highway (Vta04)', color='#34495e')
    rects2 = ax.bar(x + width/2, switches_v02, width, label='Suburban (Vta02)', color='#1abc9c')
    ax.set_title("E. Way-Switch Frequency at 30 s Horizon (Switches / Min)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Switches / Min", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in strats], rotation=25, ha='right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=9)

    # Panel 6: Association Coverage / Recall at 30s
    ax = axes[2, 1]
    cov_v04 = [data['Vta04']['30s']['strategies'][s[0]]['coverage_pct'] for s in strats]
    cov_v02 = [data['Vta02']['30s']['strategies'][s[0]]['coverage_pct'] for s in strats]
    rects3 = ax.bar(x - width/2, cov_v04, width, label='Highway (Vta04)', color='#2980b9')
    rects4 = ax.bar(x + width/2, cov_v02, width, label='Suburban (Vta02)', color='#8e44ad')
    ax.set_title("F. Association Coverage at 30 s Horizon (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Valid Updates (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in strats], rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=9)

    plt.suptitle("Stage C8-3.1: Offline Temporal & Topological Map Association Audit Dashboard", fontsize=15, fontweight='bold', y=0.98)
    fig_path = FIG_DIR / "c8_3_1_association_audit.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Dashboard] Figure saved to {fig_path}")


if __name__ == '__main__':
    run_c8_3_1_benchmark()
