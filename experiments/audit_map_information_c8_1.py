"""
SIH26168 - Stage C8-1: Offline Map Information Utility & Observability Audit
Script: experiments/audit_map_information_c8_1.py

Comprehensive offline information-theoretic and observability analysis:
- Experiment 1: Information Matrix & Rank/Conditioning Analysis (Subspace observability)
- Experiment 2: Multi-Level ESKF Uncertainty Stress Test (5m, 10m, 20m, 25m, 50m, 100m)
- Experiment 3: The Map Observability Envelope (Theoretical covariance reduction across paired benchmarks)
- Experiment 4: Statistical Gating & Candidate Rejection Analysis (NIS and Chi-Square gating)

STRICT RULE: Pure offline characterization. Zero ESKF modifications, zero filter changes, zero navigation runs.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.orientation import geodetic_to_enu
from src.map.osm_parser import download_osm_bbox, parse_osm_network
from src.map.geometry import RoadNetworkIndex, wrap_angle_rad
from src.map.information import (
    compute_map_jacobians, compute_map_covariance,
    compute_fisher_information, compute_covariance_reduction, compute_nis
)

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# EXPERIMENT 1: INFORMATION MATRIX & RANK/CONDITIONING AUDIT
# ==============================================================================

def run_experiment_1_information_matrix(trips=['Vta02', 'Vta04']) -> dict:
    """
    Computes the Fisher Information Matrix I_map = H^T * R^-1 * H across road segments.
    Verifies rank, observable subspace, and contrast between straight vs curved geometry.
    """
    print("\n" + "="*80)
    print(" EXPERIMENT 1: Information Matrix & Observable Subspace Analysis")
    print("="*80)

    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        lats = df_v['veh_lat'].values
        lons = df_v['veh_lon'].values
        lat0, lon0 = float(lats[0]), float(lons[0])

        cache_name = f"{trip}_network"
        osm_path = download_osm_bbox(float(lons.min() - 0.003), float(lats.min() - 0.003),
                                     float(lons.max() + 0.003), float(lats.max() + 0.003),
                                     cache_name=cache_name)
        network = parse_osm_network(osm_path, lat0, lon0)
        segments = network['segments']

        # Sample headings across segments
        headings = np.array([s['heading_rad'] for s in segments[::10]])
        curvatures = []
        # Estimate segment-to-segment curvature
        for i in range(len(segments) - 1):
            if segments[i]['way_id'] == segments[i+1]['way_id']:
                dh = abs(wrap_angle_rad(segments[i+1]['heading_rad'] - segments[i]['heading_rad']))
                ds = (segments[i]['length'] + segments[i+1]['length']) / 2.0
                curvatures.append(float(dh / (ds + 1e-3)))
            else:
                curvatures.append(0.0)
        curvatures = np.array(curvatures)

        # 1. Condition A: Lateral Only
        H_perp, _, _ = compute_map_jacobians(0.0)
        R_perp = np.array([[2.5 ** 2]])
        I_perp, rank_perp, cond_perp, eigs_perp = compute_fisher_information(H_perp, R_perp)

        # Verify exact along-track null space: t = [0, 1, 0, ...]
        t_null = np.zeros(15)
        t_null[1] = 1.0  # Along road (North)
        along_null_val = float(t_null.T @ I_perp @ t_null)

        # 2. Condition B: Heading Only
        _, H_psi, _ = compute_map_jacobians(0.0)
        R_psi = np.array([[np.radians(5.0) ** 2]])
        I_psi, rank_psi, cond_psi, eigs_psi = compute_fisher_information(H_psi, R_psi)

        # 3. Condition C: Joint Lateral + Heading (Straight road: rho = 0)
        _, _, H_joint = compute_map_jacobians(0.0)
        R_joint_straight = compute_map_covariance(sigma_lane=2.5, sigma_psi_rad=np.radians(5.0), curvature_rad_m=0.0)
        I_joint_str, rank_joint_str, cond_joint_str, eigs_joint_str = compute_fisher_information(H_joint, R_joint_straight)

        # 4. Condition C-curve: Joint Lateral + Heading on Curve (kappa = 0.05 rad/m -> R = 20m curve)
        R_joint_curve = compute_map_covariance(sigma_lane=2.5, sigma_psi_rad=np.radians(5.0), curvature_rad_m=0.05)
        I_joint_cur, rank_joint_cur, cond_joint_cur, eigs_joint_cur = compute_fisher_information(H_joint, R_joint_curve)

        results[trip] = {
            'trip': trip,
            'segments_analyzed': len(segments),
            'mean_curvature_rad_m': float(np.mean(curvatures)),
            'p95_curvature_rad_m': float(np.percentile(curvatures, 95)),
            'lateral_only': {
                'rank': rank_perp,
                'non_zero_eigenvalue': eigs_perp[0] if len(eigs_perp) > 0 else 0.0,
                'along_track_null_space_projection': along_null_val
            },
            'heading_only': {
                'rank': rank_psi,
                'non_zero_eigenvalue': eigs_psi[0] if len(eigs_psi) > 0 else 0.0
            },
            'joint_straight': {
                'rank': rank_joint_str,
                'condition_number': cond_joint_str,
                'eigenvalues': eigs_joint_str
            },
            'joint_curve': {
                'rank': rank_joint_cur,
                'condition_number': cond_joint_cur,
                'eigenvalues': eigs_joint_cur
            }
        }

        print(f"[{trip}] Lateral Constraint: Rank = {rank_perp} | Along-Track Information = {along_null_val:.6f} (Strict Null Space)")
        print(f"       Heading Constraint: Rank = {rank_psi} | Heading Information = {eigs_psi[0]:.2f} rad^-2")
        print(f"       Joint Constraint (Straight): Rank = {rank_joint_str} | Condition No. = {cond_joint_str:.1f} | Eigs = {[round(x, 2) for x in eigs_joint_str]}")
        print(f"       Joint Constraint (Curve):    Rank = {rank_joint_cur} | Condition No. = {cond_joint_cur:.1f} | Eigs = {[round(x, 2) for x in eigs_joint_cur]}")

    return results


# ==============================================================================
# EXPERIMENT 2: MULTI-LEVEL ESKF UNCERTAINTY STRESS TEST
# ==============================================================================

def run_experiment_2_uncertainty_stress_test(trips=['Vta02', 'Vta04']) -> dict:
    """
    Evaluates map information behavior across realistic ESKF position uncertainty levels:
    sigma_p in [5m, 10m, 20m, 25m, 50m, 100m].
    Computes candidate counts, residuals, NIS, and candidate acceptance/rejection rates.
    """
    print("\n" + "="*80)
    print(" EXPERIMENT 2: Multi-Level ESKF Uncertainty Stress Test")
    print("="*80)

    uncertainty_levels = [5.0, 10.0, 20.0, 25.0, 50.0, 100.0]
    chi2_95 = 5.991  # 2 DOF chi-square 95% threshold
    chi2_99 = 9.210  # 2 DOF chi-square 99% threshold

    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        lats = df_v['veh_lat'].values
        lons = df_v['veh_lon'].values
        vx = df_v['veh_speed_ms'].values
        lat0, lon0 = float(lats[0]), float(lons[0])

        cache_name = f"{trip}_network"
        osm_path = download_osm_bbox(float(lons.min() - 0.003), float(lats.min() - 0.003),
                                     float(lons.max() + 0.003), float(lats.max() + 0.003),
                                     cache_name=cache_name)
        network = parse_osm_network(osm_path, lat0, lon0)
        index = RoadNetworkIndex(network['segments'])

        # Compute ENU trajectory
        enu_pts = np.array([geodetic_to_enu(lat, lon, lat0, lon0)[:2] for lat, lon in zip(lats, lons)])
        diff_enu = np.diff(enu_pts, axis=0, prepend=[enu_pts[0]])
        vbox_heading_rad = np.arctan2(diff_enu[:, 0], diff_enu[:, 1])

        mask_motion = (vx > 2.0) & (np.linalg.norm(diff_enu, axis=1) > 0.1)
        eval_indices = np.where(mask_motion)[0][::5]  # Downsample to 2 Hz for efficiency

        trip_level_results = {}

        for sig_p in uncertainty_levels:
            # Scale realistic heading uncertainty with position drift
            sig_psi_deg = min(30.0, 2.0 + 0.28 * sig_p)
            sig_psi_rad = np.radians(sig_psi_deg)

            # Prior covariance matrix P^- (15x15)
            P_diag = np.ones(15)
            P_diag[0] = sig_p ** 2  # East pos
            P_diag[1] = sig_p ** 2  # North pos
            P_diag[2] = 1.0         # Up pos
            P_diag[3:6] = 1.0 ** 2  # Velocity
            P_diag[6:8] = np.radians(0.5) ** 2  # Roll, Pitch
            P_diag[8] = sig_psi_rad ** 2        # Yaw
            P_diag[9:12] = 0.05 ** 2            # Accel bias
            P_diag[12:15] = 0.005 ** 2          # Gyro bias
            P_prior = np.diag(P_diag)

            R_mat = compute_map_covariance(sigma_lane=2.5, sigma_psi_rad=np.radians(5.0))

            nis_list = []
            cand_counts = []
            accepted_95_count = 0
            accepted_99_count = 0
            multi_accepted_count = 0  # Ambiguous accepted roads

            for idx in eval_indices:
                p = enu_pts[idx]
                psi = vbox_heading_rad[idx]

                # Search radius scaled to 2.5 * sigma_p (capped at 100m)
                R_search = min(100.0, max(15.0, 2.5 * sig_p))
                cands = index.query_candidates(p, radius_m=R_search, veh_heading_rad=psi, heading_gate_rad=np.radians(30.0))
                cand_counts.append(len(cands))

                if len(cands) == 0:
                    continue

                # Check NIS for nearest candidate
                best = cands[0]
                _, _, H_joint = compute_map_jacobians(best['road_heading_rad'])
                r_vec = np.array([best['distance_m'] - (-0.90), best['delta_heading_rad']])
                nis = compute_nis(r_vec, H_joint, P_prior, R_mat)
                nis_list.append(nis)

                if nis <= chi2_95:
                    accepted_95_count += 1
                if nis <= chi2_99:
                    accepted_99_count += 1

                # Check how many distinct roads pass the chi-square gate
                pass_roads = set()
                for c in cands:
                    _, _, H_c = compute_map_jacobians(c['road_heading_rad'])
                    r_c = np.array([c['distance_m'] - (-0.90), c['delta_heading_rad']])
                    if compute_nis(r_c, H_c, P_prior, R_mat) <= chi2_95:
                        pass_roads.add(c['way_id'])
                if len(pass_roads) >= 2:
                    multi_accepted_count += 1

            nis_arr = np.array(nis_list)
            total_tested = len(eval_indices)

            trip_level_results[f"sigma_{int(sig_p)}m"] = {
                'sigma_p_m': sig_p,
                'sigma_psi_deg': sig_psi_deg,
                'search_radius_m': R_search,
                'avg_candidate_count': float(np.mean(cand_counts)),
                'p95_candidate_count': float(np.percentile(cand_counts, 95)),
                'nis_median': float(np.median(nis_arr)) if len(nis_arr) > 0 else 0.0,
                'nis_mean': float(np.mean(nis_arr)) if len(nis_arr) > 0 else 0.0,
                'nis_p95': float(np.percentile(nis_arr, 95)) if len(nis_arr) > 0 else 0.0,
                'acceptance_rate_chi2_95': float(accepted_95_count / total_tested),
                'acceptance_rate_chi2_99': float(accepted_99_count / total_tested),
                'ambiguous_accepted_rate': float(multi_accepted_count / total_tested)
            }

            print(f"[{trip}] sigma_p={sig_p:3.0f}m (psi={sig_psi_deg:4.1f}°): Avg Cands={np.mean(cand_counts):4.1f} | NIS Median={np.median(nis_arr):4.2f}, P95={np.percentile(nis_arr, 95):5.2f} | Accept(95%)={accepted_95_count/total_tested*100:5.1f}% | Multi-Accept Ambiguity={multi_accepted_count/total_tested*100:4.1f}%")

        results[trip] = trip_level_results

    return results


# ==============================================================================
# EXPERIMENT 3: THE MAP OBSERVABILITY ENVELOPE (THEORETICAL COVARIANCE REDUCTION)
# ==============================================================================

def run_experiment_3_observability_envelope(trips=['Vta02', 'Vta04']) -> dict:
    """
    Constructs the Map Observability Envelope by computing the exact analytical covariance reduction:
      Delta P = P^- - ( (P^-)^-1 + H^T R^-1 H )^-1
    across the 6 paired error/uncertainty benchmarks:
      (5m / 2°), (10m / 5°), (20m / 10°), (25m / 15°), (50m / 20°), (100m / 30°).
    Disaggregates into cross-track variance, along-track variance, and heading variance.
    """
    print("\n" + "="*80)
    print(" EXPERIMENT 3: The Map Observability Envelope (Theoretical Covariance Reduction)")
    print("="*80)

    benchmarks = [
        (5.0, 2.0, "5m / 2° (Initial Outage, t~3s)"),
        (10.0, 5.0, "10m / 5° (Short Outage, t~7s)"),
        (20.0, 10.0, "20m / 10° (Medium Outage, t~12s)"),
        (25.0, 15.0, "25m / 15° (Critical Boundary, t~18s)"),
        (50.0, 20.0, "50m / 20° (Extended Outage, t~30s)"),
        (100.0, 30.0, "100m / 30° (Severe Outage, t~60s)")
    ]

    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        lats = df_v['veh_lat'].values
        lons = df_v['veh_lon'].values
        lat0, lon0 = float(lats[0]), float(lons[0])

        cache_name = f"{trip}_network"
        osm_path = download_osm_bbox(float(lons.min() - 0.003), float(lats.min() - 0.003),
                                     float(lons.max() + 0.003), float(lats.max() + 0.003),
                                     cache_name=cache_name)
        network = parse_osm_network(osm_path, lat0, lon0)
        index = RoadNetworkIndex(network['segments'])

        # Sample 100 motion epochs
        enu_pts = np.array([geodetic_to_enu(lat, lon, lat0, lon0)[:2] for lat, lon in zip(lats[::100], lons[::100])])
        diff_enu = np.diff(enu_pts, axis=0, prepend=[enu_pts[0]])
        headings_rad = np.arctan2(diff_enu[:, 0], diff_enu[:, 1])

        trip_envelope = {}

        for sig_p, sig_psi_deg, desc in benchmarks:
            sig_psi_rad = np.radians(sig_psi_deg)

            # Prior covariance P^- (15x15)
            P_diag = np.ones(15)
            P_diag[0] = sig_p ** 2
            P_diag[1] = sig_p ** 2
            P_diag[2] = 1.0
            P_diag[3:6] = 1.0 ** 2
            P_diag[6:8] = np.radians(0.5) ** 2
            P_diag[8] = sig_psi_rad ** 2
            P_diag[9:12] = 0.05 ** 2
            P_diag[12:15] = 0.005 ** 2
            P_prior = np.diag(P_diag)

            # Evaluate reductions across conditions
            cond_reductions = {'lateral_only': [], 'heading_only': [], 'joint': []}

            for p, psi in zip(enu_pts, headings_rad):
                H_perp, H_psi, H_joint = compute_map_jacobians(psi)
                R_perp = np.array([[2.5 ** 2]])
                R_psi = np.array([[np.radians(5.0) ** 2]])
                R_joint = compute_map_covariance(sigma_lane=2.5, sigma_psi_rad=np.radians(5.0))

                red_lat = compute_covariance_reduction(P_prior, H_perp, R_perp, psi)
                red_head = compute_covariance_reduction(P_prior, H_psi, R_psi, psi)
                red_joint = compute_covariance_reduction(P_prior, H_joint, R_joint, psi)

                cond_reductions['lateral_only'].append(red_lat)
                cond_reductions['heading_only'].append(red_head)
                cond_reductions['joint'].append(red_joint)

            # Average variance reductions
            def avg_metrics(red_list):
                return {
                    'var_cross_prior_m2': float(np.mean([r['var_cross_prior'] for r in red_list])),
                    'var_cross_post_m2': float(np.mean([r['var_cross_post'] for r in red_list])),
                    'var_cross_reduction_pct': float(np.mean([r['var_cross_reduction_pct'] for r in red_list])),
                    'var_along_prior_m2': float(np.mean([r['var_along_prior'] for r in red_list])),
                    'var_along_post_m2': float(np.mean([r['var_along_post'] for r in red_list])),
                    'var_along_reduction_pct': float(np.mean([r['var_along_reduction_pct'] for r in red_list])),
                    'var_yaw_prior_deg2': float(np.mean([r['var_yaw_prior_deg2'] for r in red_list])),
                    'var_yaw_post_deg2': float(np.mean([r['var_yaw_post_deg2'] for r in red_list])),
                    'var_yaw_reduction_pct': float(np.mean([r['var_yaw_reduction_pct'] for r in red_list]))
                }

            summary = {
                'benchmark_label': desc,
                'sigma_p_m': sig_p,
                'sigma_psi_deg': sig_psi_deg,
                'lateral_only': avg_metrics(cond_reductions['lateral_only']),
                'heading_only': avg_metrics(cond_reductions['heading_only']),
                'joint': avg_metrics(cond_reductions['joint'])
            }

            trip_envelope[f"bench_{int(sig_p)}m_{int(sig_psi_deg)}deg"] = summary

            j = summary['joint']
            print(f"[{trip}] {desc:<40}: Cross Red={j['var_cross_reduction_pct']:5.1f}% ({j['var_cross_prior_m2']:.1f}->{j['var_cross_post_m2']:.1f} m²) | Along Red={j['var_along_reduction_pct']:4.1f}% | Yaw Red={j['var_yaw_reduction_pct']:5.1f}% ({j['var_yaw_prior_deg2']:.1f}->{j['var_yaw_post_deg2']:.1f} deg²)")

        results[trip] = trip_envelope

    return results


# ==============================================================================
# EXPERIMENT 4: STATISTICAL GATING & CANDIDATE REJECTION ANALYSIS
# ==============================================================================

def run_experiment_4_statistical_gating(trips=['Vta02', 'Vta04']) -> dict:
    """
    Evaluates false candidate rejection rates under statistical Chi-Square NIS gating
    vs topological heading gating alone. Tests whether NIS gating safely rejects wrong roads.
    """
    print("\n" + "="*80)
    print(" EXPERIMENT 4: Statistical Chi-Square Gating & Spurious Road Rejection")
    print("="*80)

    results = {}
    chi2_95 = 5.991

    for trip in trips:
        df_p, df_v = load_trip(trip)
        lats = df_v['veh_lat'].values
        lons = df_v['veh_lon'].values
        lat0, lon0 = float(lats[0]), float(lons[0])

        cache_name = f"{trip}_network"
        osm_path = download_osm_bbox(float(lons.min() - 0.003), float(lats.min() - 0.003),
                                     float(lons.max() + 0.003), float(lats.max() + 0.003),
                                     cache_name=cache_name)
        network = parse_osm_network(osm_path, lat0, lon0)
        index = RoadNetworkIndex(network['segments'])

        enu_pts = np.array([geodetic_to_enu(lat, lon, lat0, lon0)[:2] for lat, lon in zip(lats[::20], lons[::20])])
        diff_enu = np.diff(enu_pts, axis=0, prepend=[enu_pts[0]])
        headings_rad = np.arctan2(diff_enu[:, 0], diff_enu[:, 1])

        # Test at moderate uncertainty: sigma_p = 15m, sigma_psi = 6 deg
        P_prior = np.diag([15.0**2, 15.0**2, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, np.radians(6.0)**2, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        R_mat = compute_map_covariance(sigma_lane=2.5, sigma_psi_rad=np.radians(5.0))

        true_accept = 0
        false_accept = 0
        total_queries = 0

        for p, psi in zip(enu_pts, headings_rad):
            # Query all candidates within 30m with vehicle heading
            cands = index.query_candidates(p, radius_m=30.0, veh_heading_rad=psi)
            if len(cands) == 0:
                continue

            # Identify true road as the closest heading-aligned road
            aligned = [c for c in cands if abs(c['delta_heading_rad']) < np.radians(30.0)]
            if len(aligned) == 0:
                continue
            true_way = aligned[0]['way_id']

            for c in cands:
                total_queries += 1
                _, _, H_c = compute_map_jacobians(c['road_heading_rad'])
                r_c = np.array([c['distance_m'] - (-0.90), c['delta_heading_rad']])
                nis = compute_nis(r_c, H_c, P_prior, R_mat)

                is_true_road = (c['way_id'] == true_way)
                is_accepted = (nis <= chi2_95)

                if is_true_road and is_accepted:
                    true_accept += 1
                elif not is_true_road and is_accepted:
                    false_accept += 1

        results[trip] = {
            'trip': trip,
            'total_candidate_evaluations': total_queries,
            'true_road_acceptance_rate': float(true_accept / (len(enu_pts) + 1e-6)),
            'spurious_candidate_acceptance_rate': float(false_accept / (total_queries + 1e-6)),
            'chi2_threshold_used': chi2_95
        }

        print(f"[{trip}] True Road Acceptance Rate = {results[trip]['true_road_acceptance_rate']*100:.1f}% | Spurious Candidate Acceptance Rate = {results[trip]['spurious_candidate_acceptance_rate']*100:.2f}%")

    return results


# ==============================================================================
# PUBLICATION DASHBOARD & MASTER JSON EXPORT
# ==============================================================================

def generate_c8_1_dashboard(exp1_res, exp2_res, exp3_res, exp4_res):
    """Generates a 6-panel diagnostic publication dashboard for Stage C8-1."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Stage C8-1: Offline Map Information Utility & Observability Audit", fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Observable Subspace Eigenvalues
    ax = axes[0, 0]
    eigs_str = exp1_res['Vta04']['joint_straight']['eigenvalues']
    eigs_cur = exp1_res['Vta04']['joint_curve']['eigenvalues']
    x_bars = np.arange(len(eigs_str))
    ax.bar(x_bars - 0.15, eigs_str, width=0.3, label='Straight Road (rho=0)', color='royalblue', alpha=0.8)
    ax.bar(x_bars + 0.15, eigs_cur, width=0.3, label='Curved Road (rho=0.5)', color='coral', alpha=0.8)
    ax.set_xticks(x_bars)
    ax.set_xticklabels(['Lateral Mode (m^-2)', 'Heading Mode (rad^-2)'])
    ax.set_title("1. Fisher Information Eigenvalues ($H^T R^{-1} H$)", fontweight='bold')
    ax.set_ylabel("Information Quantity [log scale]")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Cross-Track Variance Reduction vs Benchmark Level
    ax = axes[0, 1]
    bench_keys = list(exp3_res['Vta04'].keys())
    sig_p_vals = [exp3_res['Vta04'][k]['sigma_p_m'] for k in bench_keys]
    red_cross_lat = [exp3_res['Vta04'][k]['lateral_only']['var_cross_reduction_pct'] for k in bench_keys]
    red_cross_joint = [exp3_res['Vta04'][k]['joint']['var_cross_reduction_pct'] for k in bench_keys]
    red_cross_sub = [exp3_res['Vta02'][k]['joint']['var_cross_reduction_pct'] for k in bench_keys]
    ax.plot(sig_p_vals, red_cross_joint, 'go-', linewidth=2.5, label='Joint (Vta04 Highway)')
    ax.plot(sig_p_vals, red_cross_sub, 'bs--', linewidth=2, label='Joint (Vta02 Suburban)')
    ax.plot(sig_p_vals, red_cross_lat, 'r^:', linewidth=1.5, label='Lateral Only')
    ax.set_title(r"2. Cross-Track Variance Reduction ($\Delta P_{\perp}$)", fontweight='bold')
    ax.set_xlabel(r"Prior Position Uncertainty $\sigma_p$ [m]")
    ax.set_ylabel("Cross-Track Variance Reduction [%]")
    ax.set_ylim(50, 102)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Heading Variance Reduction vs Benchmark Level
    ax = axes[0, 2]
    red_yaw_head = [exp3_res['Vta04'][k]['heading_only']['var_yaw_reduction_pct'] for k in bench_keys]
    red_yaw_joint = [exp3_res['Vta04'][k]['joint']['var_yaw_reduction_pct'] for k in bench_keys]
    red_yaw_sub = [exp3_res['Vta02'][k]['joint']['var_yaw_reduction_pct'] for k in bench_keys]
    ax.plot(sig_p_vals, red_yaw_joint, 'go-', linewidth=2.5, label='Joint (Vta04 Highway)')
    ax.plot(sig_p_vals, red_yaw_sub, 'bs--', linewidth=2, label='Joint (Vta02 Suburban)')
    ax.plot(sig_p_vals, red_yaw_head, 'm^:', linewidth=1.5, label='Heading Only')
    ax.set_title(r"3. Heading Variance Reduction ($\Delta P_{\psi}$)", fontweight='bold')
    ax.set_xlabel(r"Prior Position Uncertainty $\sigma_p$ [m]")
    ax.set_ylabel("Heading Variance Reduction [%]")
    ax.set_ylim(0, 102)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 4: Along-Track Null Information Confirmation
    ax = axes[1, 0]
    red_along_lat = [exp3_res['Vta04'][k]['lateral_only']['var_along_reduction_pct'] for k in bench_keys]
    red_along_joint = [exp3_res['Vta04'][k]['joint']['var_along_reduction_pct'] for k in bench_keys]
    ax.plot(sig_p_vals, red_along_lat, 'ro-', linewidth=2, label='Lateral Only (Strict Null)')
    ax.plot(sig_p_vals, red_along_joint, 'b^--', linewidth=2, label='Joint Lateral + Heading')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title("4. Along-Track Variance Reduction (Preservation Check)", fontweight='bold')
    ax.set_xlabel(r"Prior Position Uncertainty $\sigma_p$ [m]")
    ax.set_ylabel("Along-Track Variance Reduction [%]")
    ax.set_ylim(-1, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Normalized Innovation Squared (NIS) vs Prior Uncertainty
    ax = axes[1, 1]
    u_keys = list(exp2_res['Vta04'].keys())
    u_sigmas = [exp2_res['Vta04'][k]['sigma_p_m'] for k in u_keys]
    nis_med4 = [exp2_res['Vta04'][k]['nis_median'] for k in u_keys]
    nis_p95_4 = [exp2_res['Vta04'][k]['nis_p95'] for k in u_keys]
    nis_med2 = [exp2_res['Vta02'][k]['nis_median'] for k in u_keys]
    ax.plot(u_sigmas, nis_med4, 'go-', linewidth=2, label='NIS Median (Vta04)')
    ax.plot(u_sigmas, nis_p95_4, 'g^--', linewidth=1.5, label='NIS P95 (Vta04)')
    ax.plot(u_sigmas, nis_med2, 'bs-', linewidth=2, label='NIS Median (Vta02)')
    ax.axhline(5.99, color='red', linestyle='--', linewidth=1.5, label=r'$\chi_2^2(0.95)$ Threshold (5.99)')
    ax.set_title("5. Normalized Innovation Squared (NIS) vs. Uncertainty", fontweight='bold')
    ax.set_xlabel(r"Prior Position Uncertainty $\sigma_p$ [m]")
    ax.set_ylabel("NIS (2 DOF)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 6: Candidate Ambiguity & Multi-Acceptance Rate
    ax = axes[1, 2]
    amb_accept4 = [exp2_res['Vta04'][k]['ambiguous_accepted_rate'] * 100 for k in u_keys]
    amb_accept2 = [exp2_res['Vta02'][k]['ambiguous_accepted_rate'] * 100 for k in u_keys]
    ax.plot(u_sigmas, amb_accept4, 'go-', linewidth=2.5, label='Vta04 Highway')
    ax.plot(u_sigmas, amb_accept2, 'bs--', linewidth=2.5, label='Vta02 Suburban')
    ax.axvline(25.0, color='red', linestyle=':', linewidth=2, label='25m Feasibility Boundary')
    ax.set_title("6. Ambiguous Road Multi-Acceptance Rate", fontweight='bold')
    ax.set_xlabel(r"Prior Position Uncertainty $\sigma_p$ [m]")
    ax.set_ylabel("Multi-Road Acceptance Rate [%]")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = FIG_DIR / "c8_1_map_information_audit.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\n[Dashboard] Saved 6-panel diagnostic dashboard to: {out_fig}")


def main():
    print("="*80)
    print(" SIH26168 - Stage C8-1: Offline Map Information Utility & Observability Audit")
    print("="*80)

    # Run Experiments 1 through 4
    exp1 = run_experiment_1_information_matrix()
    exp2 = run_experiment_2_uncertainty_stress_test()
    exp3 = run_experiment_3_observability_envelope()
    exp4 = run_experiment_4_statistical_gating()

    # Generate Publication Dashboard
    generate_c8_1_dashboard(exp1, exp2, exp3, exp4)

    # Export Master JSON
    master_deliverable = {
        'metadata': {
            'stage': 'C8-1',
            'description': 'Offline Map Information Utility & Observability Audit',
            'script': 'experiments/audit_map_information_c8_1.py',
            'rules': 'Strict offline information analysis. Zero ESKF modifications. Quarantined VBOX ground truth.'
        },
        'experiment_1_information_matrix': exp1,
        'experiment_2_uncertainty_stress_test': exp2,
        'experiment_3_observability_envelope': exp3,
        'experiment_4_statistical_gating': exp4
    }

    out_json = RES_DIR / "c8_1_map_information_audit.json"
    with open(out_json, "w") as f:
        json.dump(master_deliverable, f, indent=2)
    print(f"\n[Deliverable] Exported master JSON results to: {out_json}")


if __name__ == '__main__':
    main()
