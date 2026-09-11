"""
Stage C8-11.5: Data-Quality-Gated Rolling Calibration Audit

DIAGNOSTIC ONLY -- does NOT modify production navigation code.

Hypothesis:
"Calibrate only when the available pre-outage data satisfies sufficient motion,
straightness, magnetic stability, and sample-count requirements;
otherwise retain the previous calibration estimate."

Compares four strategies on IDENTICAL candidate outage locations:
A -- Fixed window (5s, 10s, 20s, 30s baselines)
B -- Quality-gated fixed window (5s, 10s, 20s, 30s)
C -- Variable-length quality-gated (5 -> 10 -> 20 -> 30 -> 60 s)
D -- Oracle diagnostic (post-hoc vehicle reference analysis)

Records 12 per-outage metrics for every outage and strategy.
Evaluates the "More good samples vs More seconds" statistical hypothesis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import compute_leveling_matrix, detect_stationary_period


# ============================================================================
# Circular statistics helpers (identical to C8-11.3 / C8-11.4)
# ============================================================================

def angle_diff_deg(a, b):
    """Signed difference a - b wrapped to [-180, 180] degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles, weights=None):
    if len(angles) == 0:
        return np.nan
    r = np.radians(angles)
    if weights is None:
        s, c = np.sum(np.sin(r)), np.sum(np.cos(r))
    else:
        s, c = np.sum(weights * np.sin(r)), np.sum(weights * np.cos(r))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def circular_std_deg(angles):
    if len(angles) <= 1:
        return 0.0
    r = np.radians(angles)
    R = np.sqrt(np.mean(np.sin(r))**2 + np.mean(np.cos(r))**2)
    R = np.clip(R, 1e-12, 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))


def circular_median_deg(angles):
    if len(angles) == 0:
        return np.nan
    if len(angles) == 1:
        return float(angles[0] % 360.0)
    cm = circular_mean_deg(angles)
    res = angle_diff_deg(angles, cm)
    return float((cm + np.median(res)) % 360.0)


# ============================================================================
# Data preparation (identical pipeline to C8-11.3 / C8-11.4)
# ============================================================================

def prepare_trip(trip_name: str) -> Dict[str, Any]:
    """Load trip and compute all derived quantities. Returns dict."""
    df_p, df_v = load_trip(trip_name)
    n = len(df_p)
    dt = 0.1
    time_s = np.arange(n) * dt

    speed_ms = df_p["phone_speed_kmh"].values / 3.6
    gnss_bearing = df_p["phone_bearing_deg"].values
    accel = df_p[["accel_x", "accel_y", "accel_z"]].values
    mag = df_p[["mag_x", "mag_y", "mag_z"]].values
    gyro = df_p[["gyro_x", "gyro_y", "gyro_z"]].values
    lat = df_p["phone_lat"].values
    lon = df_p["phone_lon"].values
    cum_dist_m = np.cumsum(speed_ms * dt)

    veh_speed_ms = df_v["veh_speed_kmh"].values / 3.6
    veh_heading = df_v["veh_heading_deg"].values

    stat_idx = detect_stationary_period(accel, speed_ms)
    g_ref = np.median(accel[stat_idx], axis=0) if len(stat_idx) >= 15 else df_p[["grav_x", "grav_y", "grav_z"]].median().values
    R_level, _, _ = compute_leveling_matrix(g_ref)

    mag_lev = (R_level @ mag.T).T
    cx = 0.5 * (np.max(mag_lev[:, 0]) + np.min(mag_lev[:, 0]))
    cy = 0.5 * (np.max(mag_lev[:, 1]) + np.min(mag_lev[:, 1]))
    cz = 0.5 * (np.max(mag_lev[:, 2]) + np.min(mag_lev[:, 2]))
    m_c = mag_lev - np.array([cx, cy, cz])
    psi_mag = np.degrees(np.arctan2(-m_c[:, 0], -m_c[:, 2])) % 360.0

    mag_norm = np.linalg.norm(mag, axis=1)
    baseline_B = float(np.median(mag_norm))

    gyro_lev = (R_level @ gyro.T).T
    tr_raw = np.abs(np.degrees(gyro_lev[:, 2]))
    tr_smooth = np.convolve(tr_raw, np.ones(10) / 10.0, mode="same")

    is_moving = (speed_ms >= 3.0) & (~np.isnan(gnss_bearing))
    is_straight = is_moving & (tr_smooth < 3.0)

    gnss_offset = angle_diff_deg(gnss_bearing, psi_mag)
    gt_offset = angle_diff_deg(veh_heading, psi_mag)

    return {
        "n": n, "dt": dt, "time_s": time_s, "speed_ms": speed_ms,
        "gnss_bearing": gnss_bearing, "veh_heading": veh_heading,
        "veh_speed_ms": veh_speed_ms, "psi_mag": psi_mag,
        "mag_norm": mag_norm, "baseline_B": baseline_B,
        "cum_dist_m": cum_dist_m, "lat": lat, "lon": lon,
        "is_moving": is_moving, "is_straight": is_straight,
        "gnss_offset": gnss_offset, "gt_offset": gt_offset,
    }


# ============================================================================
# Quality Gate Evaluator
# ============================================================================

def evaluate_quality_gate(
    D: Dict[str, Any],
    pre_slice: np.ndarray,
    straight_samps: np.ndarray,
    min_straight_samples: int = 15,
    max_dispersion_deg: float = 6.0,
    max_mag_std_uT: float = 3.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates data quality requirements on candidate pre-outage window.
    
    Returns:
        passed (bool)
        reason (str)
        stats (dict of metrics)
    """
    n_pre = len(pre_slice)
    n_st = len(straight_samps)
    frac_st = float(n_st / n_pre) if n_pre > 0 else 0.0

    if n_st < min_straight_samples:
        return False, "insufficient_straight_samples", {
            "n_valid_samples": n_st,
            "straight_fraction": frac_st,
            "mean_speed_ms": float(np.mean(D["speed_ms"][straight_samps])) if n_st > 0 else 0.0,
            "course_dispersion_deg": np.nan,
            "mag_norm_mean": np.nan,
            "mag_norm_std": np.nan,
            "mag_norm_dev_pct": np.nan,
        }

    mean_spd = float(np.mean(D["speed_ms"][straight_samps]))
    if mean_spd < 3.0:
        return False, "low_speed", {
            "n_valid_samples": n_st,
            "straight_fraction": frac_st,
            "mean_speed_ms": mean_spd,
            "course_dispersion_deg": np.nan,
            "mag_norm_mean": np.nan,
            "mag_norm_std": np.nan,
            "mag_norm_dev_pct": np.nan,
        }

    offsets = D["gnss_offset"][straight_samps]
    dispersion = circular_std_deg(offsets)
    if dispersion > max_dispersion_deg:
        return False, "excessive_course_dispersion", {
            "n_valid_samples": n_st,
            "straight_fraction": frac_st,
            "mean_speed_ms": mean_spd,
            "course_dispersion_deg": dispersion,
            "mag_norm_mean": float(np.mean(D["mag_norm"][straight_samps])),
            "mag_norm_std": float(np.std(D["mag_norm"][straight_samps])),
            "mag_norm_dev_pct": float(np.abs(np.mean(D["mag_norm"][straight_samps]) - D["baseline_B"]) / D["baseline_B"] * 100.0),
        }

    b_norms = D["mag_norm"][straight_samps]
    b_mean = float(np.mean(b_norms))
    b_std = float(np.std(b_norms))
    b_dev_pct = float(np.abs(b_mean - D["baseline_B"]) / D["baseline_B"] * 100.0)

    if b_std > max_mag_std_uT:
        return False, "magnetic_instability", {
            "n_valid_samples": n_st,
            "straight_fraction": frac_st,
            "mean_speed_ms": mean_spd,
            "course_dispersion_deg": dispersion,
            "mag_norm_mean": b_mean,
            "mag_norm_std": b_std,
            "mag_norm_dev_pct": b_dev_pct,
        }

    return True, "pass", {
        "n_valid_samples": n_st,
        "straight_fraction": frac_st,
        "mean_speed_ms": mean_spd,
        "course_dispersion_deg": dispersion,
        "mag_norm_mean": b_mean,
        "mag_norm_std": b_std,
        "mag_norm_dev_pct": b_dev_pct,
    }


# ============================================================================
# Main audit execution
# ============================================================================

OUTAGE_DUR_S = 30.0
CANDIDATE_WINDOWS_S = [5, 10, 20, 30, 60]
FIXED_BASELINE_WINDOWS_S = [5, 10, 20, 30]


def run_c8_11_5_audit():
    print("=" * 78)
    print("STAGE C8-11.5: DATA-QUALITY-GATED ROLLING CALIBRATION AUDIT")
    print("=" * 78)

    results = {
        "meta": {
            "stage": "C8-11.5",
            "description": "Data-Quality-Gated Rolling Calibration Audit",
            "trips": ["Vta02", "Vta04"],
            "outage_duration_s": OUTAGE_DUR_S,
            "fixed_windows_s": FIXED_BASELINE_WINDOWS_S,
            "expansion_windows_s": CANDIDATE_WINDOWS_S,
            "quality_gates": {
                "strict": {
                    "min_straight_samples": 15,
                    "min_speed_ms": 3.0,
                    "max_dispersion_deg": 6.0,
                    "max_mag_std_uT": 3.0
                },
                "moderate": {
                    "min_straight_samples": 15,
                    "min_speed_ms": 3.0,
                    "max_dispersion_deg": 8.0,
                    "max_mag_std_uT": 3.5
                }
            },
            "constraints": {
                "anti_circularity": "All operational decisions use smartphone channels only; vehicle GT is post-hoc diagnostic only.",
                "identical_outage_locations": "All strategies evaluated on identical candidate outage points. Rejections counted and included in error.",
                "temporal_integrity": "max(t_calib) < t_outage_start enforced strictly for all outages."
            }
        },
        "trips": {}
    }

    for trip_name in ["Vta02", "Vta04"]:
        print(f"\n{'=' * 78}")
        print(f"TRIP: {trip_name}")
        print(f"{'=' * 78}")

        D = prepare_trip(trip_name)
        dt = D["dt"]
        n = D["n"]
        outage_pts = int(OUTAGE_DUR_S / dt)

        # -----------------------------------------------------------------
        # Generate dense candidate outages (IDENTICAL grid to C8-11.4)
        # -----------------------------------------------------------------
        step_s = 15.0 if trip_name == "Vta02" else 10.0
        step_pts = int(step_s / dt)
        min_start = int(60.0 / dt)

        dense_starts = []
        for s in range(min_start, n - outage_pts, step_pts):
            out_sl = np.arange(s, s + outage_pts)
            if np.mean(D["veh_speed_ms"][out_sl]) < 2.5:
                continue
            if np.sum(~np.isnan(D["veh_heading"][out_sl])) < outage_pts * 0.8:
                continue
            pre60 = np.arange(max(0, s - int(60.0 / dt)), s)
            if np.sum(D["is_moving"][pre60]) < 20:
                continue
            dense_starts.append(s)

        print(f"  Dense candidate outages generated: {len(dense_starts)}")

        # Initial frozen calibration from first 60s of motion (pre-outage baseline)
        first_moving = np.where(D["is_moving"])[0]
        init_slice = first_moving[:int(60.0 / dt)] if len(first_moving) >= int(60.0 / dt) else first_moving
        init_offset = circular_mean_deg(D["gnss_offset"][init_slice])
        print(f"  Initial state calibration offset:  {init_offset:5.2f} deg (from first {len(init_slice)*dt:.1f}s motion)")

        # Temporal integrity verification
        for s in dense_starts:
            for w in CANDIDATE_WINDOWS_S:
                w_pts = int(w / dt)
                pre_start = max(0, s - w_pts)
                assert pre_start < s, f"Temporal leakage: start {pre_start} >= outage {s}"

        trip_output: Dict[str, Any] = {
            "num_outages": len(dense_starts),
            "step_s": step_s,
            "init_offset_deg": float(init_offset),
            "strategies": {},
            "gate_breakdown": {},
            "statistical_test": {},
            "oracle_confusion_matrix": {},
            "outage_records": []
        }

        # =================================================================
        # 1. EVALUATE STRATEGY A: FIXED WINDOW BASELINES (5, 10, 20, 30 s)
        # =================================================================
        print(f"\n[STRATEGY A] Fixed Window Baselines (W in {FIXED_BASELINE_WINDOWS_S} s)")
        strat_a_res = {}
        strat_a_logs = {w: [] for w in FIXED_BASELINE_WINDOWS_S}

        for w in FIXED_BASELINE_WINDOWS_S:
            w_pts = int(w / dt)
            last_offset = init_offset
            last_t_s = 0.0
            last_dist_m = 0.0

            outage_maes = []
            rejections = 0
            ages_s = []
            dists_m = []

            for s in dense_starts:
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]

                # Strategy A naive acceptance: N_straight >= 10
                accepted = (len(st_samps) >= 10)
                n_val = len(st_samps)
                frac_st = float(n_val / len(pre_slice)) if len(pre_slice) > 0 else 0.0
                mean_spd = float(np.mean(D["speed_ms"][st_samps])) if n_val > 0 else 0.0
                cstd = circular_std_deg(D["gnss_offset"][st_samps]) if n_val >= 2 else np.nan
                b_norms = D["mag_norm"][st_samps] if n_val > 0 else np.array([])
                b_std = float(np.std(b_norms)) if n_val >= 2 else np.nan

                if accepted:
                    calib_off = circular_mean_deg(D["gnss_offset"][st_samps])
                    last_offset = calib_off
                    last_t_s = float(s * dt)
                    last_dist_m = float(D["cum_dist_m"][s])
                    age_s = 0.0
                    dist_m = 0.0
                else:
                    rejections += 1
                    calib_off = last_offset
                    age_s = float(s * dt - last_t_s)
                    dist_m = float(D["cum_dist_m"][s] - last_dist_m)

                ages_s.append(age_s)
                dists_m.append(dist_m)

                # Evaluate true outage MAE
                out_sl = np.arange(s, s + outage_pts)
                pred_hdg = (D["psi_mag"][out_sl] + calib_off) % 360.0
                err = np.abs(angle_diff_deg(pred_hdg, D["veh_heading"][out_sl]))
                mae = float(np.mean(err))
                outage_maes.append(mae)

                strat_a_logs[w].append({
                    "outage_start_s": float(s * dt),
                    "accepted": accepted,
                    "window_used_s": w if accepted else 0,
                    "n_valid_samples": n_val,
                    "straight_fraction": frac_st,
                    "mean_speed_ms": mean_spd,
                    "course_dispersion_deg": cstd,
                    "mag_norm_std_uT": b_std,
                    "calib_offset_deg": calib_off,
                    "post_outage_mae_deg": mae,
                    "calib_age_s": age_s,
                    "dist_since_calib_m": dist_m,
                    "is_catastrophic": bool(mae > 20.0)
                })

            maes_arr = np.array(outage_maes)
            strat_a_res[f"{w}s"] = {
                "mean_mae_deg": float(np.mean(maes_arr)),
                "median_mae_deg": float(np.median(maes_arr)),
                "rmse_deg": float(np.sqrt(np.mean(maes_arr**2))),
                "p95_mae_deg": float(np.percentile(maes_arr, 95)),
                "max_mae_deg": float(np.max(maes_arr)),
                "rejections": int(rejections),
                "rejection_rate_pct": float(rejections / len(dense_starts) * 100.0),
                "mean_calib_age_s": float(np.mean(ages_s)),
                "mean_dist_since_calib_m": float(np.mean(dists_m)),
                "catastrophic_rate_pct": float(np.mean(maes_arr > 20.0) * 100.0)
            }
            print(f"  W={w:2d}s | Mean: {np.mean(maes_arr):5.2f} deg | Med: {np.median(maes_arr):5.2f} deg | "
                  f"P95: {np.percentile(maes_arr, 95):5.2f} deg | Max: {np.max(maes_arr):5.2f} deg | "
                  f"Rej: {rejections:2d}/{len(dense_starts)} ({rejections/len(dense_starts)*100:4.1f}%) | "
                  f"Age: {np.mean(ages_s):5.1f}s | Catastrophic(>20deg): {np.mean(maes_arr > 20.0)*100:4.1f}%")

        trip_output["strategies"]["strategy_a_fixed"] = strat_a_res

        # =================================================================
        # 2. EVALUATE STRATEGY B: QUALITY-GATED FIXED WINDOW (5, 10, 20, 30 s)
        # =================================================================
        print(f"\n[STRATEGY B] Quality-Gated Fixed Window (Strict: N>=15, cstd<=6deg, bstd<=3uT)")
        strat_b_res = {}
        strat_b_logs = {w: [] for w in FIXED_BASELINE_WINDOWS_S}

        for w in FIXED_BASELINE_WINDOWS_S:
            w_pts = int(w / dt)
            last_offset = init_offset
            last_t_s = 0.0
            last_dist_m = 0.0

            outage_maes = []
            rejections = 0
            ages_s = []
            dists_m = []

            for s in dense_starts:
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]

                passed, reason, q_stats = evaluate_quality_gate(
                    D, pre_slice, st_samps,
                    min_straight_samples=15,
                    max_dispersion_deg=6.0,
                    max_mag_std_uT=3.0
                )

                if passed:
                    calib_off = circular_mean_deg(D["gnss_offset"][st_samps])
                    last_offset = calib_off
                    last_t_s = float(s * dt)
                    last_dist_m = float(D["cum_dist_m"][s])
                    age_s = 0.0
                    dist_m = 0.0
                else:
                    rejections += 1
                    calib_off = last_offset
                    age_s = float(s * dt - last_t_s)
                    dist_m = float(D["cum_dist_m"][s] - last_dist_m)

                ages_s.append(age_s)
                dists_m.append(dist_m)

                # True outage MAE
                out_sl = np.arange(s, s + outage_pts)
                pred_hdg = (D["psi_mag"][out_sl] + calib_off) % 360.0
                err = np.abs(angle_diff_deg(pred_hdg, D["veh_heading"][out_sl]))
                mae = float(np.mean(err))
                outage_maes.append(mae)

                strat_b_logs[w].append({
                    "outage_start_s": float(s * dt),
                    "accepted": passed,
                    "rejection_reason": reason if not passed else "none",
                    "window_used_s": w if passed else 0,
                    "n_valid_samples": q_stats["n_valid_samples"],
                    "straight_fraction": q_stats["straight_fraction"],
                    "mean_speed_ms": q_stats["mean_speed_ms"],
                    "course_dispersion_deg": q_stats["course_dispersion_deg"],
                    "mag_norm_std_uT": q_stats["mag_norm_std"],
                    "calib_offset_deg": calib_off,
                    "post_outage_mae_deg": mae,
                    "calib_age_s": age_s,
                    "dist_since_calib_m": dist_m,
                    "is_catastrophic": bool(mae > 20.0)
                })

            maes_arr = np.array(outage_maes)
            strat_b_res[f"{w}s"] = {
                "mean_mae_deg": float(np.mean(maes_arr)),
                "median_mae_deg": float(np.median(maes_arr)),
                "rmse_deg": float(np.sqrt(np.mean(maes_arr**2))),
                "p95_mae_deg": float(np.percentile(maes_arr, 95)),
                "max_mae_deg": float(np.max(maes_arr)),
                "rejections": int(rejections),
                "rejection_rate_pct": float(rejections / len(dense_starts) * 100.0),
                "mean_calib_age_s": float(np.mean(ages_s)),
                "mean_dist_since_calib_m": float(np.mean(dists_m)),
                "catastrophic_rate_pct": float(np.mean(maes_arr > 20.0) * 100.0)
            }
            print(f"  W={w:2d}s | Mean: {np.mean(maes_arr):5.2f} deg | Med: {np.median(maes_arr):5.2f} deg | "
                  f"P95: {np.percentile(maes_arr, 95):5.2f} deg | Max: {np.max(maes_arr):5.2f} deg | "
                  f"Rej: {rejections:2d}/{len(dense_starts)} ({rejections/len(dense_starts)*100:4.1f}%) | "
                  f"Age: {np.mean(ages_s):5.1f}s | Catastrophic(>20deg): {np.mean(maes_arr > 20.0)*100:4.1f}%")

        trip_output["strategies"]["strategy_b_quality_gated_fixed"] = strat_b_res

        # Also run Moderate Quality Gate for Strategy B (sensitivity check)
        strat_b_mod_res = {}
        for w in FIXED_BASELINE_WINDOWS_S:
            w_pts = int(w / dt)
            last_offset = init_offset
            last_t_s = 0.0
            last_dist_m = 0.0
            outage_maes = []
            rejections = 0
            for s in dense_starts:
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]
                passed, _, _ = evaluate_quality_gate(
                    D, pre_slice, st_samps,
                    min_straight_samples=15,
                    max_dispersion_deg=8.0,
                    max_mag_std_uT=3.5
                )
                if passed:
                    calib_off = circular_mean_deg(D["gnss_offset"][st_samps])
                    last_offset = calib_off
                    last_t_s = float(s * dt)
                    last_dist_m = float(D["cum_dist_m"][s])
                else:
                    rejections += 1
                    calib_off = last_offset
                out_sl = np.arange(s, s + outage_pts)
                pred_hdg = (D["psi_mag"][out_sl] + calib_off) % 360.0
                err = np.abs(angle_diff_deg(pred_hdg, D["veh_heading"][out_sl]))
                outage_maes.append(float(np.mean(err)))
            maes_arr = np.array(outage_maes)
            strat_b_mod_res[f"{w}s"] = {
                "mean_mae_deg": float(np.mean(maes_arr)),
                "median_mae_deg": float(np.median(maes_arr)),
                "p95_mae_deg": float(np.percentile(maes_arr, 95)),
                "rejections": int(rejections),
                "rejection_rate_pct": float(rejections / len(dense_starts) * 100.0)
            }
        trip_output["strategies"]["strategy_b_moderate_gate_fixed"] = strat_b_mod_res

        # =================================================================
        # 3. EVALUATE STRATEGY C: VARIABLE-LENGTH QUALITY-GATED
        # =================================================================
        print(f"\n[STRATEGY C] Variable-Length Quality-Gated (5 -> 10 -> 20 -> 30 -> 60 s)")
        last_offset = init_offset
        last_t_s = 0.0
        last_dist_m = 0.0

        strat_c_logs = []
        c_maes = []
        c_windows_used = []
        c_rejections = 0
        c_ages_s = []
        c_dists_m = []

        for s in dense_starts:
            accepted = False
            chosen_w = 0
            chosen_stats = {}

            for w in CANDIDATE_WINDOWS_S:
                w_pts = int(w / dt)
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]

                passed, reason, q_stats = evaluate_quality_gate(
                    D, pre_slice, st_samps,
                    min_straight_samples=15,
                    max_dispersion_deg=6.0,
                    max_mag_std_uT=3.0
                )

                if passed:
                    accepted = True
                    chosen_w = w
                    calib_off = circular_mean_deg(D["gnss_offset"][st_samps])
                    last_offset = calib_off
                    last_t_s = float(s * dt)
                    last_dist_m = float(D["cum_dist_m"][s])
                    age_s = 0.0
                    dist_m = 0.0
                    chosen_stats = q_stats
                    break

            if not accepted:
                c_rejections += 1
                chosen_w = 0
                calib_off = last_offset
                age_s = float(s * dt - last_t_s)
                dist_m = float(D["cum_dist_m"][s] - last_dist_m)
                chosen_stats = {
                    "n_valid_samples": 0,
                    "straight_fraction": 0.0,
                    "mean_speed_ms": 0.0,
                    "course_dispersion_deg": np.nan,
                    "mag_norm_std": np.nan,
                }

            c_windows_used.append(chosen_w)
            c_ages_s.append(age_s)
            c_dists_m.append(dist_m)

            out_sl = np.arange(s, s + outage_pts)
            pred_hdg = (D["psi_mag"][out_sl] + calib_off) % 360.0
            err = np.abs(angle_diff_deg(pred_hdg, D["veh_heading"][out_sl]))
            mae = float(np.mean(err))
            c_maes.append(mae)

            strat_c_logs.append({
                "outage_start_s": float(s * dt),
                "accepted": accepted,
                "window_used_s": chosen_w,
                "n_valid_samples": chosen_stats["n_valid_samples"],
                "straight_fraction": chosen_stats["straight_fraction"],
                "mean_speed_ms": chosen_stats["mean_speed_ms"],
                "course_dispersion_deg": chosen_stats["course_dispersion_deg"],
                "mag_norm_std_uT": chosen_stats["mag_norm_std"],
                "calib_offset_deg": calib_off,
                "post_outage_mae_deg": mae,
                "calib_age_s": age_s,
                "dist_since_calib_m": dist_m,
                "is_catastrophic": bool(mae > 20.0)
            })

        c_maes_arr = np.array(c_maes)
        w_counts = {int(k): int(v) for k, v in zip(*np.unique(c_windows_used, return_counts=True))}
        trip_output["strategies"]["strategy_c_variable_length"] = {
            "mean_mae_deg": float(np.mean(c_maes_arr)),
            "median_mae_deg": float(np.median(c_maes_arr)),
            "rmse_deg": float(np.sqrt(np.mean(c_maes_arr**2))),
            "p95_mae_deg": float(np.percentile(c_maes_arr, 95)),
            "max_mae_deg": float(np.max(c_maes_arr)),
            "rejections": int(c_rejections),
            "rejection_rate_pct": float(c_rejections / len(dense_starts) * 100.0),
            "mean_calib_age_s": float(np.mean(c_ages_s)),
            "mean_dist_since_calib_m": float(np.mean(c_dists_m)),
            "catastrophic_rate_pct": float(np.mean(c_maes_arr > 20.0) * 100.0),
            "window_usage_histogram": w_counts
        }
        print(f"  Strategy C | Mean: {np.mean(c_maes_arr):5.2f} deg | Med: {np.median(c_maes_arr):5.2f} deg | "
              f"P95: {np.percentile(c_maes_arr, 95):5.2f} deg | Max: {np.max(c_maes_arr):5.2f} deg | "
              f"Rej: {c_rejections:2d}/{len(dense_starts)} ({c_rejections/len(dense_starts)*100:4.1f}%) | "
              f"Age: {np.mean(c_ages_s):5.1f}s | Catastrophic(>20deg): {np.mean(c_maes_arr > 20.0)*100:4.1f}%")
        print(f"  Window Usage Histogram: {w_counts}")

        # Variable-length with Moderate Gate
        c_mod_maes = []
        c_mod_rejections = 0
        c_mod_windows = []
        last_offset = init_offset
        for s in dense_starts:
            accepted = False
            chosen_w = 0
            for w in CANDIDATE_WINDOWS_S:
                w_pts = int(w / dt)
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]
                passed, _, _ = evaluate_quality_gate(
                    D, pre_slice, st_samps,
                    min_straight_samples=15,
                    max_dispersion_deg=8.0,
                    max_mag_std_uT=3.5
                )
                if passed:
                    accepted = True
                    chosen_w = w
                    calib_off = circular_mean_deg(D["gnss_offset"][st_samps])
                    last_offset = calib_off
                    break
            if not accepted:
                c_mod_rejections += 1
                chosen_w = 0
                calib_off = last_offset
            c_mod_windows.append(chosen_w)
            out_sl = np.arange(s, s + outage_pts)
            pred_hdg = (D["psi_mag"][out_sl] + calib_off) % 360.0
            err = np.abs(angle_diff_deg(pred_hdg, D["veh_heading"][out_sl]))
            c_mod_maes.append(float(np.mean(err)))
        c_mod_arr = np.array(c_mod_maes)
        w_mod_counts = {int(k): int(v) for k, v in zip(*np.unique(c_mod_windows, return_counts=True))}
        trip_output["strategies"]["strategy_c_moderate_gate"] = {
            "mean_mae_deg": float(np.mean(c_mod_arr)),
            "median_mae_deg": float(np.median(c_mod_arr)),
            "p95_mae_deg": float(np.percentile(c_mod_arr, 95)),
            "rejections": int(c_mod_rejections),
            "rejection_rate_pct": float(c_mod_rejections / len(dense_starts) * 100.0),
            "window_usage_histogram": w_mod_counts
        }

        # =================================================================
        # 4. EVALUATE STRATEGY D: ORACLE DIAGNOSTIC (POST-HOC REFERENCE)
        # =================================================================
        print(f"\n[STRATEGY D] Oracle Diagnostic (Post-Hoc Vehicle Reference Analysis)")
        oracle_maes = []
        oracle_best_windows = []
        oracle_per_window_maes = {w: [] for w in CANDIDATE_WINDOWS_S}

        # Gate Confusion Matrix collectors (using 10 deg MAE as good threshold)
        # For each candidate window at each outage:
        # was it accepted by gate? was its true error <= 10 deg?
        tp, fp, tn, fn = 0, 0, 0, 0

        for s in dense_starts:
            out_sl = np.arange(s, s + outage_pts)
            gt_h = D["veh_heading"][out_sl]

            best_mae = 999.0
            best_w = 0

            for w in CANDIDATE_WINDOWS_S:
                w_pts = int(w / dt)
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]

                passed, _, _ = evaluate_quality_gate(
                    D, pre_slice, st_samps,
                    min_straight_samples=15,
                    max_dispersion_deg=6.0,
                    max_mag_std_uT=3.0
                )

                if len(st_samps) >= 5:
                    off = circular_mean_deg(D["gnss_offset"][st_samps])
                    pred = (D["psi_mag"][out_sl] + off) % 360.0
                    mae_w = float(np.mean(np.abs(angle_diff_deg(pred, gt_h))))
                    oracle_per_window_maes[w].append(mae_w)

                    if mae_w < best_mae:
                        best_mae = mae_w
                        best_w = w

                    # Confusion matrix check
                    is_truly_good = (mae_w <= 10.0)
                    if passed and is_truly_good:
                        tp += 1
                    elif passed and not is_truly_good:
                        fp += 1
                    elif not passed and not is_truly_good:
                        tn += 1
                    elif not passed and is_truly_good:
                        fn += 1
                else:
                    oracle_per_window_maes[w].append(np.nan)
                    if not passed:
                        tn += 1

            if best_mae < 999.0:
                oracle_maes.append(best_mae)
                oracle_best_windows.append(best_w)
            else:
                # Fallback to initial
                pred = (D["psi_mag"][out_sl] + init_offset) % 360.0
                mae_init = float(np.mean(np.abs(angle_diff_deg(pred, gt_h))))
                oracle_maes.append(mae_init)
                oracle_best_windows.append(0)

        orc_maes_arr = np.array(oracle_maes)
        orc_counts = {int(k): int(v) for k, v in zip(*np.unique(oracle_best_windows, return_counts=True))}

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

        trip_output["strategies"]["strategy_d_oracle"] = {
            "mean_mae_deg": float(np.mean(orc_maes_arr)),
            "median_mae_deg": float(np.median(orc_maes_arr)),
            "rmse_deg": float(np.sqrt(np.mean(orc_maes_arr**2))),
            "p95_mae_deg": float(np.percentile(orc_maes_arr, 95)),
            "max_mae_deg": float(np.max(orc_maes_arr)),
            "oracle_optimal_window_histogram": orc_counts
        }
        trip_output["oracle_confusion_matrix"] = {
            "threshold_good_mae_deg": 10.0,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy
        }

        print(f"  Strategy D (Oracle Best Ceiling): Mean MAE = {np.mean(orc_maes_arr):5.2f} deg | "
              f"Med = {np.median(orc_maes_arr):5.2f} deg | P95 = {np.percentile(orc_maes_arr, 95):5.2f} deg")
        print(f"  Oracle Optimal Window Distribution: {orc_counts}")
        print(f"  Gate Confusion Matrix (Threshold <= 10 deg):")
        print(f"    TP={tp:3d} | FP={fp:3d} | TN={tn:3d} | FN={fn:3d}")
        print(f"    Precision: {precision*100:5.1f}% | Recall: {recall*100:5.1f}% | "
              f"Specificity: {specificity*100:5.1f}% | Accuracy: {accuracy*100:5.1f}%")

        # =================================================================
        # 5. SPECIAL STATISTICAL TEST: SAMPLES VS SECONDS
        # =================================================================
        print(f"\n[STATISTICAL TEST] More Good Samples vs More Seconds")
        all_calibs = []
        for s in dense_starts:
            out_sl = np.arange(s, s + outage_pts)
            gt_h = D["veh_heading"][out_sl]

            for w in CANDIDATE_WINDOWS_S:
                w_pts = int(w / dt)
                pre_slice = np.arange(max(0, s - w_pts), s)
                st_samps = pre_slice[D["is_straight"][pre_slice]]

                if len(st_samps) >= 5:
                    off = circular_mean_deg(D["gnss_offset"][st_samps])
                    pred = (D["psi_mag"][out_sl] + off) % 360.0
                    mae = float(np.mean(np.abs(angle_diff_deg(pred, gt_h))))
                    cstd = circular_std_deg(D["gnss_offset"][st_samps])
                    b_norms = D["mag_norm"][st_samps]
                    b_std = float(np.std(b_norms))
                    b_dev = float(np.abs(np.mean(b_norms) - D["baseline_B"]) / D["baseline_B"])
                    mean_spd = float(np.mean(D["speed_ms"][st_samps]))

                    all_calibs.append({
                        "outage_idx": s,
                        "w_s": w,
                        "n_samples": len(st_samps),
                        "frac_straight": len(st_samps) / len(pre_slice),
                        "mean_speed_ms": mean_spd,
                        "course_dispersion_deg": cstd,
                        "mag_norm_std_uT": b_std,
                        "mag_norm_dev_pct": b_dev * 100.0,
                        "post_outage_mae_deg": mae
                    })

        corr_results = {}
        maes_arr = np.array([c["post_outage_mae_deg"] for c in all_calibs])
        print(f"  Evaluated N = {len(all_calibs)} candidate calibration segments:")
        print(f"  {'Predictor':>24s} | {'Pearson r':>10s} | {'p-value':>10s} | {'Spearman rho':>12s} | {'p-value':>10s}")
        print(f"  " + "-" * 74)

        for feat in ["w_s", "n_samples", "frac_straight", "mean_speed_ms",
                     "course_dispersion_deg", "mag_norm_std_uT", "mag_norm_dev_pct"]:
            vals = np.array([c[feat] for c in all_calibs])
            r_p, p_p = stats.pearsonr(vals, maes_arr)
            r_s, p_s = stats.spearmanr(vals, maes_arr)
            corr_results[feat] = {
                "pearson_r": float(r_p),
                "pearson_p": float(p_p),
                "spearman_rho": float(r_s),
                "spearman_p": float(p_s)
            }
            print(f"  {feat:>24s} | {r_p:+10.3f} | {p_p:10.4f} | {r_s:+12.3f} | {p_s:10.4f}")

        # Within Common-Support (where all windows have >= 10 straight samples)
        cs_starts = []
        for s in dense_starts:
            all_w_ok = True
            for w in [5, 10, 20, 30]:
                w_pts = int(w / dt)
                pre_slice = np.arange(max(0, s - w_pts), s)
                if np.sum(D["is_straight"][pre_slice]) < 10:
                    all_w_ok = False
                    break
            if all_w_ok:
                cs_starts.append(s)

        cs_calibs = [c for c in all_calibs if c["outage_idx"] in cs_starts and c["w_s"] in [5, 10, 20, 30]]
        cs_corr = {}
        if len(cs_calibs) > 0:
            cs_maes = np.array([c["post_outage_mae_deg"] for c in cs_calibs])
            print(f"\n  Within Common-Support Subset (N_outages = {len(cs_starts)}, N_calibs = {len(cs_calibs)}):")
            print(f"  {'Predictor':>24s} | {'Pearson r':>10s} | {'p-value':>10s} | {'Spearman rho':>12s} | {'p-value':>10s}")
            print(f"  " + "-" * 74)
            for feat in ["w_s", "n_samples", "frac_straight", "mean_speed_ms",
                         "course_dispersion_deg", "mag_norm_std_uT", "mag_norm_dev_pct"]:
                vals = np.array([c[feat] for c in cs_calibs])
                r_p, p_p = stats.pearsonr(vals, cs_maes)
                r_s, p_s = stats.spearmanr(vals, cs_maes)
                cs_corr[feat] = {
                    "pearson_r": float(r_p),
                    "pearson_p": float(p_p),
                    "spearman_rho": float(r_s),
                    "spearman_p": float(p_s)
                }
                print(f"  {feat:>24s} | {r_p:+10.3f} | {p_p:10.4f} | {r_s:+12.3f} | {p_s:10.4f}")

        trip_output["statistical_test"] = {
            "all_calibrations": corr_results,
            "common_support_calibrations": cs_corr,
            "num_common_support_outages": len(cs_starts)
        }

        # Save merged outage logs for trip
        # Each record merges Strategy A (30s), Strategy B (10s), Strategy C, and Strategy D
        for i, s in enumerate(dense_starts):
            trip_output["outage_records"].append({
                "outage_index": i,
                "outage_start_s": float(s * dt),
                "distance_km": float(D["cum_dist_m"][s] / 1000.0),
                "strategy_a_30s": strat_a_logs[30][i],
                "strategy_b_10s": strat_b_logs[10][i],
                "strategy_b_30s": strat_b_logs[30][i],
                "strategy_c": strat_c_logs[i],
                "oracle_optimal_window_s": oracle_best_windows[i],
                "oracle_best_mae_deg": oracle_maes[i]
            })

        results["trips"][trip_name] = trip_output

    # Save JSON results
    out_json = REPO_ROOT / "results" / "c8_11_5_quality_gated_calibration.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive JSON results to: {out_json}")

    return results


if __name__ == "__main__":
    run_c8_11_5_audit()
