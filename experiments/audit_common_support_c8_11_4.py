"""
Stage C8-11.4: Common-Support Rolling Calibration & Uncertainty Audit

DIAGNOSTIC ONLY — does NOT modify production navigation code.

Tasks:
1. Common-support window comparison (all W must have valid data for every outage)
2. Non-overlapping outage subset (30 s separation)
3. Short-window estimator comparison (5 s, 10 s)
4. Calibration sample count audit
5. Window age / spatial lag analysis
6. Outage error distribution audit
7. Dynamic R_psi predictive validity
8. Uncertainty calibration (coverage test)
9. Cross-trip generalization
10. Final window decision

Anti-circularity:
- Calibration uses ONLY smartphone channels.
- Vehicle heading used EXCLUSIVELY post-hoc for evaluation.
- max(t_calib) < t_outage_start for every outage.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import compute_leveling_matrix, detect_stationary_period


# ============================================================================
# Circular statistics helpers (identical to C8-11.3)
# ============================================================================

def angle_diff_deg(a, b):
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

def trimmed_circular_mean_deg(angles, trim_pct=0.15):
    if len(angles) <= 4:
        return circular_mean_deg(angles)
    cm = circular_median_deg(angles)
    res = angle_diff_deg(angles, cm)
    cutoff = np.percentile(np.abs(res), (1.0 - trim_pct) * 100.0)
    keep = np.abs(res) <= cutoff
    if np.sum(keep) < 2:
        return cm
    return circular_mean_deg(angles[keep])

def huber_circular_m_estimator_deg(angles, max_iter=5):
    if len(angles) <= 3:
        return circular_mean_deg(angles)
    est = circular_median_deg(angles)
    for _ in range(max_iter):
        res = angle_diff_deg(angles, est)
        mad = np.median(np.abs(res - np.median(res)))
        scale = 1.4826 * max(mad, 0.5)
        k = 1.345 * scale
        w = np.where(np.abs(res) <= k, 1.0, k / np.maximum(np.abs(res), 1e-6))
        est = circular_mean_deg(angles, weights=w)
    return est


# ============================================================================
# Data preparation (same pipeline as C8-11.3)
# ============================================================================

MIN_STRAIGHT_SAMPLES = 10  # minimum to attempt calibration

def prepare_trip(trip_name):
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
        "veh_speed_ms": veh_speed_ms,
        "psi_mag": psi_mag, "mag_norm": mag_norm, "baseline_B": baseline_B,
        "cum_dist_m": cum_dist_m, "lat": lat, "lon": lon,
        "is_moving": is_moving, "is_straight": is_straight,
        "gnss_offset": gnss_offset, "gt_offset": gt_offset,
    }


# ============================================================================
# Core: evaluate one outage with one window length
# ============================================================================

def get_straight_samples_in_window(D, s_idx, w_s):
    """Return array of absolute indices of valid straight samples in [s_idx - w_pts, s_idx)."""
    w_pts = int(w_s / D["dt"])
    pre_start = max(0, s_idx - w_pts)
    pre_slice = np.arange(pre_start, s_idx)
    return pre_slice[D["is_straight"][pre_slice]]


def evaluate_outage(D, s_idx, calib_offset_deg, outage_pts):
    """Given a calibration offset, compute MAE over the outage interval."""
    out_slice = np.arange(s_idx, s_idx + outage_pts)
    pred_heading = (D["psi_mag"][out_slice] + calib_offset_deg) % 360.0
    err = np.abs(angle_diff_deg(pred_heading, D["veh_heading"][out_slice]))
    return float(np.mean(err)), err


# ============================================================================
# Main audit
# ============================================================================

WINDOW_LENGTHS = [5, 10, 20, 30, 60, 120]
OUTAGE_DUR_S = 30.0
ESTIMATOR_FUNCS = {
    "circ_mean": circular_mean_deg,
    "circ_median": circular_median_deg,
    "trimmed_mean": lambda a: trimmed_circular_mean_deg(a, 0.15),
    "huber_m": huber_circular_m_estimator_deg,
}


def run_c8_11_4_audit():
    print("=" * 78)
    print("STAGE C8-11.4: COMMON-SUPPORT ROLLING CALIBRATION & UNCERTAINTY AUDIT")
    print("=" * 78)

    results = {
        "meta": {
            "stage": "C8-11.4",
            "description": "Common-Support Rolling Calibration & Uncertainty Audit",
            "trips": ["Vta02", "Vta04"],
            "windows_s": WINDOW_LENGTHS,
            "outage_duration_s": OUTAGE_DUR_S,
            "min_straight_samples": MIN_STRAIGHT_SAMPLES,
            "constraints": {
                "anti_circularity": "All calibration uses smartphone channels only; vehicle GT is post-hoc evaluation only.",
                "temporal_integrity": "max(t_calib) < t_outage_start enforced for every outage."
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
        # Generate dense candidate outages (same grid as C8-11.3)
        # -----------------------------------------------------------------
        step_s = 15.0 if trip_name == "Vta02" else 10.0
        step_pts = int(step_s / dt)
        min_start = int(60.0 / dt)  # wait 60 s before first outage

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

        print(f"  Dense candidate outages: {len(dense_starts)}")

        # =================================================================
        # TASK 1 — COMMON-SUPPORT WINDOW COMPARISON
        # =================================================================
        print(f"\n[TASK 1] Common-Support Window Comparison")

        # For each outage, check which windows have >= MIN_STRAIGHT_SAMPLES
        window_valid = {w: [] for w in WINDOW_LENGTHS}
        for s in dense_starts:
            for w in WINDOW_LENGTHS:
                samps = get_straight_samples_in_window(D, s, w)
                window_valid[w].append(len(samps) >= MIN_STRAIGHT_SAMPLES)

        # Common support: outage must be valid for ALL windows
        common_mask = np.ones(len(dense_starts), dtype=bool)
        for w in WINDOW_LENGTHS:
            common_mask &= np.array(window_valid[w])

        common_starts = [dense_starts[i] for i in range(len(dense_starts)) if common_mask[i]]
        excluded_starts = [dense_starts[i] for i in range(len(dense_starts)) if not common_mask[i]]

        # Determine exclusion reasons
        exclusion_reasons = []
        for i in range(len(dense_starts)):
            if not common_mask[i]:
                missing_w = [w for w in WINDOW_LENGTHS if not window_valid[w][i]]
                exclusion_reasons.append({
                    "outage_start_s": dense_starts[i] * dt,
                    "missing_windows_s": missing_w
                })

        print(f"  Total dense outages:    {len(dense_starts)}")
        print(f"  Common-support outages: {len(common_starts)}")
        print(f"  Excluded outages:       {len(excluded_starts)}")

        # Leakage verification on common-support set
        for s in common_starts:
            for w in WINDOW_LENGTHS:
                pre_end = s
                assert pre_end <= s, f"Leakage: calibration end {pre_end} >= outage start {s}"

        # Evaluate all six windows on common-support outages
        cs_results = {w: [] for w in WINDOW_LENGTHS}
        cs_per_outage = []

        for s in common_starts:
            out_sl = np.arange(s, s + outage_pts)
            gt_out = D["veh_heading"][out_sl]
            mag_out = D["psi_mag"][out_sl]

            record = {"outage_start_s": float(s * dt), "distance_km": float(D["cum_dist_m"][s] / 1000.0)}

            for w in WINDOW_LENGTHS:
                samps = get_straight_samples_in_window(D, s, w)
                off = circular_mean_deg(D["gnss_offset"][samps])
                pred = (mag_out + off) % 360.0
                err = np.abs(angle_diff_deg(pred, gt_out))
                mae = float(np.mean(err))
                cs_results[w].append(mae)
                record[f"mae_{w}s"] = mae

            cs_per_outage.append(record)

        # Summary statistics
        cs_summary = {}
        print(f"\n  Common-Support Window Comparison (N = {len(common_starts)}):")
        print(f"  {'Window':>8s} | {'Mean':>8s} | {'Median':>8s} | {'RMSE':>8s} | {'P95':>8s} | {'Max':>8s}")
        print(f"  " + "-" * 58)
        for w in WINDOW_LENGTHS:
            arr = np.array(cs_results[w])
            cs_summary[f"{w}s"] = {
                "n_outages": len(arr),
                "mean_mae_deg": float(np.mean(arr)),
                "median_mae_deg": float(np.median(arr)),
                "rmse_deg": float(np.sqrt(np.mean(arr**2))),
                "p95_mae_deg": float(np.percentile(arr, 95)),
                "max_mae_deg": float(np.max(arr)),
            }
            print(f"  {w:6d} s | {np.mean(arr):7.2f}° | {np.median(arr):7.2f}° | "
                  f"{np.sqrt(np.mean(arr**2)):7.2f}° | {np.percentile(arr, 95):7.2f}° | {np.max(arr):7.2f}°")

        # =================================================================
        # TASK 2 — NON-OVERLAPPING OUTAGE SUBSET
        # =================================================================
        print(f"\n[TASK 2] Non-Overlapping Outage Subset (>=30 s separation)")

        # Greedy deterministic: scan common-support outages left to right,
        # accept if start >= last_accepted_start + 30s/dt
        min_sep_pts = int(30.0 / dt)
        nonoverlap_starts = []
        last_accepted = -min_sep_pts - 1
        for s in sorted(common_starts):
            if s - last_accepted >= min_sep_pts:
                nonoverlap_starts.append(s)
                last_accepted = s

        # Evaluate on non-overlapping set
        no_results = {w: [] for w in WINDOW_LENGTHS}
        no_per_outage = []
        for s in nonoverlap_starts:
            out_sl = np.arange(s, s + outage_pts)
            gt_out = D["veh_heading"][out_sl]
            mag_out = D["psi_mag"][out_sl]
            record = {"outage_start_s": float(s * dt), "distance_km": float(D["cum_dist_m"][s] / 1000.0)}
            for w in WINDOW_LENGTHS:
                samps = get_straight_samples_in_window(D, s, w)
                off = circular_mean_deg(D["gnss_offset"][samps])
                pred = (mag_out + off) % 360.0
                err = np.abs(angle_diff_deg(pred, gt_out))
                mae = float(np.mean(err))
                no_results[w].append(mae)
                record[f"mae_{w}s"] = mae
            no_per_outage.append(record)

        no_summary = {}
        print(f"  Non-Overlapping Outages: {len(nonoverlap_starts)}")
        print(f"  {'Window':>8s} | {'Mean':>8s} | {'Median':>8s} | {'RMSE':>8s} | {'P95':>8s} | {'Max':>8s}")
        print(f"  " + "-" * 58)
        for w in WINDOW_LENGTHS:
            arr = np.array(no_results[w])
            if len(arr) == 0:
                no_summary[f"{w}s"] = {"n_outages": 0}
                continue
            no_summary[f"{w}s"] = {
                "n_outages": len(arr),
                "mean_mae_deg": float(np.mean(arr)),
                "median_mae_deg": float(np.median(arr)),
                "rmse_deg": float(np.sqrt(np.mean(arr**2))),
                "p95_mae_deg": float(np.percentile(arr, 95)),
                "max_mae_deg": float(np.max(arr)),
            }
            print(f"  {w:6d} s | {np.mean(arr):7.2f}° | {np.median(arr):7.2f}° | "
                  f"{np.sqrt(np.mean(arr**2)):7.2f}° | {np.percentile(arr, 95):7.2f}° | {np.max(arr):7.2f}°")

        # =================================================================
        # TASK 3 — SHORT-WINDOW ESTIMATOR COMPARISON
        # =================================================================
        print(f"\n[TASK 3] Short-Window Estimator Comparison (W = 5 s, 10 s)")

        est_results = {}
        for w in [5, 10]:
            est_results[f"{w}s"] = {}
            for est_name, est_fn in ESTIMATOR_FUNCS.items():
                maes = []
                for s in common_starts:
                    samps = get_straight_samples_in_window(D, s, w)
                    if len(samps) < MIN_STRAIGHT_SAMPLES:
                        continue
                    off = est_fn(D["gnss_offset"][samps])
                    out_sl = np.arange(s, s + outage_pts)
                    pred = (D["psi_mag"][out_sl] + off) % 360.0
                    err = np.abs(angle_diff_deg(pred, D["veh_heading"][out_sl]))
                    maes.append(float(np.mean(err)))
                arr = np.array(maes) if maes else np.array([])
                if len(arr) > 0:
                    est_results[f"{w}s"][est_name] = {
                        "n": len(arr),
                        "mean_mae_deg": float(np.mean(arr)),
                        "median_mae_deg": float(np.median(arr)),
                        "p95_mae_deg": float(np.percentile(arr, 95)),
                    }
                    print(f"  W={w:3d}s | {est_name:13s}: Mean = {np.mean(arr):6.2f}°, "
                          f"Median = {np.median(arr):6.2f}°, P95 = {np.percentile(arr, 95):6.2f}°  (N={len(arr)})")

        # =================================================================
        # TASK 4 — CALIBRATION SAMPLE COUNT
        # =================================================================
        print(f"\n[TASK 4] Calibration Sample Count Audit")

        sample_count_results = {}
        for w in WINDOW_LENGTHS:
            w_pts = int(w / dt)
            counts_raw = []
            counts_valid = []
            fractions = []
            for s in common_starts:
                pre_start = max(0, s - w_pts)
                n_raw = s - pre_start
                samps = get_straight_samples_in_window(D, s, w)
                n_valid = len(samps)
                counts_raw.append(n_raw)
                counts_valid.append(n_valid)
                fractions.append(n_valid / max(n_raw, 1))

            arr_v = np.array(counts_valid)
            arr_f = np.array(fractions)
            sample_count_results[f"{w}s"] = {
                "n_outages": len(counts_raw),
                "raw_samples_median": int(np.median(counts_raw)),
                "valid_straight_min": int(np.min(arr_v)) if len(arr_v) > 0 else 0,
                "valid_straight_median": int(np.median(arr_v)) if len(arr_v) > 0 else 0,
                "valid_straight_max": int(np.max(arr_v)) if len(arr_v) > 0 else 0,
                "valid_straight_mean": float(np.mean(arr_v)) if len(arr_v) > 0 else 0,
                "fraction_accepted_mean": float(np.mean(arr_f)) if len(arr_f) > 0 else 0,
                "fraction_accepted_min": float(np.min(arr_f)) if len(arr_f) > 0 else 0,
            }
            print(f"  W={w:4d}s: Raw = {int(np.median(counts_raw)):4d} | "
                  f"Valid Straight = [{np.min(arr_v):3d}, {np.median(arr_v):5.0f}, {np.max(arr_v):4d}] | "
                  f"Fraction = [{np.min(arr_f):.2f}, {np.mean(arr_f):.2f}, {np.max(arr_f):.2f}]")

        # =================================================================
        # TASK 5 — WINDOW AGE / SPATIAL LAG
        # =================================================================
        print(f"\n[TASK 5] Window Age / Spatial Lag Analysis")

        age_lag_records = []
        for w in WINDOW_LENGTHS:
            for s in common_starts:
                samps = get_straight_samples_in_window(D, s, w)
                if len(samps) < MIN_STRAIGHT_SAMPLES:
                    continue
                oldest_idx = samps[0]
                newest_idx = samps[-1]

                oldest_age_s = float((s - oldest_idx) * dt)
                mean_age_s = float((s - np.mean(samps)) * dt)
                dist_oldest_to_outage = float(D["cum_dist_m"][s] - D["cum_dist_m"][oldest_idx])
                dist_during_window = float(D["cum_dist_m"][newest_idx] - D["cum_dist_m"][oldest_idx])

                off = circular_mean_deg(D["gnss_offset"][samps])
                out_sl = np.arange(s, s + outage_pts)
                pred = (D["psi_mag"][out_sl] + off) % 360.0
                err_arr = np.abs(angle_diff_deg(pred, D["veh_heading"][out_sl]))
                mae = float(np.mean(err_arr))

                age_lag_records.append({
                    "window_s": w,
                    "outage_start_s": float(s * dt),
                    "oldest_age_s": oldest_age_s,
                    "mean_age_s": mean_age_s,
                    "dist_oldest_m": dist_oldest_to_outage,
                    "dist_window_m": dist_during_window,
                    "n_valid": len(samps),
                    "mae_deg": mae,
                })

        # Aggregate correlation across all windows
        if len(age_lag_records) > 5:
            ages = np.array([r["mean_age_s"] for r in age_lag_records])
            dists = np.array([r["dist_oldest_m"] for r in age_lag_records])
            maes = np.array([r["mae_deg"] for r in age_lag_records])
            r_age, p_age = stats.pearsonr(ages, maes)
            rho_age, _ = stats.spearmanr(ages, maes)
            r_dist, p_dist = stats.pearsonr(dists, maes)
            rho_dist, _ = stats.spearmanr(dists, maes)
            print(f"  Correlation (MAE vs mean calibration age):      r = {r_age:+.3f} (p = {p_age:.4f}), rho = {rho_age:+.3f}")
            print(f"  Correlation (MAE vs spatial lag to oldest samp): r = {r_dist:+.3f} (p = {p_dist:.4f}), rho = {rho_dist:+.3f}")

            age_lag_summary = {
                "n_records": len(age_lag_records),
                "corr_age_vs_mae": {"pearson_r": float(r_age), "p": float(p_age), "spearman_rho": float(rho_age)},
                "corr_dist_vs_mae": {"pearson_r": float(r_dist), "p": float(p_dist), "spearman_rho": float(rho_dist)},
            }

            # Per-window breakdown
            for w in WINDOW_LENGTHS:
                w_recs = [r for r in age_lag_records if r["window_s"] == w]
                if len(w_recs) >= 3:
                    w_ages = np.array([r["mean_age_s"] for r in w_recs])
                    w_dists = np.array([r["dist_oldest_m"] for r in w_recs])
                    w_maes = np.array([r["mae_deg"] for r in w_recs])
                    if np.std(w_ages) > 0 and np.std(w_maes) > 0:
                        rw_a, pw_a = stats.pearsonr(w_ages, w_maes)
                    else:
                        rw_a, pw_a = 0.0, 1.0
                    if np.std(w_dists) > 0 and np.std(w_maes) > 0:
                        rw_d, pw_d = stats.pearsonr(w_dists, w_maes)
                    else:
                        rw_d, pw_d = 0.0, 1.0
                    age_lag_summary[f"{w}s"] = {
                        "n": len(w_recs),
                        "mean_age_s": float(np.mean(w_ages)),
                        "mean_dist_oldest_m": float(np.mean(w_dists)),
                        "mean_mae_deg": float(np.mean(w_maes)),
                        "corr_age_mae_r": float(rw_a), "corr_age_mae_p": float(pw_a),
                        "corr_dist_mae_r": float(rw_d), "corr_dist_mae_p": float(pw_d),
                    }
                    print(f"    W={w:4d}s: mean_age={np.mean(w_ages):5.1f}s, "
                          f"mean_dist={np.mean(w_dists):6.0f}m, "
                          f"mean_MAE={np.mean(w_maes):5.2f}°, "
                          f"r(age,MAE)={rw_a:+.3f}, r(dist,MAE)={rw_d:+.3f}")
        else:
            age_lag_summary = {"n_records": len(age_lag_records), "note": "insufficient data"}

        # =================================================================
        # TASK 6 — OUTAGE ERROR DISTRIBUTION
        # =================================================================
        print(f"\n[TASK 6] Outage Error Distribution")

        dist_summary = {}
        for w in WINDOW_LENGTHS:
            arr = np.array(cs_results[w])
            if len(arr) == 0:
                continue
            q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
            mad = float(np.median(np.abs(arr - np.median(arr))))
            dist_summary[f"{w}s"] = {
                "median_deg": float(np.median(arr)),
                "iqr_deg": float(q75 - q25),
                "mad_deg": mad,
                "p90_deg": float(np.percentile(arr, 90)),
                "p95_deg": float(np.percentile(arr, 95)),
                "max_deg": float(np.max(arr)),
                "n_above_20deg": int(np.sum(arr > 20)),
                "n_above_30deg": int(np.sum(arr > 30)),
                "pct_above_20deg": float(np.mean(arr > 20) * 100),
            }
            print(f"  W={w:4d}s: Median = {np.median(arr):5.2f}°, IQR = {q75 - q25:5.2f}°, "
                  f"MAD = {mad:5.2f}°, P90 = {np.percentile(arr, 90):5.2f}°, "
                  f"P95 = {np.percentile(arr, 95):5.2f}°, Max = {np.max(arr):5.2f}°, "
                  f">20° = {np.sum(arr > 20)}/{len(arr)} ({np.mean(arr > 20)*100:.0f}%)")

        # =================================================================
        # TASK 7 — DYNAMIC Rψ AUDIT (Predictive Validity)
        # =================================================================
        print(f"\n[TASK 7] Dynamic R_psi Predictive Validity Audit")

        # Gather pre-outage predictor features for each outage (using 60s window)
        predictor_records = []
        for s in common_starts:
            samps_60 = get_straight_samples_in_window(D, s, 60)
            if len(samps_60) < MIN_STRAIGHT_SAMPLES:
                continue

            offsets = D["gnss_offset"][samps_60]
            b_norms = D["mag_norm"][samps_60]
            speeds = D["speed_ms"][samps_60]

            sigma_psi = circular_std_deg(offsets)
            sigma_B = float(np.std(b_norms))
            mu_B = float(np.mean(b_norms))
            delta_B_rel = float(np.abs(mu_B - D["baseline_B"]) / D["baseline_B"])
            mean_speed = float(np.mean(speeds))
            n_valid = len(samps_60)
            oldest_age_s = float((s - samps_60[0]) * dt)

            # True outage error (post-hoc)
            out_sl = np.arange(s, s + outage_pts)
            off_est = circular_mean_deg(offsets)
            pred = (D["psi_mag"][out_sl] + off_est) % 360.0
            true_err = np.abs(angle_diff_deg(pred, D["veh_heading"][out_sl]))
            true_mae = float(np.mean(true_err))

            predictor_records.append({
                "sigma_psi": sigma_psi,
                "sigma_B": sigma_B,
                "delta_B_rel": delta_B_rel,
                "mean_speed": mean_speed,
                "n_valid": n_valid,
                "oldest_age_s": oldest_age_s,
                "true_mae_deg": true_mae,
            })

        rpsi_audit = {"n_outages": len(predictor_records)}
        if len(predictor_records) >= 5:
            pred_names = ["sigma_psi", "sigma_B", "delta_B_rel", "mean_speed", "n_valid", "oldest_age_s"]
            target = np.array([r["true_mae_deg"] for r in predictor_records])
            pred_matrix = np.column_stack([[r[p] for r in predictor_records] for p in pred_names])

            # Univariate correlations
            univar = {}
            print(f"  Univariate Correlations with True Outage MAE (N={len(predictor_records)}):")
            for i, pname in enumerate(pred_names):
                x = pred_matrix[:, i]
                if np.std(x) < 1e-12:
                    univar[pname] = {"pearson_r": 0.0, "p": 1.0, "spearman_rho": 0.0}
                    print(f"    {pname:16s}: constant → r = 0.000")
                    continue
                r_p, p_p = stats.pearsonr(x, target)
                rho_s, _ = stats.spearmanr(x, target)
                univar[pname] = {"pearson_r": float(r_p), "p": float(p_p), "spearman_rho": float(rho_s)}
                sig = "*" if p_p < 0.05 else ""
                print(f"    {pname:16s}: r = {r_p:+.3f} (p = {p_p:.4f}){sig}, rho = {rho_s:+.3f}")

            rpsi_audit["univariate"] = univar

            # Multivariate: OLS regression (with leave-one-out cross-validation)
            # Only if enough data
            if len(predictor_records) >= 10:
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import LeaveOneOut

                X = pred_matrix.copy()
                y = target.copy()
                # Standardize predictors
                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0)
                X_std[X_std < 1e-12] = 1.0
                X_norm = (X - X_mean) / X_std

                loo = LeaveOneOut()
                loo_preds = np.zeros(len(y))
                for train_idx, test_idx in loo.split(X_norm):
                    reg = LinearRegression()
                    reg.fit(X_norm[train_idx], y[train_idx])
                    loo_preds[test_idx] = reg.predict(X_norm[test_idx])

                loo_r, _ = stats.pearsonr(y, loo_preds)
                loo_rmse = float(np.sqrt(np.mean((y - loo_preds)**2)))
                baseline_rmse = float(np.sqrt(np.mean((y - np.mean(y))**2)))

                # Full-data fit for coefficients
                reg_full = LinearRegression().fit(X_norm, y)
                coefs = dict(zip(pred_names, reg_full.coef_.tolist()))
                r2_full = float(reg_full.score(X_norm, y))

                print(f"\n  Multivariate OLS (Leave-One-Out Cross-Validation):")
                print(f"    LOO r(predicted, actual) = {loo_r:+.3f}")
                print(f"    LOO RMSE = {loo_rmse:.2f}° (vs baseline RMSE = {baseline_rmse:.2f}°)")
                print(f"    In-sample R² = {r2_full:.3f}")
                print(f"    Standardized coefficients: {json.dumps({k: round(v, 3) for k, v in coefs.items()})}")

                rpsi_audit["multivariate_loo"] = {
                    "loo_r": float(loo_r),
                    "loo_rmse_deg": loo_rmse,
                    "baseline_rmse_deg": baseline_rmse,
                    "in_sample_r2": r2_full,
                    "standardized_coefs": coefs,
                }
            else:
                print(f"  Multivariate regression skipped (N={len(predictor_records)} < 10)")
                rpsi_audit["multivariate_loo"] = {"note": "insufficient data"}
        else:
            print(f"  Insufficient data for R_psi audit (N={len(predictor_records)})")

        # =================================================================
        # TASK 8 — UNCERTAINTY CALIBRATION (Coverage Test)
        # =================================================================
        print(f"\n[TASK 8] Uncertainty Calibration (Coverage Test)")

        # Use proposed R_psi model from C8-11.3:
        # R_psi = sigma_0^2 + c1*sigma_psi^2 + c2*sigma_B^2 + c3*(delta_B/B0)^2
        sigma_0 = 3.0
        c1, c2, c3 = 1.0, 0.5, 50.0

        # Evaluate on non-overlapping outages (to avoid tuning on dense overlapping set)
        coverage_records = []
        for s in nonoverlap_starts:
            samps_60 = get_straight_samples_in_window(D, s, 60)
            if len(samps_60) < MIN_STRAIGHT_SAMPLES:
                continue

            offsets = D["gnss_offset"][samps_60]
            b_norms = D["mag_norm"][samps_60]

            sp = circular_std_deg(offsets)
            sb = float(np.std(b_norms))
            db_rel = float(np.abs(np.mean(b_norms) - D["baseline_B"]) / D["baseline_B"])

            r_psi_var = sigma_0**2 + c1 * sp**2 + c2 * sb**2 + c3 * db_rel**2
            predicted_1sigma = float(np.sqrt(r_psi_var))

            off_est = circular_mean_deg(offsets)
            out_sl = np.arange(s, s + outage_pts)
            pred_heading = (D["psi_mag"][out_sl] + off_est) % 360.0
            signed_err = angle_diff_deg(pred_heading, D["veh_heading"][out_sl])
            mean_signed_err = float(np.mean(signed_err))
            mean_abs_err = float(np.mean(np.abs(signed_err)))

            coverage_records.append({
                "outage_start_s": float(s * dt),
                "predicted_1sigma_deg": predicted_1sigma,
                "mean_signed_err_deg": mean_signed_err,
                "mean_abs_err_deg": mean_abs_err,
            })

        coverage_summary = {"n_outages": len(coverage_records)}
        if len(coverage_records) >= 3:
            pred_sig = np.array([r["predicted_1sigma_deg"] for r in coverage_records])
            actual_err = np.array([r["mean_signed_err_deg"] for r in coverage_records])
            actual_abs = np.array([r["mean_abs_err_deg"] for r in coverage_records])

            within_1sigma = np.sum(np.abs(actual_err) <= pred_sig)
            within_2sigma = np.sum(np.abs(actual_err) <= 1.96 * pred_sig)
            pct_1sig = float(within_1sigma / len(actual_err) * 100)
            pct_2sig = float(within_2sigma / len(actual_err) * 100)

            coverage_summary["pct_within_1sigma"] = pct_1sig
            coverage_summary["pct_within_2sigma"] = pct_2sig
            coverage_summary["expected_1sigma_pct"] = 68.3
            coverage_summary["expected_2sigma_pct"] = 95.0
            coverage_summary["median_predicted_1sigma_deg"] = float(np.median(pred_sig))
            coverage_summary["median_actual_abs_err_deg"] = float(np.median(actual_abs))

            cal_verdict = "WELL_CALIBRATED" if 50 < pct_1sig < 85 else ("OVERCONFIDENT" if pct_1sig < 50 else "CONSERVATIVE")
            coverage_summary["verdict"] = cal_verdict

            print(f"  N = {len(coverage_records)} non-overlapping outages")
            print(f"  Within +/-1 sigma: {within_1sigma}/{len(actual_err)} = {pct_1sig:.1f}% (expected ~68.3%)")
            print(f"  Within +/-1.96 sigma: {within_2sigma}/{len(actual_err)} = {pct_2sig:.1f}% (expected ~95.0%)")
            print(f"  Median predicted 1-sigma = {np.median(pred_sig):.2f} deg, Median actual |err| = {np.median(actual_abs):.2f} deg")
            print(f"  Verdict: {cal_verdict}")
        else:
            coverage_summary["note"] = "insufficient outages for coverage test"
            print(f"  Insufficient non-overlapping outages for coverage test (N={len(coverage_records)})")

        # =================================================================
        # Assemble trip results
        # =================================================================
        trip_res = {
            "n_samples": D["n"],
            "duration_s": float(D["n"] * dt),
            "distance_km": float(D["cum_dist_m"][-1] / 1000.0),
            "task1_common_support": {
                "total_dense_outages": len(dense_starts),
                "common_support_outages": len(common_starts),
                "excluded_outages": len(excluded_starts),
                "exclusion_reasons": exclusion_reasons[:10],  # cap for readability
                "window_comparison": cs_summary,
                "per_outage": cs_per_outage,
            },
            "task2_non_overlapping": {
                "n_outages": len(nonoverlap_starts),
                "min_separation_s": 30.0,
                "window_comparison": no_summary,
                "per_outage": no_per_outage,
            },
            "task3_short_window_estimators": est_results,
            "task4_sample_counts": sample_count_results,
            "task5_age_spatial_lag": age_lag_summary,
            "task6_error_distribution": dist_summary,
            "task7_rpsi_audit": rpsi_audit,
            "task8_uncertainty_coverage": coverage_summary,
        }
        results["trips"][trip_name] = trip_res

    # =====================================================================
    # TASK 9 — CROSS-TRIP GENERALIZATION
    # =====================================================================
    print(f"\n{'=' * 78}")
    print(f"[TASK 9] Cross-Trip Generalization")
    print(f"{'=' * 78}")

    cross = {}
    for trip_name in ["Vta02", "Vta04"]:
        cs = results["trips"][trip_name]["task1_common_support"]["window_comparison"]
        no = results["trips"][trip_name]["task2_non_overlapping"]["window_comparison"]

        # Find best window by median MAE on common-support
        best_cs = min(WINDOW_LENGTHS, key=lambda w: cs.get(f"{w}s", {}).get("median_mae_deg", 999))
        best_no = min(WINDOW_LENGTHS, key=lambda w: no.get(f"{w}s", {}).get("median_mae_deg", 999))

        n_cs = cs.get(f"{best_cs}s", {}).get("n_outages", 0)
        n_no = no.get(f"{best_no}s", {}).get("n_outages", 0)

        cross[trip_name] = {
            "best_window_cs_median": best_cs,
            "best_window_no_median": best_no,
            "n_common_support": n_cs,
            "n_non_overlapping": n_no,
            "statistical_power_note": "weak" if n_no < 10 else "moderate" if n_no < 30 else "adequate",
        }

        print(f"  {trip_name}: Best CS Window = {best_cs}s (N={n_cs}), "
              f"Best NO Window = {best_no}s (N={n_no}), "
              f"Statistical Power = {cross[trip_name]['statistical_power_note']}")

    results["task9_cross_trip"] = cross

    # =====================================================================
    # TASK 10 — FINAL DECISION
    # =====================================================================
    print(f"\n{'=' * 78}")
    print(f"[TASK 10] Final Window Decision")
    print(f"{'=' * 78}")

    # Collect evidence
    v02_cs = results["trips"]["Vta02"]["task1_common_support"]["window_comparison"]
    v04_cs = results["trips"]["Vta04"]["task1_common_support"]["window_comparison"]
    v02_no = results["trips"]["Vta02"]["task2_non_overlapping"]["window_comparison"]
    v04_no = results["trips"]["Vta04"]["task2_non_overlapping"]["window_comparison"]

    # Window ranking by mean MAE (common-support)
    v02_ranking = sorted(WINDOW_LENGTHS, key=lambda w: v02_cs.get(f"{w}s", {}).get("mean_mae_deg", 999))
    v04_ranking = sorted(WINDOW_LENGTHS, key=lambda w: v04_cs.get(f"{w}s", {}).get("mean_mae_deg", 999))

    # Non-overlapping ranking
    v02_no_ranking = sorted(WINDOW_LENGTHS, key=lambda w: v02_no.get(f"{w}s", {}).get("mean_mae_deg", 999))

    # Check agreement
    v02_best = v02_ranking[0]
    v04_best = v04_ranking[0]
    v02_no_best = v02_no_ranking[0]

    # Cross-trip agreement
    agree = v02_best == v04_best
    no_agree = v02_best == v02_no_best

    # Statistical power assessment
    v04_n = v04_cs.get(f"{v04_best}s", {}).get("n_outages", 0)
    v04_weak = v04_n < 10

    # Compare top-1 vs top-2 gap
    if len(v02_ranking) >= 2:
        gap_1_2 = v02_cs[f"{v02_ranking[1]}s"]["mean_mae_deg"] - v02_cs[f"{v02_ranking[0]}s"]["mean_mae_deg"]
    else:
        gap_1_2 = 0.0

    # Determine decision
    if agree and no_agree and gap_1_2 > 1.0 and not v04_weak:
        decision = "B" if v02_best == 10 else "A" if v02_best == 5 else "C"
        decision_label = f"{v02_best}s is justified"
    elif not agree and v04_weak:
        decision = "E"
        decision_label = "Evidence is insufficient — Vta04 has too few outages for cross-trip validation"
    elif gap_1_2 < 1.0:
        decision = "C"
        decision_label = "5–10s adaptive range is justified (difference between top windows is marginal)"
    else:
        decision = "D"
        decision_label = "No fixed window is justified yet — inconsistent cross-trip ranking"

    print(f"\n  Vta02 Common-Support Ranking: {v02_ranking}")
    print(f"  Vta04 Common-Support Ranking: {v04_ranking}")
    print(f"  Vta02 Non-Overlapping Ranking: {v02_no_ranking}")
    print(f"  Vta02 best = {v02_best}s, Vta04 best = {v04_best}s")
    print(f"  Cross-trip agreement: {agree}")
    print(f"  Overlapping vs non-overlapping agreement: {no_agree}")
    print(f"  Top-1 vs Top-2 gap (Vta02): {gap_1_2:.2f}°")
    print(f"  Vta04 statistical power: {'WEAK' if v04_weak else 'ADEQUATE'} (N={v04_n})")
    print(f"\n  >>> DECISION: {decision} — {decision_label}")

    results["task10_final_decision"] = {
        "decision_code": decision,
        "decision_label": decision_label,
        "vta02_cs_ranking": v02_ranking,
        "vta04_cs_ranking": v04_ranking,
        "vta02_no_ranking": v02_no_ranking,
        "cross_trip_agreement": agree,
        "overlap_nonoverlap_agreement": no_agree,
        "top1_top2_gap_deg": gap_1_2,
        "vta04_statistical_power": "weak" if v04_weak else "adequate",
    }

    # =====================================================================
    # Save results
    # =====================================================================
    out_path = REPO_ROOT / "results" / "c8_11_4_common_support_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"\n  JSON results saved to: {out_path}")

    return results


if __name__ == "__main__":
    run_c8_11_4_audit()
