"""
SIH26168 - Stage C7-C: Offline Road-Grade & Longitudinal Error Audit
Script: experiments/audit_road_grade_c7_c.py

Comprehensive offline physical audit:
  Phase 1: Establish Actual Road-Grade Signal (VBOX quantities: theta_g = atan(v_z / v_x), a_g,x = g * sin(theta_g))
           Full distributional metrics, fraction > 0.5°, 1°, 2°, speed/accel correlations.
  Phase 2: Separate Road Grade from Vehicle Dynamic Pitch (Steady |a_x|<0.3, Accel a_x>1.0, Brake a_x<-1.5)
  Phase 3: Smartphone Observability Feasibility (Candidate A: Phone gravity tilt; Candidate B: Vertical GPS velocity; Candidate C: Barometer audit)
  Phase 4: Explanatory Power (Linear regression: e_a = beta * g * sin(theta_g) + r; R^2, residual std, cross-trip stability)
  Phase 5: Lag Audit (Cross-correlation of e_a vs g * sin(theta_g) over tau in [-5s, +5s])
  Phase 6: Navigation Relevance & Counterfactual Upper Bound (Drift recovery under perfect grade knowledge across 10s, 20s, 30s, 60s)

STRICT RULE: Pure offline characterization. Zero filter changes, zero ML, zero parameter tuning.
"""

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
from src.preprocessing.gravity_alignment import align_phone_to_vehicle

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def compute_distribution_metrics(values: np.ndarray) -> dict:
    """Computes comprehensive distributional metrics for an array."""
    if len(values) == 0:
        return {
            'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'p5': 0.0, 'p50': 0.0, 'p95': 0.0, 'min': 0.0, 'max': 0.0
        }
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'p5': float(np.percentile(values, 5)),
        'p50': float(np.percentile(values, 50)),
        'p95': float(np.percentile(values, 95)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


# ==============================================================================
# PHASE 1: ESTABLISH ACTUAL ROAD-GRADE SIGNAL
# ==============================================================================

def run_phase_1_road_grade_profile(trips=['Vta02', 'Vta04']) -> dict:
    """Computes VBOX-derived road grade theta_g = atan(v_z / v_x) and a_g,x = g * sin(theta_g)."""
    print("\n--- PHASE 1: VBOX Actual Road-Grade Signal ---")
    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx = df_v['veh_speed_ms'].values
        vz = df_v['veh_vert_vel_kmh'].values / 3.6
        ax_vbox = df_v['veh_accel_long_ms2'].values
        dv_dt = np.gradient(vx, 0.1)

        # Phone aligned forward acceleration
        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx)
        ax_phone = acc_v[:, 0]

        # Valid motion mask (speed > 2.0 m/s for stable grade angle)
        mask_motion = vx > 2.0
        theta_g_rad = np.zeros(n)
        theta_g_rad[mask_motion] = np.arctan2(vz[mask_motion], vx[mask_motion])
        theta_g_deg = np.degrees(theta_g_rad)
        g_sin_theta = 9.80665 * np.sin(theta_g_rad)

        # Acceleration discrepancies
        e_a_kin = ax_phone - dv_dt
        e_a_can = ax_phone - ax_vbox

        # Threshold exceedance fractions
        frac_gt_05 = float(np.mean(np.abs(theta_g_deg[mask_motion]) > 0.5))
        frac_gt_10 = float(np.mean(np.abs(theta_g_deg[mask_motion]) > 1.0))
        frac_gt_20 = float(np.mean(np.abs(theta_g_deg[mask_motion]) > 2.0))

        # Correlations
        r_grade_speed = float(np.corrcoef(theta_g_deg[mask_motion], vx[mask_motion])[0, 1])
        r_grade_ax = float(np.corrcoef(theta_g_deg[mask_motion], dv_dt[mask_motion])[0, 1])
        r_ea_kin_grade = float(np.corrcoef(e_a_kin[mask_motion], g_sin_theta[mask_motion])[0, 1])
        r_ea_can_grade = float(np.corrcoef(e_a_can[mask_motion], g_sin_theta[mask_motion])[0, 1])

        grade_stats = compute_distribution_metrics(theta_g_deg[mask_motion])
        g_leak_stats = compute_distribution_metrics(g_sin_theta[mask_motion])
        ea_kin_stats = compute_distribution_metrics(e_a_kin[mask_motion])
        ea_can_stats = compute_distribution_metrics(e_a_can[mask_motion])

        results[trip] = {
            'trip': trip,
            'motion_epochs': int(np.sum(mask_motion)),
            'motion_fraction': float(np.mean(mask_motion)),
            'grade_deg_stats': grade_stats,
            'g_sin_theta_ms2_stats': g_leak_stats,
            'ea_kin_stats': ea_kin_stats,
            'ea_can_stats': ea_can_stats,
            'fraction_abs_grade_gt_0_5_deg': frac_gt_05,
            'fraction_abs_grade_gt_1_0_deg': frac_gt_10,
            'fraction_abs_grade_gt_2_0_deg': frac_gt_20,
            'corr_grade_vs_speed': r_grade_speed,
            'corr_grade_vs_accel': r_grade_ax,
            'corr_ea_kin_vs_g_sin_theta': r_ea_kin_grade,
            'corr_ea_can_vs_g_sin_theta': r_ea_can_grade
        }

        print(f"[{trip}] Grade: Mean={grade_stats['mean']:+.3f}°, Std={grade_stats['std']:.3f}°, P5={grade_stats['p5']:+.3f}°, P95={grade_stats['p95']:+.3f}°, Range=[{grade_stats['min']:+.2f}°, {grade_stats['max']:+.2f}°]")
        print(f"       Gravity Leakage g*sin(theta): Mean={g_leak_stats['mean']:+.4f}, Std={g_leak_stats['std']:.4f}, Max={g_leak_stats['max']:.4f} m/s^2")
        print(f"       Exceedance: >0.5°: {frac_gt_05*100:.1f}% | >1.0°: {frac_gt_10*100:.1f}% | >2.0°: {frac_gt_20*100:.1f}%")
        print(f"       Corr(ea_kin, g*sin(theta)): r={r_ea_kin_grade:+.4f} | Corr(ea_can, g*sin(theta)): r={r_ea_can_grade:+.4f}")

    return results


# ==============================================================================
# PHASE 2: SEPARATE ROAD GRADE FROM DYNAMIC SUSPENSION PITCH
# ==============================================================================

def run_phase_2_dynamic_pitch_separation(trips=['Vta02', 'Vta04']) -> dict:
    """
    Separates topographic road grade from dynamic suspension pitch compliance
    by stratifying into: Steady (|a_x| < 0.3), Accel (a_x > 1.0), Brake (a_x < -1.5).
    """
    print("\n--- PHASE 2: Road Grade vs. Dynamic Suspension Pitch Separation ---")
    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx = df_v['veh_speed_ms'].values
        vz = df_v['veh_vert_vel_kmh'].values / 3.6
        ax_vbox = df_v['veh_accel_long_ms2'].values
        dv_dt = np.gradient(vx, 0.1)

        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx)
        ax_phone = acc_v[:, 0]

        mask_motion = vx > 2.0
        theta_g = np.zeros(n)
        theta_g[mask_motion] = np.arctan2(vz[mask_motion], vx[mask_motion])
        g_sin_theta = 9.80665 * np.sin(theta_g)
        e_a_kin = ax_phone - dv_dt

        # Dynamic pitch proxy: difference between chassis accelerometer (which feels suspension pitch tilt)
        # and kinematic acceleration dv/dt: Delta_pitch_acc = a_x_vbox - dv/dt ≈ g * sin(theta_suspension)
        delta_pitch_acc = ax_vbox - dv_dt

        regimes = {
            'steady (|ax|<0.3)': mask_motion & (np.abs(dv_dt) < 0.3),
            'accel (ax>1.0)': mask_motion & (dv_dt > 1.0),
            'brake (ax<-1.5)': mask_motion & (dv_dt < -1.5)
        }

        regime_data = {}
        for r_name, mask in regimes.items():
            if np.sum(mask) < 5:
                continue
            r_grade = float(np.corrcoef(e_a_kin[mask], g_sin_theta[mask])[0, 1]) if np.std(g_sin_theta[mask]) > 1e-6 else 0.0
            r_susp = float(np.corrcoef(e_a_kin[mask], delta_pitch_acc[mask])[0, 1]) if np.std(delta_pitch_acc[mask]) > 1e-6 else 0.0

            regime_data[r_name] = {
                'count': int(np.sum(mask)),
                'std_ea_kin': float(np.std(e_a_kin[mask])),
                'std_grade_leak': float(np.std(g_sin_theta[mask])),
                'std_susp_pitch': float(np.std(delta_pitch_acc[mask])),
                'corr_ea_vs_road_grade': r_grade,
                'corr_ea_vs_susp_pitch': r_susp,
                'mean_ea_kin': float(np.mean(e_a_kin[mask])),
                'mean_grade_leak': float(np.mean(g_sin_theta[mask])),
                'mean_susp_pitch': float(np.mean(delta_pitch_acc[mask]))
            }
            print(f"[{trip} - {r_name}] N={np.sum(mask):4d} | std(ea)={np.std(e_a_kin[mask]):.3f} | std(grade)={np.std(g_sin_theta[mask]):.3f} | std(susp)={np.std(delta_pitch_acc[mask]):.3f} | r(ea,grade)={r_grade:+.3f} | r(ea,susp)={r_susp:+.3f}")

        results[trip] = regime_data

    return results


# ==============================================================================
# PHASE 3: SMARTPHONE OBSERVABILITY FEASIBILITY
# ==============================================================================

def run_phase_3_smartphone_observability(trips=['Vta02', 'Vta04']) -> dict:
    """
    Evaluates whether smartphone sensor channels can observe road grade:
      Candidate A: Phone gravity vector / attitude tilt
      Candidate B: Smartphone vertical GPS velocity (via numerical altitude diff)
      Candidate C: Barometer audit (checks availability)
    """
    print("\n--- PHASE 3: Smartphone Observability Feasibility Audit ---")
    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx_vbox = df_v['veh_speed_ms'].values
        vz_vbox = df_v['veh_vert_vel_kmh'].values / 3.6
        mask_motion = vx_vbox > 2.0

        # True VBOX grade
        theta_vbox_deg = np.zeros(n)
        theta_vbox_deg[mask_motion] = np.degrees(np.arctan2(vz_vbox[mask_motion], vx_vbox[mask_motion]))

        # Candidate A: Phone gravity vector pitch
        # Gravity channels in df_p
        if 'grav_x' in df_p.columns and 'grav_y' in df_p.columns and 'grav_z' in df_p.columns:
            gx = df_p['grav_x'].values
            gy = df_p['grav_y'].values
            gz = df_p['grav_z'].values
            phone_pitch_grav_deg = np.degrees(np.arctan2(gx, np.sqrt(gy**2 + gz**2)))
            r_grav_vbox = float(np.corrcoef(phone_pitch_grav_deg[mask_motion], theta_vbox_deg[mask_motion])[0, 1])
        else:
            phone_pitch_grav_deg = np.zeros(n)
            r_grav_vbox = 0.0

        # Candidate B: Smartphone vertical GPS velocity
        phone_alt = df_p['phone_alt'].values
        phone_vz = np.gradient(phone_alt, 0.1)  # 10 Hz gradient of 1 Hz quantized alt
        vx_phone = df_p['phone_speed_ms'].values if 'phone_speed_ms' in df_p.columns else vx_vbox
        theta_phone_gps_deg = np.zeros(n)
        mask_phone_motion = mask_motion & (vx_phone > 2.0)
        theta_phone_gps_deg[mask_phone_motion] = np.degrees(np.arctan2(phone_vz[mask_phone_motion], vx_phone[mask_phone_motion]))

        r_gps_vbox = float(np.corrcoef(theta_phone_gps_deg[mask_phone_motion], theta_vbox_deg[mask_phone_motion])[0, 1])

        # Candidate C: Barometer audit
        baro_cols = [c for c in df_p.columns if any(k in c.lower() for k in ['baro', 'press', 'air', 'pressure'])]
        has_baro = len(baro_cols) > 0

        results[trip] = {
            'trip': trip,
            'candidate_a_gravity_pitch_corr': r_grav_vbox,
            'candidate_a_gravity_pitch_std_deg': float(np.std(phone_pitch_grav_deg)),
            'candidate_b_gps_alt_grade_corr': r_gps_vbox,
            'candidate_b_gps_alt_grade_std_deg': float(np.std(theta_phone_gps_deg[mask_phone_motion])),
            'candidate_c_has_barometer': bool(has_baro),
            'candidate_c_baro_channels': baro_cols,
            'vbox_grade_std_deg': float(np.std(theta_vbox_deg[mask_motion]))
        }

        print(f"[{trip}] Cand A (Phone Grav Tilt): r={r_grav_vbox:+.3f} | Cand B (Phone GPS Alt Grade): r={r_gps_vbox:+.3f} (Std={np.std(theta_phone_gps_deg[mask_phone_motion]):.2f}° vs True={np.std(theta_vbox_deg[mask_motion]):.2f}°) | Cand C (Baro Present): {has_baro}")

    return results


# ==============================================================================
# PHASE 4: THE EXPLANATORY POWER TEST (R^2 AND REGRESSION)
# ==============================================================================

def run_phase_4_explanatory_power(trips=['Vta02', 'Vta04']) -> dict:
    """
    Evaluates: e_a(t) = beta * g * sin(theta_g(t)) + r(t)
    Computes beta, R^2, residual standard deviation, and regime-specific R^2 across trips.
    """
    print("\n--- PHASE 4: Explanatory Power Test (Linear Regression & R^2) ---")
    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx = df_v['veh_speed_ms'].values
        vz = df_v['veh_vert_vel_kmh'].values / 3.6
        ax_vbox = df_v['veh_accel_long_ms2'].values
        dv_dt = np.gradient(vx, 0.1)

        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx)
        ax_phone = acc_v[:, 0]

        mask_motion = vx > 2.0
        theta_g = np.zeros(n)
        theta_g[mask_motion] = np.arctan2(vz[mask_motion], vx[mask_motion])
        g_sin_theta = 9.80665 * np.sin(theta_g)

        # Acceleration errors
        e_a_kin = ax_phone - dv_dt
        e_a_can = ax_phone - ax_vbox

        # Overall regression: e_a = beta * g_sin_theta + alpha
        X = np.vstack([g_sin_theta[mask_motion], np.ones(np.sum(mask_motion))]).T
        beta_kin, alpha_kin = np.linalg.lstsq(X, e_a_kin[mask_motion], rcond=None)[0]
        beta_can, alpha_can = np.linalg.lstsq(X, e_a_can[mask_motion], rcond=None)[0]

        pred_kin = beta_kin * g_sin_theta[mask_motion] + alpha_kin
        pred_can = beta_can * g_sin_theta[mask_motion] + alpha_can

        var_total_kin = float(np.var(e_a_kin[mask_motion]))
        var_res_kin = float(np.var(e_a_kin[mask_motion] - pred_kin))
        r2_kin = float(1.0 - var_res_kin / (var_total_kin + 1e-12))

        var_total_can = float(np.var(e_a_can[mask_motion]))
        var_res_can = float(np.var(e_a_can[mask_motion] - pred_can))
        r2_can = float(1.0 - var_res_can / (var_total_can + 1e-12))

        # Regime-specific R^2
        regimes = {
            'steady (|ax|<0.3)': mask_motion & (np.abs(dv_dt) < 0.3),
            'accel (ax>1.0)': mask_motion & (dv_dt > 1.0),
            'brake (ax<-1.5)': mask_motion & (dv_dt < -1.5)
        }
        regime_r2 = {}
        for r_name, m in regimes.items():
            if np.sum(m) > 10:
                r_val = float(np.corrcoef(e_a_kin[m], g_sin_theta[m])[0, 1]) if np.std(g_sin_theta[m]) > 1e-6 else 0.0
                regime_r2[r_name] = {
                    'count': int(np.sum(m)),
                    'r': r_val,
                    'r2': float(r_val**2),
                    'std_ea': float(np.std(e_a_kin[m])),
                    'std_grade': float(np.std(g_sin_theta[m]))
                }

        results[trip] = {
            'trip': trip,
            'kinematic_ref': {
                'beta': float(beta_kin),
                'alpha': float(alpha_kin),
                'r2': r2_kin,
                'std_total_ea': float(np.sqrt(var_total_kin)),
                'std_residual_ea': float(np.sqrt(var_res_kin)),
                'variance_explained_pct': float(r2_kin * 100.0)
            },
            'can_ref': {
                'beta': float(beta_can),
                'alpha': float(alpha_can),
                'r2': r2_can,
                'std_total_ea': float(np.sqrt(var_total_can)),
                'std_residual_ea': float(np.sqrt(var_res_can)),
                'variance_explained_pct': float(r2_can * 100.0)
            },
            'regimes': regime_r2
        }

        print(f"[{trip}] Kinematic Ref: beta={beta_kin:+.3f} | R^2={r2_kin:.6f} ({r2_kin*100:.3f}%) | Std: Total={np.sqrt(var_total_kin):.3f} -> Residual={np.sqrt(var_res_kin):.3f} m/s^2")
        print(f"       CAN Chassis Ref: beta={beta_can:+.3f} | R^2={r2_can:.6f} ({r2_can*100:.3f}%) | Std: Total={np.sqrt(var_total_can):.3f} -> Residual={np.sqrt(var_res_can):.3f} m/s^2")

    return results


# ==============================================================================
# PHASE 5: LAG AUDIT
# ==============================================================================

def run_phase_5_lag_audit(trips=['Vta02', 'Vta04']) -> dict:
    """
    Audits cross-correlation of e_a(t) vs g * sin(theta_g(t + tau)) over tau in [-5s, +5s].
    Reports best lag and peak correlation.
    """
    print("\n--- PHASE 5: Lag Audit of e_a vs Road Grade ---")
    results = {}

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx = df_v['veh_speed_ms'].values
        vz = df_v['veh_vert_vel_kmh'].values / 3.6
        dv_dt = np.gradient(vx, 0.1)

        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx)
        ax_phone = acc_v[:, 0]

        mask_motion = vx > 2.0
        theta_g = np.zeros(n)
        theta_g[mask_motion] = np.arctan2(vz[mask_motion], vx[mask_motion])
        g_sin_theta = 9.80665 * np.sin(theta_g)
        e_a_kin = ax_phone - dv_dt

        # Lag sweep: -5s to +5s (-50 to +50 samples)
        max_lag = 50
        lags_s = np.arange(-max_lag, max_lag + 1) * 0.1
        corrs = []

        x = e_a_kin - np.mean(e_a_kin)
        y = g_sin_theta - np.mean(g_sin_theta)

        for lag_idx in range(-max_lag, max_lag + 1):
            if lag_idx < 0:
                c = np.corrcoef(x[-lag_idx:], y[:lag_idx])[0, 1]
            elif lag_idx > 0:
                c = np.corrcoef(x[:-lag_idx], y[lag_idx:])[0, 1]
            else:
                c = np.corrcoef(x, y)[0, 1]
            corrs.append(float(c))

        corrs = np.array(corrs)
        opt_idx = int(np.argmax(np.abs(corrs)))
        best_lag_s = float(lags_s[opt_idx])
        peak_corr = float(corrs[opt_idx])
        zero_lag_corr = float(corrs[max_lag])

        results[trip] = {
            'trip': trip,
            'best_lag_s': best_lag_s,
            'peak_corr': peak_corr,
            'zero_lag_corr': zero_lag_corr,
            'lags_s': lags_s.tolist(),
            'correlations': corrs.tolist()
        }

        print(f"[{trip}] Peak Corr={peak_corr:+.4f} at Lag={best_lag_s:+.1f}s | Zero-Lag Corr={zero_lag_corr:+.4f}")

    return results


# ==============================================================================
# PHASE 6: NAVIGATION RELEVANCE & COUNTERFACTUAL UPPER BOUND
# ==============================================================================

def run_phase_6_counterfactual_navigation(trips=['Vta02', 'Vta04']) -> dict:
    """
    Evaluates: "Even with perfect VBOX grade information, how much drift can theoretically be recovered?"
    Simulates offline open-loop dead reckoning under:
      1. Raw Measured Accel: a_x_raw = a_x_phone
      2. Counterfactual Grade-Corrected: a_x_corr = a_x_phone - g * sin(theta_g)
    Evaluates across horizons 10s, 20s, 30s, 60s.
    """
    print("\n--- PHASE 6: Counterfactual Navigation Upper Bound ---")
    results = {}
    horizons_s = [10, 20, 30, 60]
    dt = 0.1

    for trip in trips:
        df_p, df_v = load_trip(trip)
        n = min(len(df_p), len(df_v))
        df_p = df_p.iloc[:n].copy()
        df_v = df_v.iloc[:n].copy()

        vx = df_v['veh_speed_ms'].values
        vz = df_v['veh_vert_vel_kmh'].values / 3.6

        raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
        raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
        acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx)
        ax_phone = acc_v[:, 0]

        mask_motion = vx > 2.0
        theta_g = np.zeros(n)
        theta_g[mask_motion] = np.arctan2(vz[mask_motion], vx[mask_motion])
        g_sin_theta = 9.80665 * np.sin(theta_g)
        ax_corr = ax_phone - g_sin_theta

        horizon_results = {}
        for h in horizons_s:
            w_len = int(round(h / dt))
            if w_len > n:
                continue
            step = max(1, w_len // 2)
            windows = [(i, i + w_len) for i in range(0, n - w_len + 1, step)]

            drifts_raw = []
            drifts_corr = []

            for i0, i1 in windows:
                v_true = vx[i0:i1]
                p_true = np.cumsum(v_true) * dt

                # Raw integration
                v_raw = v_true[0] + np.cumsum(ax_phone[i0:i1]) * dt
                p_raw = np.cumsum(v_raw) * dt
                drifts_raw.append(abs(p_raw[-1] - p_true[-1]))

                # Grade-corrected integration
                v_c = v_true[0] + np.cumsum(ax_corr[i0:i1]) * dt
                p_c = np.cumsum(v_c) * dt
                drifts_corr.append(abs(p_c[-1] - p_true[-1]))

            mean_raw = float(np.mean(drifts_raw))
            mean_corr = float(np.mean(drifts_corr))
            p90_raw = float(np.percentile(drifts_raw, 90))
            p90_corr = float(np.percentile(drifts_corr, 90))
            delta_mean = mean_corr - mean_raw
            delta_pct = float((delta_mean / (mean_raw + 1e-12)) * 100.0)

            horizon_results[f"{h}s"] = {
                'horizon_s': h,
                'window_count': len(windows),
                'mean_drift_raw_m': mean_raw,
                'mean_drift_corrected_m': mean_corr,
                'p90_drift_raw_m': p90_raw,
                'p90_drift_corrected_m': p90_corr,
                'delta_mean_m': delta_mean,
                'delta_percent': delta_pct
            }
            print(f"[{trip} - {h:2d}s] N={len(windows):3d} | Raw Mean={mean_raw:6.2f}m | Grade-Corr Mean={mean_corr:6.2f}m | Delta={delta_mean:+5.2f}m ({delta_pct:+5.2f}%)")

        results[trip] = horizon_results

    return results


# ==============================================================================
# DIAGNOSTIC DASHBOARD VISUALIZATION
# ==============================================================================

def generate_diagnostic_figure(phase1_data: dict, phase2_data: dict, phase3_data: dict,
                               phase4_data: dict, phase5_data: dict, phase6_data: dict):
    """Generates a publication-grade 6-panel diagnostic dashboard for Stage C7-C."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stage C7-C: Comprehensive Offline Road-Grade & Longitudinal Error Audit", fontsize=14, fontweight='bold', y=0.98)

    # --------------------------------------------------------------------------
    # Panel 1: VBOX Actual Road-Grade Distribution (Phase 1)
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    df_p, df_v = load_trip('Vta02')
    vx2 = df_v['veh_speed_ms'].values
    vz2 = df_v['veh_vert_vel_kmh'].values / 3.6
    m2 = vx2 > 2.0
    grade2_deg = np.degrees(np.arctan2(vz2[m2], vx2[m2]))

    df_p4, df_v4 = load_trip('Vta04')
    vx4 = df_v4['veh_speed_ms'].values
    vz4 = df_v4['veh_vert_vel_kmh'].values / 3.6
    m4 = vx4 > 2.0
    grade4_deg = np.degrees(np.arctan2(vz4[m4], vx4[m4]))

    ax1.hist(grade2_deg, bins=50, density=True, alpha=0.6, color='#2980b9', label=f"Vta02 (std={np.std(grade2_deg):.2f}°)")
    ax1.hist(grade4_deg, bins=50, density=True, alpha=0.6, color='#e67e22', label=f"Vta04 (std={np.std(grade4_deg):.2f}°)")
    ax1.axvline(0, color='black', linestyle=':', alpha=0.7)
    ax1.set_xlabel("VBOX Road Grade $\\theta_g$ [degrees]")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Phase 1: Ground-Truth Road Grade Distribution", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 2: Acceleration Error vs. Road-Grade Gravity Leakage (Phase 4 Scatter)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values[:2000]
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values[:2000]
    acc_v, _, _, _ = align_phone_to_vehicle(raw_acc, raw_gyro, vx2[:2000])
    ax_phone2 = acc_v[:, 0]
    dv_dt2 = np.gradient(vx2[:2000], 0.1)
    ea2 = ax_phone2 - dv_dt2
    g_leak2 = 9.80665 * np.sin(np.radians(grade2_deg[:2000]))

    ax2.scatter(g_leak2, ea2, alpha=0.25, s=12, color='#8e44ad', label=f"Vta02 Points (r={phase1_data['Vta02']['corr_ea_kin_vs_g_sin_theta']:.3f})")
    # Linear fit line
    b = phase4_data['Vta02']['kinematic_ref']['beta']
    a = phase4_data['Vta02']['kinematic_ref']['alpha']
    g_range = np.linspace(g_leak2.min(), g_leak2.max(), 100)
    ax2.plot(g_range, b * g_range + a, 'r-', lw=2, label=f"Fit Line (R² = {phase4_data['Vta02']['kinematic_ref']['r2']:.4f})")
    ax2.set_xlabel("Gravity Leakage $g \\sin\\theta_g$ [m/s$^2$]")
    ax2.set_ylabel("Longitudinal Error $e_a$ [m/s$^2$]")
    ax2.set_title("Phase 4: Scatter & Explanatory Power ($R^2 \\approx 0$)", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 3: Dynamic Pitch Separation Across Regimes (Phase 2)
    # --------------------------------------------------------------------------
    ax3 = axes[0, 2]
    reg_names = ['Steady\n(|ax|<0.3)', 'Accel\n(ax>1.0)', 'Brake\n(ax<-1.5)']
    r_grade = [
        phase2_data['Vta02']['steady (|ax|<0.3)']['corr_ea_vs_road_grade'],
        phase2_data['Vta02']['accel (ax>1.0)']['corr_ea_vs_road_grade'],
        phase2_data['Vta02']['brake (ax<-1.5)']['corr_ea_vs_road_grade']
    ]
    r_susp = [
        phase2_data['Vta02']['steady (|ax|<0.3)']['corr_ea_vs_susp_pitch'],
        phase2_data['Vta02']['accel (ax>1.0)']['corr_ea_vs_susp_pitch'],
        phase2_data['Vta02']['brake (ax<-1.5)']['corr_ea_vs_susp_pitch']
    ]
    x_idx = np.arange(len(reg_names))
    w = 0.35
    ax3.bar(x_idx - w/2, r_grade, width=w, label='Corr(ea, Road Grade)', color='#3498db')
    ax3.bar(x_idx + w/2, r_susp, width=w, label='Corr(ea, Susp Pitch)', color='#e74c3c')
    ax3.axhline(0, color='black', linestyle='-', lw=0.8)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(reg_names, fontsize=8.5)
    ax3.set_ylabel("Correlation Coefficient $r$")
    ax3.set_title("Phase 2: Road Grade vs. Dynamic Suspension Pitch", fontweight='bold')
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 4: Smartphone Observability Audit (Phase 3)
    # --------------------------------------------------------------------------
    ax4 = axes[1, 0]
    trips_list = ['Vta02', 'Vta04']
    r_cand_a = [phase3_data[t]['candidate_a_gravity_pitch_corr'] for t in trips_list]
    r_cand_b = [phase3_data[t]['candidate_b_gps_alt_grade_corr'] for t in trips_list]
    x_obs = np.arange(len(trips_list))
    w_obs = 0.3
    ax4.bar(x_obs - w_obs/2, r_cand_a, width=w_obs, label='Cand A (Phone Grav Tilt)', color='#1abc9c')
    ax4.bar(x_obs + w_obs/2, r_cand_b, width=w_obs, label='Cand B (Phone GPS Alt Grade)', color='#f39c12')
    ax4.axhline(0, color='black', linestyle='-', lw=0.8)
    ax4.set_xticks(x_obs)
    ax4.set_xticklabels(['Vta02 (Suburban)', 'Vta04 (Highway)'], fontsize=9)
    ax4.set_ylabel("Correlation with VBOX Grade")
    ax4.set_title("Phase 3: Smartphone Grade Observability", fontweight='bold')
    ax4.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left', fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 5: Lag Audit of Acceleration Error vs. Grade (Phase 5)
    # --------------------------------------------------------------------------
    ax5 = axes[1, 1]
    for trip, col in [('Vta02', '#2ecc71'), ('Vta04', '#e67e22')]:
        lags = np.array(phase5_data[trip]['lags_s'])
        c_lags = np.array(phase5_data[trip]['correlations'])
        ax5.plot(lags, c_lags, label=f"{trip} (peak r={phase5_data[trip]['peak_corr']:.3f} at {phase5_data[trip]['best_lag_s']:+.1f}s)", color=col, lw=2)
    ax5.axvline(0, color='gray', linestyle=':')
    ax5.axhline(0, color='black', linestyle='-', lw=0.5)
    ax5.set_xlabel("Time Lag $\\tau$ [seconds]")
    ax5.set_ylabel("Correlation $r(e_a(t), g\\sin\\theta_g(t+\\tau))$")
    ax5.set_title("Phase 5: Cross-Correlation Lag Audit", fontweight='bold')
    ax5.set_ylim([-0.25, 0.25])
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='lower left', fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 6: Counterfactual Navigation Upper Bound (Phase 6)
    # --------------------------------------------------------------------------
    ax6 = axes[1, 2]
    horizons = [10, 20, 30, 60]
    v2_raw = [phase6_data['Vta02'][f'{h}s']['mean_drift_raw_m'] for h in horizons]
    v2_corr = [phase6_data['Vta02'][f'{h}s']['mean_drift_corrected_m'] for h in horizons]
    v4_raw = [phase6_data['Vta04'][f'{h}s']['mean_drift_raw_m'] for h in horizons]
    v4_corr = [phase6_data['Vta04'][f'{h}s']['mean_drift_corrected_m'] for h in horizons]

    ax6.plot(horizons, v2_raw, 'bo-', label='Vta02 Raw Baseline', lw=1.8)
    ax6.plot(horizons, v2_corr, 'b*--', label='Vta02 Grade-Corrected', lw=1.8)
    ax6.plot(horizons, v4_raw, 'ms-', label='Vta04 Raw Baseline', lw=1.8)
    ax6.plot(horizons, v4_corr, 'm^--', label='Vta04 Grade-Corrected', lw=1.8)

    ax6.set_xlabel("Outage Duration [seconds]")
    ax6.set_ylabel("Mean Along-Track Drift [m]")
    ax6.set_title("Phase 6: Counterfactual Drift Recovery Upper Bound", fontweight='bold')
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    plot_path = FIG_DIR / "c7_c_road_grade_audit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved master diagnostic plot to: {plot_path}")


# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("STAGE C7-C: OFFLINE ROAD-GRADE & LONGITUDINAL ERROR AUDIT")
    print("=" * 80)

    # Phase 1: Establish Road Grade Signal
    p1_data = run_phase_1_road_grade_profile(trips=['Vta02', 'Vta04'])

    # Phase 2: Separate Grade from Dynamic Pitch
    p2_data = run_phase_2_dynamic_pitch_separation(trips=['Vta02', 'Vta04'])

    # Phase 3: Smartphone Observability Feasibility
    p3_data = run_phase_3_smartphone_observability(trips=['Vta02', 'Vta04'])

    # Phase 4: Explanatory Power (R^2 & Regression)
    p4_data = run_phase_4_explanatory_power(trips=['Vta02', 'Vta04'])

    # Phase 5: Lag Audit
    p5_data = run_phase_5_lag_audit(trips=['Vta02', 'Vta04'])

    # Phase 6: Navigation Relevance & Counterfactual Upper Bound
    p6_data = run_phase_6_counterfactual_navigation(trips=['Vta02', 'Vta04'])

    # Generate Visualization
    generate_diagnostic_figure(p1_data, p2_data, p3_data, p4_data, p5_data, p6_data)

    # Master JSON Export
    master_results = {
        'metadata': {
            'stage': 'C7-C',
            'description': 'Offline Road-Grade & Longitudinal Error Audit',
            'script': 'experiments/audit_road_grade_c7_c.py'
        },
        'phase_1_road_grade_profile': p1_data,
        'phase_2_dynamic_pitch_separation': p2_data,
        'phase_3_smartphone_observability': p3_data,
        'phase_4_explanatory_power': p4_data,
        'phase_5_lag_audit': p5_data,
        'phase_6_counterfactual_navigation': p6_data
    }

    json_path = RES_DIR / "c7_c_road_grade_audit.json"
    with open(json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"Exported master JSON to: {json_path}")
    print("\nSTAGE C7-C AUDIT COMPLETE.")


if __name__ == "__main__":
    main()
