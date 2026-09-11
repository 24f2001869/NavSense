"""
SIH26168 - Stage C6: Controlled Navigation-Error Decomposition Diagnostic
Module: experiments/run_error_decomposition_c6.py

Executes a rigorous counterfactual sensitivity suite on the frozen Stage C5
architecture (Bounded Adaptive C0 / BCAC with NHC) to isolate the physical
mechanisms driving inertial dead-reckoning drift:
  - C6-A: Heading sensitivity (delta_psi in [0, 0.5, 1, 2, 5] deg)
  - C6-B: Accelerometer bias sensitivity (b_ax in [0, 0.02, 0.05, 0.1, 0.2] m/s^2)
  - C6-C: Mounting tilt & gravity leakage sensitivity (pitch & roll in [0, 0.25, 0.5, 1, 2] deg)
  - C6-D: Non-Holonomic Constraint (NHC) strength diagnostic (None, Normal, Tight, Relaxed)
  - C6-E: Combined factorial matrix & nonlinear coupling analysis

Primary diagnostic trip: Vta02
Held-out confirmation trip: Vta04
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip
from src.preprocessing.gravity_alignment import align_phone_to_vehicle
from src.preprocessing.orientation import geodetic_to_enu
from src.navigation.eskf import ESKF3D
from src.navigation.nhc import NonHolonomicConstraint

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def compute_bcac_series(acc_v, dt=0.1, w_b_s=10.0, sigma_c0=0.291):
    """
    Computes frozen Bounded Adaptive C0 (BCAC) process noise series sigma_a(k).
    sigma_a(k) = 0.291 * clip(sqrt(J_norm(k)), 0.75, 1.35)
    where J(k) is trailing jerk RMS and B(k) is 10s causal rolling median.
    """
    n = len(acc_v)
    w_k = 10  # 1.0 s trailing jerk window
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
    sigma_series = sigma_c0 * mult
    return sigma_series, j_norm, b, jerk_rms


def run_single_outage(
    kw, w_dur, dt,
    acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
    delta_psi_deg=0.0,
    delta_pitch_deg=0.0,
    delta_roll_deg=0.0,
    bias_ax=0.0,
    bias_ay=0.0,
    nhc_mode='normal',
    record_trajectory=False
):
    """
    Simulates a single outage window with controlled perturbation injections.
    """
    init_h = float(heading[kw]) + delta_psi_deg
    init_pitch = delta_pitch_deg
    init_roll = delta_roll_deg

    eskf = ESKF3D(
        init_pos_enu=(gt_e[kw], gt_n[kw], gt_u[kw]),
        init_vel_enu=(gt_ve[kw], gt_vn[kw], gt_vu[kw]),
        init_heading_deg=init_h,
        init_pitch_deg=init_pitch,
        init_roll_deg=init_roll,
        init_ba=ba_stat,
        R_vp=np.eye(3),
        sigma_a=0.291,
        gravity=9.80665
    )

    if nhc_mode == 'none':
        nhc = None
    elif nhc_mode == 'tight':
        nhc = NonHolonomicConstraint(sigma_lat=0.1, sigma_vert=0.1)
    elif nhc_mode == 'relaxed':
        nhc = NonHolonomicConstraint(sigma_lat=2.0, sigma_vert=2.0)
    else:  # normal
        nhc = NonHolonomicConstraint(sigma_lat=0.5, sigma_vert=0.5)

    traj_time = []
    traj_pos_err = []
    traj_lat_err = []
    traj_long_err = []
    traj_vel_err = []
    traj_lat_vel = []

    for step_k in range(kw, kw + w_dur):
        eskf.sigma_a = sigma_series[step_k]

        # Apply synthetic acceleration bias in vehicle frame
        ax_inj = acc_v[step_k, 0] + bias_ax
        ay_inj = acc_v[step_k, 1] + bias_ay
        az_inj = acc_v[step_k, 2]

        eskf.predict(ax_inj, ay_inj, az_inj,
                     gyro_v[step_k, 0], gyro_v[step_k, 1], gyro_v[step_k, 2], dt)

        if nhc is not None:
            nhc.update_eskf(eskf)

        if record_trajectory:
            t_curr = (step_k - kw + 1) * dt
            st = eskf.get_state()
            p_curr = st['pos_n'][:2]
            v_curr = st['vel_n'][:2]
            gt_p_curr = np.array([gt_e[step_k + 1], gt_n[step_k + 1]])
            gt_v_curr = np.array([gt_ve[step_k + 1], gt_vn[step_k + 1]])
            curr_h_rad = np.radians(heading[step_k + 1])

            e_vec = p_curr - gt_p_curr
            # Decompose into along-track (longitudinal) and cross-track (lateral)
            u_fwd = np.array([np.sin(curr_h_rad), np.cos(curr_h_rad)])
            u_lat = np.array([-np.cos(curr_h_rad), np.sin(curr_h_rad)])

            e_long = float(np.dot(e_vec, u_fwd))
            e_lat = float(np.dot(e_vec, u_lat))
            e_tot = float(np.linalg.norm(e_vec))
            v_err = float(np.linalg.norm(v_curr - gt_v_curr))

            C_v_n = eskf.attitude.get_dcm()
            vel_v = C_v_n.T @ eskf.vel_n
            lat_v = float(vel_v[1])

            traj_time.append(t_curr)
            traj_pos_err.append(e_tot)
            traj_lat_err.append(e_lat)
            traj_long_err.append(e_long)
            traj_vel_err.append(v_err)
            traj_lat_vel.append(lat_v)

    st_end = eskf.get_state()
    gt_p_end = np.array([gt_e[kw + w_dur], gt_n[kw + w_dur]])
    gt_v_end = np.array([gt_ve[kw + w_dur], gt_vn[kw + w_dur]])
    end_h_rad = np.radians(heading[kw + w_dur])

    e_vec_end = st_end['pos_n'][:2] - gt_p_end
    u_fwd_end = np.array([np.sin(end_h_rad), np.cos(end_h_rad)])
    u_lat_end = np.array([-np.cos(end_h_rad), np.sin(end_h_rad)])

    final_long_err = float(np.dot(e_vec_end, u_fwd_end))
    final_lat_err = float(np.dot(e_vec_end, u_lat_end))
    final_tot_err = float(np.linalg.norm(e_vec_end))
    final_vel_err = float(np.linalg.norm(st_end['vel_n'][:2] - gt_v_end))

    C_v_n_end = eskf.attitude.get_dcm()
    vel_v_end = C_v_n_end.T @ eskf.vel_n
    final_lat_vel = float(vel_v_end[1])

    out = {
        'final_drift_m': final_tot_err,
        'final_lat_err_m': final_lat_err,
        'final_long_err_m': final_long_err,
        'final_vel_err_ms': final_vel_err,
        'final_lat_vel_ms': final_lat_vel
    }
    if record_trajectory:
        out['traj'] = {
            'time': traj_time,
            'pos_err': traj_pos_err,
            'lat_err': traj_lat_err,
            'long_err': traj_long_err,
            'vel_err': traj_vel_err,
            'lat_vel': traj_lat_vel
        }
    return out


def evaluate_consolidated_sweep(
    trip_data,
    horizons=[5.0, 10.0, 20.0, 30.0, 60.0],
    stride_s=40.0,
    dt=0.1,
    experiments_dict=None
):
    """
    Evaluates all unique counterfactual experiment configurations in a single consolidated
    pass per window to eliminate redundant baseline runs and provide live progress logging.
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

    n = len(acc_v)
    stride_k = int(round(stride_s / dt))
    results = {}

    for h in horizons:
        w_dur = int(round(h / dt))
        h_key = f"{int(h)}s"
        window_starts = list(range(50, n - w_dur - 1, stride_k))
        n_win = len(window_starts)
        t_h0 = time.time()

        print(f"  --> Simulating Horizon {int(h)}s ({n_win} windows x {len(experiments_dict)} conditions = {n_win * len(experiments_dict)} runs)...", flush=True)

        results[h_key] = {
            'horizon_s': h,
            'n_windows': n_win,
            'experiments': {exp_name: {
                'drift_m': [], 'lat_err_m': [], 'long_err_m': [], 'vel_err_ms': [], 'lat_vel_ms': []
            } for exp_name in experiments_dict}
        }

        for w_idx, kw in enumerate(window_starts):
            for exp_name, exp_cfg in experiments_dict.items():
                sim_res = run_single_outage(
                    kw, w_dur, dt,
                    acc_v, gyro_v, heading, gt_e, gt_n, gt_u, gt_ve, gt_vn, gt_vu, ba_stat, sigma_series,
                    delta_psi_deg=exp_cfg.get('delta_psi_deg', 0.0),
                    delta_pitch_deg=exp_cfg.get('delta_pitch_deg', 0.0),
                    delta_roll_deg=exp_cfg.get('delta_roll_deg', 0.0),
                    bias_ax=exp_cfg.get('bias_ax', 0.0),
                    bias_ay=exp_cfg.get('bias_ay', 0.0),
                    nhc_mode=exp_cfg.get('nhc_mode', 'normal'),
                    record_trajectory=False
                )
                results[h_key]['experiments'][exp_name]['drift_m'].append(sim_res['final_drift_m'])
                results[h_key]['experiments'][exp_name]['lat_err_m'].append(sim_res['final_lat_err_m'])
                results[h_key]['experiments'][exp_name]['long_err_m'].append(sim_res['final_long_err_m'])
                results[h_key]['experiments'][exp_name]['vel_err_ms'].append(sim_res['final_vel_err_ms'])
                results[h_key]['experiments'][exp_name]['lat_vel_ms'].append(sim_res['final_lat_vel_ms'])

            if (w_idx + 1) % 3 == 0 or (w_idx + 1) == n_win:
                elapsed_h = time.time() - t_h0
                rate = (w_idx + 1) * len(experiments_dict) / max(elapsed_h, 0.1)
                pct = ((w_idx + 1) / n_win) * 100
                print(f"      [Horizon {int(h)}s] Window {w_idx+1}/{n_win} ({pct:.0f}%) complete ({elapsed_h:.1f} s, {rate:.1f} runs/s)", flush=True)

    return results


def prepare_trip_data(trip_name, dt=0.1):
    """
    Loads, aligns, and prepares inertial and reference data for a given trip.
    """
    print(f"\nLoading and preparing trip {trip_name}...", flush=True)
    df_p, df_v = load_trip(trip_name)

    raw_acc = df_p[['accel_x', 'accel_y', 'accel_z']].values
    raw_gyro = df_p[['gyro_x', 'gyro_y', 'gyro_z']].values
    speed = df_v['veh_speed_ms'].values
    can_acc = df_v['veh_accel_long_ms2'].values
    heading = df_v['veh_heading_deg'].values

    acc_v, gyro_v, R_vp, angles_deg = align_phone_to_vehicle(raw_acc, raw_gyro, speed)
    ba_stat = np.array([-0.147147, -0.012351, 0.003124], dtype=np.float64)
    lat0, lon0 = df_v['veh_lat'].iloc[0], df_v['veh_lon'].iloc[0]
    gt_e, gt_n, gt_u = geodetic_to_enu(df_v['veh_lat'].values, df_v['veh_lon'].values, lat0, lon0)

    h_rad = np.radians(heading)
    gt_ve = speed * np.sin(h_rad)
    gt_vn = speed * np.cos(h_rad)
    gt_vu = df_v['veh_vert_vel_kmh'].values / 3.6 if 'veh_vert_vel_kmh' in df_v.columns else np.zeros_like(speed)

    sigma_series, j_norm, b, jerk_rms = compute_bcac_series(acc_v, dt=dt, w_b_s=10.0, sigma_c0=0.291)

    trip_dict = {
        'trip_name': trip_name,
        'acc_v': acc_v,
        'gyro_v': gyro_v,
        'speed': speed,
        'can_acc': can_acc,
        'heading': heading,
        'gt_e': gt_e,
        'gt_n': gt_n,
        'gt_u': gt_u,
        'gt_ve': gt_ve,
        'gt_vn': gt_vn,
        'gt_vu': gt_vu,
        'ba_stat': ba_stat,
        'sigma_series': sigma_series,
        'j_norm': j_norm,
        'b': b,
        'jerk_rms': jerk_rms,
        'mean_speed': float(np.mean(speed)),
        'duration_s': float((len(speed) - 1) * dt)
    }
    print(f"  {trip_name}: {len(speed)} epochs ({trip_dict['duration_s']:.1f} s), Mean speed = {trip_dict['mean_speed']:.2f} m/s", flush=True)
    return trip_dict


def main():
    print("===============================================================================")
    print("STAGE C6: CONTROLLED NAVIGATION-ERROR DECOMPOSITION DIAGNOSTIC")
    print("===============================================================================")
    start_total_time = time.time()
    dt = 0.1

    # Load primary diagnostic dataset Vta02 and confirmation trip Vta04
    trip_v2 = prepare_trip_data("Vta02", dt=dt)
    trip_v4 = prepare_trip_data("Vta04", dt=dt)

    # Consolidated 21 Unique Experiment Configurations
    # -------------------------------------------------------------
    all_experiments = {
        # Baseline Reference (Neutral)
        'Baseline': {'label': 'Baseline (Nominal BCAC+NHC)'},

        # C6-A: Heading Sensitivity
        'H_0p5deg': {'delta_psi_deg': 0.5, 'label': 'Heading +0.5 deg'},
        'H_1p0deg': {'delta_psi_deg': 1.0, 'label': 'Heading +1.0 deg'},
        'H_2p0deg': {'delta_psi_deg': 2.0, 'label': 'Heading +2.0 deg'},
        'H_5p0deg': {'delta_psi_deg': 5.0, 'label': 'Heading +5.0 deg'},

        # C6-B: Accelerometer Bias Sensitivity
        'B_0p02ms2': {'bias_ax': 0.02, 'label': 'Forward Bias +0.02 m/s²'},
        'B_0p05ms2': {'bias_ax': 0.05, 'label': 'Forward Bias +0.05 m/s²'},
        'B_0p10ms2': {'bias_ax': 0.10, 'label': 'Forward Bias +0.10 m/s²'},
        'B_0p20ms2': {'bias_ax': 0.20, 'label': 'Forward Bias +0.20 m/s²'},

        # C6-C: Mounting Tilt Sensitivity (Pitch & Roll)
        'Pitch_0p25deg': {'delta_pitch_deg': 0.25, 'label': 'Pitch +0.25 deg'},
        'Pitch_0p50deg': {'delta_pitch_deg': 0.50, 'label': 'Pitch +0.50 deg'},
        'Pitch_1p00deg': {'delta_pitch_deg': 1.00, 'label': 'Pitch +1.00 deg'},
        'Pitch_2p00deg': {'delta_pitch_deg': 2.00, 'label': 'Pitch +2.00 deg'},
        'Roll_0p50deg': {'delta_roll_deg': 0.50, 'label': 'Roll +0.50 deg'},
        'Roll_1p00deg': {'delta_roll_deg': 1.00, 'label': 'Roll +1.00 deg'},
        'Roll_2p00deg': {'delta_roll_deg': 2.00, 'label': 'Roll +2.00 deg'},

        # C6-D: NHC Constraint Strength
        'NHC_None': {'nhc_mode': 'none', 'label': 'No NHC (Pure Dead-Reckoning)'},
        'NHC_Relaxed': {'nhc_mode': 'relaxed', 'label': 'Relaxed NHC (sigma=2.0)'},
        'NHC_Tight': {'nhc_mode': 'tight', 'label': 'Tightened NHC (sigma=0.1)'},

        # C6-E: Combinations
        'H_plus_B': {'delta_psi_deg': 1.0, 'bias_ax': 0.10, 'label': 'H+B (+1 deg, +0.1 m/s²)'},
        'H_plus_T': {'delta_psi_deg': 1.0, 'delta_pitch_deg': 1.0, 'label': 'H+T (+1 deg, Pitch +1 deg)'},
        'All_Combined': {'delta_psi_deg': 1.0, 'bias_ax': 0.10, 'delta_pitch_deg': 1.0, 'nhc_mode': 'none', 'label': 'All (H+B+T+N)'}
    }

    # Sub-experiment dictionary mappings for analysis and display
    sub_heading = {
        'Baseline': {'label': 'Baseline (0 deg)'},
        'H_0p5deg': {'label': '+0.5 deg'},
        'H_1p0deg': {'label': '+1.0 deg'},
        'H_2p0deg': {'label': '+2.0 deg'},
        'H_5p0deg': {'label': '+5.0 deg'},
    }
    sub_bias = {
        'Baseline': {'label': 'Baseline (0 m/s²)'},
        'B_0p02ms2': {'label': '+0.02 m/s²'},
        'B_0p05ms2': {'label': '+0.05 m/s²'},
        'B_0p10ms2': {'label': '+0.10 m/s²'},
        'B_0p20ms2': {'label': '+0.20 m/s²'},
    }
    sub_tilt = {
        'Baseline': {'label': 'Baseline (0 deg)'},
        'Pitch_0p25deg': {'label': 'Pitch +0.25 deg'},
        'Pitch_0p50deg': {'label': 'Pitch +0.50 deg'},
        'Pitch_1p00deg': {'label': 'Pitch +1.00 deg'},
        'Pitch_2p00deg': {'label': 'Pitch +2.00 deg'},
        'Roll_0p50deg': {'label': 'Roll +0.50 deg'},
        'Roll_1p00deg': {'label': 'Roll +1.00 deg'},
        'Roll_2p00deg': {'label': 'Roll +2.00 deg'},
    }
    sub_nhc = {
        'NHC_None': {'label': 'No NHC (Pure DR)'},
        'NHC_Relaxed': {'label': 'Relaxed NHC (sigma=2.0)'},
        'Baseline': {'label': 'Normal NHC (sigma=0.5)'},
        'NHC_Tight': {'label': 'Tightened NHC (sigma=0.1)'},
    }
    sub_factorial = {
        'Baseline': {'label': 'Baseline (0)'},
        'H_1p0deg': {'label': 'H1 (+1.0 deg)'},
        'B_0p10ms2': {'label': 'B1 (+0.10 m/s²)'},
        'Pitch_1p00deg': {'label': 'T1 (Pitch +1.0 deg)'},
        'NHC_None': {'label': 'N1 (No NHC)'},
        'H_plus_B': {'label': 'H+B (+1 deg, +0.1 m/s²)'},
        'H_plus_T': {'label': 'H+T (+1 deg, +1 deg)'},
        'All_Combined': {'label': 'All (H+B+T+N)'},
    }

    # -------------------------------------------------------------
    # RUN DIAGNOSTICS ON PRIMARY TRIP Vta02
    # -------------------------------------------------------------
    print("\n===============================================================================", flush=True)
    print("[PHASE 1] Running Consolidated Primary Sensitivity Diagnostics on Vta02...", flush=True)
    print("===============================================================================", flush=True)

    t0 = time.time()
    vta02_horizons = [10.0, 20.0, 30.0, 60.0]
    res_v2 = evaluate_consolidated_sweep(
        trip_v2, horizons=vta02_horizons, stride_s=80.0, dt=dt, experiments_dict=all_experiments
    )
    print(f"  Vta02 consolidated diagnostic suite completed in {time.time() - t0:.1f} s", flush=True)

    # -------------------------------------------------------------
    # RUN CONFIRMATION SWEEPS ON HELD-OUT TRIP Vta04
    # -------------------------------------------------------------
    print("\n===============================================================================", flush=True)
    print("[PHASE 2] Running Consolidated Confirmation Diagnostics on Held-Out Vta04...", flush=True)
    print("===============================================================================", flush=True)

    t1 = time.time()
    vta04_horizons = [10.0, 30.0, 60.0]
    res_v4 = evaluate_consolidated_sweep(
        trip_v4, horizons=vta04_horizons, stride_s=20.0, dt=dt, experiments_dict=all_experiments
    )
    print(f"  Vta04 consolidated confirmation suite completed in {time.time() - t1:.1f} s", flush=True)

    # -------------------------------------------------------------
    # TIME-SERIES TRAJECTORY DYNAMICS (Representative Window Audit)
    # -------------------------------------------------------------
    print("\n[PHASE 3] Extracting Time-Series Dynamics on Representative 30-s Window...", flush=True)
    # Pick a dynamic cruising window from Vta02 (e.g. kw = 500, t = 50.0 s, speed ~ 15 m/s)
    kw_rep = 500
    w_dur_rep = 300  # 30s
    rep_trajs = {}
    for exp_name in sub_factorial.keys():
        exp_cfg = all_experiments[exp_name]
        res_rep = run_single_outage(
            kw_rep, w_dur_rep, dt,
            trip_v2['acc_v'], trip_v2['gyro_v'], trip_v2['heading'],
            trip_v2['gt_e'], trip_v2['gt_n'], trip_v2['gt_u'],
            trip_v2['gt_ve'], trip_v2['gt_vn'], trip_v2['gt_vu'],
            trip_v2['ba_stat'], trip_v2['sigma_series'],
            delta_psi_deg=exp_cfg.get('delta_psi_deg', 0.0),
            delta_pitch_deg=exp_cfg.get('delta_pitch_deg', 0.0),
            delta_roll_deg=exp_cfg.get('delta_roll_deg', 0.0),
            bias_ax=exp_cfg.get('bias_ax', 0.0),
            bias_ay=exp_cfg.get('bias_ay', 0.0),
            nhc_mode=exp_cfg.get('nhc_mode', 'normal'),
            record_trajectory=True
        )
        rep_trajs[exp_name] = res_rep['traj']

    # -------------------------------------------------------------
    # STATISTICAL SYNTHESIS & PRINT SUMMARY TABLES
    # -------------------------------------------------------------
    print("\n===============================================================================", flush=True)
    print("EMPIRICAL SENSITIVITY BENCHMARK SUMMARY (Vta02 - 30 s Horizons)", flush=True)
    print("===============================================================================", flush=True)

    print("\n--- Sub-Experiment C6-A: Heading Sensitivity ---")
    print(f"{'Condition':<20} | {'Mean Drift (m)':>14} | {'Cross-Track (m)':>15} | {'Along-Track (m)':>15} | {'Vel Err (m/s)':>13}")
    print("-" * 85)
    for k, cfg in sub_heading.items():
        dr = res_v2['30s']['experiments'][k]['drift_m']
        lat = res_v2['30s']['experiments'][k]['lat_err_m']
        lng = res_v2['30s']['experiments'][k]['long_err_m']
        ve = res_v2['30s']['experiments'][k]['vel_err_ms']
        print(f"{cfg['label']:<20} | {np.mean(dr):14.2f} | {np.mean(np.abs(lat)):15.2f} | {np.mean(np.abs(lng)):15.2f} | {np.mean(ve):13.2f}")

    print("\n--- Sub-Experiment C6-B: Accelerometer Bias Sensitivity ---")
    print(f"{'Condition':<20} | {'Mean Drift (m)':>14} | {'Along-Track (m)':>15} | {'Cross-Track (m)':>15} | {'Vel Err (m/s)':>13}")
    print("-" * 85)
    for k, cfg in sub_bias.items():
        dr = res_v2['30s']['experiments'][k]['drift_m']
        lng = res_v2['30s']['experiments'][k]['long_err_m']
        lat = res_v2['30s']['experiments'][k]['lat_err_m']
        ve = res_v2['30s']['experiments'][k]['vel_err_ms']
        print(f"{cfg['label']:<20} | {np.mean(dr):14.2f} | {np.mean(np.abs(lng)):15.2f} | {np.mean(np.abs(lat)):15.2f} | {np.mean(ve):13.2f}")

    print("\n--- Sub-Experiment C6-C: Mounting Tilt & Gravity Leakage ---")
    print(f"{'Condition':<20} | {'Mean Drift (m)':>14} | {'Along-Track (m)':>15} | {'Cross-Track (m)':>15} | {'Lat Vel (m/s)':>13}")
    print("-" * 85)
    for k, cfg in sub_tilt.items():
        dr = res_v2['30s']['experiments'][k]['drift_m']
        lng = res_v2['30s']['experiments'][k]['long_err_m']
        lat = res_v2['30s']['experiments'][k]['lat_err_m']
        lv = res_v2['30s']['experiments'][k]['lat_vel_ms']
        print(f"{cfg['label']:<20} | {np.mean(dr):14.2f} | {np.mean(np.abs(lng)):15.2f} | {np.mean(np.abs(lat)):15.2f} | {np.mean(np.abs(lv)):13.2f}")

    print("\n--- Sub-Experiment C6-D: NHC Constraint Strength ---")
    print(f"{'Condition':<25} | {'Mean Drift (m)':>14} | {'Cross-Track (m)':>15} | {'Lat Vel (m/s)':>13}")
    print("-" * 75)
    for k, cfg in sub_nhc.items():
        dr = res_v2['30s']['experiments'][k]['drift_m']
        lat = res_v2['30s']['experiments'][k]['lat_err_m']
        lv = res_v2['30s']['experiments'][k]['lat_vel_ms']
        print(f"{cfg['label']:<25} | {np.mean(dr):14.2f} | {np.mean(np.abs(lat)):15.2f} | {np.mean(np.abs(lv)):13.2f}")

    print("\n--- Sub-Experiment C6-E: Combined Factorial Matrix & Coupling Analysis ---")
    print(f"{'Condition':<25} | {'Mean Drift (m)':>14} | {'Excess Drift (m)':>16} | {'Coupling Ratio Chi':>20}")
    print("-" * 80)
    base_drift_v2 = np.mean(res_v2['30s']['experiments']['Baseline']['drift_m'])
    d_h1 = np.mean(res_v2['30s']['experiments']['H_1p0deg']['drift_m']) - base_drift_v2
    d_b1 = np.mean(res_v2['30s']['experiments']['B_0p10ms2']['drift_m']) - base_drift_v2
    d_t1 = np.mean(res_v2['30s']['experiments']['Pitch_1p00deg']['drift_m']) - base_drift_v2
    d_n1 = np.mean(res_v2['30s']['experiments']['NHC_None']['drift_m']) - base_drift_v2

    coupling_metrics = {}
    for k, cfg in sub_factorial.items():
        dr = np.mean(res_v2['30s']['experiments'][k]['drift_m'])
        excess = dr - base_drift_v2
        chi_str = "N/A"
        chi_val = None
        if k == 'H_plus_B':
            linear_sum = d_h1 + d_b1
            chi_val = excess / linear_sum if linear_sum > 0 else 1.0
            chi_str = f"{chi_val:.3f} (H+B)"
        elif k == 'H_plus_T':
            linear_sum = d_h1 + d_t1
            chi_val = excess / linear_sum if linear_sum > 0 else 1.0
            chi_str = f"{chi_val:.3f} (H+T)"
        elif k == 'All_Combined':
            linear_sum = d_h1 + d_b1 + d_t1 + d_n1
            chi_val = excess / linear_sum if linear_sum > 0 else 1.0
            chi_str = f"{chi_val:.3f} (All)"
        coupling_metrics[k] = {'mean_drift': float(dr), 'excess_drift': float(excess), 'coupling_ratio': chi_val}
        print(f"{cfg['label']:<25} | {dr:14.2f} | {excess:16.2f} | {chi_str:>20}")

    # -------------------------------------------------------------
    # RENDER 5 PUBLICATION-GRADE DIAGNOSTIC FIGURES
    # -------------------------------------------------------------
    print("\n[PHASE 4] Generating Publication Diagnostic Figures...", flush=True)

    # -------------------------------------------------------------
    # FIGURE 1: C6-A Heading Sensitivity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    psi_angles = [0.0, 0.5, 1.0, 2.0, 5.0]
    psi_keys = ['Baseline', 'H_0p5deg', 'H_1p0deg', 'H_2p0deg', 'H_5p0deg']

    # Panel A: Drift vs Heading Error across Horizons
    for h in [10.0, 20.0, 30.0, 60.0]:
        h_key = f"{int(h)}s"
        dr_means = [np.mean(res_v2[h_key]['experiments'][k]['drift_m']) for k in psi_keys]
        axes[0, 0].plot(psi_angles, dr_means, marker='o', lw=2, label=f'{int(h)}s Horizon')
    axes[0, 0].set_title('(A) Vta02 Multi-Horizon Drift vs. Heading Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Heading Perturbation $\\delta\\psi$ (deg)', fontsize=10)
    axes[0, 0].set_ylabel('Mean Total Position Error (m)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=9.5)

    # Panel B: Along-Track vs. Cross-Track Error (30 s)
    lat_errs = [np.mean(np.abs(res_v2['30s']['experiments'][k]['lat_err_m'])) for k in psi_keys]
    long_errs = [np.mean(np.abs(res_v2['30s']['experiments'][k]['long_err_m'])) for k in psi_keys]
    axes[0, 1].plot(psi_angles, lat_errs, marker='s', lw=2.2, color='crimson', label='Cross-Track (Lateral) Error')
    axes[0, 1].plot(psi_angles, long_errs, marker='^', lw=2.2, color='navy', label='Along-Track (Longitudinal) Error')
    # Theoretical cross-track linear model: e_lat ~ v * psi_rad * t
    v_mean = trip_v2['mean_speed']
    t_30 = 30.0
    theo_lat = [v_mean * np.radians(p) * t_30 for p in psi_angles]
    axes[0, 1].plot(psi_angles, theo_lat, ls='--', color='gray', lw=1.5, label=f'Theoretical $v\\psi t$ ($v={v_mean:.1f}$ m/s)')
    axes[0, 1].set_title('(B) 30-s Directional Error Decomposition vs. Theory', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Heading Perturbation $\\delta\\psi$ (deg)', fontsize=10)
    axes[0, 1].set_ylabel('Mean Error Magnitude (m)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(fontsize=9.5)

    # Panel C: Time-Resolved Error Growth (Representative Window)
    t_axis = rep_trajs['Baseline']['time']
    axes[1, 0].plot(t_axis, rep_trajs['Baseline']['pos_err'], lw=2, color='black', label='Baseline (0 deg)')
    axes[1, 0].plot(t_axis, rep_trajs['H_1p0deg']['pos_err'], lw=2, color='crimson', label='H1 (+1.0 deg)')
    axes[1, 0].plot(t_axis, rep_trajs['H_1p0deg']['lat_err'], lw=2, color='orange', ls='--', label='H1 Cross-Track')
    axes[1, 0].set_title('(C) Time-Resolved Drift Growth over 30 s Outage', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Time into Blackout (s)', fontsize=10)
    axes[1, 0].set_ylabel('Error (m)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(fontsize=9.5)

    # Panel D: Vta02 vs Vta04 Cross-Trip Heading Sensitivity (30 s)
    dr_v2 = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in psi_keys]
    dr_v4 = [np.mean(res_v4['30s']['experiments'][k]['drift_m']) for k in psi_keys]
    axes[1, 1].plot(psi_angles, dr_v2, marker='o', lw=2.2, color='blue', label='Vta02 Primary')
    axes[1, 1].plot(psi_angles, dr_v4, marker='s', lw=2.2, color='green', label='Vta04 Confirmation')
    axes[1, 1].set_title('(D) Cross-Trip Confirmation: Vta02 vs. Vta04', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Heading Perturbation $\\delta\\psi$ (deg)', fontsize=10)
    axes[1, 1].set_ylabel('Mean Total Drift (m)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9.5)

    plt.tight_layout()
    fig1_path = FIG_DIR / "c6_heading_sensitivity.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 1: {fig1_path}", flush=True)

    # -------------------------------------------------------------
    # FIGURE 2: C6-B Accelerometer Bias Sensitivity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    bias_vals = [0.0, 0.02, 0.05, 0.10, 0.20]
    bias_keys = ['Baseline', 'B_0p02ms2', 'B_0p05ms2', 'B_0p10ms2', 'B_0p20ms2']

    # Panel A: Multi-Horizon Drift vs Accel Bias
    for h in [10.0, 20.0, 30.0, 60.0]:
        h_key = f"{int(h)}s"
        dr_means = [np.mean(res_v2[h_key]['experiments'][k]['drift_m']) for k in bias_keys]
        axes[0, 0].plot(bias_vals, dr_means, marker='o', lw=2, label=f'{int(h)}s Horizon')
    axes[0, 0].set_title('(A) Vta02 Multi-Horizon Drift vs. Forward Accel Bias', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Forward Bias $b_{a,x}$ (m/s²)', fontsize=10)
    axes[0, 0].set_ylabel('Mean Total Position Error (m)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=9.5)

    # Panel B: Empirical vs. Quadratic Law (0.5 * b * t^2)
    dr_30_bias = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in bias_keys]
    base_30_b0 = dr_30_bias[0]
    empirical_excess = [d - base_30_b0 for d in dr_30_bias]
    theo_quadratic = [0.5 * b * (30.0**2) for b in bias_vals]

    axes[0, 1].plot(bias_vals, empirical_excess, marker='s', lw=2.2, color='crimson', label='Empirical Excess Drift $\\Delta e_p$')
    axes[0, 1].plot(bias_vals, theo_quadratic, ls='--', color='navy', lw=2, label='Ideal Quadratic Law $\\frac{1}{2} b_a T^2$ ($T=30$s)')
    axes[0, 1].set_title('(B) Empirical Excess vs. Ideal Quadratic Law', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Forward Bias $b_{a,x}$ (m/s²)', fontsize=10)
    axes[0, 1].set_ylabel('Excess Position Drift (m)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(fontsize=9.5)

    # Panel C: Along-Track vs. Cross-Track Response
    long_bias = [np.mean(np.abs(res_v2['30s']['experiments'][k]['long_err_m'])) for k in bias_keys]
    lat_bias = [np.mean(np.abs(res_v2['30s']['experiments'][k]['lat_err_m'])) for k in bias_keys]
    axes[1, 0].plot(bias_vals, long_bias, marker='^', lw=2.2, color='navy', label='Along-Track (Longitudinal)')
    axes[1, 0].plot(bias_vals, lat_bias, marker='s', lw=2.2, color='crimson', label='Cross-Track (Lateral)')
    axes[1, 0].set_title('(C) 30-s Directional Error under Forward Bias', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Forward Bias $b_{a,x}$ (m/s²)', fontsize=10)
    axes[1, 0].set_ylabel('Mean Error (m)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(fontsize=9.5)

    # Panel D: Cross-Trip Confirmation (30 s)
    dr_v2_bias = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in bias_keys]
    dr_v4_bias = [np.mean(res_v4['30s']['experiments'][k]['drift_m']) for k in bias_keys]
    axes[1, 1].plot(bias_vals, dr_v2_bias, marker='o', lw=2.2, color='blue', label='Vta02 Primary')
    axes[1, 1].plot(bias_vals, dr_v4_bias, marker='s', lw=2.2, color='green', label='Vta04 Confirmation')
    axes[1, 1].set_title('(D) Cross-Trip Confirmation: Vta02 vs. Vta04', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Forward Bias $b_{a,x}$ (m/s²)', fontsize=10)
    axes[1, 1].set_ylabel('Mean Total Drift (m)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9.5)

    plt.tight_layout()
    fig2_path = FIG_DIR / "c6_accel_bias_sensitivity.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 2: {fig2_path}", flush=True)

    # -------------------------------------------------------------
    # FIGURE 3: C6-C Mounting Tilt & Gravity Leakage
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    pitch_angles = [0.0, 0.25, 0.50, 1.00, 2.00]
    pitch_keys = ['Baseline', 'Pitch_0p25deg', 'Pitch_0p50deg', 'Pitch_1p00deg', 'Pitch_2p00deg']
    roll_angles = [0.0, 0.50, 1.00, 2.00]
    roll_keys = ['Baseline', 'Roll_0p50deg', 'Roll_1p00deg', 'Roll_2p00deg']

    # Panel A: Pitch (Longitudinal Gravity Leakage)
    for h in [10.0, 20.0, 30.0, 60.0]:
        h_key = f"{int(h)}s"
        dr_means = [np.mean(res_v2[h_key]['experiments'][k]['drift_m']) for k in pitch_keys]
        axes[0, 0].plot(pitch_angles, dr_means, marker='o', lw=2, label=f'{int(h)}s Horizon')
    axes[0, 0].set_title('(A) Pitch Tilt Multi-Horizon Drift (Forward Leakage)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Pitch Perturbation $\\delta\\theta$ (deg)', fontsize=10)
    axes[0, 0].set_ylabel('Mean Total Drift (m)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=9.5)

    # Panel B: Roll vs. Pitch Asymmetry (30 s)
    dr_30_pitch = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in pitch_keys]
    dr_30_roll = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in roll_keys]
    axes[0, 1].plot(pitch_angles, dr_30_pitch, marker='o', lw=2.2, color='crimson', label='Pitch (Forward Leakage - Unconstrained)')
    axes[0, 1].plot(roll_angles, dr_30_roll, marker='s', lw=2.2, color='navy', label='Roll (Lateral Leakage - NHC Constrained)')
    axes[0, 1].set_title('(B) Pitch vs. Roll Asymmetric Response (30 s)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Angular Perturbation (deg)', fontsize=10)
    axes[0, 1].set_ylabel('Mean Total Drift (m)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(fontsize=9.5)

    # Panel C: Lateral Velocity under Roll vs. Pitch
    lv_roll = [np.mean(np.abs(res_v2['30s']['experiments'][k]['lat_vel_ms'])) for k in roll_keys]
    lv_pitch = [np.mean(np.abs(res_v2['30s']['experiments'][k]['lat_vel_ms'])) for k in roll_keys]
    axes[1, 0].plot(roll_angles, lv_roll, marker='s', lw=2.2, color='navy', label='Roll Error')
    axes[1, 0].plot(roll_angles, lv_pitch, marker='o', lw=2.2, color='crimson', label='Pitch Error')
    axes[1, 0].set_title('(C) Lateral Velocity Response ($v_y^v$) under Tilt', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Tilt Perturbation (deg)', fontsize=10)
    axes[1, 0].set_ylabel('Mean Lateral Velocity (m/s)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(fontsize=9.5)

    # Panel D: Gravity Leakage Acceleration vs. Theory
    leakage_acc = [9.80665 * np.sin(np.radians(a)) for a in pitch_angles]
    axes[1, 1].plot(pitch_angles, leakage_acc, marker='^', lw=2.2, color='purple', label='$a_{leak} = g \\sin\\delta\\theta$')
    axes[1, 1].set_title('(D) Theoretical Horizontal Gravity Leakage', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Tilt Error $\\delta\\theta$ (deg)', fontsize=10)
    axes[1, 1].set_ylabel('Leakage Acceleration (m/s²)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9.5)

    plt.tight_layout()
    fig3_path = FIG_DIR / "c6_tilt_gravity_leakage.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 3: {fig3_path}", flush=True)

    # -------------------------------------------------------------
    # FIGURE 4: C6-D NHC Constraint Strength
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    nhc_keys = ['NHC_None', 'NHC_Relaxed', 'Baseline', 'NHC_Tight']
    nhc_labels = ['No NHC', 'Relaxed (2.0)', 'Normal (0.5)', 'Tight (0.1)']

    # Panel A: Horizon Drift across NHC Regimes
    h_labels_v2 = [f"{int(h)}s" for h in vta02_horizons]
    x = np.arange(len(h_labels_v2))
    width = 0.2
    for idx, (nk, nl) in enumerate(zip(nhc_keys, nhc_labels)):
        means = [np.mean(res_v2[hl]['experiments'][nk]['drift_m']) for hl in h_labels_v2]
        axes[0, 0].bar(x + idx * width - 0.3, means, width, label=nl, alpha=0.85)
    axes[0, 0].set_title('(A) Vta02 Drift across NHC Constraint Strengths', fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(h_labels_v2)
    axes[0, 0].set_ylabel('Mean Drift (m)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend(fontsize=9.5)

    # Panel B: Lateral Velocity Error Distribution (30 s)
    lat_vel_distributions = [res_v2['30s']['experiments'][nk]['lat_vel_ms'] for nk in nhc_keys]
    axes[0, 1].boxplot(lat_vel_distributions, tick_labels=nhc_labels)
    axes[0, 1].set_title('(B) Lateral Velocity Distribution ($v_y^v$) at 30 s', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Lateral Velocity (m/s)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)

    # Panel C: Cross-Track Error Reduction vs. No NHC (30 s)
    lat_err_vals = [np.mean(np.abs(res_v2['30s']['experiments'][nk]['lat_err_m'])) for nk in nhc_keys]
    axes[1, 0].bar(nhc_labels, lat_err_vals, color=['crimson', 'orange', 'teal', 'darkgreen'], alpha=0.85)
    axes[1, 0].set_title('(C) Cross-Track Error Suppression across Regimes', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Mean Cross-Track Error (m)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)

    # Panel D: Vta04 Held-Out NHC Drift Comparison (30 s & 60 s)
    v4_30s = [np.mean(res_v4['30s']['experiments'][nk]['drift_m']) for nk in nhc_keys]
    v4_60s = [np.mean(res_v4['60s']['experiments'][nk]['drift_m']) for nk in nhc_keys]
    x_v4 = np.arange(len(nhc_labels))
    axes[1, 1].bar(x_v4 - 0.15, v4_30s, 0.3, label='30s Horizon', color='steelblue', alpha=0.85)
    axes[1, 1].bar(x_v4 + 0.15, v4_60s, 0.3, label='60s Horizon', color='coral', alpha=0.85)
    axes[1, 1].set_title('(D) Vta04 Held-Out NHC Constraint Response', fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(x_v4)
    axes[1, 1].set_xticklabels(nhc_labels)
    axes[1, 1].set_ylabel('Mean Drift (m)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9.5)

    plt.tight_layout()
    fig4_path = FIG_DIR / "c6_nhc_regime_comparison.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 4: {fig4_path}", flush=True)

    # -------------------------------------------------------------
    # FIGURE 5: C6-E Combined Factorial Matrix & Coupling
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fact_keys = ['Baseline', 'H_1p0deg', 'B_0p10ms2', 'Pitch_1p00deg', 'NHC_None', 'H_plus_B', 'H_plus_T', 'All_Combined']
    fact_labels = ['Base', 'H1 (+1°)', 'B1 (+0.1)', 'T1 (P+1°)', 'N1 (NoNHC)', 'H+B', 'H+T', 'All (H+B+T+N)']

    # Panel A: 30-s Mean Drift Comparison (Vta02)
    means_v2 = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) for k in fact_keys]
    colors = ['gray', 'crimson', 'royalblue', 'purple', 'darkorange', 'brown', 'magenta', 'black']
    axes[0, 0].bar(fact_labels, means_v2, color=colors, alpha=0.85)
    axes[0, 0].set_title('(A) Vta02 30-s Factorial Drift Comparison', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Position Error (m)', fontsize=10)
    axes[0, 0].set_xticklabels(fact_labels, rotation=25, ha='right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.4)

    # Panel B: Superposition Test (Linear Sum vs. Actual Combo)
    combos = ['H+B', 'H+T', 'All']
    actual_excess = [
        np.mean(res_v2['30s']['experiments']['H_plus_B']['drift_m']) - base_drift_v2,
        np.mean(res_v2['30s']['experiments']['H_plus_T']['drift_m']) - base_drift_v2,
        np.mean(res_v2['30s']['experiments']['All_Combined']['drift_m']) - base_drift_v2,
    ]
    linear_excess = [
        d_h1 + d_b1,
        d_h1 + d_t1,
        d_h1 + d_b1 + d_t1 + d_n1
    ]
    x_c = np.arange(len(combos))
    axes[0, 1].bar(x_c - 0.15, actual_excess, 0.3, label='Actual Combined $\\Delta e$', color='crimson', alpha=0.85)
    axes[0, 1].bar(x_c + 0.15, linear_excess, 0.3, label='Linear Sum $\\sum \\Delta e_i$', color='navy', alpha=0.85)
    axes[0, 1].set_title('(B) Superposition Test: Actual vs. Linear Sum', fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x_c)
    axes[0, 1].set_xticklabels(combos)
    axes[0, 1].set_ylabel('Excess Drift $\\Delta e$ (m)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend(fontsize=9.5)

    # Panel C: Coupling Ratio Chi across Horizons (H+B and H+T)
    chi_hb_list = []
    chi_ht_list = []
    hor_list = ['10s', '20s', '30s', '60s']
    for hl in hor_list:
        b_dr = np.mean(res_v2[hl]['experiments']['Baseline']['drift_m'])
        dh = np.mean(res_v2[hl]['experiments']['H_1p0deg']['drift_m']) - b_dr
        db = np.mean(res_v2[hl]['experiments']['B_0p10ms2']['drift_m']) - b_dr
        dt_p = np.mean(res_v2[hl]['experiments']['Pitch_1p00deg']['drift_m']) - b_dr

        ex_hb = np.mean(res_v2[hl]['experiments']['H_plus_B']['drift_m']) - b_dr
        ex_ht = np.mean(res_v2[hl]['experiments']['H_plus_T']['drift_m']) - b_dr

        chi_hb_list.append(ex_hb / (dh + db) if (dh + db) > 0 else 1.0)
        chi_ht_list.append(ex_ht / (dh + dt_p) if (dh + dt_p) > 0 else 1.0)

    axes[1, 0].plot(hor_list, chi_hb_list, marker='o', lw=2.2, color='brown', label='Coupling Ratio $\\chi(H+B)$')
    axes[1, 0].plot(hor_list, chi_ht_list, marker='s', lw=2.2, color='magenta', label='Coupling Ratio $\\chi(H+T)$')
    axes[1, 0].axhline(1.0, ls='--', color='black', lw=1.2, label='Linear Independence ($\\chi=1.0$)')
    axes[1, 0].set_title('(C) Coupling Ratio Evolution across Horizons', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Outage Horizon', fontsize=10)
    axes[1, 0].set_ylabel('Coupling Ratio $\\chi = \\Delta e_{combo} / \\sum \\Delta e_i$', fontsize=10)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend(fontsize=9.5)

    # Panel D: Vta02 vs. Vta04 Cross-Trip Sensitivity Ranking
    ranking_keys = ['H_1p0deg', 'B_0p10ms2', 'Pitch_1p00deg', 'NHC_None']
    ranking_labels = ['Heading (+1°)', 'Bias (+0.1 m/s²)', 'Tilt (P+1°)', 'No NHC']
    v2_dr_rank = [np.mean(res_v2['30s']['experiments'][k]['drift_m']) - base_drift_v2 for k in ranking_keys]

    base_v4 = np.mean(res_v4['30s']['experiments']['Baseline']['drift_m'])
    v4_dr_rank = [np.mean(res_v4['30s']['experiments'][k]['drift_m']) - base_v4 for k in ranking_keys]

    x_r = np.arange(len(ranking_labels))
    axes[1, 1].bar(x_r - 0.15, v2_dr_rank, 0.3, label='Vta02 Excess Drift', color='blue', alpha=0.85)
    axes[1, 1].bar(x_r + 0.15, v4_dr_rank, 0.3, label='Vta04 Excess Drift', color='green', alpha=0.85)
    axes[1, 1].set_title('(D) Mechanism Sensitivity Ranking: Vta02 vs. Vta04', fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(x_r)
    axes[1, 1].set_xticklabels(ranking_labels, rotation=15, ha='right', fontsize=9.5)
    axes[1, 1].set_ylabel('Excess Drift Above Baseline (m)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend(fontsize=9.5)

    plt.tight_layout()
    fig5_path = FIG_DIR / "c6_combined_coupling_matrix.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 5: {fig5_path}", flush=True)

    # -------------------------------------------------------------
    # EXPORT STRUCTURED JSON DELIVERABLE
    # -------------------------------------------------------------
    def serialize_res(res_dict):
        clean = {}
        for hk, hdata in res_dict.items():
            clean[hk] = {
                'horizon_s': hdata['horizon_s'],
                'n_windows': hdata['n_windows'],
                'experiments': {}
            }
            for exp_k, exp_vals in hdata['experiments'].items():
                clean[hk]['experiments'][exp_k] = {
                    'mean_drift_m': float(np.mean(exp_vals['drift_m'])),
                    'median_drift_m': float(np.median(exp_vals['drift_m'])),
                    'p90_drift_m': float(np.percentile(exp_vals['drift_m'], 90)),
                    'mean_lat_err_m': float(np.mean(np.abs(exp_vals['lat_err_m']))),
                    'mean_long_err_m': float(np.mean(np.abs(exp_vals['long_err_m']))),
                    'mean_vel_err_ms': float(np.mean(exp_vals['vel_err_ms'])),
                    'mean_lat_vel_ms': float(np.mean(np.abs(exp_vals['lat_vel_ms'])))
                }
        return clean

    c6_output = {
        'status': 'Complete',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'frozen_architecture': 'Bounded Adaptive C0 (BCAC)',
        'vta02_primary': serialize_res(res_v2),
        'vta04_confirmation': serialize_res(res_v4),
        'coupling_analysis_vta02_30s': coupling_metrics,
        'figures': [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
            str(fig3_path.relative_to(REPO_ROOT)),
            str(fig4_path.relative_to(REPO_ROOT)),
            str(fig5_path.relative_to(REPO_ROOT))
        ]
    }

    json_path = RES_DIR / "c6_error_decomposition.json"
    with open(json_path, 'w') as f:
        json.dump(c6_output, f, indent=2)
    print(f"\nSaved structured JSON deliverable: {json_path}", flush=True)

    print(f"\n>>> Stage C6 Diagnostic Pipeline Successfully Completed in {time.time() - start_total_time:.1f} s <<<", flush=True)


if __name__ == "__main__":
    main()
