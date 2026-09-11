"""
SIH26168 - Stage C10: Golden Reference Test Session Generator
Script: experiments/export_golden_reference_session.py

Exports a complete, multi-regime time-series session (1789 epochs at 10 Hz)
spanning GNSS Lock, Causal Bias Tracking, 60s GNSS Blackout, Dead-Reckoning,
and GNSS Reacquisition on trip Vta04.

Saves:
- data/golden_reference_session.json: Structured session for Kotlin unit testing
- data/golden_reference_session.csv: Tabular time-series
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.navigation.dead_reckoning_engine import DeadReckoningEngine, EngineConfig, GNSSMeasurement
from experiments.evaluate_phone_speed_c8_7 import prepare_trip_phone_data
from experiments.validate_end_to_end_c8_11_8 import prepare_trip_for_validation

DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_golden_reference():
    print("=" * 78)
    print("STAGE C10: GENERATING GOLDEN REFERENCE SESSION DATASET")
    print("=" * 78)

    # 1. Train Causal RF Model
    print("1. Training causal RF speed model...")
    d_vta02_raw = prepare_trip_phone_data('Vta02')
    rf_model = RandomForestRegressor(n_estimators=35, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_model.fit(d_vta02_raw['features'], d_vta02_raw['gt_speed'])

    # 2. Ingest Vta04
    print("2. Ingesting Vta04 trip telemetry...")
    D = prepare_trip_for_validation('Vta04', rf_model)
    n = D['n']
    dt = D['dt']

    # Define Outage Interval: 60s blackout from epoch 600 to 1200 (60.0s to 120.0s)
    BLACKOUT_START = 600
    BLACKOUT_END = 1200

    print(f"3. Configuring production C9.1 DeadReckoningEngine on Vta04 (N={n} epochs)...")
    print(f"   Blackout window: epoch {BLACKOUT_START} -> {BLACKOUT_END} (t = {BLACKOUT_START*dt:.1f}s -> {BLACKOUT_END*dt:.1f}s)")

    cfg = EngineConfig(
        sigma_speed=0.60,
        sigma_lat_0=0.50,
        mag_norm_tol=0.08,
        mag_db_dt_tol=5.0,
        enable_vert_nhc=True,
        turn_rate_vnhc_gate_deg_s=3.0,
        turn_rate_compass_gate_deg_s=3.0
    )

    engine = DeadReckoningEngine(
        config=cfg,
        road_index=D['road_index'],
        init_pos_enu=(float(D['gt_e'][0]), float(D['gt_n'][0]), float(D['gt_u'][0])),
        init_vel_enu=(float(D['gt_ve'][0]), float(D['gt_vn'][0]), float(D['gt_vu'][0])),
        init_heading_deg=float(D['heading'][0]),
        R_vp=np.eye(3),
        ba_stat=D['ba_stat'].copy(),
        baseline_mag_uT=float(D['baseline_B'])
    )

    epochs_data = []
    csv_rows = []

    print("4. Executing engine simulation and recording ground-truth telemetry...")
    for step in range(n):
        t_now = step * dt
        is_outage = (BLACKOUT_START <= step < BLACKOUT_END)

        # Raw Phone IMU and Magnetometer
        acc_step = D['acc_v'][step]   # In leveled vehicle frame
        gyro_step = D['gyro_v'][step]
        mag_step = D['mag_raw'][step]
        ml_spd = float(D['ml_speed'][step])
        cal_mag_h = float((D['psi_mag'][step] + D['init_offset']) % 360.0)
        db_dt = float(D['db_dt'][step])

        # GNSS packet
        if not is_outage:
            gnss = GNSSMeasurement(
                valid=True,
                pos_enu=np.array([D['gt_e'][step], D['gt_n'][step], D['gt_u'][step]]),
                vel_enu=np.array([D['gt_ve'][step], D['gt_vn'][step], D['gt_vu'][step]]),
                heading_deg=float(D['heading'][step]),
                accuracy_m=3.0
            )
        else:
            gnss = GNSSMeasurement(valid=False)

        # Execute Engine Step
        res = engine.step(
            accel_raw=acc_step,
            gyro_raw=gyro_step,
            speed_est=ml_spd,
            dt=dt,
            mag_raw=mag_step,
            gnss=gnss,
            psi_mag_cal_deg=cal_mag_h,
            db_dt=db_dt
        )

        diag = res['diagnostics']
        pos_out = res['pos_enu']
        vel_out = res['vel_enu']

        epoch_record = {
            'step': step,
            'time_s': round(t_now, 3),
            'accel': [round(float(x), 6) for x in acc_step],
            'gyro': [round(float(x), 6) for x in gyro_step],
            'mag': [round(float(x), 4) for x in mag_step],
            'ml_speed': round(ml_spd, 4),
            'db_dt': round(db_dt, 4),
            'psi_mag_cal_deg': round(cal_mag_h, 3),
            'gnss_valid': bool(gnss.valid),
            'gnss_pos_enu': [round(float(x), 4) for x in gnss.pos_enu] if gnss.valid else [0.0, 0.0, 0.0],
            'gnss_vel_enu': [round(float(x), 4) for x in gnss.vel_enu] if gnss.valid else [0.0, 0.0, 0.0],
            'gt_pos_enu': [round(float(D['gt_e'][step]), 4), round(float(D['gt_n'][step]), 4), round(float(D['gt_u'][step]), 4)],
            'gt_vel_enu': [round(float(D['gt_ve'][step]), 4), round(float(D['gt_vn'][step]), 4), round(float(D['gt_vu'][step]), 4)],
            'gt_heading_deg': round(float(D['heading'][step]), 3),
            # Python Reference Outputs
            'ref_pos_enu': [round(float(x), 5) for x in pos_out],
            'ref_vel_enu': [round(float(x), 5) for x in vel_out],
            'ref_heading_deg': round(float(res['heading_deg']), 4),
            'ref_pitch_deg': round(float(res['pitch_deg']), 4),
            'ref_roll_deg': round(float(res['roll_deg']), 4),
            'ref_pos_sigma_m': round(float(res['pos_sigma_m']), 4),
            'ref_heading_sigma_deg': round(float(res['heading_sigma_deg']), 4),
            'nav_state': diag['nav_state'],
            'nhc_active': bool(diag['nhc_active']),
            'vnhc_active': bool(diag['vnhc_active']),
            'compass_active': bool(diag['compass_active']),
            'map_active': bool(diag['map_active']),
            'zupt_active': bool(diag['zupt_active']),
            'gnss_active': bool(diag['gnss_active'])
        }
        epochs_data.append(epoch_record)

        csv_rows.append({
            'step': step,
            'time_s': t_now,
            'ax': acc_step[0], 'ay': acc_step[1], 'az': acc_step[2],
            'gx': gyro_step[0], 'gy': gyro_step[1], 'gz': gyro_step[2],
            'mx': mag_step[0], 'my': mag_step[1], 'mz': mag_step[2],
            'ml_speed': ml_spd,
            'gnss_valid': int(gnss.valid),
            'ref_pos_e': pos_out[0], 'ref_pos_n': pos_out[1], 'ref_pos_u': pos_out[2],
            'ref_vel_e': vel_out[0], 'ref_vel_n': vel_out[1], 'ref_vel_u': vel_out[2],
            'ref_heading': res['heading_deg'],
            'nav_state': diag['nav_state']
        })

    # Save Session JSON
    out_json = DATA_DIR / "golden_reference_session.json"
    session_manifest = {
        'meta': {
            'trip_name': 'Vta04',
            'dt': dt,
            'n_epochs': n,
            'duration_s': round(n * dt, 2),
            'blackout_start_epoch': BLACKOUT_START,
            'blackout_end_epoch': BLACKOUT_END,
            'blackout_duration_s': round((BLACKOUT_END - BLACKOUT_START) * dt, 2),
            'init_pos_enu': [float(D['gt_e'][0]), float(D['gt_n'][0]), float(D['gt_u'][0])],
            'init_vel_enu': [float(D['gt_ve'][0]), float(D['gt_vn'][0]), float(D['gt_vu'][0])],
            'init_heading_deg': float(D['heading'][0]),
            'ba_stat': [float(x) for x in D['ba_stat']],
            'baseline_mag_uT': float(D['baseline_B'])
        },
        'epochs': epochs_data
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(session_manifest, f, indent=2)
    print(f"Saved Golden Reference JSON: {out_json} ({len(epochs_data)} epochs)")

    # Save Session CSV
    out_csv = DATA_DIR / "golden_reference_session.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Saved Golden Reference CSV: {out_csv}")

    print("=" * 78)
    print("GOLDEN REFERENCE SESSION GENERATION COMPLETE!")
    print("=" * 78)


if __name__ == "__main__":
    generate_golden_reference()
