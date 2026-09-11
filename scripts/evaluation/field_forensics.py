import os
import glob
import json
import datetime
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/field_analysis/plots', exist_ok=True)

def audit_file_timing(f):
    df = pd.read_csv(f)
    fname = os.path.basename(f)
    is_raw = 'timestamp_ns' in df.columns
    if is_raw:
        ts_s = df['timestamp_ns'].values.astype(float) * 1e-9
        utc_ms = df['utc_time_ms'].values.astype(float)
    else:
        ts_s = df['timestamp_ms'].values.astype(float) * 1e-3
        utc_ms = df['timestamp_ms'].values.astype(float)
        
    dt = np.diff(ts_s)
    
    # Timing metrics
    res = {
        'file': fname,
        'rows': len(df),
        'duration_s': float(ts_s[-1] - ts_s[0]),
        'mean_dt_ms': float(np.mean(dt) * 1000),
        'median_dt_ms': float(np.median(dt) * 1000),
        'std_dt_ms': float(np.std(dt) * 1000),
        'min_dt_ms': float(np.min(dt) * 1000),
        'max_dt_ms': float(np.max(dt) * 1000),
        'p01_dt_ms': float(np.percentile(dt, 1) * 1000),
        'p05_dt_ms': float(np.percentile(dt, 5) * 1000),
        'p95_dt_ms': float(np.percentile(dt, 95) * 1000),
        'p99_dt_ms': float(np.percentile(dt, 99) * 1000),
        'effective_hz': float(1.0 / np.mean(dt)),
        'zero_dt_count': int(np.sum(dt == 0)),
        'negative_dt_count': int(np.sum(dt < 0)),
        'jitter_iqr_ms': float((np.percentile(dt, 75) - np.percentile(dt, 25)) * 1000),
        'large_gaps_gt_200ms': int(np.sum(dt > 0.2)),
        'large_gaps_gt_500ms': int(np.sum(dt > 0.5)),
    }
    return res

def analyze_stationary_baseline(f):
    df = pd.read_csv(f)
    acc = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = df[['mag_x', 'mag_y', 'mag_z']].values
    
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    mag_norm = np.linalg.norm(mag, axis=1)
    
    res = {
        'file': os.path.basename(f),
        'rows': len(df),
        'accel': {
            'mean_xyz': acc.mean(axis=0).tolist(),
            'std_xyz': acc.std(axis=0).tolist(),
            'rms_xyz': np.sqrt(np.mean(acc**2, axis=0)).tolist(),
            'peak_to_peak_xyz': (acc.max(axis=0) - acc.min(axis=0)).tolist(),
            'norm_mean': float(acc_norm.mean()),
            'norm_std': float(acc_norm.std()),
            'norm_min': float(acc_norm.min()),
            'norm_max': float(acc_norm.max())
        },
        'gyro': {
            'mean_xyz': gyro.mean(axis=0).tolist(),
            'std_xyz': gyro.std(axis=0).tolist(),
            'rms_xyz': np.sqrt(np.mean(gyro**2, axis=0)).tolist(),
            'peak_to_peak_xyz': (gyro.max(axis=0) - gyro.min(axis=0)).tolist(),
            'norm_mean': float(gyro_norm.mean()),
            'norm_std': float(gyro_norm.std()),
            'norm_max': float(gyro_norm.max())
        },
        'mag': {
            'mean_xyz': mag.mean(axis=0).tolist(),
            'std_xyz': mag.std(axis=0).tolist(),
            'norm_mean': float(mag_norm.mean()),
            'norm_std': float(mag_norm.std())
        }
    }
    
    if 'ml_speed' in df.columns:
        s = df['ml_speed'].values
        res['tcn_speed'] = {
            'min': float(s.min()),
            'max': float(s.max()),
            'mean': float(s.mean()),
            'median': float(np.median(s)),
            'p95': float(np.percentile(s, 95)),
            'std': float(s.std())
        }
        
    if 'pos_e' in df.columns:
        pe, pn, pu = df['pos_e'].values, df['pos_n'].values, df['pos_u'].values
        res['eskf_drift'] = {
            'pos_e_drift_m': float(pe[-1] - pe[0]),
            'pos_n_drift_m': float(pn[-1] - pn[0]),
            'pos_u_drift_m': float(pu[-1] - pu[0]),
            'pos_h_drift_m': float(np.sqrt((pe[-1] - pe[0])**2 + (pn[-1] - pn[0])**2)),
            'pos_e_span_m': float(pe.max() - pe.min()),
            'pos_n_span_m': float(pn.max() - pn.min()),
            'pos_u_span_m': float(pu.max() - pu.min()),
            'heading_deg_mean': float(df['heading_deg'].mean()),
            'heading_deg_std': float(df['heading_deg'].std())
        }
    return res

def analyze_outages_in_file(f):
    df = pd.read_csv(f)
    fname = os.path.basename(f)
    
    # State transitions
    states = df['engine_state'].values
    ts_s = df['timestamp_ms'].values.astype(float) * 1e-3
    pe = df['pos_e'].values
    pn = df['pos_n'].values
    pu = df['pos_u'].values
    ml_s = df['ml_speed'].values
    gnss_s = df['gnss_speed'].values
    heading = df['heading_deg'].values
    
    # GNSS lat/lon
    lat = df['gnss_lat'].values
    lon = df['gnss_lon'].values
    
    outage_mask = (states == 'DEAD_RECKONING')
    # Find contiguous intervals of True
    intervals = []
    in_outage = False
    start_idx = 0
    
    for i in range(len(states)):
        if outage_mask[i] and not in_outage:
            in_outage = True
            start_idx = i
        elif not outage_mask[i] and in_outage:
            in_outage = False
            intervals.append((start_idx, i - 1))
    if in_outage:
        intervals.append((start_idx, len(states) - 1))
        
    outage_reports = []
    for idx, (s_idx, e_idx) in enumerate(intervals):
        dur = ts_s[e_idx] - ts_s[s_idx]
        rows = e_idx - s_idx + 1
        
        # GNSS displacement over this interval (ground truth walking motion if GNSS was still tracking)
        # Note: In software blackout, hardware GNSS in background still logged to CSV!
        lat0, lon0 = lat[s_idx], lon[s_idx]
        lat1, lon1 = lat[e_idx], lon[e_idx]
        dlat_m = (lat1 - lat0) * 111139.0
        dlon_m = (lon1 - lon0) * 111139.0 * np.cos(np.radians(lat0))
        gnss_disp_h = np.sqrt(dlat_m**2 + dlon_m**2)
        
        # ESKF predicted displacement over this interval
        eskf_disp_e = pe[e_idx] - pe[s_idx]
        eskf_disp_n = pn[e_idx] - pn[s_idx]
        eskf_disp_u = pu[e_idx] - pu[s_idx]
        eskf_disp_h = np.sqrt(eskf_disp_e**2 + eskf_disp_n**2)
        
        # Drift error between ESKF and GNSS displacement
        disp_error_h = abs(eskf_disp_h - gnss_disp_h)
        drift_rate_m_s = disp_error_h / dur if dur > 0 else 0
        
        outage_reports.append({
            'outage_id': idx + 1,
            'start_idx': int(s_idx),
            'end_idx': int(e_idx),
            'start_time_s': float(ts_s[s_idx] - ts_s[0]),
            'end_time_s': float(ts_s[e_idx] - ts_s[0]),
            'duration_s': float(dur),
            'row_count': int(rows),
            'gnss_disp_h_m': float(gnss_disp_h),
            'eskf_disp_h_m': float(eskf_disp_h),
            'eskf_disp_u_m': float(eskf_disp_u),
            'disp_error_h_m': float(disp_error_h),
            'drift_rate_m_s': float(drift_rate_m_s),
            'ml_speed_mean': float(np.mean(ml_s[s_idx:e_idx+1])),
            'gnss_speed_mean': float(np.mean(gnss_s[s_idx:e_idx+1])),
            'heading_start': float(heading[s_idx]),
            'heading_end': float(heading[e_idx]),
            'heading_change': float(heading[e_idx] - heading[s_idx])
        })
        
    return outage_reports

def main():
    files = sorted(glob.glob('data/field/files/*.csv'))
    
    # 1. Timing audit across all files
    timings = [audit_file_timing(f) for f in files]
    with open('results/field_analysis/timing_audit.json', 'w') as fp:
        json.dump(timings, fp, indent=2)
    print("=== TIMING AUDIT ===")
    for t in timings:
        print(f"{t['file']}: Rate={t['effective_hz']:.2f} Hz, Mean dt={t['mean_dt_ms']:.1f}ms, P99={t['p99_dt_ms']:.1f}ms, Zero dt={t['zero_dt_count']}, Gaps > 200ms={t['large_gaps_gt_200ms']}, Gaps > 500ms={t['large_gaps_gt_500ms']}")

    # 2. Stationary Baseline analysis for stationary candidates
    print("\n=== STATIONARY BASELINE ANALYSIS ===")
    stat_files = [
        'data/field/files/idr_telemetry_20260910_201119.csv',
        'data/field/files/idr_telemetry_20260910_201331.csv'
    ]
    stat_res = [analyze_stationary_baseline(f) for f in stat_files]
    with open('results/field_analysis/stationary_baseline.json', 'w') as fp:
        json.dump(stat_res, fp, indent=2)
    for s in stat_res:
        print(f"{s['file']}:")
        print(f"  Accel norm mean={s['accel']['norm_mean']:.4f} std={s['accel']['norm_std']:.4f}")
        print(f"  Gyro norm mean={s['gyro']['norm_mean']:.5f} std={s['gyro']['norm_std']:.5f} max={s['gyro']['norm_max']:.5f}")
        if 'tcn_speed' in s:
            print(f"  TCN Speed: mean={s['tcn_speed']['mean']:.3f} max={s['tcn_speed']['max']:.3f}")
        if 'eskf_drift' in s:
            print(f"  ESKF Horiz Span={s['eskf_drift']['pos_e_span_m']:.2f}m E, {s['eskf_drift']['pos_n_span_m']:.2f}m N, Vert Span={s['eskf_drift']['pos_u_span_m']:.2f}m")

    # 3. Outage analysis for 202714 (simulated outage toggling)
    outages_202714 = analyze_outages_in_file('data/field/files/idr_telemetry_20260910_202714.csv')
    with open('results/field_analysis/outages_202714.json', 'w') as fp:
        json.dump(outages_202714, fp, indent=2)
    print(f"\n=== SIMULATED OUTAGE ANALYSIS: 202714 (Found {len(outages_202714)} outage intervals) ===")
    for o in outages_202714:
        print(f"  Outage #{o['outage_id']}: Dur={o['duration_s']:.1f}s ({o['start_time_s']:.1f}s -> {o['end_time_s']:.1f}s), GNSS Disp={o['gnss_disp_h_m']:.1f}m, ESKF Disp={o['eskf_disp_h_m']:.1f}m, Vert={o['eskf_disp_u_m']:.1f}m, Drift Rate={o['drift_rate_m_s']:.2f} m/s, TCN Spd={o['ml_speed_mean']:.2f} m/s")

    # 4. Outage analysis for 201650 and 203838
    outages_201650 = analyze_outages_in_file('data/field/files/idr_telemetry_20260910_201650.csv')
    outages_203838 = analyze_outages_in_file('data/field/files/idr_telemetry_20260910_203838.csv')
    with open('results/field_analysis/outages_201650.json', 'w') as fp:
        json.dump(outages_201650, fp, indent=2)
    with open('results/field_analysis/outages_203838.json', 'w') as fp:
        json.dump(outages_203838, fp, indent=2)
    print(f"\n=== OUTAGES IN 201650: {len(outages_201650)} intervals, Total outage rows={sum(o['row_count'] for o in outages_201650)}")
    print(f"=== OUTAGES IN 203838: {len(outages_203838)} intervals, Total outage rows={sum(o['row_count'] for o in outages_203838)}")

if __name__ == '__main__':
    main()
