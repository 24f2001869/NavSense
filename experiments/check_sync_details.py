"""
Check GPS speed cross-correlation and timestamp synchronization across rows in S vs V.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip, find_trip_dir

for trip in ["Vta02", "Vta03", "Vta04"]:
    print(f"\n===================================================================")
    print(f"SYNCHRONIZATION AUDIT FOR {trip}")
    print(f"===================================================================")
    trip_dir = find_trip_dir(trip)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')
    
    # Check column names
    print("Phone cols:", [c for c in df_s.columns if 'SPEED' in c.upper() or 'TIME' in c.upper() or 'DATE' in c.upper()])
    print("Veh cols:", [c for c in df_v.columns if 'VELOCITY' in c.upper() or 'TIME' in c.upper() or 'SPEED' in c.upper()])
    
    # Check lengths
    print(f"Raw rows: S={len(df_s)}, V={len(df_v)}")
    
    n = min(len(df_s), len(df_v))
    s_sub = df_s.iloc[:n].reset_index(drop=True)
    v_sub = df_v.iloc[:n].reset_index(drop=True)
    
    # Phone GPS speed vs VBOX velocity
    col_v_speed = [c for c in v_sub.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    col_s_speed = [c for c in s_sub.columns if 'GPS SPEED' in c.upper()][0]
    
    v_speed = v_sub[col_v_speed].to_numpy()
    s_speed = s_sub[col_s_speed].to_numpy()
    
    # Check correlation row-by-row
    valid = (~np.isnan(v_speed)) & (~np.isnan(s_speed))
    r_row = np.corrcoef(v_speed[valid], s_speed[valid])[0, 1]
    print(f"Row-by-Row GPS Speed Correlation: r = {r_row:.4f}")
    
    # Check lag sweep between phone speed and vehicle speed
    lags = np.arange(-50, 51) # +/- 5 seconds
    corrs = []
    for lag in lags:
        if lag < 0:
            s_s = s_speed[:lag]
            v_s = v_speed[-lag:]
        elif lag > 0:
            s_s = s_speed[lag:]
            v_s = v_speed[:-lag]
        else:
            s_s = s_speed
            v_s = v_speed
        mask = (~np.isnan(s_s)) & (~np.isnan(v_s))
        if np.sum(mask) > 100:
            corrs.append(np.corrcoef(s_s[mask], v_s[mask])[0, 1])
        else:
            corrs.append(0.0)
            
    best_lag = lags[np.argmax(corrs)]
    best_corr = np.max(corrs)
    print(f"Best Speed Lag: {best_lag} samples ({best_lag*0.1:+.2f} s), Max Corr = {best_corr:.4f}")
    print(f"Corr at 0 lag = {corrs[50]:.4f}")
    
    # Also check acceleration correlation between CAN accel and Phone leveled accel / raw accel
    # In V-*.csv: ' Indicated Longitudinal Acceleration (g)'
    col_can_acc = [c for c in v_sub.columns if 'LONGITUDINAL' in c.upper()][0]
    col_s_ax = [c for c in s_sub.columns if 'ACCELEROMETER X' in c.upper()][0]
    can_acc = v_sub[col_can_acc].to_numpy() * 9.80665
    s_ax = s_sub[col_s_ax].to_numpy()
    
    # Lag sweep between CAN accel and raw Phone ax
    acc_corrs = []
    for lag in lags:
        if lag < 0:
            s_a = s_ax[:lag]
            c_a = can_acc[-lag:]
        elif lag > 0:
            s_a = s_ax[lag:]
            c_a = can_acc[:-lag]
        else:
            s_a = s_ax
            c_a = can_acc
        mask = (~np.isnan(s_a)) & (~np.isnan(c_a))
        if np.sum(mask) > 100:
            acc_corrs.append(np.corrcoef(s_a[mask], c_a[mask])[0, 1])
        else:
            acc_corrs.append(0.0)
            
    best_acc_lag = lags[np.argmax(acc_corrs)]
    best_acc_corr = np.max(acc_corrs)
    print(f"Best Accel Lag (s_ax vs can_acc): {best_acc_lag} samples ({best_acc_lag*0.1:+.2f} s), Max Corr = {best_acc_corr:.4f}")
    print(f"Accel Corr at 0 lag = {acc_corrs[50]:.4f}")
