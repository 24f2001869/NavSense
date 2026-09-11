"""
Deep alignment test: Check if S and V in IO-VNBD are aligned row-by-row or if there is a time lag.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import find_trip_dir

for trip in ["Vta02", "Vta03", "Vta04"]:
    print(f"\n===================================================================")
    print(f"ALIGNMENT ANALYSIS FOR {trip}")
    print(f"===================================================================")
    trip_dir = find_trip_dir(trip)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')
    
    s_lat = df_s[[c for c in df_s.columns if 'LATITUDE' in c.upper()][0]].to_numpy()
    s_lon = df_s[[c for c in df_s.columns if 'LONGITUDE' in c.upper()][0]].to_numpy()
    s_spd = df_s[[c for c in df_s.columns if 'SPEED' in c.upper()][0]].to_numpy() * 3.6  # convert m/s to km/h!
    s_ax = df_s[[c for c in df_s.columns if 'ACCELEROMETER X' in c.upper()][0]].to_numpy()
    
    v_lat = df_v[[c for c in df_v.columns if 'LATITUDE' in c.upper()][0]].to_numpy()
    v_lon = df_v[[c for c in df_v.columns if 'LONGITUDE' in c.upper()][0]].to_numpy()
    v_spd = df_v[[c for c in df_v.columns if 'VELOCITY (KM/HR)' in c.upper()][0]].to_numpy()
    v_ax = df_v[[c for c in df_v.columns if 'LONGITUDINAL' in c.upper()][0]].to_numpy() * 9.80665
    
    # 1. Coordinate distance at row 0
    d_lat_0 = (s_lat[0] - v_lat[0]) * 111320
    d_lon_0 = (s_lon[0] - v_lon[0]) * 111320 * np.cos(np.radians(s_lat[0]))
    dist_0 = np.sqrt(d_lat_0**2 + d_lon_0**2)
    print(f"Row 0 Position Distance: {dist_0:.2f} meters (dLat={d_lat_0:.2f}m, dLon={d_lon_0:.2f}m)")
    
    # 2. Coordinate distance at last row
    d_lat_end = (s_lat[-1] - v_lat[-1]) * 111320
    d_lon_end = (s_lon[-1] - v_lon[-1]) * 111320 * np.cos(np.radians(s_lat[-1]))
    dist_end = np.sqrt(d_lat_end**2 + d_lon_end**2)
    print(f"Last Row Position Distance: {dist_end:.2f} meters (dLat={d_lat_end:.2f}m, dLon={d_lon_end:.2f}m)")
    
    # 3. Speed cross-correlation sweep (-100 to +100 samples, i.e. +/- 10 seconds)
    lags = np.arange(-100, 101)
    speed_corrs = []
    for lag in lags:
        if lag < 0:
            s_ = s_spd[:lag]
            v_ = v_spd[-lag:]
        elif lag > 0:
            s_ = s_spd[lag:]
            v_ = v_spd[:-lag]
        else:
            s_ = s_spd
            v_ = v_spd
        mask = (~np.isnan(s_)) & (~np.isnan(v_))
        speed_corrs.append(np.corrcoef(s_[mask], v_[mask])[0, 1])
        
    best_speed_lag = lags[np.argmax(speed_corrs)]
    print(f"Best GPS Speed Lag: {best_speed_lag} samples ({best_speed_lag*0.1:+.2f} s), Max Corr: {np.max(speed_corrs):.4f}")
    print(f"GPS Speed Corr at 0 lag: {speed_corrs[100]:.4f}")
    
    # 4. In Vta03: why did it have negative correlation? Let's check s_spd vs v_spd in Vta03!
    if trip == "Vta03":
        print("\nVta03 Speed values preview (first 10 samples):")
        print("Phone Speed (m/s * 3.6 = km/h):", s_spd[:10])
        print("VBOX Speed (km/h):            ", v_spd[:10])
        print("Vta03 Speed values preview (last 10 samples):")
        print("Phone Speed (m/s * 3.6 = km/h):", s_spd[-10:])
        print("VBOX Speed (km/h):            ", v_spd[-10:])
