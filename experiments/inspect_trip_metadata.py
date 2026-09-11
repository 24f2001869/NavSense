"""
Inspect coordinates, speeds, and timestamps of Vta03 and Vta04 to understand the alignment.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_trip, find_trip_dir

for trip in ["Vta02", "Vta03", "Vta04"]:
    print(f"\n===================================================================")
    print(f"TRIP: {trip}")
    print(f"===================================================================")
    trip_dir = find_trip_dir(trip)
    s_path = list(trip_dir.glob("S-*.csv"))[0]
    v_path = list(trip_dir.glob("V-*.csv"))[0]
    
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path, encoding='latin1')
    
    # Phone columns
    s_lat_col = [c for c in df_s.columns if 'LATITUDE' in c.upper()][0]
    s_lon_col = [c for c in df_s.columns if 'LONGITUDE' in c.upper()][0]
    s_spd_col = [c for c in df_s.columns if 'SPEED' in c.upper()][0]
    s_time_col = [c for c in df_s.columns if 'TIME SINCE START' in c.upper()][0]
    s_date_col = [c for c in df_s.columns if 'DATE' in c.upper()][0]
    
    # Veh columns
    v_lat_col = [c for c in df_v.columns if 'LATITUDE' in c.upper()][0]
    v_lon_col = [c for c in df_v.columns if 'LONGITUDE' in c.upper()][0]
    v_spd_col = [c for c in df_v.columns if 'VELOCITY (KM/HR)' in c.upper()][0]
    v_time_col = [c for c in df_v.columns if 'TIME SINCE START' in c.upper()][0]
    
    print(f"Phone Start Time ({s_time_col}): {df_s[s_time_col].iloc[0]} -> End: {df_s[s_time_col].iloc[-1]} (Duration: {(df_s[s_time_col].iloc[-1] - df_s[s_time_col].iloc[0])/1000.0:.1f}s)")
    print(f"Phone Start Date: {df_s[s_date_col].iloc[0]} -> End Date: {df_s[s_date_col].iloc[-1]}")
    print(f"Veh Start Time ({v_time_col}): {df_v[v_time_col].iloc[0]} -> End: {df_v[v_time_col].iloc[-1]} (Duration: {df_v[v_time_col].iloc[-1] - df_v[v_time_col].iloc[0]:.1f}s)")
    
    print(f"Phone Start Lat/Lon: ({df_s[s_lat_col].iloc[0]:.6f}, {df_s[s_lon_col].iloc[0]:.6f}) -> End: ({df_s[s_lat_col].iloc[-1]:.6f}, {df_s[s_lon_col].iloc[-1]:.6f})")
    print(f"Veh Start Lat/Lon:   ({df_v[v_lat_col].iloc[0]:.6f}, {df_v[v_lon_col].iloc[0]:.6f}) -> End: ({df_v[v_lat_col].iloc[-1]:.6f}, {df_v[v_lon_col].iloc[-1]:.6f})")
    
    # Check max speeds
    print(f"Phone Max Speed: {df_s[s_spd_col].max():.2f} km/h, Veh Max Speed: {df_v[v_spd_col].max():.2f} km/h")
    print(f"Phone Mean Speed: {df_s[s_spd_col].mean():.2f} km/h, Veh Mean Speed: {df_v[v_spd_col].mean():.2f} km/h")
