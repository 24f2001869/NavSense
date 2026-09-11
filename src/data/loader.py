"""
SIH26168 - IO-VNBD Dataset Loader
Robust ingestion and canonical normalization for Smartphone (S) and Vehicle (V) streams.
Handles latin1 encoding, non-ascii units (m/s², µT, °), whitespace stripping,
and standardizes to clean SI units.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "IO-VNBD"

def _normalize_phone_columns(df):
    mapping = {}
    for col in df.columns:
        c = col.strip()
        cu = c.upper()
        if 'LATITUDE' in cu: mapping[col] = 'phone_lat'
        elif 'LONGITUDE' in cu: mapping[col] = 'phone_lon'
        elif 'ALTITUDE' in cu: mapping[col] = 'phone_alt'
        elif 'GPS SPEED' in cu: mapping[col] = 'phone_speed_kmh'
        elif 'ACCURACY' in cu: mapping[col] = 'phone_accuracy_m'
        elif 'GPS ORIENTATION' in cu: mapping[col] = 'phone_bearing_deg'
        elif 'SATELLITES' in cu: mapping[col] = 'phone_sats'
        elif 'TIME SINCE START' in cu: mapping[col] = 'time_ms'
        elif 'DATE' in cu: mapping[col] = 'date_str'
        elif 'ACCELEROMETER X' in cu: mapping[col] = 'accel_x'
        elif 'ACCELEROMETER Y' in cu: mapping[col] = 'accel_y'
        elif 'ACCELEROMETER Z' in cu: mapping[col] = 'accel_z'
        elif 'GRAVITY X' in cu: mapping[col] = 'grav_x'
        elif 'GRAVITY Y' in cu: mapping[col] = 'grav_y'
        elif 'GRAVITY Z' in cu: mapping[col] = 'grav_z'
        elif 'GYROSCOPE YAW' in cu: mapping[col] = 'gyro_z'
        elif 'GYROSCOPE PITCH' in cu: mapping[col] = 'gyro_x'
        elif 'GYROSCOPE ROLL' in cu: mapping[col] = 'gyro_y'
        elif 'MAGNETIC FIELD X' in cu: mapping[col] = 'mag_x'
        elif 'MAGNETIC FIELD Y' in cu: mapping[col] = 'mag_y'
        elif 'MAGNETIC FIELD Z' in cu: mapping[col] = 'mag_z'
        elif 'ORIENTATION (YAW)' in cu or 'AZIMUTH' in cu: mapping[col] = 'ori_yaw_deg'
        elif 'ORIENTATION (PITCH)' in cu: mapping[col] = 'ori_pitch_deg'
        elif 'ORIENTATION (ROLL' in cu: mapping[col] = 'ori_roll_deg'
    return df.rename(columns=mapping)

def _normalize_veh_columns(df):
    mapping = {}
    for col in df.columns:
        c = col.strip()
        cu = c.upper()
        if 'NO OF GPS SATELLITES' in cu: mapping[col] = 'veh_sats'
        elif 'TIME SINCE START OF DAY' in cu: mapping[col] = 'time_s'
        elif 'LATITUDE' in cu: mapping[col] = 'veh_lat'
        elif 'LONGITUDE' in cu: mapping[col] = 'veh_lon'
        elif cu == 'VELOCITY (KM/HR)' or ('VELOCITY' in cu and 'VERTICAL' not in cu): mapping[col] = 'veh_speed_kmh'
        elif 'HEADING' in cu: mapping[col] = 'veh_heading_deg'
        elif 'HEIGHT' in cu: mapping[col] = 'veh_height_km'
        elif 'VERTICAL VELOCITY' in cu: mapping[col] = 'veh_vert_vel_kmh'
        elif 'SAMPLE PERIOD' in cu: mapping[col] = 'sample_period_s'
        elif 'STEERING ANGLE' in cu: mapping[col] = 'steer_angle_deg'
        elif 'WHEEL SPEED FRONT LEFT' in cu: mapping[col] = 'wheel_fl_rads'
        elif 'WHEEL SPEED FRONT RIGHT' in cu: mapping[col] = 'wheel_fr_rads'
        elif 'WHEEL SPEED REAR LEFT' in cu: mapping[col] = 'wheel_rl_rads'
        elif 'WHEEL SPEED REAR RIGHT' in cu: mapping[col] = 'wheel_rr_rads'
        elif 'YAW RATE' in cu: mapping[col] = 'yaw_rate_degs'
        elif 'INDICATED VEHICLE SPEED' in cu: mapping[col] = 'indicated_speed_kmh'
        elif 'INDICATED LONGITUDINAL ACCELERATION' in cu: mapping[col] = 'veh_accel_long_g'
        elif 'INDICATED LATERAL ACCELERATION' in cu: mapping[col] = 'veh_accel_lat_g'
        elif 'HANDBRAKE' in cu: mapping[col] = 'handbrake'
        elif 'ENGINE SPEED' in cu: mapping[col] = 'engine_rpm'
        elif 'BRAKE PRESSURE' in cu: mapping[col] = 'brake_pressure_psi'
        elif 'BRAKE POSITION' in cu: mapping[col] = 'brake_pos'
        elif 'BATTERY VOLTAGE' in cu: mapping[col] = 'battery_voltage'
        elif 'ACCELERATOR' in cu: mapping[col] = 'throttle_pos'
    return df.rename(columns=mapping)

def find_trip_dir(trip_name="Vta02", driver="Vta (Driver E)"):
    categorised = DATASET_ROOT / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / driver / trip_name
    if categorised.exists():
        return categorised
    for p in DATASET_ROOT.rglob(trip_name):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find trip directory for {trip_name}")

def load_trip(trip_dir_or_name):
    """
    Loads and standardizes synchronized smartphone and vehicle CSVs.
    Returns:
        df_phone: DataFrame with clean phone sensor channels
        df_veh: DataFrame with clean vehicle ground truth channels
    """
    if isinstance(trip_dir_or_name, str) and not Path(trip_dir_or_name).exists():
        trip_dir = find_trip_dir(trip_dir_or_name)
    else:
        trip_dir = Path(trip_dir_or_name)

    s_files = list(trip_dir.glob("S-*.csv"))
    v_files = list(trip_dir.glob("V-*.csv"))

    if not s_files or not v_files:
        raise FileNotFoundError(f"Trip folder {trip_dir} must contain both S-*.csv and V-*.csv")

    s_path = s_files[0]
    v_path = v_files[0]

    # Verify these are real binary CSV payloads, not LFS stubs
    if s_path.stat().st_size < 1000 or v_path.stat().st_size < 1000:
        raise ValueError(f"Trip files in {trip_dir} appear to be Git LFS pointers. Run download_lfs.py first.")

    # Read CSVs
    df_phone_raw = pd.read_csv(s_path, encoding='latin1')
    df_veh_raw = pd.read_csv(v_path, encoding='latin1')

    # Normalize column names
    df_phone = _normalize_phone_columns(df_phone_raw)
    df_veh = _normalize_veh_columns(df_veh_raw)

    # Relative time starting at 0.0s
    if 'time_ms' in df_phone.columns:
        df_phone['time_s'] = (df_phone['time_ms'] - df_phone['time_ms'].iloc[0]) / 1000.0
    else:
        df_phone['time_s'] = np.arange(len(df_phone)) * 0.1

    if 'time_s' in df_veh.columns:
        df_veh['time_rel_s'] = df_veh['time_s'] - df_veh['time_s'].iloc[0]
    else:
        df_veh['time_rel_s'] = np.arange(len(df_veh)) * 0.1

    # Standardize velocities to m/s
    if 'veh_speed_kmh' in df_veh.columns:
        df_veh['veh_speed_ms'] = df_veh['veh_speed_kmh'] / 3.6
    if 'phone_speed_kmh' in df_phone.columns:
        df_phone['phone_speed_ms'] = df_phone['phone_speed_kmh'] / 3.6

    # Accelerations from g to m/s^2
    if 'veh_accel_long_g' in df_veh.columns:
        df_veh['veh_accel_long_ms2'] = df_veh['veh_accel_long_g'] * 9.80665
    if 'veh_accel_lat_g' in df_veh.columns:
        df_veh['veh_accel_lat_ms2'] = df_veh['veh_accel_lat_g'] * 9.80665

    # Align lengths
    n = min(len(df_phone), len(df_veh))
    df_phone = df_phone.iloc[:n].reset_index(drop=True)
    df_veh = df_veh.iloc[:n].reset_index(drop=True)

    return df_phone, df_veh

if __name__ == "__main__":
    s, v = load_trip("Vta02")
    print(f"Loaded Vta02: {len(s)} rows.")
    print("Clean Phone columns:", list(s.columns))
    print("Clean Veh columns:", list(v.columns))
