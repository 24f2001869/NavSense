#!/usr/bin/env python3
"""
Phase 5.1 — Real Smartphone Field-Data Forensic Analysis
=========================================================
Performs a complete forensic inventory and analysis of every telemetry CSV
pulled from the Android phone via ADB.

Key constraints:
- The phone was NOT vehicle-mounted.  All moving recordings are PEDESTRIAN (walking).
- The locked nav architecture (TCN-kin → ESKF → V1 damping) is NOT modified.
- This script is OBSERVATION ONLY.

Output:  results/phase5_1_field/  (report + figures)
"""

import os, sys, json, textwrap, warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT   = Path(rstr(PROJECT_ROOT))
FIELD  = ROOT / "data" / "field" / "files"
OUT    = ROOT / "results" / "phase5_1_field"
OUT.mkdir(parents=True, exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))

# ─────────────────────────────────────────────────────────────
# 1. LOAD ALL FILES
# ─────────────────────────────────────────────────────────────
def load_idr(path):
    """Load an idr_telemetry_*.csv file."""
    df = pd.read_csv(path)
    # timestamp_ms is elapsedRealtime in ms
    df["t_sec"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    df["dt_ms"] = df["timestamp_ms"].diff()
    return df

def load_raw(path):
    """Load a raw_nord_*.csv file."""
    df = pd.read_csv(path)
    df["t_sec"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 1e9
    df["dt_ms"] = df["timestamp_ns"].diff() / 1e6
    return df

# Discover files
idr_files = sorted(FIELD.glob("idr_telemetry_*.csv"))
raw_files = sorted(FIELD.glob("raw_nord_*.csv"))
all_files = sorted(list(idr_files) + list(raw_files), key=lambda p: p.name)

print(f"Found {len(idr_files)} IDR telemetry files, {len(raw_files)} raw sensor files")
print(f"Total: {len(all_files)} files\n")

# ─────────────────────────────────────────────────────────────
# 2. DATASET INVENTORY & MANIFEST
# ─────────────────────────────────────────────────────────────
manifest_rows = []

dfs = {}  # name -> df

for f in all_files:
    is_raw = f.name.startswith("raw_")
    try:
        if is_raw:
            df = load_raw(f)
            ts_col = "timestamp_ns"
            first_ts = df[ts_col].iloc[0]
            last_ts  = df[ts_col].iloc[-1]
            dur_sec  = (last_ts - first_ts) / 1e9
        else:
            df = load_idr(f)
            ts_col = "timestamp_ms"
            first_ts = df[ts_col].iloc[0]
            last_ts  = df[ts_col].iloc[-1]
            dur_sec  = (last_ts - first_ts) / 1000.0
    except Exception as e:
        print(f"ERROR loading {f.name}: {e}")
        continue

    dfs[f.name] = df

    dt = df["dt_ms"].dropna()
    n_rows    = len(df)
    file_size = f.stat().st_size
    med_dt    = dt.median()
    mean_dt   = dt.mean()
    p01_dt    = np.percentile(dt, 1)
    p99_dt    = np.percentile(dt, 99)
    eff_rate  = 1000.0 / med_dt if med_dt > 0 else 0
    n_dup     = (dt == 0).sum()
    n_nan     = int(df.isna().sum().sum())

    # Monotonicity
    if is_raw:
        mono = (df["timestamp_ns"].diff().dropna() > 0).all()
    else:
        mono = (df["timestamp_ms"].diff().dropna() > 0).all()

    # GNSS availability
    if "engine_state" in df.columns:
        n_gnss = (df["engine_state"] == "GNSS_LOCKED").sum()
        gnss_dur = n_gnss / eff_rate if eff_rate > 0 else 0
        n_dr   = (df["engine_state"] == "DEAD_RECKONING").sum()
    elif "gnss_lat" in df.columns:
        n_gnss = (df["gnss_lat"] != 0).sum()
        gnss_dur = n_gnss / eff_rate if eff_rate > 0 else 0
        n_dr   = 0
    else:
        n_gnss = 0; gnss_dur = 0; n_dr = 0

    # Columns
    cols = list(df.columns)

    # Parse filename timestamp
    parts = f.stem.split("_")
    if is_raw:
        fn_ts_str = parts[-2] + "_" + parts[-1]
    else:
        fn_ts_str = parts[-2] + "_" + parts[-1]
    try:
        fn_dt = datetime.strptime(fn_ts_str, "%Y%m%d_%H%M%S")
        fn_dt = fn_dt.replace(tzinfo=IST)
    except:
        fn_dt = None

    row = {
        "filename": f.name,
        "file_size_bytes": file_size,
        "n_rows": n_rows,
        "ts_column": ts_col,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration_sec": round(dur_sec, 2),
        "duration_str": f"{int(dur_sec//60)}m {dur_sec%60:.1f}s",
        "median_dt_ms": round(med_dt, 2),
        "mean_dt_ms": round(mean_dt, 2),
        "p01_dt_ms": round(p01_dt, 2),
        "p99_dt_ms": round(p99_dt, 2),
        "eff_sample_rate_hz": round(eff_rate, 2),
        "monotonic": bool(mono),
        "n_duplicate_ts": int(n_dup),
        "n_nan_cells": n_nan,
        "n_gnss_fixes": int(n_gnss),
        "n_dead_reckoning": int(n_dr),
        "gnss_avail_sec": round(gnss_dur, 1),
        "filename_datetime_IST": str(fn_dt) if fn_dt else "UNKNOWN",
        "n_columns": len(cols),
        "columns": cols,
        "is_raw": is_raw,
    }
    manifest_rows.append(row)
    print(f"  {f.name}: {n_rows} rows, {dur_sec:.1f}s, {eff_rate:.1f} Hz, "
          f"GNSS={n_gnss}, DR={n_dr}")

# Save manifest
manifest_path = OUT / "file_manifest.json"
with open(manifest_path, "w") as fp:
    json.dump(manifest_rows, fp, indent=2, default=str)
print(f"\nManifest saved to {manifest_path}\n")

# ─────────────────────────────────────────────────────────────
# 3. RECONSTRUCT CHRONOLOGY & INFER TEST TYPE
# ─────────────────────────────────────────────────────────────
def infer_test_type(name, df, m):
    """Infer test type from data characteristics."""
    is_raw = m["is_raw"]

    if is_raw:
        # Raw sensor file — check GNSS speed and motion
        if "gnss_speed" in df.columns:
            max_speed = df["gnss_speed"].max()
            mean_speed = df["gnss_speed"].mean()
        else:
            max_speed = 0; mean_speed = 0

        if "protocol_marker" in df.columns:
            has_outage = (df["protocol_marker"] != "NORMAL").any()
        else:
            has_outage = False

        if max_speed < 0.3 and mean_speed < 0.1:
            return "STATIONARY_SENSOR_CAPTURE", 0.8
        elif has_outage:
            return "MOVING_SIMULATED_GNSS_OUTAGE", 0.6
        else:
            return "MOVING_GNSS_ON", 0.5

    # IDR telemetry
    has_gnss = "engine_state" in df.columns
    if has_gnss:
        n_gnss = (df["engine_state"] == "GNSS_LOCKED").sum()
        n_dr   = (df["engine_state"] == "DEAD_RECKONING").sum()
        frac_dr = n_dr / len(df) if len(df) > 0 else 0
    else:
        n_gnss = 0; n_dr = 0; frac_dr = 0

    # Check motion via GNSS speed and ML speed
    if "gnss_speed" in df.columns:
        max_gnss_spd = df["gnss_speed"].max()
        mean_gnss_spd = df["gnss_speed"].mean()
    else:
        max_gnss_spd = 0; mean_gnss_spd = 0

    if "ml_speed" in df.columns:
        max_ml_spd = df["ml_speed"].max()
        mean_ml_spd = df["ml_speed"].mean()
    else:
        max_ml_spd = 0; mean_ml_spd = 0

    dur = m["duration_sec"]

    # Check accelerometer variance as motion proxy
    if "accel_x" in df.columns:
        accel_std = df[["accel_x","accel_y","accel_z"]].std().mean()
    else:
        accel_std = 0

    # Check position displacement
    if "pos_e" in df.columns and "pos_n" in df.columns:
        disp = np.sqrt(
            (df["pos_e"].iloc[-1] - df["pos_e"].iloc[0])**2 +
            (df["pos_n"].iloc[-1] - df["pos_n"].iloc[0])**2
        )
    else:
        disp = 0

    # GNSS lat/lon displacement
    if "gnss_lat" in df.columns and "gnss_lon" in df.columns:
        lat0, lon0 = df["gnss_lat"].iloc[0], df["gnss_lon"].iloc[0]
        lat1, lon1 = df["gnss_lat"].iloc[-1], df["gnss_lon"].iloc[-1]
        gnss_disp = np.sqrt(((lat1-lat0)*111320)**2 + ((lon1-lon0)*111320*np.cos(np.radians(lat0)))**2)
    else:
        gnss_disp = 0

    # Very short recording with little data
    if dur < 15 and len(df) < 50:
        if max_gnss_spd < 0.3 and accel_std < 0.5:
            return "STATIONARY_ENGINE_OFF", 0.6
        return "UNKNOWN_SHORT", 0.3

    # Stationary: no motion, no displacement
    if max_gnss_spd < 0.3 and gnss_disp < 5 and accel_std < 0.5:
        return "STATIONARY_ENGINE_OFF", 0.7

    # Has outage toggling?
    if frac_dr > 0.1 and frac_dr < 0.9 and n_dr > 20 and n_gnss > 20:
        return "MOVING_SIMULATED_GNSS_OUTAGE", 0.8

    # Predominantly dead reckoning from start
    if frac_dr > 0.9:
        if max_gnss_spd < 0.3 and accel_std < 0.5:
            return "STATIONARY_ENGINE_OFF", 0.5
        elif mean_ml_spd > 0.5:
            return "MOVING_GNSS_OFF", 0.6
        else:
            return "STATIONARY_DR_MODE", 0.5

    # Moving with GNSS
    if mean_gnss_spd > 0.3 or gnss_disp > 10:
        return "MOVING_GNSS_ON", 0.7

    # If we reach here, try accel
    if accel_std > 0.5:
        return "MOVING_GNSS_ON", 0.4

    return "UNKNOWN", 0.3


chrono = []
for m in manifest_rows:
    name = m["filename"]
    df = dfs[name]
    test_type, conf = infer_test_type(name, df, m)
    chrono.append({
        "filename": name,
        "fn_datetime": m["filename_datetime_IST"],
        "duration": m["duration_str"],
        "duration_sec": m["duration_sec"],
        "n_rows": m["n_rows"],
        "inferred_test": test_type,
        "confidence": conf,
    })

# Sort by filename timestamp
chrono.sort(key=lambda x: x["fn_datetime"])

print("=" * 80)
print("RECONSTRUCTED CHRONOLOGY")
print("=" * 80)
for c in chrono:
    print(f"  {c['filename']:45s}  {c['duration']:>10s}  {c['inferred_test']:35s}  conf={c['confidence']:.1f}")
print()

# ─────────────────────────────────────────────────────────────
# 4. SENSOR TIMING AUDIT (ALL FILES)
# ─────────────────────────────────────────────────────────────
print("=" * 80)
print("SENSOR TIMING AUDIT")
print("=" * 80)

timing_report = []
for m in manifest_rows:
    name = m["filename"]
    df = dfs[name]
    dt = df["dt_ms"].dropna()

    # Check for gaps (> 3x median)
    med = dt.median()
    if med > 0:
        gaps = dt[dt > 3 * med]
        n_gaps = len(gaps)
        max_gap = dt.max()
    else:
        n_gaps = 0; max_gap = 0

    # Check for burst delivery (dt < 0.5 * median)
    if med > 0:
        bursts = dt[dt < 0.5 * med]
        n_bursts = len(bursts)
    else:
        n_bursts = 0

    entry = {
        "filename": name,
        "n_rows": m["n_rows"],
        "eff_rate_hz": m["eff_sample_rate_hz"],
        "median_dt_ms": m["median_dt_ms"],
        "mean_dt_ms": m["mean_dt_ms"],
        "std_dt_ms": round(dt.std(), 2) if len(dt) > 0 else 0,
        "p01_dt_ms": m["p01_dt_ms"],
        "p99_dt_ms": m["p99_dt_ms"],
        "min_dt_ms": round(dt.min(), 2) if len(dt) > 0 else 0,
        "max_dt_ms": round(dt.max(), 2) if len(dt) > 0 else 0,
        "monotonic": m["monotonic"],
        "n_gaps_gt_3x": n_gaps,
        "n_bursts_lt_half": n_bursts,
        "n_dup_ts": m["n_duplicate_ts"],
    }
    timing_report.append(entry)
    print(f"  {name}: rate={entry['eff_rate_hz']:.1f}Hz, "
          f"dt={entry['median_dt_ms']:.1f}±{entry['std_dt_ms']:.1f}ms, "
          f"mono={'OK' if entry['monotonic'] else 'FAIL'}, "
          f"gaps={n_gaps}, bursts={n_bursts}, dups={entry['n_dup_ts']}")

print()

# ─────────────────────────────────────────────────────────────
# HELPER: Common sensor stats
# ─────────────────────────────────────────────────────────────
def sensor_stats(series, name=""):
    """Return dict of stats for a sensor series."""
    s = series.dropna()
    if len(s) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": len(s),
        "mean": round(float(s.mean()), 5),
        "std": round(float(s.std()), 5),
        "min": round(float(s.min()), 5),
        "max": round(float(s.max()), 5),
        "median": round(float(s.median()), 5),
        "p01": round(float(np.percentile(s, 1)), 5),
        "p05": round(float(np.percentile(s, 5)), 5),
        "p95": round(float(np.percentile(s, 95)), 5),
        "p99": round(float(np.percentile(s, 99)), 5),
        "rms": round(float(np.sqrt((s**2).mean())), 5),
        "ptp": round(float(s.max() - s.min()), 5),
    }

def print_sensor_block(stats_list, title):
    """Pretty-print a list of sensor stat dicts."""
    print(f"\n  {title}")
    print(f"  {'Axis':<8s} {'Mean':>10s} {'Std':>10s} {'RMS':>10s} {'PtP':>10s} {'Min':>10s} {'Max':>10s}")
    for st in stats_list:
        if st.get("count", 0) == 0:
            continue
        print(f"  {st['name']:<8s} {st['mean']:>10.4f} {st['std']:>10.4f} "
              f"{st['rms']:>10.4f} {st['ptp']:>10.4f} {st['min']:>10.4f} {st['max']:>10.4f}")

# ─────────────────────────────────────────────────────────────
# 5. PER-FILE DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────
file_analyses = {}

for m in manifest_rows:
    name = m["filename"]
    df = dfs[name]
    analysis = {"filename": name, "test_type": None, "sensor": {}, "nav": {}}

    # Find the inferred test type
    for c in chrono:
        if c["filename"] == name:
            analysis["test_type"] = c["inferred_test"]
            break

    print("=" * 80)
    print(f"DEEP ANALYSIS: {name}  [{analysis['test_type']}]")
    print(f"  Duration: {m['duration_str']}, Rows: {m['n_rows']}, Rate: {m['eff_sample_rate_hz']} Hz")
    print("=" * 80)

    # -- Accelerometer --
    accel_cols = [c for c in ["accel_x","accel_y","accel_z"] if c in df.columns]
    if accel_cols:
        accel_stats = [sensor_stats(df[c], c.split("_")[-1].upper()) for c in accel_cols]
        # Magnitude
        amag = np.sqrt(sum(df[c]**2 for c in accel_cols))
        accel_stats.append(sensor_stats(amag, "MAG"))
        print_sensor_block(accel_stats, "ACCELEROMETER (m/s²)")
        analysis["sensor"]["accel"] = accel_stats

    # -- Gyroscope --
    gyro_cols = [c for c in ["gyro_x","gyro_y","gyro_z"] if c in df.columns]
    if gyro_cols:
        gyro_stats = [sensor_stats(df[c], c.split("_")[-1].upper()) for c in gyro_cols]
        gmag = np.sqrt(sum(df[c]**2 for c in gyro_cols))
        gyro_stats.append(sensor_stats(gmag, "MAG"))
        print_sensor_block(gyro_stats, "GYROSCOPE (rad/s)")
        analysis["sensor"]["gyro"] = gyro_stats

    # -- Magnetometer --
    mag_cols = [c for c in ["mag_x","mag_y","mag_z"] if c in df.columns]
    if mag_cols:
        mag_stats = [sensor_stats(df[c], c.split("_")[-1].upper()) for c in mag_cols]
        mmag = np.sqrt(sum(df[c]**2 for c in mag_cols))
        mag_stats.append(sensor_stats(mmag, "MAG"))
        print_sensor_block(mag_stats, "MAGNETOMETER (µT)")
        analysis["sensor"]["mag"] = mag_stats

    # -- Gravity (raw files only) --
    grav_cols = [c for c in ["gravity_x","gravity_y","gravity_z"] if c in df.columns]
    if grav_cols:
        grav_stats = [sensor_stats(df[c], c.split("_")[-1].upper()) for c in grav_cols]
        grmag = np.sqrt(sum(df[c]**2 for c in grav_cols))
        grav_stats.append(sensor_stats(grmag, "MAG"))
        print_sensor_block(grav_stats, "GRAVITY VECTOR (m/s²)")
        analysis["sensor"]["gravity"] = grav_stats

    # -- Linear Acceleration (raw files only) --
    la_cols = [c for c in ["lin_accel_x","lin_accel_y","lin_accel_z"] if c in df.columns]
    if la_cols:
        la_stats = [sensor_stats(df[c], c.split("_")[-1].upper()) for c in la_cols]
        print_sensor_block(la_stats, "LINEAR ACCELERATION (m/s²)")
        analysis["sensor"]["lin_accel"] = la_stats

    # -- GNSS --
    if "gnss_speed" in df.columns:
        gnss_spd = sensor_stats(df["gnss_speed"], "speed")
        gnss_acc = sensor_stats(df.get("gnss_accuracy", df.get("gnss_acc_h", pd.Series())), "acc_h")
        print(f"\n  GNSS Speed: mean={gnss_spd['mean']:.3f}, max={gnss_spd['max']:.3f}, "
              f"std={gnss_spd['std']:.3f} m/s")
        if gnss_acc.get("count", 0) > 0:
            print(f"  GNSS Accuracy: mean={gnss_acc['mean']:.1f}, "
                  f"max={gnss_acc['max']:.1f}, min={gnss_acc['min']:.1f} m")
        analysis["nav"]["gnss_speed"] = gnss_spd
        analysis["nav"]["gnss_acc"] = gnss_acc

    # -- GNSS Position Displacement --
    if "gnss_lat" in df.columns:
        lat = df["gnss_lat"].values
        lon = df["gnss_lon"].values
        # Total displacement
        dlat = (lat[-1] - lat[0]) * 111320
        dlon = (lon[-1] - lon[0]) * 111320 * np.cos(np.radians(lat[0]))
        gnss_disp = np.sqrt(dlat**2 + dlon**2)
        # Max displacement from start
        d_all = np.sqrt(((lat - lat[0])*111320)**2 +
                        ((lon - lon[0])*111320*np.cos(np.radians(lat[0])))**2)
        max_disp = d_all.max()
        print(f"  GNSS Displacement: start→end = {gnss_disp:.1f} m, max from start = {max_disp:.1f} m")
        analysis["nav"]["gnss_displacement_m"] = round(gnss_disp, 2)
        analysis["nav"]["gnss_max_displacement_m"] = round(max_disp, 2)

    # -- TCN / ML Speed --
    if "ml_speed" in df.columns:
        ml = sensor_stats(df["ml_speed"], "ml_speed")
        print(f"\n  TCN Speed: mean={ml['mean']:.3f}, max={ml['max']:.3f}, "
              f"min={ml['min']:.3f}, std={ml['std']:.3f}, p95={ml['p95']:.3f} m/s")
        analysis["nav"]["ml_speed"] = ml

    # -- ESKF Position --
    if "pos_e" in df.columns:
        pe = df["pos_e"].values
        pn = df["pos_n"].values
        pu = df["pos_u"].values if "pos_u" in df.columns else np.zeros_like(pe)
        eskf_disp = np.sqrt((pe[-1]-pe[0])**2 + (pn[-1]-pn[0])**2)
        eskf_max = np.sqrt((pe - pe[0])**2 + (pn - pn[0])**2).max()
        vert_range = float(pu.max() - pu.min()) if "pos_u" in df.columns else 0
        print(f"\n  ESKF Position: disp={eskf_disp:.1f} m, max_from_start={eskf_max:.1f} m, "
              f"vert_range={vert_range:.1f} m")
        analysis["nav"]["eskf_displacement_m"] = round(eskf_disp, 2)
        analysis["nav"]["eskf_max_displacement_m"] = round(eskf_max, 2)
        analysis["nav"]["eskf_vertical_range_m"] = round(vert_range, 2)

    # -- Heading --
    if "heading_deg" in df.columns:
        hdg = df["heading_deg"].dropna()
        print(f"  Heading: mean={hdg.mean():.1f}°, std={hdg.std():.1f}°, "
              f"range=[{hdg.min():.1f}°, {hdg.max():.1f}°]")
        analysis["nav"]["heading_mean"] = round(float(hdg.mean()), 2)
        analysis["nav"]["heading_std"] = round(float(hdg.std()), 2)

    # -- Engine State Distribution --
    if "engine_state" in df.columns:
        state_counts = df["engine_state"].value_counts().to_dict()
        print(f"\n  Engine State: {state_counts}")
        analysis["nav"]["engine_states"] = state_counts

        # Detect transitions
        transitions = (df["engine_state"] != df["engine_state"].shift()).sum() - 1
        print(f"  State transitions: {transitions}")
        analysis["nav"]["state_transitions"] = int(transitions)

    # -- NHC / VNHC flags --
    for flag in ["nhc_enabled", "vnhc_enabled", "compass_accepted", "map_accepted"]:
        if flag in df.columns:
            vals = df[flag]
            if vals.dtype == object:
                n_true = (vals == "true").sum()
            else:
                n_true = vals.sum()
            print(f"  {flag}: {n_true}/{len(df)} ({100*n_true/len(df):.1f}%)")

    file_analyses[name] = analysis

# ─────────────────────────────────────────────────────────────
# 6. GNSS STATE MACHINE VERIFICATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GNSS STATE MACHINE AUDIT")
print("=" * 80)

state_machine_report = {}

for m in manifest_rows:
    name = m["filename"]
    if m["is_raw"]:
        continue  # raw files don't have engine_state
    df = dfs[name]
    if "engine_state" not in df.columns:
        continue

    states = df["engine_state"].values
    ts = df["timestamp_ms"].values

    # Find all contiguous segments
    segments = []
    seg_start = 0
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            segments.append({
                "state": states[seg_start],
                "start_idx": seg_start,
                "end_idx": i-1,
                "start_ts": ts[seg_start],
                "end_ts": ts[i-1],
                "duration_sec": (ts[i-1] - ts[seg_start]) / 1000.0,
                "n_rows": i - seg_start,
            })
            seg_start = i
    segments.append({
        "state": states[seg_start],
        "start_idx": seg_start,
        "end_idx": len(states)-1,
        "start_ts": ts[seg_start],
        "end_ts": ts[-1],
        "duration_sec": (ts[-1] - ts[seg_start]) / 1000.0,
        "n_rows": len(states) - seg_start,
    })

    n_outages = sum(1 for s in segments if s["state"] == "DEAD_RECKONING")
    outage_durations = [s["duration_sec"] for s in segments if s["state"] == "DEAD_RECKONING"]

    print(f"\n  {name}:")
    print(f"    Total segments: {len(segments)}")
    print(f"    GNSS_LOCKED segments: {sum(1 for s in segments if s['state']=='GNSS_LOCKED')}")
    print(f"    DEAD_RECKONING segments: {n_outages}")
    if outage_durations:
        print(f"    Outage durations: min={min(outage_durations):.1f}s, "
              f"max={max(outage_durations):.1f}s, mean={np.mean(outage_durations):.1f}s")
    
    # Check for stale GNSS during DR
    stale_gnss_count = 0
    for seg in segments:
        if seg["state"] == "DEAD_RECKONING" and seg["n_rows"] > 5:
            dr_slice = df.iloc[seg["start_idx"]:seg["end_idx"]+1]
            if "gnss_lat" in dr_slice.columns:
                # Check if GNSS position changes during DR
                lat_unique = dr_slice["gnss_lat"].nunique()
                lon_unique = dr_slice["gnss_lon"].nunique()
                if lat_unique > 1 or lon_unique > 1:
                    stale_gnss_count += 1

    if stale_gnss_count > 0:
        print(f"    ⚠️  STALE GNSS DETECTED: {stale_gnss_count} DR segments with changing GNSS coords")
    else:
        print(f"    ✅ No stale GNSS leakage during DR segments")

    # Check GNSS reacquisition discontinuities
    reacq_jumps = []
    for i, seg in enumerate(segments):
        if seg["state"] == "GNSS_LOCKED" and i > 0 and segments[i-1]["state"] == "DEAD_RECKONING":
            # Transition from DR to GNSS
            dr_end_idx = segments[i-1]["end_idx"]
            gnss_start_idx = seg["start_idx"]
            if "pos_e" in df.columns:
                pe_before = df["pos_e"].iloc[dr_end_idx]
                pn_before = df["pos_n"].iloc[dr_end_idx]
                pe_after  = df["pos_e"].iloc[gnss_start_idx]
                pn_after  = df["pos_n"].iloc[gnss_start_idx]
                jump = np.sqrt((pe_after-pe_before)**2 + (pn_after-pn_before)**2)
                reacq_jumps.append(jump)

    if reacq_jumps:
        print(f"    Reacquisition jumps: min={min(reacq_jumps):.2f}m, "
              f"max={max(reacq_jumps):.2f}m, mean={np.mean(reacq_jumps):.2f}m")

    state_machine_report[name] = {
        "n_segments": len(segments),
        "n_outages": n_outages,
        "outage_durations": outage_durations,
        "stale_gnss": stale_gnss_count,
        "reacq_jumps": reacq_jumps,
        "segments": segments[:50],  # Cap for JSON
    }

# ─────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────

# Figure 1: Timing jitter for all files
fig, axes = plt.subplots(len(dfs), 1, figsize=(14, 3*len(dfs)),
                          squeeze=False)
for i, (name, df) in enumerate(dfs.items()):
    ax = axes[i, 0]
    dt = df["dt_ms"].dropna()
    ax.plot(df["t_sec"].iloc[1:len(dt)+1], dt.values, lw=0.3, alpha=0.6)
    ax.axhline(dt.median(), color="red", lw=1, ls="--", label=f"median={dt.median():.1f}ms")
    ax.set_ylabel("dt (ms)")
    ax.set_title(f"{name}  [rate={1000/dt.median():.1f}Hz]", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(dt.max() * 1.2, dt.median() * 5))
axes[-1, 0].set_xlabel("Time (s)")
fig.suptitle("Phase 5.1 — Sensor Timing Jitter (All Files)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "fig1_timing_jitter.png", dpi=150)
plt.close(fig)
print("\nSaved fig1_timing_jitter.png")

# Figure 2: Per-file sensor overview dashboards
for name, df in dfs.items():
    is_raw = name.startswith("raw_")
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(f"Phase 5.1 — Sensor Dashboard: {name}", fontsize=13, fontweight="bold")

    # Accel XYZ
    ax = fig.add_subplot(gs[0, 0])
    for c in ["accel_x","accel_y","accel_z"]:
        if c in df.columns:
            ax.plot(df["t_sec"], df[c], lw=0.4, label=c, alpha=0.7)
    ax.set_ylabel("m/s²"); ax.set_title("Accelerometer"); ax.legend(fontsize=7)

    # Gyro XYZ
    ax = fig.add_subplot(gs[0, 1])
    for c in ["gyro_x","gyro_y","gyro_z"]:
        if c in df.columns:
            ax.plot(df["t_sec"], df[c], lw=0.4, label=c, alpha=0.7)
    ax.set_ylabel("rad/s"); ax.set_title("Gyroscope"); ax.legend(fontsize=7)

    # Mag XYZ
    ax = fig.add_subplot(gs[0, 2])
    for c in ["mag_x","mag_y","mag_z"]:
        if c in df.columns:
            ax.plot(df["t_sec"], df[c], lw=0.4, label=c, alpha=0.7)
    ax.set_ylabel("µT"); ax.set_title("Magnetometer"); ax.legend(fontsize=7)

    # GNSS Speed vs ML Speed
    ax = fig.add_subplot(gs[1, 0])
    if "gnss_speed" in df.columns:
        ax.plot(df["t_sec"], df["gnss_speed"], lw=0.8, label="GNSS", alpha=0.8)
    if "ml_speed" in df.columns:
        ax.plot(df["t_sec"], df["ml_speed"], lw=0.8, label="TCN", alpha=0.8)
    ax.set_ylabel("m/s"); ax.set_title("Speed Comparison"); ax.legend(fontsize=7)

    # GNSS Trajectory (lat/lon)
    ax = fig.add_subplot(gs[1, 1])
    if "gnss_lat" in df.columns:
        lat = df["gnss_lat"].values
        lon = df["gnss_lon"].values
        # Convert to meters from start
        de = (lon - lon[0]) * 111320 * np.cos(np.radians(lat[0]))
        dn = (lat - lat[0]) * 111320
        ax.plot(de, dn, lw=0.5, alpha=0.7)
        ax.scatter(de[0], dn[0], c="green", s=50, zorder=5, label="Start")
        ax.scatter(de[-1], dn[-1], c="red", s=50, zorder=5, label="End")
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        ax.set_title("GNSS Trajectory"); ax.legend(fontsize=7)
        ax.set_aspect("equal")

    # ESKF Trajectory
    ax = fig.add_subplot(gs[1, 2])
    if "pos_e" in df.columns:
        ax.plot(df["pos_e"], df["pos_n"], lw=0.5, alpha=0.7)
        ax.scatter(df["pos_e"].iloc[0], df["pos_n"].iloc[0], c="green", s=50, zorder=5, label="Start")
        ax.scatter(df["pos_e"].iloc[-1], df["pos_n"].iloc[-1], c="red", s=50, zorder=5, label="End")
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        ax.set_title("ESKF Trajectory"); ax.legend(fontsize=7)
        ax.set_aspect("equal")

    # Heading
    ax = fig.add_subplot(gs[2, 0])
    if "heading_deg" in df.columns:
        ax.plot(df["t_sec"], df["heading_deg"], lw=0.5, alpha=0.7)
        ax.set_ylabel("deg"); ax.set_title("Heading")
    if "gnss_bearing" in df.columns:
        ax.plot(df["t_sec"], df["gnss_bearing"], lw=0.5, alpha=0.5, label="GNSS bearing")
        ax.legend(fontsize=7)

    # GNSS Accuracy
    ax = fig.add_subplot(gs[2, 1])
    acc_col = "gnss_accuracy" if "gnss_accuracy" in df.columns else ("gnss_acc_h" if "gnss_acc_h" in df.columns else None)
    if acc_col:
        ax.plot(df["t_sec"], df[acc_col], lw=0.5, alpha=0.7)
        ax.set_ylabel("m"); ax.set_title("GNSS Horizontal Accuracy")

    # Engine State
    ax = fig.add_subplot(gs[2, 2])
    if "engine_state" in df.columns:
        state_map = {"GNSS_LOCKED": 1, "DEAD_RECKONING": 0}
        state_num = df["engine_state"].map(state_map).fillna(-1)
        ax.fill_between(df["t_sec"], state_num, alpha=0.5, step="post")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["DR", "GNSS"])
        ax.set_title("Engine State")
    elif "protocol_marker" in df.columns:
        pm_map = {"NORMAL": 1, "OUTAGE": 0}
        pm_num = df["protocol_marker"].map(pm_map).fillna(-1)
        ax.fill_between(df["t_sec"], pm_num, alpha=0.5, step="post")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["OUTAGE", "NORMAL"])
        ax.set_title("Protocol Marker")

    # Vertical position
    ax = fig.add_subplot(gs[3, 0])
    if "pos_u" in df.columns:
        ax.plot(df["t_sec"], df["pos_u"], lw=0.5, alpha=0.7, label="ESKF Up")
    if "gnss_alt" in df.columns:
        alt = df["gnss_alt"]
        ax.plot(df["t_sec"], alt - alt.iloc[0], lw=0.5, alpha=0.5, label="GNSS Alt (rel)")
    ax.set_ylabel("m"); ax.set_title("Vertical"); ax.legend(fontsize=7)

    # dt timing
    ax = fig.add_subplot(gs[3, 1])
    dt = df["dt_ms"].dropna()
    ax.hist(dt[dt < dt.median()*3], bins=80, alpha=0.7, edgecolor="none")
    ax.axvline(dt.median(), color="red", ls="--", lw=1.5, label=f"median={dt.median():.1f}ms")
    ax.set_xlabel("dt (ms)"); ax.set_title("Timing Histogram"); ax.legend(fontsize=7)

    # Accel magnitude PSD (if enough data)
    ax = fig.add_subplot(gs[3, 2])
    if "accel_x" in df.columns and len(df) > 256:
        amag = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
        fs = 1000.0 / df["dt_ms"].median() if df["dt_ms"].median() > 0 else 100
        try:
            f_psd, psd = signal.welch(amag.values, fs=fs, nperseg=min(256, len(amag)//2))
            ax.semilogy(f_psd, psd, lw=0.8)
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
            ax.set_title("Accel Magnitude PSD")
        except:
            ax.text(0.5, 0.5, "PSD failed", transform=ax.transAxes, ha="center")

    safe_name = name.replace(".csv", "")
    fig.savefig(OUT / f"fig2_dashboard_{safe_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig2_dashboard_{safe_name}.png")


# Figure 3: TCN speed distribution comparison across all files
fig, axes = plt.subplots(1, 1, figsize=(12, 5))
for name, df in dfs.items():
    if "ml_speed" in df.columns:
        axes.hist(df["ml_speed"].dropna(), bins=60, alpha=0.4,
                  label=f"{name[:30]}...", density=True, edgecolor="none")
axes.set_xlabel("TCN Speed (m/s)")
axes.set_ylabel("Density")
axes.set_title("Phase 5.1 — TCN Speed Distributions (All Recordings)")
axes.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig3_tcn_speed_distributions.png", dpi=150)
plt.close(fig)
print("Saved fig3_tcn_speed_distributions.png")

# Figure 4: Orientation analysis for walking files
for name, df in dfs.items():
    if name.startswith("raw_") and all(c in df.columns for c in ["gravity_x","gravity_y","gravity_z"]):
        # Compute roll/pitch from gravity
        gx = df["gravity_x"].values
        gy = df["gravity_y"].values
        gz = df["gravity_z"].values
        roll  = np.degrees(np.arctan2(gy, gz))
        pitch = np.degrees(np.arctan2(-gx, np.sqrt(gy**2 + gz**2)))

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(df["t_sec"], roll, lw=0.4, alpha=0.7)
        axes[0].set_ylabel("Roll (°)"); axes[0].set_title("Phone Orientation from Gravity")
        axes[1].plot(df["t_sec"], pitch, lw=0.4, alpha=0.7)
        axes[1].set_ylabel("Pitch (°)")

        if "rot_vec_x" in df.columns:
            # Compute yaw from rotation vector
            qx = df["rot_vec_x"].values
            qy = df["rot_vec_y"].values
            qz = df["rot_vec_z"].values
            qw = df["rot_vec_w"].values
            yaw = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
            axes[2].plot(df["t_sec"], yaw, lw=0.4, alpha=0.7)
        axes[2].set_ylabel("Yaw (°)"); axes[2].set_xlabel("Time (s)")

        safe_name = name.replace(".csv", "")
        fig.suptitle(f"Orientation: {name}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / f"fig4_orientation_{safe_name}.png", dpi=130)
        plt.close(fig)
        print(f"Saved fig4_orientation_{safe_name}.png")

    elif not name.startswith("raw_") and "heading_deg" in df.columns:
        # For IDR files, we have limited orientation info
        # Use accelerometer to estimate tilt
        if all(c in df.columns for c in ["accel_x","accel_y","accel_z"]):
            ax_v = df["accel_x"].values
            ay_v = df["accel_y"].values
            az_v = df["accel_z"].values
            roll  = np.degrees(np.arctan2(ay_v, az_v))
            pitch = np.degrees(np.arctan2(-ax_v, np.sqrt(ay_v**2 + az_v**2)))

            fa = file_analyses.get(name, {})
            tt = fa.get("test_type", "")
            # Only plot for moving recordings
            if "MOVING" in tt or "WALKING" in tt or len(df) > 500:
                fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
                axes[0].plot(df["t_sec"], roll, lw=0.3, alpha=0.6)
                axes[0].set_ylabel("Accel Roll (°)")
                axes[0].set_title(f"Orientation (from accel): {name}")
                axes[1].plot(df["t_sec"], pitch, lw=0.3, alpha=0.6)
                axes[1].set_ylabel("Accel Pitch (°)")
                axes[2].plot(df["t_sec"], df["heading_deg"], lw=0.3, alpha=0.6)
                axes[2].set_ylabel("Heading (°)"); axes[2].set_xlabel("Time (s)")
                fig.tight_layout()
                safe_name = name.replace(".csv", "")
                fig.savefig(OUT / f"fig4_orientation_{safe_name}.png", dpi=130)
                plt.close(fig)
                print(f"Saved fig4_orientation_{safe_name}.png")


# Figure 5: GNSS outage analysis for files with outage toggling
for name, df in dfs.items():
    if "engine_state" not in df.columns:
        continue
    sm = state_machine_report.get(name)
    if sm is None or sm["n_outages"] < 2:
        continue

    segments = sm["segments"]
    outage_segs = [s for s in segments if s["state"] == "DEAD_RECKONING"]

    if not outage_segs:
        continue

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot speed with DR shading
    ax = axes[0]
    if "gnss_speed" in df.columns:
        ax.plot(df["t_sec"], df["gnss_speed"], lw=0.8, label="GNSS Speed", alpha=0.8)
    if "ml_speed" in df.columns:
        ax.plot(df["t_sec"], df["ml_speed"], lw=0.8, label="TCN Speed", alpha=0.8)
    # Shade DR periods
    t0 = df["timestamp_ms"].iloc[0]
    for seg in outage_segs:
        t_start = (seg["start_ts"] - t0) / 1000.0
        t_end   = (seg["end_ts"] - t0) / 1000.0
        ax.axvspan(t_start, t_end, alpha=0.15, color="red")
    ax.set_ylabel("m/s"); ax.set_title(f"GNSS Outage Analysis: {name}")
    ax.legend(fontsize=8)

    # Plot ESKF position displacement during DR
    ax = axes[1]
    if "pos_e" in df.columns:
        disp = np.sqrt((df["pos_e"] - df["pos_e"].iloc[0])**2 +
                        (df["pos_n"] - df["pos_n"].iloc[0])**2)
        ax.plot(df["t_sec"], disp, lw=0.8, alpha=0.7, label="ESKF disp from start")
    for seg in outage_segs:
        t_start = (seg["start_ts"] - t0) / 1000.0
        t_end   = (seg["end_ts"] - t0) / 1000.0
        ax.axvspan(t_start, t_end, alpha=0.15, color="red")
    ax.set_ylabel("m"); ax.legend(fontsize=8)

    # Heading
    ax = axes[2]
    if "heading_deg" in df.columns:
        ax.plot(df["t_sec"], df["heading_deg"], lw=0.5, alpha=0.7)
    for seg in outage_segs:
        t_start = (seg["start_ts"] - t0) / 1000.0
        t_end   = (seg["end_ts"] - t0) / 1000.0
        ax.axvspan(t_start, t_end, alpha=0.15, color="red")
    ax.set_ylabel("Heading (°)"); ax.set_xlabel("Time (s)")

    safe_name = name.replace(".csv", "")
    fig.tight_layout()
    fig.savefig(OUT / f"fig5_outage_{safe_name}.png", dpi=130)
    plt.close(fig)
    print(f"Saved fig5_outage_{safe_name}.png")


# ─────────────────────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PHASE 5.1 — SUMMARY")
print("=" * 80)

print("\n--- FILE INVENTORY ---")
for m in manifest_rows:
    print(f"  {m['filename']:45s}  {m['file_size_bytes']:>10d} bytes  "
          f"{m['n_rows']:>6d} rows  {m['duration_str']:>10s}  "
          f"{m['eff_sample_rate_hz']:>6.1f} Hz")

print("\n--- CHRONOLOGY ---")
for c in chrono:
    print(f"  {c['fn_datetime']:30s}  {c['filename']:45s}  "
          f"{c['duration']:>10s}  {c['inferred_test']:35s}  conf={c['confidence']:.1f}")

print("\n--- TIMING HEALTH ---")
all_mono = all(t["monotonic"] for t in timing_report)
total_gaps = sum(t["n_gaps_gt_3x"] for t in timing_report)
total_dups = sum(t["n_dup_ts"] for t in timing_report)
rates = [t["eff_rate_hz"] for t in timing_report]
print(f"  Monotonicity: {'ALL OK' if all_mono else 'FAILURES DETECTED'}")
print(f"  Total gaps (>3x median): {total_gaps}")
print(f"  Total duplicate timestamps: {total_dups}")
print(f"  Sample rates: {min(rates):.1f} – {max(rates):.1f} Hz")

print("\n--- KEY OBSERVATIONS ---")
for name, fa in file_analyses.items():
    ml = fa["nav"].get("ml_speed")
    if ml and ml.get("count", 0) > 0:
        print(f"  {name}: TCN mean={ml['mean']:.3f}, max={ml['max']:.3f}, "
              f"min={ml['min']:.3f} m/s")

# GNSS state machine summary
for name, sm in state_machine_report.items():
    print(f"  {name}: {sm['n_outages']} outage(s), stale_gnss={sm['stale_gnss']}")

# Save timing and state machine reports
with open(OUT / "timing_report.json", "w") as fp:
    json.dump(timing_report, fp, indent=2)
with open(OUT / "state_machine_report.json", "w") as fp:
    json.dump(state_machine_report, fp, indent=2, default=str)
with open(OUT / "chronology.json", "w") as fp:
    json.dump(chrono, fp, indent=2)

print(f"\nAll results saved to {OUT}")
print("Phase 5.1 forensic analysis complete.")
