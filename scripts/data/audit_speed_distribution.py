#!/usr/bin/env python3
"""
Speed Distribution Audit across Train, Validation, and Test Splits
Script: scratch/audit_speed_distribution.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset_builder import IOVNBDDatasetBuilder

HELD_OUT_TEST_TRIP_NAMES = [
    'Vta21', 'Vta22', 'Vta23', 'Vta24', 'Vta25', 'Vta26', 'Vta27', 'Vta28',
    'Vtb09', 'Vtb10', 'Vtb11', 'Vtb12',
    'Vw12', 'Vw13', 'Vw14a', 'Vw14b', 'Vw15', 'Vw16a',
    'V-Vfa02'
]

VAL_TRIP_NAMES = [
    'Vta19', 'Vta20',
    'Vtb08',
    'Vw10', 'Vw11',
    'V-Vfa01'
]

BINS = [
    (0.0, 1.0, "0–1 m/s (0–3.6 km/h)"),
    (1.0, 3.0, "1–3 m/s (3.6–10.8 km/h)"),
    (3.0, 5.0, "3–5 m/s (10.8–18 km/h)"),
    (5.0, 10.0, "5–10 m/s (18–36 km/h)"),
    (10.0, 15.0, "10–15 m/s (36–54 km/h)"),
    (15.0, 20.0, "15–20 m/s (54–72 km/h)"),
    (20.0, 25.0, "20–25 m/s (72–90 km/h)"),
    (25.0, 30.0, "25–30 m/s (90–108 km/h)"),
    (30.0, 100.0, "30+ m/s (108+ km/h)"),
]

def load_split_speeds(builder, trip_list):
    all_speeds = []
    trip_stats = []
    for t in trip_list:
        df = builder.load_clean_trip(t)
        v = df['can_speed_mps'].values.astype(np.float32)
        all_speeds.append(v)
        trip_stats.append({
            'trip_name': t['trip_name'],
            'driver': t['driver'],
            'samples': len(v),
            'min_mps': float(np.min(v)),
            'max_mps': float(np.max(v)),
            'mean_mps': float(np.mean(v)),
            'p50_mps': float(np.median(v)),
            'p95_mps': float(np.percentile(v, 95)),
        })
    return np.concatenate(all_speeds), trip_stats

def compute_bin_distribution(speeds):
    total = len(speeds)
    results = []
    for low, high, label in BINS:
        mask = (speeds >= low) & (speeds < high) if high < 100.0 else (speeds >= low)
        count = int(np.sum(mask))
        pct = (count / total) * 100.0 if total > 0 else 0.0
        results.append({
            'bin_label': label,
            'low': low,
            'high': high,
            'count': count,
            'pct': pct
        })
    return results

def main():
    print("=" * 80)
    print("AUDITING TRUE-SPEED DISTRIBUTION ACROSS TRAIN, VAL, TEST")
    print("=" * 80)

    builder = IOVNBDDatasetBuilder([
        Path('data/raw/IO-VNBD-repo/Synchronised V abd S datasets/Categorised IOVNB Dataset'),
    ])
    all_trips = builder.discover_trips()
    car_trips = [t for t in all_trips if t['driver'] in ['Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Vf (Driver E)']]

    train_trips = [t for t in car_trips if t['trip_name'] not in HELD_OUT_TEST_TRIP_NAMES and t['trip_name'] not in VAL_TRIP_NAMES]
    val_trips = [t for t in car_trips if t['trip_name'] in VAL_TRIP_NAMES]
    test_trips = [t for t in car_trips if t['trip_name'] in HELD_OUT_TEST_TRIP_NAMES]

    print(f"Trips count -> Train: {len(train_trips)}, Val: {len(val_trips)}, Test: {len(test_trips)}")

    train_speeds, train_trip_stats = load_split_speeds(builder, train_trips)
    val_speeds, val_trip_stats = load_split_speeds(builder, val_trips)
    test_speeds, test_trip_stats = load_split_speeds(builder, test_trips)

    print(f"Sample count -> Train: {len(train_speeds)}, Val: {len(val_speeds)}, Test: {len(test_speeds)}")

    train_dist = compute_bin_distribution(train_speeds)
    val_dist = compute_bin_distribution(val_speeds)
    test_dist = compute_bin_distribution(test_speeds)

    # Print comparative table
    print("\n" + "=" * 100)
    print(f"{'Speed Bin':30s} | {'Train Samples':>13s} {'Train %':>8s} | {'Val Samples':>11s} {'Val %':>8s} | {'Test Samples':>12s} {'Test %':>8s}")
    print("-" * 100)
    
    summary_rows = []
    for i in range(len(BINS)):
        b_label = BINS[i][2]
        tr_c, tr_p = train_dist[i]['count'], train_dist[i]['pct']
        va_c, va_p = val_dist[i]['count'], val_dist[i]['pct']
        te_c, te_p = test_dist[i]['count'], test_dist[i]['pct']
        print(f"{b_label:30s} | {tr_c:13,d} {tr_p:7.2f}% | {va_c:11,d} {va_p:7.2f}% | {te_c:12,d} {te_p:7.2f}%")
        summary_rows.append({
            'speed_bin': b_label,
            'low_mps': BINS[i][0],
            'high_mps': BINS[i][1],
            'train_count': tr_c,
            'train_pct': round(tr_p, 2),
            'val_count': va_c,
            'val_pct': round(va_p, 2),
            'test_count': te_c,
            'test_pct': round(te_p, 2),
        })

    print("=" * 100)
    print(f"{'TOTAL':30s} | {len(train_speeds):13,d} {100.00:7.2f}% | {len(val_speeds):11,d} {100.00:7.2f}% | {len(test_speeds):12,d} {100.00:7.2f}%")
    print("=" * 100)

    # Calculate aggregate summary stats per split
    def get_split_summary(speeds, name):
        return {
            'split': name,
            'total_samples': len(speeds),
            'duration_hours': round(len(speeds) * 0.1 / 3600, 2),
            'mean_mps': round(float(np.mean(speeds)), 2),
            'std_mps': round(float(np.std(speeds)), 2),
            'median_mps': round(float(np.median(speeds)), 2),
            'p05_mps': round(float(np.percentile(speeds, 5)), 2),
            'p95_mps': round(float(np.percentile(speeds, 95)), 2),
            'p99_mps': round(float(np.percentile(speeds, 99)), 2),
            'max_mps': round(float(np.max(speeds)), 2),
            'standstill_pct_under_1mps': round(float(np.mean(speeds < 1.0) * 100.0), 2),
            'low_speed_pct_under_5mps': round(float(np.mean(speeds < 5.0) * 100.0), 2),
            'mid_speed_pct_5_to_20mps': round(float(np.mean((speeds >= 5.0) & (speeds < 20.0)) * 100.0), 2),
            'high_speed_pct_20_to_25mps': round(float(np.mean((speeds >= 20.0) & (speeds < 25.0)) * 100.0), 2),
            'extreme_speed_pct_over_25mps': round(float(np.mean(speeds >= 25.0) * 100.0), 2),
        }

    stats = {
        'train': get_split_summary(train_speeds, 'Train (39 trips)'),
        'val': get_split_summary(val_speeds, 'Val (6 trips)'),
        'test': get_split_summary(test_speeds, 'Test (19 trips)'),
        'bins': summary_rows
    }

    out_file = PROJECT_ROOT / "results" / "expanded_tcn_benchmark" / "speed_distribution_audit.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved distribution audit to {out_file}")

if __name__ == '__main__':
    main()
