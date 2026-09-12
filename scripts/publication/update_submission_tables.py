"""
update_submission_tables.py
---------------------------
Regenerates Table 1 to Table 7 in clean CSV format for The Journal of Navigation submission,
fully grounded in authoritative frozen evaluation ledgers and verified dataset statistics.
"""

import csv
from pathlib import Path

TABLES_DIR = Path('publication/journal_of_navigation_submission/tables')
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def write_table_01():
    with open(TABLES_DIR / 'Table_01.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Environment / Category", "Trips (N)", "Duration (h)", "Synchronized Epochs", "Distance (km)", "Mean Speed (km/h)", "Notes & Operational Profile"])
        w.writerow(["Suburban Town (Vta)", "30", "3.57", "128,618", "147.9", "41.4", "Frequent intersections, traffic signals, stop-and-go"])
        w.writerow(["Dense Urban (Vtb)", "12", "3.18", "114,437", "163.6", "51.5", "Multi-lane arterials, urban street canyons, variable speeds"])
        w.writerow(["Winding Mountain / Rural (Vw)", "20", "7.32", "263,579", "409.0", "55.9", "Continuous curvature, elevation changes, unbanked turns"])
        w.writerow(["High-Speed Motorway (V-Vfa)", "2", "2.19", "79,009", "181.9", "82.7", "Sustained high-speed cruising (80–118 km/h), minimal turns"])
        w.writerow(["Complete Passenger-Car Corpus", "64", "16.27", "585,643", "902.4", "55.5", "Synchronized 10 Hz telemetry; Driver E; Ford Fiesta test vehicle"])
    print("Updated Table_01.csv")

def write_table_02():
    with open(TABLES_DIR / 'Table_02.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Data Split", "Trip Count (N)", "Synchronized Epochs", "Duration (h)", "Distance (km)", "Driver Identity", "Partitioning Protocol"])
        w.writerow(["Training Split", "39", "448,993", "12.47", "634.5", "Driver E", "Whole-trip disjoint; zero temporal or route overlap"])
        w.writerow(["Validation Split", "6", "21,230", "0.59", "27.2", "Driver E", "Model selection, checkpoint early stopping, loss tuning"])
        w.writerow(["Held-Out Test Split", "19", "115,420", "3.21", "240.6", "Driver E", "Strictly unseen test routes; multi-horizon evaluation"])
        w.writerow(["Total Synchronized Corpus", "64", "585,643", "16.27", "902.4", "Driver E", "Usable sliding sequence samples: 579,307 (100-sample buffer)"])
    print("Updated Table_02.csv")

def write_table_03():
    with open(TABLES_DIR / 'Table_03.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Outage Horizon", "Navigation Architecture", "Evaluated Episodes (N)", "Mean Drift (m)", "Macro Distance Drift (%)", "SIH Pass Count (<10%)", "Route Pass Rate (%)"])
        w.writerow(["10 s Outage", "Pure Strapdown Kinematics", "18", "23.66", "83.54", "6 / 18", "33.3%"])
        w.writerow(["10 s Outage", "Learned Static TCN", "18", "26.83", "106.44", "5 / 18", "27.8%"])
        w.writerow(["10 s Outage", "Fixed Damped Momentum", "18", "23.49", "111.37", "6 / 18", "33.3%"])
        w.writerow(["10 s Outage", "Adaptive Regime Fusion (Proposed)", "18", "25.06", "111.53", "5 / 18", "27.8%"])
        w.writerow(["20 s Outage", "Pure Strapdown Kinematics", "17", "57.01", "147.98", "4 / 17", "23.5%"])
        w.writerow(["20 s Outage", "Learned Static TCN", "17", "45.73", "123.95", "5 / 17", "29.4%"])
        w.writerow(["20 s Outage", "Fixed Damped Momentum", "17", "65.81", "193.51", "4 / 17", "23.5%"])
        w.writerow(["20 s Outage", "Adaptive Regime Fusion (Proposed)", "17", "49.27", "135.51", "4 / 17", "23.5%"])
        w.writerow(["30 s Outage", "Pure Strapdown Kinematics", "16", "109.57", "118.22", "2 / 16", "12.5%"])
        w.writerow(["30 s Outage", "Learned Static TCN", "16", "54.62", "65.57", "5 / 16", "31.2%"])
        w.writerow(["30 s Outage", "Fixed Damped Momentum", "16", "98.07", "127.13", "2 / 16", "12.5%"])
        w.writerow(["30 s Outage", "Adaptive Regime Fusion (Proposed)", "16", "61.83", "71.38", "4 / 16", "25.0%"])
        w.writerow(["60 s Outage", "Pure Strapdown Kinematics", "13", "324.21", "89.66", "0 / 13", "0.0%"])
        w.writerow(["60 s Outage", "Learned Static TCN", "13", "95.53", "18.26", "3 / 13", "23.1%"])
        w.writerow(["60 s Outage", "Fixed Damped Momentum", "13", "173.71", "40.75", "1 / 13", "7.7%"])
        w.writerow(["60 s Outage", "Adaptive Regime Fusion (Proposed)", "13", "96.65", "16.76", "3 / 13", "23.1%"])
    print("Updated Table_03.csv")

def write_table_04():
    with open(TABLES_DIR / 'Table_04.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Evaluation Metric", "Current TCN (6 Training Trips)", "Expanded TCN (39 Training Trips)", "Relative Performance Delta (%)"])
        w.writerow(["Training Scope", "6 trips (1.1 h, 40,890 epochs)", "39 trips (12.5 h, 448,993 epochs)", "+550% training route exposure"])
        w.writerow(["Velocity MAE (m/s)", "6.17 m/s", "2.76 m/s", "-55.27% error reduction"])
        w.writerow(["Velocity RMSE (m/s)", "6.96 m/s", "3.59 m/s", "-48.42% error reduction"])
        w.writerow(["Signed Bias (m/s)", "-3.06 m/s", "-0.41 m/s", "-86.60% bias reduction"])
        w.writerow(["30-Second Dead-Reckoning Drift (m)", "153.31 m", "52.64 m", "-65.67% drift reduction"])
        w.writerow(["60-Second Dead-Reckoning Drift (m)", "263.55 m", "92.96 m", "-64.72% drift reduction"])
    print("Updated Table_04.csv")

def write_table_05():
    with open(TABLES_DIR / 'Table_05.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Configuration Variant", "Forward Speed", "Lateral Constraint", "Attitude Feedback", "Mean 60-s Drift (m)", "Relative Drift vs. Baseline (%)", "Mean NIS_x", "Filter State"])
        w.writerow(["Variant A (Pure-INS Baseline)", "None", "None", "None", "2488.7", "295.1%", "0.00", "Divergent"])
        w.writerow(["Variant D (Coupled 3D NHC alone)", "None", "Coupled (v_y, v_z)", "Full", "1166.7", "138.3%", "0.00", "Marginal"])
        w.writerow(["Variant E (Learned TCN Alone)", "TCN v_x", "None", "None", "843.4", "100.0% (Ref)", "6.55", "Stable"])
        w.writerow(["Variant F (TCN + Unconditional Lateral NHC)", "TCN v_x", "Coupled (v_y = 0)", "Full", "1226.9", "145.5%", "82.12", "Degraded"])
        w.writerow(["Variant H (TCN + Full Coupled 3D NHC)", "TCN v_x", "Coupled (v_y, v_z = 0)", "Full", "1240.8", "147.1%", "87.85", "Degraded"])
        w.writerow(["Variant V1 (Proposed Decoupled Damping)", "TCN v_x", "Decoupled Damping", "None (K_y[6:15]=0)", "382.7", "45.4%", "6.60", "Stable"])
    print("Updated Table_05.csv")

def write_table_06():
    with open(TABLES_DIR / 'Table_06.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Variant ID", "Map Matching Mode", "Heading Feedback", "Envelope Gate", "Mean 60-s Drift (m)", "Heading Error (deg)", "Delta vs. M0 (%)", "Regression Rate (%)"])
        w.writerow(["M0 (Baseline)", "Open-Loop / None", "Disabled", "No", "578.3", "63.4°", "+0.0%", "0.0%"])
        w.writerow(["M1 (Shadow Map)", "Shadow 5-Gate MHT", "Disabled", "No", "578.3", "63.4°", "+0.0%", "0.0%"])
        w.writerow(["M2 (Closed-Loop Heading)", "Heading Constraint", "Enabled (Direct)", "No", "1303.9", "73.5°", "-125.5%", "47.0%"])
        w.writerow(["M3 (Lateral Map)", "Lateral Constraint", "Disabled", "No", "597.7", "65.5°", "-3.3%", "28.9%"])
        w.writerow(["M4 (Joint Map)", "Joint Heading & Lateral", "Enabled", "No", "829.8", "68.2°", "-43.5%", "33.7%"])
        w.writerow(["M5 (Envelope-Gated)", "Joint Heading & Lateral", "Enabled", "Yes", "807.9", "67.5°", "-39.7%", "22.9%"])
    print("Updated Table_06.csv")

def write_table_07():
    with open(TABLES_DIR / 'Table_07.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Road Category", "Trips (N)", "Mean Ref Distance (m)", "Pure Strapdown Drift (m)", "Pure Macro Drift (%)", "Adaptive Fusion Drift (m)", "Adaptive Macro Drift (%)", "SIH Pass Count (<10%)"])
        w.writerow(["Suburban Town (Vta)", "7", "575.1", "301.6", "127.4%", "95.1", "21.6%", "1 / 7 (14.3%)"])
        w.writerow(["Winding Mountain (Vw)", "4", "1293.9", "446.1", "39.7%", "108.2", "9.7%", "2 / 4 (50.0%)"])
        w.writerow(["High-Speed Motorway (V-Vfa)", "1", "1460.3", "313.9", "25.4%", "157.2", "11.7%", "0 / 1 (0.0%)"])
        w.writerow(["Stationary Control (Vw15)", "1", "1.3", "5.6", "450.5%", "0.82", "63.8%", "0 / 1 (0.0%)"])
        w.writerow(["Overall Benchmark", "13", "820.2", "324.21", "89.66%", "96.65", "16.76%", "3 / 13 (23.08%)"])
    print("Updated Table_07.csv")

if __name__ == '__main__':
    write_table_01()
    write_table_02()
    write_table_03()
    write_table_04()
    write_table_05()
    write_table_06()
    write_table_07()
