import pandas as pd
import numpy as np

df = pd.read_csv('files/idr_telemetry_20260909_091947.csv')
t0 = df['timestamp_ms'].iloc[0]
df['time_s'] = (df['timestamp_ms'] - t0) / 1000.0

print(f"Total Epochs: {len(df)}")
print(f"Total Duration: {df['time_s'].iloc[-1]:.2f} s")

# Detect transitions
df['state_change'] = df['engine_state'] != df['engine_state'].shift(1)
transitions = df[df['state_change']]

print("\n--- STATE MACHINE TRANSITIONS ---")
for idx, row in transitions.iterrows():
    print(f"Epoch {idx:3d} (t = {row['time_s']:6.2f}s): State = {row['engine_state']:16s} | pos = [{row['pos_e']:6.2f}, {row['pos_n']:6.2f}, {row['pos_u']:6.2f}] | heading = {row['heading_deg']:6.2f} deg")

# Inspect outage details
outage_epochs = df[df['engine_state'] == 'DEAD_RECKONING']
if len(outage_epochs) > 0:
    t_start = outage_epochs['time_s'].iloc[0]
    t_end = outage_epochs['time_s'].iloc[-1]
    dur = t_end - t_start
    print(f"\n--- OUTAGE DURATION ---")
    print(f"Outage Start: t = {t_start:.2f} s (Epoch {outage_epochs.index[0]})")
    print(f"Outage End:   t = {t_end:.2f} s (Epoch {outage_epochs.index[-1]})")
    print(f"Outage Duration: {dur:.2f} s ({len(outage_epochs)} epochs)")
    
    # Drift during outage
    p0 = outage_epochs[['pos_e', 'pos_n', 'pos_u']].iloc[0].values
    p1 = outage_epochs[['pos_e', 'pos_n', 'pos_u']].iloc[-1].values
    drift = np.linalg.norm(p1 - p0)
    print(f"Accumulated dead-reckoning displacement: {drift:.2f} m")
