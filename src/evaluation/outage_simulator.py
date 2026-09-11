"""
SIH26168 - Step 4: Artificial GNSS Outage Simulator
Simulates GPS/GNSS signal blackouts (e.g. tunnels, dense urban canyons, spoofing/jamming)
by masking GNSS position and velocity observations during specified intervals.
"""

import numpy as np
import pandas as pd

class GNSSOutageSimulator:
    def __init__(self, outage_windows=None):
        """
        outage_windows: list of tuples (start_sec, duration_sec),
                        e.g. [(100.0, 30.0), (300.0, 60.0)]
        """
        self.outage_windows = outage_windows or []

    def add_outage(self, start_s, duration_s):
        self.outage_windows.append((float(start_s), float(duration_s)))
        return self

    def is_outage(self, current_time_s):
        """Checks whether the given timestamp falls inside any defined outage window."""
        for start_s, dur_s in self.outage_windows:
            if start_s <= current_time_s < (start_s + dur_s):
                return True
        return False

    def generate_outage_mask(self, timestamps):
        """
        Returns a boolean array where True indicates GNSS is AVAILABLE,
        and False indicates GNSS is in OUTAGE.
        """
        times = np.asarray(timestamps)
        available = np.ones(len(times), dtype=bool)
        for start_s, dur_s in self.outage_windows:
            in_outage = (times >= start_s) & (times < start_s + dur_s)
            available[in_outage] = False
        return available

    def apply_to_dataframe(self, df, time_col='time_s'):
        """
        Creates a copy of df where GNSS fields during outages are set to NaN.
        """
        df_masked = df.copy()
        mask = self.generate_outage_mask(df_masked[time_col].values)

        gnss_cols = [c for c in df_masked.columns if any(k in c.lower() for k in ['lat', 'lon', 'alt', 'speed', 'bearing', 'heading'])]
        for col in gnss_cols:
            if col in df_masked.columns:
                df_masked.loc[~mask, col] = np.nan

        df_masked['gnss_available'] = mask
        return df_masked

if __name__ == "__main__":
    t = np.arange(0, 100, 0.1)
    sim = GNSSOutageSimulator().add_outage(20.0, 30.0).add_outage(70.0, 10.0)
    mask = sim.generate_outage_mask(t)
    print(f"Total points: {len(t)}, Available: {np.sum(mask)}, Outage: {np.sum(~mask)}")
