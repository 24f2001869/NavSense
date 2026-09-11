"""
SIH26168 - Step 3: Coordinate Systems & Geodetic Transformations
Utilities for Euler angles, Direction Cosine Matrices (DCM),
and WGS84 Geodetic to local East-North-Up (ENU) Cartesian frame conversion.
"""

import numpy as np

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0         # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2 # Square of eccentricity

def euler_to_rot_matrix(roll, pitch, yaw):
    """
    Computes 3x3 rotation matrix for standard Z-Y-X (yaw-pitch-roll) sequence.
    Angles in radians.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr]
    ])

    R_y = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    R_z = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return R_z @ R_y @ R_x

def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Converts WGS84 latitude, longitude, altitude to Earth-Centered Earth-Fixed (ECEF) coords."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
    x = (N + alt_m) * np.cos(phi) * np.cos(lam)
    y = (N + alt_m) * np.cos(phi) * np.sin(lam)
    z = (N * (1.0 - WGS84_E2) + alt_m) * np.sin(phi)
    return x, y, z

def geodetic_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg, alt_m=0.0, alt0_m=0.0):
    """
    Converts geodetic coordinates to local East-North-Up (ENU) Cartesian meters
    relative to a reference anchor (lat0, lon0, alt0).
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    phi0 = np.radians(lat0_deg)
    lam0 = np.radians(lon0_deg)

    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_lam = np.sin(lam0)
    cos_lam = np.cos(lam0)

    east = -sin_lam * dx + cos_lam * dy
    north = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    up = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return east, north, up

def enu_to_geodetic(east, north, lat0_deg, lon0_deg, up=0.0):
    """
    Approximate local ENU to geodetic conversion for small to medium scale navigation.
    """
    phi0 = np.radians(lat0_deg)
    # Radii of curvature
    R_m = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * np.sin(phi0) ** 2) ** 1.5
    R_p = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(phi0) ** 2)

    d_lat = north / (R_m + up)
    d_lon = east / ((R_p + up) * np.cos(phi0))

    lat = lat0_deg + np.degrees(d_lat)
    lon = lon0_deg + np.degrees(d_lon)
    return lat, lon
