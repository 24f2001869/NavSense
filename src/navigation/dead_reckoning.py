"""
SIH26168 - Step 4: Pure IMU Dead Reckoning (Inertial Navigation Baseline)
Demonstrates the fundamental classical limitation of low-cost MEMS IMUs:
Double numerical integration of unconstrained acceleration causes quadratic
and cubic position drift over time.
"""

import numpy as np

def run_pure_imu_dead_reckoning(accel_veh, gyro_veh, dt=0.1, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0), init_yaw_rad=0.0):
    """
    Performs pure mechanization double-integration of IMU channels:
      1. Integrate gyro Z (yaw rate) to obtain vehicle heading psi(t).
      2. Project horizontal body acceleration (longitudinal, lateral) into ENU navigation frame.
      3. Trapezoidal integration to velocity (v_east, v_north).
      4. Trapezoidal integration to position (east, north).

    accel_veh: (N, 3) [a_forward, a_lateral, a_vertical]
    gyro_veh:  (N, 3) [omega_pitch, omega_roll, omega_yaw]
    """
    n = len(accel_veh)
    east = np.zeros(n)
    north = np.zeros(n)
    vel_e = np.zeros(n)
    vel_n = np.zeros(n)
    yaw = np.zeros(n)

    east[0], north[0] = init_pos
    vel_e[0], vel_n[0] = init_vel
    yaw[0] = init_yaw_rad

    # Body accelerations (remove mean vertical to eliminate gravity residual)
    a_fwd = accel_veh[:, 0]
    a_lat = accel_veh[:, 1]
    omega_z = gyro_veh[:, 2]

    for k in range(1, n):
        # 1. Heading integration
        yaw[k] = yaw[k - 1] + omega_z[k] * dt

        # 2. Body acceleration into ENU frame
        # Convention: Heading 0 rad = North, pi/2 = East (clockwise navigation azimuth)
        # East accel = a_fwd * sin(yaw) + a_lat * cos(yaw)
        # North accel = a_fwd * cos(yaw) - a_lat * sin(yaw)
        acc_e = a_fwd[k] * np.sin(yaw[k]) + a_lat[k] * np.cos(yaw[k])
        acc_n = a_fwd[k] * np.cos(yaw[k]) - a_lat[k] * np.sin(yaw[k])

        # 3. Trapezoidal integration of velocity
        vel_e[k] = vel_e[k - 1] + 0.5 * (acc_e + (a_fwd[k-1]*np.sin(yaw[k-1]) + a_lat[k-1]*np.cos(yaw[k-1]))) * dt
        vel_n[k] = vel_n[k - 1] + 0.5 * (acc_n + (a_fwd[k-1]*np.cos(yaw[k-1]) - a_lat[k-1]*np.sin(yaw[k-1]))) * dt

        # 4. Position integration
        east[k] = east[k - 1] + 0.5 * (vel_e[k] + vel_e[k - 1]) * dt
        north[k] = north[k - 1] + 0.5 * (vel_n[k] + vel_n[k - 1]) * dt

    return east, north, vel_e, vel_n, yaw

def run_heading_velocity_dead_reckoning(forward_speed, heading_rad, dt=0.1, init_pos=(0.0, 0.0)):
    """
    Dead reckoning using measured/predicted forward speed and heading (Hodometer DR).
    p_k = p_{k-1} + v_k * [sin(psi), cos(psi)] * dt
    """
    n = len(forward_speed)
    east = np.zeros(n)
    north = np.zeros(n)
    east[0], north[0] = init_pos

    for k in range(1, n):
        v = forward_speed[k]
        psi = heading_rad[k]
        east[k] = east[k - 1] + v * np.sin(psi) * dt
        north[k] = north[k - 1] + v * np.cos(psi) * dt

    return east, north
