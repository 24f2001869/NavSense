"""
SIH26168 - Stage C8-1: Offline Map Information Utility & Observability Formulation
Script: src/map/information.py

Mathematical Formulation:
- Formulates decoupled measurement models:
  1. Measurement A (Lateral Cross-track):
     r_perp = d_perp(p) - mu_lane
     H_perp = [n_E, n_N, 0, 0_1x3, 0_1x3, 0_1x3, 0_1x3]  (1x15)
  2. Measurement B (Road Tangent Heading):
     r_psi = wrap(psi - psi_road)
     H_psi = [0_1x3, 0_1x3, 0, 0, 1, 0_1x3, 0_1x3]       (1x15)
  3. Joint Measurement C:
     H = [H_perp; H_psi]                                  (2x15)
     R = [sigma_lane^2,  rho * sigma_lane * sigma_psi;
          rho * sigma_lane * sigma_psi,  sigma_psi^2]     (2x2)

- Information-Theoretic Functions:
  - Fisher Information Matrix: I_map = H^T * R^-1 * H (15x15)
  - Observable subspace eigenvalues, rank, and condition number.
  - Analytical covariance reduction:
    Delta_P = P^- - ( (P^-)^-1 + H^T * R^-1 * H )^-1
  - State projection into along-track and cross-track coordinate frames.
  - Normalized Innovation Squared (NIS):
    NIS = r^T * (H * P^- * H^T + R)^-1 * r
"""

import numpy as np


def compute_map_jacobians(road_heading_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes measurement Jacobians H_perp (1x15), H_psi (1x15), and joint H (2x15)
    for a 15-state ESKF error state:
      delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]^T
    In local ENU frame:
      - East is index 0, North is index 1, Up is index 2.
      - Heading (yaw from North clockwise) is attitude index 8 (index 2 of delta_theta).
    """
    # Unit normal pointing to the right of the road:
    # If tangent t = [sin(psi), cos(psi)], then normal n = [cos(psi), -sin(psi)]
    n_E = np.cos(road_heading_rad)
    n_N = -np.sin(road_heading_rad)

    H_perp = np.zeros((1, 15))
    H_perp[0, 0] = n_E
    H_perp[0, 1] = n_N

    H_psi = np.zeros((1, 15))
    H_psi[0, 8] = 1.0  # Yaw error in attitude block (index 6, 7, 8 -> roll, pitch, yaw)

    H_joint = np.vstack([H_perp, H_psi])  # (2, 15)

    return H_perp, H_psi, H_joint


def compute_map_covariance(sigma_lane: float = 2.5, sigma_psi_rad: float = np.radians(5.0),
                           curvature_rad_m: float = 0.0) -> np.ndarray:
    """
    Computes measurement covariance R (2x2) with optional curvature-dependent cross-coupling rho(kappa).
    On straight roads (curvature ~ 0), rho = 0. On curved roads, position and heading geometry couple.
    """
    rho = float(np.tanh(curvature_rad_m * 20.0))  # Smoothly bound cross-coupling in (-0.5, +0.5)
    rho = np.clip(rho, -0.5, 0.5)

    var_lane = sigma_lane ** 2
    var_psi = sigma_psi_rad ** 2
    cov_cross = rho * sigma_lane * sigma_psi_rad

    R = np.array([
        [var_lane, cov_cross],
        [cov_cross, var_psi]
    ])
    return R


def compute_fisher_information(H: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, int, float, list[float]]:
    """
    Computes the Fisher Information Matrix I_map = H^T * R^-1 * H (15x15).
    Returns information matrix, rank, condition number of observable subspace, and non-zero eigenvalues.
    """
    R_inv = np.linalg.inv(R)
    I_map = H.T @ R_inv @ H  # (15, 15)

    # Spectral analysis
    eigvals = np.linalg.eigvalsh(I_map)
    tol = 1e-10 * np.max(eigvals)
    non_zero_eigs = [float(ev) for ev in eigvals if ev > tol]
    rank = len(non_zero_eigs)

    cond_num = float(non_zero_eigs[-1] / (non_zero_eigs[0] + 1e-12)) if rank > 0 else 0.0

    return I_map, rank, cond_num, non_zero_eigs


def compute_covariance_reduction(P_prior: np.ndarray, H: np.ndarray, R: np.ndarray,
                                 road_heading_rad: float) -> dict:
    """
    Computes analytical state covariance update:
      P^+ = (I - K*H) * P^- * (I - K*H)^T + K * R * K^T  (Joseph form)
    and disaggregates variance reduction along the vehicle track:
      - Cross-track horizontal variance (perpendicular to road)
      - Along-track horizontal variance (parallel to road)
      - Heading (yaw) variance
    """
    S = H @ P_prior @ H.T + R  # (M, M)
    K = P_prior @ H.T @ np.linalg.inv(S)  # (15, M)

    I_15 = np.eye(15)
    IKH = I_15 - K @ H
    P_post = IKH @ P_prior @ IKH.T + K @ R @ K.T  # (15, 15)

    # Unit vectors parallel and perpendicular to road:
    # Tangent t = [sin(psi), cos(psi)], Normal n = [cos(psi), -sin(psi)]
    t_vec = np.array([np.sin(road_heading_rad), np.cos(road_heading_rad)])
    n_vec = np.array([np.cos(road_heading_rad), -np.sin(road_heading_rad)])

    # Horizontal position covariance 2x2
    P_pos_prior = P_prior[0:2, 0:2]
    P_pos_post = P_post[0:2, 0:2]

    var_along_prior = float(t_vec.T @ P_pos_prior @ t_vec)
    var_along_post = float(t_vec.T @ P_pos_post @ t_vec)

    var_cross_prior = float(n_vec.T @ P_pos_prior @ n_vec)
    var_cross_post = float(n_vec.T @ P_pos_post @ n_vec)

    var_yaw_prior = float(P_prior[8, 8])
    var_yaw_post = float(P_post[8, 8])

    return {
        'P_post': P_post,
        'K_gain': K,
        'var_cross_prior': var_cross_prior,
        'var_cross_post': var_cross_post,
        'var_cross_reduction_pct': float((var_cross_prior - var_cross_post) / (var_cross_prior + 1e-12) * 100.0),
        'var_along_prior': var_along_prior,
        'var_along_post': var_along_post,
        'var_along_reduction_pct': float((var_along_prior - var_along_post) / (var_along_prior + 1e-12) * 100.0),
        'var_yaw_prior_deg2': float(np.degrees(np.degrees(var_yaw_prior))),
        'var_yaw_post_deg2': float(np.degrees(np.degrees(var_yaw_post))),
        'var_yaw_reduction_pct': float((var_yaw_prior - var_yaw_post) / (var_yaw_prior + 1e-12) * 100.0)
    }


def compute_nis(residual: np.ndarray, H: np.ndarray, P_prior: np.ndarray, R: np.ndarray) -> float:
    """Computes Normalized Innovation Squared (NIS): r^T * (H * P^- * H^T + R)^-1 * r."""
    r = np.asarray(residual).flatten()
    S = H @ P_prior @ H.T + R
    S_inv = np.linalg.inv(S)
    nis = float(r.T @ S_inv @ r)
    return nis
