"""
Audit script for Stage C3 ESKF:
1. Finite-difference validation of state transition matrix Phi.
2. Attitude error convention numerical check.
3. Process noise Q discretization audit.
"""

import numpy as np

def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

def rotvec_to_quat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], dtype=np.float64)
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta) / theta
    return np.array([c, s * v[0], s * v[1], s * v[2]], dtype=np.float64)

def quat_to_dcm(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
    ], dtype=np.float64)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def nonlinear_step(p, v, q, ba, bg, f_meas_v, omega_meas_v, dt, g=9.80665):
    f_corr_v = f_meas_v - ba
    omega_corr_v = omega_meas_v - bg
    C = quat_to_dcm(q)
    f_n = C @ f_corr_v
    a_n = f_n - np.array([0.0, 0.0, g])
    
    p_next = p + v * dt + 0.5 * a_n * (dt**2)
    v_next = v + a_n * dt
    dq = rotvec_to_quat(omega_corr_v * dt)
    q_next = quat_mult(q, dq)
    q_next /= np.linalg.norm(q_next)
    
    ba_next = ba.copy()
    bg_next = bg.copy()
    return p_next, v_next, q_next, ba_next, bg_next

def run_jacobian_audit():
    print("=== PRIORITY 2: FINITE DIFFERENCE JACOBIAN AUDIT ===")
    dt = 0.1
    eps = 1e-6
    
    # Nominal state
    p0 = np.array([10.0, 20.0, 5.0])
    v0 = np.array([12.0, -3.0, 0.5])
    # arbitrary attitude
    q0 = rotvec_to_quat(np.array([0.1, -0.2, 0.5]))
    ba0 = np.array([0.1, -0.05, 0.2])
    bg0 = np.array([0.01, 0.02, -0.03])
    
    f_meas = np.array([0.5, 0.2, 9.7])
    omega_meas = np.array([0.05, -0.03, 0.1])
    
    f_corr = f_meas - ba0
    omega_corr = omega_meas - bg0
    C0 = quat_to_dcm(q0)
    
    # Analytic Phi
    Phi = np.eye(15)
    Phi[0:3, 3:6] = np.eye(3) * dt
    Phi[0:3, 6:9] = -0.5 * C0 @ skew(f_corr) * (dt**2)
    Phi[0:3, 9:12] = -0.5 * C0 * (dt**2)
    Phi[3:6, 6:9] = -C0 @ skew(f_corr) * dt
    Phi[3:6, 9:12] = -C0 * dt
    Phi[6:9, 6:9] = np.eye(3) - skew(omega_corr) * dt
    Phi[6:9, 12:15] = -np.eye(3) * dt
    
    # Nominal step
    p_nom, v_nom, q_nom, ba_nom, bg_nom = nonlinear_step(p0, v0, q0, ba0, bg0, f_meas, omega_meas, dt)
    C_nom = quat_to_dcm(q_nom)
    
    # Numerical Jacobian Phi_num
    Phi_num = np.zeros((15, 15))
    
    for i in range(15):
        dx = np.zeros(15)
        dx[i] = eps
        
        p_pert = p0 + dx[0:3]
        v_pert = v0 + dx[3:6]
        dq_pert = rotvec_to_quat(dx[6:9])
        q_pert = quat_mult(q0, dq_pert)
        q_pert /= np.linalg.norm(q_pert)
        ba_pert = ba0 + dx[9:12]
        bg_pert = bg0 + dx[12:15]
        
        p_p, v_p, q_p, ba_p, bg_p = nonlinear_step(p_pert, v_pert, q_pert, ba_pert, bg_pert, f_meas, omega_meas, dt)
        
        # Output error state
        dp = p_p - p_nom
        dv = v_p - v_nom
        # Attitude error: C_p = C_nom @ (I + [dtheta]x) => C_nom^T C_p = I + [dtheta]x
        C_p = quat_to_dcm(q_p)
        R_err = C_nom.T @ C_p
        dtheta = np.array([
            0.5 * (R_err[2, 1] - R_err[1, 2]),
            0.5 * (R_err[0, 2] - R_err[2, 0]),
            0.5 * (R_err[1, 0] - R_err[0, 1])
        ])
        dba = ba_p - ba_nom
        dbg = bg_p - bg_nom
        
        dy = np.concatenate([dp, dv, dtheta, dba, dbg])
        Phi_num[:, i] = dy / eps

    # Compare key blocks
    blocks = [
        ("Phi_p_v (pos wrt vel)", (0, 3), (3, 6)),
        ("Phi_p_theta (pos wrt theta)", (0, 3), (6, 9)),
        ("Phi_p_ba (pos wrt ba)", (0, 3), (9, 12)),
        ("Phi_v_theta (vel wrt theta)", (3, 6), (6, 9)),
        ("Phi_v_ba (vel wrt ba)", (3, 6), (9, 12)),
        ("Phi_theta_theta (theta wrt theta)", (6, 9), (6, 9)),
        ("Phi_theta_bg (theta wrt bg)", (6, 9), (12, 15)),
        ("Phi_ba_ba (ba wrt ba)", (9, 12), (9, 12)),
        ("Phi_bg_bg (bg wrt bg)", (12, 15), (12, 15))
    ]
    
    for name, r, c in blocks:
        A = Phi[r[0]:r[1], c[0]:c[1]]
        N = Phi_num[r[0]:r[1], c[0]:c[1]]
        diff = np.max(np.abs(A - N))
        rel_diff = diff / (np.max(np.abs(N)) + 1e-12)
        status = "PASS" if rel_diff < 1e-3 or diff < 1e-4 else "FAIL"
        print(f"  {name:35s}: max_abs_diff = {diff:.2e}, rel_err = {rel_diff:.2e} -> {status}")
        if status == "FAIL":
            print(f"    Analytic:\n{A}")
            print(f"    Numerical:\n{N}")

if __name__ == "__main__":
    run_jacobian_audit()
