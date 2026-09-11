package com.sih26168.idr.engine;

import java.util.Arrays;

/**
 * 15-State Error-State Kalman Filter (ESKF3D) for INS/GNSS integration.
 * Direct Java port of src/navigation/eskf.py.
 *
 * Error state delta_x: [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)]^T
 */
public class ESKF3D {
    // Nominal navigation states
    public Vector3 pos_n;
    public Vector3 vel_n;
    public Vector3 ba;
    public Vector3 bg;
    public double g_val = 9.80665;

    public AttitudeEstimator3D attitude;

    // Error state covariance P (15x15)
    public Matrix P;

    // Noise parameters
    public double sigma_a;
    public double sigma_g;
    public double sigma_ba;
    public double sigma_bg;

    // GNSS measurement covariance R (6x6)
    public Matrix R;

    // Measurement Jacobian H (6x15)
    public Matrix H;

    // History tracking
    public Vector3 last_a_n = new Vector3();
    public Vector3 last_f_corr_v = new Vector3();
    public Vector3 last_omega_corr_v = new Vector3();

    public ESKF3D(double sigma_a, double sigma_g, double sigma_ba, double sigma_bg) {
        this(
            new Vector3(), new Vector3(), 0.0, 0.0, 0.0,
            Matrix.identity(3), new Vector3(), new Vector3(),
            sigma_a, sigma_g, sigma_ba, sigma_bg,
            2.0, 3.0, 0.2, 0.5, 9.80665
        );
    }

    public ESKF3D(
        Vector3 initPosEnu,
        Vector3 initVelEnu,
        double initHeadingDeg,
        double initPitchDeg,
        double initRollDeg,
        Matrix R_vp,
        Vector3 initBa,
        Vector3 initBg,
        double sigma_a,
        double sigma_g,
        double sigma_ba,
        double sigma_bg,
        double r_pos,
        double r_pos_z,
        double r_vel,
        double r_vel_z,
        double gravity
    ) {
        this.pos_n = initPosEnu != null ? initPosEnu.copy() : new Vector3();
        this.vel_n = initVelEnu != null ? initVelEnu.copy() : new Vector3();
        this.ba = initBa != null ? initBa.copy() : new Vector3();
        this.bg = initBg != null ? initBg.copy() : new Vector3();
        this.g_val = gravity;

        this.attitude = new AttitudeEstimator3D(initHeadingDeg, initPitchDeg, initRollDeg, R_vp);

        this.sigma_a = sigma_a;
        this.sigma_g = sigma_g;
        this.sigma_ba = sigma_ba;
        this.sigma_bg = sigma_bg;

        // Initialize P (15x15)
        double[] pDiag = new double[]{
            1.5 * 1.5, 1.5 * 1.5, 3.0 * 3.0,
            0.1 * 0.1, 0.1 * 0.1, 0.2 * 0.2,
            Math.toRadians(2.0) * Math.toRadians(2.0),
            Math.toRadians(2.0) * Math.toRadians(2.0),
            Math.toRadians(5.0) * Math.toRadians(5.0),
            0.15 * 0.15, 0.15 * 0.15, 0.15 * 0.15,
            Math.toRadians(0.5) * Math.toRadians(0.5),
            Math.toRadians(0.5) * Math.toRadians(0.5),
            Math.toRadians(0.5) * Math.toRadians(0.5)
        };
        this.P = Matrix.diagonal(pDiag);

        // GNSS R (6x6)
        double[] rDiag = new double[]{
            r_pos * r_pos, r_pos * r_pos, r_pos_z * r_pos_z,
            r_vel * r_vel, r_vel * r_vel, r_vel_z * r_vel_z
        };
        this.R = Matrix.diagonal(rDiag);

        // GNSS H (6x15)
        this.H = new Matrix(6, 15);
        for (int i = 0; i < 3; i++) {
            this.H.set(i, i, 1.0);
            this.H.set(3 + i, 3 + i, 1.0);
        }
    }

    public void predict(double accelX, double accelY, double accelZ,
                        double gyroX, double gyroY, double gyroZ, double dt) {
        if (dt <= 0.0) return;

        // 1. Measured vectors in vehicle body frame
        Vector3 fMeasP = new Vector3(accelX, accelY, accelZ);
        Vector3 omegaMeasP = new Vector3(gyroX, gyroY, gyroZ);

        double[] fMeasVArr = attitude.R_vp.multiply(new double[]{fMeasP.x, fMeasP.y, fMeasP.z});
        Vector3 fMeasV = new Vector3(fMeasVArr[0], fMeasVArr[1], fMeasVArr[2]);

        double[] omegaMeasVArr = attitude.R_vp.multiply(new double[]{omegaMeasP.x, omegaMeasP.y, omegaMeasP.z});
        Vector3 omegaMeasV = new Vector3(omegaMeasVArr[0], omegaMeasVArr[1], omegaMeasVArr[2]);

        // 2. Subtract estimated sensor biases
        Vector3 fCorrV = fMeasV.sub(ba);
        Vector3 omegaCorrV = omegaMeasV.sub(bg);
        this.last_f_corr_v = fCorrV;
        this.last_omega_corr_v = omegaCorrV;

        // 3. Specific force in navigation frame
        Matrix C_v_n = attitude.getDcm();
        double[] fCorrNArr = C_v_n.multiply(new double[]{fCorrV.x, fCorrV.y, fCorrV.z});
        Vector3 fCorrN = new Vector3(fCorrNArr[0], fCorrNArr[1], fCorrNArr[2]);

        // 4. Kinematic acceleration with gravity compensation
        Vector3 aN = new Vector3(fCorrN.x, fCorrN.y, fCorrN.z - g_val);
        this.last_a_n = aN;

        // 5. Nominal state propagation (Trapezoidal integration)
        pos_n = pos_n.add(vel_n.scale(dt)).add(aN.scale(0.5 * dt * dt));
        vel_n = vel_n.add(aN.scale(dt));

        // Attitude quaternion propagation
        Quaternion dq = Quaternion.fromRotationVector(omegaCorrV.scale(dt));
        attitude.q_nv = Quaternion.multiply(attitude.q_nv, dq);
        attitude.q_nv.normalize();

        // 6. Discrete State Transition Matrix Phi (15x15)
        Matrix Phi = Matrix.identity(15);
        Matrix skewF = Matrix.skew(fCorrV);
        Matrix skewW = Matrix.skew(omegaCorrV);

        // delta_p block
        for (int i = 0; i < 3; i++) {
            Phi.set(i, 3 + i, dt);
        }
        Matrix C_skewF = C_v_n.multiply(skewF);
        Matrix pTheta = C_skewF.scale(-0.5 * dt * dt);
        Matrix pBa = C_v_n.scale(-0.5 * dt * dt);
        Phi.setBlock(0, 6, pTheta);
        Phi.setBlock(0, 9, pBa);

        // delta_v block
        Matrix vTheta = C_skewF.scale(-dt);
        Matrix vBa = C_v_n.scale(-dt);
        Phi.setBlock(3, 6, vTheta);
        Phi.setBlock(3, 9, vBa);

        // delta_theta block
        Matrix thetaTheta = Matrix.identity(3).sub(skewW.scale(dt));
        Matrix thetaBg = Matrix.identity(3).scale(-dt);
        Phi.setBlock(6, 6, thetaTheta);
        Phi.setBlock(6, 12, thetaBg);

        // 7. Discrete Process Noise Covariance Q_d (15x15)
        double[] qDiag = new double[15];
        double qP = (1.0 / 3.0) * (sigma_a * sigma_a) * (dt * dt * dt);
        double qV = (sigma_a * sigma_a) * dt;
        double qTheta = (sigma_g * sigma_g) * dt;
        double qBa = (sigma_ba * sigma_ba) * dt;
        double qBg = (sigma_bg * sigma_bg) * dt;
        for (int i = 0; i < 3; i++) {
            qDiag[i] = qP;
            qDiag[3 + i] = qV;
            qDiag[6 + i] = qTheta;
            qDiag[9 + i] = qBa;
            qDiag[12 + i] = qBg;
        }
        Matrix Qd = Matrix.diagonal(qDiag);

        // 8. Covariance propagation: P = Phi * P * Phi^T + Qd
        Matrix P_new = Phi.multiply(P).multiply(Phi.transpose()).add(Qd);
        P_new.symmetrize();
        this.P = P_new;
    }

    public void updateGnss(Vector3 gnssPosEnu, Vector3 gnssVelEnu) {
        double[] z = new double[]{
            gnssPosEnu.x, gnssPosEnu.y, gnssPosEnu.z,
            gnssVelEnu.x, gnssVelEnu.y, gnssVelEnu.z
        };
        double[] y = new double[]{
            z[0] - pos_n.x, z[1] - pos_n.y, z[2] - pos_n.z,
            z[3] - vel_n.x, z[4] - vel_n.y, z[5] - vel_n.z
        };

        // S = H * P * H^T + R (6x6)
        Matrix S = H.multiply(P).multiply(H.transpose()).add(R);
        Matrix S_inv = S.invert();

        // K = P * H^T * S_inv (15x6)
        Matrix K = P.multiply(H.transpose()).multiply(S_inv);

        // delta_x = K * y (15)
        double[] deltaX = K.multiply(y);

        // State Error Injection
        pos_n.x += deltaX[0];
        pos_n.y += deltaX[1];
        pos_n.z += deltaX[2];

        vel_n.x += deltaX[3];
        vel_n.y += deltaX[4];
        vel_n.z += deltaX[5];

        Vector3 dThetaB = new Vector3(deltaX[6], deltaX[7], deltaX[8]);
        Quaternion dqCorr = Quaternion.fromRotationVector(dThetaB);
        attitude.q_nv = Quaternion.multiply(attitude.q_nv, dqCorr);
        attitude.q_nv.normalize();

        ba.x += deltaX[9];
        ba.y += deltaX[10];
        ba.z += deltaX[11];

        bg.x += deltaX[12];
        bg.y += deltaX[13];
        bg.z += deltaX[14];

        // Joseph-form Covariance Update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        Matrix IKH = Matrix.identity(15).sub(K.multiply(H));
        Matrix P_new = IKH.multiply(P).multiply(IKH.transpose()).add(K.multiply(R).multiply(K.transpose()));
        P_new.symmetrize();
        this.P = P_new;
    }

    /**
     * Reacquires GNSS after an extended outage or dead-reckoning period.
     * Prevents large accumulated position drift (e.g. 50m - 1000m) from cross-corrupting
     * velocity, attitude (dTheta), and sensor biases (ba, bg) through linear Kalman cross-covariances.
     *
     * Mathematical Rationale:
     * - Position is directly re-anchored to the absolute GNSS position measurement.
     * - Position error covariance block is re-initialized to GNSS measurement noise (sigma_pos^2).
     * - Stale cross-covariances between position and the other 12 error states are cleared.
     * - Velocity is updated directly from GNSS velocity measurement using its decoupled Kalman gain.
     * - Attitude quaternion and sensor biases are strictly protected (zero kick).
     */
    public void reacquireGnss(Vector3 gnssPosEnu, Vector3 gnssVelEnu, double accuracyM) {
        // 1. Direct position re-anchoring to absolute GNSS position
        this.pos_n = gnssPosEnu.copy();

        // 2. Direct velocity re-anchoring to absolute GNSS Doppler velocity
        this.vel_n = gnssVelEnu.copy();

        // 3. Reset position and velocity covariance blocks to measurement noise
        double rPos = Math.max(accuracyM * accuracyM, 4.0);
        P.set(0, 0, rPos);
        P.set(1, 1, rPos);
        P.set(2, 2, rPos * 2.25); // vertical GNSS accuracy typically ~1.5x horizontal

        double rVel = 0.25; // (0.5 m/s)^2 velocity variance
        P.set(3, 3, rVel);
        P.set(4, 4, rVel);
        P.set(5, 5, rVel * 2.25);

        // 4. Clear stale position & velocity cross-covariances with attitude and biases
        for (int i = 0; i < 6; i++) {
            for (int j = 6; j < 15; j++) {
                P.set(i, j, 0.0);
                P.set(j, i, 0.0);
            }
        }
        // Also clear cross-covariances between position and velocity
        for (int i = 0; i < 3; i++) {
            for (int j = 3; j < 6; j++) {
                P.set(i, j, 0.0);
                P.set(j, i, 0.0);
            }
        }
        P.symmetrize();
    }

    /**
     * TCN Forward Speed Observation (1 Hz).
     * Strictly enforces K_x[6:15] = 0 (zero attitude feedback).
     */
    public double updateTcnSpeed(double speedMps, double sigmaSpeed) {
        Matrix C_v_n = attitude.getDcm();
        Matrix C_n_v = C_v_n.transpose();
        double[] velVArr = C_n_v.multiply(new double[]{vel_n.x, vel_n.y, vel_n.z});

        double rx = speedMps - velVArr[0];
        Matrix Hx = new Matrix(1, 15);
        Hx.set(0, 3, C_n_v.get(0, 0));
        Hx.set(0, 4, C_n_v.get(0, 1));
        Hx.set(0, 5, C_n_v.get(0, 2));

        Matrix P_HxT = P.multiply(Hx.transpose());
        double Hx_P_HxT = Hx.multiply(P_HxT).get(0, 0);
        double Sx = Hx_P_HxT + sigmaSpeed * sigmaSpeed;
        double nis = (rx * rx) / Math.max(Sx, 1e-6);

        if (nis > 9.0) {
            return nis; // Chi2 outlier gated out
        }

        Matrix Kx = P_HxT.scale(1.0 / Math.max(Sx, 1e-6));
        // Strict attitude and bias freeze: Kx[6:15] = 0
        for (int i = 6; i < 15; i++) {
            Kx.set(i, 0, 0.0);
        }

        // State correction
        pos_n.x += Kx.get(0, 0) * rx;
        pos_n.y += Kx.get(1, 0) * rx;
        pos_n.z += Kx.get(2, 0) * rx;

        vel_n.x += Kx.get(3, 0) * rx;
        vel_n.y += Kx.get(4, 0) * rx;
        vel_n.z += Kx.get(5, 0) * rx;

        // Joseph-form covariance update
        Matrix IKH = Matrix.identity(15).sub(Kx.multiply(Hx));
        Matrix R_mat = new Matrix(1, 1, new double[]{sigmaSpeed * sigmaSpeed});
        Matrix P_new = IKH.multiply(P).multiply(IKH.transpose()).add(Kx.multiply(R_mat).multiply(Kx.transpose()));
        P_new.symmetrize();
        this.P = P_new;

        return nis;
    }

    /**
     * Decoupled Pure-Velocity Lateral Damping (Variant V1: 10 Hz).
     * Strictly enforces Ky[6:15] = 0 (zero attitude feedback).
     */
    public double updateDecoupledLateralVelocityDamping(double sigmaNhc) {
        Matrix C_v_n = attitude.getDcm();
        Matrix C_n_v = C_v_n.transpose();
        double[] velVArr = C_n_v.multiply(new double[]{vel_n.x, vel_n.y, vel_n.z});

        double ry = 0.0 - velVArr[1];
        Matrix Hy = new Matrix(1, 15);
        Hy.set(0, 3, C_n_v.get(1, 0));
        Hy.set(0, 4, C_n_v.get(1, 1));
        Hy.set(0, 5, C_n_v.get(1, 2));

        Matrix P_HyT = P.multiply(Hy.transpose());
        double Hy_P_HyT = Hy.multiply(P_HyT).get(0, 0);
        double Sy = Hy_P_HyT + sigmaNhc * sigmaNhc;
        double nis = (ry * ry) / Math.max(Sy, 1e-6);

        Matrix Ky = P_HyT.scale(1.0 / Math.max(Sy, 1e-6));
        // Strict attitude and bias freeze: Ky[6:15] = 0
        for (int i = 6; i < 15; i++) {
            Ky.set(i, 0, 0.0);
        }

        // State correction (velocity only)
        vel_n.x += Ky.get(3, 0) * ry;
        vel_n.y += Ky.get(4, 0) * ry;
        vel_n.z += Ky.get(5, 0) * ry;

        // Joseph-form covariance update
        Matrix IKH = Matrix.identity(15).sub(Ky.multiply(Hy));
        Matrix R_mat = new Matrix(1, 1, new double[]{sigmaNhc * sigmaNhc});
        Matrix P_new = IKH.multiply(P).multiply(IKH.transpose()).add(Ky.multiply(R_mat).multiply(Ky.transpose()));
        P_new.symmetrize();
        this.P = P_new;

        return nis;
    }
}
