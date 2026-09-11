package com.sih26168.idr.engine;

/**
 * Continuous 3D Attitude Estimator tracking attitude quaternion q_nv and DCM C_v^n.
 * Direct Java port of src/navigation/attitude.py.
 */
public class AttitudeEstimator3D {
    public Matrix R_vp;
    public Quaternion q_nv;
    public double g_val = 9.80665;

    public Vector3 last_omega_v = new Vector3();
    public Vector3 last_f_v = new Vector3();

    public AttitudeEstimator3D(double initHeadingDeg, double initPitchDeg, double initRollDeg, Matrix R_vp) {
        if (R_vp != null) {
            this.R_vp = R_vp.copy();
        } else {
            this.R_vp = Matrix.identity(3);
        }
        this.q_nv = Quaternion.fromEuler(initHeadingDeg, initPitchDeg, initRollDeg);
        this.q_nv.normalize();
    }

    public void setMountingMatrix(Matrix R_vp) {
        this.R_vp = R_vp.copy();
    }

    public Quaternion predict(double gyroX, double gyroY, double gyroZ, double dt) {
        if (dt <= 0.0) {
            return q_nv;
        }
        Vector3 omegaP = new Vector3(gyroX, gyroY, gyroZ);
        double[] omegaVArr = R_vp.multiply(new double[]{omegaP.x, omegaP.y, omegaP.z});
        Vector3 omegaV = new Vector3(omegaVArr[0], omegaVArr[1], omegaVArr[2]);
        this.last_omega_v = omegaV;

        Vector3 deltaTheta = omegaV.scale(dt);
        Quaternion dq = Quaternion.fromRotationVector(deltaTheta);
        this.q_nv = Quaternion.multiply(this.q_nv, dq);
        this.q_nv.normalize();
        return this.q_nv;
    }

    public boolean updateGravity(double accelX, double accelY, double accelZ, double ka, double accelGate) {
        Vector3 fP = new Vector3(accelX, accelY, accelZ);
        double[] fVArr = R_vp.multiply(new double[]{fP.x, fP.y, fP.z});
        Vector3 fV = new Vector3(fVArr[0], fVArr[1], fVArr[2]);
        this.last_f_v = fV;

        double fNorm = fV.norm();
        if (Math.abs(fNorm - g_val) > accelGate) {
            return false;
        }

        Vector3 fHatV = fV.scale(1.0 / fNorm);
        Matrix C = getDcm();
        Vector3 gPredV = new Vector3(C.get(2, 0), C.get(2, 1), C.get(2, 2));

        Vector3 eTilt = fHatV.cross(gPredV);
        Vector3 corrVec = eTilt.scale(ka);
        Quaternion dqTilt = Quaternion.fromRotationVector(corrVec);

        this.q_nv = Quaternion.multiply(this.q_nv, dqTilt);
        this.q_nv.normalize();
        return true;
    }

    public Matrix getDcm() {
        return q_nv.toDcm();
    }

    public double getYawDeg() {
        return q_nv.getYawDeg();
    }

    public double getPitchDeg() {
        return q_nv.getPitchDeg();
    }

    public double getRollDeg() {
        return q_nv.getRollDeg();
    }

    public Quaternion getQuaternion() {
        return q_nv.copy();
    }

    public double getQuaternionNorm() {
        return q_nv.norm();
    }

    /**
     * Instantly sets the yaw heading angle while preserving current pitch and roll.
     */
    public void setHeadingDeg(double headingDeg) {
        double pitch = getPitchDeg();
        double roll = getRollDeg();
        this.q_nv = Quaternion.fromEuler(headingDeg, pitch, roll);
        this.q_nv.normalize();
    }

    /**
     * Smoothly nudges yaw towards targetHeadingDeg by weight alpha (0.0 < alpha <= 1.0),
     * properly wrapping angular differences [-180, +180].
     */
    public void alignHeading(double targetHeadingDeg, double alpha) {
        double currentYaw = getYawDeg();
        double diff = (targetHeadingDeg - currentYaw) % 360.0;
        if (diff > 180.0) diff -= 360.0;
        else if (diff < -180.0) diff += 360.0;

        double newYaw = (currentYaw + alpha * diff) % 360.0;
        if (newYaw < 0.0) newYaw += 360.0;
        double pitch = getPitchDeg();
        double roll = getRollDeg();
        this.q_nv = Quaternion.fromEuler(newYaw, pitch, roll);
        this.q_nv.normalize();
    }
}
