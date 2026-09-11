package com.sih26168.idr.engine;

/**
 * Hamilton unit quaternion representation and attitude kinematics.
 * Matches python attitude.py and eskf.py quaternion conventions.
 */
public class Quaternion {
    public double w;
    public double x;
    public double y;
    public double z;

    public Quaternion() {
        this(1.0, 0.0, 0.0, 0.0);
    }

    public Quaternion(double w, double x, double y, double z) {
        this.w = w;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Quaternion copy() {
        return new Quaternion(w, x, y, z);
    }

    public double norm() {
        return Math.sqrt(w * w + x * x + y * y + z * z);
    }

    public void normalize() {
        double n = norm();
        if (n > 1e-12) {
            w /= n;
            x /= n;
            y /= n;
            z /= n;
        } else {
            w = 1.0;
            x = 0.0;
            y = 0.0;
            z = 0.0;
        }
    }

    /**
     * Hamilton quaternion multiplication: q = q1 (x) q2
     */
    public static Quaternion multiply(Quaternion q1, Quaternion q2) {
        return new Quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        );
    }

    /**
     * Converts rotation vector v into a unit quaternion.
     */
    public static Quaternion fromRotationVector(Vector3 v) {
        double th = v.norm();
        if (th < 1e-12) {
            return new Quaternion(1.0, 0.0, 0.0, 0.0);
        }
        double s = Math.sin(0.5 * th) / th;
        return new Quaternion(
            Math.cos(0.5 * th),
            s * v.x,
            s * v.y,
            s * v.z
        );
    }

    /**
     * Constructs initial quaternion from Euler angles (yaw, pitch, roll in degrees).
     * Navigation frame is ENU (East, North, Up).
     */
    public static Quaternion fromEuler(double yawDeg, double pitchDeg, double rollDeg) {
        double psi = Math.toRadians(yawDeg);
        double theta = Math.toRadians(pitchDeg);
        double phi = Math.toRadians(rollDeg);

        double cosPsi = Math.cos(psi);
        double sinPsi = Math.sin(psi);
        Matrix R_yaw = new Matrix(3, 3);
        R_yaw.set(0, 0, sinPsi);  R_yaw.set(0, 1, -cosPsi); R_yaw.set(0, 2, 0.0);
        R_yaw.set(1, 0, cosPsi);  R_yaw.set(1, 1,  sinPsi); R_yaw.set(1, 2, 0.0);
        R_yaw.set(2, 0, 0.0);     R_yaw.set(2, 1,  0.0);    R_yaw.set(2, 2, 1.0);

        double cp = Math.cos(theta);
        double sp = Math.sin(theta);
        Matrix R_pitch = new Matrix(3, 3);
        R_pitch.set(0, 0,  cp); R_pitch.set(0, 1, 0.0); R_pitch.set(0, 2, sp);
        R_pitch.set(1, 0, 0.0); R_pitch.set(1, 1, 1.0); R_pitch.set(1, 2, 0.0);
        R_pitch.set(2, 0, -sp); R_pitch.set(2, 1, 0.0); R_pitch.set(2, 2, cp);

        double cr = Math.cos(phi);
        double sr = Math.sin(phi);
        Matrix R_roll = new Matrix(3, 3);
        R_roll.set(0, 0, 1.0); R_roll.set(0, 1, 0.0); R_roll.set(0, 2, 0.0);
        R_roll.set(1, 0, 0.0); R_roll.set(1, 1,  cr); R_roll.set(1, 2, -sr);
        R_roll.set(2, 0, 0.0); R_roll.set(2, 1,  sr); R_roll.set(2, 2,  cr);

        Matrix C = R_yaw.multiply(R_pitch).multiply(R_roll);
        return fromDcm(C);
    }

    public static Quaternion fromDcm(Matrix C) {
        double tr = C.get(0, 0) + C.get(1, 1) + C.get(2, 2);
        Quaternion q = new Quaternion();
        if (tr > 0.0) {
            double s = Math.sqrt(tr + 1.0) * 2.0;
            q.w = 0.25 * s;
            q.x = (C.get(2, 1) - C.get(1, 2)) / s;
            q.y = (C.get(0, 2) - C.get(2, 0)) / s;
            q.z = (C.get(1, 0) - C.get(0, 1)) / s;
        } else if ((C.get(0, 0) > C.get(1, 1)) && (C.get(0, 0) > C.get(2, 2))) {
            double s = Math.sqrt(1.0 + C.get(0, 0) - C.get(1, 1) - C.get(2, 2)) * 2.0;
            q.w = (C.get(2, 1) - C.get(1, 2)) / s;
            q.x = 0.25 * s;
            q.y = (C.get(0, 1) + C.get(1, 0)) / s;
            q.z = (C.get(0, 2) + C.get(2, 0)) / s;
        } else if (C.get(1, 1) > C.get(2, 2)) {
            double s = Math.sqrt(1.0 + C.get(1, 1) - C.get(0, 0) - C.get(2, 2)) * 2.0;
            q.w = (C.get(0, 2) - C.get(2, 0)) / s;
            q.x = (C.get(0, 1) + C.get(1, 0)) / s;
            q.y = 0.25 * s;
            q.z = (C.get(1, 2) + C.get(2, 1)) / s;
        } else {
            double s = Math.sqrt(1.0 + C.get(2, 2) - C.get(0, 0) - C.get(1, 1)) * 2.0;
            q.w = (C.get(1, 0) - C.get(0, 1)) / s;
            q.x = (C.get(0, 2) + C.get(2, 0)) / s;
            q.y = (C.get(1, 2) + C.get(2, 1)) / s;
            q.z = 0.25 * s;
        }
        q.normalize();
        return q;
    }

    /**
     * Computes the 3x3 Direction Cosine Matrix C_v^n transforming vectors
     * from Vehicle frame to Navigation ENU frame: v_n = C_v^n @ v_v.
     */
    public Matrix toDcm() {
        Matrix C = new Matrix(3, 3);
        C.set(0, 0, 1.0 - 2.0 * (y * y + z * z));
        C.set(0, 1, 2.0 * (x * y - w * z));
        C.set(0, 2, 2.0 * (x * z + w * y));

        C.set(1, 0, 2.0 * (x * y + w * z));
        C.set(1, 1, 1.0 - 2.0 * (x * x + z * z));
        C.set(1, 2, 2.0 * (y * z - w * x));

        C.set(2, 0, 2.0 * (x * z - w * y));
        C.set(2, 1, 2.0 * (y * z + w * x));
        C.set(2, 2, 1.0 - 2.0 * (x * x + y * y));
        return C;
    }

    public double getYawDeg() {
        Matrix C = toDcm();
        double psiRad = Math.atan2(C.get(0, 0), C.get(1, 0));
        double yawDeg = Math.toDegrees(psiRad);
        return (yawDeg % 360.0 + 360.0) % 360.0;
    }

    public double getPitchDeg() {
        Matrix C = toDcm();
        double sinPitch = Math.max(-1.0, Math.min(1.0, C.get(2, 0)));
        return Math.toDegrees(Math.asin(sinPitch));
    }

    public double getRollDeg() {
        Matrix C = toDcm();
        return Math.toDegrees(Math.atan2(C.get(2, 1), C.get(2, 2)));
    }
}
