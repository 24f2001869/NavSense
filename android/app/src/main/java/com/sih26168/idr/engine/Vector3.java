package com.sih26168.idr.engine;

/**
 * High-performance 3D vector implementation with vector algebra operations.
 */
public class Vector3 {
    public double x;
    public double y;
    public double z;

    public Vector3() {
        this(0.0, 0.0, 0.0);
    }

    public Vector3(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vector3(double[] arr) {
        this(arr[0], arr[1], arr[2]);
    }

    public Vector3 copy() {
        return new Vector3(x, y, z);
    }

    public Vector3 add(Vector3 v) {
        return new Vector3(x + v.x, y + v.y, z + v.z);
    }

    public Vector3 sub(Vector3 v) {
        return new Vector3(x - v.x, y - v.y, z - v.z);
    }

    public Vector3 minus(Vector3 v) {
        return sub(v);
    }

    public Vector3 scale(double s) {
        return new Vector3(x * s, y * s, z * s);
    }

    public double dot(Vector3 v) {
        return x * v.x + y * v.y + z * v.z;
    }

    public Vector3 cross(Vector3 v) {
        return new Vector3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    public double norm() {
        return Math.sqrt(x * x + y * y + z * z);
    }

    public Vector3 normalized() {
        double n = norm();
        if (n < 1e-12) return new Vector3(0.0, 0.0, 0.0);
        return scale(1.0 / n);
    }

    public double[] toArray() {
        return new double[]{x, y, z};
    }

    @Override
    public String toString() {
        return String.format("[%.4f, %.4f, %.4f]", x, y, z);
    }
}
