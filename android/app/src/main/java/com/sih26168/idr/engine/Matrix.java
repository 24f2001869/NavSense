package com.sih26168.idr.engine;

import java.util.Arrays;

/**
 * High-performance, numerically stable Matrix library for 15-state ESKF.
 * Includes analytic 3x3 inversion, robust Gauss-Jordan matrix inversion,
 * and Joseph-stabilized Kalman covariance updates.
 */
public class Matrix {
    public final int rows;
    public final int cols;
    public final double[] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    public Matrix(int rows, int cols, double[] data) {
        this.rows = rows;
        this.cols = cols;
        this.data = Arrays.copyOf(data, rows * cols);
    }

    public static Matrix identity(int n) {
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            m.data[i * n + i] = 1.0;
        }
        return m;
    }

    public static Matrix diagonal(double[] diag) {
        int n = diag.length;
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            m.data[i * n + i] = diag[i];
        }
        return m;
    }

    public Matrix copy() {
        return new Matrix(rows, cols, data);
    }

    public double get(int r, int c) {
        return data[r * cols + c];
    }

    public void set(int r, int c, double val) {
        data[r * cols + c] = val;
    }

    public void addVal(int r, int c, double val) {
        data[r * cols + c] += val;
    }

    public Matrix add(Matrix b) {
        if (rows != b.rows || cols != b.cols) {
            throw new IllegalArgumentException("Matrix dimensions mismatch in add");
        }
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) {
            out.data[i] = this.data[i] + b.data[i];
        }
        return out;
    }

    public Matrix sub(Matrix b) {
        if (rows != b.rows || cols != b.cols) {
            throw new IllegalArgumentException("Matrix dimensions mismatch in sub");
        }
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) {
            out.data[i] = this.data[i] - b.data[i];
        }
        return out;
    }

    public Matrix scale(double s) {
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) {
            out.data[i] = this.data[i] * s;
        }
        return out;
    }

    public Matrix multiply(Matrix b) {
        if (cols != b.rows) {
            throw new IllegalArgumentException("Matrix dimensions mismatch in multiply: " + cols + " vs " + b.rows);
        }
        Matrix out = new Matrix(rows, b.cols);
        for (int i = 0; i < rows; i++) {
            int i_cols = i * cols;
            int i_bcols = i * b.cols;
            for (int k = 0; k < cols; k++) {
                double a_ik = this.data[i_cols + k];
                if (a_ik == 0.0) continue;
                int k_bcols = k * b.cols;
                for (int j = 0; j < b.cols; j++) {
                    out.data[i_bcols + j] += a_ik * b.data[k_bcols + j];
                }
            }
        }
        return out;
    }

    public double[] multiply(double[] v) {
        if (cols != v.length) {
            throw new IllegalArgumentException("Matrix-vector dimensions mismatch: " + cols + " vs " + v.length);
        }
        double[] out = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            int i_cols = i * cols;
            for (int j = 0; j < cols; j++) {
                sum += data[i_cols + j] * v[j];
            }
            out[i] = sum;
        }
        return out;
    }

    public Matrix transpose() {
        Matrix out = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            int i_cols = i * cols;
            for (int j = 0; j < cols; j++) {
                out.data[j * rows + i] = this.data[i_cols + j];
            }
        }
        return out;
    }

    public void setBlock(int startR, int startC, Matrix block) {
        for (int i = 0; i < block.rows; i++) {
            for (int j = 0; j < block.cols; j++) {
                this.set(startR + i, startC + j, block.get(i, j));
            }
        }
    }

    public Matrix getBlock(int startR, int startC, int numR, int numC) {
        Matrix block = new Matrix(numR, numC);
        for (int i = 0; i < numR; i++) {
            for (int j = 0; j < numC; j++) {
                block.set(i, j, this.get(startR + i, startC + j));
            }
        }
        return block;
    }

    public void symmetrize() {
        if (rows != cols) return;
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                double avg = 0.5 * (data[i * cols + j] + data[j * cols + i]);
                data[i * cols + j] = avg;
                data[j * cols + i] = avg;
            }
        }
    }

    /**
     * Analytic 3x3 matrix inversion via adjugate matrix.
     */
    public Matrix invert3x3() {
        if (rows != 3 || cols != 3) {
            throw new IllegalArgumentException("invert3x3 only supports 3x3 matrices");
        }
        double a00 = get(0, 0), a01 = get(0, 1), a02 = get(0, 2);
        double a10 = get(1, 0), a11 = get(1, 1), a12 = get(1, 2);
        double a20 = get(2, 0), a21 = get(2, 1), a22 = get(2, 2);

        double det = a00 * (a11 * a22 - a12 * a21) -
                     a01 * (a10 * a22 - a12 * a20) +
                     a02 * (a10 * a21 - a11 * a20);

        if (Math.abs(det) < 1e-15) {
            throw new ArithmeticException("Matrix 3x3 is singular (det=" + det + ")");
        }

        double invDet = 1.0 / det;
        Matrix inv = new Matrix(3, 3);
        inv.set(0, 0, (a11 * a22 - a12 * a21) * invDet);
        inv.set(0, 1, (a02 * a21 - a01 * a22) * invDet);
        inv.set(0, 2, (a01 * a12 - a02 * a11) * invDet);

        inv.set(1, 0, (a12 * a20 - a10 * a22) * invDet);
        inv.set(1, 1, (a00 * a22 - a02 * a20) * invDet);
        inv.set(1, 2, (a02 * a10 - a00 * a12) * invDet);

        inv.set(2, 0, (a10 * a21 - a11 * a20) * invDet);
        inv.set(2, 1, (a01 * a20 - a00 * a21) * invDet);
        inv.set(2, 2, (a00 * a11 - a01 * a10) * invDet);

        return inv;
    }

    /**
     * General matrix inversion using Gauss-Jordan elimination with partial pivoting.
     */
    public Matrix invert() {
        if (rows != cols) {
            throw new IllegalArgumentException("Inversion requires a square matrix");
        }
        if (rows == 3) {
            return invert3x3();
        }

        int n = rows;
        double[][] a = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = get(i, j);
            }
            a[i][i + n] = 1.0;
        }

        for (int i = 0; i < n; i++) {
            int pivot = i;
            double maxVal = Math.abs(a[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(a[k][i]) > maxVal) {
                    maxVal = Math.abs(a[k][i]);
                    pivot = k;
                }
            }

            if (maxVal < 1e-15) {
                throw new ArithmeticException("Matrix is singular during Gauss-Jordan inversion at row " + i);
            }

            if (pivot != i) {
                double[] tmp = a[i];
                a[i] = a[pivot];
                a[pivot] = tmp;
            }

            double scale = a[i][i];
            for (int j = i; j < 2 * n; j++) {
                a[i][j] /= scale;
            }

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = a[k][i];
                    if (factor != 0.0) {
                        for (int j = i; j < 2 * n; j++) {
                            a[k][j] -= factor * a[i][j];
                        }
                    }
                }
            }
        }

        Matrix inv = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv.set(i, j, a[i][j + n]);
            }
        }
        return inv;
    }

    public static Matrix skew(Vector3 v) {
        Matrix s = new Matrix(3, 3);
        s.set(0, 1, -v.z);
        s.set(0, 2,  v.y);
        s.set(1, 0,  v.z);
        s.set(1, 2, -v.x);
        s.set(2, 0, -v.y);
        s.set(2, 1,  v.x);
        return s;
    }
}
