package com.sih26168.idr.engine;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Causal rolling feature extractor for real-time mobile execution (10 Hz).
 * Computes the 16 multi-domain statistical, kinematic, and vibration features
 * over a 1.0-second trailing window (10 epochs @ 10 Hz) for CausalSpeedModel.
 */
public class LiveFeatureExtractor {

    private final int windowSize;
    private final Deque<Vector3> accWindow = new ArrayDeque<>();
    private final Deque<Vector3> gyroWindow = new ArrayDeque<>();
    private final Deque<Double> jerkWindow = new ArrayDeque<>();
    private Vector3 lastAcc = null;
    private double lastMagHeading = 0.0;
    private boolean hasLastMagHeading = false;

    public LiveFeatureExtractor(int windowSize) {
        this.windowSize = windowSize;
    }

    public LiveFeatureExtractor() {
        this(10); // 1.0 second @ 10 Hz
    }

    public synchronized double[] extractFeatures(
        Vector3 accV,
        Vector3 gyroV,
        double pitchDeg,
        double rollDeg,
        double calMagHeadingDeg,
        boolean isStationary,
        double dt
    ) {
        // 1. Update Jerk
        double jerk = 0.0;
        if (lastAcc != null && dt > 1e-4) {
            double dx = accV.x - lastAcc.x;
            double dy = accV.y - lastAcc.y;
            double dz = accV.z - lastAcc.z;
            jerk = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt;
        }
        lastAcc = new Vector3(accV.x, accV.y, accV.z);

        // 2. Manage sliding window ring buffers
        accWindow.addLast(new Vector3(accV.x, accV.y, accV.z));
        gyroWindow.addLast(new Vector3(gyroV.x, gyroV.y, gyroV.z));
        jerkWindow.addLast(jerk);

        while (accWindow.size() > windowSize) accWindow.removeFirst();
        while (gyroWindow.size() > windowSize) gyroWindow.removeFirst();
        while (jerkWindow.size() > windowSize) jerkWindow.removeFirst();

        int n = accWindow.size();

        // 3. Compute rolling statistics
        double sumAx = 0.0, sumSqAx = 0.0;
        double sumSqAy = 0.0, sumAy = 0.0;
        double sumSqAz = 0.0, sumAz = 0.0;
        double sumSqJerk = 0.0;

        for (Vector3 a : accWindow) {
            sumAx += a.x;
            sumSqAx += a.x * a.x;
            sumAy += a.y;
            sumSqAy += a.y * a.y;
            sumAz += a.z;
            sumSqAz += a.z * a.z;
        }

        for (Double j : jerkWindow) {
            sumSqJerk += j * j;
        }

        double axMean = sumAx / n;
        double axVar = Math.max(0.0, (sumSqAx / n) - (axMean * axMean));
        double axStd = Math.sqrt(axVar);

        double ayMean = sumAy / n;
        double ayVar = Math.max(0.0, (sumSqAy / n) - (ayMean * ayMean));
        double ayStd = Math.sqrt(ayVar);

        double azMean = sumAz / n;
        double azVar = Math.max(0.0, (sumSqAz / n) - (azMean * azMean));
        double azStd = Math.sqrt(azVar);
        double azRms = Math.sqrt(sumSqAz / n);

        double jerkRms = Math.sqrt(sumSqJerk / n);

        // Gyro rolling stats
        double sumGyroZ = 0.0, sumSqGyroZ = 0.0;
        double sumGyroNorm = 0.0;

        for (Vector3 g : gyroWindow) {
            sumGyroZ += g.z;
            sumSqGyroZ += g.z * g.z;
            sumGyroNorm += Math.sqrt(g.x * g.x + g.y * g.y + g.z * g.z);
        }

        double gyroZMean = sumGyroZ / n;
        double gyroZVar = Math.max(0.0, (sumSqGyroZ / n) - (gyroZMean * gyroZMean));
        double gyroYawStd = Math.sqrt(gyroZVar);
        double gyroNormMean = sumGyroNorm / n;

        // Compass rate dpsi/dt (deg/s)
        double dpsiMag = 0.0;
        if (hasLastMagHeading && dt > 1e-4) {
            double diff = ((calMagHeadingDeg - lastMagHeading + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
            dpsiMag = diff / dt;
        }
        lastMagHeading = calMagHeadingDeg;
        hasLastMagHeading = true;

        // 4. Assemble canonical 16-feature vector
        double[] feat = new double[16];
        feat[0] = accV.x;
        feat[1] = accV.y;
        feat[2] = accV.z;
        feat[3] = axMean;
        feat[4] = axStd;
        feat[5] = ayStd;
        feat[6] = azStd;
        feat[7] = azRms;
        feat[8] = jerkRms;
        feat[9] = gyroV.z;
        feat[10] = gyroNormMean;
        feat[11] = gyroYawStd;
        feat[12] = pitchDeg;
        feat[13] = rollDeg;
        feat[14] = dpsiMag;
        feat[15] = isStationary ? 1.0 : 0.0;

        return feat;
    }
}
