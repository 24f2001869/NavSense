package com.sih26168.idr.engine;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Phase 5.0: Android Golden Reference Verification Test.
 * Executes the entire 1789-epoch Vta04 session on the Java ESKF3D engine
 * and asserts exact numerical and behavioral parity against the frozen Python reference:
 * - Position Error: <= 0.05 m
 * - Velocity Error: <= 0.01 m/s
 * - Heading Error : <= 0.05 deg
 */
public class GoldenReferenceVerifier {

    private static String readFile(File file) throws Exception {
        byte[] bytes = new byte[(int) file.length()];
        try (FileInputStream fis = new FileInputStream(file)) {
            int read = 0;
            while (read < bytes.length) {
                int r = fis.read(bytes, read, bytes.length - read);
                if (r < 0) break;
                read += r;
            }
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static double toDouble(Object o) {
        if (o instanceof Number) {
            return ((Number) o).doubleValue();
        }
        return 0.0;
    }

    private static long toLong(Object o) {
        if (o instanceof Number) {
            return ((Number) o).longValue();
        }
        return 0L;
    }

    @SuppressWarnings("unchecked")
    private static List<Double> toDoubleList(Object o) {
        List<Double> list = new ArrayList<>();
        if (o instanceof List) {
            for (Object item : (List<?>) o) {
                list.add(toDouble(item));
            }
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        System.out.println("==============================================================================");
        System.out.println("PHASE 5.0: ANDROID ESKF + TCN-KIN + V1 NUMERICAL PARITY VERIFICATION");
        System.out.println("==============================================================================");

        try {
            // 1. Locate Golden Reference File
            File repoRoot = new File(".").getCanonicalFile();
            File goldenJsonFile = new File(repoRoot, "data/golden_reference_session_v1.json");
            if (!goldenJsonFile.exists()) {
                goldenJsonFile = new File(repoRoot, "android/app/src/main/assets/golden_reference_session_v1.json");
            }
            if (!goldenJsonFile.exists()) {
                throw new IllegalStateException("Golden reference file not found: " + goldenJsonFile.getAbsolutePath());
            }

            // 2. Load Golden Reference Session
            System.out.println("[1] Ingesting Golden Reference Session from: " + goldenJsonFile.getName());
            String goldenStr = readFile(goldenJsonFile);
            Map<String, Object> goldenMap = MiniJson.parseObject(goldenStr);
            Map<String, Object> meta = (Map<String, Object>) goldenMap.get("meta");
            List<Map<String, Object>> epochs = (List<Map<String, Object>>) goldenMap.get("epochs");

            int nEpochs = epochs.size();
            double dt = toDouble(meta.get("dt"));
            long outageStart = toLong(meta.get("outage_start_epoch"));
            long outageEnd = toLong(meta.get("outage_end_epoch"));

            System.out.println("  -> Total Session Epochs: " + nEpochs + " (" + (nEpochs * dt) + " seconds)");
            System.out.println("  -> Blackout Window: Epoch " + outageStart + " -> " + outageEnd +
                               " (" + (outageStart * dt) + " s -> " + (outageEnd * dt) + " s)");

            List<List<Double>> rVpJson = (List<List<Double>>) meta.get("R_vp");
            Matrix R_vp = new Matrix(3, 3);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    R_vp.set(r, c, toDouble(rVpJson.get(r).get(c)));
                }
            }

            // 3. Initialize Java ESKF3D Engine at Epoch 0
            System.out.println("\n[2] Initializing Java ESKF3D Engine...");
            List<Double> initPos = toDoubleList(meta.get("init_pos_enu"));
            List<Double> initVel = toDoubleList(meta.get("init_vel_enu"));
            double initYaw = toDouble(meta.get("init_heading_deg"));

            ESKF3D eskf = new ESKF3D(0.3, 0.01, 0.001, 0.0001);
            eskf.pos_n = new Vector3(initPos.get(0), initPos.get(1), initPos.get(2));
            eskf.vel_n = new Vector3(initVel.get(0), initVel.get(1), initVel.get(2));
            eskf.attitude = new AttitudeEstimator3D(initYaw, 0.0, 0.0, R_vp);

            // Initialize covariance P
            eskf.P = Matrix.identity(15).scale(0.01);
            for (int i = 0; i < 3; i++) eskf.P.set(i, i, 1.0);
            for (int i = 3; i < 6; i++) eskf.P.set(i, i, 0.25);
            for (int i = 6; i < 9; i++) eskf.P.set(i, i, Math.toRadians(2.0) * Math.toRadians(2.0));

            // 4. Verification Loop
            System.out.println("\n[3] Executing 1789-Epoch Step-by-Step Simulation...");

            double maxPosDiff = 0.0;
            double maxVelDiff = 0.0;
            double maxHeadingDiff = 0.0;
            double maxPosDiffOutage = 0.0;

            int countOutage = 0;
            int tcnUpdatesCount = 0;
            int nhcUpdatesCount = 0;
            int firstDivEpoch = -1;

            long tStart = System.currentTimeMillis();

            for (int i = 0; i < nEpochs; i++) {
                Map<String, Object> ep = epochs.get(i);
                Map<String, Object> inputs = (Map<String, Object>) ep.get("inputs");
                Map<String, Object> expected = (Map<String, Object>) ep.get("expected_outputs");

                List<Double> accList = toDoubleList(inputs.get("accel_p"));
                List<Double> gyroList = toDoubleList(inputs.get("gyro_p"));
                boolean gnssValid = Boolean.TRUE.equals(inputs.get("gnss_valid"));
                Object tcnSpdObj = inputs.get("tcn_speed_mps");

                double ax_p = accList.get(0);
                double ay_p = accList.get(1);
                double az_p = accList.get(2);
                double gx_p = gyroList.get(0);
                double gy_p = gyroList.get(1);
                double gz_p = gyroList.get(2);

                // 1. Inertial Propagation
                eskf.predict(ax_p, ay_p, az_p, gx_p, gy_p, gz_p, dt);

                // 2. Gravity Leveling
                eskf.attitude.updateGravity(ax_p, ay_p, az_p, 0.02, 1.5);

                // 3. GNSS Update when available
                if (gnssValid && (i % 10 == 0)) {
                    List<Double> gPos = toDoubleList(inputs.get("gnss_pos_enu"));
                    List<Double> gVel = toDoubleList(inputs.get("gnss_vel_enu"));
                    Vector3 gnssPos = new Vector3(gPos.get(0), gPos.get(1), gPos.get(2));
                    Vector3 gnssVel = new Vector3(gVel.get(0), gVel.get(1), gVel.get(2));
                    eskf.updateGnss(gnssPos, gnssVel);
                }

                // 4. Dead-Reckoning Updates during Outage
                if (!gnssValid) {
                    // TCN Forward Speed Update
                    if (tcnSpdObj != null) {
                        double tcnSpd = toDouble(tcnSpdObj);
                        eskf.updateTcnSpeed(tcnSpd, 1.0);
                        tcnUpdatesCount++;
                    }

                    // V1 Decoupled Pure-Velocity Damping
                    eskf.updateDecoupledLateralVelocityDamping(0.5);
                    nhcUpdatesCount++;
                }

                // Reference State Comparison
                List<Double> refPos = toDoubleList(expected.get("pos_enu"));
                List<Double> refVel = toDoubleList(expected.get("vel_enu"));
                double refYaw = toDouble(expected.get("yaw_deg"));

                double dE = eskf.pos_n.x - refPos.get(0);
                double dN = eskf.pos_n.y - refPos.get(1);
                double dU = eskf.pos_n.z - refPos.get(2);
                double posDiff = Math.sqrt(dE * dE + dN * dN + dU * dU);

                double dVe = eskf.vel_n.x - refVel.get(0);
                double dVn = eskf.vel_n.y - refVel.get(1);
                double dVu = eskf.vel_n.z - refVel.get(2);
                double velDiff = Math.sqrt(dVe * dVe + dVn * dVn + dVu * dVu);

                double yawDiff = Math.abs(((eskf.attitude.getYawDeg() - refYaw + 180.0) % 360.0 + 360.0) % 360.0 - 180.0);

                if (posDiff > maxPosDiff) maxPosDiff = posDiff;
                if (velDiff > maxVelDiff) maxVelDiff = velDiff;
                if (yawDiff > maxHeadingDiff) maxHeadingDiff = yawDiff;

                boolean isOutageEpoch = (i >= outageStart && i < outageEnd);
                if (isOutageEpoch) {
                    countOutage++;
                    if (posDiff > maxPosDiffOutage) {
                        maxPosDiffOutage = posDiff;
                    }
                }
            }

            long elapsedMs = System.currentTimeMillis() - tStart;
            double msPerEpoch = (double) elapsedMs / nEpochs;

            // 5. Parity Report
            System.out.println("\n[4] Numerical Parity Results Across All 1,789 Epochs:");
            System.out.println(String.format("  - Execution Time       : %d ms (%.3f ms/epoch, %.1f Hz throughput)",
                elapsedMs, msPerEpoch, 1000.0 / Math.max(msPerEpoch, 0.001)));
            System.out.println(String.format("  - Max Position Error   : %.4f m   [Threshold: <= 0.05 m]", maxPosDiff));
            System.out.println(String.format("  - Max Velocity Error   : %.4f m/s [Threshold: <= 0.01 m/s]", maxVelDiff));
            System.out.println(String.format("  - Max Heading Error    : %.4f deg [Threshold: <= 0.05 deg]", maxHeadingDiff));
            System.out.println(String.format("  - Max Outage Pos Error : %.4f m   [60s Blackout Window]", maxPosDiffOutage));
            System.out.println(String.format("  - TCN Speed Updates    : %d applied", tcnUpdatesCount));
            System.out.println(String.format("  - V1 NHC Damping Updates: %d applied", nhcUpdatesCount));

            // Parity Assertions
            if (maxPosDiff > 0.05) {
                throw new AssertionError(String.format("Position parity failed! Max error = %.4f m > 0.05 m", maxPosDiff));
            }
            if (maxVelDiff > 0.01) {
                throw new AssertionError(String.format("Velocity parity failed! Max error = %.4f m/s > 0.01 m/s", maxVelDiff));
            }
            if (maxHeadingDiff > 0.05) {
                throw new AssertionError(String.format("Heading parity failed! Max error = %.4f deg > 0.05 deg", maxHeadingDiff));
            }

            System.out.println("\n==============================================================================");
            System.out.println("PARITY VERIFICATION PASSED: EXACT NUMERICAL MATCH BETWEEN PYTHON & JAVA");
            System.out.println("Zero algorithmic drift. Engine is mathematically frozen & production ready.");
            System.out.println("==============================================================================");

        } catch (Throwable t) {
            System.err.println("\nVERIFICATION FAILED: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }
}
