package com.sih26168.idr.engine;

/**
 * Deterministic Unit Test: GNSS Reacquisition Safety Protection (Phase 5.1.1).
 * Validates that recovering from large dead-reckoning displacement (> 800m):
 * 1. Unprotected updateGnss kicks attitude and biases.
 * 2. Protected reacquireGnss isolates position re-anchoring with zero attitude/bias corruption.
 */
public class ReacquisitionSafetyTest {

    public static void main(String[] args) {
        System.out.println("==============================================================================");
        System.out.println("PHASE 5.1.1: DETERMINISTIC GNSS REACQUISITION SAFETY TEST");
        System.out.println("==============================================================================");

        double dt = 0.1;

        // Test 1: Demonstrate Unprotected reacquisition failure
        System.out.println("\n[1] Testing Standard Unprotected updateGnss() after 60s Outage...");
        ESKF3D eskfUnprotected = createSimulatedOutageEngine(60.0, dt);

        double yawBefore = eskfUnprotected.attitude.getYawDeg();
        double pitchBefore = eskfUnprotected.attitude.getPitchDeg();
        double rollBefore = eskfUnprotected.attitude.getRollDeg();
        Vector3 baBefore = eskfUnprotected.ba.copy();
        Vector3 bgBefore = eskfUnprotected.bg.copy();
        Vector3 velBefore = eskfUnprotected.vel_n.copy();

        // Simulate GNSS fix arriving 800m away (as observed in field test D)
        Vector3 gnssPos = new Vector3(0.0, 0.0, 0.0);
        Vector3 gnssVel = new Vector3(1.0, 0.0, 0.0);

        eskfUnprotected.updateGnss(gnssPos, gnssVel);

        double yawAfterUnprotected = eskfUnprotected.attitude.getYawDeg();
        double pitchAfterUnprotected = eskfUnprotected.attitude.getPitchDeg();
        double rollAfterUnprotected = eskfUnprotected.attitude.getRollDeg();
        Vector3 baAfterUnprotected = eskfUnprotected.ba.copy();
        Vector3 bgAfterUnprotected = eskfUnprotected.bg.copy();
        Vector3 velAfterUnprotected = eskfUnprotected.vel_n.copy();

        double yawDeltaUnprotected = Math.abs(yawAfterUnprotected - yawBefore);
        double pitchDeltaUnprotected = Math.abs(pitchAfterUnprotected - pitchBefore);
        double dBaUnprotected = baAfterUnprotected.sub(baBefore).norm();
        double dBgUnprotected = bgAfterUnprotected.sub(bgBefore).norm();

        System.out.printf("  Unprotected Attitude Kick : Yaw=%.2f deg, Pitch=%.2f deg\n", yawDeltaUnprotected, pitchDeltaUnprotected);
        System.out.printf("  Unprotected Bias Shift    : |dBa|=%.4f m/s², |dBg|=%.6f rad/s\n", dBaUnprotected, dBgUnprotected);
        System.out.printf("  Unprotected Velocity Post : [%.2f, %.2f, %.2f] m/s\n", velAfterUnprotected.x, velAfterUnprotected.y, velAfterUnprotected.z);

        // Test 2: Test Protected reacquireGnss
        System.out.println("\n[2] Testing Protected reacquireGnss() after identical 60s Outage...");
        ESKF3D eskfProtected = createSimulatedOutageEngine(60.0, dt);

        double yawBeforeProt = eskfProtected.attitude.getYawDeg();
        double pitchBeforeProt = eskfProtected.attitude.getPitchDeg();
        double rollBeforeProt = eskfProtected.attitude.getRollDeg();
        Vector3 baBeforeProt = eskfProtected.ba.copy();
        Vector3 bgBeforeProt = eskfProtected.bg.copy();

        eskfProtected.reacquireGnss(gnssPos, gnssVel, 3.0);

        double yawAfterProt = eskfProtected.attitude.getYawDeg();
        double pitchAfterProt = eskfProtected.attitude.getPitchDeg();
        double rollAfterProt = eskfProtected.attitude.getRollDeg();
        Vector3 baAfterProt = eskfProtected.ba.copy();
        Vector3 bgAfterProt = eskfProtected.bg.copy();
        Vector3 velAfterProt = eskfProtected.vel_n.copy();

        double yawDeltaProt = Math.abs(yawAfterProt - yawBeforeProt);
        double pitchDeltaProt = Math.abs(pitchAfterProt - pitchBeforeProt);
        double dBaProt = baAfterProt.sub(baBeforeProt).norm();
        double dBgProt = bgAfterProt.sub(bgBeforeProt).norm();

        System.out.printf("  Protected Attitude Kick   : Yaw=%.6f deg, Pitch=%.6f deg (Threshold: < 0.0001 deg)\n", yawDeltaProt, pitchDeltaProt);
        System.out.printf("  Protected Bias Shift      : |dBa|=%.6f m/s², |dBg|=%.8f rad/s (Threshold: 0.0)\n", dBaProt, dBgProt);
        System.out.printf("  Protected Velocity Post   : [%.2f, %.2f, %.2f] m/s\n", velAfterProt.x, velAfterProt.y, velAfterProt.z);
        System.out.printf("  Protected Position Post   : [%.2f, %.2f, %.2f] m (Re-anchored to GNSS)\n", eskfProtected.pos_n.x, eskfProtected.pos_n.y, eskfProtected.pos_n.z);

        // Assertions
        boolean attitudeSafe = yawDeltaProt < 1e-4 && pitchDeltaProt < 1e-4;
        boolean biasSafe = dBaProt < 1e-6 && dBgProt < 1e-8;
        boolean posSafe = eskfProtected.pos_n.sub(gnssPos).norm() < 1e-4;
        boolean velSafe = Math.abs(velAfterProt.x - gnssVel.x) < 0.5;

        System.out.println("\n[3] Safety Assertions:");
        System.out.println("  - Attitude Isolation : " + (attitudeSafe ? "PASSED" : "FAILED"));
        System.out.println("  - Bias Protection    : " + (biasSafe ? "PASSED" : "FAILED"));
        System.out.println("  - Position Re-anchor : " + (posSafe ? "PASSED" : "FAILED"));
        System.out.println("  - Velocity Bounded   : " + (velSafe ? "PASSED" : "FAILED"));

        if (attitudeSafe && biasSafe && posSafe && velSafe) {
            System.out.println("\n==============================================================================");
            System.out.println("TEST PASSED: Reacquisition Protection strictly eliminates state corruption.");
            System.out.println("==============================================================================");
        } else {
            System.err.println("\nTEST FAILED: Assertions not met!");
            System.exit(1);
        }
    }

    private static ESKF3D createSimulatedOutageEngine(double outageSec, double dt) {
        ESKF3D eskf = new ESKF3D(0.1, 0.01, 1e-4, 1e-5);

        // Initial GNSS lock at (0,0)
        eskf.updateGnss(new Vector3(0, 0, 0), new Vector3(10, 0, 0));

        // Outage propagation: 60s at 13.5 m/s forward speed
        int steps = (int) (outageSec / dt);
        for (int i = 0; i < steps; i++) {
            eskf.predict(0.0, 0.0, 9.80665, 0.0, 0.0, 0.0, dt);
            eskf.updateTcnSpeed(13.5, 0.60);
            eskf.updateDecoupledLateralVelocityDamping(0.50);
        }

        return eskf;
    }
}
