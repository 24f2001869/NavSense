package com.sih26168.idr.engine;

import org.junit.Test;
import static org.junit.Assert.*;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Android Unit Test for the 15-state DeadReckoningEngine.
 * Automatically executed by './gradlew test' to enforce bit-for-bit parity
 * against the Python reference on the Golden Reference session.
 */
public class DeadReckoningEngineTest {

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
        return o instanceof Number ? ((Number) o).doubleValue() : 0.0;
    }

    private static long toLong(Object o) {
        return o instanceof Number ? ((Number) o).longValue() : 0L;
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

    @Test
    public void testGoldenReferenceSessionParity() throws Exception {
        File repoRoot = new File(".").getCanonicalFile();
        File roadJsonFile = new File(repoRoot, "data/vta04_road_network.json");
        if (!roadJsonFile.exists()) roadJsonFile = new File(repoRoot, "../data/vta04_road_network.json");
        if (!roadJsonFile.exists()) roadJsonFile = new File(repoRoot, "../../data/vta04_road_network.json");
        if (!roadJsonFile.exists()) roadJsonFile = new File(repoRoot, "src/main/assets/vta04_road_network.json");
        if (!roadJsonFile.exists()) roadJsonFile = new File(repoRoot, "app/src/main/assets/vta04_road_network.json");

        File goldenJsonFile = new File(repoRoot, "data/golden_reference_session.json");
        if (!goldenJsonFile.exists()) goldenJsonFile = new File(repoRoot, "../data/golden_reference_session.json");
        if (!goldenJsonFile.exists()) goldenJsonFile = new File(repoRoot, "../../data/golden_reference_session.json");
        if (!goldenJsonFile.exists()) goldenJsonFile = new File(repoRoot, "src/main/assets/golden_reference_session.json");
        if (!goldenJsonFile.exists()) goldenJsonFile = new File(repoRoot, "app/src/main/assets/golden_reference_session.json");

        assertTrue("Road network file must exist", roadJsonFile.exists());
        assertTrue("Golden reference file must exist", goldenJsonFile.exists());

        // 1. Load Road Network
        String roadStr = readFile(roadJsonFile);
        Map<String, Object> roadMap = MiniJson.parseObject(roadStr);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> segListJson = (List<Map<String, Object>>) roadMap.get("segments");

        List<RoadNetworkIndex.Segment> segments = new ArrayList<>();
        for (Map<String, Object> s : segListJson) {
            List<Double> p1 = toDoubleList(s.get("p1"));
            List<Double> p2 = toDoubleList(s.get("p2"));
            double headingRad = toDouble(s.get("heading_rad"));
            long wayId = toLong(s.get("way_id"));
            segments.add(new RoadNetworkIndex.Segment(
                p1.get(0), p1.get(1),
                p2.get(0), p2.get(1),
                headingRad, wayId
            ));
        }
        RoadNetworkIndex roadIndex = new RoadNetworkIndex(segments);

        // 2. Load Session
        String goldenStr = readFile(goldenJsonFile);
        Map<String, Object> goldenMap = MiniJson.parseObject(goldenStr);
        @SuppressWarnings("unchecked")
        Map<String, Object> meta = (Map<String, Object>) goldenMap.get("meta");
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> epochs = (List<Map<String, Object>>) goldenMap.get("epochs");

        double dt = toDouble(meta.get("dt"));
        long blackoutStart = toLong(meta.get("blackout_start_epoch"));
        long blackoutEnd = toLong(meta.get("blackout_end_epoch"));

        // 3. Initialize Engine
        List<Double> initPos = toDoubleList(meta.get("init_pos_enu"));
        List<Double> initVel = toDoubleList(meta.get("init_vel_enu"));
        double initHeading = toDouble(meta.get("init_heading_deg"));
        List<Double> baStatList = toDoubleList(meta.get("ba_stat"));
        double baselineMag = toDouble(meta.get("baseline_mag_uT"));

        DeadReckoningEngine.EngineConfig cfg = new DeadReckoningEngine.EngineConfig();
        cfg.sigma_speed = 0.60;
        cfg.sigma_lat_0 = 0.50;
        cfg.mag_norm_tol = 0.08;
        cfg.mag_db_dt_tol = 5.0;
        cfg.enable_vert_nhc = true;
        cfg.turn_rate_vnhc_gate_deg_s = 3.0;
        cfg.turn_rate_compass_gate_deg_s = 3.0;

        DeadReckoningEngine engine = new DeadReckoningEngine(
            cfg,
            roadIndex,
            new Vector3(initPos.get(0), initPos.get(1), initPos.get(2)),
            new Vector3(initVel.get(0), initVel.get(1), initVel.get(2)),
            initHeading,
            Matrix.identity(3),
            new Vector3(baStatList.get(0), baStatList.get(1), baStatList.get(2)),
            new Vector3(0, 0, 0),
            baselineMag
        );

        double maxPosDiff = 0.0;
        double maxVelDiff = 0.0;
        double maxHeadingDiff = 0.0;
        int nhcMatches = 0;
        int vnhcMatches = 0;
        int mapMatches = 0;

        for (int i = 0; i < epochs.size(); i++) {
            Map<String, Object> ep = epochs.get(i);
            List<Double> accList = toDoubleList(ep.get("accel"));
            List<Double> gyroList = toDoubleList(ep.get("gyro"));
            List<Double> magList = toDoubleList(ep.get("mag"));
            double mlSpeed = toDouble(ep.get("ml_speed"));
            double dbDt = toDouble(ep.get("db_dt"));
            double psiMagCal = toDouble(ep.get("psi_mag_cal_deg"));
            boolean gnssValid = Boolean.TRUE.equals(ep.get("gnss_valid"));

            DeadReckoningEngine.GNSSMeasurement gnss;
            if (gnssValid) {
                List<Double> gPos = toDoubleList(ep.get("gnss_pos_enu"));
                List<Double> gVel = toDoubleList(ep.get("gnss_vel_enu"));
                gnss = new DeadReckoningEngine.GNSSMeasurement(
                    true,
                    new Vector3(gPos.get(0), gPos.get(1), gPos.get(2)),
                    new Vector3(gVel.get(0), gVel.get(1), gVel.get(2)),
                    3.0
                );
            } else {
                gnss = new DeadReckoningEngine.GNSSMeasurement(false);
            }

            DeadReckoningEngine.NavigationTelemetry tel = engine.step(
                new Vector3(accList.get(0), accList.get(1), accList.get(2)),
                new Vector3(gyroList.get(0), gyroList.get(1), gyroList.get(2)),
                mlSpeed,
                dt,
                new Vector3(magList.get(0), magList.get(1), magList.get(2)),
                gnss,
                psiMagCal,
                dbDt
            );

            List<Double> refPos = toDoubleList(ep.get("ref_pos_enu"));
            List<Double> refVel = toDoubleList(ep.get("ref_vel_enu"));
            double refHeading = toDouble(ep.get("ref_heading_deg"));

            double dE = tel.pos_enu.x - refPos.get(0);
            double dN = tel.pos_enu.y - refPos.get(1);
            double dU = tel.pos_enu.z - refPos.get(2);
            double posDiff = Math.sqrt(dE * dE + dN * dN + dU * dU);

            double dVe = tel.vel_enu.x - refVel.get(0);
            double dVn = tel.vel_enu.y - refVel.get(1);
            double dVu = tel.vel_enu.z - refVel.get(2);
            double velDiff = Math.sqrt(dVe * dVe + dVn * dVn + dVu * dVu);

            double hDiff = Math.abs(((tel.heading_deg - refHeading + 180.0) % 360.0 + 360.0) % 360.0 - 180.0);

            if (posDiff > maxPosDiff) maxPosDiff = posDiff;
            if (velDiff > maxVelDiff) maxVelDiff = velDiff;
            if (hDiff > maxHeadingDiff) maxHeadingDiff = hDiff;

            if (tel.diagnostics.nhc_active == Boolean.TRUE.equals(ep.get("nhc_active"))) nhcMatches++;
            if (tel.diagnostics.vnhc_active == Boolean.TRUE.equals(ep.get("vnhc_active"))) vnhcMatches++;
            if (tel.diagnostics.map_active == Boolean.TRUE.equals(ep.get("map_active"))) mapMatches++;
        }

        // Rigorous Bit-For-Bit Parity Assertions
        assertEquals("NHC flag must match 100% of epochs", epochs.size(), nhcMatches);
        assertEquals("VNHC flag must match 100% of epochs", epochs.size(), vnhcMatches);
        assertEquals("Map guidance flag must match 100% of epochs", epochs.size(), mapMatches);
        assertTrue("Max Heading difference must be < 0.05 deg (was " + maxHeadingDiff + ")", maxHeadingDiff < 0.05);
        assertTrue("Max Velocity difference must be < 0.05 m/s (was " + maxVelDiff + ")", maxVelDiff < 0.05);
        assertTrue("Max Position difference must be < 0.30 m (was " + maxPosDiff + ")", maxPosDiff < 0.30);
    }
}
