package com.sih26168.idr.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 2D Spatial Road Network Index for fast candidate queries and map matching.
 * Direct Java port of src/map/geometry.py.
 */
public class RoadNetworkIndex {

    public static class Segment {
        public final double p1_e;
        public final double p1_n;
        public final double p2_e;
        public final double p2_n;
        public final double delta_e;
        public final double delta_n;
        public final double length_sq;
        public final double heading_rad;
        public final long way_id;

        public Segment(double p1_e, double p1_n, double p2_e, double p2_n, double heading_rad, long way_id) {
            this.p1_e = p1_e;
            this.p1_n = p1_n;
            this.p2_e = p2_e;
            this.p2_n = p2_n;
            this.delta_e = p2_e - p1_e;
            this.delta_n = p2_n - p1_n;
            this.length_sq = delta_e * delta_e + delta_n * delta_n;
            this.heading_rad = heading_rad;
            this.way_id = way_id;
        }
    }

    public static class Candidate {
        public final int segmentIdx;
        public final long wayId;
        public final double distanceM;
        public final double roadHeadingRad;
        public final double roadHeadingDeg;
        public final double projE;
        public final double projN;

        public Candidate(int segmentIdx, long wayId, double distanceM, double roadHeadingRad, double projE, double projN) {
            this.segmentIdx = segmentIdx;
            this.wayId = wayId;
            this.distanceM = distanceM;
            this.roadHeadingRad = roadHeadingRad;
            this.roadHeadingDeg = Math.toDegrees(roadHeadingRad);
            this.projE = projE;
            this.projN = projN;
        }
    }

    private final List<Segment> segments;

    public RoadNetworkIndex(List<Segment> segments) {
        this.segments = segments != null ? segments : new ArrayList<>();
    }

    public int getNumSegments() {
        return segments.size();
    }

    public List<Segment> getSegments() {
        return segments;
    }

    public static double wrapAngleRad(double angle) {
        double twoPi = 2.0 * Math.PI;
        double a = (angle + Math.PI) % twoPi;
        if (a < 0.0) a += twoPi;
        return a - Math.PI;
    }

    /**
     * Queries road candidates within radius_m and heading_gate_rad.
     */
    public List<Candidate> queryCandidates(double e, double n, double radiusM,
                                           Double vehHeadingRad, Double headingGateRad) {
        List<Candidate> candidates = new ArrayList<>();
        double rSq = radiusM * radiusM;

        for (int i = 0; i < segments.size(); i++) {
            Segment seg = segments.get(i);
            double ap_e = e - seg.p1_e;
            double ap_n = n - seg.p1_n;

            double dot = ap_e * seg.delta_e + ap_n * seg.delta_n;
            double t = Math.max(0.0, Math.min(1.0, dot / (seg.length_sq + 1e-12)));

            double projE = seg.p1_e + t * seg.delta_e;
            double projN = seg.p1_n + t * seg.delta_n;

            double diffE = e - projE;
            double diffN = n - projN;
            double distSq = diffE * diffE + diffN * diffN;

            if (distSq <= rSq) {
                if (vehHeadingRad != null && headingGateRad != null) {
                    double deltaHeading = wrapAngleRad(vehHeadingRad - seg.heading_rad);
                    if (Math.abs(deltaHeading) > headingGateRad) {
                        continue;
                    }
                }
                candidates.add(new Candidate(i, seg.way_id, Math.sqrt(distSq), seg.heading_rad, projE, projN));
            }
        }

        Collections.sort(candidates, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate o1, Candidate o2) {
                return Double.compare(o1.distanceM, o2.distanceM);
            }
        });

        return candidates;
    }
}
