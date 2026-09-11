package com.sih26168.idr.engine;

import java.util.List;

/**
 * Manages OSM vector road candidate selection, ambiguity rejection, and heading innovation.
 * Direct Java port of src/navigation/map_constraints.py.
 */
public class MapConstraintManager {

    public static class HeadingInnovation {
        public final double y_psi;
        public final Matrix H_psi; // 1x15

        public HeadingInnovation(double y_psi, Matrix H_psi) {
            this.y_psi = y_psi;
            this.H_psi = H_psi;
        }
    }

    private final RoadNetworkIndex index;
    public final double searchRadiusM;
    public final double headingGateRad;
    public final double sigmaPsi;
    public final double nisGate1dof;
    public final double updateIntervalSec;
    public final double ambiguityMarginM;

    public MapConstraintManager(
        RoadNetworkIndex roadIndex,
        double searchRadiusM,
        double headingGateDeg,
        double sigmaPsiDeg,
        double updateIntervalSec
    ) {
        this.index = roadIndex;
        this.searchRadiusM = searchRadiusM;
        this.headingGateRad = Math.toRadians(headingGateDeg);
        this.sigmaPsi = Math.toRadians(sigmaPsiDeg);
        this.nisGate1dof = 6.635; // chi_1^2 at 99%
        this.updateIntervalSec = updateIntervalSec;
        this.ambiguityMarginM = 3.0;
    }

    public RoadNetworkIndex.Candidate selectCandidate(double posE, double posN, double headingDeg) {
        if (index == null) return null;

        double vehHeadingRad = Math.toRadians(headingDeg);
        List<RoadNetworkIndex.Candidate> candidates = index.queryCandidates(
            posE, posN, searchRadiusM, vehHeadingRad, headingGateRad
        );

        if (candidates == null || candidates.isEmpty()) {
            return null;
        }

        // Ambiguity Check across distinct road ways
        if (candidates.size() > 1) {
            RoadNetworkIndex.Candidate cand0 = candidates.get(0);
            RoadNetworkIndex.Candidate cand1 = candidates.get(1);
            if (cand0.wayId != cand1.wayId) {
                double distDiff = Math.abs(cand0.distanceM - cand1.distanceM);
                if (distDiff < ambiguityMarginM) {
                    return null; // Ambiguous junction / parallel road fork
                }
            }
        }

        return candidates.get(0);
    }

    public HeadingInnovation computeHeadingInnovation(double headingDeg, RoadNetworkIndex.Candidate candidate) {
        double psiRoad = candidate.roadHeadingRad;
        double psiHat = Math.toRadians(headingDeg);
        double y_psi = RoadNetworkIndex.wrapAngleRad(psiRoad - psiHat);

        Matrix H_psi = new Matrix(1, 15);
        H_psi.set(0, 8, -1.0); // Yaw error in body attitude block

        return new HeadingInnovation(y_psi, H_psi);
    }
}
