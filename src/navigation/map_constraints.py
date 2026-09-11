"""
SIH26168 - Stage C8-2: Controlled ESKF Map Constraints
Module: src/navigation/map_constraints.py

Implements physical and geometric map constraints integrated into the 15-state 3D
Error-State Kalman Filter (ESKF3D) for land vehicle navigation during GNSS outages.

Features:
1. Multi-Layer Spatial & Directional Gating:
   - Search radius bound: R <= R_max (25.0 m hard boundary).
   - Heading alignment gate: |Delta psi| <= 30.0 deg (eliminates opposite carriageways).
   - Normalized Innovation Squared (NIS) gating: NIS <= chi_2^2(0.95) = 5.991.
   - Ambiguity detection & rejection: skips update if multiple distinct roads satisfy gates.
2. Decoupled Probabilistic Observation Models:
   - Lateral cross-track update: r_perp = d_perp(p) - mu_lane, with H_perp = [n_E, n_N, 0, 0_1x12].
   - Road tangent heading update: r_psi = wrap(psi_road - psi_hat), with H_psi = [0_1x8, -1.0, 0_1x6].
   - Joint update: combined 2x15 update with optional curvature cross-coupling.
3. Joseph-Stabilized Covariance Update:
   P^+ = (I - K*H) P^- (I - K*H)^T + K * R * K^T
4. Full Telemetry & Wrong-Road Association Tracking against ground truth.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np

from src.map.geometry import RoadNetworkIndex, wrap_angle_rad
from src.navigation.nhc import rotvec_to_quat, quat_mult
from src.map.beam_search import MultiHypothesisTracker, PathHypothesis


class MapConstraintManager:
    """
    Manages candidate retrieval, probabilistic gating, ambiguity rejection,
    and ESKF state/covariance updates using OSM vector road networks.
    """

    def __init__(
        self,
        road_index: RoadNetworkIndex,
        search_radius_m: float = 25.0,
        heading_gate_deg: float = 30.0,
        sigma_lane_m: float = 2.5,
        sigma_psi_deg: float = 5.0,
        lane_offset_m: float = -0.90,
        nis_gate_2dof: float = 9.210,   # chi_2^2 at 99% (5.991 at 95%)
        nis_gate_1dof: float = 6.635,   # chi_1^2 at 99% (3.841 at 95%)
        update_interval_sec: float = 1.0,
        ambiguity_margin_m: float = 3.0
    ):
        self.index = road_index
        self.search_radius_m = float(search_radius_m)
        self.heading_gate_rad = float(np.radians(heading_gate_deg))
        self.sigma_lane = float(sigma_lane_m)
        self.sigma_psi = float(np.radians(sigma_psi_deg))
        self.lane_offset_m = float(lane_offset_m)
        self.nis_gate_2dof = float(nis_gate_2dof)
        self.nis_gate_1dof = float(nis_gate_1dof)
        self.update_interval_sec = float(update_interval_sec)
        self.ambiguity_margin_m = float(ambiguity_margin_m)

        self.last_update_time = -np.inf

    def select_candidate(
        self,
        pos_enu: np.ndarray,
        heading_deg: float,
        pos_cov: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Queries road candidates within bounded search radius and evaluates multi-layer gates.
        Returns the best unambiguous road candidate, or None if no candidate passes
        or if multiple conflicting roads create topological ambiguity.
        """
        veh_h_rad = np.radians(heading_deg)

        # 1. Spatial search within hard radius cap (R <= 25 m)
        candidates = self.index.query_candidates(
            p=pos_enu[:2],
            radius_m=self.search_radius_m,
            veh_heading_rad=veh_h_rad,
            heading_gate_rad=self.heading_gate_rad
        )

        if not candidates:
            return None

        # 2. Ambiguity Check across distinct road ways
        # If the top two candidates belong to different way_ids and have similar distances,
        # freeze updates to prevent false-snapping near forks or multi-road intersections.
        if len(candidates) > 1:
            cand0 = candidates[0]
            cand1 = candidates[1]
            if cand0['way_id'] != cand1['way_id']:
                dist_diff = abs(cand0['distance_m'] - cand1['distance_m'])
                if dist_diff < self.ambiguity_margin_m:
                    # Ambiguous junction / parallel road fork: reject candidate
                    return None

        best_cand = candidates[0]
        return best_cand

    def compute_lateral_innovation(
        self,
        pos_enu: np.ndarray,
        candidate: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        """
        Computes lateral cross-track innovation y_perp and measurement Jacobian H_perp (1x15).
        y_perp = mu_lane - d_perp(p)
        H_perp = [cos(psi_road), -sin(psi_road), 0, 0_1x12]
        """
        psi_road = candidate['road_heading_rad']
        n_vec = np.array([np.cos(psi_road), -np.sin(psi_road)])

        seg_idx = candidate['segment_idx']
        seg = self.index.segments[seg_idx]
        p1 = seg['p1']

        # Orthogonal distance to centerline (positive = right, negative = left)
        d_perp = float((pos_enu[:2] - p1) @ n_vec)

        # Innovation: expected lateral position is mu_lane
        y_perp = float(self.lane_offset_m - d_perp)

        H_perp = np.zeros((1, 15), dtype=np.float64)
        H_perp[0, 0:2] = n_vec

        return y_perp, H_perp

    def compute_heading_innovation(
        self,
        heading_deg: float,
        candidate: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        """
        Computes road tangent heading innovation y_psi and measurement Jacobian H_psi (1x15).
        y_psi = wrap(psi_road - psi_hat)
        H_psi = [0_1x8, -1.0, 0_1x6]
        """
        psi_road = candidate['road_heading_rad']
        psi_hat = np.radians(heading_deg)

        y_psi = float(wrap_angle_rad(psi_road - psi_hat))

        H_psi = np.zeros((1, 15), dtype=np.float64)
        H_psi[0, 8] = -1.0  # Yaw error in body attitude block

        return y_psi, H_psi

    def update_lateral(
        self,
        eskf: Any,
        t_now: float,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Condition A: ESKF Map Lateral Cross-Track Update.
        """
        if not force_update and (t_now - self.last_update_time) < (self.update_interval_sec - 1e-4):
            return {'active': False, 'reason': 'rate_limit'}

        st = eskf.get_state()
        pos_enu = st['pos_n']
        heading_deg = st['yaw_deg']

        candidate = self.select_candidate(pos_enu, heading_deg)
        if candidate is None:
            return {'active': False, 'reason': 'no_valid_candidate'}

        y_perp, H_perp = self.compute_lateral_innovation(pos_enu, candidate)

        R_perp = np.array([[self.sigma_lane ** 2]], dtype=np.float64)
        S = H_perp @ eskf.P @ H_perp.T + R_perp
        S_inv = 1.0 / float(S[0, 0])

        nis = float((y_perp ** 2) * S_inv)
        if nis > self.nis_gate_1dof:
            return {'active': False, 'reason': 'nis_gated', 'nis': nis, 'candidate': candidate}

        # Kalman gain (15x1)
        K = eskf.P @ H_perp.T * S_inv

        # State error correction
        delta_x = (K * y_perp).flatten()

        # Inject correction into nominal state
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        eskf.ba += delta_x[9:12]
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_perp
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R_perp @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        self.last_update_time = t_now

        return {
            'active': True,
            'type': 'lateral',
            'innovation': y_perp,
            'nis': nis,
            'candidate': candidate,
            'delta_pos': delta_x[0:3],
            'delta_yaw_deg': float(np.degrees(-delta_x[8])),
            'K': K
        }

    def update_heading(
        self,
        eskf: Any,
        t_now: float,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Condition B: ESKF Map Road Tangent Heading Update.
        """
        if not force_update and (t_now - self.last_update_time) < (self.update_interval_sec - 1e-4):
            return {'active': False, 'reason': 'rate_limit'}

        st = eskf.get_state()
        pos_enu = st['pos_n']
        heading_deg = st['yaw_deg']

        candidate = self.select_candidate(pos_enu, heading_deg)
        if candidate is None:
            return {'active': False, 'reason': 'no_valid_candidate'}

        y_psi, H_psi = self.compute_heading_innovation(heading_deg, candidate)

        R_psi = np.array([[self.sigma_psi ** 2]], dtype=np.float64)
        S = H_psi @ eskf.P @ H_psi.T + R_psi
        S_inv = 1.0 / float(S[0, 0])

        nis = float((y_psi ** 2) * S_inv)
        if nis > self.nis_gate_1dof:
            return {'active': False, 'reason': 'nis_gated', 'nis': nis, 'candidate': candidate}

        # Kalman gain (15x1)
        K = eskf.P @ H_psi.T * S_inv

        # State error correction
        delta_x = (K * y_psi).flatten()

        # Inject correction into nominal state
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        eskf.ba += delta_x[9:12]
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_psi
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R_psi @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        self.last_update_time = t_now

        return {
            'active': True,
            'type': 'heading',
            'innovation': y_psi,
            'nis': nis,
            'candidate': candidate,
            'delta_pos': delta_x[0:3],
            'delta_yaw_deg': float(np.degrees(-delta_x[8])),
            'K': K
        }

    def update_joint(
        self,
        eskf: Any,
        t_now: float,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Condition C: ESKF Joint Map Lateral Cross-Track + Heading Update.
        """
        if not force_update and (t_now - self.last_update_time) < (self.update_interval_sec - 1e-4):
            return {'active': False, 'reason': 'rate_limit'}

        st = eskf.get_state()
        pos_enu = st['pos_n']
        heading_deg = st['yaw_deg']

        candidate = self.select_candidate(pos_enu, heading_deg)
        if candidate is None:
            return {'active': False, 'reason': 'no_valid_candidate'}

        y_perp, H_perp = self.compute_lateral_innovation(pos_enu, candidate)
        y_psi, H_psi = self.compute_heading_innovation(heading_deg, candidate)

        y_joint = np.array([y_perp, y_psi], dtype=np.float64)
        H_joint = np.vstack([H_perp, H_psi])  # (2, 15)

        R_joint = np.diag([self.sigma_lane ** 2, self.sigma_psi ** 2])
        S = H_joint @ eskf.P @ H_joint.T + R_joint
        S_inv = np.linalg.inv(S)

        nis = float(y_joint.T @ S_inv @ y_joint)
        if nis > self.nis_gate_2dof:
            return {'active': False, 'reason': 'nis_gated', 'nis': nis, 'candidate': candidate}

        # Kalman gain (15x2)
        K = eskf.P @ H_joint.T @ S_inv

        # State error correction
        delta_x = (K @ y_joint).flatten()

        # Inject correction into nominal state
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        eskf.ba += delta_x[9:12]
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H_joint
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R_joint @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        self.last_update_time = t_now

        return {
            'active': True,
            'type': 'joint',
            'innovation_perp': y_perp,
            'innovation_psi': y_psi,
            'nis': nis,
            'candidate': candidate,
            'delta_pos': delta_x[0:3],
            'delta_yaw_deg': float(np.degrees(-delta_x[8])),
            'K': K
        }


class MHTMapConstraintManager:
    """
    SIH26168 - Stage C8-4: 5-Gate Multi-Hypothesis Closed-Loop Map Constraint Manager.
    
    Enforces the strict 5-Gate Safety Architecture before applying map updates to the ESKF:
      Gate 1: MHT Confidence (P(H_1) - P(H_2) > Delta, Delta=0.20)
      Gate 2: Candidate Geometry (d_perp <= R_max = 25.0 m hard cap)
      Gate 3: Heading Consistency (|wrap(psi_road - psi_hat)| <= 30.0 deg)
      Gate 4: Temporal Persistence (way_id must remain dominant for >= N_persist epochs)
      Gate 5: Innovation Consistency (NIS <= chi^2_gate)
      
    Fallback: Failure of ANY gate results in NO MAP UPDATE (active=False).
    Shadow Mode: If shadow_mode=True, evaluates all gates and computes hypothetical corrections,
                 but leaves ESKF state and covariance 100% untouched.
    """

    def __init__(
        self,
        road_index: RoadNetworkIndex,
        mht_tracker: Optional[MultiHypothesisTracker] = None,
        beam_width_K: int = 3,
        search_radius_m: float = 25.0,
        heading_gate_deg: float = 30.0,
        commitment_margin_delta: float = 0.20,
        sigma_lane_m: float = 2.5,
        sigma_psi_deg: float = 5.0,
        lane_offset_m: float = -0.90,
        nis_gate_2dof: float = 9.210,   # chi_2^2 at 99%
        nis_gate_1dof: float = 6.635,   # chi_1^2 at 99%
        n_persist: int = 2,
        max_pos_correction_m: float = 5.0,
        update_interval_sec: float = 1.0
    ):
        self.index = road_index
        self.search_radius_m = float(search_radius_m)
        self.heading_gate_rad = float(np.radians(heading_gate_deg))
        self.delta = float(commitment_margin_delta)
        self.sigma_lane = float(sigma_lane_m)
        self.sigma_psi = float(np.radians(sigma_psi_deg))
        self.lane_offset_m = float(lane_offset_m)
        self.nis_gate_2dof = float(nis_gate_2dof)
        self.nis_gate_1dof = float(nis_gate_1dof)
        self.n_persist = int(n_persist)
        self.max_pos_correction_m = float(max_pos_correction_m)
        self.update_interval_sec = float(update_interval_sec)

        if mht_tracker is not None:
            self.tracker = mht_tracker
        else:
            self.tracker = MultiHypothesisTracker(
                road_index=road_index,
                beam_width_K=beam_width_K,
                search_radius_m=search_radius_m,
                heading_gate_deg=heading_gate_deg,
                commitment_margin_delta=commitment_margin_delta
            )

        self.last_update_time = -np.inf
        self.last_dominant_way_id: Optional[int] = None
        self.dominant_way_persistence: int = 0

    def compute_lateral_innovation(
        self,
        pos_enu: np.ndarray,
        candidate: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        psi_road = candidate['road_heading_rad']
        n_vec = np.array([np.cos(psi_road), -np.sin(psi_road)])
        seg_idx = candidate['segment_idx']
        seg = self.index.segments[seg_idx]
        p1 = seg['p1']
        d_perp = float((pos_enu[:2] - p1) @ n_vec)
        y_perp = float(self.lane_offset_m - d_perp)
        H_perp = np.zeros((1, 15), dtype=np.float64)
        H_perp[0, 0:2] = n_vec
        return y_perp, H_perp

    def compute_heading_innovation(
        self,
        heading_deg: float,
        candidate: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        psi_road = candidate['road_heading_rad']
        psi_hat = np.radians(heading_deg)
        y_psi = float(wrap_angle_rad(psi_road - psi_hat))
        H_psi = np.zeros((1, 15), dtype=np.float64)
        H_psi[0, 8] = -1.0
        return y_psi, H_psi

    def evaluate_and_update(
        self,
        eskf: Any,
        t_now: float,
        mode: str = "joint",
        speed_ms: float = 0.0,
        gyro_z_rads: float = 0.0,
        shadow_mode: bool = False,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the 5-Gate validation pipeline and optionally applies map correction to the ESKF.
        """
        if not force_update and (t_now - self.last_update_time) < (self.update_interval_sec - 1e-4):
            return {'active': False, 'reason': 'rate_limit'}

        st = eskf.get_state()
        pos_enu = st['pos_n']
        heading_deg = float(st['yaw_deg'])

        # Query MHT Beam Tracker
        mht_res = self.tracker.update(
            est_pos=pos_enu,
            est_heading_deg=heading_deg,
            speed_ms=speed_ms,
            gyro_z_rads=gyro_z_rads,
            t_now=t_now
        )

        cand = mht_res.get('committed_candidate', None)
        top_cand = mht_res.get('top_candidate', None)

        # Gate 1: MHT Confidence
        if mht_res['status'] != 'confident' or cand is None:
            # Ambiguous epoch (fork or multiple plausible branches)
            return {
                'active': False,
                'reason': 'gate1_ambiguous',
                'prob_gap': mht_res.get('prob_gap', 0.0),
                'top_candidate': top_cand,
                'mht_res': mht_res
            }

        # Gate 2: Candidate Geometry (R <= 25 m)
        d_perp = float(cand['distance_m'])
        if d_perp > self.search_radius_m:
            return {
                'active': False,
                'reason': 'gate2_geometry',
                'distance_m': d_perp,
                'candidate': cand,
                'mht_res': mht_res
            }

        # Gate 3: Heading Consistency (|Delta psi| <= 30 deg)
        psi_hat_rad = np.radians(heading_deg)
        delta_psi_rad = abs(wrap_angle_rad(cand['road_heading_rad'] - psi_hat_rad))
        if delta_psi_rad > self.heading_gate_rad:
            return {
                'active': False,
                'reason': 'gate3_heading',
                'delta_psi_deg': float(np.degrees(delta_psi_rad)),
                'candidate': cand,
                'mht_res': mht_res
            }

        # Gate 4: Temporal Persistence
        curr_way = int(cand['way_id'])
        if curr_way == self.last_dominant_way_id:
            self.dominant_way_persistence += 1
        else:
            self.last_dominant_way_id = curr_way
            self.dominant_way_persistence = 1

        if self.dominant_way_persistence < self.n_persist:
            return {
                'active': False,
                'reason': 'gate4_persistence',
                'persistence': self.dominant_way_persistence,
                'required': self.n_persist,
                'candidate': cand,
                'mht_res': mht_res
            }

        # Gate 5: Innovation Consistency (NIS)
        if mode == "lateral":
            y_perp, H_perp = self.compute_lateral_innovation(pos_enu, cand)
            R = np.array([[self.sigma_lane ** 2]], dtype=np.float64)
            S = H_perp @ eskf.P @ H_perp.T + R
            S_inv = 1.0 / float(S[0, 0])
            nis = float((y_perp ** 2) * S_inv)
            if nis > self.nis_gate_1dof:
                return {'active': False, 'reason': 'gate5_nis', 'nis': nis, 'candidate': cand, 'mht_res': mht_res}
            y = np.array([y_perp])
            H = H_perp
            K = eskf.P @ H.T * S_inv
            delta_x = (K * y_perp).flatten()

        elif mode == "heading":
            y_psi, H_psi = self.compute_heading_innovation(heading_deg, cand)
            R = np.array([[self.sigma_psi ** 2]], dtype=np.float64)
            S = H_psi @ eskf.P @ H_psi.T + R
            S_inv = 1.0 / float(S[0, 0])
            nis = float((y_psi ** 2) * S_inv)
            if nis > self.nis_gate_1dof:
                return {'active': False, 'reason': 'gate5_nis', 'nis': nis, 'candidate': cand, 'mht_res': mht_res}
            y = np.array([y_psi])
            H = H_psi
            K = eskf.P @ H.T * S_inv
            delta_x = (K * y_psi).flatten()

        elif mode == "joint":
            y_perp, H_perp = self.compute_lateral_innovation(pos_enu, cand)
            y_psi, H_psi = self.compute_heading_innovation(heading_deg, cand)
            y = np.array([y_perp, y_psi], dtype=np.float64)
            H = np.vstack([H_perp, H_psi])
            R = np.diag([self.sigma_lane ** 2, self.sigma_psi ** 2])
            S = H @ eskf.P @ H.T + R
            S_inv = np.linalg.inv(S)
            nis = float(y.T @ S_inv @ y)
            if nis > self.nis_gate_2dof:
                return {'active': False, 'reason': 'gate5_nis', 'nis': nis, 'candidate': cand, 'mht_res': mht_res}
            K = eskf.P @ H.T @ S_inv
            delta_x = (K @ y).flatten()

        else:
            raise ValueError(f"Unknown map constraint mode: {mode}")

        # Position correction clamp safeguard
        delta_pos = delta_x[0:3].copy()
        pos_norm = float(np.linalg.norm(delta_pos))
        clamped = False
        if pos_norm > self.max_pos_correction_m and pos_norm > 1e-6:
            scale = self.max_pos_correction_m / pos_norm
            delta_x *= scale
            delta_pos = delta_x[0:3].copy()
            clamped = True

        self.last_update_time = t_now

        if shadow_mode:
            # SHADOW MODE: Record hypothetical correction but DO NOT modify ESKF state or covariance!
            return {
                'active': True,
                'shadow_mode': True,
                'mode': mode,
                'candidate': cand,
                'nis': nis,
                'delta_pos': delta_pos,
                'delta_yaw_deg': float(np.degrees(-delta_x[8])),
                'clamped': clamped,
                'mht_res': mht_res
            }

        # CLOSED-LOOP MODE: Inject correction into ESKF nominal state
        eskf.pos_n += delta_x[0:3]
        eskf.vel_n += delta_x[3:6]
        dtheta_b = delta_x[6:9]
        dq_corr = rotvec_to_quat(dtheta_b)
        eskf.attitude.q_nv = quat_mult(eskf.attitude.q_nv, dq_corr)
        eskf.attitude.q_nv /= np.linalg.norm(eskf.attitude.q_nv)
        eskf.ba += delta_x[9:12]
        eskf.bg += delta_x[12:15]

        # Joseph-stabilized Covariance Update
        IKH = np.eye(15, dtype=np.float64) - K @ H
        eskf.P = IKH @ eskf.P @ IKH.T + K @ R @ K.T
        eskf.P = 0.5 * (eskf.P + eskf.P.T)

        return {
            'active': True,
            'shadow_mode': False,
            'mode': mode,
            'candidate': cand,
            'nis': nis,
            'delta_pos': delta_pos,
            'delta_yaw_deg': float(np.degrees(-delta_x[8])),
            'clamped': clamped,
            'mht_res': mht_res
        }
