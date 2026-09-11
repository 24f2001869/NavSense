"""
SIH26168 - Stage C8-3.2: Multi-Hypothesis Beam Search (MHT) for Topological Map Matching
Module: src/map/beam_search.py

Features:
- Path-level hypothesis representation (maintains full segment history, transition costs,
  and recursive accumulated likelihood scores).
- Directed graph branch expansion along topological road successors.
- Transparent score decomposition:
    S = w_d * S_d + w_psi * S_psi + w_trans * S_trans + w_kin * S_kin
- Gyro turn rate vs road curvature kinematic consistency.
- Delayed commitment logic:
    If P(H_1) - P(H_2) <= Delta, status is AMBIGUOUS (no premature snapping).
    If P(H_1) - P(H_2) > Delta, status is CONFIDENT.
- Bounded beam pruning to top K hypotheses (K in {1, 2, 3, 5}).
- Fork detection, tracking, and recovery telemetry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.map.geometry import RoadNetworkIndex, wrap_angle_rad


@dataclass
class PathHypothesis:
    """
    Represents a discrete path hypothesis through the OSM road network graph.
    """
    seg_idx: int
    way_id: int
    path_history: List[int] = field(default_factory=list)
    accumulated_score: float = 0.0
    last_d_perp: float = 0.0
    last_delta_psi: float = 0.0
    last_kinematic_err: float = 0.0
    last_trans_score: float = 1.0
    posterior_prob: float = 1.0
    is_fork_branch: bool = False
    fork_origin_time: float = 0.0

    def copy(self) -> 'PathHypothesis':
        return PathHypothesis(
            seg_idx=self.seg_idx,
            way_id=self.way_id,
            path_history=list(self.path_history),
            accumulated_score=self.accumulated_score,
            last_d_perp=self.last_d_perp,
            last_delta_psi=self.last_delta_psi,
            last_kinematic_err=self.last_kinematic_err,
            last_trans_score=self.last_trans_score,
            posterior_prob=self.posterior_prob,
            is_fork_branch=self.is_fork_branch,
            fork_origin_time=self.fork_origin_time
        )


class MultiHypothesisTracker:
    """
    Maintains a beam of up to K path hypotheses through the OSM road network.
    """

    def __init__(
        self,
        road_index: RoadNetworkIndex,
        beam_width_K: int = 3,
        search_radius_m: float = 25.0,
        heading_gate_deg: float = 30.0,
        commitment_margin_delta: float = 0.0,
        decay_gamma: float = 0.85,
        w_d: float = 0.35,
        w_psi: float = 0.25,
        w_trans: float = 0.20,
        w_kin: float = 0.20,
        sigma_d: float = 5.0,
        sigma_psi_deg: float = 15.0,
        sigma_omega_rads: float = 0.08,
        min_hypothesis_score: float = 0.25
    ):
        self.index = road_index
        self.K = max(1, int(beam_width_K))
        self.search_radius_m = float(search_radius_m)
        self.heading_gate_rad = float(np.radians(heading_gate_deg))
        self.delta = float(commitment_margin_delta)
        self.decay_gamma = float(decay_gamma)

        # Transparent score weights
        self.w_d = float(w_d)
        self.w_psi = float(w_psi)
        self.w_trans = float(w_trans)
        self.w_kin = float(w_kin)

        self.sigma_d = float(sigma_d)
        self.sigma_psi_rad = float(np.radians(sigma_psi_deg))
        self.sigma_omega = float(sigma_omega_rads)
        self.min_hypothesis_score = float(min_hypothesis_score)

        self.beam: List[PathHypothesis] = []
        self.last_update_time: float = -np.inf

    def reset(self):
        """Clears all active hypotheses."""
        self.beam = []
        self.last_update_time = -np.inf

    def compute_single_epoch_score(
        self,
        cand: Dict[str, Any],
        prev_seg_idx: Optional[int],
        speed_ms: float,
        gyro_z_rads: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes transparent normalized score components for a candidate segment.
        """
        d = float(cand['distance_m'])
        delta_psi = float(cand['delta_heading_rad']) if cand['delta_heading_rad'] is not None else 0.0
        seg_idx = int(cand['segment_idx'])

        # 1. Distance likelihood S_d in [0, 1]
        s_d = float(np.exp(-0.5 * (d / self.sigma_d) ** 2))

        # 2. Heading alignment likelihood S_psi in [0, 1]
        s_psi = float(np.exp(-0.5 * (delta_psi / self.sigma_psi_rad) ** 2))

        # 3. Topological transition score S_trans in [0, 1]
        s_trans = float(self.index.topological_transition_score(prev_seg_idx, seg_idx))

        # 4. Kinematic consistency S_kin in [0, 1]
        # Compare measured yaw rate omega_z with segment curvature kappa * speed
        s_kin = 1.0
        if prev_seg_idx is not None and speed_ms > 1.0:
            h_curr = float(self.index.headings_rad[seg_idx])
            h_prev = float(self.index.headings_rad[prev_seg_idx])
            d_psi = float(wrap_angle_rad(h_curr - h_prev))
            # Arc length approximate
            seg_len = float(self.index.segments[seg_idx]['length'])
            kappa = d_psi / max(2.0, seg_len)
            omega_expected = kappa * speed_ms
            omega_err = abs(gyro_z_rads - omega_expected)
            s_kin = float(np.exp(-0.5 * (omega_err / self.sigma_omega) ** 2))

        total = (
            self.w_d * s_d +
            self.w_psi * s_psi +
            self.w_trans * s_trans +
            self.w_kin * s_kin
        )

        components = {
            's_d': s_d,
            's_psi': s_psi,
            's_trans': s_trans,
            's_kin': s_kin,
            'total': total,
            'd_perp': d,
            'delta_psi': delta_psi
        }
        return total, components

    def update(
        self,
        est_pos: np.ndarray,
        est_heading_deg: float,
        speed_ms: float = 0.0,
        gyro_z_rads: float = 0.0,
        t_now: float = 0.0
    ) -> Dict[str, Any]:
        """
        Processes one epoch update of the multi-hypothesis beam tracker.
        Returns:
            - 'status': 'confident', 'ambiguous', or 'no_candidates'
            - 'committed_candidate': candidate dict or None (if ambiguous)
            - 'top_candidate': candidate dict of Rank 1 hypothesis
            - 'beam_hypotheses': list of active hypotheses
            - 'beam_size': int
            - 'is_fork': bool
        """
        est_h_rad = np.radians(est_heading_deg)
        cands = self.index.query_candidates(
            p=est_pos[:2],
            radius_m=self.search_radius_m,
            veh_heading_rad=est_h_rad,
            heading_gate_rad=self.heading_gate_rad
        )

        if not cands:
            # No candidates in reach
            return {
                'status': 'no_candidates',
                'committed_candidate': None,
                'top_candidate': None,
                'beam_hypotheses': self.beam,
                'beam_size': len(self.beam),
                'is_fork': False,
                'top_prob': 0.0,
                'prob_gap': 0.0
            }

        # Case 1: Initial epoch (beam is empty)
        if not self.beam:
            scored_inits = []
            seen_ways = set()
            for c in cands:
                score, comp = self.compute_single_epoch_score(
                    cand=c, prev_seg_idx=None, speed_ms=speed_ms, gyro_z_rads=gyro_z_rads
                )
                w_id = int(c['way_id'])
                # Prefer diverse physical ways for initial beam
                if w_id not in seen_ways or len(scored_inits) < self.K:
                    seen_ways.add(w_id)
                    h = PathHypothesis(
                        seg_idx=int(c['segment_idx']),
                        way_id=w_id,
                        path_history=[int(c['segment_idx'])],
                        accumulated_score=score,
                        last_d_perp=comp['d_perp'],
                        last_delta_psi=comp['delta_psi'],
                        last_trans_score=1.0,
                        posterior_prob=1.0 / min(len(cands), self.K)
                    )
                    scored_inits.append((score, h, c))

            scored_inits.sort(key=lambda x: x[0], reverse=True)
            self.beam = [h for _, h, _ in scored_inits[:self.K]]
            self._update_posterior_probabilities()

            top_h = self.beam[0]
            top_cand = next(c for c in cands if c['segment_idx'] == top_h.seg_idx)
            is_fork = len(self.beam) > 1 and (self.beam[0].way_id != self.beam[1].way_id)
            prob_gap = (self.beam[0].posterior_prob - self.beam[1].posterior_prob) if len(self.beam) > 1 else 1.0
            is_confident = (prob_gap > self.delta) or (self.K == 1)

            return {
                'status': 'confident' if is_confident else 'ambiguous',
                'committed_candidate': top_cand if is_confident else None,
                'top_candidate': top_cand,
                'beam_hypotheses': self.beam,
                'beam_size': len(self.beam),
                'is_fork': is_fork,
                'top_prob': self.beam[0].posterior_prob,
                'prob_gap': prob_gap
            }

        # Case 2: Recursive beam expansion along candidate set
        expanded_hypotheses: List[Tuple[float, PathHypothesis, Dict[str, Any]]] = []

        cand_map = {c['segment_idx']: c for c in cands}

        for h in self.beam:
            for c in cands:
                c_idx = c['segment_idx']
                # Check topological reachability
                is_connected = self.index.are_topologically_connected(h.seg_idx, c_idx)
                if not is_connected and len(cands) > 1:
                    # Strongly discourage jumping across disconnected streets if connected ones exist
                    continue

                epoch_score, comp = self.compute_single_epoch_score(
                    cand=c, prev_seg_idx=h.seg_idx, speed_ms=speed_ms, gyro_z_rads=gyro_z_rads
                )

                # Recursive score accumulation with decay
                accum_score = self.decay_gamma * h.accumulated_score + epoch_score

                # Check if this transition represents a fork branch
                is_branch = h.is_fork_branch or (c['way_id'] != h.way_id)
                fork_t = h.fork_origin_time if h.is_fork_branch else (t_now if is_branch else 0.0)

                new_h = PathHypothesis(
                    seg_idx=c_idx,
                    way_id=int(c['way_id']),
                    path_history=h.path_history + [c_idx],
                    accumulated_score=accum_score,
                    last_d_perp=comp['d_perp'],
                    last_delta_psi=comp['delta_psi'],
                    last_trans_score=comp['s_trans'],
                    is_fork_branch=is_branch,
                    fork_origin_time=fork_t
                )
                expanded_hypotheses.append((accum_score, new_h, c))

        # Fallback: if all expansions were rejected by strict connectivity, allow best unconstrained
        if not expanded_hypotheses:
            for h in self.beam:
                for c in cands:
                    epoch_score, comp = self.compute_single_epoch_score(
                        cand=c, prev_seg_idx=h.seg_idx, speed_ms=speed_ms, gyro_z_rads=gyro_z_rads
                    )
                    # Heavy transition penalty for disconnected hop
                    accum_score = self.decay_gamma * h.accumulated_score + epoch_score - 1.0
                    new_h = PathHypothesis(
                        seg_idx=c['segment_idx'],
                        way_id=int(c['way_id']),
                        path_history=h.path_history + [c['segment_idx']],
                        accumulated_score=accum_score,
                        last_d_perp=comp['d_perp'],
                        last_delta_psi=comp['delta_psi'],
                        last_trans_score=comp['s_trans']
                    )
                    expanded_hypotheses.append((accum_score, new_h, c))

        # Deduplication: if multiple hypotheses end on the same segment, retain the one with highest score
        best_per_seg: Dict[int, Tuple[float, PathHypothesis, Dict[str, Any]]] = {}
        for item in expanded_hypotheses:
            score, h, c = item
            s_idx = h.seg_idx
            if s_idx not in best_per_seg or score > best_per_seg[s_idx][0]:
                best_per_seg[s_idx] = item

        deduped = list(best_per_seg.values())
        deduped.sort(key=lambda x: x[0], reverse=True)

        # Beam Pruning to top K
        self.beam = [h for _, h, _ in deduped[:self.K]]
        self._update_posterior_probabilities()

        top_h = self.beam[0]
        top_cand = cand_map.get(top_h.seg_idx, None)

        # Fork detection and delayed commitment evaluation
        is_fork = len(self.beam) > 1 and (self.beam[0].way_id != self.beam[1].way_id)
        prob_gap = (self.beam[0].posterior_prob - self.beam[1].posterior_prob) if len(self.beam) > 1 else 1.0

        # Delayed commitment: require confidence margin Delta to commit
        is_confident = (prob_gap > self.delta) or (self.K == 1)

        return {
            'status': 'confident' if is_confident else 'ambiguous',
            'committed_candidate': top_cand if is_confident else None,
            'top_candidate': top_cand,
            'beam_hypotheses': self.beam,
            'beam_size': len(self.beam),
            'is_fork': is_fork,
            'top_prob': self.beam[0].posterior_prob,
            'prob_gap': prob_gap
        }

    def _update_posterior_probabilities(self):
        """Computes normalized softmax posterior probabilities across the active beam."""
        if not self.beam:
            return
        if len(self.beam) == 1:
            self.beam[0].posterior_prob = 1.0
            return

        scores = np.array([h.accumulated_score for h in self.beam])
        max_s = np.max(scores)
        exp_s = np.exp(np.clip(scores - max_s, -20.0, 0.0))
        probs = exp_s / (np.sum(exp_s) + 1e-12)

        for i, p in enumerate(probs):
            self.beam[i].posterior_prob = float(p)
