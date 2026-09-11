"""
SIH26168 - Stage C8: Geometric Projection & Spatial Query Utilities
Script: src/map/geometry.py

Features:
- Fast vectorized point-to-segment orthogonal projection and distance calculation.
- Signed lateral cross-track distance (left/right of centerline).
- Segment tangent heading error calculation with proper 2*pi wrapping.
- Spatial candidate query within uncertainty radius R.
- Candidate multiplicity and directional alignment pruning.
"""

import numpy as np


def wrap_angle_rad(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps an angle or array of angles in radians to [-pi, +pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class RoadNetworkIndex:
    """
    Vectorized 2D spatial index for fast orthogonal distance, candidate retrieval,
    and tangent heading calculation against an OSM road segment database.
    """

    def __init__(self, segments: list[dict]):
        self.segments = segments
        self.num_segments = len(segments)

        if self.num_segments == 0:
            self.p1 = np.empty((0, 2))
            self.p2 = np.empty((0, 2))
            self.delta = np.empty((0, 2))
            self.lengths_sq = np.empty(0)
            self.tangents = np.empty((0, 2))
            self.headings_rad = np.empty(0)
            self.way_ids = np.empty(0, dtype=int)
            return

        self.p1 = np.array([s['p1'] for s in segments])  # (M, 2)
        self.p2 = np.array([s['p2'] for s in segments])  # (M, 2)
        self.delta = self.p2 - self.p1                  # (M, 2)
        self.lengths_sq = np.sum(self.delta ** 2, axis=1) # (M,)
        self.tangents = np.array([s['tangent'] for s in segments]) # (M, 2)
        self.headings_rad = np.array([s['heading_rad'] for s in segments]) # (M,)
        self.way_ids = np.array([s['way_id'] for s in segments])
        self.u_nodes = np.array([s.get('u_node', -1) for s in segments], dtype=np.int64)
        self.v_nodes = np.array([s.get('v_node', -1) for s in segments], dtype=np.int64)

        # Build graph topological adjacency maps
        from collections import defaultdict
        self.node_to_outgoing = defaultdict(list)
        self.node_to_incoming = defaultdict(list)
        self.way_to_segments = defaultdict(list)
        for idx, s in enumerate(segments):
            u = s.get('u_node', -1)
            v = s.get('v_node', -1)
            w = s.get('way_id', -1)
            if u != -1:
                self.node_to_outgoing[u].append(idx)
            if v != -1:
                self.node_to_incoming[v].append(idx)
            if w != -1:
                self.way_to_segments[w].append(idx)

    def are_topologically_connected(self, seg_idx_prev: int, seg_idx_curr: int) -> bool:
        """
        Determines whether two segments are topologically connected in the road network:
        1. They belong to the identical OSM way (same physical road).
        2. Direct directed junction: the end node of prev is the start node of curr.
        3. Undirected junction: they share any junction node.
        """
        if seg_idx_prev < 0 or seg_idx_curr < 0:
            return False
        if seg_idx_prev == seg_idx_curr:
            return True
        if self.way_ids[seg_idx_prev] == self.way_ids[seg_idx_curr]:
            return True
        v_prev = self.v_nodes[seg_idx_prev]
        u_curr = self.u_nodes[seg_idx_curr]
        if v_prev != -1 and v_prev == u_curr:
            return True
        u_prev = self.u_nodes[seg_idx_prev]
        v_curr = self.v_nodes[seg_idx_curr]
        # Any shared junction node
        prev_nodes = {u_prev, v_prev} - {-1}
        curr_nodes = {u_curr, v_curr} - {-1}
        return len(prev_nodes & curr_nodes) > 0

    def topological_transition_score(self, seg_idx_prev: int | None, seg_idx_curr: int) -> float:
        """
        Computes transition score in [0.0, 1.0] based on road network graph connectivity.
        """
        if seg_idx_prev is None or seg_idx_prev < 0:
            return 1.0  # Neutral prior if no previous segment
        if seg_idx_prev == seg_idx_curr:
            return 1.0
        if self.way_ids[seg_idx_prev] == self.way_ids[seg_idx_curr]:
            return 0.95
        v_prev = self.v_nodes[seg_idx_prev]
        u_curr = self.u_nodes[seg_idx_curr]
        if v_prev != -1 and v_prev == u_curr:
            return 0.85
        # Shared junction node
        u_prev = self.u_nodes[seg_idx_prev]
        v_curr = self.v_nodes[seg_idx_curr]
        prev_nodes = {u_prev, v_prev} - {-1}
        curr_nodes = {u_curr, v_curr} - {-1}
        if len(prev_nodes & curr_nodes) > 0:
            return 0.60
        return 0.0

    def continuity_score(self, seg_idx_prev: int | None, seg_idx_curr: int) -> float:
        """
        Computes continuity score: 1.0 if identical way, else 0.0.
        """
        if seg_idx_prev is None or seg_idx_prev < 0:
            return 1.0
        return 1.0 if (self.way_ids[seg_idx_prev] == self.way_ids[seg_idx_curr]) else 0.0

    def query_point(self, p: np.ndarray | tuple) -> dict:
        """
        Computes orthogonal distance and projection to all segments for a 2D point p = [e, n].
        Returns nearest segment details.
        """
        if self.num_segments == 0:
            return {
                'nearest_dist': np.inf, 'nearest_idx': -1,
                'proj_point': np.zeros(2), 'signed_d_perp': 0.0,
                'road_heading': 0.0, 'segment': None
            }

        p = np.asarray(p[:2], dtype=float)
        ap = p - self.p1  # (M, 2)

        # Projection fraction t in [0, 1]
        dot_prod = ap[:, 0] * self.delta[:, 0] + ap[:, 1] * self.delta[:, 1]
        t = np.clip(dot_prod / (self.lengths_sq + 1e-12), 0.0, 1.0)  # (M,)

        # Projected closest point on each segment
        proj = self.p1 + t[:, np.newaxis] * self.delta  # (M, 2)

        # Orthogonal distance
        dist_sq = np.sum((p - proj) ** 2, axis=1)  # (M,)
        min_idx = int(np.argmin(dist_sq))
        min_dist = float(np.sqrt(dist_sq[min_idx]))

        # Signed lateral offset: 2D cross product (p - p1) x tangent
        # In ENU: tangent = [t_e, t_n]. Cross product = ap_e * t_n - ap_n * t_e
        # Positive = right of centerline, Negative = left of centerline
        tan = self.tangents[min_idx]
        ap_best = ap[min_idx]
        signed_cross = ap_best[0] * tan[1] - ap_best[1] * tan[0]

        return {
            'nearest_dist': min_dist,
            'nearest_idx': min_idx,
            'proj_point': proj[min_idx],
            'signed_d_perp': float(signed_cross),
            'road_heading': float(self.headings_rad[min_idx]),
            'segment': self.segments[min_idx],
            'all_dists': np.sqrt(dist_sq),
            'all_projs': proj
        }

    def query_candidates(self, p: np.ndarray | tuple, radius_m: float,
                         veh_heading_rad: float | None = None,
                         heading_gate_rad: float | None = None) -> list[dict]:
        """
        Retrieves all candidate road segments within a given search radius R.
        Optionally filters candidates whose tangent heading differs from vehicle heading
        by more than heading_gate_rad.
        """
        res = self.query_point(p)
        dists = res['all_dists']
        mask = dists <= radius_m

        if np.sum(mask) == 0:
            return []

        cand_indices = np.where(mask)[0]
        candidates = []

        for idx in cand_indices:
            seg = self.segments[idx]
            d = float(dists[idx])
            h_road = float(self.headings_rad[idx])

            delta_heading = None
            if veh_heading_rad is not None:
                delta_heading = float(wrap_angle_rad(veh_heading_rad - h_road))
                if heading_gate_rad is not None and abs(delta_heading) > heading_gate_rad:
                    continue

            candidates.append({
                'segment_idx': int(idx),
                'way_id': int(seg['way_id']),
                'name': seg['name'],
                'highway': seg['highway'],
                'is_oneway': seg['is_oneway'],
                'lanes': seg['lanes'],
                'u_node': int(seg.get('u_node', -1)),
                'v_node': int(seg.get('v_node', -1)),
                'distance_m': d,
                'road_heading_rad': h_road,
                'delta_heading_rad': delta_heading,
                'proj_point': res['all_projs'][idx]
            })

        # Sort by distance
        candidates.sort(key=lambda c: c['distance_m'])
        return candidates
