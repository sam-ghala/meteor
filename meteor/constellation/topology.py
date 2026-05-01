"""
File created at: 2026-04-30 11:20:24
Author: Sam Ghalayini
meteor/meteor/constellation/topology.py

Topology snapshot at a single time t

Construction sequence:
1. Bulid intra-plane edges per shell
2. Build candidate inter-plane edges per shell
3. Build candidate cross-shell edges
4. Prune for hardware limits
5. Build ground-access edges (use RF, not laser terminals)
6. Mirror all surviving edges for bidirectional flows

Note: Bent-pipe relays are distingused in path computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from meteor.config.constellation import ConstellationConfig
from meteor.constellation import isl as isl_mod
from meteor.constellation.isl import (
    ISLType,
    apply_terminal_budget,
    build_flexible_budget_array,
)
from meteor.constellation.kinematics import constellations_positions, pairwise_distance
from meteor.ground.gateways import Gateway, gateway_position_array, get_global_gateways

logger = logging.getLogger(__name__)


@dataclass
class Topology:
    """Complete topology snapshot at time t"""

    # identify topology
    t: float
    config: ConstellationConfig

    # per satellite array
    sat_xyz: np.ndarray = field(repr=False)
    sat_lats_deg: np.ndarray = field(repr=False)
    sat_lons_deg: np.ndarray = field(repr=False)
    shell_idx: np.ndarray = field(repr=False)

    # per gateway arrays
    gateways: tuple[Gateway, ...] = field(default_factory=tuple, repr=False)
    gw_xyz: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 3)))

    # edges
    edge_src: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, dtype=np.int64))
    edge_dst: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, dtype=np.int64))
    edge_delay_s: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    edge_capacity_mbps: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    edge_isl_type: np.ndarray = field(
        repr=False, default_factory=lambda: np.empty(0, dtype=np.int8)
    )

    # public counts
    @property
    def n_satellites(self) -> int:
        return len(self.sat_xyz)

    @property
    def n_gateways(self) -> int:
        return len(self.gateways)

    @property
    def n_nodes(self) -> int:
        return self.n_satellites + self.n_gateways

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def gateway_id_offset(self) -> int:
        return self.n_satellites

    def is_gateway(self, node_id: int) -> bool:
        return node_id >= self.gateway_id_offset

    def gateway_for_id(self, node_id: int) -> Gateway:
        if not self.is_gateway(node_id):
            raise ValueError(f"node {node_id} is a satellite, not a gateway")
        return self.gateways[node_id - self.gateway_id_offset]

    # construct topology
    @classmethod
    def from_config(
        cls, config: ConstellationConfig, t: float, gateways: list[Gateway] | None = None
    ) -> Topology:
        logger.debug(
            f"Building topology at t={t} for {config.n_satellites} satellites across {config.n_shells}, terminal limited={config.is_terminal_limited}"
        )

        # 1. positions
        sat_xyz, sat_lats, sat_lons = constellations_positions(config, t)
        shell_idx_arr = _build_shell_idx(config)

        # 2. gateways
        if config.ground_access_enabled:
            gw_list = list(gateways) if gateways is not None else get_global_gateways()
        else:
            gw_list = []
        gw_xyz_arr = gateway_position_array(gw_list, config.physics.earth_radius_km)

        # 3. intra-plane
        intra_edges = _build_intra_plane_edges(config, sat_xyz)

        # 4. inter-plane and cross shells
        inter_candidates = _build_inter_plane_candidates(config, sat_xyz, sat_lats)
        cross_candidates = _build_cross_shell_candidates(config, sat_xyz)

        # 5. pruning for terminal budgets
        flex_edges = _combine_candidate_arrays(inter_candidates, cross_candidates)
        if config.is_terminal_limited and len(flex_edges[0]) > 0:
            budget = build_flexible_budget_array(config)
            accepted = apply_terminal_budget(
                candidate_src=flex_edges[0],
                candidate_dst=flex_edges[1],
                candidate_dist=flex_edges[5],
                candidate_type=flex_edges[4],
                flexible_budget=budget,
            )
            flex_edges = tuple(arr[accepted] for arr in flex_edges)
            logger.debug(
                f"Terminal heuristics: kept {int(accepted.sum())} / {len(accepted)} flexible terminals"
            )

        # 6. ground access edges
        flex_src, flex_dst, flex_delay, flex_cap, flex_type, _flex_dist = flex_edges

        # 7. mirror to make bidirectional
        if config.ground_access_enabled and len(gw_list) > 0:
            ga_edges = _build_ground_access_edges(config, sat_xyz, gw_xyz_arr, gw_list)
        else:
            ga_edges = _empty_edge_tuple()

        # make "El Topo" not the movie, the Topology
        all_groups = [intra_edges, (flex_src, flex_dst, flex_delay, flex_cap, flex_type), ga_edges]
        src_unidir = (
            np.concatenate([g[0] for g in all_groups])
            if any(len(g[0]) for g in all_groups)
            else np.empty(0, dtype=np.int64)
        )
        dst_unidir = (
            np.concatenate([g[1] for g in all_groups])
            if any(len(g[1]) for g in all_groups)
            else np.empty(0, dtype=np.int64)
        )
        delay_unidir = (
            np.concatenate([g[2] for g in all_groups])
            if any(len(g[2]) for g in all_groups)
            else np.empty(0)
        )
        cap_unidir = (
            np.concatenate([g[3] for g in all_groups])
            if any(len(g[3]) for g in all_groups)
            else np.empty(0)
        )
        type_unidir = (
            np.concatenate([g[4] for g in all_groups])
            if any(len(g[4]) for g in all_groups)
            else np.empty(0, dtype=np.int8)
        )

        edge_src = np.concatenate([src_unidir, dst_unidir])
        edge_dst = np.concatenate([dst_unidir, src_unidir])
        edge_delay = np.concatenate([delay_unidir, delay_unidir])
        edge_cap = np.concatenate([cap_unidir, cap_unidir])
        edge_type = np.concatenate([type_unidir, type_unidir]).astype(np.int8)

        topo = cls(
            t=t,
            config=config,
            sat_xyz=sat_xyz,
            sat_lats_deg=sat_lats,
            sat_lons_deg=sat_lons,
            shell_idx=shell_idx_arr,
            gateways=tuple(gw_list),
            gw_xyz=gw_xyz_arr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_delay_s=edge_delay,
            edge_capacity_mbps=edge_cap,
            edge_isl_type=edge_type,
        )
        logger.debug(
            f"Built:\n  {topo.n_satellites} satellites\n  {topo.n_gateways} gateways\n  {topo.n_nodes} total nodes\n  {topo.n_edges}"
        )
        return topo

    # public api
    def get_edge_set(self) -> frozenset:
        """Frozenset of (src, dst, isl_type) tuples for topology holding time comparison"""
        return frozenset(
            zip(
                self.edge_src.tolist(),
                self.edge_dst.tolist(),
                self.edge_isl_type.tolist(),
                strict=True,
            )
        )

    def edges_of_type(self, isl_type: ISLType) -> tuple[np.ndarray, np.ndarray]:
        mask = self.edge_isl_type == int(isl_type)
        return self.edge_src[mask], self.edge_dst[mask]

    def edge_count_by_type(self) -> dict[ISLType, int]:
        return {t: int((self.edge_isl_type == int(t)).sum()) for t in ISLType}

    def to_networkx(self):
        import networkx as nx

        G = nx.DiGraph()
        for i in range(self.n_satellites):
            G.add_node(
                int(i),
                kind="satellite",
                shell=int(self.shell_idx[i]),
                lat=float(self.sat_lats_deg[i]),
                lon=float(self.sat_lons_deg[i]),
            )
        for i, gw in enumerate(self.gateways):
            G.add_node(
                int(self.gateway_id_offset + i),
                kind="gateway",
                gw_id=gw.id,
                lat=gw.lat,
                lon=gw.lon,
                has_server=gw.has_server,
            )
        edges = [
            (
                int(s),
                int(d),
                {
                    "delay": float(self.edge_delay_s[k]),
                    "capacity": float(self.edge_capacity_mbps[k]),
                    "isl_type": int(self.edge_isl_type[k]),
                },
            )
            for k, (s, d) in enumerate(zip(self.edge_src, self.edge_dst, strict=True))
        ]
        G.add_edges_from(edges)
        return G


# helpers for construction steps aboce in from_config


def _empty_edge_tuple():
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int8),
    )


def _empty_candidate_tuple():
    return _empty_edge_tuple() + (np.empty(0, dtype=np.float64),)


def _build_shell_idx(config: ConstellationConfig) -> np.ndarray:
    shell_idx = np.empty(config.n_satellites, dtype=np.int8)
    offsets = config.shell_offsets
    for i in range(config.n_shells):
        shell_idx[offsets[i] : offsets[i + 1]] = i
    return shell_idx


def _build_intra_plane_edges(config: ConstellationConfig, sat_xyz: np.ndarray):
    """intra-plane edges per shell"""
    speed = config.physics.speed_of_light_km_s
    src_chunks, dst_chunks, delay_chunks, cap_chunks, type_chunks = [], [], [], [], []
    offsets = config.shell_offsets

    for i, shell in enumerate(config.shells):
        local_pairs = isl_mod.intra_plane_pairs(shell)
        if len(local_pairs) == 0:
            continue
        src = local_pairs[:, 0] + offsets[i]
        dst = local_pairs[:, 1] + offsets[i]
        dist = pairwise_distance(sat_xyz[src], sat_xyz[dst])

        src_chunks.append(src)
        dst_chunks.append(dst)
        delay_chunks.append(dist / speed)
        cap_chunks.append(np.full(len(src), shell.hardware.isl_capacity_mbps))
        type_chunks.append(np.full(len(src), int(ISLType.INTRA_PLANE), dtype=np.int8))

    if not src_chunks:
        return _empty_edge_tuple()
    return (
        np.concatenate(src_chunks),
        np.concatenate(dst_chunks),
        np.concatenate(delay_chunks),
        np.concatenate(cap_chunks),
        np.concatenate(type_chunks),
    )


def _build_inter_plane_candidates(
    config: ConstellationConfig,
    sat_xyz: np.ndarray,
    sat_lats: np.ndarray,
):
    """inter-plane candidates per shell"""
    speed = config.physics.speed_of_light_km_s
    src_chunks, dst_chunks, delay_chunks, cap_chunks, type_chunks, dist_chunks = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    offsets = config.shell_offsets

    for i, shell in enumerate(config.shells):
        shell_start = offsets[i]
        shell_end = offsets[i + 1]
        shell_lats = sat_lats[shell_start:shell_end]

        local_pairs = isl_mod.inter_plane_pairs(
            shell, shell_lats, config.isl_thresholds, distances=None
        )
        if len(local_pairs) == 0:
            continue
        src = local_pairs[:, 0] + shell_start
        dst = local_pairs[:, 1] + shell_start
        dist = pairwise_distance(sat_xyz[src], sat_xyz[dst])

        src_chunks.append(src)
        dst_chunks.append(dst)
        delay_chunks.append(dist / speed)
        cap_chunks.append(np.full(len(src), shell.hardware.isl_capacity_mbps))
        type_chunks.append(np.full(len(src), int(ISLType.INTER_PLANE), dtype=np.int8))
        dist_chunks.append(dist)

    if not src_chunks:
        return _empty_candidate_tuple()
    return (
        np.concatenate(src_chunks),
        np.concatenate(dst_chunks),
        np.concatenate(delay_chunks),
        np.concatenate(cap_chunks),
        np.concatenate(type_chunks),
        np.concatenate(dist_chunks),
    )


def _build_cross_shell_candidates(config: ConstellationConfig, sat_xyz: np.ndarray):
    """Cross-shell laser candidates"""
    if not (config.cross_shell_enabled and config.n_shells > 1):
        return _empty_candidate_tuple()

    speed = config.physics.speed_of_light_km_s
    offsets = config.shell_offsets
    src_chunks, dst_chunks, delay_chunks, cap_chunks, type_chunks, dist_chunks = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(config.n_shells):
        for j in range(i + 1, config.n_shells):
            shell_i = config.shells[i]
            shell_j = config.shells[j]
            xyz_i = sat_xyz[offsets[i] : offsets[i + 1]]
            xyz_j = sat_xyz[offsets[j] : offsets[j + 1]]
            link_cap = min(shell_i.hardware.isl_capacity_mbps, shell_j.hardware.isl_capacity_mbps)

            local_a, local_b = isl_mod.cross_shell_laser_pairs(
                xyz_i,
                xyz_j,
                config.isl_thresholds,
            )
            if len(local_a) == 0:
                continue
            src = local_a + offsets[i]
            dst = local_b + offsets[j]
            dist = pairwise_distance(sat_xyz[src], sat_xyz[dst])

            src_chunks.append(src)
            dst_chunks.append(dst)
            delay_chunks.append(dist / speed)
            cap_chunks.append(np.full(len(src), link_cap))
            type_chunks.append(np.full(len(src), int(ISLType.CROSS_SHELL_LASER), dtype=np.int8))
            dist_chunks.append(dist)

    if not src_chunks:
        return _empty_candidate_tuple()
    return (
        np.concatenate(src_chunks),
        np.concatenate(dst_chunks),
        np.concatenate(delay_chunks),
        np.concatenate(cap_chunks),
        np.concatenate(type_chunks),
        np.concatenate(dist_chunks),
    )


def _combine_candidate_arrays(*candidate_tuples):
    """stack multiple candidate-edge tuples into a single one before pruning"""
    parts = [t for t in candidate_tuples if len(t[0]) > 0]
    if not parts:
        return _empty_candidate_tuple()
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(6))


def _build_ground_access_edges(
    config: ConstellationConfig,
    sat_xyz: np.ndarray,
    gw_xyz: np.ndarray,
    gw_list: list,
):
    """sat to gateway edges (seperate connection from ISLs, these use RF (Ka-Band and others))"""
    sat_idx, gw_idx, distances = isl_mod.ground_access_pairs(sat_xyz, gw_xyz, config.isl_thresholds)
    if len(sat_idx) == 0:
        return _empty_edge_tuple()

    speed = config.physics.speed_of_light_km_s
    delay = distances / speed
    n_sats = config.n_satellites
    gw_global = gw_idx + n_sats

    sat_uplinks = np.array(
        [config.shells[_lookup_shell(config, int(s))].hardware.ground_uplink_mbps for s in sat_idx]
    )
    gw_uplinks = np.array([gw_list[g].uplink_mbps for g in gw_idx])
    capacity = np.minimum(sat_uplinks, gw_uplinks)

    types = np.full(len(sat_idx), int(ISLType.GROUND_ACCESS), dtype=np.int8)
    return (
        sat_idx.astype(np.int64),
        gw_global.astype(np.int64),
        delay,
        capacity,
        types,
    )


def _lookup_shell(config: ConstellationConfig, sat_global_id: int) -> int:
    offsets = config.shell_offsets
    return int(np.searchsorted(offsets[1:], sat_global_id, side="right"))
