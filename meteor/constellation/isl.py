"""
File created at: 2026-04-30 11:20:16
Author: Sam Ghalayini
meteor/meteor/constellation/isl.py

Defining ISL classification and feasability rules

Defines which pairs can form ISLs and what type they are

Link Types:
- INTRA_PLANE       - same shell, same plane, ring-adjacent, always active
- INTER_PLANE       - same shell, adjacent planes, break above lat_cutoff
- GROUND_ACCESS     - satellite to gateway, break when satellite elvation drops below ground_access_min_elevation_deg. Used for both traffic and bent pipe bridges
"""

from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np

from meteor.config.constellation import ISLThresholds
from meteor.config.hardware import TopologyRule
from meteor.config.orbital import ShellConfig
from meteor.constellation.kinematics import elev_matrix

logger = logging.getLogger(__name__)


class ISLType(IntEnum):
    """
    ISL link type
    """

    INTRA_PLANE = 0
    INTER_PLANE = 1
    GROUND_ACCESS = 2


# intra-shell links
def intra_plane_pairs(shell: ShellConfig) -> np.ndarray:
    """
    Local (src,dst) pairs for intra-plane links: each sat connects to next sat in same plane
    Returns one direction, mirrored for bidirectionality
    """
    n_planes = shell.n_planes
    sats_per_plane = shell.sats_per_plane
    n_pairs = n_planes * sats_per_plane

    src = np.arange(n_pairs, dtype=np.int64)
    plane = src // sats_per_plane
    sat = src % sats_per_plane
    next_sat = (sat + 1) % sats_per_plane
    dst = plane * sats_per_plane + next_sat

    return np.stack([src, dst], axis=-1)


def inter_plane_pairs(
    shell: ShellConfig,
    sat_xyz_shell: np.ndarray,
    lats_deg: np.ndarray,
    thresholds: ISLThresholds,
    *,
    topology_rule: TopologyRule,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local (src, dst) pairs for inter-plane links, cutoff by latitude and (optinally) distance threshold.
    """
    n_planes = shell.n_planes
    sats_per_plane = shell.sats_per_plane
    lat_cutoff = shell.lat_cutoff_deg

    if n_planes < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    # build candidate src dst paris for adjacent planes
    src_list, dst_list = [], []
    for p in range(n_planes - 1):
        sat_indices = np.arange(sats_per_plane, dtype=np.int64)

        if topology_rule == "four_isl":
            keep = np.ones(sats_per_plane, dtype=bool)
        elif topology_rule == "three_isl_bricks":
            keep = (p + sat_indices) % 2 == 0
        else:
            raise ValueError(f"unknown topology rule {topology_rule!r}")
        if not keep.any():
            continue
        kept = sat_indices[keep]

        local_src = p * sats_per_plane + kept
        local_dst = (p + 1) * sats_per_plane + kept
        src_list.append(local_src)
        dst_list.append(local_dst)

    if not src_list:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)

    diff = sat_xyz_shell[src] - sat_xyz_shell[dst]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))

    feasible = (np.abs(lats_deg[src]) <= lat_cutoff) & (np.abs(lats_deg[dst]) <= lat_cutoff)

    if np.isfinite(thresholds.inter_plane_max_km):
        feasible = feasible & (distances <= thresholds.inter_plane_max_km)

    pairs = np.stack([src[feasible], dst[feasible]], axis=-1)
    kept_distances = distances[feasible]
    return pairs, kept_distances


# ground access
def ground_access_pairs(
    sat_xyz: np.ndarray,
    gw_xyz: np.ndarray,
    thresholds: ISLThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All feasible sat-gateway pairs based on minimum elevation angle"""
    if len(sat_xyz) == 0 or len(gw_xyz) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    elev = elev_matrix(sat_xyz, gw_xyz)
    feasible = elev >= thresholds.ground_access_min_elevation_deg
    if not feasible.any():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    sat_idx, gw_idx = np.where(feasible)
    diff = sat_xyz[sat_idx] - gw_xyz[gw_idx]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    return sat_idx.astype(np.int64), gw_idx.astype(np.int64), distances
