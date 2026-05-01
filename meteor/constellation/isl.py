"""
File created at: 2026-04-30 11:20:16
Author: Sam Ghalayini
meteor/meteor/constellation/isl.py

Defining ISL classification and feasability rules

Defines which pairs can form ISLs and what type they are

Link Types:
- INTRA_PLANE       - same shell, same plane, ring-adjacent, always active
- INTER_PLANE       - same shell, adjacent planes, break above lat_cutoff
- CROSS_SHELL_LASER - different_shells, direct laser, break beyond cross_shell_laser_max_km
- GROUND_ACCESS     - satellite to gateway, break when satellite elvation drops below ground_access_min_elevation_deg. Used for both traffic and bent pipe bridges
"""

from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from meteor.config.constellation import ConstellationConfig, ISLThresholds
from meteor.config.hardware import UNLIMITED_TERMINALS
from meteor.config.orbital import ShellConfig
from meteor.constellation.kinematics import elev_matrix

logger = logging.getLogger(__name__)


class ISLType(IntEnum):
    """
    ISL link type
    """

    INTRA_PLANE = 0
    INTER_PLANE = 1
    CROSS_SHELL_LASER = 2
    GROUND_ACCESS = 3


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
    lats_deg: np.ndarray,
    thresholds: ISLThresholds,
    distances: np.ndarray | None = None,
) -> np.ndarray:
    """
    Local (src, dst) pairs for inter-plane links, cutoff by latitude and (optinally) distance threshold.
    """
    n_planes = shell.n_planes
    sats_per_plane = shell.sats_per_plane
    lat_cutoff = shell.lat_cutoff_deg

    src_list, dst_list = [], []
    for p in range(n_planes - 1):
        local_src = p * sats_per_plane + np.arange(sats_per_plane)
        local_dst = (p + 1) * sats_per_plane + np.arange(sats_per_plane)
        src_list.append(local_src)
        dst_list.append(local_dst)

    if not src_list:
        return np.empty((0, 2), dtype=np.int64)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)

    lat_ok = (np.abs(lats_deg[src]) <= lat_cutoff) & (np.abs(lats_deg[dst]) <= lat_cutoff)
    feasible = lat_ok

    if np.isfinite(thresholds.inter_plane_max_km):
        if distances is None:
            raise ValueError("inter_plane_max_km is finite but no distances provided")
        feasible = feasible & (distances <= thresholds.inter_plane_max_km)

    return np.stack([src[feasible], dst[feasible]], axis=-1)


# cross shell laser generation
def cross_shell_laser_pairs(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    thresholds: ISLThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each satellite in shell A, find nearest satellite in shell B within laser thresholds
    List my be pruned due to limited terminal heuristic
    """
    if len(xyz_a) == 0 or len(xyz_b) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    tree_b = cKDTree(xyz_b)
    distances, nearest_b = tree_b.query(xyz_a, k=1)
    feasible = distances <= thresholds.cross_shell_laser_max_km
    return np.where(feasible)[0], nearest_b[feasible]


# ground access
def ground_access_pairs(
    sat_xyz: np.ndarray,
    gw_xyz: np.ndarray,
    thresholds: ISLThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All feasible sat-gateway pairs based on minimum elevation angle"""
    if len(sat_xyz) == 0 or len(gw_xyz) == 0:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return empty_int, empty_int, empty_float

    elev = elev_matrix(sat_xyz, gw_xyz)
    feasible = elev >= thresholds.ground_access_min_elevation_deg
    if not feasible.any():
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return empty_int, empty_int, empty_float

    sat_idx, gw_idx = np.where(feasible)
    diff = sat_xyz[sat_idx] - gw_xyz[gw_idx]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    return sat_idx.astype(np.int64), gw_idx.astype(np.int64), distances


# terminal-budget pairing heuristic
# =================================


def apply_terminal_budget(
    candidate_src: np.ndarray,
    candidate_dst: np.ndarray,
    candidate_dist: np.ndarray,
    candidate_type: np.ndarray,
    flexible_budget: np.ndarray,
) -> np.ndarray:
    """
    Greedy distance based pairing for non-intra-plane edges for flexible terminal budgets

    Args:
        src : (E,) global IDs for sources
        dst : (E,) global IDs for destinations
        dist: (E,) distances for sorting
        type: (E,) ISL types
        budget(N,) per-satellite remaining terminals (1 or 2 or unlimited)

    Returns:
        accepted: (E,) boolean mask of accepted edges
    """
    n_candidates = len(candidate_src)
    if n_candidates == 0:
        return np.empty(0, dtype=bool)

    # sort
    order = np.argsort(candidate_dist, kind="stable")
    accepted = np.zeros(n_candidates, dtype=bool)

    for k in order:
        s = int(candidate_src[k])
        d = int(candidate_dst[k])
        s_unlim = flexible_budget[s] == UNLIMITED_TERMINALS
        d_unlim = flexible_budget[d] == UNLIMITED_TERMINALS

        if (s_unlim or flexible_budget[s] > 0) and (d_unlim or flexible_budget[d] > 0):
            accepted[k] = True
            if not s_unlim:
                flexible_budget[s] -= 1
            if not d_unlim:
                flexible_budget[d] -= 1
    return accepted


def build_flexible_budget_array(config: ConstellationConfig) -> np.ndarray:
    """
    Construct the per-satellite flexible-terminal budget array, indexed by global satellite ID
    """
    budget = np.empty(config.n_satellites, dtype=np.int64)
    offsets = config.shell_offsets
    for i, shell in enumerate(config.shells):
        if shell.hardware.is_unlimited:
            budget[offsets[i] : offsets[i + 1]] = UNLIMITED_TERMINALS
        else:
            budget[offsets[i] : offsets[i + 1]] = shell.hardware.n_flexible_terminals
    return budget
