"""
File created at: 2026-04-30 11:19:37
Author: Sam Ghalayini
meteor/meteor/config/constellation.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from meteor.config.orbital import PHYSICS, PhysicalConstants, ShellConfig


@dataclass(frozen=True)
class ISLThresholds:
    """
    Feasibility thresholds for ISLs

    intra-shell links dont break
    inter-shell links break at a max threshold distance
    Cross shell ground relay links break when sat-relay elevation angle < 25 degrees
    """

    # intra-shell distance thresholds
    intra_plane_max_km: float = math.inf
    inter_plane_max_km: float = 5016.0  # math.inf

    # cross shell ground relay threshold elevation degree
    ground_access_min_elevation_deg: float = 25.0


@dataclass(frozen=True)
class ConstellationConfig:
    """
    Full Constellation tuple consisting of shells plus global ISL rules
    """

    shells: tuple[ShellConfig, ...]
    isl_thresholds: ISLThresholds = field(default_factory=ISLThresholds)
    ground_access_enabled: bool = True
    physics: PhysicalConstants = field(default_factory=lambda: PHYSICS)

    def __post_init__(self):
        ids = [s.shell_id for s in self.shells]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate shell IDs: {ids}")
        if len(self.shells) == 0:
            raise ValueError("ConstellationConfig must contain at least one shell")

    # counts
    @property
    def n_shells(self) -> int:
        return len(self.shells)

    @property
    def n_satellites(self) -> int:
        return sum(s.n_satellites for s in self.shells)

    # global node id mapping for satllites
    @property
    def shell_offsets(self) -> np.ndarray:
        """total satellite count per shell to create offsets"""
        offsets = np.zeros(self.n_shells + 1, dtype=np.int64)
        for i, shell in enumerate(self.shells):
            offsets[i + 1] = offsets[i] + shell.n_satellites
        return offsets

    def global_id(self, shell_idx: int, plane: int, sat_idx: int) -> int:
        """Convert to global node id"""
        shell = self.shells[shell_idx]
        local = plane * shell.sats_per_plane + sat_idx
        return int(self.shell_offsets[shell_idx] + local)

    def shell_of(self, global_id: int) -> int:
        offsets = self.shell_offsets
        return int(np.searchsorted(offsets[1:], global_id, side="right"))
