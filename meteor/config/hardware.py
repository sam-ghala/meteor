"""
File created at: 2026-05-01 10:57:27
Author: Sam Ghalayini
meteor/meteor/config/hardware.py

Hardware configs for satellite buses
"""

from __future__ import annotations

from dataclasses import dataclass

# for unlimited terminals
UNLIMITED_TERMINALS = -1


@dataclass(frozen=True)
class HardwareConfig:
    """
    hardware-level satellite specs that drive ISL topology changes

    Terminal counts:
        n_laser_terminasl : total optical ISL terminals on each satellite
            -1 = unlimited
            >= 1 enables greedy distance based pairings
        n_intra_plane_reserved: terminals reserved for intra-plane links (same orbit)
            with terminals = 3, 2 links for same orbit and one flexible one
            if n_laser_termainls =-1 this is ignored

    Capacity:
        isl_capacity_mbps : per link bandwidth ISL (200Gbps)
        ground up/down link capacity : RF link (50Mbps)
    """

    hw_id: str
    n_laser_terminals: int = UNLIMITED_TERMINALS
    n_intra_plane_reserved: int = 0
    isl_capacity_mbps: float = 200.0
    ground_uplink_mbps: float = 50.0
    ground_downlink_mbps: float = 50.0

    def __post_init__(self):
        if self.n_laser_terminals != UNLIMITED_TERMINALS:
            if self.n_laser_terminals < 1:
                raise ValueError(
                    f"n_laser_terminals must be >= 1 or {UNLIMITED_TERMINALS} "
                    f"(unlimited), got {self.n_laser_terminals}"
                )
            if self.n_intra_plane_reserved < 0:
                raise ValueError("n_intra_plane_reserved must be >= 0")
            if self.n_intra_plane_reserved > self.n_laser_terminals:
                raise ValueError(
                    f"n_intra_plane_reserved ({self.n_intra_plane_reserved}) "
                    f"cannot exceed n_laser_terminals ({self.n_laser_terminals})"
                )

    @property
    def is_unlimited(self) -> bool:
        return self.n_laser_terminals == UNLIMITED_TERMINALS

    @property
    def n_flexible_terminals(self) -> int:
        """Terminals available for non-intra-plane links"""
        if self.is_unlimited:
            return UNLIMITED_TERMINALS
        return self.n_laser_terminals - self.n_intra_plane_reserved


# predefined hardware specs

# no hardware terminal constraints
HW_UNLIMITED = HardwareConfig(
    hw_id="sate_unlimited",
    n_laser_terminals=UNLIMITED_TERMINALS,
)

# starlink V1.5 (Gen1 lasers): 4 terminals
# two inter- and two intra-plane
HW_V1_5 = HardwareConfig(
    hw_id="starlink_v1_5",
    n_laser_terminals=4,
    n_intra_plane_reserved=2,
    isl_capacity_mbps=200.0,
)

# starlink v3 Gen2: 3 lasers, one flexible
HW_V3 = HardwareConfig(
    hw_id="starlink_v3",
    n_laser_terminals=3,
    n_intra_plane_reserved=2,
    isl_capacity_mbps=200.0,
)
