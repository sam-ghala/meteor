"""
File created at: 2026-05-01 10:57:27
Author: Sam Ghalayini
meteor/meteor/config/hardware.py

Hardware configs for satellite buses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TopologyRule = Literal["four_isl", "three_isl_bricks"]


@dataclass(frozen=True)
class HardwareConfig:
    """
    hardware-level satellite specs that drive ISL topology changes

    Terminal counts:
        FOUR_ISL uses +Grid topology
        THREE_ISL uses brick-laying topology

    Capacity:
        isl_capacity_mbps : per link bandwidth ISL (200Gbps)
        ground up/down link capacity : RF link (50Mbps)
    """

    hw_id: str
    topology_rule: TopologyRule
    n_laser_terminals: int
    isl_capacity_mbps: float = 200.0
    ground_uplink_mbps: float = 50.0
    ground_downlink_mbps: float = 50.0

    def __post_init__(self):
        expected_terminals = {
            "four_isl": 4,
            "three_isl_bricks": 3,
        }
        if self.topology_rule not in expected_terminals:
            raise ValueError(
                f"Unknown topology rule: {self.topology_rule!r}"
                f"Supported topology rules: {sorted(expected_terminals.keys())}"
            )
        expected = expected_terminals[self.topology_rule]
        if self.n_laser_terminals != expected:
            raise ValueError(
                f"topology_rule={self.topology_rule!r} expects "
                f"n_laser_terminals={expected}, got {self.n_laser_terminals}"
            )


# predefined hardware specs

# +Grid
HW_FOUR_ISL = HardwareConfig(
    hw_id="four_isl", topology_rule="four_isl", n_laser_terminals=4, isl_capacity_mbps=200.0
)

# Realistic GEN2 constellations hardware have 3 ISLs
HW_THREE_ISL = HardwareConfig(
    hw_id="three_isl_bricks",
    topology_rule="three_isl_bricks",
    n_laser_terminals=3,
    isl_capacity_mbps=200.0,
)
