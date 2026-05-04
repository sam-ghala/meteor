"""
File created at: 2026-04-30 23:13:54
Author: Sam Ghalayini
meteor/meteor/config/__init__.py
"""

from meteor.config.constellation import ConstellationConfig, ISLThresholds
from meteor.config.hardware import HW_FOUR_ISL, HW_THREE_ISL, HardwareConfig, TopologyRule
from meteor.config.orbital import PHYSICS, PhysicalConstants, ShellConfig
from meteor.config.presets import (
    GEN2_FULL_FOUR_ISL,
    GEN2_FULL_THREE_ISL,
    GEN2_S1_ONLY_FOUR_ISL,
    GEN2_S1_ONLY_THREE_ISL,
    IRIDIUM_FOUR_ISL,
    IRIDIUM_THREE_ISL,
    STARLINK_FULL_FOUR_ISL,
    STARLINK_FULL_THREE_ISL,
    STARLINK_MID_FOUR_ISL,
    STARLINK_MID_THREE_ISL,
    STARLINK_S1_ONLY_FOUR_ISL,
    STARLINK_S1_ONLY_THREE_ISL,
    STARLINK_TWO_SHELL_FOUR_ISL,
    STARLINK_TWO_SHELL_THREE_ISL,
)
from meteor.config.registry import get_preset

__all__ = [
    # core configs
    "ShellConfig",
    "ConstellationConfig",
    "ISLThresholds",
    "PhysicalConstants",
    "PHYSICS",
    "TopologyRule",
    "HardwareConfig",
    "get_preset",
    "HW_FOUR_ISL",
    "HW_THREE_ISL",
    # starlink constellations
    "STARLINK_S1_ONLY_FOUR_ISL",
    "STARLINK_S1_ONLY_THREE_ISL",
    "STARLINK_TWO_SHELL_FOUR_ISL",
    "STARLINK_TWO_SHELL_THREE_ISL",
    "STARLINK_FULL_FOUR_ISL",
    "STARLINK_FULL_THREE_ISL",
    "STARLINK_MID_FOUR_ISL",
    "STARLINK_MID_THREE_ISL",
    # iridium
    "IRIDIUM_FOUR_ISL",
    "IRIDIUM_THREE_ISL",
    # Starlink GEN2
    "GEN2_S1_ONLY_FOUR_ISL",
    "GEN2_S1_ONLY_THREE_ISL",
    "GEN2_FULL_FOUR_ISL",
    "GEN2_FULL_THREE_ISL",
]
