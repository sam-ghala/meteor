"""
File created at: 2026-04-30 23:13:54
Author: Sam Ghalayini
meteor/meteor/config/__init__.py
"""

from meteor.config.constellation import ConstellationConfig, ISLThresholds
from meteor.config.hardware import HW_UNLIMITED, HW_V1_5, HW_V3, UNLIMITED_TERMINALS, HardwareConfig
from meteor.config.orbital import PHYSICS, PhysicalConstants, ShellConfig
from meteor.config.presets import (
    GEN2_FULL,
    GEN2_S1,
    GEN2_S1_ONLY,
    GEN2_S2,
    GEN2_S3,
    GEN2_S4,
    GEN2_S5,
    GEN2_S6_POLAR,
    IRIDIUM,
    IRIDIUM_SHELL,
    STARLINK_FULL,
    STARLINK_FULL_CROSS_SHELL_LASERS,
    STARLINK_MID_SMALL,
    STARLINK_S1,
    STARLINK_S1_ONLY,
    STARLINK_S2,
    STARLINK_S3,
    STARLINK_S4,
    STARLINK_TWO_SHELL,
)

__all__ = [
    # core configs
    "ShellConfig",
    "ConstellationConfig",
    "ISLThresholds",
    "PhysicalConstants",
    "PHYSICS",
    "UNLIMITED_TERMINALS",
    "HardwareConfig",
    # starlink shells
    "STARLINK_S1",
    "STARLINK_S2",
    "STARLINK_S3",
    "STARLINK_S4",
    # starlink constellations
    "STARLINK_S1_ONLY",
    "STARLINK_TWO_SHELL",
    "STARLINK_FULL",
    "STARLINK_FULL_CROSS_SHELL_LASERS",
    "STARLINK_MID_SMALL",
    # iridium
    "IRIDIUM_SHELL",
    "IRIDIUM",
    # Starlink GEN2
    "GEN2_S1",
    "GEN2_S2",
    "GEN2_S3",
    "GEN2_S4",
    "GEN2_S5",
    "GEN2_S6_POLAR",
    "GEN2_S1_ONLY",
    "GEN2_FULL",
    # hardware
    "HW_UNLIMITED",
    "HW_V1_5",
    "HW_V3",
]
