"""
File created at: 2026-05-01 23:21:36
Author: Sam Ghalayini
meteor/meteor/config/registry.py

Lookup table for YAML configs that reference Python objects by name
"""

from __future__ import annotations

from meteor.config.constellation import ConstellationConfig
from meteor.config.presets import (
    GEN2_FULL,
    GEN2_S1_ONLY,
    IRIDIUM,
    STARLINK_FULL,
    STARLINK_FULL_CROSS_SHELL_LASERS,
    STARLINK_MID_SMALL,
    STARLINK_S1_ONLY,
)

PRESETS: dict[str, ConstellationConfig] = {
    "IRIDUM": IRIDIUM,
    "STARLINK_S1_ONLY": STARLINK_S1_ONLY,
    "STARLINK_MID_SMALL": STARLINK_MID_SMALL,
    "STARLINK_FULL_CROSS_SHELL_LASERS": STARLINK_FULL_CROSS_SHELL_LASERS,
    "STARLINK_FULL": STARLINK_FULL,
    "GEN2_S1_ONLY": GEN2_S1_ONLY,
    "GEN2_FULL": GEN2_FULL,
}


def get_preset(name: str) -> ConstellationConfig:
    """Lookup preset by name"""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset: {name!r}, Available: {available}")
    return PRESETS[name]
