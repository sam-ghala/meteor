"""
File created at: 2026-05-01 23:21:36
Author: Sam Ghalayini
meteor/meteor/config/registry.py

Lookup table for YAML configs that reference Python objects by name
"""

from __future__ import annotations

from meteor.config.constellation import ConstellationConfig
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

PRESETS: dict[str, ConstellationConfig] = {
    "GEN2_FULL_FOUR_ISL": GEN2_FULL_FOUR_ISL,
    "GEN2_FULL_THREE_ISL": GEN2_FULL_THREE_ISL,
    "GEN2_S1_ONLY_FOUR_ISL": GEN2_S1_ONLY_FOUR_ISL,
    "GEN2_S1_ONLY_THREE_ISL": GEN2_S1_ONLY_THREE_ISL,
    "IRIDIUM_THREE_ISL": IRIDIUM_THREE_ISL,
    "IRIDIUM_FOUR_ISL": IRIDIUM_FOUR_ISL,
    "STARLINK_FULL_FOUR_ISL": STARLINK_FULL_FOUR_ISL,
    "STARLINK_FULL_THREE_ISL": STARLINK_FULL_THREE_ISL,
    "STARLINK_MID_FOUR_ISL": STARLINK_MID_FOUR_ISL,
    "STARLINK_MID_THREE_ISL": STARLINK_MID_THREE_ISL,
    "STARLINK_S1_ONLY_FOUR_ISL": STARLINK_S1_ONLY_FOUR_ISL,
    "STARLINK_S1_ONLY_THREE_ISL": STARLINK_S1_ONLY_THREE_ISL,
    "STARLINK_TWO_SHELL_FOUR_ISL": STARLINK_TWO_SHELL_FOUR_ISL,
    "STARLINK_TWO_SHELL_THREE_ISL": STARLINK_TWO_SHELL_THREE_ISL,
}


def get_preset(name: str) -> ConstellationConfig:
    """Lookup preset by name"""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset: {name!r}, Available: {available}")
    return PRESETS[name]
