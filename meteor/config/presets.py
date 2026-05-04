"""
File created at: 2026-04-30 11:19:50
Author: Sam Ghalayini
meteor/meteor/config/presets.py

References:
- FCC 21-48: https://docs.fcc.gov/public/attachments/fcc-21-48a1.pdf
"""

from __future__ import annotations

from meteor.config.constellation import ConstellationConfig
from meteor.config.hardware import HW_FOUR_ISL, HW_THREE_ISL, HardwareConfig
from meteor.config.orbital import ShellConfig


def _shell(
    shell_id: str,
    altitude_km: float,
    inclination_deg: float,
    n_planes: int,
    sats_per_plane: int,
    hardware: HardwareConfig,
) -> ShellConfig:
    """Helper to construct a ShellConfig"""  # after refactoring and remaking so many shells this is a nice function to have
    return ShellConfig(
        shell_id=shell_id,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        n_planes=n_planes,
        sats_per_plane=sats_per_plane,
        hardware=hardware,
    )


# construction functions for all shells
# =====================================


# iridium shells
def _iridium_shell(hw: HardwareConfig) -> ShellConfig:
    return _shell("iridium780", 780.0, 86.4, 6, 11, hw)


# starlink small for iterations
def _starlink_s1_8x(hw):
    return _shell("starlink_s1_550_8x", 550.0, 53.0, 9, 22, hw)


def _starlink_s2_8x(hw):
    return _shell("starlink_s2_540_8x", 540.0, 53.2, 9, 22, hw)


# starlink full
def _starlink_s1(hw: HardwareConfig):
    return _shell("starlink_s1_550", 550.0, 53.0, 72, 22, hw)


def _starlink_s2(hw: HardwareConfig):
    return _shell("starlink_s2_540", 540.0, 53.2, 72, 22, hw)


def _starlink_s3(hw: HardwareConfig):
    return _shell("starlink_s3_570", 570.0, 70.0, 36, 20, hw)


def _starlink_s4(hw: HardwareConfig):
    return _shell("starlink_s4_560", 560.0, 97.6, 6, 58, hw)


# gen2 starlink
def _gen2_s1(hw: HardwareConfig):
    return _shell("gen2_s1_525", 525.0, 53.0, 28, 60, hw)


def _gen2_s2(hw: HardwareConfig):
    return _shell("gen2_s2_530", 530.0, 43.0, 28, 60, hw)


def _gen2_s3(hw: HardwareConfig):
    return _shell("gen2_s3_535", 535.0, 33.0, 28, 60, hw)


def _gen2_s4(hw: HardwareConfig):
    return _shell("gen2_s4_345", 345.0, 48.0, 24, 48, hw)


def _gen2_s5(hw: HardwareConfig):
    return _shell("gen2_s5_350", 350.0, 38.0, 24, 48, hw)


def _gen2_s6(hw: HardwareConfig):
    return _shell("gen2_s6_360_polar", 360.0, 96.9, 12, 30, hw)


# iridium constellation
# =====================
IRIDIUM_FOUR_ISL = ConstellationConfig(
    shells=(_iridium_shell(HW_FOUR_ISL),),
)
IRIDIUM_THREE_ISL = ConstellationConfig(
    shells=(_iridium_shell(HW_THREE_ISL),),
)

# starlink small constellations
# =============================
STARLINK_S1_ONLY_FOUR_ISL = ConstellationConfig(
    shells=(_starlink_s1(HW_FOUR_ISL),),
)
STARLINK_S1_ONLY_THREE_ISL = ConstellationConfig(
    shells=(_starlink_s1(HW_THREE_ISL),),
)

STARLINK_TWO_SHELL_FOUR_ISL = ConstellationConfig(
    shells=(_starlink_s1(HW_FOUR_ISL), _starlink_s2(HW_FOUR_ISL))
)
STARLINK_TWO_SHELL_THREE_ISL = ConstellationConfig(
    shells=(_starlink_s1(HW_THREE_ISL), _starlink_s2(HW_THREE_ISL))
)

STARLINK_MID_THREE_ISL = ConstellationConfig(
    shells=(_starlink_s1_8x(HW_THREE_ISL), _starlink_s2_8x(HW_THREE_ISL))
)
STARLINK_MID_FOUR_ISL = ConstellationConfig(
    shells=(_starlink_s1_8x(HW_FOUR_ISL), _starlink_s2_8x(HW_FOUR_ISL))
)

# starlink full constellation
# ===========================
STARLINK_FULL_THREE_ISL = ConstellationConfig(
    shells=(
        _starlink_s1(HW_THREE_ISL),
        _starlink_s2(HW_THREE_ISL),
        _starlink_s3(HW_THREE_ISL),
        _starlink_s4(HW_THREE_ISL),
    )
)
STARLINK_FULL_FOUR_ISL = ConstellationConfig(
    shells=(
        _starlink_s1(HW_FOUR_ISL),
        _starlink_s2(HW_FOUR_ISL),
        _starlink_s3(HW_FOUR_ISL),
        _starlink_s4(HW_FOUR_ISL),
    )
)

# gen2 starlink constellation
# ===========================
GEN2_S1_ONLY_THREE_ISL = ConstellationConfig(shells=(_gen2_s1(HW_THREE_ISL),))
GEN2_S1_ONLY_FOUR_ISL = ConstellationConfig(shells=(_gen2_s1(HW_FOUR_ISL),))

GEN2_FULL_THREE_ISL = ConstellationConfig(
    shells=(
        _gen2_s1(HW_THREE_ISL),
        _gen2_s2(HW_THREE_ISL),
        _gen2_s3(HW_THREE_ISL),
        _gen2_s4(HW_THREE_ISL),
        _gen2_s5(HW_THREE_ISL),
        _gen2_s6(HW_THREE_ISL),
    )
)
GEN2_FULL_FOUR_ISL = ConstellationConfig(
    shells=(
        _gen2_s1(HW_FOUR_ISL),
        _gen2_s2(HW_FOUR_ISL),
        _gen2_s3(HW_FOUR_ISL),
        _gen2_s4(HW_FOUR_ISL),
        _gen2_s5(HW_FOUR_ISL),
        _gen2_s6(HW_FOUR_ISL),
    )
)
