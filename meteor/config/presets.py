"""
File created at: 2026-04-30 11:19:50
Author: Sam Ghalayini
meteor/meteor/config/presets.py

References:
- FCC 21-48: https://docs.fcc.gov/public/attachments/fcc-21-48a1.pdf
"""

from __future__ import annotations

from meteor.config.constellation import ConstellationConfig
from meteor.config.hardware import HW_V3
from meteor.config.orbital import ShellConfig

# iridium
# =======
IRIDIUM_SHELL = ShellConfig(
    shell_id="iridium_780",
    altitude_km=780.0,
    inclination_deg=86.4,
    n_planes=6,
    sats_per_plane=11,
)
IRIDIUM = ConstellationConfig(
    shells=(IRIDIUM_SHELL,),
)

# mid-sized constallation (396 satellites)
# ========================================
_S1_REDUCED_8X = ShellConfig(
    shell_id="starlink_s1_550_reduced_8x",
    altitude_km=550.0,
    inclination_deg=53.0,
    n_planes=9,
    sats_per_plane=22,
)

_S2_REDUCED_8X = ShellConfig(
    shell_id="starlink_s2_540_reduced_8x",
    altitude_km=540.0,
    inclination_deg=53.2,
    n_planes=9,
    sats_per_plane=22,
)
STARLINK_MID_SMALL = ConstellationConfig(
    shells=(_S1_REDUCED_8X, _S2_REDUCED_8X),
    ground_access_enabled=True,
)

# individual starlink shells
# ==========================
STARLINK_S1 = ShellConfig(
    shell_id="starlink_s1_550",
    altitude_km=550.0,
    inclination_deg=53.0,
    n_planes=72,
    sats_per_plane=22,
)

STARLINK_S2 = ShellConfig(
    shell_id="starlink_s2_540",
    altitude_km=540.0,
    inclination_deg=53.2,
    n_planes=72,
    sats_per_plane=22,
)

STARLINK_S3 = ShellConfig(
    shell_id="starlink_s3_570",
    altitude_km=570.0,
    inclination_deg=70.0,
    n_planes=36,
    sats_per_plane=20,
)

STARLINK_S4 = ShellConfig(
    shell_id="starlink_s4_560",
    altitude_km=560.0,
    inclination_deg=97.6,
    n_planes=6,
    sats_per_plane=58,
)
# starlink constallations at different scales
# ===========================================
STARLINK_S1_ONLY = ConstellationConfig(
    shells=(STARLINK_S1,),
)

STARLINK_TWO_SHELL = ConstellationConfig(
    shells=(
        STARLINK_S1,
        STARLINK_S2,
    ),
    cross_shell_lasers_enabled=True,
    ground_access_enabled=False,
)

STARLINK_FULL = ConstellationConfig(
    shells=(
        STARLINK_S1,
        STARLINK_S2,
        STARLINK_S3,
        STARLINK_S4,
    ),
    cross_shell_lasers_enabled=True,
    ground_access_enabled=True,
)

STARLINK_FULL_CROSS_SHELL_LASERS = ConstellationConfig(
    shells=(STARLINK_S1, STARLINK_S2, STARLINK_S3, STARLINK_S4),
    cross_shell_lasers_enabled=True,
    ground_access_enabled=False,
)

# GEN2 Constellation, V3 satellites (FCC, January 2026)
# =====================================================
GEN2_S1 = ShellConfig(
    shell_id="gen2_s1_525",
    altitude_km=525.0,
    inclination_deg=53.0,
    n_planes=28,
    sats_per_plane=60,
    hardware=HW_V3,
)

GEN2_S2 = ShellConfig(
    shell_id="gen2_s2_530",
    altitude_km=530.0,
    inclination_deg=43.0,
    n_planes=28,
    sats_per_plane=60,
    hardware=HW_V3,
)

GEN2_S3 = ShellConfig(
    shell_id="gen2_s3_535",
    altitude_km=535.0,
    inclination_deg=33.0,
    n_planes=28,
    sats_per_plane=60,
    hardware=HW_V3,
)

GEN2_S4 = ShellConfig(
    shell_id="gen2_s4_345",
    altitude_km=345.0,
    inclination_deg=48.0,
    n_planes=24,
    sats_per_plane=48,
    hardware=HW_V3,
)

GEN2_S5 = ShellConfig(
    shell_id="gen2_s5_350",
    altitude_km=350.0,
    inclination_deg=38.0,
    n_planes=24,
    sats_per_plane=48,
    hardware=HW_V3,
)

GEN2_S6_POLAR = ShellConfig(
    shell_id="gen2_s6_360_polar",
    altitude_km=360.0,
    inclination_deg=96.9,
    n_planes=12,
    sats_per_plane=30,
    hardware=HW_V3,
)

# GEN2 Constellations
# ===================
GEN2_S1_ONLY = ConstellationConfig(shells=(GEN2_S1,))

GEN2_FULL = ConstellationConfig(
    shells=(GEN2_S1, GEN2_S2, GEN2_S3, GEN2_S4, GEN2_S5, GEN2_S6_POLAR),
    cross_shell_lasers_enabled=True,
    ground_access_enabled=True,
)
