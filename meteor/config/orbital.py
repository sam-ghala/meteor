"""
File created at: 2026-04-30 11:19:23
Author: Sam Ghalayini
meteor/meteor/config/orbital.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from meteor.config.hardware import HW_UNLIMITED, HardwareConfig


@dataclass(frozen=True)
class PhysicalConstants:
    """Universal constants shared across constellations"""

    earth_radius_km: float = 6371.0
    earth_rot_rate_rad_s: float = (
        2 * math.pi / 86400.0
    )  # one full revolution (2pi) / seconds in a day
    speed_of_light_km_s: float = 299792.458
    gravitational_parameter_km3_s2: float = 398600.4418  # gravational constant * earths mass


PHYSICS = PhysicalConstants()


@dataclass(frozen=True)
class ShellConfig:
    """
    Configuration for a single orbital shell at a set altitude

    A shell is a homogeneous set of satellies at the same altitude and inclination
    evenly distributed across n planes and with sat_per_plane satellites per plane

    Node IDs within a shell are local from 0 to (n_planes * sats_per_plane - 1),
    global ID's are assigned by ConstellationConfig with shell offsets
    """

    shell_id: str  # "starlink_s1_550"

    # geometry
    altitude_km: float  # orbital altitude above earth surface
    inclination_deg: float  # orbital plane inclination, 0 degree inclination is equatorial orbit, 90 degrees is polar orbit
    n_planes: int  # number of orbital planes
    sats_per_plane: int  # number of satellites per plane / one orbit loop
    # all satellites travel over the earth in great circles

    raan_offset_deg: float = (
        0.0  # global rotation of all RAANs (Right Ascension of the Ascending Nodes)
    )
    # range [0, 360], 0 degree alignment fixed to a star
    phase_offset_factor: float = 1.0  # inter-plane phase offset
    # the satellites in orbit 2 are shifted by half the distance to orbit 1
    # interlace stallites like brick laying for maximum coverage

    lat_cutoff_deg: float = 75.0  # inter-plane links break above this latitude
    hardware: HardwareConfig = field(default_factory=lambda: HW_UNLIMITED)

    @property
    def n_satellites(self) -> int:
        """Total satellites in this shell"""
        return self.n_planes * self.sats_per_plane

    @property
    def orbital_period_s(self) -> float:
        """Compute orbital period from altitude using Kepler's third law"""
        # T = 2 pi sqrt(a^3 / mu)
        a = PHYSICS.earth_radius_km + self.altitude_km
        return 2 * math.pi * math.sqrt(a**3 / PHYSICS.gravitational_parameter_km3_s2)

    @property
    def angular_velocity_rad_s(self) -> float:
        """Mean motion (orbital angular velocity)"""
        return 2 * math.pi / self.orbital_period_s
