"""
File created at: 2026-04-30 11:20:10
Author: Sam Ghalayini
meteor/meteor/constellation/kinematics.py

satellite position computation
mapping where each satellite is at timestep t
"""

from __future__ import annotations

import logging
import math

import numpy as np

from meteor.config.constellation import ConstellationConfig
from meteor.config.orbital import PHYSICS, ShellConfig

logger = logging.getLogger(__name__)


def shell_positions(
    shell: ShellConfig,
    t: float,
    physics=PHYSICS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Positions for all satellites in a single shell at time t
    Returns:
        xyz: (n_sats, 3) ECEF positoins in km, side note: ECI (Earth-Centered Inertial) vs. ECEF (Earth-Centered, Earth-Fixed). ECI is a non-rotating frame fixed with respect to stars, ECEF rotates with the Earth
        lats_deg: (n_sats,) latitudes in degrees
        lons_deg: (n_sats,) Earth-fixed longitudes in degrees [-180, 180]
    """
    omega = shell.angular_velocity_rad_s
    incl = math.radians(shell.inclination_deg)
    raan_offset = math.radians(shell.raan_offset_deg)
    R = physics.earth_radius_km + shell.altitude_km

    plane_idx = np.arange(shell.n_planes)
    sat_idx = np.arange(shell.sats_per_plane)

    raans = (plane_idx / shell.n_planes) * 2 * math.pi + raan_offset
    phase_offsets = plane_idx * (math.pi / shell.n_planes) * shell.phase_offset_factor

    base_theta = (sat_idx / shell.sats_per_plane) * 2 * math.pi
    theta = base_theta[None, :] + phase_offsets[:, None] + omega * t

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_incl = math.sin(incl)
    cos_incl = math.cos(incl)

    lat_rad = np.arcsin(sin_incl * sin_theta)
    lon_rad = (
        np.arctan2(cos_incl * sin_theta, cos_theta)
        + raans[:, None]
        - physics.earth_rot_rate_rad_s * t
    )
    lon_rad = (lon_rad + math.pi) % (2 * math.pi) - math.pi

    cos_lat = np.cos(lat_rad)
    x = R * cos_lat * np.cos(lon_rad)
    y = R * cos_lat * np.sin(lon_rad)
    z = R * np.sin(lat_rad)

    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    lats_deg = np.degrees(lat_rad).ravel()
    lons_deg = np.degrees(lon_rad).ravel()

    return xyz, lats_deg, lons_deg


def constellations_positions(
    config: ConstellationConfig,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked positions across all shells at time t, index by global IDs"""
    n_total = config.n_satellites
    xyz = np.empty((n_total, 3), dtype=np.float64)
    lats = np.empty(n_total, dtype=np.float64)
    lons = np.empty(n_total, dtype=np.float64)

    offsets = config.shell_offsets
    for i, shell in enumerate(config.shells):
        s_xyz, s_lats, s_lons = shell_positions(shell, t, physics=config.physics)
        start, end = offsets[i], offsets[i + 1]
        xyz[start:end] = s_xyz
        lats[start:end] = s_lats
        lons[start:end] = s_lons

    return xyz, lats, lons


# helpful geometry functions


def pairwise_distance(xyz_a: np.ndarray, xyz_b: np.ndarray) -> np.ndarray:
    """Euclidian distance between corresponding rows of two position arrays"""
    diff = xyz_a - xyz_b
    return np.sqrt(np.sum(diff * diff, axis=-1))


def gateway_xyz(
    lat_deg: float,
    lon_deg: float,
    earth_radius_km: float = PHYSICS.earth_radius_km,
) -> np.ndarray:
    """Convert (lat, lon) ground point to ECEF cartisian"""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    cos_lat = math.cos(lat)
    return np.array(
        [
            earth_radius_km * cos_lat * math.cos(lon),
            earth_radius_km * cos_lat * math.sin(lon),
            earth_radius_km * math.sin(lat),
        ]
    )


def gateways_xyz(
    lats_deg: np.ndarray, lons_deg: np.ndarray, earth_radius_km: float = PHYSICS.earth_radius_km
) -> np.ndarray:
    """Vectorized version of gateway_xyz"""
    lats = np.radians(lats_deg)
    lons = np.radians(lons_deg)
    cos_lat = np.cos(lats)
    return np.stack(
        [
            earth_radius_km * cos_lat * np.cos(lons),
            earth_radius_km * cos_lat * np.sin(lons),
            earth_radius_km * np.sin(lats),
        ],
        axis=-1,
    )


def elev_matrix(sat_xyz: np.ndarray, gw_xyz: np.ndarray) -> np.ndarray:
    """
    Elevation angle of each satellie from each gateway
    Returns
        - (n_sats, n_gws) matrix in degrees, negative degrees are below the horizion
    Calculate angle between line-of-sight and zenith then subtrack elevation to get angle based off of the horizion, 25degree cutoff makes sense in terms of the horizion
    """
    los = sat_xyz[:, None, :] - gw_xyz[None, :, :]  # (n_sats, n_gws, 3)
    los_norm = np.linalg.norm(los, axis=-1)  # (n_sats, n_gws)
    gw_norm = np.linalg.norm(gw_xyz, axis=-1)  # (n_g,)

    cos_zenith = np.sum(los * gw_xyz[None, :, :], axis=-1) / (los_norm * gw_norm[None, :])
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return 90.0 - np.degrees(np.arccos(cos_zenith))
