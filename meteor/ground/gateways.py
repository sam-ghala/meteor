"""
File created at: 2026-05-01 08:14:35
Author: Sam Ghalayini
meteor/meteor/constellation/gateways.py

Ground gateways server two purposes in METEOR:
- Internet access and have attached server computational capacity
- Bent pipe relay connection back to another satellite
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gateway:
    id: str
    lat: float
    lon: float
    has_server: bool = False
    mu_s: float = 0.0  # CPU cycles / sec
    uplink_mbps: float = 50.0
    downlink_mbps: float = 50.0


def get_global_gateways() -> list[Gateway]:
    standard_mu = 10e9
    return [
        Gateway("gw_seattle", 47.602, -122.3321, has_server=True, mu_s=standard_mu),
        Gateway("gw_new_york", 40.7128, -74.0060, has_server=True, mu_s=standard_mu),
        Gateway("gw_texas", 31.9686, -99.9018, has_server=False),
        Gateway("gw_london", 51.5074, -0.1278, has_server=True, mu_s=standard_mu),
        Gateway("gw_frankfurt", 50.1109, 8.6821, has_server=False),
        Gateway("gw_tokyo", 35.6762, 139.6503, has_server=True, mu_s=standard_mu),
        Gateway("gw_singapore", 1.3521, 103.8198, has_server=False),
        Gateway("gw_sydney", -33.8688, 151.2093, has_server=True, mu_s=standard_mu),
        Gateway("gw_sao_paulo", -23.5505, -46.6333, has_server=False),
        Gateway("gw_cape_town", -33.9249, 18.4241, has_server=False),
    ]


def gateway_position_array(
    gateways: list[Gateway],
    earth_radius_km: float,
) -> np.ndarray:
    if not gateways:
        return np.empty((0, 3), dtype=np.float64)
    from meteor.constellation.kinematics import gateways_xyz

    lats = np.array([gw.lat for gw in gateways])
    lons = np.array([gw.lon for gw in gateways])
    return gateways_xyz(lats, lons, earth_radius_km)
