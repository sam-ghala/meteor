"""
File created at: 2026-03-25 15:52:52
Author: Sam Ghalayini
meteor/data/constellations.py
"""

# Imports
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from data.ground_stations import get_global_gateways

# Dataclass
"""
Store physical parameters
"""


@dataclass
class orbital_config:
    planes: int = 22  # count
    sat_per_plane: int = 18  # count
    altitude: int = 550  # km
    inclination: float = 53.0  # Degrees
    lat_cutoff: float = 75.0  # Degrees
    C_e: float = 200.0  # Mbps
    uplink: float = 50.0  # Mbps
    downlink: float = 50.0  # Mbps
    orbital_period: float = 5739.0  # seconds
    earth_rot_speed: float = 2 * math.pi / 86400.0  # radians per second
    earth_radius: float = 6371.0  # km
    speed_of_light = 299792.458  # km/s


class Topology:
    """
    Complete topology snapshot at a given time t.

    Inputs: time_t (seconds), config (orbital parameters)
    Outputs: NetworkX DiGraph with satellite nodes, ISL edges,
             gateway-to-satellite mapping, server locations.

    Node IDs are integers: node_id = plane * sat_per_plane + sat_idx
    """

    def __init__(self, time_t: float, config: orbital_config = orbital_config()):
        self.time_t = time_t
        self.config = config
        self.N = config.planes * config.sat_per_plane
        self.graph = nx.DiGraph()

        self.gateways = get_global_gateways()
        self.gateway_mapping = {}  # gateway_id -> sat node_id (int)

        self._lats = np.zeros(self.N)
        self._lons = np.zeros(self.N)
        self._xyz = np.zeros((self.N, 3))

        self._compute_all_positions()
        self._build_nodes()
        self._build_edges()
        self._map_ground_stations()

    # Position computation
    def _compute_all_positions(self):
        """Compute lat/lon/xyz for all satellites at self.time_t from orbital mechantics."""
        cfg = self.config
        omega = 2 * math.pi / cfg.orbital_period
        incl = math.radians(cfg.inclination)
        R = cfg.earth_radius + cfg.altitude

        for p in range(cfg.planes):
            raan = (p / cfg.planes) * 2 * math.pi
            phase_offset = p * math.pi / cfg.planes

            for s in range(cfg.sat_per_plane):
                nid = p * cfg.sat_per_plane + s
                theta = (s / cfg.sat_per_plane) * 2 * math.pi + phase_offset + omega * self.time_t

                # Geocentric lat/lon
                lat_rad = math.asin(math.sin(incl) * math.sin(theta))
                lon_rad = (
                    math.atan2(math.cos(incl) * math.sin(theta), math.cos(theta))
                    + raan
                    - cfg.earth_rot_speed * self.time_t
                )
                lon_rad = (lon_rad + math.pi) % (2 * math.pi) - math.pi

                self._lats[nid] = math.degrees(lat_rad)
                self._lons[nid] = math.degrees(lon_rad)

                # straight line distance
                cos_lat = math.cos(lat_rad)
                self._xyz[nid] = [
                    R * cos_lat * math.cos(lon_rad),
                    R * cos_lat * math.sin(lon_rad),
                    R * math.sin(lat_rad),
                ]

    def _build_nodes(self):
        """ "
        Add satellite nodes, location information as well as if its spot beam connected with a server
        """
        cfg = self.config
        for p in range(cfg.planes):
            for s in range(cfg.sat_per_plane):
                nid = p * cfg.sat_per_plane + s
                self.graph.add_node(
                    nid,
                    plane=p,
                    sat_idx=s,
                    lat=self._lats[nid],
                    lon=self._lons[nid],
                    has_server=False,
                    mu_server=0.0,
                    uplink=cfg.uplink,
                    downlink=cfg.downlink,
                )

    def _build_edges(self):
        """
        Add ISL edges. Each pair is added exactly once in both directions.
        Intra-orbit: ring within each plane (always active).
        Inter-orbit: same index in adjacent plane (deactivate above lat_cutoff).
        """
        cfg = self.config

        for p in range(cfg.planes):
            for s in range(cfg.sat_per_plane):
                nid = p * cfg.sat_per_plane + s

                # --- Intra-orbit: forward link only (s -> s+1) ---
                next_s = (s + 1) % cfg.sat_per_plane
                next_id = p * cfg.sat_per_plane + next_s
                delay = self._isl_delay(nid, next_id)
                self.graph.add_edge(nid, next_id, capacity=cfg.C_e, delay=delay)
                self.graph.add_edge(next_id, nid, capacity=cfg.C_e, delay=delay)

                # --- Inter-orbit: rightward link only (p -> p+1) ---
                if p < cfg.planes - 1:
                    if abs(self._lats[nid]) <= cfg.lat_cutoff:
                        right_id = (p + 1) * cfg.sat_per_plane + s
                        if abs(self._lats[right_id]) <= cfg.lat_cutoff:
                            delay = self._isl_delay(nid, right_id)
                            self.graph.add_edge(nid, right_id, capacity=cfg.C_e, delay=delay)
                            self.graph.add_edge(right_id, nid, capacity=cfg.C_e, delay=delay)

    def _isl_delay(self, src_id: int, dst_id: int) -> float:
        """Straight-line distance between two satellites / speed of light."""
        diff = self._xyz[src_id] - self._xyz[dst_id]
        dist_km = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        return dist_km / self.config.speed_of_light

    def _map_ground_stations(self):
        """
        Map each gateway to its closest overhead satellite.
        Uses vectorised haversine over all satellites.
        """
        cfg = self.config
        sat_lats_rad = np.radians(self._lats)
        sat_lons_rad = np.radians(self._lons)

        for gw in self.gateways:
            gw_lat = math.radians(gw.lat)
            gw_lon = math.radians(gw.lon)

            # Vectorised haversine: gateway vs all satellites
            dlat = sat_lats_rad - gw_lat
            dlon = sat_lons_rad - gw_lon
            a = (
                np.sin(dlat / 2) ** 2
                + math.cos(gw_lat) * np.cos(sat_lats_rad) * np.sin(dlon / 2) ** 2
            )
            dists = 2 * cfg.earth_radius * np.arcsin(np.sqrt(a))

            closest_id = int(np.argmin(dists))
            self.gateway_mapping[gw.id] = closest_id

            if gw.has_server:
                self.graph.nodes[closest_id]["has_server"] = True
                self.graph.nodes[closest_id]["mu_server"] = gw.mu_s

    def _ground_haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Haversine distance on Earth's surface (km)."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return self.config.earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # public functions
    def node_id(self, plane: int, sat_idx: int) -> int:
        """Convert (plane, sat_idx) to integer node ID."""
        return plane * self.config.sat_per_plane + sat_idx

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_servers(self) -> dict:
        """Return {gateway_id: Gateway} for gateways that have servers."""
        return {gw.id: gw for gw in self.gateways if gw.has_server}

    def get_server_sat_ids(self) -> list:
        """Return list of satellite node IDs that have servers attached."""
        return [nid for nid, d in self.graph.nodes(data=True) if d["has_server"]]

    def get_snapshot(self) -> dict:
        return {
            "time_t": self.time_t,
            "graph": self.graph,
            "n_nodes": self.N,
            "n_edges": self.graph.number_of_edges(),
            "gateways": self.gateways,
            "servers": self.get_servers(),
            "server_sat_ids": self.get_server_sat_ids(),
            "gateway_mapping": self.gateway_mapping,
        }
