"""
File created at: 2026-03-25 15:52:52
Author: Sam Ghalayini
meteor/data/constellations.py
"""
# Imports
from dataclasses import dataclass
import numpy as np
import networkx as nx

# Dataclass
"""
Hold all information for satellite constellation
"""
@dataclass
class orbital_config:
    planes : int = 22 # count
    sat_per_plane : int = 18 # count
    altitude : int = 550 # km
    inclination : float = 53.0 # Degrees
    lat_cutoff : float = 75.0 # Degrees
    C_e : float = 200.0 # Mbps
    uplink : float = 50.0 # Mbps
    downlink : float = 50.0 # Mbps

# Compute satellite positions for given time t

# build edge list
# dict d = {src, dst, distance_km, prop_delay_s, capacity_mbps}

# build networkX graph

# add ground stations

# Topology Class
class Topology: 
    """
    Handles complete topology
    inputs: time t
    outputs : topology snapshot, node list, edge list, network graph, gateway-satellite mapping, server locations
    """
    def __init__(self, time_t, config=orbital_config()):
        self.time_t = time_t
        self.config = config
        self.graph = nx.DiGraph()

        self._build_nodes()
        self._build_edges()

    def _build_nodes(self):
        """
        Build all nodes in constellation from dataclass and using starfield 
        """
        # use skyfield to calculate exact lat/lon for self.time_t

        for p in range(self.config.planes):
            for s in range(self.config.sat_per_plane):
                node_id = f"sat_{p}_{s}"

                tmp_lat = np.random.uniform(-80,80)
                tmp_lon = np.random.uniform(-180,180)

                self.graph.add_node(
                    node_id,
                    plane = p,
                    sat_idx=s,
                    lat=tmp_lat,
                    lon=tmp_lon,
                    has_server=False,
                    uplink=self.config.uplink,
                    downlink=self.config.downlink
                )

    def _build_edges(self):
        """
        add edges to graph, ISLS based on grid structure and lat constraints
        """
        for node,data in self.graph.nodes(data=True):
            p = data['plane']
            s = data['sat_idx']
            lat = data['lat']

            # intra-orbit links (always active)
            next_sat = f"sat_{p}_{(s+1) % self.config.sat_per_plane}"
            self.graph.add_edge(
                node, 
                next_sat, 
                capacity=self.config.C_e,
                delay=self._calc_prop_delay(node, next_sat)
            )
            # inter-orbit links (deactivate if lat > cutoff because of angle at poles)
            if abs(lat) <= self.config.lat_cutoff:
                right_plane = f"sat_{(p+1) % self.config.planes}_{s}"
                self.graph.add_edge(
                    node, 
                    right_plane,
                    capacity=self.config.C_e,
                    delay=self._calc_prop_delay(node, right_plane))

    def _calc_prop_delay(self, src, dst) -> float:
        # calc dist between and divide by speed of light
        return 0.01

    def get_graph(self):
        return self.graph

