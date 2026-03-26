"""
File created at: 2026-03-25 18:35:33
Author: Sam Ghalayini
meteor/data/traffic.py
"""
# imports
from dataclasses import dataclass
from data.constellation import Topology
import numpy as np
from typing import List

# traffic class properties
@dataclass
class class_params:
    class_id : int
    d_f : float
    L_f : float
    W_f : float
    tau : float
    w : float 

CLASS_NAMES = ['voice', 'video', 'file']
DEFAULT_CLASS_WEIGHTS = [0.60, 0.25, 0.15]
CLASS_PARAMS = {
    0 : [0, 0.064, 640, 0, 0.050, 0.6],
    1 : [8.0, 8e6, 0, 0.500, 0.25],
    2 : [50.0, 50e6, 0, 10.0, 0.15]
    }
DEFAULT_OFFLOAD_PROB = np.array([0.0, 0.05, 0.30]) # voice, video, and file offloading probability

@dataclass
class FlowTable:
    flow_id : np.ndarray
    src_sat : np.ndarray
    dst_sat : np.ndarray
    class_id : np.ndarray
    d_f : np.ndarray
    L_f : np.ndarray
    W_f : np.ndarray
    tau : np.ndarray
    w : np.ndarray
    is_offload : np.ndarray

    @property
    def n_flows(self) -> int:
        """ Returns total number of flows in table"""
        return len(self.flow_id)

def build_population_weights(topology : Topology) -> np.ndarray:
    """Create probability distribution over satellite IDs based off of population density, determines where traffic originates and terminates from, minimal to no traffic in oceans and deserts or when inclination degree is reached"""
    population_centers = np.array([ # city name, lat, lon, population in millions
        ("tokyo", 0, 0, 37) 
    ])

    graph = topology.get_graph()
    sats = np.zeros((graph.number_of_nodes(),))
    for i, pop_center in enumerate(population_centers):
        tmp = np.zeros((graph.number_of_nodes(),))
        for nid, d in graph.nodes(data=True):
            tmp[nid] = topology._ground_haversine(pop_center[1], pop_center[2], d['lat'], d['lon'])
        sats[i] = min(tmp)
    for s in sats:
        if s == 0:
            s = 0.05
    sats = sats.softmax()
    return sats

def generate_flows(topology : Topology, n_flows : int, class_mix, offload_prob, rng) -> FlowTable:
    """Sample a cmoplete flow table for one TE interval"""
    # generate demands from populus areas, maybe I need to adjust the number of cities lat/lon
    # global network of only 369 satellites won't work, I think I need to research the minimum amount of satllites 
    # use a greater topology presets I think, something that makes sense so there are satellites over cities and satellites over desert and oceans 
    # 369 seems too small
    
    return None
    
def compute_total_demand(flows) -> float:
    """sum the demand for all flows in Mbps"""
    return flows.d_f.sum()

def compute_network_capacity(topology : Topology) -> float:
    """estimate total network throughput capcaity"""
    return topology.graph.number_of_edges() * topology.config.C_e

def scale_to_load(topology, n_flows_initial, target_load, class_mix, offload_prob, rng) -> FlowTable:
    """ generate flow table whose total demand is a fraction of total network, not completely loading the network"""
    return None

def get_offloading_flows(flows) -> np.ndarray:
    """ return indicies of offloading flow table"""
    return np.where(flows.is_offload)[0]

def get_class_flows(flows, class_id) -> np.ndarray:
    """return indices of a certain data class"""
    return np.where(flows.class_id == class_id)[0]

def summarize_flows(flows) -> dict:
    """print/log stats about flow table
    total flow count, per class flow count, percentages, offloading flow count, total demand in mbps, per class demand, min/max/mean workload, num of unique sat sources and dst, just a quick check"""
    return {}

