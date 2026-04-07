"""
File created at: 2026-03-25 18:35:33
Author: Sam Ghalayini
meteor/data/traffic.py
"""

# imports
from dataclasses import dataclass

import numpy as np

from data.constellation import Topology


# traffic class properties
@dataclass(frozen=True)
class class_params:
    class_id: int
    name: str
    d_f: float
    L_f: float
    tau: float
    w: float


CLASS_NAMES = ["voice", "video", "file"]
CLASS_PARAMS = {
    0: class_params(0, "voice", d_f=0.064, L_f=640.0, tau=0.050, w=0.5),
    1: class_params(1, "video", d_f=8.0, L_f=8e6, tau=0.500, w=0.3),
    2: class_params(2, "file", d_f=50.0, L_f=50e6, tau=10.0, w=0.2),
}

DEFAULT_CLASS_WEIGHTS = np.array([0.60, 0.25, 0.15])
DEFAULT_OFFLOAD_PROB = np.array([0.0, 0.05, 0.30])  # voice, video, and file offloading probability
# precompute lookup arrays of each data type param
_D_F = np.array([CLASS_PARAMS[c].d_f for c in range(3)])
_L_F = np.array([CLASS_PARAMS[c].L_f for c in range(3)])
_TAU = np.array([CLASS_PARAMS[c].tau for c in range(3)])
_W = np.array([CLASS_PARAMS[c].w for c in range(3)])


@dataclass
class FlowTable:
    flow_id: np.ndarray
    src_sat: np.ndarray
    dst_sat: np.ndarray
    class_id: np.ndarray
    d_f: np.ndarray
    L_f: np.ndarray
    W_f: np.ndarray
    tau: np.ndarray
    w: np.ndarray
    is_offload: np.ndarray

    @property
    def n_flows(self) -> int:
        """Returns total number of flows in table"""
        return len(self.flow_id)


# population weighting, so satellites have realistic and higher loads near population centers
# (lat, lon, population from current statistics)
POPULATION_CENTERS = [
    (35.68, 139.69, 37),  # Tokyo
    (28.61, 77.21, 30),  # Delhi
    (31.23, 121.47, 27),  # Shanghai
    (-23.55, -46.63, 22),  # São Paulo
    (19.43, -99.13, 21),  # Mexico City
    (30.04, 31.24, 20),  # Cairo
    (19.08, 72.88, 20),  # Mumbai
    (39.90, 116.40, 20),  # Beijing
    (23.81, 90.41, 17),  # Dhaka
    (34.69, 135.50, 19),  # Osaka
    (40.71, -74.01, 18),  # New York
    (24.86, 67.01, 16),  # Karachi
    (41.01, 28.98, 15),  # Istanbul
    (29.43, 106.91, 15),  # Chongqing
    (22.57, 88.36, 15),  # Kolkata
    (6.52, 3.38, 15),  # Lagos
    (-22.91, -43.17, 13),  # Rio
    (48.86, 2.35, 11),  # Paris
    (55.76, 37.62, 12),  # Moscow
    (51.51, -0.13, 9),  # London
]


def build_population_weights(topology: Topology) -> np.ndarray:
    """Create probability distribution over satellite IDs based off of population density, determines where traffic originates and terminates from, minimal to no traffic in oceans and deserts or when inclination degree is reached"""
    N = topology.N
    weights = np.zeros(N)
    R = topology.config.earth_radius
    sat_lats = np.array([topology.graph.nodes[i]["lat"] for i in range(N)])
    sat_lons = np.array([topology.graph.nodes[i]["lon"] for i in range(N)])
    sat_lats_rad = np.radians(sat_lats)
    sat_lons_rad = np.radians(sat_lons)

    for lat, lon, pop in POPULATION_CENTERS:
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # haversine city vs all satelites
        lat_diff = sat_lats_rad - lat_rad
        lon_diff = sat_lons_rad - lon_rad
        a = (
            np.sin(lat_diff / 2) ** 2
            + np.cos(lat_rad) * np.cos(sat_lats_rad) * np.sin(lon_diff / 2) ** 2
        )
        distances = 2 * R * np.arcsin(np.sqrt(a))
        closest = int(np.argmin(distances))
        weights[closest] += pop

    weights += 0.1  # non zero probability for all satellites
    weights /= weights.sum()
    return weights


def generate_flows(
    topology: Topology,
    n_flows: int,
    class_mix: np.ndarray = DEFAULT_CLASS_WEIGHTS,
    offload_prob: np.ndarray = DEFAULT_OFFLOAD_PROB,
    rng: np.random.Generator | None = None,
) -> FlowTable:
    """Sample a cmoplete flow table for one TE interval"""
    if rng is None:
        rng = np.random.default_rng()
    N = topology.N
    pop_weights = build_population_weights(topology)

    # sample data type classes
    class_ids = rng.choice(3, size=n_flows, p=class_mix)
    # sample source sats
    src_sats = rng.choice(N, size=n_flows, p=pop_weights)
    # sample offloading
    is_offload = rng.random(n_flows) < offload_prob[class_ids]
    # sampled dest sats
    dst_sats = np.full(n_flows, -1, dtype=int)

    d_f = _D_F[class_ids].copy()
    L_f = _L_F[class_ids].copy()
    tau = _TAU[class_ids].copy()
    w = _W[class_ids].copy()

    # src != dst collisions
    comm_mask = ~is_offload
    n_comm = comm_mask.sum()
    if n_comm > 0:
        dst_sats[comm_mask] = rng.choice(N, size=n_comm, p=pop_weights)

        collisions = comm_mask & (dst_sats == src_sats)
        while collisions.any():
            n_colided = collisions.sum()
            dst_sats[collisions] = rng.choice(N, size=n_colided, p=pop_weights)
            collisions = comm_mask & (dst_sats == src_sats)

    # sample workloads for offloading flows
    W_f = np.zeros(n_flows)
    n_offload = is_offload.sum()
    if n_offload > 0:
        W_f[is_offload] = 10.0 ** rng.uniform(7, 9, size=n_offload)

    return FlowTable(
        flow_id=np.arange(n_flows),
        src_sat=src_sats,
        dst_sat=dst_sats,
        class_id=class_ids,
        d_f=d_f,
        L_f=L_f,
        W_f=W_f,
        tau=tau,
        w=w,
        is_offload=is_offload,
    )


# utility and summarizing functions


def compute_total_demand(flows) -> float:
    """sum the demand for all flows in Mbps"""
    return float(flows.d_f.sum())


def compute_network_capacity(topology: Topology) -> float:
    """estimate total network throughput capcaity"""
    total = sum(d["capacity"] for _, _, d in topology.graph.edges(data=True))
    return total / 2.0


def scale_to_load(
    topology: Topology,
    target_load: float,
    class_mix: np.ndarray = DEFAULT_CLASS_WEIGHTS,
    offload_prob: np.ndarray = DEFAULT_OFFLOAD_PROB,
    rng: np.random.Generator | None = None,
) -> FlowTable:
    """generate flow table whose total demand is a fraction of total network, not completely loading the network"""
    if rng is None:
        rng = np.random.default_rng()

    net_capacity = compute_network_capacity(topology)
    target_demand = target_load * net_capacity

    # average demand per flow
    avg_demand = sum(class_mix[c] * CLASS_PARAMS[c].d_f for c in range(3))
    n_flows = max(1, int(target_demand / avg_demand))
    # genereate flows for time interval
    flows = generate_flows(topology, n_flows, class_mix, offload_prob, rng)

    return flows


def get_offloading_flows(flows: FlowTable) -> np.ndarray:
    """return indicies of offloading flow table"""
    return np.where(flows.is_offload)[0]


def get_class_flows(flows: FlowTable, class_id: int) -> np.ndarray:
    """return indices of a certain data class"""
    return np.where(flows.class_id == class_id)[0]


def summarize_flows(flows: FlowTable) -> dict:
    """print/log stats about flow table
    total flow count, per class flow count, percentages, offloading flow count, total demand in mbps, per class demand, min/max/mean workload, num of unique sat sources and dst, just a quick check"""
    n = flows.n_flows
    offload_idx = get_offloading_flows(flows)
    offload_W = flows.W_f[offload_idx]

    # summairize some key stats from flows
    n_voice = int((flows.class_id == 0).sum())
    n_video = int((flows.class_id == 1).sum())
    n_file = int((flows.class_id == 2).sum())
    demand_voice = float(flows.d_f[flows.class_id == 0].sum())
    demand_video = float(flows.d_f[flows.class_id == 1].sum())
    demand_file = float(flows.d_f[flows.class_id == 2].sum())
    summary = {
        "n_flows": n,
        "n_voice": n_voice,
        "n_video": n_video,
        "n_file": n_file,
        "ct_voice": 100 * n_voice / n if n else 0.0,
        "ct_video": 100 * n_video / n if n else 0.0,
        "ct_file": 100 * n_file / n if n else 0.0,
        "n_offload": len(offload_idx),
        "unique_sources": int(np.unique(flows.src_sat).size),
        "unique_destinations": int(np.unique(flows.dst_sat[flows.dst_sat >= 0]).size),
        "total_demand_mbps": demand_voice + demand_video + demand_file,
        "voice_demand_mbps": demand_voice,
        "video_demand_mbps": demand_video,
        "file_demand_mbps": demand_file,
    }

    if len(offload_W) > 0:
        summary["workload_mean"] = float(offload_W.mean())  # type: ignore
    else:
        summary["workload_mean"] = 0

    return summary
