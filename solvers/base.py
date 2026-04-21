"""
File created at: 2026-04-10 08:37:47
Author: Sam Ghalayini
meteor/solvers/base.py

define standard contract that every solver fits to for comparison
"""

from dataclasses import dataclass, field

import numpy as np

from data.paths import PathData
from data.traffic import FlowTable


@dataclass
class TEconfig:
    M_big: float = 1e6
    w_c: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.3, 0.2]))  # class weights
    M_drop: float = 1000.0
    lam: float = 10.0
    epsilon: float = 0.0001  # min bandwidth
    delta: float = 1.0  # time interval


@dataclass
class TEresult:
    x: np.ndarray  # (P,) bandwidth for each path
    y: np.ndarray  # (F,S) server assignment for each offloading flow
    z: np.ndarray  # (F,) admission indicator
    v: np.ndarray  # (F,) deadline slack (seconds)
    solve_time: float

    @property
    def num_flows(self) -> int:
        return len(self.z)

    @property
    def num_paths(self) -> int:
        return len(self.x)


def delay(
    result: TEresult,
    flows: FlowTable,
    path_data: PathData,
    server_mus: list[float],
    config: TEconfig,
) -> np.ndarray:
    """
    Compute time-delay for time interval solution
    inputs:
        - bandwidth - traffic allocation (mbps per path)
        - server_assign - for each offloading flow, which server is computing its workload
        - admission - which flows were admitted for this time interval
        - slack - when the solver cannot satify al deadlines then include a deadline slack variable in which each admitted flow can increase the deadline by a certain amount, each non-admitted flow has M_big penalty for dropping the flow
    outputs:
        - weighted sum of normalized per-class delays
    """

    delays = np.zeros((flows.n_flows,))
    for f in flows.flow_id:
        if result.z[f] < 0.5:
            continue
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        x_f = result.x[start:end]
        total_bw = x_f.sum()
        if total_bw < 1e-12:  # near 0 allocation
            continue
        # transmission
        transmission_delay = flows.L_f[f] / (total_bw * 1e6)  # bits / mbps -> second

        # propagation delay
        weights = x_f / total_bw
        prop_delays_f = path_data.prop_delays[start:end]
        prop_delay = np.dot(weights, prop_delays_f)

        # communication propagation
        if not flows.is_offload[f]:
            comp_delay = 0

        # offloading computation
        else:
            assigned = np.where(result.y[f] > 0.5)[0]
            if len(assigned) == 0:
                comp_delay = 0
            else:
                s = assigned[0]
                mu_s = server_mus[s]
                comp_delay = flows.W_f[f] / mu_s

        delays[f] = transmission_delay + prop_delay + comp_delay

    return delays
