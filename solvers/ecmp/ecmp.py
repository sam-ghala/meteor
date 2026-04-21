"""
File created at: 2026-04-12 10:59:42
Author: Sam Ghalayini
meteor/solvers/ecmp/ecmp.py
Equal Cost Multi-Path
"""

import time

import numpy as np

from data.paths import PathData
from data.traffic import FlowTable
from solvers.base import TEconfig, TEresult


def solve_ecmp(
    flows: FlowTable,
    path_data: PathData,
    server_mus: list[float],
    config: TEconfig,
) -> TEresult:
    """
    distribute bandwidth across each path evenly baseline, and pick closest server
    """
    t0 = time.perf_counter()
    x = np.zeros(path_data.n_paths)
    y = np.zeros((flows.n_flows, len(server_mus)))
    z = np.ones(flows.n_flows)
    v = np.zeros(flows.n_flows)

    for f in flows.flow_id:
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        n_paths = end - start

        if n_paths == 0:
            z[f] = 0.0
            continue

        if not flows.is_offload[f]:
            share = flows.d_f[f] / n_paths
            for p in range(start, end):
                x[p] = share
        else:
            server_id = -1
            server_prop = np.inf
            for p in range(start, end):
                if path_data.paths[p].server_id is not None:
                    if path_data.paths[p].prop_delay < server_prop:
                        server_prop = path_data.paths[p].prop_delay
                        server_id = path_data.paths[p].server_id

            if server_id < 0:
                z[f] = 0.0
                continue

            y[f, server_id] = 1

            server_paths = [
                p for p in range(start, end) if path_data.paths[p].server_id == server_id
            ]
            share = flows.d_f[f] / len(server_paths)

            for p in server_paths:
                x[p] = share

    return TEresult(x, y, z, v, solve_time=(time.perf_counter() - t0))
