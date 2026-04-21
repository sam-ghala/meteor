"""
File created at: 2026-04-12 14:22:33
Author: Sam Ghalayini
meteor/solvers/class_blind/throughput_gurobi.py

Maximize throughput with all physical network constraints.
Same constraints as QoS Gurobi, different objective.
"""

# %%
import time
from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from data.paths import PathData
from data.traffic import FlowTable
from solvers.base import TEconfig, TEresult


def solve_gurobi(
    flows: FlowTable,
    path_data: PathData,
    server_mus: list[float],
    config: TEconfig,
) -> TEresult:
    t0 = time.perf_counter()

    F = flows.n_flows
    P = path_data.n_paths
    S = len(server_mus)
    E = path_data.n_edges
    Ce = 200.0
    C_up = 50.0
    C_dn = 50.0

    model = gp.Model("meteor_throughput")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 60)

    # variables
    x = model.addMVar(P, lb=0.0, name="x")  # traffic allocation
    z = model.addMVar(F, vtype=GRB.BINARY, name="z")  # admission
    y = model.addMVar((F, S), vtype=GRB.BINARY, name="y")  # server assignment

    # precompute flow-to-satellite mappings
    src_flows = defaultdict(list)
    dst_flows = defaultdict(list)
    for f in range(F):
        src_flows[int(flows.src_sat[f])].append(f)
        if not flows.is_offload[f]:
            dst_flows[int(flows.dst_sat[f])].append(f)

    # Network capacity constraints

    # (8) ISL link capacity
    model.addMConstr(path_data.Phi.T, x, GRB.LESS_EQUAL, np.full(E, Ce), name="link_cap")

    # (9) uplink capacity
    for n, flist in src_flows.items():
        terms = []
        for f in flist:
            start = path_data.flow_path_start[f]
            end = path_data.flow_path_end[f]
            if end > start:
                terms.append(x[start:end].sum())
        if terms:
            model.addConstr(gp.quicksum(terms) <= C_up, name=f"uplink_{n}")

    # (10) downlink capacity
    for n, flist in dst_flows.items():
        terms = []
        for f in flist:
            start = path_data.flow_path_start[f]
            end = path_data.flow_path_end[f]
            if end > start:
                terms.append(x[start:end].sum())
        if terms:
            model.addConstr(gp.quicksum(terms) <= C_dn, name=f"downlink_{n}")

    # Per-flow admission constraints

    # (11) demand cap: sum x_fp <= d_f * z_f
    for f in range(F):
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        if end > start:
            model.addConstr(x[start:end].sum() <= flows.d_f[f] * z[f], name=f"demand_{f}")

    # (12) minimum bandwidth: sum x_fp >= epsilon * z_f
    for f in range(F):
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        if end > start:
            model.addConstr(x[start:end].sum() >= config.epsilon * z[f], name=f"minbw_{f}")

    # Server assignment constraints

    # (13) each admitted offloading flow assigned to exactly one server
    for f in range(F):
        if flows.is_offload[f]:
            model.addConstr(y[f, :].sum() == z[f], name=f"server_assign_{f}")
        else:
            model.addConstr(y[f, :].sum() == 0, name=f"no_server_{f}")

    # (14) server capacity
    for s in range(S):
        model.addConstr(
            gp.quicksum(y[f, s] * flows.W_f[f] for f in range(F) if flows.is_offload[f])
            <= server_mus[s] * config.delta,
            name=f"server_cap_{s}",
        )

    # (15) path-server coupling
    for f in range(F):
        if not flows.is_offload[f]:
            continue
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        for p in range(start, end):
            s = path_data.paths[p].server_id
            if s is not None:
                model.addConstr(x[p] <= flows.d_f[f] * y[f, s], name=f"path_server_{f}_{p}")

    # Objective: maximize throughput
    model.setObjective(x.sum(), GRB.MAXIMIZE)

    # solve
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        x_sol = x.X
        z_sol = z.X
        y_sol = y.X
    else:
        print(f"Gurobi status: {model.status}")
        x_sol = np.zeros(P)
        z_sol = np.zeros(F)
        y_sol = np.zeros((F, S))

    v_sol = np.zeros(F)

    solve_time = time.perf_counter() - t0
    return TEresult(x_sol, y_sol, z_sol, v_sol, solve_time)


# %%
