"""
File created at: 2026-04-19 18:21:11
Author: Sam Ghalayini
meteor/solvers/qos_gurobi/qos_gurobi.py
QoS-aware Gurobi
"""

# %%
import time
from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from data.paths import PathData
from data.traffic import CLASS_PARAMS, FlowTable
from solvers.base import TEconfig, TEresult


def solve_qos_gurobi(
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

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("meteor_qos", env=env)
    model.setParam("TimeLimit", 120)

    # Decision variables
    x = model.addMVar(P, lb=0.0, name="x")  # bandwidth per path
    z = model.addMVar(F, vtype=GRB.BINARY, name="z")  # admission
    y = model.addMVar((F, S), vtype=GRB.BINARY, name="y")  # server assignment
    v = model.addMVar(F, lb=0.0, name="v")  # deadline slack

    # Precompute T_approx: constant delay estimate per flow
    # T_approx = L_f/(d_f·1e6) + min_p(tau_p) + W_f/max(mu_s)
    #          (full demand)   (shortest path)   (fastest server)
    T_approx = np.zeros(F)
    for f in range(F):
        # transmission: L_f / (d_f * 1e6) — assumes full demand allocated
        if flows.d_f[f] > 0:
            T_approx[f] += flows.L_f[f] / (flows.d_f[f] * 1e6)

        # propagation: min over all paths — assumes mean path used
        # use min for optimistic flow admission and mean for a more realistic admission
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        if end > start:
            T_approx[f] += path_data.prop_delays[start:end].mean()

        # computation: W_f / max(mu_s) — assumes fastest server
        # use max() for assuming the fastest server
        # but this leads to a larger delay beacuse offloading flows that come from the same city/satellite will go to the same server
        # use np.mean()
        if flows.is_offload[f] and flows.W_f[f] > 0:
            T_approx[f] += flows.W_f[f] / np.mean(server_mus)

    # Precompute flow-to-satellite mappings
    src_flows = defaultdict(list)
    dst_flows = defaultdict(list)
    for f in range(F):
        src_flows[int(flows.src_sat[f])].append(f)
        if not flows.is_offload[f]:
            dst_flows[int(flows.dst_sat[f])].append(f)

    n_per_class = np.array([(flows.class_id == c).sum() for c in range(3)])

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

    # Deadline-derived minimum bandwidth (linearization)
    # force: sum x_fp >= (L_f / tau_c) * z_f  so transmission delay <= tau_c
    for f in range(F):
        c = int(flows.class_id[f])
        cp = CLASS_PARAMS[c]
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        if end > start:
            min_prop = path_data.prop_delays[start:end].min()
            remaining = cp.tau - min_prop
            if remaining > 0:
                min_bw = flows.L_f[f] / (remaining * 1e6)
            else:
                min_bw = flows.d_f[f]
            if min_bw > config.epsilon:
                model.addConstr(x[start:end].sum() >= min_bw * z[f], name=f"deadline_bw_{f}")

    # Server assignment constraints

    # (13) server assignment
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

    # QoS deadline constraint (7) — uses T_approx (constant)
    #
    # EXACT:   T_f <= tau_c + v_f + M_big·(1-z_f)      [T_f is a variable]
    # APPROX:  T_approx <= tau_c + v_f + M_big·(1-z_f)  [T_approx is a constant]
    #
    # When z_f=1: pay for slack violation if there is any
    # When z_f=0: M_drop violation
    for f in range(F):
        c = int(flows.class_id[f])
        cp = CLASS_PARAMS[c]
        model.addConstr(
            T_approx[f] <= cp.tau + v[f] + config.M_big * (1 - z[f]), name=f"deadline_{f}"
        )

    # Objective (linearized)
    obj = gp.LinExpr()

    # Term 1: admission cost
    # z_f · T_approx + (1-z_f) · M_drop = z_f · (T_approx - M_drop) + M_drop
    for f in range(F):
        c = int(flows.class_id[f])
        cp = CLASS_PARAMS[c]
        n_c = n_per_class[c]
        if n_c == 0:
            continue
        scale = config.w_c[c] / (n_c * cp.tau)
        coeff = (
            T_approx[f] - config.M_drop
        ) * scale  # negative number, good because we are minimizing
        obj += coeff * z[f]

    # Term 2: bandwidth incentive (linearization)
    # First-order Taylor of L_f/(sumx_fp·1e6) around sumx_fp = d_f:
    #   ≈ L_f/(d_f·1e6) - L_f/(d_f²·1e6) · (sumx_fp - d_f)
    # The gradient term -L_f/(d_f²·1e6) gives the optimizer a reason
    # to allocate bandwidth even though T_approx is fixed.
    for f in range(F):
        c = int(flows.class_id[f])
        cp = CLASS_PARAMS[c]
        n_c = n_per_class[c]
        if n_c == 0 or flows.d_f[f] < 1e-12:
            continue
        bw_weight = (
            config.w_c[c] / (n_c * cp.tau) * flows.L_f[f] / (flows.d_f[f] ** 2 * 1e6)
        )  # first order taylor approx f(x) + f'(x) * (a - x), a being full demand/optimistic solution
        # a in taylor approx is "expected operating point" of the function we want a first order/tangent line for
        start = path_data.flow_path_start[f]
        end = path_data.flow_path_end[f]
        for p in range(start, end):
            obj += -bw_weight * x[p]

    # Term 3: deadline violation penalty
    for f in range(F):
        obj += config.lam * v[f]

    model.setObjective(obj, GRB.MINIMIZE)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        x_sol = x.X
        z_sol = z.X
        y_sol = y.X
        v_sol = v.X
    else:
        print(f"Gurobi status: {model.status}")
        x_sol = np.zeros(P)
        z_sol = np.zeros(F)
        y_sol = np.zeros((F, S))
        v_sol = np.zeros(F)

    # n_offload_admitted = sum(1 for f in range(F) if flows.is_offload[f] and z_sol[f] > 0.5)
    # print(f"  offloading: {n_offload_admitted} / {int(flows.is_offload.sum())} admitted")

    solve_time = time.perf_counter() - t0
    return TEresult(x_sol, y_sol, z_sol, v_sol, solve_time)


# %%
