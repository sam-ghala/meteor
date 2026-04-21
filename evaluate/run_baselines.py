"""
File created at: 2026-04-12 14:33:24
Author: Sam Ghalayini
meteor/run_baselines.py
"""

# %%
import numpy as np

from data.constellation import Topology
from data.paths import build_path_data, get_link_loads
from data.traffic import (
    CLASS_PARAMS,
    compute_network_capacity,
    scale_to_load,
    summarize_flows,
)
from solvers.base import TEconfig, delay
from solvers.class_blind.throughput_gurobi import solve_gurobi as solve_throughput
from solvers.ecmp.ecmp import solve_ecmp
from solvers.qos_gurobi.qos_gurobi import solve_qos_gurobi


def run_solver(name, solve_fn, flows, path_data, server_mus, config):
    print(f"\n --- {name} starting ---")
    result = solve_fn(flows, path_data, server_mus, config)
    delays = delay(result, flows, path_data, server_mus, config)
    loads = np.asarray(get_link_loads(path_data, result.x)).flatten()
    print(f"\n --- {name} finished ---")
    return result, delays, loads


def main():
    rng = np.random.default_rng(42)

    # build topology
    topo = Topology(time_t=0)
    graph = topo.get_graph()
    server_sat_ids = topo.get_server_sat_ids()
    server_mus = [graph.nodes[s]["mu_server"] for s in server_sat_ids]
    print(f"  {topo.N} sats, {graph.number_of_edges()} edges, {len(server_sat_ids)} servers")

    # generate flows
    target_load = 1  # percentage of maximum load on the constellation
    flows = scale_to_load(topo, target_load, rng=rng)
    s = summarize_flows(flows)  # get summary of flows
    net_cap = compute_network_capacity(topo)
    print(f"  target load: {target_load:.0%} of {net_cap:,.0f} Mbps")
    print(
        f"  {s['n_flows']} flows ({s['n_voice']} voice, {s['n_video']} video, {s['n_file']} file)"
    )
    print(f"  {s['n_offload']} offloading, {s['total_demand_mbps']:.1f} Mbps total demand")

    # compute paths
    path_data = build_path_data(graph, flows, server_sat_ids, k=10, k_per_server=3)
    print(f"  {path_data.n_paths} paths, {path_data.n_edges} edges")

    config = TEconfig()
    F = flows.n_flows

    # run solvers
    ecmp_r, ecmp_d, ecmp_l = run_solver("ECMP", solve_ecmp, flows, path_data, server_mus, config)
    thru_r, thru_d, thru_l = run_solver(
        "Throughput", solve_throughput, flows, path_data, server_mus, config
    )
    qos_r, qos_d, qos_l = run_solver("QoS", solve_qos_gurobi, flows, path_data, server_mus, config)

    # comparison table
    names = ["ECMP", "Throughput", "QoS"]
    results = [ecmp_r, thru_r, qos_r]
    delays_list = [ecmp_d, thru_d, qos_d]
    loads_list = [ecmp_l, thru_l, qos_l]

    print(f"\n{'=' * (24 + 14 * len(names))}")
    header = f"  {'':22}" + "".join(f"{n:>14}" for n in names)
    print(header)
    print(f"  {'-' * (22 + 14 * len(names))}")
    print(f"  {'throughput (Mbps)':<22}" + "".join(f"{r.x.sum():>14.1f}" for r in results))
    print(f"  {'admitted':<22}" + "".join(f"{r.z.sum():>14.0f}" for r in results))
    print(f"  {'max link load':<22}" + "".join(f"{ld.max():>14.1f}" for ld in loads_list))
    print(f"  {'solve time (s)':<22}" + "".join(f"{r.solve_time:>14.4f}" for r in results))

    # per-class mean delay
    for c, cname in enumerate(["voice", "video", "file"]):
        vals = []
        for d, r in zip(delays_list, results, strict=False):
            mask = (flows.class_id == c) & (r.z > 0.5)
            vals.append(d[mask].mean() * 1000 if mask.sum() > 0 else 0.0)
        print(f"  {cname + ' mean (ms)':<22}" + "".join(f"{v:>14.2f}" for v in vals))

    print(f"  {'-' * (22 + 14 * len(names))}")

    # per-class O_c
    for c, cname in enumerate(["voice", "video", "file"]):
        cp = CLASS_PARAMS[c]
        vals = []
        for d, r in zip(delays_list, results, strict=False):
            mask = flows.class_id == c
            admitted = r.z[mask] > 0.5
            if admitted.sum() > 0:
                vals.append(d[mask][admitted].mean() / cp.tau)
            else:
                vals.append(config.M_drop / cp.tau)
        print(f"  {cname + ' O_c':<22}" + "".join(f"{v:>14.4f}" for v in vals))

    # total slack
    print(f"  {'total slack (s)':<22}" + "".join(f"{r.v.sum():>14.4f}" for r in results))

    # per-class violation rate
    for c, cname in enumerate(["voice", "video", "file"]):
        cp = CLASS_PARAMS[c]
        vals = []
        for d, r in zip(delays_list, results, strict=False):
            mask = (flows.class_id == c) & (r.z > 0.5)
            if mask.sum() > 0:
                violated = (d[mask] > cp.tau).sum()
                vals.append(100.0 * violated / mask.sum())
            else:
                vals.append(100.0)
        print(f"  {cname + ' violate %':<22}" + "".join(f"{v:>14.1f}" for v in vals))

    # per-class max delay
    for c, cname in enumerate(["voice", "video", "file"]):
        vals = []
        for d, r in zip(delays_list, results, strict=False):
            mask = (flows.class_id == c) & (r.z > 0.5)
            vals.append(d[mask].max() * 1000 if mask.sum() > 0 else 0.0)
        print(f"  {cname + ' max (ms)':<22}" + "".join(f"{v:>14.1f}" for v in vals))

    # satisfied demand %
    sat_vals = []
    for r in results:
        admitted = r.z > 0.5
        if admitted.sum() > 0:
            demanded = flows.d_f[admitted].sum()
            allocated = sum(
                r.x[path_data.flow_path_start[f] : path_data.flow_path_end[f]].sum()
                for f in range(F)
                if r.z[f] > 0.5
            )
            sat_vals.append(100.0 * allocated / demanded if demanded > 0 else 0.0)
        else:
            sat_vals.append(0.0)
    print(f"  {'satisfied demand %':<22}" + "".join(f"{v:>14.1f}" for v in sat_vals))

    # trickle flows
    trickle_vals = []
    for r in results:
        count = 0
        for f in range(F):
            if r.z[f] > 0.5:
                bw = r.x[path_data.flow_path_start[f] : path_data.flow_path_end[f]].sum()
                if bw < 0.1 * flows.d_f[f]:
                    count += 1
        trickle_vals.append(count)
    print(f"  {'trickle flows (<10%)':<22}" + "".join(f"{v:>14}" for v in trickle_vals))

    # per-class admission count
    for c, cname in enumerate(["voice", "video", "file"]):
        vals = []
        for r in results:
            mask = (flows.class_id == c) & (r.z > 0.5)
            total = (flows.class_id == c).sum()
            vals.append(f"{mask.sum()}/{total}")
        print(f"  {cname + ' admitted':<22}" + "".join(f"{v:>14}" for v in vals))

    print(f"{'=' * (24 + 14 * len(names))}")


if __name__ == "__main__":
    main()
# %%
