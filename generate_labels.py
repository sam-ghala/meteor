"""
File created at: 2026-04-21 11:33:21
Author: Sam Ghalayini
meteor/generate_labels.py
Generate QoS Gurobi labels across topology snapshots and traffic samples.
"""

import argparse
import json
import logging
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np

from data.constellation import Topology
from data.paths import build_path_data
from data.traffic import CLASS_PARAMS, scale_to_load
from solvers.base import TEconfig, delay
from solvers.class_blind.throughput_gurobi import solve_gurobi as solve_throughput
from solvers.ecmp.ecmp import solve_ecmp
from solvers.qos_gurobi.qos_gurobi import solve_qos_gurobi

logging.getLogger("gurobipy").setLevel(logging.ERROR)


def generate_one_instance(args):
    """Generate one (topology, traffic, solution) and save to disk."""
    time_t, load, seed, config_dict, output_dir, instance_id, verbose = args

    fpath = os.path.join(output_dir, f"{instance_id}.npz")

    if os.path.exists(fpath):
        return {"id": instance_id, "status": "skipped"}

    try:
        config = TEconfig(**config_dict)
        rng = np.random.default_rng(seed)

        # build topology at this orbital time
        topo = Topology(time_t=time_t)
        graph = topo.get_graph()
        server_sat_ids = topo.get_server_sat_ids()
        server_mus = [graph.nodes[s]["mu_server"] for s in server_sat_ids]

        # generate traffic at target load
        flows = scale_to_load(topo, load, rng=rng)

        # compute candidate paths
        path_data = build_path_data(
            graph, flows, server_sat_ids, k=10, k_per_server=3, verbose=verbose
        )

        # solve QoS optimization
        result = solve_qos_gurobi(flows, path_data, server_mus, config)

        # compute real delays for quality metrics
        delays = delay(result, flows, path_data, server_mus, config)

        # solve throughput baseline
        result_tp = solve_throughput(flows, path_data, server_mus, config)
        delays_tp = delay(result_tp, flows, path_data, server_mus, config)

        # solve ECMP baseline
        result_ecmp = solve_ecmp(flows, path_data, server_mus, config)
        delays_ecmp = delay(result_ecmp, flows, path_data, server_mus, config)

        # ── package data ──
        positions = np.array(
            [[graph.nodes[n]["lat"], graph.nodes[n]["lon"]] for n in range(topo.N)],
            dtype=np.float32,
        )
        Phi_coo = path_data.Phi.tocoo()

        path_server_ids = np.array(
            [p.server_id if p.server_id is not None else -1 for p in path_data.paths],
            dtype=np.int32,
        )

        path_to_flow = np.zeros(path_data.n_paths, dtype=np.int32)
        for f in range(flows.n_flows):
            path_to_flow[path_data.flow_path_start[f] : path_data.flow_path_end[f]] = f

        # ── save ──
        np.savez_compressed(
            fpath,
            # metadata
            meta=np.array(
                [
                    time_t,
                    load,
                    seed,
                    flows.n_flows,
                    path_data.n_paths,
                    path_data.n_edges,
                    topo.N,
                    result.solve_time,
                    result_tp.solve_time,
                    result_ecmp.solve_time,
                ],
                dtype=np.float64,
            ),
            # topology
            positions=positions,
            edges=np.array(list(graph.edges()), dtype=np.int32),
            server_sat_ids=np.array(server_sat_ids, dtype=np.int32),
            server_mus=np.array(server_mus, dtype=np.float64),
            # flow features
            d_f=flows.d_f.astype(np.float32),
            L_f=flows.L_f.astype(np.float32),
            W_f=flows.W_f.astype(np.float32),
            class_id=flows.class_id.astype(np.int8),
            is_offload=flows.is_offload.astype(np.int8),
            src_sat=flows.src_sat.astype(np.int32),
            dst_sat=flows.dst_sat.astype(np.int32),
            # path structure
            flow_path_start=path_data.flow_path_start.astype(np.int32),
            flow_path_end=path_data.flow_path_end.astype(np.int32),
            prop_delays=path_data.prop_delays.astype(np.float32),
            path_server_ids=path_server_ids,
            path_to_flow=path_to_flow,
            # Phi sparse (COO)
            Phi_row=Phi_coo.row.astype(np.int32),
            Phi_col=Phi_coo.col.astype(np.int32),
            Phi_data=Phi_coo.data.astype(np.float32),
            Phi_shape=np.array(Phi_coo.shape, dtype=np.int32),
            # labels
            x=result.x.astype(np.float32),
            z=result.z.astype(np.float32),
            y=result.y.astype(np.float32),
            v=result.v.astype(np.float32),
            # evaluation
            delays=delays.astype(np.float32),
            # throughput labels
            x_tp=result_tp.x.astype(np.float32),
            z_tp=result_tp.z.astype(np.float32),
            y_tp=result_tp.y.astype(np.float32),
            delays_tp=delays_tp.astype(np.float32),
            # hops
            hop_counts=path_data.hop_counts.astype(np.int32),
            # ECMP labels
            x_ecmp=result_ecmp.x.astype(np.float32),
            z_ecmp=result_ecmp.z.astype(np.float32),
            y_ecmp=result_ecmp.y.astype(np.float32),
            delays_ecmp=delays_ecmp.astype(np.float32),
        )

        # ── summary metrics ──
        n_admitted = int((result.z > 0.5).sum())
        voice_mask = (flows.class_id == 0) & (result.z > 0.5)
        video_mask = (flows.class_id == 1) & (result.z > 0.5)
        voice_oc = delays[voice_mask].mean() / CLASS_PARAMS[0].tau if voice_mask.sum() > 0 else -1
        video_oc = delays[video_mask].mean() / CLASS_PARAMS[1].tau if video_mask.sum() > 0 else -1

        return {
            "id": instance_id,
            "status": "ok",
            "time_t": time_t,
            "load": load,
            "seed": int(seed),
            "n_flows": flows.n_flows,
            "n_paths": path_data.n_paths,
            "n_admitted": n_admitted,
            "solve_time": round(result.solve_time, 2),
            "voice_oc": round(float(voice_oc), 4),
            "video_oc": round(float(video_oc), 4),
            "file_mb": round(os.path.getsize(fpath) / 1e6, 1),
        }

    except Exception as e:
        return {
            "id": instance_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate METEOR QoS Gurobi labels")
    parser.add_argument("--output_dir", default="/workspace/labels")
    parser.add_argument("--n_time_steps", type=int, default=50)
    parser.add_argument("--n_traffic_seeds", type=int, default=10)
    parser.add_argument("--loads", nargs="+", type=float, default=[0.15, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time_limit", type=int, default=120)
    parser.add_argument(
        "--verbose", action="store_true", help="show path computation progress bars"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_dict = {
        "M_big": 10.0,
        "w_c": [0.60, 0.25, 0.15],
        "M_drop": 10.0,
        "lam": 10.0,
        "epsilon": 0.0001,
        "delta": 1.0,
    }

    # build job list across orbital period
    orbital_period = 5760.0  # ~96 min for 550km LEO
    time_steps = np.linspace(0, orbital_period, args.n_time_steps, endpoint=False)

    jobs = []
    for t_idx, time_t in enumerate(time_steps):
        for load in args.loads:
            for seed_idx in range(args.n_traffic_seeds):
                seed = t_idx * 10000 + seed_idx * 100 + int(load * 100)
                instance_id = f"t{t_idx:03d}_load{int(load * 100):02d}_s{seed_idx:02d}"
                jobs.append(
                    (
                        time_t,
                        load,
                        seed,
                        config_dict,
                        args.output_dir,
                        instance_id,
                        args.verbose,
                    )
                )

    total = len(jobs)

    # ── header ──
    print(f"\n{'=' * 70}")
    print(" METEOR Label Generation")
    print(f" {total} instances | {args.workers} workers | loads {args.loads}")
    print(
        f" {args.n_time_steps} topologies × {args.n_traffic_seeds} seeds × {len(args.loads)} loads"
    )
    print(f" output: {args.output_dir}")
    print(f"{'=' * 70}")

    # ── run ──
    t_start = time.perf_counter()
    results = []
    done, ok, failed = 0, 0, 0
    header_printed = False

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(generate_one_instance, jobs):
            if not header_printed:
                print(
                    f"\n  {'progress':<33} {'instance':<20} {'flows/admitted':<16} "
                    f"{'solve':<8} {'voice/video O_c':<16} {'throughput'}"
                )
                header_printed = True

            done += 1
            elapsed = time.perf_counter() - t_start

            if result["status"] == "failed":
                failed += 1
                print(f"\n  ✗ {result['id']}: {result['error']}")

            elif result["status"] == "ok":
                ok += 1
                rate = ok / elapsed * 3600
                eta = (total - done) / (done / elapsed)

                bar_len = 30
                filled = int(bar_len * done / total)
                bar = "█" * filled + "░" * (bar_len - filled)

                print(
                    f"\r  [{bar}] {done}/{total} "
                    f"| {result['id']:<18} "
                    f"| {result['n_flows']}f {result['n_admitted']}a{'':<4} "
                    f"| {result['solve_time']:>5}s "
                    f"| v={result['voice_oc']} d={result['video_oc']:<6} "
                    f"| {rate:.0f}/hr ETA {eta/3600:.1f}h",
                    end="",
                    flush=True,
                )

            results.append(result)

    # ── summary ──
    elapsed = time.perf_counter() - t_start
    print(f"\n\n{'=' * 70}")
    print(f" Done in {elapsed / 3600:.1f}h")
    print(f" {ok} completed | {failed} failed | {total - ok - failed} skipped")

    if ok > 0:
        solve_times = [r["solve_time"] for r in results if r["status"] == "ok"]
        sizes = [r["file_mb"] for r in results if r["status"] == "ok"]
        print(f" Solve time: {np.mean(solve_times):.1f}s avg, {max(solve_times):.1f}s max")
        print(f" File size:  {np.mean(sizes):.1f}MB avg, {sum(sizes):.0f}MB total")

    index_path = os.path.join(args.output_dir, "index.json")
    print(f" Index: {index_path}")
    print(f"{'=' * 70}")

    # ── save index ──
    with open(index_path, "w") as f:
        json.dump(
            {
                "total": total,
                "completed": ok,
                "failed": failed,
                "skipped": sum(1 for r in results if r["status"] == "skipped"),
                "config": config_dict,
                "loads": args.loads,
                "n_time_steps": args.n_time_steps,
                "n_traffic_seeds": args.n_traffic_seeds,
                "instances": [r for r in results if r["status"] == "ok"],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
