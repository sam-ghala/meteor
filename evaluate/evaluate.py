"""
File created at: 2026-04-21 15:21:24
Author: Sam Ghalayini
meteor/evaluate/evaluate.py
Standardized evaluation for any solver output.
Works with:
  1. Live solver runs (x, z, y, delays arrays + flow/path data)
  2. Saved .npz dataset files (loads everything from disk)

Outputs a flat metrics dict that can be saved as JSON and loaded into pandas.
"""

import json
import os

import numpy as np
from scipy.sparse import coo_matrix

from data.traffic import CLASS_PARAMS

CLASS_NAMES = {0: "voice", 1: "video", 2: "file"}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_arrays(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    delays: np.ndarray,
    d_f: np.ndarray,
    L_f: np.ndarray,
    class_id: np.ndarray,
    is_offload: np.ndarray,
    prop_delays: np.ndarray,
    flow_path_start: np.ndarray,
    flow_path_end: np.ndarray,
    Phi_row: np.ndarray,
    Phi_col: np.ndarray,
    Phi_data: np.ndarray,
    Phi_shape: tuple,
    solver_name: str = "unknown",
    solve_time: float = 0.0,
    Ce: float = 200.0,
    v: np.ndarray = None,
) -> dict:
    """
    Compute all evaluation metrics from raw arrays.
    This is the core function — everything else calls this.
    """
    F = len(z)
    P = len(x)

    if v is None:
        v = np.zeros(F)

    # ── link loads ──
    Phi = coo_matrix((Phi_data, (Phi_row, Phi_col)), shape=tuple(Phi_shape)).tocsr()
    link_loads = np.asarray(Phi.T @ x).flatten()
    max_link_load = float(link_loads.max()) if len(link_loads) > 0 else 0.0

    # ── global metrics ──
    admitted_mask = z > 0.5
    n_admitted = int(admitted_mask.sum())
    total_throughput = float(x.sum())

    # satisfied demand
    demanded = 0.0
    allocated = 0.0
    trickle_count = 0
    for f in range(F):
        if z[f] > 0.5:
            start, end = flow_path_start[f], flow_path_end[f]
            bw = x[start:end].sum()
            demanded += d_f[f]
            allocated += bw
            if bw < 0.1 * d_f[f]:
                trickle_count += 1

    satisfied_demand_pct = 100.0 * allocated / demanded if demanded > 0 else 0.0
    if np.isnan(satisfied_demand_pct) or np.isinf(satisfied_demand_pct):
        satisfied_demand_pct = 0.0

    # ── per-class metrics ──
    per_class = {}
    for c in range(3):
        cname = CLASS_NAMES[c]
        cp = CLASS_PARAMS[c]
        mask = class_id == c
        n_c = int(mask.sum())

        if n_c == 0:
            per_class[cname] = {
                "n_total": 0,
                "n_admitted": 0,
                "admission_rate": 0.0,
                "O_c": 0.0,
                "mean_delay_ms": 0.0,
                "max_delay_ms": 0.0,
                "p95_delay_ms": 0.0,
                "violation_rate": 0.0,
            }
            continue

        admitted_c = mask & admitted_mask
        n_admitted_c = int(admitted_c.sum())

        # O_c: mean normalized delay (admitted get real delay, dropped get M_drop)
        if n_admitted_c > 0:
            mean_delay = float(delays[admitted_c].mean())
            max_delay = float(delays[admitted_c].max())
            p95_delay = float(np.percentile(delays[admitted_c], 95))
            violated = int((delays[admitted_c] > cp.tau).sum())
            violation_rate = violated / n_admitted_c
            O_c = mean_delay / cp.tau
        else:
            mean_delay = 0.0
            max_delay = 0.0
            p95_delay = 0.0
            violation_rate = 1.0
            O_c = 10.0 / cp.tau  # M_drop / tau

        per_class[cname] = {
            "n_total": n_c,
            "n_admitted": n_admitted_c,
            "admission_rate": n_admitted_c / n_c,
            "O_c": O_c,
            "mean_delay_ms": mean_delay * 1000,
            "max_delay_ms": max_delay * 1000,
            "p95_delay_ms": p95_delay * 1000,
            "violation_rate": violation_rate,
        }

    return {
        # metadata
        "solver_name": solver_name,
        "solve_time": solve_time,
        "n_flows": F,
        "n_paths": P,
        # global
        "n_admitted": n_admitted,
        "admission_rate": n_admitted / F if F > 0 else 0.0,
        "total_throughput_mbps": total_throughput,
        "max_link_load": max_link_load,
        "satisfied_demand_pct": satisfied_demand_pct,
        "trickle_flows": trickle_count,
        "total_slack": float(v.sum()),
        # per-class
        "voice": per_class["voice"],
        "video": per_class["video"],
        "file": per_class["file"],
    }


def evaluate_npz(fpath: str) -> dict:
    """
    Evaluate all three solvers (QoS, throughput, ECMP) from a saved .npz file.
    Returns dict with keys "qos", "throughput", "ecmp", each containing metrics.
    Also includes instance metadata.
    """
    d = np.load(fpath)

    meta = d["meta"]
    time_t, load, seed = meta[0], meta[1], meta[2]
    n_flows, n_paths = int(meta[3]), int(meta[4])

    # shared arrays
    common = dict(
        d_f=d["d_f"],
        L_f=d["L_f"],
        class_id=d["class_id"].astype(int),
        is_offload=d["is_offload"].astype(bool),
        prop_delays=d["prop_delays"],
        flow_path_start=d["flow_path_start"],
        flow_path_end=d["flow_path_end"],
        Phi_row=d["Phi_row"],
        Phi_col=d["Phi_col"],
        Phi_data=d["Phi_data"],
        Phi_shape=d["Phi_shape"],
    )

    results = {
        "instance_id": os.path.basename(fpath).replace(".npz", ""),
        "time_t": float(time_t),
        "load": float(load),
        "seed": int(seed),
        "n_flows": n_flows,
        "n_paths": n_paths,
    }

    # QoS
    results["qos"] = evaluate_arrays(
        x=d["x"],
        z=d["z"],
        y=d["y"],
        delays=d["delays"],
        solver_name="qos_gurobi",
        solve_time=float(meta[7]) if len(meta) > 7 else 0.0,
        v=d["v"] if "v" in d else None,
        **common,
    )

    # Throughput
    if "x_tp" in d:
        results["throughput"] = evaluate_arrays(
            x=d["x_tp"],
            z=d["z_tp"],
            y=d["y_tp"],
            delays=d["delays_tp"],
            solver_name="throughput",
            solve_time=float(meta[8]) if len(meta) > 8 else 0.0,
            **common,
        )

    # ECMP
    if "x_ecmp" in d:
        results["ecmp"] = evaluate_arrays(
            x=d["x_ecmp"],
            z=d["z_ecmp"],
            y=d["y_ecmp"],
            delays=d["delays_ecmp"],
            solver_name="ecmp",
            **common,
        )

    return results


def evaluate_dataset(label_dir: str, output_dir: str = None) -> list[dict]:
    """
    Evaluate all .npz files in a label directory.
    Optionally saves per-instance JSONs and a combined results.json.
    Returns list of result dicts.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npz")])
    all_results = []

    for i, fname in enumerate(npz_files):
        fpath = os.path.join(label_dir, fname)
        try:
            result = evaluate_npz(fpath)
            all_results.append(result)

            if output_dir:
                out_path = os.path.join(output_dir, fname.replace(".npz", ".json"))
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, cls=NumpyEncoder)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(npz_files)}] {fname}")

        except Exception as e:
            print(f"  FAILED {fname}: {e}")

    # save combined
    if output_dir:
        combined_path = os.path.join(output_dir, "results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print(f"  Saved {len(all_results)} results to {combined_path}")

    return all_results


def print_comparison(results: dict):
    """
    Pretty-print comparison of solvers from one evaluate_npz result.
    """
    solvers = []
    for key in ["ecmp", "throughput", "qos"]:
        if key in results:
            solvers.append((key, results[key]))

    if not solvers:
        print("No solver results found.")
        return

    names = [s[0] for s in solvers]
    metrics = [s[1] for s in solvers]

    w = 14  # column width

    print(f"\n{'=' * (22 + w * len(names))}")
    print(
        f"  Instance: {results.get('instance_id', '?')}  |  "
        f"Load: {results.get('load', '?')}  |  "
        f"Flows: {results.get('n_flows', '?')}"
    )
    print(f"{'=' * (22 + w * len(names))}")
    header = f"  {'':20}" + "".join(f"{n:>{w}}" for n in names)
    print(header)
    print(f"  {'-' * (20 + w * len(names))}")

    # global metrics
    rows = [
        ("throughput (Mbps)", "total_throughput_mbps", ".1f"),
        ("admitted", "n_admitted", "d"),
        ("max link load", "max_link_load", ".1f"),
        ("solve time (s)", "solve_time", ".4f"),
        ("satisfied demand %", "satisfied_demand_pct", ".1f"),
        ("trickle flows", "trickle_flows", "d"),
        ("total slack (s)", "total_slack", ".4f"),
    ]
    for label, key, fmt in rows:
        vals = "".join(f"{m[key]:>{w}{fmt}}" for m in metrics)
        print(f"  {label:<20}{vals}")

    print(f"  {'-' * (20 + w * len(names))}")

    # per-class metrics
    for cname in ["voice", "video", "file"]:
        print(f"  --- {cname} ---")
        class_rows = [
            ("  O_c", "O_c", ".4f"),
            ("  mean (ms)", "mean_delay_ms", ".2f"),
            ("  max (ms)", "max_delay_ms", ".1f"),
            ("  p95 (ms)", "p95_delay_ms", ".1f"),
            ("  violate %", "violation_rate", ".1%"),
            ("  admitted", "n_admitted", "d"),
        ]
        for label, key, fmt in class_rows:
            if fmt == ".1%":
                vals = "".join(f"{m[cname][key]:>{w}{fmt}}" for m in metrics)
            else:
                vals = "".join(f"{m[cname][key]:>{w}{fmt}}" for m in metrics)
            print(f"  {label:<20}{vals}")

    print(f"{'=' * (22 + w * len(names))}")


def print_dataset_summary(all_results: list[dict]):
    """
    Print aggregate statistics across all instances in a dataset.
    Groups by load level and shows mean metrics per solver.
    """
    if not all_results:
        print("No results to summarize.")
        return

    # group by load
    loads = sorted(set(r["load"] for r in all_results))
    solvers = ["ecmp", "throughput", "qos"]
    available_solvers = [s for s in solvers if s in all_results[0]]

    w = 12

    print(f"\n{'=' * 70}")
    print(f" Dataset Summary ({len(all_results)} instances)")
    print(f"{'=' * 70}")

    for cname in ["voice", "video", "file"]:
        print(f"\n  {cname.upper()} O_c by load:")
        header = f"  {'load':<8}" + "".join(f"{s:>{w}}" for s in available_solvers)
        print(header)
        print(f"  {'-' * (8 + w * len(available_solvers))}")

        for load in loads:
            load_results = [r for r in all_results if abs(r["load"] - load) < 0.01]
            vals = ""
            for solver in available_solvers:
                ocs = [r[solver][cname]["O_c"] for r in load_results if solver in r]
                mean_oc = np.mean(ocs) if ocs else -1
                vals += f"{mean_oc:>{w}.4f}"
            print(f"  {load:<8.2f}{vals}")

    # admission summary
    print("\n  Admission rate by load:")
    header = f"  {'load':<8}" + "".join(f"{s:>{w}}" for s in available_solvers)
    print(header)
    print(f"  {'-' * (8 + w * len(available_solvers))}")

    for load in loads:
        load_results = [r for r in all_results if abs(r["load"] - load) < 0.01]
        vals = ""
        for solver in available_solvers:
            rates = [r[solver]["admission_rate"] for r in load_results if solver in r]
            mean_rate = np.mean(rates) if rates else -1
            vals += f"{mean_rate:>{w}.1%}"
        print(f"  {load:<8.2f}{vals}")

    # solve time summary
    print("\n  Solve time (s) by load:")
    header = f"  {'load':<8}" + "".join(f"{s:>{w}}" for s in available_solvers)
    print(header)
    print(f"  {'-' * (8 + w * len(available_solvers))}")

    for load in loads:
        load_results = [r for r in all_results if abs(r["load"] - load) < 0.01]
        vals = ""
        for solver in available_solvers:
            times = [r[solver]["solve_time"] for r in load_results if solver in r]
            mean_time = np.mean(times) if times else -1
            vals += f"{mean_time:>{w}.2f}"
        print(f"  {load:<8.2f}{vals}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate METEOR dataset")
    parser.add_argument("label_dir", help="Directory with .npz label files")
    parser.add_argument("--output_dir", default=None, help="Save per-instance JSONs here")
    parser.add_argument("--single", default=None, help="Evaluate and print one .npz file")
    args = parser.parse_args()

    if args.single:
        result = evaluate_npz(args.single)
        print_comparison(result)
    else:
        results = evaluate_dataset(args.label_dir, args.output_dir)
        print_dataset_summary(results)
