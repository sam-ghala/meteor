"""
File created at: 2026-04-21 15:21:49
Author: Sam Ghalayini
meteor/evaluate/plot_results.py
Generate publication-ready figures from METEOR dataset labels.
Reads .npz files, evaluates all baselines, produces 6 figures.
Usage:
    python plot_results.py ./labels
    python plot_results.py ./labels --output_dir ./figures/paper
    python plot_results.py ./labels --single_load 0.50
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from data.traffic import CLASS_PARAMS
from evaluate.evaluate import evaluate_npz

# ── styling ──

SOLVER_STYLE = {
    "ecmp": {"color": "#7f8c8d", "marker": "s", "ls": "--", "label": "ECMP"},
    "throughput": {"color": "#e74c3c", "marker": "^", "ls": "-.", "label": "Throughput"},
    "qos": {"color": "#2ecc71", "marker": "o", "ls": "-", "label": "QoS Gurobi"},
}
CLASS_COLORS = {"voice": "#3498db", "video": "#e67e22", "file": "#9b59b6"}
CLASSES = ["voice", "video", "file"]
SOLVERS = ["ecmp", "throughput", "qos"]

FIG_SINGLE = (3.5, 2.5)
FIG_DOUBLE = (7.16, 2.8)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ── data loading ──


def load_all_results(label_dir: str) -> list[dict]:
    npz_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npz")])
    results = []
    for i, fname in enumerate(npz_files):
        try:
            result = evaluate_npz(os.path.join(label_dir, fname))
            results.append(result)
        except Exception as e:
            print(f"  FAILED {fname}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i+1}/{len(npz_files)}")
    print(f"  {len(results)} instances loaded")
    return results


def group_by_load(results):
    groups = defaultdict(list)
    for r in results:
        groups[r["load"]].append(r)
    return dict(sorted(groups.items()))


def get_class_metric(load_results, solver, cname, metric):
    return [r[solver][cname][metric] for r in load_results if solver in r]


def get_global_metric(load_results, solver, metric):
    return [r[solver][metric] for r in load_results if solver in r]


def pick_load(loads, target):
    if target is None:
        return loads[len(loads) // 2]
    return min(loads, key=lambda ld: abs(ld - target))


def save_fig(fig, output_dir, name):
    for ext in [".png"]:  # [".pdf", ".png"]:
        fig.savefig(os.path.join(output_dir, name + ext))
    plt.close(fig)
    print(f"  saved {name}")


# ── Figure 1: O_c vs Load ──


def fig_oc_vs_load(results, output_dir):
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())

    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=False)

    for ax, cname in zip(axes, CLASSES, strict=False):
        for solver in SOLVERS:
            s = SOLVER_STYLE[solver]
            means, stds = [], []
            for load in loads:
                vals = get_class_metric(by_load[load], solver, cname, "O_c")
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            ax.errorbar(
                loads,
                means,
                yerr=stds,
                capsize=2,
                capthick=0.8,
                color=s["color"],
                marker=s["marker"],
                ms=4,
                ls=s["ls"],
                lw=1.2,
                label=s["label"],
            )

        ax.axhline(y=1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(cname.capitalize())
        ax.set_xlabel("Network Load")
        ax.set_ylabel("$O_c$")
        ax.set_xticks(loads)
        ax.set_xticklabels([f"{ld:.0%}" for ld in loads])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_oc_vs_load")


# ── Figure 2: Admission Priority Bars ──


def fig_admission_bars(results, output_dir, target_load=None):
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())
    target = pick_load(loads, target_load)
    load_results = by_load[target]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(SOLVERS))
    width = 0.22

    for i, cname in enumerate(CLASSES):
        means = []
        for solver in SOLVERS:
            vals = get_class_metric(load_results, solver, cname, "admission_rate")
            means.append(np.mean(vals) if vals else 0)

        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            label=cname.capitalize(),
            color=CLASS_COLORS[cname],
            alpha=0.85,
        )
        for bar, val in zip(bars, means, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_ylabel("Admission Rate")
    ax.set_title(f"Per-Class Admission at {target:.0%} Load")
    ax.set_xticks(x)
    ax.set_xticklabels([SOLVER_STYLE[s]["label"] for s in SOLVERS])
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_admission_bars")


# ── Figure 3: Solve Time vs Load ──


def fig_solve_time(results, output_dir):
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for solver in SOLVERS:
        s = SOLVER_STYLE[solver]
        means = []
        for load in loads:
            vals = get_global_metric(by_load[load], solver, "solve_time")
            means.append(np.mean(vals) if vals else 1e-6)

        ax.semilogy(
            loads,
            means,
            color=s["color"],
            marker=s["marker"],
            ms=4,
            ls=s["ls"],
            lw=1.2,
            label=s["label"],
        )

    ax.set_xlabel("Network Load")
    ax.set_ylabel("Solve Time (s)")
    ax.set_title("Solver Runtime")
    ax.set_xticks(loads)
    ax.set_xticklabels([f"{ld:.0%}" for ld in loads])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_solve_time")


# ── Figure 4: Delay CDF ──


def fig_delay_cdf(label_dir, results, output_dir, target_load=None):
    """Needs label_dir to read raw delay arrays from .npz files."""
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())
    target = pick_load(loads, target_load)
    load_results = by_load[target]

    solver_keys = {
        "ecmp": ("delays_ecmp", "z_ecmp"),
        "throughput": ("delays_tp", "z_tp"),
        "qos": ("delays", "z"),
    }

    deadlines = {c: CLASS_PARAMS[i].tau * 1000 for i, c in enumerate(CLASSES)}

    # collect raw delays
    delay_data = {s: {c: [] for c in CLASSES} for s in SOLVERS}

    for r in load_results:
        instance_id = r.get("instance_id", "")
        fpath = os.path.join(label_dir, f"{instance_id}.npz")
        if not os.path.exists(fpath):
            continue

        d = np.load(fpath)
        class_id = d["class_id"].astype(int)

        for solver, (delay_key, z_key) in solver_keys.items():
            if delay_key not in d:
                continue
            delays_arr = d[delay_key]
            z_arr = d[z_key]

            for c_idx, cname in enumerate(CLASSES):
                mask = (class_id == c_idx) & (z_arr > 0.5)
                if mask.sum() > 0:
                    delay_data[solver][cname].extend((delays_arr[mask] * 1000).tolist())

    # plot
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)

    for ax, cname in zip(axes, CLASSES, strict=False):
        for solver in SOLVERS:
            s = SOLVER_STYLE[solver]
            vals = sorted(delay_data[solver][cname])
            if not vals:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf_y, color=s["color"], ls=s["ls"], lw=1.2, label=s["label"])

        dl = deadlines[cname]
        ax.axvline(x=dl, color="red", ls=":", lw=0.8, alpha=0.7)
        ax.text(dl * 1.1, 0.05, f"τ={dl:.0f}ms", fontsize=6, color="red")

        ax.set_title(cname.capitalize())
        ax.set_xlabel("Delay (ms)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("CDF")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle(f"Delay CDF at {target:.0%} Load", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig4_delay_cdf")


# ── Figure 5: Violation Rate vs Load ──


def fig_violation_rate(results, output_dir):
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())

    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)

    for ax, cname in zip(axes, CLASSES, strict=False):
        for solver in SOLVERS:
            s = SOLVER_STYLE[solver]
            means = []
            for load in loads:
                vals = get_class_metric(by_load[load], solver, cname, "violation_rate")
                means.append(np.mean(vals) if vals else 0)

            ax.plot(
                loads,
                means,
                color=s["color"],
                marker=s["marker"],
                ms=4,
                ls=s["ls"],
                lw=1.2,
                label=s["label"],
            )

        ax.set_title(cname.capitalize())
        ax.set_xlabel("Network Load")
        ax.set_xticks(loads)
        ax.set_xticklabels([f"{ld:.0%}" for ld in loads])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Violation Rate")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig5_violation_rate")


# ── Figure 6: Admission Rate vs Load (per solver, per class lines) ──


def fig_admission_vs_load(results, output_dir):
    by_load = group_by_load(results)
    loads = sorted(by_load.keys())

    fig, axes = plt.subplots(1, len(SOLVERS), figsize=FIG_DOUBLE, sharey=True)

    for ax, solver in zip(axes, SOLVERS, strict=False):
        for cname in CLASSES:
            means = []
            for load in loads:
                vals = get_class_metric(by_load[load], solver, cname, "admission_rate")
                means.append(np.mean(vals) if vals else 0)

            ax.plot(
                loads,
                means,
                color=CLASS_COLORS[cname],
                marker="o",
                ms=3,
                lw=1.2,
                label=cname.capitalize(),
            )

        ax.set_title(SOLVER_STYLE[solver]["label"])
        ax.set_xlabel("Network Load")
        ax.set_xticks(loads)
        ax.set_xticklabels([f"{ld:.0%}" for ld in loads])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    axes[0].set_ylabel("Admission Rate")
    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig6_admission_vs_load")


# ── main ──


def main():
    parser = argparse.ArgumentParser(description="Plot METEOR results")
    parser.add_argument("label_dir", help="Directory with .npz label files")
    parser.add_argument("--output_dir", default="./figures/paper")
    parser.add_argument(
        "--single_load",
        type=float,
        default=None,
        help="Target load for CDF and admission bar figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading from {args.label_dir}...")
    results = load_all_results(args.label_dir)
    if not results:
        print("No results found.")
        return

    print(f"\nGenerating figures in {args.output_dir}...")
    fig_oc_vs_load(results, args.output_dir)
    fig_admission_bars(results, args.output_dir, args.single_load)
    fig_solve_time(results, args.output_dir)
    fig_delay_cdf(args.label_dir, results, args.output_dir, args.single_load)
    fig_violation_rate(results, args.output_dir)
    fig_admission_vs_load(results, args.output_dir)
    print(f"\nDone. 6 figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
