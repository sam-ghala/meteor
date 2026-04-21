"""
File created at: 2026-04-08 13:03:33
Author: Sam Ghalayini
meteor/data/plot_data.py

Run from project root: python -m data.plot_data
"""

# %%
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data.constellation import Topology
from data.paths import build_path_data, get_link_loads
from data.traffic import (
    CLASS_NAMES,
    generate_flows,
    summarize_flows,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

FIGDIR = os.path.join("figures")
os.makedirs(FIGDIR, exist_ok=True)

# Consistent style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    }
)

# METEOR color palette
C_VOICE = "#2196F3"  # blue
C_VIDEO = "#FF9800"  # orange
C_FILE = "#4CAF50"  # green
C_SERVER = "#E53935"  # red
C_GATEWAY = "#9C27B0"  # purple
C_SAT = "#78909C"  # blue-grey
C_ISL = "#B0BEC5"  # light grey
CLASS_COLORS = [C_VOICE, C_VIDEO, C_FILE]


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Figure 1: Constellation Map
# ──────────────────────────────────────────────


def plot_constellation_map(topo):
    """Satellites on world map with gateways and servers marked."""
    fig, ax = plt.subplots(figsize=(12, 6))
    graph = topo.get_graph()
    N = topo.N

    # Coastline rectangle hint (simplified)
    ax.set_facecolor("#f0f4f8")
    ax.axhspan(-90, 90, color="white", alpha=0.3)

    # Plot ISLs as faint lines
    for u, v in graph.edges:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        # Skip edges that wrap around the map (lon diff > 180)
        if abs(lon_u - lon_v) < 90:
            ax.plot([lon_u, lon_v], [lat_u, lat_v], color=C_ISL, linewidth=0.3, alpha=0.4, zorder=1)

    # Plot all satellites
    lats = [graph.nodes[i]["lat"] for i in range(N)]
    lons = [graph.nodes[i]["lon"] for i in range(N)]
    ax.scatter(lons, lats, s=5, c=C_SAT, alpha=0.6, zorder=2, label="Satellites")

    # Highlight server satellites
    server_ids = topo.get_server_sat_ids()
    srv_lats = [graph.nodes[s]["lat"] for s in server_ids]
    srv_lons = [graph.nodes[s]["lon"] for s in server_ids]
    ax.scatter(
        srv_lons,
        srv_lats,
        s=80,
        c=C_SERVER,
        marker="^",
        edgecolors="black",
        linewidth=0.8,
        zorder=4,
        label="Edge Servers",
    )

    # Plot gateways
    for gw in topo.gateways:
        marker = "^" if gw.has_server else "s"
        color = C_SERVER if gw.has_server else C_GATEWAY
        ax.scatter(
            gw.lon,
            gw.lat,
            s=50,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            gw.id.replace("gw_", ""),
            (gw.lon, gw.lat),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Latitude cutoff lines
    cutoff = topo.config.lat_cutoff
    ax.axhline(cutoff, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-cutoff, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(182, cutoff + 1, f"±{cutoff}° cutoff", fontsize=7, color="red", alpha=0.7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("METEOR: Starlink Phase 1 Constellation (396 satellites)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    save(fig, "01_constellation_map.png")


# ──────────────────────────────────────────────
# Figure 2: Node Degree Distribution
# ──────────────────────────────────────────────


def plot_degree_distribution(topo):
    """Histogram of satellite node degrees showing ISL connectivity."""
    fig, ax = plt.subplots(figsize=(7, 4))
    graph = topo.get_graph()

    # Use out-degree (symmetric, so in-degree is same)
    degrees = [graph.degree(n) for n in graph.nodes]

    counts, bins, patches = ax.hist(
        degrees,
        bins=range(0, max(degrees) + 2),
        align="left",
        color=C_SAT,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    ax.set_xlabel("Node Degree (ISL connections)")
    ax.set_ylabel("Number of Satellites")
    ax.set_title("ISL Connectivity Distribution")
    ax.set_xticks(range(0, max(degrees) + 1))

    # Annotate
    n_full = sum(1 for d in degrees if d >= 8)  # 4 bidirectional = 8 directed
    n_polar = sum(1 for d in degrees if d < 8)
    ax.text(
        0.97,
        0.95,
        f"Full connectivity (degree ≥ 8): {n_full}\n" f"Polar (inter-orbit off): {n_polar}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.grid(axis="y", alpha=0.3)
    save(fig, "02_degree_distribution.png")


# ──────────────────────────────────────────────
# Figure 3: Traffic Demand by Satellite
# ──────────────────────────────────────────────


def plot_traffic_demand(topo, flows):
    """Bar chart of source demand per satellite, colored by class."""
    fig, ax = plt.subplots(figsize=(12, 4))
    N = topo.N

    demand_by_sat = np.zeros((3, N))  # [class, sat]
    for c in range(3):
        mask = flows.class_id == c
        for sat_id in range(N):
            sat_mask = mask & (flows.src_sat == sat_id)
            demand_by_sat[c, sat_id] = flows.d_f[sat_mask].sum()

    # Sort by total demand for cleaner visualization
    total = demand_by_sat.sum(axis=0)
    order = np.argsort(-total)
    top_n = 60  # show top 60 satellites

    x = np.arange(top_n)
    bottom = np.zeros(top_n)
    for c in range(3):
        vals = demand_by_sat[c, order[:top_n]]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=CLASS_COLORS[c],
            label=CLASS_NAMES[c],
            width=0.8,
            alpha=0.85,
        )
        bottom += vals

    ax.set_xlabel("Satellite (ranked by total source demand)")
    ax.set_ylabel("Total Source Demand (Mbps)")
    ax.set_title("Traffic Demand Distribution — Top 60 Source Satellites")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)

    save(fig, "03_traffic_demand.png")


# ──────────────────────────────────────────────
# Figure 4: Class Mix and Flow Table Sample
# ──────────────────────────────────────────────


def plot_class_mix(flows):
    """Pie chart of class distribution + summary stats."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.5]})

    # Pie chart
    counts = [(flows.class_id == c).sum() for c in range(3)]
    explode = (0.03, 0.03, 0.03)
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=CLASS_NAMES,
        colors=CLASS_COLORS,
        autopct="%1.1f%%",
        explode=explode,
        startangle=90,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax1.set_title("Traffic Class Distribution")

    # Summary table
    s = summarize_flows(flows)
    table_data = [
        ["Total flows", f"{s['n_flows']:,}"],
        ["Voice flows", f"{s['n_voice']:,}"],
        ["Video flows", f"{s['n_video']:,}"],
        ["File flows", f"{s['n_file']:,}"],
        ["Offloading", f"{s['n_offload']:,}"],
        ["Total demand", f"{s['total_demand_mbps']:,.0f} Mbps"],
        ["Voice demand", f"{s['voice_demand_mbps']:,.1f} Mbps"],
        ["Video demand", f"{s['video_demand_mbps']:,.0f} Mbps"],
        ["File demand", f"{s['file_demand_mbps']:,.0f} Mbps"],
        ["Unique srcs", f"{s['unique_sources']}"],
        ["Unique dsts", f"{s['unique_destinations']}"],
    ]

    ax2.axis("off")
    table = ax2.table(
        cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#e0e0e0")
        table[0, j].set_text_props(fontweight="bold")

    ax2.set_title("Flow Table Summary", pad=20)

    fig.suptitle("METEOR: Traffic Generation (10,000 flows)", fontsize=12, y=1.02)
    save(fig, "04_class_mix.png")


# ──────────────────────────────────────────────
# Figure 5: Path Delay Histogram
# ──────────────────────────────────────────────


def plot_path_delays(path_data, flows):
    """Histogram of propagation delays, comm vs offload."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Split delays by flow type
    comm_delays = []
    off_delays = []
    for p in path_data.paths:
        if p.server_id is not None:
            off_delays.append(p.prop_delay * 1000)  # ms
        else:
            comm_delays.append(p.prop_delay * 1000)

    bins = np.linspace(0, max(comm_delays + off_delays) * 1.05, 50)

    if comm_delays:
        ax.hist(
            comm_delays,
            bins=bins,
            alpha=0.7,
            color=C_VOICE,
            label=f"Communication ({len(comm_delays)} paths)",
            edgecolor="white",
        )
    if off_delays:
        ax.hist(
            off_delays,
            bins=bins,
            alpha=0.7,
            color=C_FILE,
            label=f"Offloading ({len(off_delays)} paths)",
            edgecolor="white",
        )

    ax.set_xlabel("Propagation Delay (ms)")
    ax.set_ylabel("Number of Paths")
    ax.set_title("Path Propagation Delay Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate stats
    all_delays = comm_delays + off_delays
    ax.axvline(np.median(all_delays), color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        np.median(all_delays) + 1,
        ax.get_ylim()[1] * 0.9,
        f"median: {np.median(all_delays):.1f} ms",
        fontsize=8,
        color="red",
    )

    save(fig, "05_path_delays.png")


# ──────────────────────────────────────────────
# Figure 6: Link Load Heatmap (geographic)
# ──────────────────────────────────────────────


def plot_link_loads(topo, path_data):
    """Color ISLs by how many paths cross them. Bottlenecks should be near cities."""
    fig, ax = plt.subplots(figsize=(12, 6))
    graph = topo.get_graph()
    N = topo.N

    ax.set_facecolor("#f0f4f8")

    # Compute loads with uniform allocation
    x_uniform = np.ones(path_data.n_paths)
    loads_array = np.asarray(get_link_loads(path_data, x_uniform)).flatten()

    # Build edge list with loads
    edge_list = list(graph.edges)
    edge_to_idx = path_data.edge_to_idx

    # Normalize for colormap
    max_load = loads_array.max() if loads_array.max() > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_load)
    cmap = plt.cm.YlOrRd

    # Draw edges colored by load
    for u, v in edge_list:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        if abs(lon_u - lon_v) > 90:
            continue

        idx = edge_to_idx[(u, v)]
        load = loads_array[idx]
        color = cmap(norm(load))
        width = 0.3 + 2.0 * (load / max_load)

        ax.plot([lon_u, lon_v], [lat_u, lat_v], color=color, linewidth=width, alpha=0.7, zorder=1)

    # Satellites as small dots
    lats = [graph.nodes[i]["lat"] for i in range(N)]
    lons = [graph.nodes[i]["lon"] for i in range(N)]
    ax.scatter(lons, lats, s=3, c="grey", alpha=0.4, zorder=2)

    # Gateways
    for gw in topo.gateways:
        ax.scatter(
            gw.lon, gw.lat, s=40, c="black", marker="s", edgecolors="white", linewidth=0.5, zorder=3
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Path Count (uniform x=1 allocation)")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("ISL Load Heatmap — Bottleneck Identification")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    save(fig, "06_link_loads.png")


# ──────────────────────────────────────────────
# Figure 7: Pipeline Status Diagram
# ──────────────────────────────────────────────


def plot_pipeline_status():
    """Block diagram showing what's built and what's next."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    blocks = [
        (0.5, "Constellation\nTopology", "done"),
        (2.3, "Traffic\nGeneration", "done"),
        (4.1, "Path\nComputation", "done"),
        (5.9, "CVXPY\nSolver", "next"),
        (7.7, "ML Model\n(DFL / Unroll)", "future"),
    ]

    colors = {"done": "#4CAF50", "next": "#FF9800", "future": "#BDBDBD"}
    # labels = {"done": "✓ Done", "next": "→ Next", "future": "Planned"}

    for x, text, status in blocks:
        rect = plt.Rectangle(
            (x, 0.4),
            1.5,
            1.2,
            facecolor=colors[status],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.75, 1.0, text, ha="center", va="center", fontsize=9, fontweight="bold", zorder=3
        )

    # Arrows between blocks
    for i in range(len(blocks) - 1):
        x_start = blocks[i][0] + 1.5
        x_end = blocks[i + 1][0]
        ax.annotate(
            "",
            xy=(x_end, 1.0),
            xytext=(x_start, 1.0),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=colors["done"],
            markersize=12,
            label="Done",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=colors["next"],
            markersize=12,
            label="Next",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=colors["future"],
            markersize=12,
            label="Planned",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9, frameon=False)

    ax.set_title("METEOR Pipeline Status", fontsize=12, pad=10)

    save(fig, "07_pipeline_status.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    print("=" * 50)
    print("METEOR — Generating Meeting Figures")
    print("=" * 50)

    # Build data
    print("\n[1] Building topology...")
    topo = Topology(time_t=0)
    graph = topo.get_graph()
    server_sat_ids = topo.get_server_sat_ids()
    print(
        f"    {topo.N} satellites, {graph.number_of_edges()} edges, "
        f"{len(server_sat_ids)} servers"
    )

    print("\n[2] Generating 10,000 flows...")
    rng = np.random.default_rng(42)
    flows = generate_flows(topo, n_flows=10000, rng=rng)
    s = summarize_flows(flows)
    print(
        f"    {s['n_flows']} flows, {s['n_offload']} offloading, "
        f"{s['total_demand_mbps']:,.0f} Mbps total demand"
    )

    print("\n[3] Computing paths (this takes ~2 min)...")
    path_data = build_path_data(graph, flows, server_sat_ids, k=10, k_per_server=3)
    print(f"    {path_data.n_paths} total paths, Phi shape {path_data.Phi.shape}")

    # Generate figures
    print("\n[4] Generating figures...")

    print("  Fig 1: Constellation map")
    plot_constellation_map(topo)

    print("  Fig 2: Degree distribution")
    plot_degree_distribution(topo)

    print("  Fig 3: Traffic demand by satellite")
    plot_traffic_demand(topo, flows)

    print("  Fig 4: Class mix and summary")
    plot_class_mix(flows)

    print("  Fig 5: Path delay histogram")
    plot_path_delays(path_data, flows)

    print("  Fig 6: Link load heatmap")
    plot_link_loads(topo, path_data)

    print("  Fig 7: Pipeline status")
    plot_pipeline_status()

    print(f"\n{'=' * 50}")
    print(f"All figures saved to {FIGDIR}/")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
# %%
