"""
File created at: 2026-04-08 13:03:33
Author: Sam Ghalayini
meteor/data/plot_data.py
Run with:
    python -m data.plot_data
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from data.constellation import Topology
from data.paths import build_path_data, get_link_loads
from data.traffic import CLASS_NAMES, generate_flows, summarize_flows

# ── output ──
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "data")
os.makedirs(FIGDIR, exist_ok=True)

# ── palette ──
C_VOICE = "#2196F3"
C_VIDEO = "#FF9800"
C_FILE = "#4CAF50"
C_SERVER = "#E53935"
C_GATEWAY = "#7B1FA2"
C_SAT = "#78909C"
C_ISL = "#B0BEC5"
CLASS_COLORS = [C_VOICE, C_VIDEO, C_FILE]

# ── IEEE style ──
plt.rcParams.update(
    {
        "font.family": "serif",
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
)

FIG_SINGLE = (3.5, 2.5)
FIG_WIDE = (7.16, 3.5)
FIG_MAP = (7.16, 4.0)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# ────────────────────────────────────────
# Figure 1a: Constellation — Flat Map
# ────────────────────────────────────────


def plot_constellation_flat(topo):
    fig = plt.figure(figsize=FIG_MAP)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#f5f8fa")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    graph = topo.get_graph()
    N = topo.N

    # ISLs
    for u, v in graph.edges:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        if abs(lon_u - lon_v) < 90:
            ax.plot(
                [lon_u, lon_v],
                [lat_u, lat_v],
                color=C_ISL,
                linewidth=0.2,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

    # satellites
    lats = [graph.nodes[i]["lat"] for i in range(N)]
    lons = [graph.nodes[i]["lon"] for i in range(N)]
    ax.scatter(
        lons,
        lats,
        s=3,
        c=C_SAT,
        alpha=0.5,
        zorder=2,
        transform=ccrs.PlateCarree(),
        label=f"Satellites ({N})",
    )

    # servers
    server_ids = topo.get_server_sat_ids()
    srv_lats = [graph.nodes[s]["lat"] for s in server_ids]
    srv_lons = [graph.nodes[s]["lon"] for s in server_ids]
    ax.scatter(
        srv_lons,
        srv_lats,
        s=50,
        c=C_SERVER,
        marker="^",
        edgecolors="black",
        linewidth=0.5,
        zorder=4,
        transform=ccrs.PlateCarree(),
        label=f"Edge Servers ({len(server_ids)})",
    )

    # gateways with labels
    for gw in topo.gateways:
        color = C_SERVER if gw.has_server else C_GATEWAY
        marker = "^" if gw.has_server else "s"
        if not gw.has_server:
            ax.scatter(
                gw.lon,
                gw.lat,
                s=30,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.4,
                zorder=3,
                transform=ccrs.PlateCarree(),
            )
        name = gw.id.replace("gw_", "").replace("_", " ").title()
        ax.text(
            gw.lon + 2,
            gw.lat + 2,
            name,
            fontsize=5,
            transform=ccrs.PlateCarree(),
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
        )

    # latitude cutoff
    cutoff = topo.config.lat_cutoff
    ax.plot(
        [-180, 180],
        [cutoff, cutoff],
        color="red",
        ls="--",
        lw=0.6,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [-180, 180],
        [-cutoff, -cutoff],
        color="red",
        ls="--",
        lw=0.6,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title("LEO Constellation: 22 Planes × 18 Satellites (550 km)", fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9, markerscale=1.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    save(fig, "01_constellation_flat.png")


# ────────────────────────────────────────
# Figure 1b: Constellation — Globe
# ────────────────────────────────────────


def plot_constellation_globe(topo):
    fig = plt.figure(figsize=(7.16, 6))

    # three views: Americas, Europe/Africa, Asia/Pacific
    views = [
        ("Americas", ccrs.Orthographic(-80, 30)),
        ("Europe & Africa", ccrs.Orthographic(15, 40)),
        ("Asia-Pacific", ccrs.Orthographic(120, 20)),
    ]

    graph = topo.get_graph()
    N = topo.N
    server_ids = topo.get_server_sat_ids()

    for i, (title, proj) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#e0e0e0", edgecolor="none")
        ax.add_feature(cfeature.OCEAN, facecolor="#dce9f2")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888")

        # ISLs
        for u, v in graph.edges:
            lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
            lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
            if abs(lon_u - lon_v) < 90:
                ax.plot(
                    [lon_u, lon_v],
                    [lat_u, lat_v],
                    color=C_ISL,
                    linewidth=0.15,
                    alpha=0.25,
                    transform=ccrs.PlateCarree(),
                    zorder=1,
                )

        # satellites
        lats = [graph.nodes[n]["lat"] for n in range(N)]
        lons = [graph.nodes[n]["lon"] for n in range(N)]
        ax.scatter(lons, lats, s=1.5, c=C_SAT, alpha=0.4, zorder=2, transform=ccrs.PlateCarree())

        # servers
        srv_lats = [graph.nodes[s]["lat"] for s in server_ids]
        srv_lons = [graph.nodes[s]["lon"] for s in server_ids]
        ax.scatter(
            srv_lons,
            srv_lats,
            s=25,
            c=C_SERVER,
            marker="^",
            edgecolors="black",
            linewidth=0.3,
            zorder=4,
            transform=ccrs.PlateCarree(),
        )

        # gateway labels
        for gw in topo.gateways:
            name = gw.id.replace("gw_", "").replace("_", " ").title()
            try:
                ax.text(
                    gw.lon,
                    gw.lat + 3,
                    name,
                    fontsize=4,
                    ha="center",
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.6
                    ),
                )
            except Exception:
                pass

        ax.set_title(title, fontsize=8, pad=4)

    fig.suptitle("METEOR Constellation — Global Coverage", fontsize=10, y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "01_constellation_globe.png")


# ────────────────────────────────────────
# Figure 2: Degree Distribution
# ────────────────────────────────────────


def plot_degree_distribution(topo):
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    graph = topo.get_graph()

    degrees = [graph.degree(n) for n in graph.nodes]
    n_full = sum(1 for d in degrees if d >= 8)
    n_polar = sum(1 for d in degrees if d < 8)

    ax.hist(
        degrees,
        bins=range(0, max(degrees) + 2),
        align="left",
        color=C_SAT,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Satellites")
    ax.set_title("ISL Connectivity")
    ax.set_xticks(range(0, max(degrees) + 1))

    ax.text(
        0.97,
        0.95,
        f"Full (degree ≥ 8): {n_full}\nPolar (< 8): {n_polar}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="#cccccc", alpha=0.9
        ),
    )

    fig.tight_layout()
    save(fig, "02_degree_distribution.png")


# ────────────────────────────────────────
# Figure 3: Traffic Demand
# ────────────────────────────────────────


def plot_traffic_demand(topo, flows):
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    N = topo.N

    demand_by_sat = np.zeros((3, N))
    for c in range(3):
        mask = flows.class_id == c
        for sat_id in range(N):
            sat_mask = mask & (flows.src_sat == sat_id)
            demand_by_sat[c, sat_id] = flows.d_f[sat_mask].sum()

    total = demand_by_sat.sum(axis=0)
    order = np.argsort(-total)
    top_n = 50

    x = np.arange(top_n)
    bottom = np.zeros(top_n)
    for c in range(3):
        vals = demand_by_sat[c, order[:top_n]]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=CLASS_COLORS[c],
            label=CLASS_NAMES[c].capitalize(),
            width=0.8,
            alpha=0.85,
        )
        bottom += vals

    ax.set_xlabel("Satellite (ranked by demand)")
    ax.set_ylabel("Demand (Mbps)")
    ax.set_title("Source Demand Distribution — Top 50 Satellites")
    ax.legend(fontsize=7)
    ax.set_xticks([])

    fig.tight_layout()
    save(fig, "03_traffic_demand.png")


# ────────────────────────────────────────
# Figure 4: Class Mix
# ────────────────────────────────────────


def plot_class_mix(flows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE, gridspec_kw={"width_ratios": [1, 1.3]})

    counts = [(flows.class_id == c).sum() for c in range(3)]
    labels = [f"{CLASS_NAMES[c].capitalize()}\n({counts[c]})" for c in range(3)]

    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        colors=CLASS_COLORS,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 7},
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_fontweight("bold")
    ax1.set_title("Class Distribution", fontsize=9)

    s = summarize_flows(flows)
    table_data = [
        ["Total flows", f"{s['n_flows']:,}"],
        ["Offloading", f"{s['n_offload']:,}"],
        ["Total demand", f"{s['total_demand_mbps']:,.0f} Mbps"],
        ["Voice demand", f"{s['voice_demand_mbps']:,.1f} Mbps"],
        ["Video demand", f"{s['video_demand_mbps']:,.0f} Mbps"],
        ["File demand", f"{s['file_demand_mbps']:,.0f} Mbps"],
        ["Unique sources", f"{s['unique_sources']}"],
        ["Unique destinations", f"{s['unique_destinations']}"],
    ]

    ax2.axis("off")
    table = ax2.table(
        cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    for j in range(2):
        table[0, j].set_facecolor("#e0e0e0")
        table[0, j].set_text_props(fontweight="bold")
    ax2.set_title("Flow Summary", fontsize=9, pad=15)

    fig.tight_layout()
    save(fig, "04_class_mix.png")


# ────────────────────────────────────────
# Figure 5: Path Delays
# ────────────────────────────────────────


def plot_path_delays(path_data):
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    comm_delays, off_delays = [], []
    for p in path_data.paths:
        d = p.prop_delay * 1000
        if p.server_id is not None:
            off_delays.append(d)
        else:
            comm_delays.append(d)

    all_delays = comm_delays + off_delays
    bins = np.linspace(0, max(all_delays) * 1.05, 40)

    if comm_delays:
        ax.hist(
            comm_delays,
            bins=bins,
            alpha=0.7,
            color=C_VOICE,
            label=f"Comm ({len(comm_delays)})",
            edgecolor="white",
            linewidth=0.3,
        )
    if off_delays:
        ax.hist(
            off_delays,
            bins=bins,
            alpha=0.7,
            color=C_FILE,
            label=f"Offload ({len(off_delays)})",
            edgecolor="white",
            linewidth=0.3,
        )

    med = np.median(all_delays)
    ax.axvline(med, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.text(med + 2, ax.get_ylim()[1] * 0.9, f"median: {med:.0f} ms", fontsize=6, color="red")

    ax.set_xlabel("Propagation Delay (ms)")
    ax.set_ylabel("Paths")
    ax.set_title("Path Delay Distribution")
    ax.legend(fontsize=6)

    fig.tight_layout()
    save(fig, "05_path_delays.png")


# ────────────────────────────────────────
# Figure 6: Bottleneck Heatmap
# ────────────────────────────────────────


def plot_bottleneck_map(topo, path_data):
    fig = plt.figure(figsize=FIG_MAP)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#f8fafb")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#aaaaaa")

    graph = topo.get_graph()
    N = topo.N

    # compute loads
    x_uniform = np.ones(path_data.n_paths)
    loads_array = np.asarray(get_link_loads(path_data, x_uniform)).flatten()
    edge_to_idx = path_data.edge_to_idx
    max_load = loads_array.max() if loads_array.max() > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_load)
    cmap = plt.cm.YlOrRd

    # draw edges
    for u, v in graph.edges:
        lat_u, lon_u = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat_v, lon_v = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        if abs(lon_u - lon_v) > 90:
            continue

        idx = edge_to_idx[(u, v)]
        load = loads_array[idx]
        if load < 1:
            continue  # skip unloaded links for clarity
        color = cmap(norm(load))
        width = 0.3 + 2.5 * (load / max_load)

        ax.plot(
            [lon_u, lon_v],
            [lat_u, lat_v],
            color=color,
            linewidth=width,
            alpha=0.8,
            zorder=2,
            transform=ccrs.PlateCarree(),
        )

    # satellites
    lats = [graph.nodes[i]["lat"] for i in range(N)]
    lons = [graph.nodes[i]["lon"] for i in range(N)]
    ax.scatter(lons, lats, s=2, c="grey", alpha=0.3, zorder=1, transform=ccrs.PlateCarree())

    # gateways
    for gw in topo.gateways:
        ax.scatter(
            gw.lon,
            gw.lat,
            s=25,
            c="black",
            marker="s",
            edgecolors="white",
            linewidth=0.3,
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
        name = gw.id.replace("gw_", "").replace("_", " ").title()
        ax.text(
            gw.lon + 2,
            gw.lat - 4,
            name,
            fontsize=4,
            zorder=5,
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=20)
    cbar.set_label("Path Count", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("ISL Bottleneck Map (uniform allocation)", fontsize=9)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False

    save(fig, "06_bottleneck_map.png")


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────


def main():
    print("=" * 50)
    print("METEOR — Generating Paper Figures")
    print("=" * 50)

    print("\n[1] Building topology...")
    topo = Topology(time_t=0)
    graph = topo.get_graph()
    server_sat_ids = topo.get_server_sat_ids()
    print(f"    {topo.N} sats, {graph.number_of_edges()} edges, " f"{len(server_sat_ids)} servers")

    print("\n[2] Generating flows...")
    rng = np.random.default_rng(42)
    flows = generate_flows(topo, n_flows=10000, rng=rng)
    s = summarize_flows(flows)
    print(f"    {s['n_flows']} flows, {s['total_demand_mbps']:,.0f} Mbps")

    print("\n[3] Computing paths...")
    path_data = build_path_data(graph, flows, server_sat_ids, k=10, k_per_server=3, verbose=True)
    print(f"    {path_data.n_paths} paths")

    print("\n[4] Generating figures...")

    print("  Fig 1a: Flat map")
    plot_constellation_flat(topo)

    print("  Fig 1b: Globe views")
    plot_constellation_globe(topo)

    print("  Fig 2: Degree distribution")
    plot_degree_distribution(topo)

    print("  Fig 3: Traffic demand")
    plot_traffic_demand(topo, flows)

    print("  Fig 4: Class mix")
    plot_class_mix(flows)

    print("  Fig 5: Path delays")
    plot_path_delays(path_data)

    print("  Fig 6: Bottleneck map")
    plot_bottleneck_map(topo, path_data)

    print(f"\n{'=' * 50}")
    print(f"All figures saved to {FIGDIR}/")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
