"""
File created at: 2026-04-05 11:45:05
Author: Sam Ghalayini
meteor/data/demo_traffic.py
"""

# %%
import numpy as np

from data.constellation import Topology
from data.traffic import generate_flows, summarize_flows


def main():
    # Build topology at t=0
    print("Building topology...")
    topo = Topology(time_t=0)
    snap = topo.get_snapshot()
    print(f"  Satellites: {snap['n_nodes']}")
    print(f"  ISL edges:  {snap['n_edges']}")
    print(f"  Gateways:   {len(snap['gateways'])}")
    print(f"  Servers:    {len(snap['server_sat_ids'])}")
    print()

    # Generate flows
    print("Generating 10,000 flows...")
    rng = np.random.default_rng(42)
    flows = generate_flows(topo, n_flows=10000, rng=rng)

    # Summarize
    s = summarize_flows(flows)
    print(f"  Total flows:    {s['n_flows']}")
    print(f"  Voice:          {s['n_voice']} ({s['ct_voice']:.1f}%)")
    print(f"  Video:          {s['n_video']} ({s['ct_video']:.1f}%)")
    print(f"  File:           {s['n_file']}  ({s['ct_file']:.1f}%)")
    print(f"  Offloading:     {s['n_offload']}")
    print(f"  Total demand:   {s['total_demand_mbps']:.1f} Mbps")
    print(f"    Voice demand: {s['voice_demand_mbps']:.1f} Mbps")
    print(f"    Video demand: {s['video_demand_mbps']:.1f} Mbps")
    print(f"    File demand:  {s['file_demand_mbps']:.1f} Mbps")
    print(f"  Unique sources: {s['unique_sources']}")
    print(f"  Unique dests:   {s['unique_destinations']}")
    if s["n_offload"] > 0:
        print(f"  Workload mean:  {s['workload_mean']:.2e} cycles")
    print()

    # Show first 10 flows
    print("First 10 flows:")
    print(
        f"  {'ID':>4} {'SRC':>4} {'DST':>4} {'CLS':>5} {'d_f':>8} {'L_f':>10} {'W_f':>10} {'OFF':>5}"
    )
    print(
        f"  {'----':>4} {'----':>4} {'----':>4} {'-----':>5} {'--------':>8} {'----------':>10} {'----------':>10} {'-----':>5}"
    )
    cls_names = ["voice", "video", "file "]
    for i in range(min(20, flows.n_flows)):
        print(
            f"  {flows.flow_id[i]:4d} "
            f"{flows.src_sat[i]:4d} "
            f"{flows.dst_sat[i]:4d} "
            f"{cls_names[flows.class_id[i]]:>5} "
            f"{flows.d_f[i]:8.3f} "
            f"{flows.L_f[i]:10.0f} "
            f"{flows.W_f[i]:10.1f} "
            f"{'  yes' if flows.is_offload[i] else '   no':>5}"
        )


if __name__ == "__main__":
    main()
# %%
