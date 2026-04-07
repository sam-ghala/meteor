"""
File created at: 2026-04-06 19:43:54
Author: Sam Ghalayini
meteor/data/paths.py
precompute k canidate routes through the constellation
outputs:
    - path lists
    - popagation delays per path
    - path indexing
    - mapping from paths to flows and severs
    - pi matrix, "which path uses which edges"
"""

# %% imports
import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from data.traffic import FlowTable


# %% data strucutres
@dataclass
class Path:
    """A single candidate path through the constellation."""

    path_id: int
    flow_id: int
    nodes: list[int]
    edges: list[tuple[int, int]]
    prop_delay: float
    hop_count: int
    server_id: int | None


@dataclass
class PathData:
    """Complete path data bundle consumed by solvers and models."""

    paths: list[Path]
    # Index arrays
    path_to_flow: np.ndarray
    path_to_server: np.ndarray
    prop_delays: np.ndarray
    hop_counts: np.ndarray

    # Per-flow path ranges
    flow_path_start: np.ndarray
    flow_path_end: np.ndarray

    # Topology mapping
    edge_to_idx: dict[tuple[int, int], int]
    n_edges: int

    # path edge matrix
    Phi: sparse.csr_matrix

    @property
    def n_paths(self) -> int:
        return len(self.paths)


def build_edge_index(graph: nx.DiGraph) -> tuple[dict[tuple[int, int], int], int]:
    """
    Assign a unique integer index to each directed edge in the graph.
    """
    edge_to_idx = {e: i for i, e in enumerate(graph.edges)}
    return (edge_to_idx, len(edge_to_idx))


def k_shortest_paths(
    graph: nx.DiGraph,
    src: int,
    dst: int,
    k: int = 10,
) -> list[tuple[list[int], float]]:
    """
    Compute the k shortest simple paths from src to dst by propagation delay.
    """
    results = []
    for path in itertools.islice(nx.shortest_simple_paths(graph, src, dst, weight="delay"), k):
        total_delay = sum(graph[u][v]["delay"] for u, v in zip(path, path[1:], strict=False))
        results.append((path, total_delay))
    return results


def compute_comm_paths(
    graph: nx.DiGraph,
    flows: FlowTable,
    k: int = 10,
) -> dict[int, list[tuple[list[int], float]]]:
    """
    Compute candidate paths for all communication flows (is_offload == False).
    Output:
        dict mapping flow_id: int -> list of (nodes, total_delay) tuples
    """
    comm_indices = np.where(~flows.is_offload)[0]
    mapping_flow_dict = {}
    for i in tqdm(comm_indices, desc="processing comm_paths..."):
        try:
            mapping_flow_dict[flows.flow_id[i]] = k_shortest_paths(
                graph, int(flows.src_sat[i]), int(flows.dst_sat[i]), k
            )
        except nx.NetworkXNoPath:
            mapping_flow_dict[flows.flow_id[i]] = []
    return mapping_flow_dict


def compute_offload_paths(
    graph: nx.DiGraph,
    flows: FlowTable,
    server_sat_ids: list[int],
    k_per_server: int = 3,
    k_total: int = 10,
) -> dict[int, list[tuple[list[int], float, int]]]:
    """
    Compute candidate paths for all offloading flows.
    For each offloading flow, compute k_per_server shortest paths to each server,
    then keep the top k_total overall by delay.
    """
    offload_indices = np.where(flows.is_offload)[0]
    mapping_flow_dict = {}
    for i in tqdm(offload_indices, desc="processing offload_paths..."):
        candidates = []
        for server_idx, server_sat in enumerate(server_sat_ids):
            # if src == server_sat, add direct path 0 hops
            if flows.src_sat[i] == server_sat:
                candidates.append(([flows.src_sat[i]], 0.0, server_idx))
                continue
            try:
                paths = k_shortest_paths(graph, int(flows.src_sat[i]), server_sat, k_per_server)
                for nodes, delay in paths:
                    candidates.append((nodes, delay, server_idx))
            except nx.NetworkXNoPath:
                continue
        candidates.sort(key=lambda x: x[1])
        mapping_flow_dict[flows.flow_id[i]] = candidates[:k_total]

    return mapping_flow_dict


def build_path_data(
    graph: nx.DiGraph,
    flows: FlowTable,
    server_sat_ids: list[int],
    k: int = 10,
    k_per_server: int = 3,
) -> PathData:
    """
    Main entry point. Computes all paths and assembles the PathData bundle.
    Output:
        PathData containing:
            paths: List[Path] of length P (total paths across all flows)
            path_to_flow: np.ndarray shape (P,)
            path_to_server: np.ndarray shape (P,), -1 for comm flow paths
            prop_delays: np.ndarray shape (P,)
            hop_counts: np.ndarray shape (P,)
            flow_path_start: np.ndarray shape (F,)
            flow_path_end: np.ndarray shape (F,)
            edge_to_idx: dict of size E
            n_edges: int (E)
            Phi: scipy.sparse.csr_matrix shape (P, E)
    """
    # build edge index
    edge_to_idx, n_edges = build_edge_index(graph)
    # compute paths for comm flows
    comm_paths = compute_comm_paths(graph, flows, k)

    # compute paths for offloading flows
    offload_paths = compute_offload_paths(graph, flows, server_sat_ids, k_per_server, k)

    global_idx = 0
    all_paths = []
    path_to_flow = []
    path_to_server = []
    prop_delays = []
    hop_counts = []
    flow_path_start = np.zeros((flows.n_flows), dtype=int)
    flow_path_end = np.zeros((flows.n_flows), dtype=int)

    for f in range(flows.n_flows):
        flow_id = int(flows.flow_id[f])
        flow_path_start[f] = global_idx
        # if offload
        if flows.is_offload[f]:
            f_paths = offload_paths.get(flow_id, [])
            for nodes, delay, server_idx in f_paths:
                path_obj = Path(
                    path_id=global_idx,
                    flow_id=flow_id,
                    nodes=nodes,
                    edges=list(zip(nodes, nodes[1:], strict=False)),
                    prop_delay=delay,
                    hop_count=len(nodes) - 1,
                    server_id=server_idx,
                )
                all_paths.append(path_obj)
                path_to_flow.append(flow_id)
                path_to_server.append(server_idx)
                prop_delays.append(delay)
                hop_counts.append(len(nodes) - 1)
                global_idx += 1
        # else
        else:
            f_paths = comm_paths.get(flow_id, [])
            for nodes, delay in f_paths:
                path_obj = Path(
                    path_id=global_idx,
                    flow_id=flow_id,
                    nodes=nodes,
                    edges=list(zip(nodes, nodes[1:], strict=False)),
                    prop_delay=delay,
                    hop_count=len(nodes) - 1,
                    server_id=None,
                )
                all_paths.append(path_obj)
                path_to_flow.append(flow_id)
                path_to_server.append(-1)
                prop_delays.append(delay)
                hop_counts.append(len(nodes) - 1)
                global_idx += 1
        flow_path_end[f] = global_idx

    Phi = sparse.lil_matrix((global_idx, n_edges), dtype=np.float64)
    for p, path_obj in enumerate(all_paths):
        for u, v in zip(path_obj.nodes, path_obj.nodes[1:], strict=False):
            col = edge_to_idx[(u, v)]
            Phi[p, col] = 1.0
    Phi = sparse.csr_matrix(Phi.tocsr())

    return PathData(
        paths=all_paths,
        path_to_flow=np.array(path_to_flow),
        path_to_server=np.array(path_to_server),
        prop_delays=np.array(prop_delays),
        hop_counts=np.array(hop_counts),
        flow_path_start=flow_path_start,
        flow_path_end=flow_path_end,
        edge_to_idx=edge_to_idx,
        n_edges=n_edges,
        Phi=Phi,
    )


# utilites


def get_flow_paths(path_data: PathData, flow_id: int) -> list[Path]:
    """
    Get all paths belonging to a specific flow.
    Output:
        List[Path] — the paths for that flow (length = flow_path_end[f] - flow_path_start[f])
    """
    start = path_data.flow_path_start[flow_id]
    end = path_data.flow_path_end[flow_id]
    return path_data.paths[start:end]


def get_link_loads(path_data: PathData, x: np.ndarray) -> np.ndarray:
    """
    Compute per-link load given bandwidth allocations.
    Output:
        loads: np.ndarray shape (E,) — total bandwidth on each link (Mbps)
    """
    return path_data.Phi.T @ x
