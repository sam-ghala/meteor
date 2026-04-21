"""
File created at: 2026-04-06 20:30:00
Author: Sam Ghalayini
meteor/tests/test_paths.py
"""

import numpy as np
import pytest

from data.constellation import Topology, orbital_config
from data.paths import (
    Path,
    build_edge_index,
    build_path_data,
    compute_comm_paths,
    compute_offload_paths,
    get_flow_paths,
    get_link_loads,
    k_shortest_paths,
)
from data.traffic import generate_flows


# helpers
def make_topology(planes=22, sats=18, lat_cutoff=75.0, time_t=0):
    cfg = orbital_config(planes=planes, sat_per_plane=sats, lat_cutoff=lat_cutoff)
    return Topology(time_t=time_t, config=cfg)


@pytest.fixture
def topo():
    return make_topology(planes=22, sats=18)


@pytest.fixture
def graph(topo):
    return topo.get_graph()


@pytest.fixture
def server_sat_ids(topo):
    return topo.get_server_sat_ids()


@pytest.fixture
def flows(topo):
    rng = np.random.default_rng(42)
    return generate_flows(topo, n_flows=100, rng=rng)


@pytest.fixture
def path_data(graph, flows, server_sat_ids):
    return build_path_data(graph, flows, server_sat_ids, k=10, k_per_server=3)


# edge indexing
class TestEdgeIndex:
    def test_count_matches_graph(self, graph):
        edge_to_idx, n_edges = build_edge_index(graph)
        assert n_edges == graph.number_of_edges()

    def test_indices_sequential(self, graph):
        edge_to_idx, n_edges = build_edge_index(graph)
        assert set(edge_to_idx.values()) == set(range(n_edges))

    def test_all_edges_present(self, graph):
        edge_to_idx, _ = build_edge_index(graph)
        for u, v in graph.edges:
            assert (u, v) in edge_to_idx


# k shortest paths on a single pair
class TestKShortestPaths:
    def test_returns_up_to_k(self, graph):
        paths = k_shortest_paths(graph, 0, 200, k=5)
        assert 1 <= len(paths) <= 5

    def test_paths_sorted_by_delay(self, graph):
        paths = k_shortest_paths(graph, 0, 200, k=10)
        delays = [d for _, d in paths]
        assert delays == sorted(delays)

    def test_path_starts_at_src(self, graph):
        paths = k_shortest_paths(graph, 0, 200, k=3)
        for nodes, _ in paths:
            assert nodes[0] == 0

    def test_path_ends_at_dst(self, graph):
        paths = k_shortest_paths(graph, 0, 200, k=3)
        for nodes, _ in paths:
            assert nodes[-1] == 200

    def test_delay_positive(self, graph):
        paths = k_shortest_paths(graph, 0, 200, k=5)
        for _, delay in paths:
            assert delay > 0

    def test_delay_matches_edge_sum(self, graph):
        """Total delay should equal sum of edge delays along the path."""
        paths = k_shortest_paths(graph, 0, 200, k=3)
        for nodes, delay in paths:
            expected = sum(graph[u][v]["delay"] for u, v in zip(nodes, nodes[1:], strict=False))
            assert abs(delay - expected) < 1e-12

    def test_consecutive_nodes_are_edges(self, graph):
        """Every consecutive pair in the path should be an edge in the graph."""
        paths = k_shortest_paths(graph, 0, 200, k=5)
        for nodes, _ in paths:
            for u, v in zip(nodes, nodes[1:], strict=False):
                assert graph.has_edge(u, v), f"Edge {u}->{v} not in graph"

    def test_delay_reasonable_ms(self, graph):
        """Cross-constellation paths should be under 150ms."""
        paths = k_shortest_paths(graph, 0, 200, k=1)
        for _, delay in paths:
            assert delay * 1000 < 150.0


# communication flow paths
class TestCommPaths:
    def test_returns_dict(self, graph, flows):
        comm = compute_comm_paths(graph, flows, k=5)
        assert isinstance(comm, dict)

    def test_only_comm_flows(self, graph, flows):
        """Should not contain offloading flow IDs."""
        comm = compute_comm_paths(graph, flows, k=5)
        off_ids = set(flows.flow_id[flows.is_offload])
        for fid in comm:
            assert fid not in off_ids

    def test_all_comm_flows_present(self, graph, flows):
        comm = compute_comm_paths(graph, flows, k=5)
        comm_ids = set(flows.flow_id[~flows.is_offload])
        for fid in comm_ids:
            assert fid in comm

    def test_paths_per_flow_at_most_k(self, graph, flows):
        k = 5
        comm = compute_comm_paths(graph, flows, k=k)
        for _, paths in comm.items():
            assert len(paths) <= k


# offloading flow paths
class TestOffloadPaths:
    def test_returns_dict(self, graph, flows, server_sat_ids):
        off = compute_offload_paths(graph, flows, server_sat_ids, k_per_server=3, k_total=10)
        assert isinstance(off, dict)

    def test_only_offload_flows(self, graph, flows, server_sat_ids):
        """Should not contain communication flow IDs."""
        off = compute_offload_paths(graph, flows, server_sat_ids)
        comm_ids = set(flows.flow_id[~flows.is_offload])
        for fid in off:
            assert fid not in comm_ids

    def test_paths_end_at_server(self, graph, flows, server_sat_ids):
        off = compute_offload_paths(graph, flows, server_sat_ids, k_per_server=3, k_total=10)
        server_set = set(server_sat_ids)
        for fid, paths in off.items():
            for nodes, _, _ in paths:
                assert nodes[-1] in server_set, f"Flow {fid} path ends at {nodes[-1]}, not a server"

    def test_server_idx_valid(self, graph, flows, server_sat_ids):
        off = compute_offload_paths(graph, flows, server_sat_ids, k_per_server=3, k_total=10)
        for _, paths in off.items():
            for _, _, srv_idx in paths:
                assert 0 <= srv_idx < len(server_sat_ids)

    def test_sorted_by_delay(self, graph, flows, server_sat_ids):
        off = compute_offload_paths(graph, flows, server_sat_ids, k_per_server=3, k_total=10)
        for _, paths in off.items():
            delays = [d for _, d, _ in paths]
            assert delays == sorted(delays)


# full path data bundle
class TestPathData:
    def test_total_path_count(self, path_data, flows):
        """With k=10, expect at most 10 paths per flow."""
        assert path_data.n_paths <= flows.n_flows * 10
        assert path_data.n_paths >= flows.n_flows  # at least 1 per flow

    def test_avg_paths_per_flow(self, path_data, flows):
        avg = path_data.n_paths / flows.n_flows
        assert 1.0 <= avg <= 10.0

    def test_every_flow_has_paths(self, path_data, flows):
        for f in range(flows.n_flows):
            n_paths_f = path_data.flow_path_end[f] - path_data.flow_path_start[f]
            assert n_paths_f >= 1, f"Flow {f} has 0 paths"

    def test_flow_path_ranges_contiguous(self, path_data, flows):
        """End of flow f should equal start of flow f+1."""
        for f in range(flows.n_flows - 1):
            assert (
                path_data.flow_path_end[f] == path_data.flow_path_start[f + 1]
            ), f"Gap between flow {f} and {f+1}"

    def test_flow_path_ranges_cover_all(self, path_data, flows):
        assert path_data.flow_path_start[0] == 0
        assert path_data.flow_path_end[-1] == path_data.n_paths

    def test_path_to_flow_consistent(self, path_data, flows):
        for f in range(flows.n_flows):
            fid = int(flows.flow_id[f])
            start = path_data.flow_path_start[f]
            end = path_data.flow_path_end[f]
            assert (
                path_data.path_to_flow[start:end] == fid
            ).all(), f"path_to_flow mismatch for flow {f}"

    def test_comm_paths_correct_endpoints(self, path_data, flows):
        """Communication flow paths should start at src and end at dst."""
        comm_mask = ~flows.is_offload
        for f in np.where(comm_mask)[0]:
            for path in get_flow_paths(path_data, f):
                assert (
                    path.nodes[0] == flows.src_sat[f]
                ), f"Flow {f} path doesn't start at src {flows.src_sat[f]}"
                assert (
                    path.nodes[-1] == flows.dst_sat[f]
                ), f"Flow {f} path doesn't end at dst {flows.dst_sat[f]}"

    def test_offload_paths_end_at_server(self, path_data, flows, server_sat_ids):
        server_set = set(server_sat_ids)
        off_mask = flows.is_offload
        for f in np.where(off_mask)[0]:
            for path in get_flow_paths(path_data, f):
                assert (
                    path.nodes[-1] in server_set
                ), f"Offload flow {f} path ends at {path.nodes[-1]}, not a server"

    def test_offload_paths_have_server_id(self, path_data, flows):
        off_mask = flows.is_offload
        for f in np.where(off_mask)[0]:
            for path in get_flow_paths(path_data, f):
                assert path.server_id is not None, f"Offload flow {f} path missing server_id"

    def test_comm_paths_server_minus_one(self, path_data, flows):
        """Communication flow paths should have server = -1 in the index array."""
        comm_mask = ~flows.is_offload
        for f in np.where(comm_mask)[0]:
            start = path_data.flow_path_start[f]
            end = path_data.flow_path_end[f]
            assert (path_data.path_to_server[start:end] == -1).all()

    def test_path_objects_are_path_type(self, path_data):
        for p in path_data.paths:
            assert isinstance(p, Path)

    def test_path_ids_sequential(self, path_data):
        for i, p in enumerate(path_data.paths):
            assert p.path_id == i


# phi sparse matrix
class TestPhi:
    def test_shape(self, path_data, graph):
        P = path_data.n_paths
        E = graph.number_of_edges()
        assert path_data.Phi.shape == (P, E)

    def test_nonzeros_reasonable(self, path_data):
        """Each path should use between 0 and 30 edges."""
        for p in range(path_data.n_paths):
            nnz = path_data.Phi[p].nnz
            assert 0 <= nnz <= 30, f"Path {p} has {nnz} edges"

    def test_values_are_ones(self, path_data):
        """Phi should only contain 1.0 entries."""
        data = path_data.Phi.data
        if len(data) > 0:
            assert (data == 1.0).all()

    def test_nonzeros_match_hop_count(self, path_data):
        """Phi row nonzeros should equal the path's hop count."""
        for p in range(path_data.n_paths):
            nnz = path_data.Phi[p].nnz
            assert (
                nnz == path_data.hop_counts[p]
            ), f"Path {p}: Phi nnz={nnz} but hop_count={path_data.hop_counts[p]}"

    def test_edge_indices_valid(self, path_data):
        """All column indices in Phi should be valid edge indices."""
        cols = path_data.Phi.indices
        assert (cols >= 0).all()
        assert (cols < path_data.n_edges).all()


# link loads utility
class TestLinkLoads:
    def test_shape(self, path_data):
        x = np.ones(path_data.n_paths)
        loads = get_link_loads(path_data, x)
        assert loads.shape == (path_data.n_edges,)

    def test_non_negative(self, path_data):
        x = np.ones(path_data.n_paths)
        loads = get_link_loads(path_data, x)
        assert (loads >= 0).all()

    def test_zero_allocation_zero_load(self, path_data):
        x = np.zeros(path_data.n_paths)
        loads = get_link_loads(path_data, x)
        assert (loads == 0).all()

    def test_some_links_loaded(self, path_data):
        """With uniform allocation, at least some links should have load > 0."""
        x = np.ones(path_data.n_paths)
        loads = get_link_loads(path_data, x)
        assert loads.max() > 0


# get flow paths utility
class TestGetFlowPaths:
    def test_returns_list_of_paths(self, path_data):
        paths = get_flow_paths(path_data, 0)
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, Path)

    def test_correct_count(self, path_data):
        n_expected = path_data.flow_path_end[0] - path_data.flow_path_start[0]
        paths = get_flow_paths(path_data, 0)
        assert len(paths) == n_expected

    def test_flow_id_matches(self, path_data, flows):
        for f in range(min(10, flows.n_flows)):
            fid = int(flows.flow_id[f])
            for path in get_flow_paths(path_data, f):
                assert path.flow_id == fid
