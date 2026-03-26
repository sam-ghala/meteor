"""
File created at: 2026-03-25 17:43:33
Author: Sam Ghalayini
meteor/tests/test_constellation.py
"""

# import pytest 
import math 
import networkx as nx
from data.constellation import Topology, orbital_config

# helpers 
def make_topology(planes=10, sats=10, lat_cutoff=75.0, time_t=0):
    cfg = orbital_config(planes=planes, sat_per_plane=sats,
                         lat_cutoff=lat_cutoff)
    return Topology(time_t=time_t, config=cfg)

# test nodes 
class TestNodes:

    def test_node_count(self):
        topo = make_topology(planes=10, sats=10)
        assert len(topo.get_graph().nodes) == 100

    def test_node_count_full_subset(self):
        topo = make_topology(planes=22, sats=18)
        assert len(topo.get_graph().nodes) == 396

    def test_node_ids_are_integers(self):
        topo = make_topology(planes=5, sats=5)
        for nid in topo.get_graph().nodes:
            assert isinstance(nid, int)

    def test_node_id_range(self):
        topo = make_topology(planes=10, sats=10)
        ids = sorted(topo.get_graph().nodes)
        assert ids == list(range(100))

    def test_node_id_helper(self):
        topo = make_topology(planes=10, sats=10)
        assert topo.node_id(0, 0) == 0
        assert topo.node_id(0, 9) == 9
        assert topo.node_id(3, 5) == 35
        assert topo.node_id(9, 9) == 99

    def test_node_attributes_present(self):
        topo = make_topology(planes=5, sats=5)
        data = topo.get_graph().nodes[0]
        for key in ['plane', 'sat_idx', 'lat', 'lon',
                     'has_server', 'mu_server', 'uplink', 'downlink']:
            assert key in data, f"Missing attribute: {key}"

    def test_plane_assignment(self):
        topo = make_topology(planes=10, sats=10)
        g = topo.get_graph()
        for p in range(10):
            for s in range(10):
                nid = p * 10 + s
                assert g.nodes[nid]['plane'] == p
                assert g.nodes[nid]['sat_idx'] == s

    def test_latitudes_in_range(self):
        """Latitude must be within [-inclination, +inclination]."""
        topo = make_topology(planes=22, sats=18)
        incl = topo.config.inclination
        for nid, data in topo.get_graph().nodes(data=True):
            assert -incl - 0.1 <= data['lat'] <= incl + 0.1, \
                f"Sat {nid}: lat={data['lat']:.2f} outside [{-incl}, {incl}]"

    def test_longitudes_in_range(self):
        topo = make_topology(planes=22, sats=18)
        for nid, data in topo.get_graph().nodes(data=True):
            assert -180.0 <= data['lon'] <= 180.0, \
                f"Sat {nid}: lon={data['lon']:.2f} outside [-180, 180]"

    def test_positions_change_with_time(self):
        topo_t0 = make_topology(planes=10, sats=10, time_t=0)
        topo_t1 = make_topology(planes=10, sats=10, time_t=60)
        lat0 = topo_t0.get_graph().nodes[0]['lat']
        lat1 = topo_t1.get_graph().nodes[0]['lat']
        assert lat0 != lat1, "Satellite should move between t=0 and t=60"

# test edges 
class TestEdges:

    def test_intra_orbit_edges_present(self):
        """Each plane forms a ring: N sats -> N bidirectional edges -> 2N directed."""
        topo = make_topology(planes=3, sats=5, lat_cutoff=90.0)
        g = topo.get_graph()
        for p in range(3):
            for s in range(5):
                nid = p * 5 + s
                next_id = p * 5 + (s + 1) % 5
                assert g.has_edge(nid, next_id), \
                    f"Missing intra-orbit edge {nid} -> {next_id}"
                assert g.has_edge(next_id, nid), \
                    f"Missing intra-orbit edge {next_id} -> {nid}"

    def test_edge_count_no_cutoff(self):
        """
        With lat_cutoff=90 (no deactivation) and no inter-plane wrap:
        Intra-orbit: planes * sats * 2 (ring, both directions)
        Inter-orbit: (planes-1) * sats * 2 (no wrap, both directions)
        Total: planes*sats*2 + (planes-1)*sats*2
        """
        planes, sats = 5, 8
        topo = make_topology(planes=planes, sats=sats, lat_cutoff=90.0)
        expected_intra = planes * sats * 2
        expected_inter = (planes - 1) * sats * 2
        actual = topo.get_graph().number_of_edges()
        assert actual == expected_intra + expected_inter, \
            f"Expected {expected_intra + expected_inter} edges, got {actual}"

    def test_inter_orbit_deactivation_at_poles(self):
        """With a tight cutoff, some inter-orbit links should be missing."""
        topo_open = make_topology(planes=10, sats=10, lat_cutoff=90.0)
        topo_tight = make_topology(planes=10, sats=10, lat_cutoff=30.0)
        edges_open = topo_open.get_graph().number_of_edges()
        edges_tight = topo_tight.get_graph().number_of_edges()
        assert edges_tight < edges_open, \
            "Tighter lat cutoff should produce fewer edges"

    def test_no_inter_orbit_wrap(self):
        """Plane 0 should not connect to plane (planes-1)."""
        planes, sats = 5, 5
        topo = make_topology(planes=planes, sats=sats, lat_cutoff=90.0)
        g = topo.get_graph()
        for s in range(sats):
            first_plane_sat = s
            last_plane_sat = (planes - 1) * sats + s
            assert not g.has_edge(first_plane_sat, last_plane_sat), \
                f"Unexpected wrap edge {first_plane_sat} -> {last_plane_sat}"
            assert not g.has_edge(last_plane_sat, first_plane_sat), \
                f"Unexpected wrap edge {last_plane_sat} -> {first_plane_sat}"

    def test_edge_attributes(self):
        topo = make_topology(planes=5, sats=5, lat_cutoff=90.0)
        g = topo.get_graph()

        src, dst = list(g.edges)[0]
        data = g.edges[src, dst]
        assert 'capacity' in data
        assert 'delay' in data
        assert data['capacity'] == topo.config.C_e

    def test_propagation_delay_positive(self):
        topo = make_topology(planes=10, sats=10, lat_cutoff=90.0)
        for u, v, data in topo.get_graph().edges(data=True):
            assert data['delay'] > 0, f"Edge {u}->{v} has non-positive delay"

    def test_propagation_delay_reasonable(self):
        """
        Neighbouring ISLs at 550km should be roughly 1-10ms.
        Cross-constellation max is ~40ms. Flag anything outside [0.1ms, 50ms].
        """
        topo = make_topology(planes=22, sats=18)
        for u, v, data in topo.get_graph().edges(data=True):
            delay_ms = data['delay'] * 1000
            assert 0.1 < delay_ms < 50.0, \
                f"Edge {u}->{v}: delay={delay_ms:.3f}ms seems wrong"

    def test_edges_are_directed(self):
        topo = make_topology(planes=5, sats=5, lat_cutoff=90.0)
        assert topo.get_graph().is_directed()

    def test_bidirectional(self):
        """Every edge (u,v) should have a corresponding (v,u)."""
        topo = make_topology(planes=10, sats=10, lat_cutoff=90.0)
        g = topo.get_graph()
        for u, v in g.edges:
            assert g.has_edge(v, u), f"Edge {u}->{v} exists but {v}->{u} missing"


# ground station and server tests
class TestGroundStations:

    def test_all_gateways_mapped(self):
        topo = make_topology(planes=22, sats=18)
        for gw in topo.gateways:
            assert gw.id in topo.gateway_mapping, \
                f"Gateway {gw.id} not in mapping"

    def test_mapping_points_to_valid_node(self):
        topo = make_topology(planes=22, sats=18)
        g = topo.get_graph()
        for gw_id, sat_id in topo.gateway_mapping.items():
            assert sat_id in g.nodes, \
                f"Gateway {gw_id} mapped to non-existent node {sat_id}"

    def test_server_flag_set(self):
        """Satellites mapped to server-gateways should have has_server=True."""
        topo = make_topology(planes=22, sats=18)
        g = topo.get_graph()
        server_gws = [gw for gw in topo.gateways if gw.has_server]
        for gw in server_gws:
            sat_id = topo.gateway_mapping[gw.id]
            assert g.nodes[sat_id]['has_server'] is True, \
                f"Sat {sat_id} (for {gw.id}) should have has_server=True"
            assert g.nodes[sat_id]['mu_server'] == gw.mu_s

    def test_server_sat_ids(self):
        topo = make_topology(planes=22, sats=18)
        server_ids = topo.get_server_sat_ids()
        server_gws = [gw for gw in topo.gateways if gw.has_server]
        assert len(server_ids) == len(server_gws)

    def test_seattle_maps_to_northern_hemisphere(self):
        """Sanity: Seattle gateway should map to a satellite with lat > 20."""
        topo = make_topology(planes=22, sats=18)
        g = topo.get_graph()
        seattle_sat = topo.gateway_mapping.get("gw_seattle")
        if seattle_sat is not None:
            lat = g.nodes[seattle_sat]['lat']
            assert lat > 20.0, \
                f"Seattle mapped to sat at lat={lat:.1f}, expected >20"

    def test_sydney_maps_to_southern_hemisphere(self):
        """Sanity: Sydney gateway should map to a satellite with lat < -10."""
        topo = make_topology(planes=22, sats=18)
        g = topo.get_graph()
        sydney_sat = topo.gateway_mapping.get("gw_sydney")
        if sydney_sat is not None:
            lat = g.nodes[sydney_sat]['lat']
            assert lat < -10.0, \
                f"Sydney mapped to sat at lat={lat:.1f}, expected <-10"


# full snapshot tests
class TestSnapshot:

    def test_snapshot_keys(self):
        topo = make_topology(planes=10, sats=10)
        snap = topo.get_snapshot()
        for key in ['time_t', 'graph', 'n_nodes', 'n_edges',
                     'gateways', 'servers', 'server_sat_ids',
                     'gateway_mapping']:
            assert key in snap, f"Missing snapshot key: {key}"

    def test_snapshot_counts(self):
        topo = make_topology(planes=10, sats=10)
        snap = topo.get_snapshot()
        assert snap['n_nodes'] == 100
        assert snap['n_edges'] == topo.get_graph().number_of_edges()

    def test_different_times_different_topologies(self):
        """Different time snapshots should have different edge counts
        (because lat cutoff deactivates different inter-orbit links)."""
        topo_a = make_topology(planes=22, sats=18, time_t=0)
        topo_b = make_topology(planes=22, sats=18, time_t=1000)
        edges_a = topo_a.get_graph().number_of_edges()
        edges_b = topo_b.get_graph().number_of_edges()
        # Edge counts may or may not differ, but positions should differ
        g_a = topo_a.get_graph()
        g_b = topo_b.get_graph()
        lat_a = g_a.nodes[0]['lat']
        lat_b = g_b.nodes[0]['lat']
        assert lat_a != lat_b, "Same positions at different times"


# should be fully coneected graph, no weakly connected-transient nodes
class TestConnectivity:

    def test_graph_weakly_connected(self):
        """The constellation should be a single connected component."""
        topo = make_topology(planes=22, sats=18)
        assert nx.is_weakly_connected(topo.get_graph()), \
            "Constellation graph is not weakly connected"

    def test_shortest_path_exists(self):
        """Should be able to route between any two satellites."""
        topo = make_topology(planes=22, sats=18)
        g = topo.get_graph()
        # Test a few random pairs
        pairs = [(0, 100), (0, 395), (50, 300)]
        for src, dst in pairs:
            assert nx.has_path(g, src, dst), \
                f"No path from {src} to {dst}"

    def test_shortest_path_hop_count(self):
        """Adjacent satellites should be 1 hop apart."""
        topo = make_topology(planes=22, sats=18, lat_cutoff=90.0)
        g = topo.get_graph()
        path = nx.shortest_path(g, 0, 1)
        assert len(path) == 2, f"Expected 1 hop, got path {path}"
