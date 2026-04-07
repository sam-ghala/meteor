"""
File created at: 2026-03-26 08:24:35
Author: Sam Ghalayini
meteor/tests/test_traffic.py
"""

import numpy as np
import pytest

from data.constellation import Topology, orbital_config
from data.traffic import (
    CLASS_PARAMS,
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_OFFLOAD_PROB,
    FlowTable,
    build_population_weights,
    compute_network_capacity,
    compute_total_demand,
    generate_flows,
    get_class_flows,
    get_offloading_flows,
    scale_to_load,
    summarize_flows,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def topo():
    cfg = orbital_config(planes=22, sat_per_plane=18)
    return Topology(time_t=0, config=cfg)


@pytest.fixture
def small_topo():
    cfg = orbital_config(planes=6, sat_per_plane=6)
    return Topology(time_t=0, config=cfg)


@pytest.fixture
def flows(topo):
    rng = np.random.default_rng(42)
    return generate_flows(topo, n_flows=5000, rng=rng)


@pytest.fixture
def small_flows(small_topo):
    rng = np.random.default_rng(42)
    return generate_flows(small_topo, n_flows=500, rng=rng)


# ──────────────────────────────────────────────
# Class params
# ──────────────────────────────────────────────


class TestClassParams:
    def test_three_classes_defined(self):
        assert len(CLASS_PARAMS) == 3

    def test_class_ids(self):
        for cid in [0, 1, 2]:
            assert cid in CLASS_PARAMS
            assert CLASS_PARAMS[cid].class_id == cid

    def test_weights_sum_to_one(self):
        total = sum(CLASS_PARAMS[c].w for c in range(3))
        assert abs(total - 1.0) < 1e-9

    def test_deadlines_ordered(self):
        assert CLASS_PARAMS[0].tau < CLASS_PARAMS[1].tau < CLASS_PARAMS[2].tau

    def test_demands_positive(self):
        for c in range(3):
            assert CLASS_PARAMS[c].d_f > 0
            assert CLASS_PARAMS[c].L_f > 0

    def test_class_mix_sums_to_one(self):
        assert abs(DEFAULT_CLASS_WEIGHTS.sum() - 1.0) < 1e-9

    def test_offload_prob_range(self):
        assert (DEFAULT_OFFLOAD_PROB >= 0).all()
        assert (DEFAULT_OFFLOAD_PROB <= 1).all()

    def test_voice_never_offloaded(self):
        assert DEFAULT_OFFLOAD_PROB[0] == 0.0


# ──────────────────────────────────────────────
# Population weights
# ──────────────────────────────────────────────


class TestPopulationWeights:
    def test_shape(self, topo):
        w = build_population_weights(topo)
        assert w.shape == (topo.N,)

    def test_sums_to_one(self, topo):
        w = build_population_weights(topo)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_no_negative(self, topo):
        w = build_population_weights(topo)
        assert (w >= 0).all()

    def test_not_uniform(self, topo):
        w = build_population_weights(topo)
        assert w.max() > w.min() * 5, "Weights should be heavily skewed, not near-uniform"

    def test_small_topology(self, small_topo):
        w = build_population_weights(small_topo)
        assert w.shape == (small_topo.N,)
        assert abs(w.sum() - 1.0) < 1e-9


# ──────────────────────────────────────────────
# Flow table structure
# ──────────────────────────────────────────────


class TestFlowTableStructure:
    def test_flow_count(self, flows):
        assert flows.n_flows == 5000

    def test_all_arrays_same_length(self, flows):
        n = flows.n_flows
        assert len(flows.flow_id) == n
        assert len(flows.src_sat) == n
        assert len(flows.dst_sat) == n
        assert len(flows.class_id) == n
        assert len(flows.d_f) == n
        assert len(flows.L_f) == n
        assert len(flows.W_f) == n
        assert len(flows.tau) == n
        assert len(flows.w) == n
        assert len(flows.is_offload) == n

    def test_flow_ids_sequential(self, flows):
        expected = np.arange(flows.n_flows)
        np.testing.assert_array_equal(flows.flow_id, expected)

    def test_class_ids_valid(self, flows):
        assert set(np.unique(flows.class_id)).issubset({0, 1, 2})


# ──────────────────────────────────────────────
# Class distribution
# ──────────────────────────────────────────────


class TestClassDistribution:
    def test_class_mix_approximately_correct(self, topo):
        rng = np.random.default_rng(99)
        f = generate_flows(topo, n_flows=10000, rng=rng)
        for cid, expected_frac in enumerate(DEFAULT_CLASS_WEIGHTS):
            actual_frac = (f.class_id == cid).mean()
            # Within 3 percentage points for 10k samples
            assert (
                abs(actual_frac - expected_frac) < 0.03
            ), f"Class {cid}: expected ~{expected_frac:.2f}, got {actual_frac:.2f}"

    def test_all_three_classes_present(self, flows):
        for cid in [0, 1, 2]:
            assert (flows.class_id == cid).sum() > 0, f"Class {cid} has zero flows"


# ──────────────────────────────────────────────
# Source / destination validity
# ──────────────────────────────────────────────


class TestSourceDestination:
    def test_src_valid_node_ids(self, flows, topo):
        assert (flows.src_sat >= 0).all()
        assert (flows.src_sat < topo.N).all()

    def test_dst_valid_for_comm_flows(self, flows, topo):
        comm = ~flows.is_offload
        dst_comm = flows.dst_sat[comm]
        assert (dst_comm >= 0).all()
        assert (dst_comm < topo.N).all()

    def test_no_self_loops(self, flows):
        comm = ~flows.is_offload
        src_comm = flows.src_sat[comm]
        dst_comm = flows.dst_sat[comm]
        collisions = (src_comm == dst_comm).sum()
        assert collisions == 0, f"{collisions} flows have src == dst"

    def test_offload_dst_is_negative_one(self, flows):
        off = flows.is_offload
        if off.any():
            assert (flows.dst_sat[off] == -1).all()

    def test_comm_dst_not_negative(self, flows):
        comm = ~flows.is_offload
        assert (flows.dst_sat[comm] >= 0).all()


# ──────────────────────────────────────────────
# Offloading flows
# ──────────────────────────────────────────────


class TestOffloading:
    def test_voice_never_offloaded(self, flows):
        voice_mask = flows.class_id == 0
        assert not (flows.is_offload & voice_mask).any(), "Voice flows should never be offloaded"

    def test_offload_has_positive_workload(self, flows):
        off = flows.is_offload
        if off.any():
            assert (flows.W_f[off] > 0).all()

    def test_comm_has_zero_workload(self, flows):
        comm = ~flows.is_offload
        assert (flows.W_f[comm] == 0).all()

    def test_workload_range(self, flows):
        off = flows.is_offload
        if off.any():
            assert flows.W_f[off].min() >= 1e7
            assert flows.W_f[off].max() <= 1e9

    def test_offload_indices_helper(self, flows):
        idx = get_offloading_flows(flows)
        assert (flows.is_offload[idx]).all()
        assert len(idx) == flows.is_offload.sum()


# ──────────────────────────────────────────────
# Per-flow parameters match class table
# ──────────────────────────────────────────────


class TestParameterConsistency:
    def test_demand_matches_class(self, flows):
        for cid in range(3):
            mask = flows.class_id == cid
            if mask.any():
                expected = CLASS_PARAMS[cid].d_f
                np.testing.assert_allclose(flows.d_f[mask], expected)

    def test_data_size_matches_class(self, flows):
        for cid in range(3):
            mask = flows.class_id == cid
            if mask.any():
                expected = CLASS_PARAMS[cid].L_f
                np.testing.assert_allclose(flows.L_f[mask], expected)

    def test_deadline_matches_class(self, flows):
        for cid in range(3):
            mask = flows.class_id == cid
            if mask.any():
                expected = CLASS_PARAMS[cid].tau
                np.testing.assert_allclose(flows.tau[mask], expected)

    def test_weight_matches_class(self, flows):
        for cid in range(3):
            mask = flows.class_id == cid
            if mask.any():
                expected = CLASS_PARAMS[cid].w
                np.testing.assert_allclose(flows.w[mask], expected)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_result(self, topo):
        f1 = generate_flows(topo, 1000, rng=np.random.default_rng(123))
        f2 = generate_flows(topo, 1000, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(f1.src_sat, f2.src_sat)
        np.testing.assert_array_equal(f1.dst_sat, f2.dst_sat)
        np.testing.assert_array_equal(f1.class_id, f2.class_id)
        np.testing.assert_array_equal(f1.W_f, f2.W_f)

    def test_different_seed_different_result(self, topo):
        f1 = generate_flows(topo, 1000, rng=np.random.default_rng(1))
        f2 = generate_flows(topo, 1000, rng=np.random.default_rng(2))
        assert not np.array_equal(f1.src_sat, f2.src_sat)


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────


class TestUtilities:
    def test_compute_total_demand(self, flows):
        total = compute_total_demand(flows)
        assert total > 0
        assert abs(total - flows.d_f.sum()) < 1e-6

    def test_network_capacity_positive(self, topo):
        cap = compute_network_capacity(topo)
        assert cap > 0

    def test_get_class_flows(self, flows):
        for cid in range(3):
            idx = get_class_flows(flows, cid)
            assert (flows.class_id[idx] == cid).all()
            assert len(idx) == (flows.class_id == cid).sum()

    def test_summarize_returns_dict(self, flows):
        s = summarize_flows(flows)
        assert isinstance(s, dict)
        assert s["n_flows"] == flows.n_flows
        assert s["n_voice"] + s["n_video"] + s["n_file"] == flows.n_flows
        assert abs(s["ct_voice"] + s["ct_video"] + s["ct_file"] - 100.0) < 0.1


# ──────────────────────────────────────────────
# Load scaling
# ──────────────────────────────────────────────


class TestScaleToLoad:
    def test_returns_flow_table(self, topo):
        rng = np.random.default_rng(42)
        f = scale_to_load(topo, target_load=0.5, rng=rng)
        assert isinstance(f, FlowTable)
        assert f.n_flows > 0

    def test_higher_load_more_demand(self, topo):
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(10)
        f_low = scale_to_load(topo, target_load=0.25, rng=rng1)
        f_high = scale_to_load(topo, target_load=0.75, rng=rng2)
        assert compute_total_demand(f_high) > compute_total_demand(f_low)

    def test_load_roughly_correct(self, topo):
        rng = np.random.default_rng(42)
        target = 0.5
        f = scale_to_load(topo, target_load=target, rng=rng)
        cap = compute_network_capacity(topo)
        actual_load = compute_total_demand(f) / cap
        # Within 20% of target (stochastic, so generous tolerance)
        assert abs(actual_load - target) < 0.20, f"Target load {target}, actual {actual_load:.3f}"


# ──────────────────────────────────────────────
# Run with: python -m pytest tests/test_traffic.py -v
# ──────────────────────────────────────────────
