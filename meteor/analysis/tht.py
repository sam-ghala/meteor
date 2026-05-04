"""
File created at: 2026-05-01 22:00:01
Author: Sam Ghalayini
meteor/meteor/analysis/tht.py

Topology holding time (THT) analysis

Sample N consecutive topology snapshots at fixed intervals
and measure how many intervals until edges are not the same from t to t+1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from meteor.config.constellation import ConstellationConfig
from meteor.constellation.topology import Topology
from meteor.ground.gateways import Gateway

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class THTResult:
    """Output of compute_tht_samples"""

    holding_times_s: np.ndarray
    n_steps: int
    n_changes: int
    dt_s: float
    total_duration_s: float

    @property
    def mean_tht_s(self) -> float:
        if len(self.holding_times_s) == 0:
            return float("nan")
        return float(np.mean(self.holding_times_s))

    @property
    def median_tht_s(self) -> float:
        if len(self.holding_times_s) == 0:
            return float("nan")
        return float(np.median(self.holding_times_s))

    @property
    def p99_tht_s(self) -> float:
        if len(self.holding_times_s) == 0:
            return float("nan")
        return float(np.percentile(self.holding_times_s, 99))

    def summary(self) -> str:
        return (
            f"THT: n_steps={self.n_steps}, n_changes={self.n_changes}, "
            f"mean={self.mean_tht_s * 1e3:.2f}ms, "
            f"median={self.median_tht_s * 1e3:.2f}ms, "
            f"p99={self.p99_tht_s * 1e3:.2f}ms, "
            f"total={self.total_duration_s:.1f}s"
        )


def compute_tht_samples(
    config: ConstellationConfig,
    *,
    n_steps: int,
    dt_s: float,
    t0_s: float = 0.0,
    gateways: list[Gateway] | None = None,
    log_every: int = 1000,
) -> THTResult:
    """Measure THT for a constellation over n-steps spaces by delta"""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 to compare snapshots")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    logger.info(
        f"Starting THT measurement: {n_steps} steps * {dt_s} s = {n_steps * dt_s} s window\n  {config.n_satellites} satellites across {config.n_shells} shells"
    )

    holding_times: list[float] = []
    current_run_steps = 1

    t = t0_s
    prev_topo = Topology.from_config(config, t=t, gateways=gateways)
    prev_edge_set = prev_topo.get_edge_set()

    for step in range(1, n_steps):
        t = t0_s + step * dt_s
        topo = Topology.from_config(config, t=t, gateways=gateways)
        edge_set = topo.get_edge_set()

        if edge_set == prev_edge_set:
            current_run_steps += 1
        else:
            holding_times.append(current_run_steps * dt_s)
            current_run_steps = 1
            prev_edge_set = edge_set

        if log_every and step % log_every == 0:
            logger.info(
                f"THT step {step}/{n_steps} t={t:.3f}, len(holding_times)={len(holding_times)}"
            )

    holding_times.append(current_run_steps * dt_s)

    holding_times_arr = np.asarray(holding_times, dtype=np.float64)
    result = THTResult(
        holding_times_s=holding_times_arr,
        n_steps=n_steps,
        n_changes=len(holding_times_arr) - 1,
        dt_s=dt_s,
        total_duration_s=n_steps * dt_s,
    )
    logger.info(result.summary())
    return result
