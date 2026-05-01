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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class THTResult:
    """Output of compute_tht_samples"""

    tht_samples: np.ndarray
    n_changes_per_step: np.ndarray
    type_changes_per_step: np.ndarray
    config_snapshot: dict
    elapsed_s: float


def compute_tht_samples(
    config: ConstellationConfig,
    n_steps: int,
    dt_s: float,
    t_start_s: float = 0.0,
    progress_every: int | None = None,
) -> THTResult:
    """Measure THT for a constellation over n-steps spaces by delta"""
    pass


def _run_lengths_to_samples(n_changes_per_step: np.ndarray, dt_s: float) -> np.ndarray:
    """
    Convert per-interval change counts into era
    An era spanning k unchanged intervals has THT= k * dt_s
    """
    pass


# summary stats
def summary_statistics(tht_samples_s: np.ndarray) -> dict:
    """
    Compute summary stats
    """
    if len(tht_samples_s) == 0:
        return {
            "n_eras": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "std_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "p25_ms": float("nan"),
            "p75_ms": float("nan"),
            "p90_ms": float("nan"),
            "p99_ms": float("nan"),
        }

    samples_ms = tht_samples_s * 1000.0
    return {
        "n_eras": int(len(samples_ms)),
        "mean_ms": float(samples_ms.mean()),
        "median_ms": float(np.median(samples_ms)),
        "std_ms": float(samples_ms.std()),
        "min_ms": float(samples_ms.min()),
        "max_ms": float(samples_ms.max()),
        "p25_ms": float(np.percentile(samples_ms, 25)),
        "p75_ms": float(np.percentile(samples_ms, 75)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
    }


def cdf_points(tht_samples_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x,y) arrays for plotting an emperical CDF
    """

    if len(tht_samples_s) == 0:
        return np.empty(0), np.empty(0)
    x = np.sort(tht_samples_s) * 1000.0
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y
