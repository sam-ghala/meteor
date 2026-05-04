"""
File created at: 2026-05-01 23:42:22
Author: Sam Ghalayini
meteor/scripts/run_tht_validation.py

Reproduce SaTE Fig 49a): THT distribution for 4 shell Starlink constellation, mean tht of ~70ms

Usage:
    python -m scripts.run_tht_validation
    python -m scripts.run_tht_validation --config configs/tht_validation.yaml

Output dir contains:
    config.yaml
    provenance.json
    samples.npz,
    summary.json,
    figure.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from meteor.analysis.tht import compute_tht_samples
from meteor.config.registry import get_preset
from meteor.utils.provenance import make_output_dir, write_provenance

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _plot_cdf(holding_times_s: np.ndarray, mean_ms: float, out_path: Path) -> None:
    """Plot THT CDF with mean reference line"""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    if len(holding_times_s) == 0:
        ax.text(0.5, 0.5, "No holding time examples", ha="center", va="center")
    else:
        sorted_ms = np.sort(holding_times_s) * 1e3
        cdf_y = np.arange(1, len(sorted_ms) + 1) / len(sorted_ms)
        ax.plot(sorted_ms, cdf_y, lw=2, label=f"METEOR (mean = {mean_ms:.1f} ms)")
        # mark mean
        our_mean_ms = float(holding_times_s.mean()) * 1000.0
        ax.axvline(
            our_mean_ms,
            linestyle="--",
            color="C2",
            linewidth=1.0,
            label=f"Mean: {our_mean_ms:.1f} ms",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Topology Holding Time (ms)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Topology Holding Time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="THT validation")
    p.add_argument("--config", required=True, type=Path, help="YAML config")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    preset_name = cfg["preset"]
    n_steps = int(cfg["n_steps"])
    dt_s = float(cfg["dt_s"])
    output_base = Path(cfg.get("output_dir", "outputs"))
    enable_ground = bool(cfg.get("ground_access_enabled", True))

    logger.info(f"THT validation: preset:{preset_name}, n_steps:{n_steps}, dt_s:{dt_s}")

    constellation = get_preset(preset_name)
    if not enable_ground:
        from dataclasses import replace

        constellation = replace(constellation, ground_access_enabled=False)

    out_dir = make_output_dir(output_base, "tht_validation")
    write_provenance(
        out_dir,
        config_path=args.config,
        extra_metadata={
            "preset": preset_name,
            "n_steps": n_steps,
            "dt_s": dt_s,
            "ground_access_enabled": enable_ground,
        },
    )

    result = compute_tht_samples(
        constellation,
        n_steps=n_steps,
        dt_s=dt_s,
        log_every=max(1, n_steps // 10),
    )

    np.save(out_dir / "holding_times.npy", result.holding_times_s)
    summary = {
        "preset": preset_name,
        "n_steps": result.n_steps,
        "n_changes": result.n_changes,
        "mean_tht_ms": result.mean_tht_s * 1e3,
        "median_tht_ms": result.median_tht_s * 1e3,
        "p99_tht_ms": result.p99_tht_s * 1e3,
        "total_duration_s": result.total_duration_s,
    }
    (out_dir / "tht_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_cdf(result.holding_times_s, result.mean_tht_s * 1e3, out_dir / "tht_cdf.png")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
