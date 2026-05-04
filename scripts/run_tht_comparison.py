"""
File created at: 2026-05-01 23:42:20
Author: Sam Ghalayini
meteor/scripts/run_tht_comparison.py

run iwth
    python scripts/run_tht_comparison.py --config configs/tht_comparison.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from meteor.analysis import compute_tht_samples
from meteor.config import get_preset
from meteor.utils import make_output_dir, write_provenance

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


logger = logging.getLogger(__name__)


def _plot_grid(results: list[tuple[str, np.ndarray, float]], out_path: Path) -> None:
    n = len(results)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False)

    for idx, (name, holding_times_s, mean_ms) in enumerate(results):
        ax = axes[idx // n_cols][idx % n_cols]
        if len(holding_times_s) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        else:
            sorted_ms = np.sort(holding_times_s) * 1e3
            cdf_y = np.arange(1, len(sorted_ms) + 1) / len(sorted_ms)
            ax.plot(sorted_ms, cdf_y, lw=1.6)
            ax.axvline(
                mean_ms, linestyle="--", color="C2", linewidth=1.0, label=f"Mean: {mean_ms:.1f} ms"
            )
        ax.set_xscale("log")
        ax.set_title(f"{name}\nmean={mean_ms:.1f}ms", fontsize=9)
        ax.set_xlabel("THT (ms)", fontsize=8)
        ax.set_ylabel("CDF", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Topology Holding Time Comparison", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="THT comparison")
    p.add_argument("--config", required=True, type=Path, help="YAML config")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    preset_name: list[str] = cfg["presets"]
    n_steps = int(cfg["n_steps"])
    dt_s = float(cfg["dt_s"])
    output_base = Path(cfg.get("output_dir", "outputs"))
    enable_ground = bool(cfg.get("ground_access_enabled", True))

    out_dir = make_output_dir(output_base, "tht_comparison")
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
    results: list[tuple[str, np.ndarray, float]] = []
    summary_rows: list[dict] = []
    for name in preset_name:
        logger.info(f"running tht for preset {name}")
        constellation = get_preset(name)
        if not enable_ground:
            constellation = replace(constellation, ground_access_enabled=False)

        result = compute_tht_samples(
            constellation,
            n_steps=n_steps,
            dt_s=dt_s,
            log_every=max(1, n_steps // 10),
        )

        np.save(out_dir / f"holding_times_{name}.npy", result.holding_times_s)
        results.append((name, result.holding_times_s, result.mean_tht_s * 1e3))
        summary_rows.append(
            {
                "preset": name,
                "n_satellites": constellation.n_satellites,
                "n_shells": constellation.n_shells,
                "n_steps": result.n_steps,
                "n_changes": result.n_changes,
                "mean_tht_ms": round(result.mean_tht_s * 1e3, 4),
                "median_tht_ms": round(result.median_tht_s * 1e3, 4),
                "p99_tht_ms": round(result.p99_tht_s * 1e3, 4),
                "total_duration_s": result.total_duration_s,
            }
        )
        logger.info(result.summary())

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    _plot_grid(results, out_dir / "tht_grid.png")
    (out_dir / "tht_summary.json").write_text(json.dumps({"rows": summary_rows}, indent=2))
    logger.info(f"Output at: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
