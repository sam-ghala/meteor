"""
File created at: 2026-05-01 22:00:31
Author: Sam Ghalayini
meteor/meteor/utils/provenance.py

Utilities for reproducible experiments

Each experiment runs:
    outputs/{experiment_name}/{date}_{git_sha}/
        config.yaml
        provenance.json
        samples.npz
        summary.json
        figure.pdf
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_git_sha(repo_root: Path | None = None) -> tuple[str, bool]:
    """Return current git SHA"""
    cwd = repo_root or Path.cwd()
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        is_dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        is_dirty = False
    return sha, is_dirty


def make_output_dir(
    base: Path,
    experiment_name: str,
    timestamp: dt.datetime | None = None,
) -> Path:
    """Format: {base_dir}/{experiment_name}/{YYYYMMDD}_{git_sha}[runN]/"""
    ts = timestamp or dt.datetime.now()
    name = f"{ts:%Y-%m-%d_%H-%M-%S}"
    out = base / experiment_name / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_provenance(
    out_dir: Path,
    config_path: Path | None,
    extra_metadata: dict | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write a reproducible .json with git state, host, timestamp, and command"""
    sha, is_dirty = get_git_sha(repo_root)
    metadata = {
        "git_sha": sha,
        "git_dirty": is_dirty,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "cwd": str(Path.cwd()),
        "config_path": str(config_path) if config_path else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(metadata, indent=2))

    if config_path is not None and config_path.exists():
        shutil.copy(config_path, out_dir / "config.yaml")

    if is_dirty:
        logger.warning(
            f"Uncomitted changes in git tree, results may not be exactly reproducible, sha: {sha}",
        )

    return prov_path
