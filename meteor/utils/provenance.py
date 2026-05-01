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

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_sha(short: bool = True) -> str:
    """Return current git SHA"""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "nogit"


def get_git_dirty() -> bool:
    """Return True if the working tree has uncommited changes"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=2
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileExistsError, OSError):
        return False


def make_output_dir(
    experiment_name: str,
    base_dir: str | Path = "outputs",
) -> Path:
    """Format: {base_dir}/{experiment_name}/{YYYYMMDD}_{git_sha}[runN]/"""
    base = Path(base_dir) / experiment_name
    base.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    sha = get_git_sha()
    stem = f"{date_str}_{sha}"

    candidate = base / stem
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    n = 2
    while True:
        candidate = base / f"{stem}_run{n}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        n += 1


def write_provenance(out_dir: Path, extra: dict | None = None) -> None:
    """Write a reproducible .json with git state, host, timestamp, and command"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(short=False),
        "git_sha_short": get_git_sha(short=True),
        "git_dirty": get_git_dirty(),
        "host": socket.gethostname(),
        "python_version": sys.version,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    if extra:
        record.update(extra)
    (out_dir / "provenance.json").write_text(json.dumps(record, indent=2, default=str))


def write_config_snapshot(out_dir: Path, config_path: Path) -> None:
    """Copy input conifg into output dir"""
    text = Path(config_path).read_text()
    (out_dir / "config.yaml").write_text(text)
