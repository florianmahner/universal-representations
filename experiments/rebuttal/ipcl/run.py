#!/usr/bin/env python3
"""Run the IPCL control end to end: factorize, score, then analyze."""
from __future__ import annotations

import subprocess
import sys

from omegaconf import OmegaConf


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    for stage in ("factorize", "score", "analyze"):
        overrides = [] if stage == "analyze" else [f"n_jobs={cfg.n_jobs}"]
        command = [
            sys.executable,
            "-m",
            f"experiments.rebuttal.ipcl.{stage}",
            *overrides,
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
