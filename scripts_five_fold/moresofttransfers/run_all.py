from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.moresofttransfers.config import RUN_EXPERIMENTS


def run(command: list[str]) -> None:
    print()
    print("=" * 88)
    print("RUNNING")
    print(" ".join(command))
    print("=" * 88)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    python = sys.executable
    torchrun = str(Path(python).with_name("torchrun"))

    for experiment in RUN_EXPERIMENTS:
        run([
            torchrun,
            "--standalone",
            "--nproc_per_node=2",
            "scripts/moresofttransfers/train.py",
            "--experiment",
            experiment,
        ])

        run([
            python,
            "scripts/moresofttransfers/evaluate.py",
            "--experiment",
            experiment,
        ])

    print()
    print("=" * 88)
    print("ALL MORE-SOFT-TRANSFER EXPERIMENTS COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
