from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


def run(command: list[str]) -> None:
    print()
    print("=" * 80)
    print("RUNNING")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    python = sys.executable

    run([python, "scripts/languageexpertfusion/prepare.py"])

    run([
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "scripts/mlpinteractionfusion/train.py",
    ])

    run([python, "scripts/languageexpertfusion/evaluate.py"])


if __name__ == "__main__":
    main()
