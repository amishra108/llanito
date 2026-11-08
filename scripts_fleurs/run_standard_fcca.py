from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = (
    ROOT / "scripts_fleurs"
)

RESULTS = (
    ROOT / "outputs" / "new_results"
)


def run(command):
    print()
    print("=" * 100)
    print(
        "[RUN]",
        " ".join(
            str(x)
            for x in command
        ),
        flush=True,
    )
    print("=" * 100)

    subprocess.run(
        [
            str(x)
            for x in command
        ],
        cwd=ROOT,
        check=True,
        env=os.environ.copy(),
    )


def audit(
    language: str,
):
    stats_path = (
        RESULTS
        / language
        / "_task_stats"
        / "task_stats_256_all_attention.pt"
    )

    basis_path = (
        RESULTS
        / language
        / "fcca"
        / "calibration"
        / "calibration_256_all_attention_v2_means.pt"
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            stats_path
        )

    if not basis_path.exists():
        raise FileNotFoundError(
            basis_path
        )

    stats = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=False,
    )

    basis = torch.load(
        basis_path,
        map_location="cpu",
        weights_only=False,
    )

    checks = {
        "stats calibration size":
            int(
                stats[
                    "calibration_size"
                ]
            )
            == 256,

        "stats calibration seed":
            int(
                stats[
                    "calibration_seed"
                ]
            )
            == 2026,

        "stats targets":
            len(
                stats[
                    "modules"
                ]
            )
            == 288,

        "basis method":
            basis.get(
                "method"
            )
            == "FCCA",

        "basis calibration size":
            int(
                basis[
                    "calibration_size"
                ]
            )
            == 256,

        "basis calibration seed":
            int(
                basis[
                    "calibration_seed"
                ]
            )
            == 2026,

        "basis rank":
            "128"
            in basis[
                "bases"
            ],

        "basis targets":
            len(
                basis[
                    "bases"
                ][
                    "128"
                ]
            )
            == 288,
    }

    for name, passed in (
        checks.items()
    ):
        print(
            f"[audit] {name}: "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

        if not passed:
            raise RuntimeError(
                f"Audit failed: {name}"
            )

    print(
        "[audit PASS] calibration/basis "
        "are exactly N=256 seed=2026 rank=128",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--whisper-language",
        required=True,
    )

    parser.add_argument(
        "--learning-rate",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--microbatch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--force-calibration",
        action="store_true",
    )

    args = parser.parse_args()

    if args.nproc_per_node < 1:
        raise ValueError(
            "--nproc-per-node must be >=1"
        )

    collector = [
        sys.executable,
        "-u",
        SCRIPTS
        / "collect_fcca_task_stats_any_language.py",
        "--language",
        args.language,
        "--whisper-language",
        args.whisper_language,
    ]

    if args.force_calibration:
        collector.append(
            "--force"
        )

    run(
        collector
    )

    builder = [
        sys.executable,
        "-u",
        SCRIPTS
        / "build_standard_fcca_basis.py",
        "--language",
        args.language,
    ]

    if args.force_calibration:
        builder.append(
            "--force"
        )

    run(
        builder
    )

    audit(
        args.language
    )

    train_args = [
        str(
            SCRIPTS
            / "train_whisper_fleurs.py"
        ),
        "--language",
        args.language,
        "--whisper-language",
        args.whisper_language,
        "--method",
        "fcca",
        "--trial",
        str(
            args.trial
        ),
        "--learning-rate",
        str(
            args.learning_rate
        ),
        "--effective-batch-size",
        str(
            args.effective_batch_size
        ),
        "--warmup-ratio",
        str(
            args.warmup_ratio
        ),
        "--microbatch",
        str(
            args.microbatch
        ),
    ]

    if args.nproc_per_node == 1:
        command = [
            sys.executable,
            "-u",
            *train_args,
        ]

    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(
                args.nproc_per_node
            ),
            *train_args,
        ]

    run(
        command
    )


if __name__ == "__main__":
    main()
