from __future__ import annotations

import csv
import sys
from pathlib import Path


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.resadapt_m import config

from scripts.train import (
    load_manifest,
    validate_no_audio_overlap,
    validate_no_duplicate_audio,
    write_csv,
)


def main() -> None:
    train_rows = load_manifest(
        config.ORIGINAL_TRAIN_MANIFEST,
    )

    validation_rows = load_manifest(
        config.ORIGINAL_VALIDATION_MANIFEST,
    )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name=(
            "Miami original train"
        ),
    )

    validate_no_duplicate_audio(
        validation_rows,
        dataset_name=(
            "Miami original validation"
        ),
    )

    validate_no_audio_overlap(
        train_rows,
        validation_rows,
        first_name="train",
        second_name="validation",
    )

    union_rows = [
        *train_rows,
        *validation_rows,
    ]

    fieldnames: list[str] = []

    for row in union_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    write_csv(
        config.TRAIN_MANIFEST,
        union_rows,
        fieldnames=fieldnames,
    )

    print()
    print(
        "Miami train+validation union created."
    )

    print(
        f"Train: "
        f"{len(train_rows)}"
    )

    print(
        f"Validation: "
        f"{len(validation_rows)}"
    )

    print(
        f"Union: "
        f"{len(union_rows)}"
    )

    print(
        f"Saved: "
        f"{config.TRAIN_MANIFEST}"
    )


if __name__ == "__main__":
    main()