from __future__ import annotations

import csv
import json
from pathlib import Path


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Input

LLANITO_UNION_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
    / "train_validation_union_manifest.csv"
)

LAUG1800_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "joint"
    / "laug1800"
    / "train_manifest.csv"
)

LLANITO_TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
    / "test_manifest.csv"
)


# Output

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "joint"
    / "laug900"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "train_manifest.csv"
)

OUTPUT_METADATA = (
    OUTPUT_ROOT
    / "metadata.json"
)

EXPECTED_AUGMENTED_ROWS = 900


# CSV

def read_csv(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV does not exist:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header:\n{path}"
            )

        rows = [
            dict(row)
            for row in reader
        ]

        return (
            list(reader.fieldnames),
            rows,
        )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# Audio paths

def audio_path_set(
    rows: list[
        dict[str, str]
    ],
) -> set[str]:
    result = set()

    for row in rows:
        value = row.get(
            "audio_path",
            "",
        ).strip()

        if not value:
            raise RuntimeError(
                "Encountered row with "
                "empty audio_path."
            )

        if value in result:
            raise RuntimeError(
                "Duplicate audio_path:\n"
                f"{value}"
            )

        result.add(
            value
        )

    return result


# Main

def main() -> None:
    (
        laug_fields,
        laug_rows,
    ) = read_csv(
        LAUG1800_MANIFEST
    )

    (
        _,
        llanito_rows,
    ) = read_csv(
        LLANITO_UNION_MANIFEST
    )

    (
        _,
        test_rows,
    ) = read_csv(
        LLANITO_TEST_MANIFEST
    )

    if "audio_path" not in laug_fields:
        raise RuntimeError(
            "laug1800 manifest has no "
            "audio_path column."
        )

    original_paths = audio_path_set(
        llanito_rows
    )

    test_paths = audio_path_set(
        test_rows
    )

    # Exact rows in laug1800 whose audio
    # files are not original Llanito
    # training recordings.

    augmented_rows = [
        row
        for row in laug_rows
        if (
            row["audio_path"].strip()
            not in original_paths
        )
    ]

    if (
        len(augmented_rows)
        != EXPECTED_AUGMENTED_ROWS
    ):
        raise RuntimeError(
            "Could not uniquely recover "
            "the expected 900 augmented "
            "rows from laug1800.\n"
            f"laug1800 rows: "
            f"{len(laug_rows)}\n"
            f"Recovered augmented rows: "
            f"{len(augmented_rows)}\n"
            f"Expected: "
            f"{EXPECTED_AUGMENTED_ROWS}"
        )

    augmented_paths = (
        audio_path_set(
            augmented_rows
        )
    )

    leakage = (
        augmented_paths
        & test_paths
    )

    if leakage:
        raise RuntimeError(
            "Augmented data overlaps "
            "the Llanito fixed test.\n"
            f"Examples: "
            f"{sorted(leakage)[:10]}"
        )

    write_csv(
        OUTPUT_MANIFEST,
        laug_fields,
        augmented_rows,
    )

    metadata = {
        "dataset_name": "laug900",
        "description": (
            "The exact 900 augmented "
            "rows already contained in "
            "the existing laug1800 "
            "training manifest, with "
            "the original Llanito rows "
            "removed by audio_path."
        ),
        "source_manifest": str(
            LAUG1800_MANIFEST
        ),
        "original_llanito_manifest": str(
            LLANITO_UNION_MANIFEST
        ),
        "fixed_test_manifest": str(
            LLANITO_TEST_MANIFEST
        ),
        "rows": len(
            augmented_rows
        ),
        "selection_rule": (
            "audio_path not present in "
            "original Llanito train+"
            "validation union"
        ),
        "augmentation_regenerated": False,
        "fixed_test_overlap": False,
        "output_manifest": str(
            OUTPUT_MANIFEST
        ),
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )

    print()
    print("=" * 80)
    print(
        "LAUG900 PREPARED"
    )
    print("=" * 80)

    print(
        f"Source rows: "
        f"{len(laug_rows)}"
    )

    print(
        f"Augmented rows: "
        f"{len(augmented_rows)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_MANIFEST}"
    )


if __name__ == "__main__":
    main()