from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.prepare import copy_cross_validation_manifests
from scripts_five_fold.fullLoRA.config import (
    FOLD_NUMBERS,
    OVERWRITE_PREPARED_MANIFESTS,
    PROCESSED_ROOT,
    SOURCE_PROCESSED_ROOT,
)


def main() -> None:
    print("=" * 80)
    print("PREPARING FULL ENCODER+DECODER LORA EXPERIMENT")
    print("=" * 80)
    print(f"Source manifests: {SOURCE_PROCESSED_ROOT}")
    print(f"Destination: {PROCESSED_ROOT}")

    summary = copy_cross_validation_manifests(
        source_root=SOURCE_PROCESSED_ROOT,
        destination_root=PROCESSED_ROOT,
        fold_numbers=FOLD_NUMBERS,
        overwrite=OVERWRITE_PREPARED_MANIFESTS,
    )

    print()
    print("PREPARATION COMPLETE")
    print(
        "The original frozen train/validation/test assignment was reused, "
        "so this experiment is directly comparable with tunewmcv."
    )

    for fold in summary["folds"]:
        print(
            f"Fold {fold['fold']}: "
            f"train={fold['train']['recordings']} | "
            f"validation={fold['validation']['recordings']}"
        )

    print(
        f"Fixed test recordings: "
        f"{summary['fixed_test']['recordings']}"
    )
    print(f"Outputs: {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
