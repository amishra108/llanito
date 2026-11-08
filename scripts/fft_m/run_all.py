from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.evaluate import (
    cleanup_model,
    evaluate_loaded_model,
)

from scripts.fft_m import config

from scripts.fft_m.train import (
    train_model,
)

from scripts.model_loader import (
    load_model,
)

from scripts.train import (
    distributed_barrier,
    is_main_process,
    load_manifest,
    validate_manifest_files_exist,
    validate_no_duplicate_audio,
)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    stage_group = (
        parser.add_mutually_exclusive_group()
    )

    stage_group.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Train and save the model, "
            "but do not evaluate."
        ),
    )

    stage_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip training and evaluate the "
            "existing final model."
        ),
    )

    training_group = (
        parser.add_mutually_exclusive_group()
    )

    training_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume interrupted training from "
            "the newest saved checkpoint."
        ),
    )

    training_group.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Delete fft_m model, checkpoints, "
            "metadata and outputs before "
            "training again from vanilla."
        ),
    )

    return parser.parse_args()


# Checkpoints

def checkpoint_number(
    path: Path,
) -> int:
    match = re.fullmatch(
        r"checkpoint-(\d+)",
        path.name,
    )

    if match is None:
        return -1

    return int(
        match.group(1)
    )


def find_latest_checkpoint() -> Path | None:
    root = (
        config.CHECKPOINT_DIRECTORY
    )

    if not root.exists():
        return None

    checkpoints = [
        path
        for path in root.iterdir()
        if (
            path.is_dir()
            and checkpoint_number(path)
            >= 0
        )
    ]

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=checkpoint_number,
    )


# Existing artifacts

def training_artifacts_exist() -> bool:
    if config.METADATA_PATH.is_file():
        return True

    if (
        config.MODEL_DIRECTORY.is_dir()
        and any(
            config.MODEL_DIRECTORY.iterdir()
        )
    ):
        return True

    if (
        config.CHECKPOINT_DIRECTORY.is_dir()
    ):
        for path in (
            config
            .CHECKPOINT_DIRECTORY
            .iterdir()
        ):
            if (
                path.is_dir()
                and checkpoint_number(path)
                >= 0
            ):
                return True

    return False


def validate_fresh_training_start() -> None:
    if not training_artifacts_exist():
        return

    latest_checkpoint = (
        find_latest_checkpoint()
    )

    message = [
        "Existing fft_m training artifacts "
        "were found.",
        "",
    ]

    if latest_checkpoint is not None:
        message.extend(
            [
                "A resumable checkpoint exists:",
                str(latest_checkpoint),
                "",
                "Use --resume to continue it, "
                "or --restart to discard it.",
            ]
        )

    else:
        message.extend(
            [
                "A model and/or metadata artifact "
                "already exists, but there is no "
                "resumable checkpoint.",
                "",
                "Use --restart if you deliberately "
                "want to train fft_m again.",
            ]
        )

    raise RuntimeError(
        "\n".join(message)
    )


# Restart

def remove_path(
    path: Path,
) -> None:
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(
            path
        )

    else:
        path.unlink()


def clear_existing_experiment() -> None:
    if not is_main_process():
        return

    print()
    print(
        "Restart requested."
    )

    print(
        "Removing existing fft_m "
        "artifacts..."
    )

    remove_path(
        config.MODEL_DIRECTORY
    )

    remove_path(
        config.CHECKPOINT_DIRECTORY
    )

    remove_path(
        config.METADATA_PATH
    )

    remove_path(
        config.OUTPUT_ROOT
    )


# Evaluation

def evaluate_final_model() -> None:
    if not config.MODEL_DIRECTORY.exists():
        raise FileNotFoundError(
            "Final model directory "
            "does not exist:\n"
            f"{config.MODEL_DIRECTORY}"
        )

    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            "Final model metadata "
            "does not exist:\n"
            f"{config.METADATA_PATH}"
        )

    rows = load_manifest(
        config.TEST_MANIFEST,
        required_columns=(
            "audio_path",
            "transcript",
            "task",
        ),
    )

    if (
        len(rows)
        != config.EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected Llanito fixed-test "
            "size.\n"
            f"Expected: "
            f"{config.EXPECTED_TEST_RECORDINGS}\n"
            f"Found: "
            f"{len(rows)}"
        )

    validate_no_duplicate_audio(
        rows,
        dataset_name=(
            "Llanito fixed test"
        ),
    )

    validate_manifest_files_exist(
        rows
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

        torch_dtype = (
            torch.float16
        )

    else:
        device = torch.device(
            "cpu"
        )

        torch_dtype = (
            torch.float32
        )

    print()
    print("=" * 80)
    print(
        "LLANITO FIXED-TEST EVALUATION: "
        "fft_m"
    )
    print("=" * 80)

    print()
    print(
        f"Model root: "
        f"{config.MODEL_ROOT}"
    )

    print(
        f"Test data: "
        f"{config.TEST_MANIFEST}"
    )

    print(
        f"Recordings: "
        f"{len(rows)}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print()
    print(
        "Loading final model..."
    )

    (
        model,
        processor,
        metadata,
    ) = load_model(
        model_root=(
            config.MODEL_ROOT
        ),
        torch_dtype=(
            torch_dtype
        ),
    )

    metadata_model_name = (
        metadata.get(
            "model_name"
        )
    )

    if (
        metadata_model_name
        != config.MODEL_NAME
    ):
        raise RuntimeError(
            "Model-name mismatch.\n"
            f"Expected: "
            f"{config.MODEL_NAME}\n"
            f"Metadata: "
            f"{metadata_model_name}"
        )

    evaluate_loaded_model(
        model_name=(
            config.MODEL_NAME
        ),
        processor=processor,
        model=model,
        rows=rows,
        output_root=(
            config.OUTPUT_ROOT
        ),
        normalisation_file=(
            config.NORMALISATION_FILE
        ),
        language=config.LANGUAGE,
        task=config.TASK,
        generation_max_length=(
            config.GENERATION_MAX_LENGTH
        ),
        generation_num_beams=(
            config.GENERATION_NUM_BEAMS
        ),
        device=device,
        audio_column="audio_path",
        transcript_column="transcript",
        normalisation_mapping_column=(
            config.NORMALISATION_MAPPING_COLUMN
        ),
        normalisation_section_markers=(
            config.NORMALISATION_SECTION_MARKERS
        ),
        generation_kwargs=None,
        dataset_name=(
            config.DATASET_NAME
        ),
    )

    cleanup_model(
        model,
        processor,
    )


# Main

def main() -> None:
    args = parse_args()

    run_training = (
        not args.evaluate_only
    )

    run_evaluation = (
        not args.train_only
    )

    if (
        args.resume
        and not run_training
    ):
        raise ValueError(
            "--resume cannot be combined "
            "with --evaluate-only."
        )

    if (
        args.restart
        and not run_training
    ):
        raise ValueError(
            "--restart cannot be combined "
            "with --evaluate-only."
        )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            "MIAMI FFT PIPELINE"
        )
        print("=" * 80)

        print()
        print(
            f"Train: "
            f"{run_training}"
        )

        print(
            f"Evaluate on Llanito: "
            f"{run_evaluation}"
        )

        if run_training:
            print(
                f"Training examples: "
                f"{config.EXPECTED_TRAINING_EXAMPLES}"
            )

            print(
                "Validation: none"
            )

            print(
                f"Seed: "
                f"{config.SEED}"
            )

            print(
                f"Learning rate: "
                f"{config.LEARNING_RATE}"
            )

            print(
                f"Warmup ratio: "
                f"{config.WARMUP_RATIO}"
            )

    if args.restart:
        clear_existing_experiment()

    distributed_barrier()

    resume_checkpoint: (
        Path | None
    ) = None

    if run_training:
        if args.resume:
            resume_checkpoint = (
                find_latest_checkpoint()
            )

            if (
                resume_checkpoint
                is None
            ):
                raise FileNotFoundError(
                    "--resume was requested "
                    "but no checkpoint was "
                    "found under:\n"
                    f"{config.CHECKPOINT_DIRECTORY}"
                )

            if is_main_process():
                print()
                print(
                    "Latest checkpoint:"
                )

                print(
                    f"  "
                    f"{resume_checkpoint}"
                )

        else:
            validate_fresh_training_start()

        train_model(
            resume_from_checkpoint=(
                resume_checkpoint
            ),
        )

        distributed_barrier()

    if not run_evaluation:
        return

    if not is_main_process():
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluate_final_model()

    print()
    print("=" * 80)
    print(
        "MIAMI FFT PIPELINE COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "Permanent artifacts:"
    )

    print(
        f"  "
        f"{config.MODEL_DIRECTORY}"
    )

    print(
        f"  "
        f"{config.METADATA_PATH}"
    )

    print(
        f"  "
        f"{config.OUTPUT_ROOT / 'predictions.csv'}"
    )

    print(
        f"  "
        f"{config.OUTPUT_ROOT / 'evaluation.json'}"
    )


if __name__ == "__main__":
    main()