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

from scripts.lora_l import config

from scripts.lora_l.train import (
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

    parser.add_argument(
        "--experiment",
        action="append",
        choices=(
            config.EXPERIMENT_NAMES
        ),
        help=(
            "Experiment to run. Can be supplied "
            "multiple times. If omitted, all "
            "Llanito LoRA experiments run."
        ),
    )

    stage_group = (
        parser.add_mutually_exclusive_group()
    )

    stage_group.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Train selected models without "
            "fixed-test evaluation."
        ),
    )

    stage_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Evaluate existing final models "
            "without training."
        ),
    )

    training_group = (
        parser.add_mutually_exclusive_group()
    )

    training_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume each selected model from "
            "its newest checkpoint."
        ),
    )

    training_group.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Delete existing artifacts for "
            "the selected models and retrain."
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


def find_latest_checkpoint(
    experiment: config.ExperimentConfig,
) -> Path | None:
    root = (
        experiment
        .checkpoint_directory
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

def training_artifacts_exist(
    experiment: config.ExperimentConfig,
) -> bool:
    if (
        experiment
        .metadata_path
        .is_file()
    ):
        return True

    if (
        experiment
        .model_directory
        .is_dir()
        and any(
            experiment
            .model_directory
            .iterdir()
        )
    ):
        return True

    if (
        experiment
        .checkpoint_directory
        .is_dir()
    ):
        for path in (
            experiment
            .checkpoint_directory
            .iterdir()
        ):
            if (
                path.is_dir()
                and checkpoint_number(
                    path
                )
                >= 0
            ):
                return True

    return False


def validate_fresh_training_start(
    experiment: config.ExperimentConfig,
) -> None:
    if not training_artifacts_exist(
        experiment
    ):
        return

    latest_checkpoint = (
        find_latest_checkpoint(
            experiment
        )
    )

    message = [
        (
            "Existing training artifacts "
            f"were found for "
            f"{experiment.model_name}."
        ),
        "",
    ]

    if latest_checkpoint is not None:
        message.extend(
            [
                "A resumable checkpoint exists:",
                str(
                    latest_checkpoint
                ),
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
                "want to retrain this model.",
            ]
        )

    raise RuntimeError(
        "\n".join(
            message
        )
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


def clear_existing_experiment(
    experiment: config.ExperimentConfig,
) -> None:
    if not is_main_process():
        return

    print()
    print(
        f"Restart requested for "
        f"{experiment.model_name}."
    )

    remove_path(
        experiment.model_directory
    )

    remove_path(
        experiment.checkpoint_directory
    )

    remove_path(
        experiment.metadata_path
    )

    remove_path(
        experiment.output_root
    )


# Selection

def select_experiments(
    requested_names: list[str] | None,
) -> list[
    config.ExperimentConfig
]:
    if not requested_names:
        return [
            config.EXPERIMENTS[
                name
            ]
            for name
            in config.EXPERIMENT_NAMES
        ]

    selected: list[
        config.ExperimentConfig
    ] = []

    seen: set[str] = set()

    for name in requested_names:
        if name in seen:
            continue

        seen.add(
            name
        )

        selected.append(
            config.EXPERIMENTS[
                name
            ]
        )

    return selected


# Fixed-test evaluation

def evaluate_final_model(
    experiment: config.ExperimentConfig,
) -> None:
    if not experiment.model_directory.exists():
        raise FileNotFoundError(
            "Final model directory "
            "does not exist:\n"
            f"{experiment.model_directory}"
        )

    if not experiment.metadata_path.exists():
        raise FileNotFoundError(
            "Final model metadata "
            "does not exist:\n"
            f"{experiment.metadata_path}"
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
        f"FIXED-TEST EVALUATION: "
        f"{experiment.model_name}"
    )
    print("=" * 80)

    print()
    print(
        f"Model root: "
        f"{experiment.model_root}"
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
            experiment.model_root
        ),
        torch_dtype=(
            torch_dtype
        ),
    )

    if (
        metadata.get(
            "model_name"
        )
        != experiment.model_name
    ):
        raise RuntimeError(
            "Model-name mismatch.\n"
            f"Expected: "
            f"{experiment.model_name}\n"
            f"Metadata: "
            f"{metadata.get('model_name')}"
        )

    evaluate_loaded_model(
        model_name=(
            experiment.model_name
        ),
        processor=processor,
        model=model,
        rows=rows,
        output_root=(
            experiment.output_root
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
            config
            .NORMALISATION_MAPPING_COLUMN
        ),
        normalisation_section_markers=(
            config
            .NORMALISATION_SECTION_MARKERS
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


# One experiment

def run_experiment(
    *,
    experiment: config.ExperimentConfig,
    run_training: bool,
    run_evaluation: bool,
    resume: bool,
    restart: bool,
) -> None:
    if is_main_process():
        print()
        print("#" * 80)
        print(
            f"EXPERIMENT: "
            f"{experiment.model_name}"
        )
        print("#" * 80)

        print()
        print(
            f"Placement: "
            f"{experiment.placement}"
        )

        print(
            f"Train: "
            f"{run_training}"
        )

        print(
            f"Evaluate fixed test: "
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

    if restart:
        clear_existing_experiment(
            experiment
        )

    distributed_barrier()

    resume_checkpoint: (
        Path | None
    ) = None

    if run_training:
        if resume:
            resume_checkpoint = (
                find_latest_checkpoint(
                    experiment
                )
            )

            if (
                resume_checkpoint
                is None
            ):
                raise FileNotFoundError(
                    "--resume was requested but "
                    "no checkpoint was found for:\n"
                    f"{experiment.model_name}\n"
                    "Checkpoint root:\n"
                    f"{experiment.checkpoint_directory}"
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
            validate_fresh_training_start(
                experiment
            )

        train_model(
            experiment=experiment,
            resume_from_checkpoint=(
                resume_checkpoint
            ),
        )

        distributed_barrier()

    if not run_evaluation:
        return

    # Fixed test happens only after
    # final training and only once.

    if is_main_process():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        evaluate_final_model(
            experiment
        )

    distributed_barrier()

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"{experiment.model_name} "
            f"PIPELINE COMPLETE"
        )
        print("=" * 80)

        print()
        print(
            "Permanent artifacts:"
        )

        print(
            f"  Model: "
            f"{experiment.model_directory}"
        )

        print(
            f"  Metadata: "
            f"{experiment.metadata_path}"
        )

        print(
            f"  Predictions: "
            f"{experiment.output_root / 'predictions.csv'}"
        )

        print(
            f"  Evaluation: "
            f"{experiment.output_root / 'evaluation.json'}"
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

    experiments = (
        select_experiments(
            args.experiment
        )
    )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            "FINAL LLANITO LoRA PIPELINE"
        )
        print("=" * 80)

        print()
        print(
            "Selected experiments:"
        )

        for experiment in experiments:
            print(
                f"  "
                f"{experiment.model_name}"
            )

        print()
        print(
            "Training data:"
        )

        print(
            f"  "
            f"{config.TRAIN_MANIFEST}"
        )

        print(
            f"  Expected recordings: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}"
        )

        print(
            "  Validation: none"
        )

        print()
        print(
            "Fixed test:"
        )

        print(
            f"  "
            f"{config.TEST_MANIFEST}"
        )

        print(
            f"  Expected recordings: "
            f"{config.EXPECTED_TEST_RECORDINGS}"
        )

        print()
        print(
            f"Train: "
            f"{run_training}"
        )

        print(
            f"Evaluate: "
            f"{run_evaluation}"
        )

        print(
            f"Seed: "
            f"{config.SEED}"
        )

    for experiment in experiments:
        run_experiment(
            experiment=experiment,
            run_training=(
                run_training
            ),
            run_evaluation=(
                run_evaluation
            ),
            resume=(
                args.resume
            ),
            restart=(
                args.restart
            ),
        )

    if is_main_process():
        print()
        print("#" * 80)
        print(
            "ALL SELECTED LLANITO LoRA "
            "EXPERIMENTS COMPLETE"
        )
        print("#" * 80)


if __name__ == "__main__":
    main()