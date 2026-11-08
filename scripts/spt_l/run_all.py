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

from scripts.spt_l import config

from scripts.spt_l.train import (
    load_saved_spt_model,
    train_model,
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
            "Experiment to run. May be "
            "supplied multiple times. "
            "If omitted, all Llanito SPT "
            "experiments run."
        ),
    )

    stage_group = (
        parser.add_mutually_exclusive_group()
    )

    stage_group.add_argument(
        "--train-only",
        action="store_true",
    )

    stage_group.add_argument(
        "--evaluate-only",
        action="store_true",
    )

    training_group = (
        parser.add_mutually_exclusive_group()
    )

    training_group.add_argument(
        "--resume",
        action="store_true",
    )

    training_group.add_argument(
        "--restart",
        action="store_true",
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
    experiment: (
        config.ExperimentConfig
    ),
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
            and checkpoint_number(
                path
            )
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
    experiment: (
        config.ExperimentConfig
    ),
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
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    if not training_artifacts_exist(
        experiment
    ):
        return

    latest = (
        find_latest_checkpoint(
            experiment
        )
    )

    message = [
        (
            "Existing training artifacts "
            "were found for "
            f"{experiment.model_name}."
        ),
        "",
    ]

    if latest is not None:
        message.extend(
            [
                "Resumable checkpoint:",
                str(latest),
                "",
                "Use --resume or --restart.",
            ]
        )

    else:
        message.extend(
            [
                "A final model and/or "
                "metadata already exists.",
                "",
                "Use --restart to retrain.",
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
    experiment: (
        config.ExperimentConfig
    ),
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
    requested_names: (
        list[str] | None
    ),
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


# Evaluation

def evaluate_final_model(
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    if not (
        experiment
        .model_directory
        .exists()
    ):
        raise FileNotFoundError(
            "Final SPT model directory "
            "does not exist:\n"
            f"{experiment.model_directory}"
        )

    if not (
        experiment
        .metadata_path
        .exists()
    ):
        raise FileNotFoundError(
            "Final metadata does not exist:\n"
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
            torch.float32
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
        f"{experiment.model_name}"
    )
    print("=" * 80)

    print()
    print(
        f"Prompt length: "
        f"{experiment.prompt_length}"
    )

    print(
        f"Test data: "
        f"{config.TEST_MANIFEST}"
    )

    print(
        f"Recordings: "
        f"{len(rows)}"
    )

    model, processor = (
        load_saved_spt_model(
            model_directory=(
                experiment
                .model_directory
            ),
            torch_dtype=(
                torch_dtype
            ),
        )
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


# One experiment

def run_experiment(
    *,
    experiment: (
        config.ExperimentConfig
    ),
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
            f"Prompt length: "
            f"{experiment.prompt_length}"
        )

        print(
            f"Train: "
            f"{run_training}"
        )

        print(
            f"Evaluate: "
            f"{run_evaluation}"
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
                    "No resumable checkpoint "
                    "was found for:\n"
                    f"{experiment.model_name}"
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

    if is_main_process():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        evaluate_final_model(
            experiment
        )

    distributed_barrier()


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
            "--resume cannot be used "
            "with --evaluate-only."
        )

    if (
        args.restart
        and not run_training
    ):
        raise ValueError(
            "--restart cannot be used "
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
            "FINAL LLANITO SPT PIPELINE"
        )
        print("=" * 80)

        print()
        print(
            "Experiments:"
        )

        for experiment in experiments:
            print(
                f"  "
                f"{experiment.model_name} "
                f"(p={experiment.prompt_length})"
            )

        print()
        print(
            f"Training union: "
            f"{config.TRAIN_MANIFEST}"
        )

        print(
            f"Training examples: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}"
        )

        print(
            f"Fixed test: "
            f"{config.TEST_MANIFEST}"
        )

        print()
        print(
            "Validation during final "
            "training: none"
        )

        print(
            f"Learning rate: "
            f"{config.LEARNING_RATE}"
        )

        print(
            f"Weight decay: "
            f"{config.WEIGHT_DECAY}"
        )

        print(
            f"FP16: "
            f"{config.USE_FP16}"
        )

        print(
            f"BF16: "
            f"{config.USE_BF16}"
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
            "ALL SELECTED LLANITO SPT "
            "EXPERIMENTS COMPLETE"
        )
        print("#" * 80)


if __name__ == "__main__":
    main()