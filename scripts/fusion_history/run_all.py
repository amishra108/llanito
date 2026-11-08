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


from scripts.fusion_history import config

from scripts.fusion_history.train import (
    build_history_model,
    load_history_state,
    train_model,
)

from scripts.evaluate import (
    cleanup_model,
    evaluate_loaded_model,
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
    )

    stage_group = (
        parser
        .add_mutually_exclusive_group()
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
        parser
        .add_mutually_exclusive_group()
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

    if not root.is_dir():
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


# Artifacts

def artifacts_exist(
    experiment: (
        config.ExperimentConfig
    ),
) -> bool:
    if (
        experiment
        .metadata_path
        .exists()
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

    return (
        find_latest_checkpoint(
            experiment
        )
        is not None
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


def clear_experiment(
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    if not is_main_process():
        return

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
    names: list[str] | None,
) -> list[
    config.ExperimentConfig
]:
    if not names:
        return [
            config.EXPERIMENTS[
                name
            ]
            for name in (
                config.EXPERIMENT_NAMES
            )
        ]

    selected = []

    seen = set()

    for name in names:
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


# Final model

def load_final_model(
    experiment: (
        config.ExperimentConfig
    ),
    *,
    torch_dtype: torch.dtype,
):
    (
        model,
        processor,
        architecture,
    ) = build_history_model(
        experiment,
        language=(
            config
            .EVALUATION_LANGUAGE
        ),
    )

    load_history_state(
        model=model,
        state_path=(
            experiment
            .model_directory
            / "history_state.pt"
        ),
    )

    model = model.to(
        dtype=torch_dtype
    )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    model.eval()

    return (
        model,
        processor,
        architecture,
    )


# Evaluation

def evaluate_final_model(
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
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
        != config
        .EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected Llanito "
            "fixed-test size."
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

    human_layers = [
        layer + 1
        for layer in (
            experiment
            .history_fusion_layers
        )
    ]

    print()
    print("=" * 80)
    print(
        "LLANITO FIXED-TEST "
        "EVALUATION: "
        f"{experiment.model_name}"
    )
    print("=" * 80)

    print(
        f"Experts: "
        f"{experiment.branch_domains}"
    )

    print(
        f"History length: "
        f"{experiment.history_length}"
    )

    print(
        f"History fusion layers: "
        f"{human_layers}"
    )

    print(
        f"Evaluation language: "
        f"{config.EVALUATION_LANGUAGE}"
    )

    (
        model,
        processor,
        architecture,
    ) = load_final_model(
        experiment,
        torch_dtype=torch_dtype,
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
        language=(
            config
            .EVALUATION_LANGUAGE
        ),
        task=config.TASK,
        generation_max_length=(
            config
            .GENERATION_MAX_LENGTH
        ),
        generation_num_beams=(
            config
            .GENERATION_NUM_BEAMS
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
    experiment: (
        config.ExperimentConfig
    ),
    run_training: bool,
    run_evaluation: bool,
    resume: bool,
    restart: bool,
) -> None:
    human_layers = [
        layer + 1
        for layer in (
            experiment
            .history_fusion_layers
        )
    ]

    if is_main_process():
        print()
        print("#" * 80)
        print(
            f"EXPERIMENT: "
            f"{experiment.model_name}"
        )
        print("#" * 80)

        print(
            f"Experts: "
            f"{experiment.branch_domains}"
        )

        print(
            f"Target: "
            f"{experiment.target_domain}"
        )

        print(
            f"Training language: "
            f"{experiment.training_language}"
        )

        print(
            f"History length: "
            f"{experiment.history_length}"
        )

        print(
            f"Fusion layers: "
            f"{human_layers}"
        )

        print(
            "Expected trainable "
            "parameters: "
            f"{experiment.expected_trainable_parameters:,}"
        )

        print(
            f"Training manifest: "
            f"{config.get_train_manifest(experiment)}"
        )

        print(
            f"Learning rate: "
            f"{config.LEARNING_RATE}"
        )

        print(
            f"Warmup steps: "
            f"{config.WARMUP_STEPS}"
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
        clear_experiment(
            experiment
        )

    distributed_barrier()

    checkpoint = None

    if run_training:
        if resume:
            checkpoint = (
                find_latest_checkpoint(
                    experiment
                )
            )

            if checkpoint is None:
                raise FileNotFoundError(
                    "No checkpoint found "
                    "for --resume."
                )

        elif not restart:
            if artifacts_exist(
                experiment
            ):
                raise RuntimeError(
                    "Existing artifacts found "
                    f"for "
                    f"{experiment.model_name}.\n"
                    "Use --resume or --restart."
                )

        train_model(
            experiment=experiment,
            resume_from_checkpoint=(
                checkpoint
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
            "RESADAPT HISTORY-FUSION "
            "PIPELINE"
        )
        print("=" * 80)

        print()
        print(
            "Whisper weights: frozen"
        )

        print(
            "Expert adapter weights: frozen"
        )

        print(
            "History compressor / GRU / "
            "output modules: trainable"
        )

        print()
        print(
            f"Learning rate: "
            f"{config.LEARNING_RATE}"
        )

        print(
            f"Warmup steps: "
            f"{config.WARMUP_STEPS}"
        )

        print(
            f"Warmup ratio: "
            f"{config.WARMUP_RATIO}"
        )

        print(
            f"Epochs: "
            f"{config.EPOCHS}"
        )

        print()
        print(
            "Final evaluation always:"
        )

        print(
            f"  {config.TEST_MANIFEST}"
        )

        print(
            f"  "
            f"{config.EXPECTED_TEST_RECORDINGS} "
            "Llanito recordings"
        )

        print()
        print(
            "Selected experiments:"
        )

        for experiment in (
            experiments
        ):
            print(
                f"  "
                f"{experiment.model_name}"
                f" | HIST"
                f"{experiment.history_length}"
                f" | "
                f"{experiment.number_of_fusion_layers} "
                "fusion points"
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
            resume=args.resume,
            restart=args.restart,
        )


if __name__ == "__main__":
    main()