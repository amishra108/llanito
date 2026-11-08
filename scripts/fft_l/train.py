from __future__ import annotations

import json
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


from scripts.fft_l import config

from scripts.model_loader import (
    load_model,
)

from scripts.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_classification,
    build_training_arguments,
    configure_whisper_generation,
    distributed_barrier,
    is_main_process,
    load_manifest,
    make_trainer,
    save_training_metadata,
    set_seed,
    update_training_metadata_after_training,
    validate_cuda_environment,
    validate_manifest_files_exist,
    validate_no_audio_overlap,
    validate_no_duplicate_audio,
)


# Data

def load_training_data() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    train_rows = load_manifest(
        config.TRAIN_MANIFEST,
        required_columns=(
            "audio_path",
            "transcript",
        ),
    )

    test_rows = load_manifest(
        config.TEST_MANIFEST,
        required_columns=(
            "audio_path",
            "transcript",
            "task",
        ),
    )

    if (
        len(train_rows)
        != config.EXPECTED_TRAINING_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected Llanito training-set "
            "size.\n"
            f"Expected: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}\n"
            f"Found:    {len(train_rows)}"
        )

    if (
        len(test_rows)
        != config.EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected Llanito fixed-test "
            "size.\n"
            f"Expected: "
            f"{config.EXPECTED_TEST_RECORDINGS}\n"
            f"Found:    {len(test_rows)}"
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name=(
            "llan_fft training"
        ),
    )

    validate_no_duplicate_audio(
        test_rows,
        dataset_name=(
            "Llanito fixed test"
        ),
    )

    validate_no_audio_overlap(
        train_rows,
        test_rows,
        first_name="training",
        second_name="fixed_test",
    )

    validate_manifest_files_exist(
        train_rows
    )

    validate_manifest_files_exist(
        test_rows
    )

    return (
        train_rows,
        test_rows,
    )


# Existing metadata

def load_existing_metadata() -> dict:
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            "Cannot resume because model "
            "metadata does not exist:\n"
            f"{config.METADATA_PATH}"
        )

    with config.METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        metadata = json.load(
            input_file
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "metadata.json must contain "
            "a JSON object."
        )

    if (
        metadata.get(
            "model_name"
        )
        != config.MODEL_NAME
    ):
        raise RuntimeError(
            "Existing metadata belongs "
            "to a different model."
        )

    return metadata


# Training

def train_model(
    *,
    resume_from_checkpoint: (
        Path | None
    ) = None,
) -> None:
    validate_cuda_environment(
        use_fp16=config.USE_FP16,
        use_bf16=config.USE_BF16,
    )

    set_seed(
        config.SEED
    )

    train_rows, _ = (
        load_training_data()
    )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            "LLANITO FINAL FULL FINE-TUNE"
        )
        print("=" * 80)

        print()
        print(
            f"Training examples: "
            f"{len(train_rows)}"
        )

        print(
            f"Base model: "
            f"{config.BASE_MODEL_ROOT}"
        )

        print(
            f"Final model: "
            f"{config.MODEL_DIRECTORY}"
        )

        print(
            f"Training seed: "
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

        print(
            f"Epochs: "
            f"{config.EPOCHS}"
        )

    # Load exact local vanilla
    # Whisper-medium baseline.

    (
        model,
        processor,
        base_metadata,
    ) = load_model(
        model_root=(
            config.BASE_MODEL_ROOT
        ),
        torch_dtype=None,
    )

    if (
        base_metadata.get(
            "model_type"
        )
        != "whisper_full"
    ):
        raise RuntimeError(
            "llan_fft requires a full "
            "Whisper base model."
        )

    # Full fine-tuning.

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = True

    # Configure tokenizer labels.

    processor.tokenizer.set_prefix_tokens(
        language=config.LANGUAGE,
        task=config.TASK,
        predict_timestamps=False,
    )

    # Disable generation cache during
    # gradient-checkpointed training.

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    # Training dataset.

    train_dataset = (
        WhisperManifestDataset(
            rows=train_rows,
            processor=processor,
            audio_column="audio_path",
            transcript_column="transcript",
        )
    )

    decoder_start_token_id = (
        model.config
        .decoder_start_token_id
    )

    if (
        decoder_start_token_id
        is None
    ):
        raise RuntimeError(
            "Whisper model has no "
            "decoder_start_token_id."
        )

    collator = (
        WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=(
                decoder_start_token_id
            ),
        )
    )

    # Final training has deliberately
    # no validation split.
    #
    # Hyperparameters were selected
    # externally with five-fold CV.
    #
    # Checkpoints are still saved once
    # per epoch so interrupted training
    # can be resumed.

    training_args = (
        build_training_arguments(
            checkpoint_directory=(
                config
                .CHECKPOINT_DIRECTORY
            ),
            seed=config.SEED,
            epochs=config.EPOCHS,
            learning_rate=(
                config.LEARNING_RATE
            ),
            weight_decay=(
                config.WEIGHT_DECAY
            ),
            warmup_steps=0,
            warmup_ratio=(
                config.WARMUP_RATIO
            ),
            train_batch_size=(
                config
                .TRAIN_BATCH_SIZE_PER_DEVICE
            ),
            gradient_accumulation_steps=(
                config
                .GRADIENT_ACCUMULATION_STEPS
            ),
            use_fp16=(
                config.USE_FP16
            ),
            use_bf16=(
                config.USE_BF16
            ),
            use_gradient_checkpointing=(
                config
                .USE_GRADIENT_CHECKPOINTING
            ),
            logging_steps=(
                config.LOGGING_STEPS
            ),
            save_total_limit=(
                config.SAVE_TOTAL_LIMIT
            ),
            dataloader_workers=(
                config.DATALOADER_WORKERS
            ),
            has_validation=False,
            evaluation_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            predict_with_generate=False,
            generation_max_length=(
                config
                .GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                config
                .GENERATION_NUM_BEAMS
            ),
            lr_scheduler_type=(
                config
                .LR_SCHEDULER_TYPE
            ),
            max_grad_norm=(
                config.MAX_GRAD_NORM
            ),
            optimizer=(
                config.OPTIMIZER
            ),
            save_only_model=False,
        )
    )

    classification = (
        build_classification(
            family=config.FAMILY,
            transfer_type=(
                config.TRANSFER_TYPE
            ),
            domain_category=(
                config.DOMAIN_CATEGORY
            ),
        )
    )

    # Metadata is written before a
    # fresh training run begins.
    #
    # When resuming, preserve the
    # existing metadata from the
    # interrupted run.

    if (
        resume_from_checkpoint
        is None
    ):
        metadata = (
            save_training_metadata(
                path=(
                    config.METADATA_PATH
                ),
                model_name=(
                    config.MODEL_NAME
                ),
                model_type=(
                    config.MODEL_TYPE
                ),
                model=model,
                training_args=(
                    training_args
                ),
                train_manifest=(
                    config.TRAIN_MANIFEST
                ),
                training_examples=(
                    len(train_rows)
                ),
                classification=(
                    classification
                ),
                base_model=(
                    config.BASE_MODEL
                ),
                validation_manifest=None,
                validation_examples=None,
                language=(
                    config.LANGUAGE
                ),
                task=config.TASK,
                hyperparameter_selection=(
                    config
                    .HYPERPARAMETER_SELECTION
                ),
                description=(
                    "Final Whisper-medium full "
                    "fine-tune on all 904 unique "
                    "non-test Llanito recordings "
                    "after five-fold Optuna "
                    "hyperparameter selection."
                ),
            )
        )

    else:
        metadata = (
            load_existing_metadata()
        )

    distributed_barrier()

    # Trainer.

    trainer = make_trainer(
        model=model,
        training_args=(
            training_args
        ),
        processor=processor,
        train_dataset=(
            train_dataset
        ),
        collator=collator,
        validation_dataset=None,
        compute_metrics=None,
        callbacks=(),
    )

    if is_main_process():
        print()

        if (
            resume_from_checkpoint
            is None
        ):
            print(
                "Starting training..."
            )

        else:
            print(
                "Resuming training from:"
            )

            print(
                f"  "
                f"{resume_from_checkpoint}"
            )

    train_result = (
        trainer.train(
            resume_from_checkpoint=(
                str(
                    resume_from_checkpoint
                )
                if resume_from_checkpoint
                is not None
                else None
            ),
        )
    )

    # Save final full Whisper model.

    trainer.save_model(
        str(
            config.MODEL_DIRECTORY
        )
    )

    if is_main_process():
        processor.save_pretrained(
            config.MODEL_DIRECTORY
        )

    distributed_barrier()

    # Mark metadata complete.

    if is_main_process():
        update_training_metadata_after_training(
            config.METADATA_PATH,
            existing_metadata=metadata,
            trainer=trainer,
            train_result=train_result,
            final_model_path=(
                config.MODEL_DIRECTORY
            ),
        )

    distributed_barrier()

    if is_main_process():
        print()
        print("=" * 80)
        print(
            "LLANITO FULL FINE-TUNE "
            "COMPLETE"
        )
        print("=" * 80)

        print()
        print(
            f"Model:    "
            f"{config.MODEL_DIRECTORY}"
        )

        print(
            f"Metadata: "
            f"{config.METADATA_PATH}"
        )


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()