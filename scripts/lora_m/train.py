from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    get_peft_model,
)


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.lora_m import config

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
    parameter_summary,
    save_json,
    save_training_metadata,
    set_seed,
    update_training_metadata_after_training,
    validate_cuda_environment,
    validate_manifest_files_exist,
    validate_no_audio_overlap,
    validate_no_duplicate_audio,
    write_csv,
)


# Existing metadata

def load_existing_metadata(
    experiment: config.ExperimentConfig,
) -> dict[str, Any]:
    if not experiment.metadata_path.exists():
        raise FileNotFoundError(
            "Cannot resume because model "
            "metadata does not exist:\n"
            f"{experiment.metadata_path}"
        )

    with experiment.metadata_path.open(
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
        != experiment.model_name
    ):
        raise RuntimeError(
            "Existing metadata belongs "
            "to a different model."
        )

    return metadata


# Miami union

def ensure_training_union() -> None:
    if config.TRAIN_MANIFEST.exists():
        return

    if is_main_process():
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
            "Created Miami train+validation "
            "union manifest."
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

    distributed_barrier()


# Data

def load_training_data() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    ensure_training_union()

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
        ),
    )

    if (
        len(train_rows)
        != config.EXPECTED_TRAINING_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected Miami final-training "
            "set size.\n"
            f"Expected: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}\n"
            f"Found: "
            f"{len(train_rows)}"
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name=(
            "Miami final training"
        ),
    )

    validate_no_duplicate_audio(
        test_rows,
        dataset_name=(
            "Miami fixed test"
        ),
    )

    validate_no_audio_overlap(
        train_rows,
        test_rows,
        first_name="final_training",
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


# LoRA targets

def resolve_lora_targets(
    model: torch.nn.Module,
    placement: config.LoraPlacement,
) -> list[str]:
    encoder_attention_suffixes = (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.out_proj",
    )

    encoder_full_suffixes = (
        *encoder_attention_suffixes,
        ".fc1",
        ".fc2",
    )

    decoder_full_suffixes = (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.out_proj",
        ".encoder_attn.q_proj",
        ".encoder_attn.k_proj",
        ".encoder_attn.v_proj",
        ".encoder_attn.out_proj",
        ".fc1",
        ".fc2",
    )

    targets: list[str] = []

    for (
        name,
        module,
    ) in model.named_modules():
        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            continue

        is_encoder = name.startswith(
            "model.encoder.layers."
        )

        is_decoder = name.startswith(
            "model.decoder.layers."
        )

        if placement == "encattn":
            if (
                is_encoder
                and name.endswith(
                    encoder_attention_suffixes
                )
            ):
                targets.append(
                    name
                )

        elif placement == "enc":
            if (
                is_encoder
                and name.endswith(
                    encoder_full_suffixes
                )
            ):
                targets.append(
                    name
                )

        elif placement == "dec":
            if (
                is_decoder
                and name.endswith(
                    decoder_full_suffixes
                )
            ):
                targets.append(
                    name
                )

        elif placement == "encdec":
            if (
                (
                    is_encoder
                    and name.endswith(
                        encoder_full_suffixes
                    )
                )
                or (
                    is_decoder
                    and name.endswith(
                        decoder_full_suffixes
                    )
                )
            ):
                targets.append(
                    name
                )

        else:
            raise ValueError(
                "Unsupported LoRA placement: "
                f"{placement!r}"
            )

    targets = sorted(
        set(
            targets
        )
    )

    if not targets:
        raise RuntimeError(
            "No LoRA target modules "
            "were found."
        )

    return targets


# LoRA validation

def validate_lora_scope(
    *,
    model: torch.nn.Module,
    placement: config.LoraPlacement,
) -> None:
    trainable_names = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError(
            "LoRA model contains zero "
            "trainable parameters."
        )

    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected non-LoRA "
            "trainable parameters:\n"
            + "\n".join(
                unexpected[:50]
            )
        )

    for name in trainable_names:
        lower_name = (
            name.casefold()
        )

        is_encoder = (
            ".encoder." in lower_name
        )

        is_decoder = (
            ".decoder." in lower_name
        )

        is_attention = any(
            value in lower_name
            for value in (
                ".q_proj.",
                ".k_proj.",
                ".v_proj.",
                ".out_proj.",
            )
        )

        is_feed_forward = any(
            value in lower_name
            for value in (
                ".fc1.",
                ".fc2.",
            )
        )

        if placement == "encattn":
            valid = (
                is_encoder
                and not is_decoder
                and is_attention
                and not is_feed_forward
            )

        elif placement == "enc":
            valid = (
                is_encoder
                and not is_decoder
                and (
                    is_attention
                    or is_feed_forward
                )
            )

        elif placement == "dec":
            valid = (
                is_decoder
                and not is_encoder
                and (
                    is_attention
                    or is_feed_forward
                )
            )

        elif placement == "encdec":
            valid = (
                (
                    is_encoder
                    or is_decoder
                )
                and (
                    is_attention
                    or is_feed_forward
                )
            )

        else:
            valid = False

        if not valid:
            raise RuntimeError(
                "LoRA parameter falls outside "
                "the requested scope.\n"
                f"Placement: {placement}\n"
                f"Parameter: {name}"
            )


# LoRA construction

def build_lora_model(
    *,
    base_model: torch.nn.Module,
    experiment: config.ExperimentConfig,
) -> tuple[
    torch.nn.Module,
    list[str],
]:
    target_modules = (
        resolve_lora_targets(
            base_model,
            experiment.placement,
        )
    )

    if is_main_process():
        print()
        print(
            "Resolved LoRA target modules:"
        )

        print(
            f"  Count: "
            f"{len(target_modules)}"
        )

    lora_configuration = (
        LoraConfig(
            inference_mode=False,
            r=config.LORA_RANK,
            lora_alpha=(
                config.LORA_ALPHA
            ),
            lora_dropout=(
                config.LORA_DROPOUT
            ),
            bias=config.LORA_BIAS,
            target_modules=(
                target_modules
            ),
            task_type=None,
            use_dora=(
                config.USE_DORA
            ),
            use_rslora=(
                config.USE_RSLORA
            ),
        )
    )

    model = get_peft_model(
        base_model,
        lora_configuration,
    )

    validate_lora_scope(
        model=model,
        placement=(
            experiment.placement
        ),
    )

    return (
        model,
        target_modules,
    )


# Architecture metadata

def build_architecture_metadata(
    *,
    experiment: config.ExperimentConfig,
    target_modules: list[str],
) -> dict[str, Any]:
    return {
        "adaptation": "lora",
        "placement": (
            experiment.placement
        ),
        "rank": config.LORA_RANK,
        "alpha": config.LORA_ALPHA,
        "dropout": (
            config.LORA_DROPOUT
        ),
        "bias": config.LORA_BIAS,
        "use_dora": (
            config.USE_DORA
        ),
        "use_rslora": (
            config.USE_RSLORA
        ),
        "target_modules": (
            target_modules
        ),
        "base_model_name_or_path": (
            config.BASE_MODEL
        ),
        "base_model_root": str(
            config.BASE_MODEL_ROOT
        ),
    }


# Training

def train_model(
    *,
    experiment: config.ExperimentConfig,
    resume_from_checkpoint: (
        Path | None
    ) = None,
) -> None:
    validate_cuda_environment(
        use_fp16=(
            config.USE_FP16
        ),
        use_bf16=(
            config.USE_BF16
        ),
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
            f"FINAL MIAMI LoRA: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

        print()
        print(
            f"Placement: "
            f"{experiment.placement}"
        )

        print(
            f"Training examples: "
            f"{len(train_rows)}"
        )

        print(
            "Validation: none"
        )

        print(
            f"Base model: "
            f"{config.BASE_MODEL_ROOT}"
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
            f"Warmup ratio: "
            f"{config.WARMUP_RATIO}"
        )

        print(
            f"LoRA rank: "
            f"{config.LORA_RANK}"
        )

        print(
            f"LoRA alpha: "
            f"{config.LORA_ALPHA}"
        )

        print(
            f"LoRA dropout: "
            f"{config.LORA_DROPOUT}"
        )

        print(
            f"Seed: "
            f"{config.SEED}"
        )

    # Plain Miami LoRA always
    # starts from vanilla Whisper.

    (
        base_model,
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
            "Plain Miami LoRA requires "
            "a full Whisper base model."
        )

    (
        model,
        target_modules,
    ) = build_lora_model(
        base_model=base_model,
        experiment=experiment,
    )

    counts = parameter_summary(
        model
    )

    if is_main_process():
        print()
        print(
            "Parameter counts:"
        )

        print(
            f"  Total: "
            f"{counts['total_parameters']:,}"
        )

        print(
            f"  Trainable: "
            f"{counts['trainable_parameters']:,}"
        )

        print(
            f"  Frozen: "
            f"{counts['frozen_parameters']:,}"
        )

        print(
            f"  Trainable %: "
            f"{counts['trainable_percent']:.6f}%"
        )

    processor.tokenizer.set_prefix_tokens(
        language=config.LANGUAGE,
        task=config.TASK,
        predict_timestamps=False,
    )

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

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
            "Whisper has no "
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

    # Final union training.
    # No validation / model selection.

    training_args = (
        build_training_arguments(
            checkpoint_directory=(
                experiment
                .checkpoint_directory
            ),
            seed=config.SEED,
            epochs=config.EPOCHS,
            learning_rate=(
                config.LEARNING_RATE
            ),
            weight_decay=(
                config.WEIGHT_DECAY
            ),
            warmup_steps=(
                config.WARMUP_STEPS
            ),
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

    architecture = (
        build_architecture_metadata(
            experiment=experiment,
            target_modules=(
                target_modules
            ),
        )
    )

    if (
        resume_from_checkpoint
        is None
    ):
        metadata = (
            save_training_metadata(
                path=(
                    experiment
                    .metadata_path
                ),
                model_name=(
                    experiment.model_name
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
                validation_manifest=None,
                validation_examples=None,
                classification=(
                    classification
                ),
                base_model=(
                    config.BASE_MODEL
                ),
                language=(
                    config.LANGUAGE
                ),
                task=config.TASK,
                hyperparameter_selection=(
                    config
                    .HYPERPARAMETER_SELECTION
                ),
                architecture=(
                    architecture
                ),
                description=(
                    experiment.description
                ),
            )
        )

        if is_main_process():
            metadata[
                "training_data_provenance"
            ] = {
                "original_train_manifest": str(
                    config
                    .ORIGINAL_TRAIN_MANIFEST
                ),
                "original_validation_manifest": str(
                    config
                    .ORIGINAL_VALIDATION_MANIFEST
                ),
                "final_union_manifest": str(
                    config.TRAIN_MANIFEST
                ),
                "test_manifest": str(
                    config.TEST_MANIFEST
                ),
            }

            save_json(
                experiment.metadata_path,
                metadata,
            )

    else:
        metadata = (
            load_existing_metadata(
                experiment
            )
        )

    distributed_barrier()

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
                "Starting final training..."
            )

        else:
            print(
                "Resuming from:"
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

    # PEFT Trainer saves the adapter
    # under models/<model_name>/model.

    trainer.save_model(
        str(
            experiment.model_directory
        )
    )

    if is_main_process():
        processor.save_pretrained(
            experiment.model_directory
        )

    distributed_barrier()

    if is_main_process():
        update_training_metadata_after_training(
            experiment.metadata_path,
            existing_metadata=metadata,
            trainer=trainer,
            train_result=train_result,
            final_model_path=(
                experiment
                .model_directory
            ),
        )

    distributed_barrier()

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"{experiment.model_name} "
            f"TRAINING COMPLETE"
        )
        print("=" * 80)

        print()
        print(
            f"Model: "
            f"{experiment.model_directory}"
        )

        print(
            f"Metadata: "
            f"{experiment.metadata_path}"
        )


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        choices=(
            config.EXPERIMENT_NAMES
        ),
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_model(
        experiment=(
            config.EXPERIMENTS[
                args.experiment
            ]
        ),
        resume_from_checkpoint=(
            args.resume_from_checkpoint
        ),
    )


if __name__ == "__main__":
    main()