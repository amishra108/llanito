from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    WhisperProcessor,
)


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.model_loader import (
    load_model,
)

from scripts.resadapt_m import config

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


# Residual adapter

class ResidualAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(
            hidden_size,
            bottleneck_size,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.up = nn.Linear(
            bottleneck_size,
            hidden_size,
        )

        nn.init.zeros_(
            self.up.weight
        )

        nn.init.zeros_(
            self.up.bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        correction = self.down(
            hidden_states
        )

        correction = self.activation(
            correction
        )

        correction = self.dropout(
            correction
        )

        correction = self.up(
            correction
        )

        return (
            hidden_states
            + correction
        )


class AdapterWrappedLayer(nn.Module):
    def __init__(
        self,
        *,
        base_layer: nn.Module,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.base_layer = (
            base_layer
        )

        self.residual_adapter = (
            ResidualAdapter(
                hidden_size=(
                    hidden_size
                ),
                bottleneck_size=(
                    bottleneck_size
                ),
                dropout=dropout,
            )
        )

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        output = self.base_layer(
            *args,
            **kwargs,
        )

        if isinstance(
            output,
            torch.Tensor,
        ):
            return (
                self.residual_adapter(
                    output
                )
            )

        if isinstance(
            output,
            tuple,
        ):
            if not output:
                raise RuntimeError(
                    "Whisper layer returned "
                    "an empty tuple."
                )

            adapted = (
                self.residual_adapter(
                    output[0]
                )
            )

            return (
                adapted,
                *output[1:],
            )

        if hasattr(
            output,
            "last_hidden_state",
        ):
            output.last_hidden_state = (
                self.residual_adapter(
                    output.last_hidden_state
                )
            )

            return output

        raise TypeError(
            "Unexpected Whisper layer "
            "output type: "
            f"{type(output)!r}"
        )


# Existing metadata

def load_existing_metadata(
    experiment: (
        config.ExperimentConfig
    ),
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

    if not is_main_process():
        distributed_barrier()
        return

    print()
    print(
        "Miami train+validation union "
        "does not exist."
    )

    print(
        "Creating it now..."
    )

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

    print(
        f"Union saved: "
        f"{config.TRAIN_MANIFEST}"
    )

    print(
        f"Original train: "
        f"{len(train_rows)}"
    )

    print(
        f"Original validation: "
        f"{len(validation_rows)}"
    )

    print(
        f"Final training set: "
        f"{len(union_rows)}"
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


# Adapter construction

def wrap_selected_layers(
    *,
    layers: nn.ModuleList,
    selected_layers: tuple[int, ...],
    hidden_size: int,
) -> nn.ModuleList:
    selected = set(
        selected_layers
    )

    wrapped: list[
        nn.Module
    ] = []

    for (
        layer_index,
        layer,
    ) in enumerate(
        layers
    ):
        if layer_index in selected:
            wrapped.append(
                AdapterWrappedLayer(
                    base_layer=layer,
                    hidden_size=(
                        hidden_size
                    ),
                    bottleneck_size=(
                        config
                        .ADAPTER_BOTTLENECK
                    ),
                    dropout=(
                        config
                        .ADAPTER_DROPOUT
                    ),
                )
            )

        else:
            wrapped.append(
                layer
            )

    return nn.ModuleList(
        wrapped
    )


def adapter_modules_for_state(
    *,
    model: nn.Module,
    experiment: (
        config.ExperimentConfig
    ),
) -> list[
    tuple[
        str,
        int,
        ResidualAdapter,
    ]
]:
    modules: list[
        tuple[
            str,
            int,
            ResidualAdapter,
        ]
    ] = []

    if experiment.adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        for layer_index in (
            experiment.adapter_layers
        ):
            layer = (
                model
                .model
                .encoder
                .layers[
                    layer_index
                ]
            )

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise TypeError(
                    "Expected encoder adapter "
                    f"at layer {layer_index}."
                )

            modules.append(
                (
                    "encoder",
                    layer_index,
                    layer.residual_adapter,
                )
            )

    if experiment.adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        for layer_index in (
            experiment.adapter_layers
        ):
            layer = (
                model
                .model
                .decoder
                .layers[
                    layer_index
                ]
            )

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise TypeError(
                    "Expected decoder adapter "
                    f"at layer {layer_index}."
                )

            modules.append(
                (
                    "decoder",
                    layer_index,
                    layer.residual_adapter,
                )
            )

    return modules


def verify_trainable_parameters(
    model: nn.Module,
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
            "No residual-adapter "
            "parameters are trainable."
        )

    unexpected = [
        name
        for name
        in trainable_names
        if (
            ".residual_adapter."
            not in name
        )
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected non-adapter "
            "parameters are trainable:\n"
            + "\n".join(
                unexpected[:50]
            )
        )


def build_residual_adapter_model(
    *,
    base_model: nn.Module,
    experiment: (
        config.ExperimentConfig
    ),
) -> nn.Module:
    # Plain Miami ResAdapt always starts
    # from vanilla Whisper-medium.

    for parameter in (
        base_model.parameters()
    ):
        parameter.requires_grad = False

    hidden_size = int(
        base_model.config.d_model
    )

    if experiment.adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        base_model.model.encoder.layers = (
            wrap_selected_layers(
                layers=(
                    base_model
                    .model
                    .encoder
                    .layers
                ),
                selected_layers=(
                    experiment
                    .adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
            )
        )

    if experiment.adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        base_model.model.decoder.layers = (
            wrap_selected_layers(
                layers=(
                    base_model
                    .model
                    .decoder
                    .layers
                ),
                selected_layers=(
                    experiment
                    .adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
            )
        )

    for (
        _,
        _,
        adapter,
    ) in adapter_modules_for_state(
        model=base_model,
        experiment=experiment,
    ):
        for parameter in (
            adapter.parameters()
        ):
            parameter.requires_grad = True

    verify_trainable_parameters(
        base_model
    )

    return base_model


# Adapter persistence

def residual_adapter_state_dict(
    *,
    model: nn.Module,
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[
    str,
    torch.Tensor,
]:
    state: dict[
        str,
        torch.Tensor,
    ] = {}

    for (
        module_name,
        layer_index,
        adapter,
    ) in adapter_modules_for_state(
        model=model,
        experiment=experiment,
    ):
        for (
            key,
            value,
        ) in adapter.state_dict().items():
            state[
                f"{module_name}."
                f"layer_{layer_index}."
                f"{key}"
            ] = (
                value
                .detach()
                .cpu()
                .clone()
            )

    if not state:
        raise RuntimeError(
            "Refusing to save an empty "
            "residual adapter."
        )

    return state


def save_residual_adapter(
    *,
    model: nn.Module,
    processor: WhisperProcessor,
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    experiment.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        residual_adapter_state_dict(
            model=model,
            experiment=experiment,
        ),
        (
            experiment.model_directory
            / "adapter_state.pt"
        ),
    )

    adapter_config = {
        "format_version": 2,
        "experiment": (
            experiment.model_name
        ),
        "base_model": (
            config.BASE_MODEL
        ),
        "base_model_root": str(
            config.BASE_MODEL_ROOT
        ),
        "adapter_module": (
            experiment.adapter_module
        ),
        "adapter_position": (
            config.ADAPTER_POSITION
        ),
        "adapter_layers": list(
            experiment.adapter_layers
        ),
        "bottleneck": (
            config.ADAPTER_BOTTLENECK
        ),
        "dropout": (
            config.ADAPTER_DROPOUT
        ),
    }

    save_json(
        (
            experiment.model_directory
            / "adapter_config.json"
        ),
        adapter_config,
    )

    processor.save_pretrained(
        experiment.model_directory
    )


# Architecture metadata

def build_architecture_metadata(
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[str, Any]:
    return {
        "adaptation": (
            "residual_adapter"
        ),
        "placement": (
            experiment.placement
        ),
        "module": (
            experiment.adapter_module
        ),
        "position": (
            config.ADAPTER_POSITION
        ),
        "layers": list(
            experiment.adapter_layers
        ),
        "bottleneck": (
            config.ADAPTER_BOTTLENECK
        ),
        "dropout": (
            config.ADAPTER_DROPOUT
        ),
        "base_model_name_or_path": (
            config.BASE_MODEL
        ),
        "base_model_root": str(
            config.BASE_MODEL_ROOT
        ),
        "source_adapter": None,
        "source_name": None,
    }


# Training

def train_model(
    *,
    experiment: (
        config.ExperimentConfig
    ),
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
            f"FINAL MIAMI RESADAPT: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

        print()
        print(
            f"Adapter module: "
            f"{experiment.adapter_module}"
        )

        print(
            f"Adapter layers: "
            f"{list(experiment.adapter_layers)}"
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
            f"Adapter bottleneck: "
            f"{config.ADAPTER_BOTTLENECK}"
        )

        print(
            f"Adapter dropout: "
            f"{config.ADAPTER_DROPOUT}"
        )

        print(
            f"Seed: "
            f"{config.SEED}"
        )

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
            "Plain Miami ResAdapt requires "
            "a full vanilla Whisper base."
        )

    model = (
        build_residual_adapter_model(
            base_model=base_model,
            experiment=experiment,
        )
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

    if decoder_start_token_id is None:
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
            experiment
        )
    )

    if resume_from_checkpoint is None:
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

        if resume_from_checkpoint is None:
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

    train_result = trainer.train(
        resume_from_checkpoint=(
            str(
                resume_from_checkpoint
            )
            if resume_from_checkpoint
            is not None
            else None
        ),
    )

    if is_main_process():
        save_residual_adapter(
            model=trainer.model,
            processor=processor,
            experiment=experiment,
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