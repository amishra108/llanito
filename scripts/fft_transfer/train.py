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
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
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


from scripts.fft_transfer import config

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
)


# Recipe helpers

def resolve_warmup(
    target_config: Any,
) -> tuple[int, float]:
    warmup_steps = int(
        getattr(
            target_config,
            "WARMUP_STEPS",
            0,
        )
    )

    warmup_ratio = float(
        getattr(
            target_config,
            "WARMUP_RATIO",
            0.0,
        )
    )

    if warmup_steps < 0:
        raise ValueError(
            "WARMUP_STEPS cannot be negative. "
            f"Got {warmup_steps} from "
            f"{target_config.__name__}."
        )

    if not (
        0.0
        <= warmup_ratio
        <= 1.0
    ):
        raise ValueError(
            "WARMUP_RATIO must lie between "
            "0.0 and 1.0. "
            f"Got {warmup_ratio} from "
            f"{target_config.__name__}."
        )

    if (
        warmup_steps > 0
        and warmup_ratio > 0.0
    ):
        raise ValueError(
            "Target recipe specifies both "
            "WARMUP_STEPS > 0 and "
            "WARMUP_RATIO > 0. "
            "Use exactly one warmup convention.\n"
            f"Recipe: {target_config.__name__}\n"
            f"WARMUP_STEPS: {warmup_steps}\n"
            f"WARMUP_RATIO: {warmup_ratio}"
        )

    return (
        warmup_steps,
        warmup_ratio,
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
        adapter: ResidualAdapter,
    ) -> None:
        super().__init__()

        self.base_layer = (
            base_layer
        )

        self.residual_adapter = (
            adapter
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

            return (
                self.residual_adapter(
                    output[0]
                ),
                *output[1:],
            )

        raise TypeError(
            "Unexpected Whisper layer "
            "output type: "
            f"{type(output)!r}"
        )


# Source validation

def validate_source_model(
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    if not (
        experiment
        .source_model_directory
        .is_dir()
    ):
        raise FileNotFoundError(
            "Source FFT model directory "
            "does not exist:\n"
            f"{experiment.source_model_directory}"
        )

    source_metadata_path = (
        experiment
        .source_model_root
        / "metadata.json"
    )

    if not source_metadata_path.exists():
        raise FileNotFoundError(
            "Source FFT metadata does "
            "not exist:\n"
            f"{source_metadata_path}"
        )

    with source_metadata_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        source_metadata = json.load(
            input_file
        )

    if (
        source_metadata.get(
            "model_type"
        )
        != "whisper_full"
    ):
        raise RuntimeError(
            "FFT transfer source must be "
            "a full Whisper model.\n"
            f"Source: "
            f"{experiment.source_model_name}\n"
            f"model_type: "
            f"{source_metadata.get('model_type')}"
        )


# LoRA targets

def collect_lora_targets(
    *,
    model: nn.Module,
    placement: str,
) -> list[str]:
    targets: list[str] = []

    projection_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    }

    feed_forward_names = {
        "fc1",
        "fc2",
    }

    for (
        module_name,
        _,
    ) in model.named_modules():
        leaf_name = (
            module_name
            .rsplit(
                ".",
                1,
            )[-1]
        )

        is_encoder = (
            module_name.startswith(
                "model.encoder.layers."
            )
        )

        is_decoder = (
            module_name.startswith(
                "model.decoder.layers."
            )
        )

        is_encoder_self_attention = (
            is_encoder
            and ".self_attn." in module_name
            and leaf_name
            in projection_names
        )

        is_encoder_ffn = (
            is_encoder
            and leaf_name
            in feed_forward_names
        )

        is_decoder_attention = (
            is_decoder
            and (
                ".self_attn."
                in module_name
                or ".encoder_attn."
                in module_name
            )
            and leaf_name
            in projection_names
        )

        is_decoder_ffn = (
            is_decoder
            and leaf_name
            in feed_forward_names
        )

        if placement == "enc":
            if (
                is_encoder_self_attention
                or is_encoder_ffn
            ):
                targets.append(
                    module_name
                )

        elif placement == "encattn":
            if is_encoder_self_attention:
                targets.append(
                    module_name
                )

        elif placement == "encdec":
            if (
                is_encoder_self_attention
                or is_encoder_ffn
                or is_decoder_attention
                or is_decoder_ffn
            ):
                targets.append(
                    module_name
                )

        else:
            raise ValueError(
                "Unsupported LoRA placement: "
                f"{placement!r}"
            )

    if not targets:
        raise RuntimeError(
            "No LoRA target modules "
            "were discovered."
        )

    return sorted(
        set(
            targets
        )
    )


# Residual-adapter wrapping

def add_residual_adapters(
    *,
    model: WhisperForConditionalGeneration,
    placement: str,
    bottleneck: int,
    dropout: float,
) -> tuple[
    WhisperForConditionalGeneration,
    str,
    tuple[int, ...],
]:
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    hidden_size = int(
        model.config.d_model
    )

    all_layers = tuple(
        range(24)
    )

    if placement == "enc":
        adapter_module = "encoder"

    elif placement == "dec":
        adapter_module = "decoder"

    elif placement == "encdec":
        adapter_module = (
            "encoder_decoder"
        )

    else:
        raise ValueError(
            "Unsupported ResAdapt placement: "
            f"{placement!r}"
        )

    if adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        encoder_layers = []

        for (
            layer_index,
            base_layer,
        ) in enumerate(
            model.model.encoder.layers
        ):
            if layer_index in all_layers:
                adapter = ResidualAdapter(
                    hidden_size=(
                        hidden_size
                    ),
                    bottleneck_size=(
                        bottleneck
                    ),
                    dropout=(
                        dropout
                    ),
                )

                encoder_layers.append(
                    AdapterWrappedLayer(
                        base_layer=(
                            base_layer
                        ),
                        adapter=adapter,
                    )
                )

            else:
                encoder_layers.append(
                    base_layer
                )

        model.model.encoder.layers = (
            nn.ModuleList(
                encoder_layers
            )
        )

    if adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        decoder_layers = []

        for (
            layer_index,
            base_layer,
        ) in enumerate(
            model.model.decoder.layers
        ):
            if layer_index in all_layers:
                adapter = ResidualAdapter(
                    hidden_size=(
                        hidden_size
                    ),
                    bottleneck_size=(
                        bottleneck
                    ),
                    dropout=(
                        dropout
                    ),
                )

                decoder_layers.append(
                    AdapterWrappedLayer(
                        base_layer=(
                            base_layer
                        ),
                        adapter=adapter,
                    )
                )

            else:
                decoder_layers.append(
                    base_layer
                )

        model.model.decoder.layers = (
            nn.ModuleList(
                decoder_layers
            )
        )

    for (
        name,
        parameter,
    ) in model.named_parameters():
        parameter.requires_grad = (
            "residual_adapter"
            in name
        )

    return (
        model,
        adapter_module,
        all_layers,
    )


# Build transfer model

def build_transfer_model(
    experiment: (
        config.ExperimentConfig
    ),
) -> tuple[
    nn.Module,
    WhisperProcessor,
    dict[str, Any],
]:
    validate_source_model(
        experiment
    )

    target_config = (
        config.get_target_family_config(
            experiment
        )
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            experiment
            .source_model_directory
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=experiment.training_language,
        task=config.TASK,
        predict_timestamps=False,
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            experiment
            .source_model_directory
        )
    )

    configure_whisper_generation(
        model,
        language=experiment.training_language,
        task=config.TASK,
        use_cache=(
            not bool(
                target_config
                .USE_GRADIENT_CHECKPOINTING
            )
        ),
    )

    if experiment.mechanism == "fft":
        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = True

        architecture = {
            "adaptation": (
                "full_finetune"
            ),
            "source_model": (
                experiment
                .source_model_name
            ),
            "source_model_root": str(
                experiment
                .source_model_root
            ),
            "source_model_directory": str(
                experiment
                .source_model_directory
            ),
        }

        return (
            model,
            processor,
            architecture,
        )

    if experiment.mechanism == "lora":
        if experiment.placement is None:
            raise ValueError(
                "LoRA experiment has "
                "no placement."
            )

        targets = (
            collect_lora_targets(
                model=model,
                placement=(
                    experiment.placement
                ),
            )
        )

        lora_configuration = (
            LoraConfig(
                r=int(
                    target_config
                    .LORA_RANK
                ),
                lora_alpha=int(
                    target_config
                    .LORA_ALPHA
                ),
                lora_dropout=float(
                    target_config
                    .LORA_DROPOUT
                ),
                bias=str(
                    target_config
                    .LORA_BIAS
                ),
                target_modules=targets,
                use_dora=bool(
                    target_config
                    .USE_DORA
                ),
                use_rslora=bool(
                    target_config
                    .USE_RSLORA
                ),
            )
        )

        model = get_peft_model(
            model,
            lora_configuration,
        )

        non_lora_trainable = [
            name
            for (
                name,
                parameter,
            ) in model.named_parameters()
            if (
                parameter.requires_grad
                and "lora_"
                not in name
            )
        ]

        if non_lora_trainable:
            raise RuntimeError(
                "Cold LoRA transfer has "
                "non-LoRA trainable parameters:\n"
                + "\n".join(
                    non_lora_trainable[:50]
                )
            )

        architecture = {
            "adaptation": "lora",
            "placement": (
                experiment.placement
            ),
            "source_model": (
                experiment
                .source_model_name
            ),
            "base_model_root": str(
                experiment
                .source_model_root
            ),
            "source_model_directory": str(
                experiment
                .source_model_directory
            ),
            "rank": int(
                target_config.LORA_RANK
            ),
            "alpha": int(
                target_config.LORA_ALPHA
            ),
            "dropout": float(
                target_config.LORA_DROPOUT
            ),
            "bias": str(
                target_config.LORA_BIAS
            ),
            "target_modules": (
                targets
            ),
        }

        return (
            model,
            processor,
            architecture,
        )

    if experiment.mechanism == "resadapt":
        if experiment.placement is None:
            raise ValueError(
                "ResAdapt experiment has "
                "no placement."
            )

        (
            model,
            adapter_module,
            adapter_layers,
        ) = add_residual_adapters(
            model=model,
            placement=(
                experiment.placement
            ),
            bottleneck=int(
                target_config
                .ADAPTER_BOTTLENECK
            ),
            dropout=float(
                target_config
                .ADAPTER_DROPOUT
            ),
        )

        non_adapter_trainable = [
            name
            for (
                name,
                parameter,
            ) in model.named_parameters()
            if (
                parameter.requires_grad
                and "residual_adapter"
                not in name
            )
        ]

        if non_adapter_trainable:
            raise RuntimeError(
                "Cold ResAdapt transfer has "
                "non-adapter trainable "
                "parameters:\n"
                + "\n".join(
                    non_adapter_trainable[:50]
                )
            )

        architecture = {
            "adaptation": (
                "residual_adapter"
            ),
            "placement": (
                experiment.placement
            ),
            "adapter_module": (
                adapter_module
            ),
            "adapter_layers": list(
                adapter_layers
            ),
            "bottleneck": int(
                target_config
                .ADAPTER_BOTTLENECK
            ),
            "dropout": float(
                target_config
                .ADAPTER_DROPOUT
            ),
            "position": (
                target_config
                .ADAPTER_POSITION
            ),
            "source_model": (
                experiment
                .source_model_name
            ),
            "base_model_root": str(
                experiment
                .source_model_root
            ),
            "source_model_directory": str(
                experiment
                .source_model_directory
            ),
        }

        return (
            model,
            processor,
            architecture,
        )

    raise ValueError(
        "Unsupported transfer mechanism: "
        f"{experiment.mechanism!r}"
    )


# Data

def load_training_data(
    experiment: (
        config.ExperimentConfig
    ),
) -> list[
    dict[str, str]
]:
    train_manifest = (
        config.get_train_manifest(
            experiment
        )
    )

    expected_training_examples = (
        config
        .get_expected_training_examples(
            experiment
        )
    )

    train_rows = load_manifest(
        train_manifest,
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
        != expected_training_examples
    ):
        raise RuntimeError(
            "Unexpected target training "
            "set size.\n"
            f"Expected: "
            f"{expected_training_examples}\n"
            f"Found: "
            f"{len(train_rows)}"
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
            f"Found: "
            f"{len(test_rows)}"
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name=(
            "transfer target training"
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
        first_name=(
            "target_training"
        ),
        second_name=(
            "llanito_fixed_test"
        ),
    )

    validate_manifest_files_exist(
        train_rows
    )

    validate_manifest_files_exist(
        test_rows
    )

    return train_rows


# Permanent ResAdapt saving

def save_residual_adapter(
    *,
    model: nn.Module,
    processor: WhisperProcessor,
    experiment: (
        config.ExperimentConfig
    ),
    architecture: dict[str, Any],
) -> None:
    directory = (
        experiment.model_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = model

    while hasattr(
        base,
        "module",
    ):
        base = base.module

    adapter_state: dict[
        str,
        torch.Tensor,
    ] = {}

    adapter_module = str(
        architecture[
            "adapter_module"
        ]
    )

    adapter_layers = {
        int(value)
        for value
        in architecture[
            "adapter_layers"
        ]
    }

    if adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        for (
            layer_index,
            layer,
        ) in enumerate(
            base.model.encoder.layers
        ):
            if (
                layer_index
                not in adapter_layers
            ):
                continue

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise RuntimeError(
                    "Expected wrapped encoder "
                    f"layer {layer_index}."
                )

            for (
                key,
                value,
            ) in (
                layer
                .residual_adapter
                .state_dict()
                .items()
            ):
                adapter_state[
                    (
                        f"encoder."
                        f"layer_{layer_index}."
                        f"{key}"
                    )
                ] = (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )

    if adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        for (
            layer_index,
            layer,
        ) in enumerate(
            base.model.decoder.layers
        ):
            if (
                layer_index
                not in adapter_layers
            ):
                continue

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise RuntimeError(
                    "Expected wrapped decoder "
                    f"layer {layer_index}."
                )

            for (
                key,
                value,
            ) in (
                layer
                .residual_adapter
                .state_dict()
                .items()
            ):
                adapter_state[
                    (
                        f"decoder."
                        f"layer_{layer_index}."
                        f"{key}"
                    )
                ] = (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )

    torch.save(
        adapter_state,
        directory
        / "adapter_state.pt",
    )

    save_json(
        directory
        / "adapter_config.json",
        {
            "format_version": 1,
            "adapter_module": (
                adapter_module
            ),
            "adapter_layers": sorted(
                adapter_layers
            ),
            "bottleneck": (
                architecture[
                    "bottleneck"
                ]
            ),
            "dropout": (
                architecture[
                    "dropout"
                ]
            ),
            "position": (
                architecture[
                    "position"
                ]
            ),
            "base_model": (
                experiment
                .source_model_name
            ),
            "base_model_root": str(
                experiment
                .source_model_root
            ),
        },
    )

    processor.save_pretrained(
        directory
    )


# Permanent model saving

def save_final_model(
    *,
    model: nn.Module,
    processor: WhisperProcessor,
    experiment: (
        config.ExperimentConfig
    ),
    architecture: dict[str, Any],
) -> None:
    experiment.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = model

    while hasattr(
        base,
        "module",
    ):
        base = base.module

    if experiment.mechanism == "fft":
        base.save_pretrained(
            experiment.model_directory
        )

        processor.save_pretrained(
            experiment.model_directory
        )

        return

    if experiment.mechanism == "lora":
        base.save_pretrained(
            experiment.model_directory
        )

        processor.save_pretrained(
            experiment.model_directory
        )

        return

    if experiment.mechanism == "resadapt":
        save_residual_adapter(
            model=base,
            processor=processor,
            experiment=experiment,
            architecture=architecture,
        )

        return

    raise ValueError(
        "Unsupported mechanism while "
        "saving final model."
    )


# Existing metadata

def load_existing_metadata(
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[str, Any]:
    if not (
        experiment
        .metadata_path
        .exists()
    ):
        raise FileNotFoundError(
            "Cannot resume because metadata "
            "does not exist:\n"
            f"{experiment.metadata_path}"
        )

    with (
        experiment.metadata_path
    ).open(
        "r",
        encoding="utf-8",
    ) as input_file:
        metadata = json.load(
            input_file
        )

    if (
        metadata.get(
            "model_name"
        )
        != experiment.model_name
    ):
        raise RuntimeError(
            "Existing metadata belongs to "
            "a different experiment."
        )

    return metadata


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
    target_config = (
        config.get_target_family_config(
            experiment
        )
    )

    (
        warmup_steps,
        warmup_ratio,
    ) = resolve_warmup(
        target_config
    )

    validate_cuda_environment(
        use_fp16=bool(
            target_config.USE_FP16
        ),
        use_bf16=bool(
            target_config.USE_BF16
        ),
    )

    set_seed(
        config.SEED
    )

    train_rows = (
        load_training_data(
            experiment
        )
    )

    (
        model,
        processor,
        architecture,
    ) = build_transfer_model(
        experiment
    )

    counts = parameter_summary(
        model
    )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"FFT TRANSFER: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

        print()
        print(
            f"Direction: "
            f"{experiment.direction}"
        )

        print(
            f"Mechanism: "
            f"{experiment.mechanism}"
        )

        print(
            f"Transfer type: "
            f"{experiment.transfer_type}"
        )

        print(
            f"Source: "
            f"{experiment.source_model_name}"
        )

        print(
            f"Source directory: "
            f"{experiment.source_model_directory}"
        )

        print(
            f"Target domain: "
            f"{experiment.target_domain}"
        )

        print(
            f"Training language: "
            f"{experiment.training_language}"
        )

        print(
            f"Target recipe: "
            f"{target_config.__name__}"
        )

        print(
            f"Placement: "
            f"{experiment.placement}"
        )

        print(
            f"Training examples: "
            f"{len(train_rows)}"
        )

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

        print()
        print(
            f"Learning rate: "
            f"{target_config.LEARNING_RATE}"
        )

        print(
            f"Epochs: "
            f"{target_config.EPOCHS}"
        )

        print(
            f"Warmup steps: "
            f"{warmup_steps}"
        )

        print(
            f"Warmup ratio: "
            f"{warmup_ratio}"
        )

        print(
            f"Seed: "
            f"{config.SEED}"
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
            epochs=float(
                target_config.EPOCHS
            ),
            learning_rate=float(
                target_config.LEARNING_RATE
            ),
            weight_decay=float(
                target_config.WEIGHT_DECAY
            ),
            warmup_steps=(
                warmup_steps
            ),
            warmup_ratio=(
                warmup_ratio
            ),
            train_batch_size=int(
                target_config
                .TRAIN_BATCH_SIZE_PER_DEVICE
            ),
            gradient_accumulation_steps=int(
                target_config
                .GRADIENT_ACCUMULATION_STEPS
            ),
            use_fp16=bool(
                target_config.USE_FP16
            ),
            use_bf16=bool(
                target_config.USE_BF16
            ),
            use_gradient_checkpointing=bool(
                target_config
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
                config.GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                config.GENERATION_NUM_BEAMS
            ),
            lr_scheduler_type=str(
                target_config
                .LR_SCHEDULER_TYPE
            ),
            max_grad_norm=float(
                target_config.MAX_GRAD_NORM
            ),
            optimizer=str(
                target_config.OPTIMIZER
            ),
            save_only_model=False,
        )
    )

    classification = (
        build_classification(
            family=(
                config.get_family(
                    experiment
                )
            ),
            transfer_type=(
                experiment.transfer_type
            ),
            domain_category=(
                experiment.direction
            ),
        )
    )

    train_manifest = (
        config.get_train_manifest(
            experiment
        )
    )

    hyperparameter_selection = {
        "method": (
            "reuse_target_domain_recipe"
        ),
        "target_domain": (
            experiment.target_domain
        ),
        "target_mechanism": (
            experiment.mechanism
        ),
        "source_recipe_module": (
            target_config.__name__
        ),
        "selected_recipe": {
            "learning_rate": float(
                target_config.LEARNING_RATE
            ),
            "epochs": float(
                target_config.EPOCHS
            ),
            "warmup_steps": (
                warmup_steps
            ),
            "warmup_ratio": (
                warmup_ratio
            ),
            "batch_size_per_device": int(
                target_config
                .TRAIN_BATCH_SIZE_PER_DEVICE
            ),
            "gradient_accumulation_steps": int(
                target_config
                .GRADIENT_ACCUMULATION_STEPS
            ),
            "weight_decay": float(
                target_config.WEIGHT_DECAY
            ),
            "fp16": bool(
                target_config.USE_FP16
            ),
            "bf16": bool(
                target_config.USE_BF16
            ),
            "seed": (
                config.SEED
            ),
            "data_seed": (
                config.SEED
            ),
        },
        "validation_during_final_training": False,
        "fixed_test_used": False,
    }

    if (
        resume_from_checkpoint
        is None
    ):
        metadata = (
            save_training_metadata(
                path=(
                    experiment.metadata_path
                ),
                model_name=(
                    experiment.model_name
                ),
                model_type=(
                    (
                        "whisper_full"
                        if experiment.mechanism
                        == "fft"
                        else (
                            "peft_lora"
                            if experiment.mechanism
                            == "lora"
                            else (
                                "residual_adapter"
                            )
                        )
                    )
                ),
                model=model,
                training_args=(
                    training_args
                ),
                train_manifest=(
                    train_manifest
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
                    experiment
                    .source_model_name
                ),
                language=(
                    experiment.training_language
                ),
                task=config.TASK,
                hyperparameter_selection=(
                    hyperparameter_selection
                ),
                architecture=(
                    architecture
                ),
                description=(
                    experiment.description
                ),
            )
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
        save_final_model(
            model=trainer.model,
            processor=processor,
            experiment=experiment,
            architecture=architecture,
        )

    distributed_barrier()

    if is_main_process():
        update_training_metadata_after_training(
            experiment.metadata_path,
            existing_metadata=metadata,
            trainer=trainer,
            train_result=train_result,
            final_model_path=(
                experiment.model_directory
            ),
        )

    distributed_barrier()

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"{experiment.model_name} "
            "TRAINING COMPLETE"
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