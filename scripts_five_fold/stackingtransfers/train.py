from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.stackingtransfers.config import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    build_runtime_config,
    fold_train_manifest,
    fold_validation_manifest,
    source_adapter_for_fold,
    llanito_fold_source,
)

from scripts_five_fold.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    distributed_barrier,
    is_main_process,
    load_manifest,
    load_task_mappings,
    parameter_summary,
    save_json,
    select_subset,
    set_seed,
    validate_cuda_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        choices=sorted(EXPERIMENTS),
    )

    return parser.parse_args()


# Miami manifest loader

MIAMI_REQUIRED_COLUMNS = (
    "example_id",
    "conversation_id",
    "speaker",
    "speaker_code",
    "split",
    "audio_path",
    "transcript",
)


def load_miami_manifest(
    path: Path,
    *,
    expected_split: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Miami manifest missing:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"Manifest has no header:\n{path}"
            )

        missing = (
            set(MIAMI_REQUIRED_COLUMNS)
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                f"{path} missing columns: "
                f"{sorted(missing)}"
            )

        rows = [
            dict(row)
            for row in reader
        ]

    expected = (
        expected_split
        .strip()
        .casefold()
    )

    wrong = sorted({
        str(row.get("split", "")).strip()
        for row in rows
        if (
            str(row.get("split", ""))
            .strip()
            .casefold()
            != expected
        )
    })

    if wrong:
        raise ValueError(
            f"{path} expected split "
            f"{expected_split!r}, found {wrong}."
        )

    for row in rows:
        row["task"] = "miami"

    return rows


# Residual adapter architecture

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
        self.dropout = nn.Dropout(dropout)

        self.up = nn.Linear(
            bottleneck_size,
            hidden_size,
        )

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        correction = self.down(hidden_states)
        correction = self.activation(correction)
        correction = self.dropout(correction)
        correction = self.up(correction)

        return hidden_states + correction


class SourceAdapterWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer

        self.source_adapter = ResidualAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
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

        return apply_adapter_to_output(
            output,
            self.source_adapter,
        )


class StackedMiamiWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer

        self.miami_adapter = ResidualAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
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

        return apply_adapter_to_output(
            output,
            self.miami_adapter,
        )


def apply_adapter_to_output(
    output: Any,
    adapter: ResidualAdapter,
) -> Any:
    if isinstance(output, torch.Tensor):
        return adapter(output)

    if isinstance(output, tuple):
        if not output:
            raise RuntimeError(
                "Whisper layer returned empty tuple."
            )

        return (
            adapter(output[0]),
            *output[1:],
        )

    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = adapter(
            output.last_hidden_state
        )

        return output

    raise TypeError(
        "Unexpected Whisper layer output type: "
        f"{type(output)!r}"
    )


# Source checkpoint loading

def load_source_metadata(
    source_directory: Path,
) -> dict[str, Any]:
    path = (
        source_directory
        / "adapter_config.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Source adapter config missing:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    layers = metadata.get(
        "adapter_layers",
        list(range(24)),
    )

    metadata["adapter_layers"] = [
        int(value)
        for value in layers
    ]

    bottleneck = metadata.get(
        "bottleneck"
    )

    if bottleneck is None:
        raise ValueError(
            "Source adapter config has no "
            "'bottleneck' value."
        )

    metadata["bottleneck"] = int(
        bottleneck
    )

    metadata["dropout"] = float(
        metadata.get(
            "dropout",
            0.0,
        )
    )

    return metadata


def load_source_state(
    source_directory: Path,
) -> dict[str, torch.Tensor]:
    path = (
        source_directory
        / "adapter_state.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Source adapter state missing:\n{path}"
        )

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(state, dict):
        raise TypeError(
            "Source adapter state must "
            "be a dictionary."
        )

    return state


def layer_state_from_saved(
    state: dict[str, torch.Tensor],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    prefixes = (
        f"encoder.layer_{layer_index}.",
        f"layer_{layer_index}.",
    )

    for prefix in prefixes:
        layer_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

        if layer_state:
            return layer_state

    raise RuntimeError(
        "No saved adapter weights for "
        f"encoder layer {layer_index}."
    )


def ensure_averaged_llanito_source(config) -> Path:
    destination = config.AVERAGED_LLANITO_SOURCE
    state_path = destination / "adapter_state.pt"
    metadata_path = destination / "adapter_config.json"

    if state_path.exists() and metadata_path.exists():
        return destination

    source_directories = [
        llanito_fold_source(config, fold)
        for fold in config.LLANITO_SOURCE_FOLDS
    ]

    for source in source_directories:
        if not source.exists():
            raise FileNotFoundError(
                f"Llanito fold source adapter missing:\n"
                f"{source}"
            )

    metadatas = [
        load_source_metadata(source)
        for source in source_directories
    ]

    reference = metadatas[0]

    for fold, metadata in zip(
        config.LLANITO_SOURCE_FOLDS[1:],
        metadatas[1:],
    ):
        for key in (
            "adapter_layers",
            "bottleneck",
        ):
            if metadata[key] != reference[key]:
                raise ValueError(
                    "Cannot average Llanito fold adapters: "
                    f"fold {fold} differs in {key}. "
                    f"reference={reference[key]!r}, "
                    f"fold={metadata[key]!r}"
                )

    states = [
        load_source_state(source)
        for source in source_directories
    ]

    reference_keys = set(
        states[0]
    )

    for fold, state in zip(
        config.LLANITO_SOURCE_FOLDS[1:],
        states[1:],
    ):
        if set(state) != reference_keys:
            raise ValueError(
                "Cannot average Llanito fold "
                f"{fold}: parameter names differ."
            )

    averaged = {}

    for key in sorted(
        reference_keys
    ):
        tensors = [
            state[key]
            for state in states
        ]

        shape = tensors[0].shape

        if any(
            tensor.shape != shape
            for tensor in tensors[1:]
        ):
            raise ValueError(
                f"Cannot average {key}: "
                "tensor shapes differ."
            )

        if (
            tensors[0].is_floating_point()
            or tensors[0].is_complex()
        ):
            averaged[key] = (
                torch.stack(
                    [
                        tensor.to(
                            torch.float32
                        )
                        for tensor in tensors
                    ],
                    dim=0,
                )
                .mean(
                    dim=0
                )
                .to(
                    dtype=tensors[0].dtype
                )
            )

        else:
            if any(
                not torch.equal(
                    tensors[0],
                    tensor,
                )
                for tensor in tensors[1:]
            ):
                raise ValueError(
                    "Non-floating tensor "
                    f"{key} differs across folds."
                )

            averaged[key] = (
                tensors[0]
                .clone()
            )

    if is_main_process():
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            averaged,
            state_path,
        )

        save_json(
            metadata_path,
            {
                "format_version": 1,
                "experiment": (
                    "llanitoresadaptenc_averaged"
                ),
                "base_model": (
                    config.BASE_MODEL
                ),
                "adapter_module": (
                    "encoder"
                ),
                "adapter_position": (
                    "all"
                ),
                "adapter_layers": (
                    reference[
                        "adapter_layers"
                    ]
                ),
                "bottleneck": (
                    reference[
                        "bottleneck"
                    ]
                ),
                "dropout": (
                    reference[
                        "dropout"
                    ]
                ),
                "averaging": (
                    "elementwise_arithmetic_mean"
                ),
                "source_folds": list(
                    config.LLANITO_SOURCE_FOLDS
                ),
                "source_adapters": [
                    str(path)
                    for path
                    in source_directories
                ],
            },
        )

        print(
            "Saved averaged Llanito "
            f"adapter to: {destination}",
            flush=True,
        )

    distributed_barrier()

    return destination


def build_source_model(
    config,
    source_directory: Path,
) -> WhisperForConditionalGeneration:
    metadata = load_source_metadata(
        source_directory
    )

    layers = metadata[
        "adapter_layers"
    ]

    if layers != list(
        range(config.ENCODER_LAYERS)
    ):
        raise ValueError(
            "This experiment requires a source "
            "adapter on all 24 encoder layers. "
            f"Saved layers: {layers}"
        )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            config.BASE_MODEL
        )
    )

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    hidden_size = int(
        model.config.d_model
    )

    wrapped = []

    for layer in (
        model.model.encoder.layers
    ):
        wrapped.append(
            SourceAdapterWrappedLayer(
                base_layer=layer,
                hidden_size=hidden_size,
                bottleneck_size=(
                    metadata["bottleneck"]
                ),
                dropout=(
                    metadata["dropout"]
                ),
            )
        )

    model.model.encoder.layers = (
        nn.ModuleList(
            wrapped
        )
    )

    state = load_source_state(
        source_directory
    )

    for layer_index in range(
        config.ENCODER_LAYERS
    ):
        layer = (
            model.model.encoder.layers[
                layer_index
            ]
        )

        if not isinstance(
            layer,
            SourceAdapterWrappedLayer,
        ):
            raise TypeError(
                "Source wrapper construction failed."
            )

        layer.source_adapter.load_state_dict(
            layer_state_from_saved(
                state,
                layer_index,
            ),
            strict=True,
        )

    return model


def source_adapter_modules(
    model: WhisperForConditionalGeneration,
    config,
) -> list[
    tuple[int, ResidualAdapter]
]:
    result = []

    for layer_index in range(
        config.ENCODER_LAYERS
    ):
        layer = (
            model.model.encoder.layers[
                layer_index
            ]
        )

        if isinstance(
            layer,
            StackedMiamiWrappedLayer,
        ):
            layer = layer.base_layer

        if not isinstance(
            layer,
            SourceAdapterWrappedLayer,
        ):
            raise TypeError(
                "Expected source adapter at "
                f"encoder layer {layer_index}."
            )

        result.append(
            (
                layer_index,
                layer.source_adapter,
            )
        )

    return result


def new_miami_adapter_modules(
    model: WhisperForConditionalGeneration,
    config,
) -> list[
    tuple[int, ResidualAdapter]
]:
    result = []

    for layer_index in range(
        config.ENCODER_LAYERS
    ):
        layer = (
            model.model.encoder.layers[
                layer_index
            ]
        )

        if not isinstance(
            layer,
            StackedMiamiWrappedLayer,
        ):
            raise TypeError(
                "Expected stacked Miami adapter "
                f"at encoder layer {layer_index}."
            )

        result.append(
            (
                layer_index,
                layer.miami_adapter,
            )
        )

    return result


def build_transfer_model(
    config,
    source_directory: Path,
) -> WhisperForConditionalGeneration:
    model = build_source_model(
        config,
        source_directory,
    )

    # Soft transfer

    if config.MODE == "soft":
        for _, adapter in (
            source_adapter_modules(
                model,
                config,
            )
        ):
            for parameter in (
                adapter.parameters()
            ):
                parameter.requires_grad = True

    # Stacked transfer

    elif config.MODE == "stacked":
        # Freeze everything that already exists:
        # vanilla Whisper + loaded source adapter.
        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = False

        hidden_size = int(
            model.config.d_model
        )

        wrapped = []

        for layer in (
            model.model.encoder.layers
        ):
            wrapped.append(
                StackedMiamiWrappedLayer(
                    base_layer=layer,
                    hidden_size=hidden_size,
                    bottleneck_size=(
                        config
                        .NEW_MIAMI_ADAPTER_BOTTLENECK
                    ),
                    dropout=(
                        config
                        .NEW_MIAMI_ADAPTER_DROPOUT
                    ),
                )
            )

        model.model.encoder.layers = (
            nn.ModuleList(
                wrapped
            )
        )

        # Only the new Miami adapters train.
        for _, adapter in (
            new_miami_adapter_modules(
                model,
                config,
            )
        ):
            for parameter in (
                adapter.parameters()
            ):
                parameter.requires_grad = True

    else:
        raise ValueError(
            f"Unknown mode: "
            f"{config.MODE!r}"
        )

    trainable = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable transfer parameters."
        )

    if config.MODE == "soft":
        unexpected = [
            name
            for name in trainable
            if (
                ".source_adapter."
                not in name
            )
        ]

    else:
        unexpected = [
            name
            for name in trainable
            if (
                ".miami_adapter."
                not in name
            )
        ]

    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters: "
            f"{unexpected[:20]}"
        )

    return model


# Saving new experiment state

def transfer_state_dict(
    model: WhisperForConditionalGeneration,
    config,
) -> dict[str, torch.Tensor]:
    state = {}

    modules = (
        source_adapter_modules(
            model,
            config,
        )
        if config.MODE == "soft"
        else new_miami_adapter_modules(
            model,
            config,
        )
    )

    for (
        layer_index,
        adapter,
    ) in modules:
        for key, value in (
            adapter.state_dict().items()
        ):
            state[
                f"encoder.layer_"
                f"{layer_index}.{key}"
            ] = (
                value.detach()
                .cpu()
                .clone()
            )

    return state


def save_transfer(
    model,
    processor,
    destination: Path,
    config,
    *,
    fold_number: int,
    source_directory: Path,
    metric: float,
    epoch,
    global_step: int,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        transfer_state_dict(
            model,
            config,
        ),
        destination
        / "adapter_state.pt",
    )

    source_metadata = (
        load_source_metadata(
            source_directory
        )
    )

    save_json(
        destination
        / "adapter_config.json",
        {
            "format_version": 1,
            "experiment": (
                config.EXPERIMENT_NAME
            ),
            "fold": (
                fold_number
            ),
            "mode": (
                config.MODE
            ),
            "direction": (
                config.DIRECTION
            ),
            "base_model": (
                config.BASE_MODEL
            ),
            "source_adapter": str(
                source_directory
            ),
            "source_bottleneck": (
                source_metadata[
                    "bottleneck"
                ]
            ),
            "source_dropout": (
                source_metadata[
                    "dropout"
                ]
            ),
            "adapter_layers": list(
                range(
                    config.ENCODER_LAYERS
                )
            ),
            "new_miami_bottleneck": (
                config
                .NEW_MIAMI_ADAPTER_BOTTLENECK
                if config.MODE
                == "stacked"
                else None
            ),
            "new_miami_dropout": (
                config
                .NEW_MIAMI_ADAPTER_DROPOUT
                if config.MODE
                == "stacked"
                else None
            ),
            "selection_metric": (
                config.SELECTION_METRIC
            ),
            "best_metric": (
                metric
            ),
            "epoch": (
                epoch
            ),
            "global_step": (
                global_step
            ),
        },
    )

    processor.save_pretrained(
        destination
    )


class SaveBestTransferCallback(
    TrainerCallback
):
    def __init__(
        self,
        *,
        config,
        processor,
        fold_number: int,
        source_directory: Path,
        destination: Path,
    ) -> None:
        self.config = config
        self.processor = processor
        self.fold_number = (
            fold_number
        )
        self.source_directory = (
            source_directory
        )
        self.destination = (
            destination
        )
        self.best = None

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        model=None,
        **kwargs,
    ):
        if (
            not metrics
            or model is None
            or not state.is_world_process_zero
        ):
            return control

        key = (
            "eval_normalised_wer"
            if (
                self.config
                .SELECTION_METRIC
                == "normalised_wer"
            )
            else "eval_wer"
        )

        if key not in metrics:
            raise KeyError(
                "Expected validation metric "
                f"{key!r}. Available: "
                f"{sorted(metrics)}"
            )

        value = float(
            metrics[key]
        )

        if (
            self.best is None
            or value < self.best
        ):
            self.best = value

            if (
                self.destination.exists()
            ):
                shutil.rmtree(
                    self.destination
                )

            save_transfer(
                model,
                self.processor,
                self.destination,
                self.config,
                fold_number=(
                    self.fold_number
                ),
                source_directory=(
                    self.source_directory
                ),
                metric=value,
                epoch=(
                    state.epoch
                ),
                global_step=(
                    state.global_step
                ),
            )

            print(
                "NEW BEST VALIDATION: "
                f"{value:.4f}%",
                flush=True,
            )

        return control


def clear_fold(
    model_directory: Path,
    output_directory: Path,
    config,
) -> None:
    if not (
        config
        .CLEAR_SELECTED_FOLD_BEFORE_TRAINING
    ):
        return

    if (
        config
        .RESUME_FROM_LATEST_CHECKPOINT
    ):
        return

    if model_directory.exists():
        shutil.rmtree(
            model_directory
        )

    if output_directory.exists():
        shutil.rmtree(
            output_directory
        )


def load_rows_for_fold(
    config,
    fold_number: int,
):
    train_manifest = (
        fold_train_manifest(
            config,
            fold_number,
        )
    )

    validation_manifest = (
        fold_validation_manifest(
            config,
            fold_number,
        )
    )

    if (
        config.TRAIN_DATASET
        == "miami"
    ):
        train_rows = (
            load_miami_manifest(
                train_manifest,
                expected_split="train",
            )
        )

        validation_rows = (
            load_miami_manifest(
                validation_manifest,
                expected_split=(
                    "validation"
                ),
            )
        )

        mappings = (
            load_task_mappings(
                None
            )
        )

    else:
        train_rows = load_manifest(
            train_manifest
        )

        validation_rows = load_manifest(
            validation_manifest
        )

        mappings = load_task_mappings(
            config
            .NORMALISATION_FILE
        )

    return (
        train_rows,
        validation_rows,
        mappings,
        train_manifest,
        validation_manifest,
    )


def train_fold(
    config,
    fold_number: int,
) -> None:
    fold_seed = (
        config.RANDOM_SEED
        + fold_number
    )

    set_seed(
        fold_seed
    )

    (
        train_rows,
        validation_rows,
        mappings,
        train_manifest,
        validation_manifest,
    ) = load_rows_for_fold(
        config,
        fold_number,
    )

    train_rows = select_subset(
        train_rows,
        config
        .MAX_TRAIN_RECORDINGS,
        fold_seed,
    )

    validation_rows = (
        select_subset(
            validation_rows,
            config
            .MAX_VALIDATION_RECORDINGS,
            fold_seed + 1000,
        )
    )

    source_directory = (
        source_adapter_for_fold(
            config,
            fold_number,
        )
    )

    if not (
        source_directory.exists()
    ):
        raise FileNotFoundError(
            f"Source adapter missing:\n"
            f"{source_directory}"
        )

    model_directory = (
        config.MODEL_ROOT
        / f"fold_{fold_number}"
    )

    output_directory = (
        config.OUTPUT_ROOT
        / f"fold_{fold_number}"
    )

    checkpoint_directory = (
        model_directory
        / "checkpoints"
    )

    best_directory = (
        model_directory
        / "adapter"
    )

    if is_main_process():
        clear_fold(
            model_directory,
            output_directory,
            config,
        )

        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    distributed_barrier()

    processor = (
        WhisperProcessor
        .from_pretrained(
            config.BASE_MODEL,
            language=(
                config.LANGUAGE
            ),
            task=(
                config.TASK
            ),
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=(
            config.LANGUAGE
        ),
        task=(
            config.TASK
        ),
        predict_timestamps=False,
    )

    model = build_transfer_model(
        config,
        source_directory,
    )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"{config.EXPERIMENT_NAME} "
            f"FOLD {fold_number}"
        )
        print("=" * 80)

        print(
            f"Mode: "
            f"{config.MODE}"
        )

        print(
            f"Direction: "
            f"{config.DIRECTION}"
        )

        print(
            f"Source: "
            f"{source_directory}"
        )

        print(
            f"Train manifest: "
            f"{train_manifest}"
        )

        print(
            "Validation manifest: "
            f"{validation_manifest}"
        )

        print(
            f"Train recordings: "
            f"{len(train_rows)}"
        )

        print(
            f"Validation recordings: "
            f"{len(validation_rows)}"
        )

        print(
            f"Parameters: "
            f"{parameter_summary(model)}"
        )

    train_dataset = (
        WhisperManifestDataset(
            train_rows,
            processor,
        )
    )

    validation_dataset = (
        WhisperManifestDataset(
            validation_rows,
            processor,
        )
    )

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=(
            model.config
            .decoder_start_token_id
        ),
    )

    compute_metrics = (
        build_compute_metrics(
            processor,
            validation_rows,
            mappings,
        )
    )

    training_args = (
        build_training_arguments(
            checkpoint_directory,
            fold_seed,
            epochs=(
                config
                .NUMBER_OF_EPOCHS
            ),
            learning_rate=(
                config
                .LEARNING_RATE
            ),
            weight_decay=(
                config
                .WEIGHT_DECAY
            ),
            warmup_steps=(
                config
                .WARMUP_STEPS
            ),
            train_batch_size=(
                config
                .TRAIN_BATCH_SIZE_PER_GPU
            ),
            validation_batch_size=(
                config
                .VALIDATION_BATCH_SIZE_PER_GPU
            ),
            gradient_accumulation_steps=(
                config
                .GRADIENT_ACCUMULATION_STEPS
            ),
            use_fp16=(
                config
                .USE_FP16
            ),
            use_bf16=(
                config
                .USE_BF16
            ),
            use_gradient_checkpointing=(
                config
                .USE_GRADIENT_CHECKPOINTING
            ),
            logging_steps=(
                config
                .LOGGING_STEPS
            ),
            save_total_limit=(
                config
                .SAVE_TOTAL_LIMIT
            ),
            generation_max_length=(
                config
                .GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                config
                .GENERATION_NUM_BEAMS
            ),
            dataloader_workers=(
                config
                .DATALOADER_WORKERS
            ),
        )
    )

    # Custom callback selects the
    # best transfer state.
    training_args.load_best_model_at_end = False
    training_args.metric_for_best_model = None
    training_args.greater_is_better = None

    callback = (
        SaveBestTransferCallback(
            config=config,
            processor=processor,
            fold_number=(
                fold_number
            ),
            source_directory=(
                source_directory
            ),
            destination=(
                best_directory
            ),
        )
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=(
            train_dataset
        ),
        eval_dataset=(
            validation_dataset
        ),
        data_collator=(
            collator
        ),
        compute_metrics=(
            compute_metrics
        ),
        callbacks=[
            callback
        ],
        processing_class=(
            processor
        ),
    )

    checkpoint = None

    if (
        config
        .RESUME_FROM_LATEST_CHECKPOINT
    ):
        checkpoint = (
            get_last_checkpoint(
                str(
                    checkpoint_directory
                )
            )
        )

    train_result = (
        trainer.train(
            resume_from_checkpoint=(
                checkpoint
            )
        )
    )

    if (
        trainer
        .is_world_process_zero()
    ):
        if callback.best is None:
            raise RuntimeError(
                "No best validation "
                "state was saved."
            )

        save_json(
            output_directory
            / "training_summary.json",
            {
                "experiment": (
                    config
                    .EXPERIMENT_NAME
                ),
                "fold": (
                    fold_number
                ),
                "mode": (
                    config.MODE
                ),
                "direction": (
                    config.DIRECTION
                ),
                "source_adapter": (
                    str(
                        source_directory
                    )
                ),
                "train_manifest": (
                    str(
                        train_manifest
                    )
                ),
                "validation_manifest": (
                    str(
                        validation_manifest
                    )
                ),
                "train_recordings": (
                    len(
                        train_rows
                    )
                ),
                "validation_recordings": (
                    len(
                        validation_rows
                    )
                ),
                "best_validation_metric": (
                    callback.best
                ),
                "selection_metric": (
                    config
                    .SELECTION_METRIC
                ),
                "best_adapter": (
                    str(
                        best_directory
                    )
                ),
                "train_metrics": (
                    train_result
                    .metrics
                ),
            },
        )

    distributed_barrier()

    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    distributed_barrier()


def main() -> None:
    print(
        "STACKING TRANSFER TRAINER STARTED",
        flush=True,
    )

    args = parse_args()

    config = build_runtime_config(
        args.experiment
    )

    validate_cuda_environment(
        use_fp16=(
            config.USE_FP16
        ),
        use_bf16=(
            config.USE_BF16
        ),
    )

    if (
        config.DIRECTION
        == "llanito_to_miami"
    ):
        ensure_averaged_llanito_source(
            config
        )

    for fold_number in (
        config.RUN_FOLDS
    ):
        train_fold(
            config,
            fold_number,
        )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"{config.EXPERIMENT_NAME} "
            "TRAINING COMPLETE"
        )
        print("=" * 80)

    distributed_barrier()


if __name__ == "__main__":
    main()