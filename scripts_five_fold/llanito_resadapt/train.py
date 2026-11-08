from __future__ import annotations

import csv
import gc
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import (
    get_last_checkpoint,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.llanito_resadapt.config import (
    ADAPTER_BOTTLENECK,
    ADAPTER_DROPOUT,
    ADAPTER_LAYERS,
    ADAPTER_MODULE,
    ADAPTER_POSITION,
    ALLOW_LEGACY_ENCODER_CHECKPOINTS,
    ALLOW_LEGACY_STATE_KEYS,
    BASE_MODEL,
    CLEAR_SELECTED_FOLD_BEFORE_TRAINING,
    DATALOADER_WORKERS,
    EXPERIMENT_NAME,
    FOLD_NUMBERS,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_TRAIN_RECORDINGS,
    MAX_VALIDATION_RECORDINGS,
    MIAMI_TRAIN_MANIFEST,
    MODEL_ROOT,
    NORMALISATION_FILE,
    NUMBER_OF_EPOCHS,
    OUTPUT_ROOT,
    PROCESSED_ROOT,
    RANDOM_SEED,
    RESUME_FROM_LATEST_CHECKPOINT,
    RUN_FOLDS,
    SAVE_TOTAL_LIMIT,
    SUMMARY_ROOT,
    TASK,
    TRAIN_BATCH_SIZE_PER_GPU,
    USE_BF16,
    USE_FP16,
    USE_GRADIENT_CHECKPOINTING,
    VALIDATION_BATCH_SIZE_PER_GPU,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

from scripts_five_fold.train import (
    ValidationMetricsCallback,
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    discover_fold_directories,
    distributed_barrier,
    extract_fold_number,
    is_main_process,
    load_manifest,
    load_task_mappings,
    make_trainer,
    parameter_summary,
    save_json,
    select_subset,
    set_seed,
    validate_cuda_environment,
    validate_manifest_disjointness,
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

        # Start as an exact identity mapping.

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
        base_layer: nn.Module,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer

        self.residual_adapter = (
            ResidualAdapter(
                hidden_size=hidden_size,
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
            return self.residual_adapter(
                output
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


# Model construction

def wrap_selected_layers(
    layers: nn.ModuleList,
    hidden_size: int,
) -> nn.ModuleList:
    selected = set(
        ADAPTER_LAYERS
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
                        ADAPTER_BOTTLENECK
                    ),
                    dropout=(
                        ADAPTER_DROPOUT
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


def build_residual_adapter_model(
) -> WhisperForConditionalGeneration:
    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL
        )
    )

    configure_whisper_generation(
        model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    # Freeze Whisper itself.

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(
        model.config.d_model
    )

    if ADAPTER_MODULE in {
        "encoder",
        "encoder_decoder",
    }:
        model.model.encoder.layers = (
            wrap_selected_layers(
                model.model.encoder.layers,
                hidden_size,
            )
        )

    if ADAPTER_MODULE in {
        "decoder",
        "encoder_decoder",
    }:
        model.model.decoder.layers = (
            wrap_selected_layers(
                model.model.decoder.layers,
                hidden_size,
            )
        )

    # Only residual adapters train.

    for (
        _module_name,
        _layer_index,
        adapter,
    ) in adapter_modules_for_state(
        model
    ):
        for parameter in (
            adapter.parameters()
        ):
            parameter.requires_grad = True

    verify_trainable_parameters(
        model
    )

    return model


def adapter_modules_for_state(
    model: WhisperForConditionalGeneration,
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

    if ADAPTER_MODULE in {
        "encoder",
        "encoder_decoder",
    }:
        for layer_index in ADAPTER_LAYERS:
            layer = (
                model.model.encoder.layers[
                    layer_index
                ]
            )

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise TypeError(
                    "Expected encoder residual "
                    "adapter at layer "
                    f"{layer_index}."
                )

            modules.append(
                (
                    "encoder",
                    layer_index,
                    layer.residual_adapter,
                )
            )

    if ADAPTER_MODULE in {
        "decoder",
        "encoder_decoder",
    }:
        for layer_index in ADAPTER_LAYERS:
            layer = (
                model.model.decoder.layers[
                    layer_index
                ]
            )

            if not isinstance(
                layer,
                AdapterWrappedLayer,
            ):
                raise TypeError(
                    "Expected decoder residual "
                    "adapter at layer "
                    f"{layer_index}."
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
    trainable = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable parameters "
            "were found."
        )

    unexpected = [
        name
        for name in trainable
        if ".residual_adapter."
        not in name
    ]

    if unexpected:
        raise RuntimeError(
            "Non-adapter parameters are "
            "trainable. Examples: "
            f"{unexpected[:10]}"
        )


# Adapter persistence

def residual_adapter_state_dict(
    model: WhisperForConditionalGeneration,
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
        model
    ):
        for (
            key,
            value,
        ) in (
            adapter.state_dict().items()
        ):
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

    return state


def save_residual_adapter(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    adapter_directory: Path,
) -> None:
    adapter_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = (
        residual_adapter_state_dict(
            model
        )
    )

    if not state:
        raise RuntimeError(
            "Refusing to save an "
            "empty adapter state."
        )

    torch.save(
        state,
        adapter_directory
        / "adapter_state.pt",
    )

    metadata = {
        "format_version": 2,
        "experiment": (
            EXPERIMENT_NAME
        ),
        "base_model": (
            BASE_MODEL
        ),
        "adapter_module": (
            ADAPTER_MODULE
        ),
        "adapter_position": (
            ADAPTER_POSITION
        ),
        "adapter_layers": (
            ADAPTER_LAYERS
        ),
        "bottleneck": (
            ADAPTER_BOTTLENECK
        ),
        "dropout": (
            ADAPTER_DROPOUT
        ),
        "training_data": (
            "joint_miami_llanito"
        ),
        "validation_data": (
            "llanito_only"
        ),
    }

    save_json(
        adapter_directory
        / "adapter_config.json",
        metadata,
    )

    processor.save_pretrained(
        adapter_directory
    )


def load_saved_adapter_config(
    adapter_directory: Path,
) -> dict[
    str,
    Any,
]:
    config_path = (
        adapter_directory
        / "adapter_config.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Residual adapter config "
            "is missing:\n"
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        saved = json.load(
            input_file
        )

    if not isinstance(
        saved,
        dict,
    ):
        raise TypeError(
            "adapter_config.json must "
            "contain a JSON object."
        )

    return saved


def canonical_saved_adapter_module(
    saved_config: dict[
        str,
        Any,
    ],
) -> str:
    saved_module = (
        saved_config.get(
            "adapter_module"
        )
    )

    if saved_module is None:
        if not (
            ALLOW_LEGACY_ENCODER_CHECKPOINTS
        ):
            raise ValueError(
                "Old checkpoint has no "
                "adapter_module metadata."
            )

        return "encoder"

    return str(
        saved_module
    )


def validate_saved_adapter_architecture(
    saved_config: dict[
        str,
        Any,
    ],
) -> None:
    saved_module = (
        canonical_saved_adapter_module(
            saved_config
        )
    )

    if saved_module != ADAPTER_MODULE:
        raise ValueError(
            "Adapter module mismatch: "
            f"saved={saved_module!r}, "
            f"current={ADAPTER_MODULE!r}"
        )

    saved_position = (
        saved_config.get(
            "adapter_position"
        )
    )

    if (
        saved_position is not None
        and saved_position
        != ADAPTER_POSITION
    ):
        raise ValueError(
            "Adapter position mismatch: "
            f"saved={saved_position!r}, "
            f"current={ADAPTER_POSITION!r}"
        )

    saved_layers = (
        saved_config.get(
            "adapter_layers"
        )
    )

    if saved_layers is not None:
        saved_layers = [
            int(layer)
            for layer in saved_layers
        ]

        if saved_layers != ADAPTER_LAYERS:
            raise ValueError(
                "Adapter layer mismatch: "
                f"saved={saved_layers}, "
                f"current={ADAPTER_LAYERS}"
            )

    saved_bottleneck = (
        saved_config.get(
            "bottleneck"
        )
    )

    if (
        saved_bottleneck is not None
        and int(saved_bottleneck)
        != ADAPTER_BOTTLENECK
    ):
        raise ValueError(
            "Adapter bottleneck mismatch: "
            f"saved={saved_bottleneck}, "
            f"current={ADAPTER_BOTTLENECK}"
        )


def load_adapter_state_file(
    state_path: Path,
) -> dict[
    str,
    torch.Tensor,
]:
    if not state_path.exists():
        raise FileNotFoundError(
            "Residual adapter weights "
            "are missing:\n"
            f"{state_path}"
        )

    try:
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        state = torch.load(
            state_path,
            map_location="cpu",
        )

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "adapter_state.pt did not "
            "contain a state dictionary."
        )

    if not state:
        raise RuntimeError(
            "adapter_state.pt contains "
            "no weights."
        )

    return state


def state_for_adapter(
    state: dict[
        str,
        torch.Tensor,
    ],
    module_name: str,
    layer_index: int,
) -> dict[
    str,
    torch.Tensor,
]:
    new_prefix = (
        f"{module_name}."
        f"layer_{layer_index}."
    )

    selected = {
        key[
            len(new_prefix):
        ]: value
        for (
            key,
            value,
        ) in state.items()
        if key.startswith(
            new_prefix
        )
    }

    if selected:
        return selected

    if (
        ALLOW_LEGACY_STATE_KEYS
        and module_name == "encoder"
    ):
        legacy_prefix = (
            f"layer_{layer_index}."
        )

        selected = {
            key[
                len(legacy_prefix):
            ]: value
            for (
                key,
                value,
            ) in state.items()
            if key.startswith(
                legacy_prefix
            )
        }

        if selected:
            return selected

    return {}


def load_residual_adapter(
    model: WhisperForConditionalGeneration,
    adapter_directory: Path,
) -> None:
    saved_config = (
        load_saved_adapter_config(
            adapter_directory
        )
    )

    validate_saved_adapter_architecture(
        saved_config
    )

    state = (
        load_adapter_state_file(
            adapter_directory
            / "adapter_state.pt"
        )
    )

    missing: list[
        str
    ] = []

    for (
        module_name,
        layer_index,
        adapter,
    ) in adapter_modules_for_state(
        model
    ):
        layer_state = (
            state_for_adapter(
                state,
                module_name,
                layer_index,
            )
        )

        if not layer_state:
            missing.append(
                f"{module_name} "
                f"layer {layer_index}"
            )
            continue

        try:
            adapter.load_state_dict(
                layer_state,
                strict=True,
            )

        except RuntimeError as error:
            raise RuntimeError(
                "Adapter weight shapes "
                "or names do not match "
                f"for {module_name} "
                f"layer {layer_index}.\n"
                "Underlying error:\n"
                f"{error}"
            ) from error

    if missing:
        sample_keys = list(
            state.keys()
        )[:20]

        raise RuntimeError(
            "Could not find weights for: "
            + ", ".join(
                missing
            )
            + "\nFirst saved state keys:\n"
            + "\n".join(
                sample_keys
            )
        )


# Miami data

MIAMI_REQUIRED_COLUMNS = (
    "example_id",
    "conversation_id",
    "speaker",
    "speaker_code",
    "split",
    "audio_path",
    "transcript",
)


def load_miami_training_rows(
    path: Path,
) -> list[
    dict[
        str,
        str,
    ]
]:
    if not path.exists():
        raise FileNotFoundError(
            "Miami training manifest "
            "is missing:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Miami training manifest "
                "has no header:\n"
                f"{path}"
            )

        missing = (
            set(
                MIAMI_REQUIRED_COLUMNS
            )
            - set(
                reader.fieldnames
            )
        )

        if missing:
            raise ValueError(
                "Miami training manifest "
                "is missing columns: "
                f"{sorted(missing)}"
            )

        rows = [
            dict(row)
            for row in reader
        ]

    if not rows:
        raise RuntimeError(
            "Miami training manifest "
            "is empty."
        )

    prepared: list[
        dict[
            str,
            str,
        ]
    ] = []

    for row in rows:
        split = str(
            row.get(
                "split",
                "",
            )
        ).strip().casefold()

        if split != "train":
            continue

        prepared_row = dict(
            row
        )

        # WhisperManifestDataset expects
        # the common transcript/task fields.

        prepared_row["task"] = (
            "miami"
        )

        prepared.append(
            prepared_row
        )

    if not prepared:
        raise RuntimeError(
            "No Miami training rows "
            "were found."
        )

    return prepared


def sample_joint_miami_rows(
    miami_rows: list[
        dict[
            str,
            str,
        ]
    ],
    *,
    target_count: int,
    seed: int,
) -> list[
    dict[
        str,
        str,
    ]
]:
    if target_count <= 0:
        raise ValueError(
            "target_count must "
            "be positive."
        )

    if target_count > len(
        miami_rows
    ):
        raise ValueError(
            "Requested more Miami rows "
            "than exist in the training "
            "manifest: "
            f"{target_count} > "
            f"{len(miami_rows)}"
        )

    rng = random.Random(
        seed
    )

    indices = list(
        range(
            len(miami_rows)
        )
    )

    rng.shuffle(
        indices
    )

    selected = [
        dict(
            miami_rows[index]
        )
        for index in (
            indices[
                :target_count
            ]
        )
    ]

    return selected


def build_joint_training_rows(
    *,
    llanito_rows: list[
        dict[
            str,
            str,
        ]
    ],
    miami_rows: list[
        dict[
            str,
            str,
        ]
    ],
    seed: int,
) -> tuple[
    list[
        dict[
            str,
            str,
        ]
    ],
    list[
        dict[
            str,
            str,
        ]
    ],
]:
    # Equal source contribution:
    # N Miami = N Llanito.

    sampled_miami = (
        sample_joint_miami_rows(
            miami_rows,
            target_count=(
                len(
                    llanito_rows
                )
            ),
            seed=seed,
        )
    )

    joint = [
        dict(row)
        for row in llanito_rows
    ]

    joint.extend(
        sampled_miami
    )

    rng = random.Random(
        seed + 5000
    )

    rng.shuffle(
        joint
    )

    return (
        joint,
        sampled_miami,
    )


# Fold helpers

def clear_selected_fold_outputs(
    model_directory: Path,
    output_directory: Path,
) -> None:
    if not (
        CLEAR_SELECTED_FOLD_BEFORE_TRAINING
    ):
        return

    if RESUME_FROM_LATEST_CHECKPOINT:
        return

    if model_directory.exists():
        print(
            "Removing previous "
            "fold model: "
            f"{model_directory}"
        )

        shutil.rmtree(
            model_directory
        )

    if output_directory.exists():
        print(
            "Removing previous fold "
            "training outputs: "
            f"{output_directory}"
        )

        shutil.rmtree(
            output_directory
        )


def load_saved_validation_metrics(
    fold_number: int,
) -> dict[
    str,
    Any,
] | None:
    path = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "final_validation_metrics.json"
    )

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return json.load(
            input_file
        )


def rebuild_training_cross_validation_summary(
) -> None:
    fold_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    missing_folds: list[
        int
    ] = []

    for fold_number in FOLD_NUMBERS:
        metrics = (
            load_saved_validation_metrics(
                fold_number
            )
        )

        if metrics is None:
            missing_folds.append(
                fold_number
            )
            continue

        fold_rows.append(
            {
                "fold": (
                    fold_number
                ),
                "normalised_wer": (
                    metrics.get(
                        "final_validation_normalised_wer"
                    )
                ),
                "raw_wer": (
                    metrics.get(
                        "final_validation_raw_wer"
                    )
                ),
            }
        )

    normalised_wers = [
        float(
            row[
                "normalised_wer"
            ]
        )
        for row in fold_rows
        if row[
            "normalised_wer"
        ] is not None
    ]

    aggregate = {
        "experiment": (
            EXPERIMENT_NAME
        ),
        "training_data": (
            "joint_miami_llanito"
        ),
        "validation_data": (
            "llanito_only"
        ),
        "adapter_module": (
            ADAPTER_MODULE
        ),
        "adapter_position": (
            ADAPTER_POSITION
        ),
        "adapter_layers": (
            ADAPTER_LAYERS
        ),
        "folds": (
            fold_rows
        ),
        "missing_folds": (
            missing_folds
        ),
        "number_of_available_folds": (
            len(
                fold_rows
            )
        ),
        "number_of_expected_folds": (
            len(
                FOLD_NUMBERS
            )
        ),
        "normalised_wer_macro_mean": (
            sum(
                normalised_wers
            )
            / len(
                normalised_wers
            )
            if normalised_wers
            else None
        ),
    }

    save_json(
        OUTPUT_ROOT
        / "cross_validation_summary.json",
        aggregate,
    )

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        SUMMARY_ROOT
        / "cross_validation_summary.json",
        aggregate,
    )


# Training

def unwrap_for_adapter_save(
    model: nn.Module,
) -> WhisperForConditionalGeneration:
    current = model

    while hasattr(
        current,
        "module",
    ):
        current = (
            current.module
        )

    if not isinstance(
        current,
        WhisperForConditionalGeneration,
    ):
        raise TypeError(
            "Could not unwrap Trainer "
            "model to "
            "WhisperForConditionalGeneration. "
            f"Got {type(current)!r}"
        )

    return current


def train_one_fold(
    fold_directory: Path,
    task_mappings: dict[
        str,
        dict[
            str,
            str,
        ],
    ],
    all_miami_rows: list[
        dict[
            str,
            str,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    fold_number = (
        extract_fold_number(
            fold_directory
        )
    )

    fold_seed = (
        RANDOM_SEED
        + fold_number
    )

    set_seed(
        fold_seed
    )

    # Llanito source data.

    llanito_train_rows = (
        load_manifest(
            fold_directory
            / "train.csv"
        )
    )

    validation_rows = (
        load_manifest(
            fold_directory
            / "validation.csv"
        )
    )

    validate_manifest_disjointness(
        llanito_train_rows,
        validation_rows,
    )

    # Apply any requested Llanito
    # training subset BEFORE choosing
    # the matching Miami count.

    llanito_train_rows = (
        select_subset(
            llanito_train_rows,
            MAX_TRAIN_RECORDINGS,
            fold_seed,
        )
    )

    validation_rows = (
        select_subset(
            validation_rows,
            MAX_VALIDATION_RECORDINGS,
            fold_seed + 1000,
        )
    )

    # Equal Miami/Llanito contribution.

    (
        train_rows,
        sampled_miami_rows,
    ) = build_joint_training_rows(
        llanito_rows=(
            llanito_train_rows
        ),
        miami_rows=(
            all_miami_rows
        ),
        seed=(
            fold_seed + 2000
        ),
    )

    model_directory = (
        MODEL_ROOT
        / f"fold_{fold_number}"
    )

    adapter_directory = (
        model_directory
        / "adapter"
    )

    checkpoint_directory = (
        model_directory
        / "checkpoints"
    )

    output_directory = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
    )

    metrics_path = (
        output_directory
        / "training_metrics.csv"
    )

    if is_main_process():
        clear_selected_fold_outputs(
            model_directory,
            output_directory,
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

    if is_main_process():
        print()
        print("#" * 80)
        print(
            f"STARTING "
            f"{EXPERIMENT_NAME} "
            f"FOLD {fold_number}"
        )
        print("#" * 80)

        print(
            "Training source: "
            "JOINT MIAMI + LLANITO"
        )

        print(
            "Validation source: "
            "LLANITO ONLY"
        )

        print(
            "Llanito training "
            "recordings: "
            f"{len(llanito_train_rows)}"
        )

        print(
            "Miami training "
            "recordings sampled: "
            f"{len(sampled_miami_rows)}"
        )

        print(
            "Total joint training "
            "recordings: "
            f"{len(train_rows)}"
        )

        print(
            "Llanito validation "
            "recordings: "
            f"{len(validation_rows)}"
        )

        print(
            f"Adapter module: "
            f"{ADAPTER_MODULE}"
        )

        print(
            f"Adapter position: "
            f"{ADAPTER_POSITION}"
        )

        print(
            f"Adapter layers: "
            f"{ADAPTER_LAYERS}"
        )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
            language=LANGUAGE,
            task=TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = (
        build_residual_adapter_model()
    )

    parameters = (
        parameter_summary(
            model
        )
    )

    if is_main_process():
        print(
            f"Parameter summary: "
            f"{parameters}"
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

    collator = (
        WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=(
                model.config
                .decoder_start_token_id
            ),
        )
    )

    training_args = (
        build_training_arguments(
            checkpoint_directory,
            fold_seed,
            epochs=(
                NUMBER_OF_EPOCHS
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            warmup_steps=(
                WARMUP_STEPS
            ),
            train_batch_size=(
                TRAIN_BATCH_SIZE_PER_GPU
            ),
            validation_batch_size=(
                VALIDATION_BATCH_SIZE_PER_GPU
            ),
            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
            use_fp16=(
                USE_FP16
            ),
            use_bf16=(
                USE_BF16
            ),
            use_gradient_checkpointing=(
                USE_GRADIENT_CHECKPOINTING
            ),
            logging_steps=(
                LOGGING_STEPS
            ),
            save_total_limit=(
                SAVE_TOTAL_LIMIT
            ),
            generation_max_length=(
                GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                GENERATION_NUM_BEAMS
            ),
            dataloader_workers=(
                DATALOADER_WORKERS
            ),
        )
    )

    trainer = (
        make_trainer(
            model=model,
            training_args=(
                training_args
            ),
            processor=processor,
            train_dataset=(
                train_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            collator=(
                collator
            ),
            compute_metrics=(
                build_compute_metrics(
                    processor,
                    validation_rows,
                    task_mappings,
                )
            ),
            callbacks=[
                ValidationMetricsCallback(
                    EXPERIMENT_NAME,
                    fold_number,
                    metrics_path,
                )
            ],
        )
    )

    checkpoint = None

    if RESUME_FROM_LATEST_CHECKPOINT:
        checkpoint = (
            get_last_checkpoint(
                str(
                    checkpoint_directory
                )
            )
        )

        if is_main_process():
            print(
                "Resume checkpoint: "
                f"{checkpoint}"
            )

    train_result = (
        trainer.train(
            resume_from_checkpoint=(
                checkpoint
            )
        )
    )

    final_validation_metrics = (
        trainer.evaluate(
            eval_dataset=(
                validation_dataset
            ),
            metric_key_prefix=(
                "final_validation"
            ),
        )
    )

    if trainer.is_world_process_zero():
        save_model = (
            unwrap_for_adapter_save(
                trainer.model
            )
        )

        save_residual_adapter(
            save_model,
            processor,
            adapter_directory,
        )

        summary = {
            "experiment": (
                EXPERIMENT_NAME
            ),
            "fold": (
                fold_number
            ),
            "base_model": (
                BASE_MODEL
            ),
            "training_data": (
                "joint_miami_llanito"
            ),
            "validation_data": (
                "llanito_only"
            ),
            "adapter_directory": (
                str(
                    adapter_directory
                )
            ),
            "adapter": {
                "module": (
                    ADAPTER_MODULE
                ),
                "position": (
                    ADAPTER_POSITION
                ),
                "layers": (
                    ADAPTER_LAYERS
                ),
                "bottleneck": (
                    ADAPTER_BOTTLENECK
                ),
                "dropout": (
                    ADAPTER_DROPOUT
                ),
            },
            "llanito_train_manifest": (
                str(
                    fold_directory
                    / "train.csv"
                )
            ),
            "miami_train_manifest": (
                str(
                    MIAMI_TRAIN_MANIFEST
                )
            ),
            "validation_manifest": (
                str(
                    fold_directory
                    / "validation.csv"
                )
            ),
            "llanito_training_recordings": (
                len(
                    llanito_train_rows
                )
            ),
            "miami_training_recordings": (
                len(
                    sampled_miami_rows
                )
            ),
            "total_training_recordings": (
                len(
                    train_rows
                )
            ),
            "validation_recordings": (
                len(
                    validation_rows
                )
            ),
            "parameters": (
                parameters
            ),
            "epochs_requested": (
                NUMBER_OF_EPOCHS
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "effective_batch_size": (
                TRAIN_BATCH_SIZE_PER_GPU
                * GRADIENT_ACCUMULATION_STEPS
                * trainer.args.world_size
            ),
            "best_checkpoint": (
                trainer.state
                .best_model_checkpoint
            ),
            "best_metric": (
                trainer.state
                .best_metric
            ),
            "train_metrics": (
                train_result.metrics
            ),
            "final_validation_metrics": (
                final_validation_metrics
            ),
        }

        save_json(
            output_directory
            / "training_summary.json",
            summary,
        )

        save_json(
            output_directory
            / "final_validation_metrics.json",
            final_validation_metrics,
        )

        print(
            "Residual adapter "
            "saved to: "
            f"{adapter_directory}"
        )

    distributed_barrier()

    result = {
        "fold": (
            fold_number
        ),
        "normalised_wer": (
            final_validation_metrics
            .get(
                "final_validation_normalised_wer"
            )
        ),
    }

    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    distributed_barrier()

    return result


# Main

def main() -> None:
    validate_cuda_environment(
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
    )

    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(
            "Llanito processed root "
            "does not exist:\n"
            f"{PROCESSED_ROOT}"
        )

    if not MIAMI_TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            "Miami training manifest "
            "does not exist:\n"
            f"{MIAMI_TRAIN_MANIFEST}"
        )

    fixed_test = (
        PROCESSED_ROOT
        / "fixed_test.csv"
    )

    if not fixed_test.exists():
        raise FileNotFoundError(
            "Authoritative Llanito "
            "fixed test is missing:\n"
            f"{fixed_test}"
        )

    fold_directories = (
        discover_fold_directories(
            PROCESSED_ROOT,
            RUN_FOLDS,
        )
    )

    if not fold_directories:
        raise RuntimeError(
            "No Llanito fold "
            "directories were discovered."
        )

    task_mappings = (
        load_task_mappings(
            NORMALISATION_FILE
        )
    )

    # Miami is loaded once and
    # independently sampled for each fold.

    all_miami_rows = (
        load_miami_training_rows(
            MIAMI_TRAIN_MANIFEST
        )
    )

    if is_main_process():
        MODEL_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_folds = [
            extract_fold_number(
                path
            )
            for path
            in fold_directories
        ]

        print()
        print("=" * 80)
        print(
            "JOINT MIAMI-LLANITO "
            "RESIDUAL ADAPTER TRAINING"
        )
        print("=" * 80)

        print(
            f"Experiment: "
            f"{EXPERIMENT_NAME}"
        )

        print(
            f"Adapter module: "
            f"{ADAPTER_MODULE}"
        )

        print(
            f"Adapter position: "
            f"{ADAPTER_POSITION}"
        )

        print(
            f"Adapter layers: "
            f"{ADAPTER_LAYERS}"
        )

        print(
            "Llanito fold root: "
            f"{PROCESSED_ROOT}"
        )

        print(
            "Miami training manifest: "
            f"{MIAMI_TRAIN_MANIFEST}"
        )

        print(
            "Available Miami "
            "training rows: "
            f"{len(all_miami_rows)}"
        )

        print(
            "Joint ratio: "
            "1 Miami : 1 Llanito"
        )

        print(
            "Checkpoint selection: "
            "LLANITO VALIDATION ONLY"
        )

        print(
            "Requested folds: "
            f"{RUN_FOLDS if RUN_FOLDS is not None else 'ALL'}"
        )

        print(
            f"Discovered folds: "
            f"{selected_folds}"
        )

    distributed_barrier()

    fold_summaries = [
        train_one_fold(
            fold_directory,
            task_mappings,
            all_miami_rows,
        )
        for fold_directory
        in fold_directories
    ]

    if is_main_process():
        rebuild_training_cross_validation_summary()

        print()
        print("=" * 80)
        print(
            f"{EXPERIMENT_NAME} "
            "TRAINING COMPLETE"
        )
        print("=" * 80)

        print(
            "Trained folds this run: "
            f"{[
                item['fold']
                for item
                in fold_summaries
            ]}"
        )

    distributed_barrier()


if __name__ == "__main__":
    main()