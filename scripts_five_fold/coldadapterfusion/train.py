from __future__ import annotations

import csv
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.coldadapterfusion.config import (
    ADAPTER_BOTTLENECK,
    ADAPTER_DROPOUT,
    ADAPTER_LAYERS,
    ADAPTER_POSITION,
    BASE_MODEL,
    BASELINE_MODEL_NAME,
    DATALOADER_WORKERS,
    EXPERIMENT_NAME,
    RUN_EXPERIMENTS,
    resolve_experiment,
    FOLD_NUMBERS,
    FORCE_RETRAIN_FUSION,
    FORCE_RETRAIN_LLANITO_ADAPTER,
    FORCE_RETRAIN_MIAMI_ADAPTER,
    FUSION_ATTENTION_DIM,
    FUSION_DROPOUT,
    FUSION_EPOCHS,
    FUSION_LEARNING_RATE,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LLANITO_ADAPTER_EPOCHS,
    LLANITO_ADAPTER_LEARNING_RATE,
    LLANITO_PROCESSED_ROOT,
    LOGGING_STEPS,
    MAX_LLANITO_TRAIN_RECORDINGS,
    MAX_LLANITO_VALIDATION_RECORDINGS,
    MAX_MIAMI_TRAIN_RECORDINGS,
    MAX_MIAMI_VALIDATION_RECORDINGS,
    MIAMI_EPOCHS,
    MIAMI_LEARNING_RATE,
    MIAMI_TRAIN_MANIFEST,
    MIAMI_VALIDATION_MANIFEST,
    MODEL_ROOT,
    NORMALISATION_FILE,
    OUTPUT_ROOT,
    RANDOM_SEED,
    RUN_FOLDS,
    SAVE_TOTAL_LIMIT,
    SUMMARY_ROOT,
    TASK,
    TRAIN_BATCH_SIZE_PER_GPU,
    TRAIN_FUSION_IF_MISSING,
    TRAIN_LLANITO_ADAPTER_IF_MISSING,
    TRAIN_MIAMI_ADAPTER_IF_MISSING,
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
    distributed_barrier,
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


# Experiment switching

def apply_experiment_variant(
    experiment_key: str,
) -> None:
    global BASE_MODEL
    global BASELINE_MODEL_NAME
    global EXPERIMENT_NAME
    global MODEL_ROOT
    global OUTPUT_ROOT
    global SUMMARY_ROOT

    settings = resolve_experiment(
        experiment_key
    )

    BASE_MODEL = settings[
        "base_model"
    ]

    BASELINE_MODEL_NAME = settings[
        "baseline_model_name"
    ]

    EXPERIMENT_NAME = settings[
        "experiment_name"
    ]

    MODEL_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold" / "models"
        / EXPERIMENT_NAME
    )

    OUTPUT_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold"
        / EXPERIMENT_NAME
    )

    SUMMARY_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold"
        / "summary"
        / EXPERIMENT_NAME
    )



# Utilities

def unwrap_model(
    model: nn.Module,
) -> WhisperForConditionalGeneration:
    current = model

    while hasattr(
        current,
        "module",
    ):
        current = current.module

    if not isinstance(
        current,
        WhisperForConditionalGeneration,
    ):
        raise TypeError(
            "Expected WhisperForConditionalGeneration after unwrapping, "
            f"got {type(current)!r}"
        )

    return current


def requested_folds() -> list[int]:
    if RUN_FOLDS is None:
        return list(
            FOLD_NUMBERS
        )

    folds = sorted(
        set(
            RUN_FOLDS
        )
    )

    invalid = [
        fold
        for fold in folds
        if fold not in FOLD_NUMBERS
    ]

    if invalid:
        raise ValueError(
            f"Invalid RUN_FOLDS values: {invalid}"
        )

    return folds


def fold_directory(
    fold_number: int,
) -> Path:
    path = (
        LLANITO_PROCESSED_ROOT
        / f"fold_{fold_number}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Llanito fold directory:\n{path}"
        )

    return path


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        return list(
            csv.DictReader(
                input_file
            )
        )


def prepare_miami_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Miami manifest does not exist:\n{path}"
        )

    rows = read_csv_rows(
        path
    )

    if not rows:
        raise RuntimeError(
            f"Miami manifest is empty:\n{path}"
        )

    # The shared project dataset loader expects audio_path and transcript.
    required = {
        "audio_path",
        "transcript",
    }

    missing = (
        required
        - set(
            rows[0].keys()
        )
    )

    if missing:
        raise ValueError(
            "Miami manifest is missing required columns "
            f"{sorted(missing)}.\n"
            f"Columns found: {list(rows[0].keys())}"
        )

    # Some shared metric code expects a task value. Miami does not need
    # Llanito-specific sentence/passage normalisation during its own stage,
    # so a neutral synthetic task label is added only when absent.
    for row in rows:
        if not row.get(
            "task"
        ):
            row["task"] = "miami"

    return rows


def adapter_exists(
    directory: Path,
) -> bool:
    return (
        (
            directory
            / "adapter_config.json"
        ).exists()
        and (
            directory
            / "adapter_state.pt"
        ).exists()
    )


def fusion_exists(
    directory: Path,
) -> bool:
    return (
        (
            directory
            / "fusion_config.json"
        ).exists()
        and (
            directory
            / "fusion_state.pt"
        ).exists()
    )


def clear_stage_paths(
    *paths: Path,
) -> None:
    if not is_main_process():
        return

    for path in paths:
        if path.exists():
            print(
                "Removing stale stage path: "
                f"{path}"
            )
            shutil.rmtree(
                path
            )


# Expert adapter

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

        # Start as exact zero correction.
        nn.init.zeros_(
            self.up.weight
        )
        nn.init.zeros_(
            self.up.bias
        )

    def correction(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        value = self.down(
            hidden_states
        )

        value = self.activation(
            value
        )

        value = self.dropout(
            value
        )

        return self.up(
            value
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return (
            hidden_states
            + self.correction(
                hidden_states
            )
        )


class SingleAdapterWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer

        self.residual_adapter = (
            ResidualAdapter(
                hidden_size=hidden_size,
                bottleneck_size=(
                    ADAPTER_BOTTLENECK
                ),
                dropout=(
                    ADAPTER_DROPOUT
                ),
            )
        )

    def _adapt_output(
        self,
        output: Any,
    ) -> Any:
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
                    "Whisper encoder layer returned an empty tuple."
                )

            return (
                self.residual_adapter(
                    output[0]
                ),
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
            "Unexpected Whisper encoder layer output type: "
            f"{type(output)!r}"
        )

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._adapt_output(
            self.base_layer(
                *args,
                **kwargs,
            )
        )


def build_single_adapter_model(
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

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(
        model.config.d_model
    )

    selected = set(
        ADAPTER_LAYERS
    )

    wrapped: list[nn.Module] = []

    for layer_index, layer in enumerate(
        model.model.encoder.layers
    ):
        if layer_index in selected:
            wrapped.append(
                SingleAdapterWrappedLayer(
                    layer,
                    hidden_size,
                )
            )
        else:
            wrapped.append(
                layer
            )

    model.model.encoder.layers = (
        nn.ModuleList(
            wrapped
        )
    )

    for (
        _,
        adapter,
    ) in single_adapter_modules(
        model
    ):
        for parameter in adapter.parameters():
            parameter.requires_grad = True

    verify_only_adapter_trainable(
        model
    )

    return model


def single_adapter_modules(
    model: WhisperForConditionalGeneration,
) -> list[
    tuple[
        int,
        ResidualAdapter,
    ]
]:
    result: list[
        tuple[
            int,
            ResidualAdapter,
        ]
    ] = []

    for layer_index in ADAPTER_LAYERS:
        layer = (
            model.model.encoder.layers[
                layer_index
            ]
        )

        if not isinstance(
            layer,
            SingleAdapterWrappedLayer,
        ):
            raise TypeError(
                "Expected SingleAdapterWrappedLayer at "
                f"encoder layer {layer_index}."
            )

        result.append(
            (
                layer_index,
                layer.residual_adapter,
            )
        )

    return result


def verify_only_adapter_trainable(
    model: nn.Module,
) -> None:
    trainable = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable adapter parameters."
        )

    unexpected = [
        name
        for name in trainable
        if ".residual_adapter." not in name
    ]

    if unexpected:
        raise RuntimeError(
            "Non-adapter parameters are trainable. "
            f"Examples: {unexpected[:10]}"
        )


def adapter_state_dict(
    model: WhisperForConditionalGeneration,
) -> dict[str, torch.Tensor]:
    state: dict[
        str,
        torch.Tensor,
    ] = {}

    for (
        layer_index,
        adapter,
    ) in single_adapter_modules(
        model
    ):
        for key, value in (
            adapter.state_dict().items()
        ):
            state[
                f"layer_{layer_index}.{key}"
            ] = (
                value.detach()
                .cpu()
                .clone()
            )

    return state


def save_adapter(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    directory: Path,
    expert_name: str,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        adapter_state_dict(
            model
        ),
        directory
        / "adapter_state.pt",
    )

    save_json(
        directory
        / "adapter_config.json",
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "expert_name": expert_name,
            "base_model": BASE_MODEL,
            "adapter_position": ADAPTER_POSITION,
            "adapter_layers": ADAPTER_LAYERS,
            "bottleneck": ADAPTER_BOTTLENECK,
            "dropout": ADAPTER_DROPOUT,
        },
    )

    processor.save_pretrained(
        directory
    )


def load_adapter_state(
    directory: Path,
) -> dict[str, torch.Tensor]:
    path = (
        directory
        / "adapter_state.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing adapter state:\n{path}"
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

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "Adapter state must be a dictionary."
        )

    return state


def validate_adapter_config(
    directory: Path,
) -> None:
    path = (
        directory
        / "adapter_config.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing adapter config:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        saved = json.load(
            input_file
        )

    if saved.get(
        "base_model"
    ) != BASE_MODEL:
        raise ValueError(
            "Adapter base-model mismatch."
        )

    if [
        int(value)
        for value in saved.get(
            "adapter_layers",
            [],
        )
    ] != ADAPTER_LAYERS:
        raise ValueError(
            "Adapter layer mismatch."
        )

    if int(
        saved.get(
            "bottleneck",
            -1,
        )
    ) != ADAPTER_BOTTLENECK:
        raise ValueError(
            "Adapter bottleneck mismatch."
        )


def load_adapter_into_single_model(
    model: WhisperForConditionalGeneration,
    directory: Path,
) -> None:
    validate_adapter_config(
        directory
    )

    state = load_adapter_state(
        directory
    )

    for (
        layer_index,
        adapter,
    ) in single_adapter_modules(
        model
    ):
        prefix = (
            f"layer_{layer_index}."
        )

        layer_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(
                prefix
            )
        }

        if not layer_state:
            raise RuntimeError(
                "No saved weights found for "
                f"adapter layer {layer_index}."
            )

        adapter.load_state_dict(
            layer_state,
            strict=True,
        )


# Cold AdapterFusion

class ColdFusionRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.attention_dim = (
            FUSION_ATTENTION_DIM
        )

        self.query = nn.Linear(
            hidden_size,
            self.attention_dim,
            bias=False,
        )

        self.key = nn.Linear(
            hidden_size,
            self.attention_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(
            FUSION_DROPOUT
        )

        # One optional bias per expert. Both begin equally preferred.
        self.expert_bias = nn.Parameter(
            torch.zeros(2)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        miami_states: torch.Tensor,
        llanito_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        # Query comes from the original frozen Whisper representation.
        query = self.query(
            self.dropout(
                hidden_states
            )
        )

        # Both expert outputs are keys in the same learned space.
        expert_states = torch.stack(
            [
                miami_states,
                llanito_states,
            ],
            dim=-2,
        )

        keys = self.key(
            self.dropout(
                expert_states
            )
        )

        # Scaled dot-product attention over exactly two experts.
        scores = (
            query.unsqueeze(
                -2
            )
            * keys
        ).sum(
            dim=-1
        )

        scores = (
            scores
            / (
                float(
                    self.attention_dim
                )
                ** 0.5
            )
        )

        scores = (
            scores
            + self.expert_bias
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        # The adapted expert states themselves are the values.
        # Since the weights sum to one and each expert is h + delta,
        # this is equivalent to h plus a learned weighted correction.
        fused = (
            expert_states
            * weights.unsqueeze(
                -1
            )
        ).sum(
            dim=-2
        )

        return (
            fused,
            weights,
        )


class ColdFusionWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer

        self.miami_adapter = ResidualAdapter(
            hidden_size=hidden_size,
            bottleneck_size=(
                ADAPTER_BOTTLENECK
            ),
            dropout=(
                ADAPTER_DROPOUT
            ),
        )

        self.llanito_adapter = ResidualAdapter(
            hidden_size=hidden_size,
            bottleneck_size=(
                ADAPTER_BOTTLENECK
            ),
            dropout=(
                ADAPTER_DROPOUT
            ),
        )

        self.fusion = ColdFusionRouter(
            hidden_size
        )

        self.last_fusion_weights: (
            torch.Tensor
            | None
        ) = None

    def _fuse(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Both experts see the exact same frozen Whisper state.
        miami_states = self.miami_adapter(
            hidden_states
        )

        llanito_states = self.llanito_adapter(
            hidden_states
        )

        fused, weights = self.fusion(
            hidden_states,
            miami_states,
            llanito_states,
        )

        self.last_fusion_weights = (
            weights.detach()
        )

        return fused

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
            return self._fuse(
                output
            )

        if isinstance(
            output,
            tuple,
        ):
            if not output:
                raise RuntimeError(
                    "Whisper encoder layer returned an empty tuple."
                )

            return (
                self._fuse(
                    output[0]
                ),
                *output[1:],
            )

        if hasattr(
            output,
            "last_hidden_state",
        ):
            output.last_hidden_state = (
                self._fuse(
                    output.last_hidden_state
                )
            )
            return output

        raise TypeError(
            "Unexpected Whisper encoder layer output type: "
            f"{type(output)!r}"
        )


def build_cold_fusion_model(
    miami_adapter_directory: Path,
    llanito_adapter_directory: Path,
) -> WhisperForConditionalGeneration:
    validate_adapter_config(
        miami_adapter_directory
    )

    validate_adapter_config(
        llanito_adapter_directory
    )

    miami_state = load_adapter_state(
        miami_adapter_directory
    )

    llanito_state = load_adapter_state(
        llanito_adapter_directory
    )

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

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(
        model.config.d_model
    )

    selected = set(
        ADAPTER_LAYERS
    )

    wrapped: list[nn.Module] = []

    for layer_index, layer in enumerate(
        model.model.encoder.layers
    ):
        if layer_index not in selected:
            wrapped.append(
                layer
            )
            continue

        fused_layer = (
            ColdFusionWrappedLayer(
                layer,
                hidden_size,
            )
        )

        prefix = (
            f"layer_{layer_index}."
        )

        miami_layer_state = {
            key[len(prefix):]: value
            for key, value in miami_state.items()
            if key.startswith(
                prefix
            )
        }

        llanito_layer_state = {
            key[len(prefix):]: value
            for key, value in llanito_state.items()
            if key.startswith(
                prefix
            )
        }

        fused_layer.miami_adapter.load_state_dict(
            miami_layer_state,
            strict=True,
        )

        fused_layer.llanito_adapter.load_state_dict(
            llanito_layer_state,
            strict=True,
        )

        wrapped.append(
            fused_layer
        )

    model.model.encoder.layers = (
        nn.ModuleList(
            wrapped
        )
    )

    # Freeze Whisper and both expert adapters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only fusion/router parameters.
    for (
        _,
        fusion,
    ) in fusion_modules(
        model
    ):
        for parameter in fusion.parameters():
            parameter.requires_grad = True

    verify_only_fusion_trainable(
        model
    )

    return model


def fusion_modules(
    model: WhisperForConditionalGeneration,
) -> list[
    tuple[
        int,
        ColdFusionRouter,
    ]
]:
    result: list[
        tuple[
            int,
            ColdFusionRouter,
        ]
    ] = []

    for layer_index in ADAPTER_LAYERS:
        layer = (
            model.model.encoder.layers[
                layer_index
            ]
        )

        if not isinstance(
            layer,
            ColdFusionWrappedLayer,
        ):
            raise TypeError(
                "Expected ColdFusionWrappedLayer at "
                f"encoder layer {layer_index}."
            )

        result.append(
            (
                layer_index,
                layer.fusion,
            )
        )

    return result


def verify_only_fusion_trainable(
    model: nn.Module,
) -> None:
    trainable = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable fusion parameters."
        )

    unexpected = [
        name
        for name in trainable
        if ".fusion." not in name
    ]

    if unexpected:
        raise RuntimeError(
            "Cold fusion stage has unexpected trainable parameters. "
            f"Examples: {unexpected[:10]}"
        )


def fusion_state_dict(
    model: WhisperForConditionalGeneration,
) -> dict[str, torch.Tensor]:
    state: dict[
        str,
        torch.Tensor,
    ] = {}

    for (
        layer_index,
        fusion,
    ) in fusion_modules(
        model
    ):
        for key, value in (
            fusion.state_dict().items()
        ):
            state[
                f"layer_{layer_index}.{key}"
            ] = (
                value.detach()
                .cpu()
                .clone()
            )

    return state


def save_fusion(
    model: WhisperForConditionalGeneration,
    directory: Path,
    miami_adapter_directory: Path,
    llanito_adapter_directory: Path,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        fusion_state_dict(
            model
        ),
        directory
        / "fusion_state.pt",
    )

    save_json(
        directory
        / "fusion_config.json",
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "base_model": BASE_MODEL,
            "adapter_layers": ADAPTER_LAYERS,
            "adapter_bottleneck": ADAPTER_BOTTLENECK,
            "fusion_attention_dim": FUSION_ATTENTION_DIM,
            "fusion_dropout": FUSION_DROPOUT,
            "miami_adapter_directory": str(
                miami_adapter_directory
            ),
            "llanito_adapter_directory": str(
                llanito_adapter_directory
            ),
        },
    )


def load_fusion_into_model(
    model: WhisperForConditionalGeneration,
    directory: Path,
) -> None:
    config_path = (
        directory
        / "fusion_config.json"
    )

    state_path = (
        directory
        / "fusion_state.pt"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing fusion config:\n{config_path}"
        )

    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing fusion state:\n{state_path}"
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

    for (
        layer_index,
        fusion,
    ) in fusion_modules(
        model
    ):
        prefix = (
            f"layer_{layer_index}."
        )

        layer_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(
                prefix
            )
        }

        if not layer_state:
            raise RuntimeError(
                f"No fusion state for layer {layer_index}."
            )

        fusion.load_state_dict(
            layer_state,
            strict=True,
        )


# Miami metrics

def levenshtein_counts(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    previous = list(
        range(
            len(hypothesis) + 1
        )
    )

    for i, ref_token in enumerate(
        reference,
        start=1,
    ):
        current = [
            i
        ]

        for j, hyp_token in enumerate(
            hypothesis,
            start=1,
        ):
            substitution_cost = (
                0
                if ref_token == hyp_token
                else 1
            )

            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1]
                    + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def build_plain_wer_metrics(
    processor: WhisperProcessor,
) -> Callable[[Any], dict[str, float]]:
    def compute_metrics(
        prediction: Any,
    ) -> dict[str, float]:
        predictions = (
            prediction.predictions
        )

        labels = (
            prediction.label_ids
        )

        if isinstance(
            predictions,
            tuple,
        ):
            predictions = (
                predictions[0]
            )

        labels = np.array(
            labels,
            copy=True,
        )

        labels[
            labels == -100
        ] = (
            processor.tokenizer
            .pad_token_id
        )

        prediction_text = (
            processor.tokenizer
            .batch_decode(
                predictions,
                skip_special_tokens=True,
            )
        )

        reference_text = (
            processor.tokenizer
            .batch_decode(
                labels,
                skip_special_tokens=True,
            )
        )

        errors = 0
        reference_words = 0

        for reference, hypothesis in zip(
            reference_text,
            prediction_text,
        ):
            ref_tokens = (
                reference.strip()
                .lower()
                .split()
            )

            hyp_tokens = (
                hypothesis.strip()
                .lower()
                .split()
            )

            errors += levenshtein_counts(
                ref_tokens,
                hyp_tokens,
            )

            reference_words += len(
                ref_tokens
            )

        wer = (
            100.0
            * errors
            / reference_words
            if reference_words
            else 0.0
        )

        # Keep the same key expected by the shared training arguments.
        return {
            "wer": wer,
            "raw_wer": wer,
            "normalised_wer": wer,
        }

    return compute_metrics


# Generic stage trainer

def train_stage(
    *,
    stage_name: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    checkpoint_directory: Path,
    output_directory: Path,
    epochs: int,
    learning_rate: float,
    seed: int,
    compute_metrics: Callable[[Any], dict[str, float]],
    callback: ValidationMetricsCallback | None,
) -> tuple[
    WhisperForConditionalGeneration,
    dict[str, Any],
]:
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
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
            seed,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            train_batch_size=(
                TRAIN_BATCH_SIZE_PER_GPU
            ),
            validation_batch_size=(
                VALIDATION_BATCH_SIZE_PER_GPU
            ),
            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
            use_fp16=USE_FP16,
            use_bf16=USE_BF16,
            use_gradient_checkpointing=(
                USE_GRADIENT_CHECKPOINTING
            ),
            logging_steps=LOGGING_STEPS,
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

    callbacks = (
        [callback]
        if callback is not None
        else []
    )

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=(
            validation_dataset
        ),
        collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print()
    print("=" * 80)
    print(
        f"TRAINING STAGE: {stage_name}"
    )
    print("=" * 80)
    print(
        f"Train recordings: {len(train_rows)}"
    )
    print(
        f"Validation recordings: {len(validation_rows)}"
    )
    print(
        f"Parameters: {parameter_summary(model)}"
    )

    train_result = trainer.train()

    final_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix=(
            "final_validation"
        ),
    )

    result = {
        "train_metrics": train_result.metrics,
        "final_validation_metrics": (
            final_metrics
        ),
        "best_checkpoint": (
            trainer.state
            .best_model_checkpoint
        ),
        "best_metric": (
            trainer.state
            .best_metric
        ),
    }

    trained_model = unwrap_model(
        trainer.model
    )

    # Trainer may have restored the best checkpoint at the end.
    # Keep the live model object until the caller saves the small module.
    del trainer
    del train_dataset
    del validation_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        trained_model,
        result,
    )


# Stage 1: Miami expert

def miami_adapter_directory() -> Path:
    return (
        MODEL_ROOT
        / "miami_adapter"
    )


def train_miami_adapter_if_needed() -> Path:
    directory = (
        miami_adapter_directory()
    )

    should_train = (
        FORCE_RETRAIN_MIAMI_ADAPTER
        or not adapter_exists(
            directory
        )
    )

    if not should_train:
        print(
            "Reusing existing Miami adapter: "
            f"{directory}"
        )
        return directory

    if not TRAIN_MIAMI_ADAPTER_IF_MISSING:
        raise FileNotFoundError(
            "Miami adapter is missing and "
            "TRAIN_MIAMI_ADAPTER_IF_MISSING=False."
        )

    clear_stage_paths(
        directory,
        MODEL_ROOT
        / "miami_adapter_checkpoints",
        OUTPUT_ROOT
        / "miami_adapter",
    )

    distributed_barrier()

    seed = (
        RANDOM_SEED
        + 100
    )

    set_seed(
        seed
    )

    train_rows = prepare_miami_rows(
        MIAMI_TRAIN_MANIFEST
    )

    validation_rows = prepare_miami_rows(
        MIAMI_VALIDATION_MANIFEST
    )

    train_rows = select_subset(
        train_rows,
        MAX_MIAMI_TRAIN_RECORDINGS,
        seed,
    )

    validation_rows = select_subset(
        validation_rows,
        MAX_MIAMI_VALIDATION_RECORDINGS,
        seed + 1,
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
    BASELINE_MODEL_NAME,
            language=LANGUAGE,
            task=TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_single_adapter_model()

    output_directory = (
        OUTPUT_ROOT
        / "miami_adapter"
    )

    callback = ValidationMetricsCallback(
        EXPERIMENT_NAME
        + "_miami_adapter",
        0,
        output_directory
        / "training_metrics.csv",
    )

    model, metrics = train_stage(
        stage_name="MIAMI EXPERT ADAPTER",
        model=model,
        processor=processor,
        train_rows=train_rows,
        validation_rows=validation_rows,
        checkpoint_directory=(
            MODEL_ROOT
            / "miami_adapter_checkpoints"
        ),
        output_directory=output_directory,
        epochs=MIAMI_EPOCHS,
        learning_rate=MIAMI_LEARNING_RATE,
        seed=seed,
        compute_metrics=(
            build_compute_metrics(
                processor,
                validation_rows,
                {},
            )
        ),
        callback=callback,
    )

    if is_main_process():
        save_adapter(
            model,
            processor,
            directory,
            expert_name="miami",
        )

        save_json(
            output_directory
            / "training_summary.json",
            {
                "stage": "miami_adapter",
                "train_manifest": str(
                    MIAMI_TRAIN_MANIFEST
                ),
                "validation_manifest": str(
                    MIAMI_VALIDATION_MANIFEST
                ),
                "train_recordings": len(
                    train_rows
                ),
                "validation_recordings": len(
                    validation_rows
                ),
                **metrics,
            },
        )

    distributed_barrier()

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return directory


# Stage 2: fold-specific Llanito expert

def llanito_adapter_directory(
    fold_number: int,
) -> Path:
    return (
        MODEL_ROOT
        / f"fold_{fold_number}"
        / "llanito_adapter"
    )


def train_llanito_adapter_if_needed(
    fold_number: int,
    task_mappings: dict[
        str,
        dict[str, str],
    ],
) -> Path:
    directory = (
        llanito_adapter_directory(
            fold_number
        )
    )

    should_train = (
        FORCE_RETRAIN_LLANITO_ADAPTER
        or not adapter_exists(
            directory
        )
    )

    if not should_train:
        print(
            "Reusing existing Llanito adapter: "
            f"{directory}"
        )
        return directory

    if not TRAIN_LLANITO_ADAPTER_IF_MISSING:
        raise FileNotFoundError(
            f"Fold {fold_number} Llanito adapter is missing and "
            "TRAIN_LLANITO_ADAPTER_IF_MISSING=False."
        )

    clear_stage_paths(
        directory,
        MODEL_ROOT
        / f"fold_{fold_number}"
        / "llanito_adapter_checkpoints",
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "llanito_adapter",
    )

    distributed_barrier()

    fold_root = fold_directory(
        fold_number
    )

    seed = (
        RANDOM_SEED
        + fold_number
    )

    set_seed(
        seed
    )

    train_rows = load_manifest(
        fold_root
        / "train.csv"
    )

    validation_rows = load_manifest(
        fold_root
        / "validation.csv"
    )

    validate_manifest_disjointness(
        train_rows,
        validation_rows,
    )

    train_rows = select_subset(
        train_rows,
        MAX_LLANITO_TRAIN_RECORDINGS,
        seed,
    )

    validation_rows = select_subset(
        validation_rows,
        MAX_LLANITO_VALIDATION_RECORDINGS,
        seed + 1000,
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
    BASELINE_MODEL_NAME,
            language=LANGUAGE,
            task=TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_single_adapter_model()

    output_directory = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "llanito_adapter"
    )

    callback = ValidationMetricsCallback(
        EXPERIMENT_NAME
        + "_llanito_adapter",
        fold_number,
        output_directory
        / "training_metrics.csv",
    )

    model, metrics = train_stage(
        stage_name=(
            f"LLANITO EXPERT ADAPTER FOLD {fold_number}"
        ),
        model=model,
        processor=processor,
        train_rows=train_rows,
        validation_rows=validation_rows,
        checkpoint_directory=(
            MODEL_ROOT
            / f"fold_{fold_number}"
            / "llanito_adapter_checkpoints"
        ),
        output_directory=output_directory,
        epochs=LLANITO_ADAPTER_EPOCHS,
        learning_rate=(
            LLANITO_ADAPTER_LEARNING_RATE
        ),
        seed=seed,
        compute_metrics=(
            build_compute_metrics(
                processor,
                validation_rows,
                task_mappings,
            )
        ),
        callback=callback,
    )

    if is_main_process():
        save_adapter(
            model,
            processor,
            directory,
            expert_name=(
                f"llanito_fold_{fold_number}"
            ),
        )

        save_json(
            output_directory
            / "training_summary.json",
            {
                "stage": "llanito_adapter",
                "fold": fold_number,
                "train_manifest": str(
                    fold_root
                    / "train.csv"
                ),
                "validation_manifest": str(
                    fold_root
                    / "validation.csv"
                ),
                "train_recordings": len(
                    train_rows
                ),
                "validation_recordings": len(
                    validation_rows
                ),
                **metrics,
            },
        )

    distributed_barrier()

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return directory


# Stage 3: fusion-only training

def fusion_directory(
    fold_number: int,
) -> Path:
    return (
        MODEL_ROOT
        / f"fold_{fold_number}"
        / "fusion"
    )


def train_fusion_if_needed(
    fold_number: int,
    miami_directory: Path,
    llanito_directory: Path,
    task_mappings: dict[
        str,
        dict[str, str],
    ],
    *,
    dependency_retrained: bool = False,
) -> Path:
    directory = (
        fusion_directory(
            fold_number
        )
    )

    should_train = (
        FORCE_RETRAIN_FUSION
        or dependency_retrained
        or not fusion_exists(
            directory
        )
    )

    if not should_train:
        print(
            "Reusing existing cold fusion: "
            f"{directory}"
        )
        return directory

    if not TRAIN_FUSION_IF_MISSING:
        raise FileNotFoundError(
            f"Fold {fold_number} fusion is missing and "
            "TRAIN_FUSION_IF_MISSING=False."
        )

    clear_stage_paths(
        directory,
        MODEL_ROOT
        / f"fold_{fold_number}"
        / "fusion_checkpoints",
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "fusion",
    )

    distributed_barrier()

    fold_root = fold_directory(
        fold_number
    )

    seed = (
        RANDOM_SEED
        + 5000
        + fold_number
    )

    set_seed(
        seed
    )

    train_rows = load_manifest(
        fold_root
        / "train.csv"
    )

    validation_rows = load_manifest(
        fold_root
        / "validation.csv"
    )

    validate_manifest_disjointness(
        train_rows,
        validation_rows,
    )

    train_rows = select_subset(
        train_rows,
        MAX_LLANITO_TRAIN_RECORDINGS,
        seed,
    )

    validation_rows = select_subset(
        validation_rows,
        MAX_LLANITO_VALIDATION_RECORDINGS,
        seed + 1000,
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
    BASELINE_MODEL_NAME,
            language=LANGUAGE,
            task=TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_cold_fusion_model(
        miami_directory,
        llanito_directory,
    )

    output_directory = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "fusion"
    )

    callback = ValidationMetricsCallback(
        EXPERIMENT_NAME
        + "_fusion",
        fold_number,
        output_directory
        / "training_metrics.csv",
    )

    model, metrics = train_stage(
        stage_name=(
            f"COLD ADAPTER FUSION FOLD {fold_number}"
        ),
        model=model,
        processor=processor,
        train_rows=train_rows,
        validation_rows=validation_rows,
        checkpoint_directory=(
            MODEL_ROOT
            / f"fold_{fold_number}"
            / "fusion_checkpoints"
        ),
        output_directory=output_directory,
        epochs=FUSION_EPOCHS,
        learning_rate=(
            FUSION_LEARNING_RATE
        ),
        seed=seed,
        compute_metrics=(
            build_compute_metrics(
                processor,
                validation_rows,
                task_mappings,
            )
        ),
        callback=callback,
    )

    if is_main_process():
        save_fusion(
            model,
            directory,
            miami_directory,
            llanito_directory,
        )

        save_json(
            output_directory
            / "training_summary.json",
            {
                "stage": "cold_adapter_fusion",
                "fold": fold_number,
                "miami_adapter": str(
                    miami_directory
                ),
                "llanito_adapter": str(
                    llanito_directory
                ),
                "train_manifest": str(
                    fold_root
                    / "train.csv"
                ),
                "validation_manifest": str(
                    fold_root
                    / "validation.csv"
                ),
                "train_recordings": len(
                    train_rows
                ),
                "validation_recordings": len(
                    validation_rows
                ),
                "trainable_parameters": (
                    parameter_summary(
                        model
                    )
                ),
                **metrics,
            },
        )

        save_json(
            output_directory
            / "final_validation_metrics.json",
            metrics[
                "final_validation_metrics"
            ],
        )

    distributed_barrier()

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return directory


# Aggregate training summary

def rebuild_cross_validation_summary() -> None:
    fold_rows: list[
        dict[str, Any]
    ] = []

    missing: list[int] = []

    for fold_number in FOLD_NUMBERS:
        path = (
            OUTPUT_ROOT
            / f"fold_{fold_number}"
            / "fusion"
            / "final_validation_metrics.json"
        )

        if not path.exists():
            missing.append(
                fold_number
            )
            continue

        with path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            metrics = json.load(
                input_file
            )

        fold_rows.append(
            {
                "fold": fold_number,
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

    normalised = [
        float(
            row["normalised_wer"]
        )
        for row in fold_rows
        if row["normalised_wer"]
        is not None
    ]

    aggregate = {
        "experiment": EXPERIMENT_NAME,
        "architecture": (
            "parallel Miami + Llanito encoder adapters "
            "with frozen experts and trainable cold fusion"
        ),
        "adapter_layers": ADAPTER_LAYERS,
        "folds": fold_rows,
        "missing_folds": missing,
        "normalised_wer_macro_mean": (
            sum(normalised)
            / len(normalised)
            if normalised
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


# Main

def run_current_experiment() -> None:
    if not LLANITO_PROCESSED_ROOT.exists():
        raise FileNotFoundError(
            "Llanito processed root is missing:\n"
            f"{LLANITO_PROCESSED_ROOT}"
        )

    if not MIAMI_TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            "Miami training manifest is missing:\n"
            f"{MIAMI_TRAIN_MANIFEST}"
        )

    if not MIAMI_VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(
            "Miami validation manifest is missing:\n"
            f"{MIAMI_VALIDATION_MANIFEST}"
        )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_mappings = load_task_mappings(
        NORMALISATION_FILE
    )

    folds = requested_folds()

    if is_main_process():
        print("=" * 80)
        print(
            "COLD ADAPTER FUSION"
        )
        print("=" * 80)
        print(
            f"Experiment: {EXPERIMENT_NAME}"
        )
        print(
            f"Starting/base model: {BASE_MODEL}"
        )
        print(
            f"Miami data: {MIAMI_TRAIN_MANIFEST}"
        )
        print(
            f"Llanito folds: {LLANITO_PROCESSED_ROOT}"
        )
        print(
            f"Adapter layers: {ADAPTER_LAYERS}"
        )
        print(
            f"Folds: {folds}"
        )
        print()
        print(
            "Architecture:"
        )
        print(
            "Whisper h -> Miami adapter -------\\"
        )
        print(
            "                                  > learned fusion -> next layer"
        )
        print(
            "Whisper h -> Llanito adapter -----/"
        )
        print(
            "Whisper and both adapters are frozen during fusion training."
        )

    distributed_barrier()

    miami_will_retrain = (
        FORCE_RETRAIN_MIAMI_ADAPTER
        or not adapter_exists(
            miami_adapter_directory()
        )
    )

    miami_directory = (
        train_miami_adapter_if_needed()
    )

    for fold_number in folds:
        llanito_will_retrain = (
            FORCE_RETRAIN_LLANITO_ADAPTER
            or not adapter_exists(
                llanito_adapter_directory(
                    fold_number
                )
            )
        )

        llanito_directory = (
            train_llanito_adapter_if_needed(
                fold_number,
                task_mappings,
            )
        )

        train_fusion_if_needed(
            fold_number,
            miami_directory,
            llanito_directory,
            task_mappings,
            dependency_retrained=(
                miami_will_retrain
                or llanito_will_retrain
            ),
        )

    if is_main_process():
        rebuild_cross_validation_summary()

        print()
        print("=" * 80)
        print(
            "COLD ADAPTER FUSION TRAINING COMPLETE"
        )
        print("=" * 80)
        print(
            f"Models: {MODEL_ROOT}"
        )
        print(
            f"Outputs: {OUTPUT_ROOT}"
        )

    distributed_barrier()


def main() -> None:
    validate_cuda_environment(
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
    )

    for experiment_index, experiment_key in enumerate(
        RUN_EXPERIMENTS,
        start=1,
    ):
        apply_experiment_variant(
            experiment_key
        )

        if is_main_process():
            print()
            print("=" * 88)
            print(
                f"BASE-MODEL EXPERIMENT "
                f"{experiment_index}/{len(RUN_EXPERIMENTS)}"
            )
            print("=" * 88)
            print(
                f"Experiment key: {experiment_key}"
            )
            print(
                f"Experiment name: {EXPERIMENT_NAME}"
            )
            print(
                f"Starting/base model: {BASE_MODEL}"
            )
            print(
                f"Model root: {MODEL_ROOT}"
            )
            print(
                f"Output root: {OUTPUT_ROOT}"
            )

        distributed_barrier()

        run_current_experiment()

        distributed_barrier()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main_process():
        print()
        print("=" * 88)
        print("ALL COLD ADAPTER FUSION EXPERIMENTS COMPLETE")
        print("=" * 88)
        print(
            "Completed: "
            + ", ".join(
                RUN_EXPERIMENTS
            )
        )


if __name__ == "__main__":
    main()
