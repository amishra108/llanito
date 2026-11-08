from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import (
    PeftConfig,
    PeftModel,
)
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


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

            adapted_hidden_states = (
                self.residual_adapter(
                    output[0]
                )
            )

            return (
                adapted_hidden_states,
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


# Metadata

def load_model_metadata(
    model_root: Path,
) -> dict[str, Any]:
    metadata_path = (
        model_root
        / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Model metadata does not exist:\n"
            f"{metadata_path}"
        )

    with metadata_path.open(
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

    if "model_type" not in metadata:
        raise ValueError(
            "metadata.json is missing "
            "'model_type'."
        )

    return metadata


# Model-directory resolution

def resolve_whisper_model_directory(
    value: Path | str,
) -> Path | str:
    value_string = str(
        value
    )

    path = Path(
        value_string
    )

    nested_model_directory = (
        path
        / "model"
    )

    if (
        path.exists()
        and nested_model_directory.is_dir()
    ):
        return nested_model_directory

    if path.exists():
        return path

    if (
        value_string
        == "openai/whisper-medium"
    ):
        local_vanilla_directory = (
            PROJECT_ROOT
            / "models"
            / "vanilla_whisper_medium"
            / "model"
        )

        if (
            local_vanilla_directory
            .is_dir()
        ):
            return (
                local_vanilla_directory
            )

    return value_string


# Standard Whisper

def load_whisper_full(
    *,
    model_directory: Path,
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
]:
    if not model_directory.exists():
        raise FileNotFoundError(
            "Model directory does not exist:\n"
            f"{model_directory}"
        )

    processor = (
        WhisperProcessor
        .from_pretrained(
            model_directory
        )
    )

    model_kwargs: dict[
        str,
        Any,
    ] = {}

    if torch_dtype is not None:
        model_kwargs[
            "torch_dtype"
        ] = torch_dtype

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            model_directory,
            **model_kwargs,
        )
    )

    return (
        model,
        processor,
    )


# LoRA base resolution

def resolve_lora_base_model(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
) -> Path | str:
    architecture = metadata.get(
        "architecture"
    )

    if isinstance(
        architecture,
        dict,
    ):
        base_model_root = (
            architecture.get(
                "base_model_root"
            )
        )

        if base_model_root:
            return (
                resolve_whisper_model_directory(
                    base_model_root
                )
            )

    peft_configuration = (
        PeftConfig
        .from_pretrained(
            model_directory
        )
    )

    peft_base = (
        peft_configuration
        .base_model_name_or_path
    )

    if peft_base:
        return (
            resolve_whisper_model_directory(
                peft_base
            )
        )

    metadata_base = metadata.get(
        "base_model"
    )

    if metadata_base:
        return (
            resolve_whisper_model_directory(
                metadata_base
            )
        )

    raise RuntimeError(
        "Could not determine the base "
        "Whisper model for this LoRA model.\n"
        f"Model directory: "
        f"{model_directory}"
    )


# PEFT LoRA

def load_peft_lora(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
]:
    if not model_directory.exists():
        raise FileNotFoundError(
            "LoRA model directory does "
            "not exist:\n"
            f"{model_directory}"
        )

    adapter_config_path = (
        model_directory
        / "adapter_config.json"
    )

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            "LoRA model directory is missing "
            "adapter_config.json:\n"
            f"{adapter_config_path}"
        )

    adapter_weights_exist = any(
        (
            model_directory
            / filename
        ).exists()
        for filename in (
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )

    if not adapter_weights_exist:
        raise FileNotFoundError(
            "LoRA model directory contains "
            "no adapter weights:\n"
            f"{model_directory}"
        )

    base_source = (
        resolve_lora_base_model(
            model_directory=(
                model_directory
            ),
            metadata=metadata,
        )
    )

    model_kwargs: dict[
        str,
        Any,
    ] = {}

    if torch_dtype is not None:
        model_kwargs[
            "torch_dtype"
        ] = torch_dtype

    base_model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            base_source,
            **model_kwargs,
        )
    )

    model = (
        PeftModel
        .from_pretrained(
            base_model,
            model_directory,
            is_trainable=False,
        )
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            model_directory
        )
    )

    return (
        model,
        processor,
    )


# JSON helper

def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "JSON file does not exist:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        value = json.load(
            input_file
        )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Expected a JSON object in:\n"
            f"{path}"
        )

    return value


# Residual-adapter state

def load_residual_adapter_state(
    path: Path,
) -> dict[
    str,
    torch.Tensor,
]:
    if not path.exists():
        raise FileNotFoundError(
            "Residual-adapter state does "
            "not exist:\n"
            f"{path}"
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
            "adapter_state.pt must contain "
            "a dictionary."
        )

    invalid_values = [
        key
        for (
            key,
            value,
        ) in state.items()
        if not isinstance(
            value,
            torch.Tensor,
        )
    ]

    if invalid_values:
        raise TypeError(
            "Residual-adapter state contains "
            "non-tensor values.\n"
            f"Invalid keys: "
            f"{invalid_values[:20]}"
        )

    return state


# Residual base resolution

def resolve_residual_adapter_base_model(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
    adapter_config: dict[str, Any],
) -> Path | str:
    architecture = metadata.get(
        "architecture"
    )

    if isinstance(
        architecture,
        dict,
    ):
        base_model_root = (
            architecture.get(
                "base_model_root"
            )
        )

        if base_model_root:
            return (
                resolve_whisper_model_directory(
                    base_model_root
                )
            )

    config_base_root = (
        adapter_config.get(
            "base_model_root"
        )
    )

    if config_base_root:
        return (
            resolve_whisper_model_directory(
                config_base_root
            )
        )

    base_model = (
        adapter_config.get(
            "base_model"
        )
        or metadata.get(
            "base_model"
        )
    )

    if base_model:
        return (
            resolve_whisper_model_directory(
                base_model
            )
        )

    raise RuntimeError(
        "Could not determine the base "
        "Whisper model for this residual "
        "adapter.\n"
        f"Model directory: "
        f"{model_directory}"
    )


# Residual state extraction

def extract_residual_layer_state(
    *,
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
    candidate_prefixes = (
        (
            f"{module_name}."
            f"layer_{layer_index}."
        ),
        (
            f"layer_{layer_index}."
        ),
    )

    for prefix in (
        candidate_prefixes
    ):
        layer_state = {
            key[
                len(prefix):
            ]: value
            for (
                key,
                value,
            ) in state.items()
            if key.startswith(
                prefix
            )
        }

        if layer_state:
            return layer_state

    raise RuntimeError(
        "Could not find residual-adapter "
        "weights for:\n"
        f"  module={module_name}\n"
        f"  layer={layer_index}"
    )


# Module dtype and device

def get_module_dtype_and_device(
    module: nn.Module,
) -> tuple[
    torch.dtype,
    torch.device,
]:
    for parameter in (
        module.parameters()
    ):
        return (
            parameter.dtype,
            parameter.device,
        )

    for buffer in (
        module.buffers()
    ):
        return (
            buffer.dtype,
            buffer.device,
        )

    raise RuntimeError(
        "Could not determine dtype/device "
        "for Whisper layer."
    )


# Residual wrapping

def wrap_residual_adapter_layers(
    *,
    layers: nn.ModuleList,
    module_name: str,
    selected_layers: list[int],
    hidden_size: int,
    bottleneck: int,
    dropout: float,
    state: dict[
        str,
        torch.Tensor,
    ],
) -> nn.ModuleList:
    selected = set(
        selected_layers
    )

    if not selected:
        raise ValueError(
            "Residual-adapter layer list "
            "must not be empty."
        )

    invalid_layers = [
        layer_index
        for layer_index in selected
        if (
            layer_index < 0
            or layer_index >= len(
                layers
            )
        )
    ]

    if invalid_layers:
        raise IndexError(
            "Residual-adapter configuration "
            "contains invalid layer indices.\n"
            f"Module: {module_name}\n"
            f"Invalid: "
            f"{sorted(invalid_layers)}\n"
            f"Available layer count: "
            f"{len(layers)}"
        )

    wrapped_layers: list[
        nn.Module
    ] = []

    for (
        layer_index,
        base_layer,
    ) in enumerate(
        layers
    ):
        if (
            layer_index
            not in selected
        ):
            wrapped_layers.append(
                base_layer
            )

            continue

        layer_dtype, layer_device = (
            get_module_dtype_and_device(
                base_layer
            )
        )

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

        # Match the reconstructed adapter
        # to the Whisper layer precision
        # before loading its saved state.

        adapter = adapter.to(
            device=layer_device,
            dtype=layer_dtype,
        )

        adapter.load_state_dict(
            extract_residual_layer_state(
                state=state,
                module_name=(
                    module_name
                ),
                layer_index=(
                    layer_index
                ),
            ),
            strict=True,
        )

        # Explicit verification so a future
        # dtype mismatch fails during loading,
        # not halfway through evaluation.

        adapter_dtype, adapter_device = (
            get_module_dtype_and_device(
                adapter
            )
        )

        if (
            adapter_dtype
            != layer_dtype
        ):
            raise RuntimeError(
                "Residual-adapter dtype "
                "does not match Whisper.\n"
                f"Module: {module_name}\n"
                f"Layer: {layer_index}\n"
                f"Whisper dtype: "
                f"{layer_dtype}\n"
                f"Adapter dtype: "
                f"{adapter_dtype}"
            )

        if (
            adapter_device
            != layer_device
        ):
            raise RuntimeError(
                "Residual-adapter device "
                "does not match Whisper.\n"
                f"Module: {module_name}\n"
                f"Layer: {layer_index}\n"
                f"Whisper device: "
                f"{layer_device}\n"
                f"Adapter device: "
                f"{adapter_device}"
            )

        wrapped_layers.append(
            AdapterWrappedLayer(
                base_layer=(
                    base_layer
                ),
                adapter=adapter,
            )
        )

    return nn.ModuleList(
        wrapped_layers
    )


# Residual adapter

def load_residual_adapter(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
]:
    if not model_directory.exists():
        raise FileNotFoundError(
            "Residual-adapter model "
            "directory does not exist:\n"
            f"{model_directory}"
        )

    adapter_config_path = (
        model_directory
        / "adapter_config.json"
    )

    adapter_state_path = (
        model_directory
        / "adapter_state.pt"
    )

    adapter_config = (
        load_json_object(
            adapter_config_path
        )
    )

    state = (
        load_residual_adapter_state(
            adapter_state_path
        )
    )

    adapter_module = str(
        adapter_config.get(
            "adapter_module",
            "",
        )
    ).strip()

    if adapter_module not in (
        "encoder",
        "decoder",
        "encoder_decoder",
    ):
        raise ValueError(
            "Unsupported residual-adapter "
            "module: "
            f"{adapter_module!r}"
        )

    raw_layers = (
        adapter_config.get(
            "adapter_layers"
        )
    )

    if not isinstance(
        raw_layers,
        list,
    ):
        raise ValueError(
            "adapter_config.json must contain "
            "'adapter_layers' as a list."
        )

    adapter_layers = [
        int(
            value
        )
        for value in raw_layers
    ]

    if not adapter_layers:
        raise ValueError(
            "adapter_layers must not be empty."
        )

    if (
        len(adapter_layers)
        != len(
            set(
                adapter_layers
            )
        )
    ):
        raise ValueError(
            "adapter_layers contains "
            "duplicate layer indices."
        )

    bottleneck = int(
        adapter_config[
            "bottleneck"
        ]
    )

    dropout = float(
        adapter_config[
            "dropout"
        ]
    )

    if bottleneck <= 0:
        raise ValueError(
            "Residual-adapter bottleneck "
            "must be positive."
        )

    if not (
        0.0
        <= dropout
        < 1.0
    ):
        raise ValueError(
            "Residual-adapter dropout "
            "must satisfy 0 <= dropout < 1."
        )

    base_source = (
        resolve_residual_adapter_base_model(
            model_directory=(
                model_directory
            ),
            metadata=metadata,
            adapter_config=(
                adapter_config
            ),
        )
    )

    model_kwargs: dict[
        str,
        Any,
    ] = {}

    if torch_dtype is not None:
        model_kwargs[
            "torch_dtype"
        ] = torch_dtype

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            base_source,
            **model_kwargs,
        )
    )

    hidden_size = int(
        model.config.d_model
    )

    # Encoder reconstruction

    if adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        model.model.encoder.layers = (
            wrap_residual_adapter_layers(
                layers=(
                    model
                    .model
                    .encoder
                    .layers
                ),
                module_name="encoder",
                selected_layers=(
                    adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
                bottleneck=(
                    bottleneck
                ),
                dropout=(
                    dropout
                ),
                state=state,
            )
        )

    # Decoder reconstruction

    if adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        model.model.decoder.layers = (
            wrap_residual_adapter_layers(
                layers=(
                    model
                    .model
                    .decoder
                    .layers
                ),
                module_name="decoder",
                selected_layers=(
                    adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
                bottleneck=(
                    bottleneck
                ),
                dropout=(
                    dropout
                ),
                state=state,
            )
        )

    # Final inference model

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    model.eval()

    processor = (
        WhisperProcessor
        .from_pretrained(
            model_directory
        )
    )

    return (
        model,
        processor,
    )

# Stacked PEFT LoRA

def load_peft_lora_stacked(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
]:
    if not model_directory.exists():
        raise FileNotFoundError(
            "Stacked LoRA model directory "
            "does not exist:\n"
            f"{model_directory}"
        )

    adapter_config_path = (
        model_directory
        / "adapter_config.json"
    )

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            "Stacked LoRA target adapter is "
            "missing adapter_config.json:\n"
            f"{adapter_config_path}"
        )

    adapter_weights_exist = any(
        (
            model_directory
            / filename
        ).exists()
        for filename in (
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )

    if not adapter_weights_exist:
        raise FileNotFoundError(
            "Stacked LoRA target adapter "
            "weights do not exist:\n"
            f"{model_directory}"
        )

    architecture = metadata.get(
        "architecture"
    )

    if not isinstance(
        architecture,
        dict,
    ):
        raise ValueError(
            "Stacked LoRA metadata must "
            "contain an architecture object."
        )

    source_model_root_value = (
        architecture.get(
            "source_model_root"
        )
    )

    if not source_model_root_value:
        raise ValueError(
            "Stacked LoRA metadata is missing "
            "architecture.source_model_root."
        )

    source_model_root = Path(
        str(
            source_model_root_value
        )
    )

    if not source_model_root.exists():
        raise FileNotFoundError(
            "Stacked LoRA source model root "
            "does not exist:\n"
            f"{source_model_root}"
        )

    # Reconstruct the ORIGINAL source LoRA.

    (
        source_model,
        _,
        source_metadata,
    ) = load_model(
        model_root=(
            source_model_root
        ),
        torch_dtype=(
            torch_dtype
        ),
    )

    if (
        source_metadata.get(
            "model_type"
        )
        != "peft_lora"
    ):
        raise RuntimeError(
            "Stacked LoRA source must be "
            "an ordinary peft_lora model.\n"
            f"Source: {source_model_root}\n"
            f"Found model_type: "
            f"{source_metadata.get('model_type')!r}"
        )

    if not isinstance(
        source_model,
        PeftModel,
    ):
        raise RuntimeError(
            "Stacked LoRA source did not "
            "reconstruct as PeftModel."
        )

    # This reproduces the state used during
    # cold-transfer training:
    #
    # vanilla Whisper
    # + source-domain LoRA
    # -> merge source LoRA into Whisper.

    merged_source_model = (
        source_model
        .merge_and_unload()
    )

    # Now apply the saved TARGET LoRA on top
    # of those source-adapted Whisper weights.

    model = (
        PeftModel
        .from_pretrained(
            merged_source_model,
            model_directory,
            is_trainable=False,
        )
    )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    model.eval()

    processor = (
        WhisperProcessor
        .from_pretrained(
            model_directory
        )
    )

    return (
        model,
        processor,
    )


# Stacked residual adapter

def load_residual_adapter_stacked(
    *,
    model_directory: Path,
    metadata: dict[str, Any],
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
]:
    if not model_directory.exists():
        raise FileNotFoundError(
            "Stacked residual-adapter model "
            "directory does not exist:\n"
            f"{model_directory}"
        )

    stacked_config_path = (
        model_directory
        / "stacked_adapter_config.json"
    )

    adapter_state_path = (
        model_directory
        / "adapter_state.pt"
    )

    stacked_config = (
        load_json_object(
            stacked_config_path
        )
    )

    target_state = (
        load_residual_adapter_state(
            adapter_state_path
        )
    )

    source_model_root_value = (
        stacked_config.get(
            "source_model_root"
        )
    )

    if not source_model_root_value:
        architecture = metadata.get(
            "architecture"
        )

        if isinstance(
            architecture,
            dict,
        ):
            source_model_root_value = (
                architecture.get(
                    "source_model_root"
                )
            )

    if not source_model_root_value:
        raise ValueError(
            "Stacked residual-adapter model "
            "does not specify source_model_root."
        )

    source_model_root = Path(
        str(
            source_model_root_value
        )
    )

    if not source_model_root.exists():
        raise FileNotFoundError(
            "Stacked residual-adapter source "
            "model root does not exist:\n"
            f"{source_model_root}"
        )

    # First reconstruct the complete source:
    #
    # vanilla Whisper + source-domain adapter.

    (
        model,
        _,
        source_metadata,
    ) = load_model(
        model_root=(
            source_model_root
        ),
        torch_dtype=(
            torch_dtype
        ),
    )

    if (
        source_metadata.get(
            "model_type"
        )
        != "residual_adapter"
    ):
        raise RuntimeError(
            "Stacked residual-adapter source "
            "must be an ordinary "
            "residual_adapter model.\n"
            f"Source: {source_model_root}\n"
            f"Found model_type: "
            f"{source_metadata.get('model_type')!r}"
        )

    adapter_module = str(
        stacked_config.get(
            "adapter_module",
            "",
        )
    ).strip()

    if adapter_module not in (
        "encoder",
        "decoder",
        "encoder_decoder",
    ):
        raise ValueError(
            "Unsupported stacked "
            "residual-adapter module: "
            f"{adapter_module!r}"
        )

    raw_layers = (
        stacked_config.get(
            "adapter_layers"
        )
    )

    if not isinstance(
        raw_layers,
        list,
    ):
        raise ValueError(
            "stacked_adapter_config.json "
            "must contain adapter_layers "
            "as a list."
        )

    adapter_layers = [
        int(
            value
        )
        for value in raw_layers
    ]

    if not adapter_layers:
        raise ValueError(
            "Stacked adapter layer list "
            "must not be empty."
        )

    bottleneck = int(
        stacked_config[
            "bottleneck"
        ]
    )

    dropout = float(
        stacked_config[
            "dropout"
        ]
    )

    if bottleneck <= 0:
        raise ValueError(
            "Stacked residual-adapter "
            "bottleneck must be positive."
        )

    if not (
        0.0
        <= dropout
        < 1.0
    ):
        raise ValueError(
            "Stacked residual-adapter dropout "
            "must satisfy 0 <= dropout < 1."
        )

    hidden_size = int(
        model.config.d_model
    )

    # The source adapter is already inside
    # each source layer.
    #
    # wrap_residual_adapter_layers therefore
    # puts the fresh target adapter OUTSIDE
    # the already reconstructed source layer:
    #
    # Whisper layer
    # -> frozen source adapter
    # -> trained target adapter.

    if adapter_module in (
        "encoder",
        "encoder_decoder",
    ):
        model.model.encoder.layers = (
            wrap_residual_adapter_layers(
                layers=(
                    model
                    .model
                    .encoder
                    .layers
                ),
                module_name="encoder",
                selected_layers=(
                    adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
                bottleneck=(
                    bottleneck
                ),
                dropout=(
                    dropout
                ),
                state=(
                    target_state
                ),
            )
        )

    if adapter_module in (
        "decoder",
        "encoder_decoder",
    ):
        model.model.decoder.layers = (
            wrap_residual_adapter_layers(
                layers=(
                    model
                    .model
                    .decoder
                    .layers
                ),
                module_name="decoder",
                selected_layers=(
                    adapter_layers
                ),
                hidden_size=(
                    hidden_size
                ),
                bottleneck=(
                    bottleneck
                ),
                dropout=(
                    dropout
                ),
                state=(
                    target_state
                ),
            )
        )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    model.eval()

    processor = (
        WhisperProcessor
        .from_pretrained(
            model_directory
        )
    )

    return (
        model,
        processor,
    )



# Generic loader

def load_model(
    *,
    model_root: Path,
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    torch.nn.Module,
    WhisperProcessor,
    dict[str, Any],
]:
    model_root = (
        model_root.resolve()
    )

    metadata = (
        load_model_metadata(
            model_root
        )
    )

    model_directory = (
        model_root
        / "model"
    )

    model_type = str(
        metadata[
            "model_type"
        ]
    ).strip()

    if model_type == "whisper_full":
        model, processor = (
            load_whisper_full(
                model_directory=(
                    model_directory
                ),
                torch_dtype=(
                    torch_dtype
                ),
            )
        )

    elif model_type == "peft_lora":
        model, processor = (
            load_peft_lora(
                model_directory=(
                    model_directory
                ),
                metadata=metadata,
                torch_dtype=(
                    torch_dtype
                ),
            )
        )

    elif model_type == "peft_lora_stacked":
        model, processor = (
            load_peft_lora_stacked(
                model_directory=(
                    model_directory
                ),
                metadata=metadata,
                torch_dtype=(
                    torch_dtype
                ),
            )
        )

    elif model_type == "residual_adapter":
        model, processor = (
            load_residual_adapter(
                model_directory=(
                    model_directory
                ),
                metadata=metadata,
                torch_dtype=(
                    torch_dtype
                ),
            )
        )

    elif (
        model_type
        == "residual_adapter_stacked"
    ):
        model, processor = (
            load_residual_adapter_stacked(
                model_directory=(
                    model_directory
                ),
                metadata=metadata,
                torch_dtype=(
                    torch_dtype
                ),
            )
        )

    elif model_type == "spt":
        raise NotImplementedError(
            "SPT loading has not been "
            "implemented yet."
        )

    elif model_type == "adapter_fusion":
        raise NotImplementedError(
            "Adapter-fusion loading has not "
            "been implemented yet."
        )

    else:
        raise ValueError(
            "Unsupported model_type: "
            f"{model_type!r}"
        )

    return (
        model,
        processor,
        metadata,
    )