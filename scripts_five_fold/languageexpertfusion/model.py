from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

from scripts_five_fold.languageexpertfusion.config import (
    BASE_MODEL,
    COLD_FUSION_ATTENTION_DIM,
    COLD_FUSION_DROPOUT,
    LANGUAGE,
    MLP_FUSION_BOTTLENECK,
    MLP_FUSION_DROPOUT,
    MLP_FUSION_HIDDEN_1,
    MLP_FUSION_HIDDEN_2,
    TASK,
    adapter_bottleneck,
    adapter_dropout,
    adapter_layers,
)
from scripts_five_fold.train import configure_whisper_generation


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_size, hidden_size)

        # Start as the exact frozen backbone.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def correction(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(
            self.dropout(
                self.activation(
                    self.down(hidden_states)
                )
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.correction(hidden_states)


class SingleAdapterWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
        architecture: str,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.residual_adapter = ResidualAdapter(
            hidden_size,
            adapter_bottleneck(architecture),
            adapter_dropout(architecture),
        )

    def _adapt(self, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return self.residual_adapter(output)

        if isinstance(output, tuple):
            return (self.residual_adapter(output[0]), *output[1:])

        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = self.residual_adapter(output.last_hidden_state)
            return output

        raise TypeError(f"Unexpected encoder output: {type(output)!r}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._adapt(self.base_layer(*args, **kwargs))


def build_expert_model(architecture: str) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    configure_whisper_generation(
        model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(model.config.d_model)
    selected = set(adapter_layers(architecture))
    wrapped: list[nn.Module] = []

    for layer_index, layer in enumerate(model.model.encoder.layers):
        if layer_index in selected:
            wrapped.append(
                SingleAdapterWrappedLayer(
                    layer,
                    hidden_size,
                    architecture,
                )
            )
        else:
            wrapped.append(layer)

    model.model.encoder.layers = nn.ModuleList(wrapped)

    for layer_index in adapter_layers(architecture):
        layer = model.model.encoder.layers[layer_index]
        if not isinstance(layer, SingleAdapterWrappedLayer):
            raise TypeError(f"Expected adapter layer at {layer_index}")
        for parameter in layer.residual_adapter.parameters():
            parameter.requires_grad = True

    return model


def adapter_state_dict(
    model: WhisperForConditionalGeneration,
    architecture: str,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}

    for layer_index in adapter_layers(architecture):
        layer = model.model.encoder.layers[layer_index]
        if not isinstance(layer, SingleAdapterWrappedLayer):
            raise TypeError(f"Expected adapter layer at {layer_index}")

        for key, value in layer.residual_adapter.state_dict().items():
            state[f"layer_{layer_index}.{key}"] = value.detach().cpu().clone()

    return state


def save_adapter(
    model: WhisperForConditionalGeneration,
    directory: Path,
    architecture: str,
    expert_name: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        adapter_state_dict(model, architecture),
        directory / "adapter_state.pt",
    )

    config = {
        "base_model": BASE_MODEL,
        "architecture": architecture,
        "expert_name": expert_name,
        "adapter_layers": adapter_layers(architecture),
        "bottleneck": adapter_bottleneck(architecture),
        "dropout": adapter_dropout(architecture),
    }

    (directory / "adapter_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def load_adapter_state(directory: Path) -> dict[str, torch.Tensor]:
    return torch.load(
        directory / "adapter_state.pt",
        map_location="cpu",
        weights_only=True,
    )


class ColdFusionRouter(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.query = nn.Linear(
            hidden_size,
            COLD_FUSION_ATTENTION_DIM,
            bias=False,
        )
        self.key = nn.Linear(
            hidden_size,
            COLD_FUSION_ATTENTION_DIM,
            bias=False,
        )
        self.dropout = nn.Dropout(COLD_FUSION_DROPOUT)
        self.expert_bias = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        spanish_states: torch.Tensor,
        english_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(self.dropout(hidden_states))

        expert_states = torch.stack(
            [spanish_states, english_states],
            dim=-2,
        )

        keys = self.key(self.dropout(expert_states))

        scores = (
            query.unsqueeze(-2) * keys
        ).sum(dim=-1) / (COLD_FUSION_ATTENTION_DIM ** 0.5)

        scores = scores + self.expert_bias
        weights = torch.softmax(scores, dim=-1)

        fused = (
            expert_states * weights.unsqueeze(-1)
        ).sum(dim=-2)

        return fused, weights


class MLPInteractionFusion(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size * 5, MLP_FUSION_HIDDEN_1),
            nn.GELU(),
            nn.Dropout(MLP_FUSION_DROPOUT),
            nn.Linear(MLP_FUSION_HIDDEN_1, MLP_FUSION_BOTTLENECK),
            nn.GELU(),
            nn.Dropout(MLP_FUSION_DROPOUT),
            nn.Linear(MLP_FUSION_BOTTLENECK, MLP_FUSION_HIDDEN_2),
            nn.GELU(),
            nn.Dropout(MLP_FUSION_DROPOUT),
            nn.Linear(MLP_FUSION_HIDDEN_2, hidden_size),
        )

        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        spanish_delta: torch.Tensor,
        english_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interaction = spanish_delta * english_delta
        disagreement = torch.abs(spanish_delta - english_delta)

        fusion_input = torch.cat(
            [
                hidden_states,
                spanish_delta,
                english_delta,
                interaction,
                disagreement,
            ],
            dim=-1,
        )

        correction = self.network(fusion_input)
        return hidden_states + correction, correction


class FusionWrappedLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        hidden_size: int,
        architecture: str,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer
        self.architecture = architecture

        self.spanish_adapter = ResidualAdapter(
            hidden_size,
            adapter_bottleneck(architecture),
            adapter_dropout(architecture),
        )
        self.english_adapter = ResidualAdapter(
            hidden_size,
            adapter_bottleneck(architecture),
            adapter_dropout(architecture),
        )

        if architecture == "cold":
            self.fusion = ColdFusionRouter(hidden_size)
        elif architecture == "mlp":
            self.fusion = MLPInteractionFusion(hidden_size)
        else:
            raise ValueError(f"Unknown architecture: {architecture!r}")

    def _fuse(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.architecture == "cold":
            spanish_states = self.spanish_adapter(hidden_states)
            english_states = self.english_adapter(hidden_states)
            fused, _ = self.fusion(
                hidden_states,
                spanish_states,
                english_states,
            )
            return fused

        spanish_delta = self.spanish_adapter.correction(hidden_states)
        english_delta = self.english_adapter.correction(hidden_states)
        fused, _ = self.fusion(
            hidden_states,
            spanish_delta,
            english_delta,
        )
        return fused

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.base_layer(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            return self._fuse(output)

        if isinstance(output, tuple):
            return (self._fuse(output[0]), *output[1:])

        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = self._fuse(output.last_hidden_state)
            return output

        raise TypeError(f"Unexpected encoder output: {type(output)!r}")


def _layer_state(
    state: dict[str, torch.Tensor],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    prefix = f"layer_{layer_index}."
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def build_fusion_model(
    architecture: str,
    spanish_adapter_directory: Path,
    english_adapter_directory: Path,
) -> WhisperForConditionalGeneration:
    spanish_state = load_adapter_state(spanish_adapter_directory)
    english_state = load_adapter_state(english_adapter_directory)

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    configure_whisper_generation(
        model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(model.config.d_model)
    selected = set(adapter_layers(architecture))
    wrapped: list[nn.Module] = []

    for layer_index, layer in enumerate(model.model.encoder.layers):
        if layer_index not in selected:
            wrapped.append(layer)
            continue

        fused_layer = FusionWrappedLayer(
            layer,
            hidden_size,
            architecture,
        )

        fused_layer.spanish_adapter.load_state_dict(
            _layer_state(spanish_state, layer_index),
            strict=True,
        )
        fused_layer.english_adapter.load_state_dict(
            _layer_state(english_state, layer_index),
            strict=True,
        )

        wrapped.append(fused_layer)

    model.model.encoder.layers = nn.ModuleList(wrapped)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_index in adapter_layers(architecture):
        layer = model.model.encoder.layers[layer_index]
        if not isinstance(layer, FusionWrappedLayer):
            raise TypeError(f"Expected fusion layer at {layer_index}")
        for parameter in layer.fusion.parameters():
            parameter.requires_grad = True

    return model


def fusion_state_dict(
    model: WhisperForConditionalGeneration,
    architecture: str,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}

    for layer_index in adapter_layers(architecture):
        layer = model.model.encoder.layers[layer_index]
        if not isinstance(layer, FusionWrappedLayer):
            raise TypeError(f"Expected fusion layer at {layer_index}")

        for key, value in layer.fusion.state_dict().items():
            state[f"layer_{layer_index}.{key}"] = value.detach().cpu().clone()

    return state


def save_fusion(
    model: WhisperForConditionalGeneration,
    directory: Path,
    architecture: str,
    spanish_adapter_directory: Path,
    english_adapter_directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        fusion_state_dict(model, architecture),
        directory / "fusion_state.pt",
    )

    config = {
        "base_model": BASE_MODEL,
        "architecture": architecture,
        "adapter_layers": adapter_layers(architecture),
        "spanish_adapter_directory": str(spanish_adapter_directory),
        "english_adapter_directory": str(english_adapter_directory),
    }

    (directory / "fusion_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def load_fusion(
    model: WhisperForConditionalGeneration,
    directory: Path,
    architecture: str,
) -> None:
    state = torch.load(
        directory / "fusion_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    for layer_index in adapter_layers(architecture):
        layer = model.model.encoder.layers[layer_index]
        if not isinstance(layer, FusionWrappedLayer):
            raise TypeError(f"Expected fusion layer at {layer_index}")

        layer.fusion.load_state_dict(
            _layer_state(state, layer_index),
            strict=True,
        )
