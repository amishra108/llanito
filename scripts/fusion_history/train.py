from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from transformers import (
    Trainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
)


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.fusion_history import config

from scripts.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_classification,
    build_training_arguments,
    configure_whisper_generation,
    distributed_barrier,
    is_main_process,
    load_manifest,
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


# JSON

def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "JSON file does not exist:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(
            handle
        )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Expected JSON object:\n"
            f"{path}"
        )

    return value


# Residual adapter

class ResidualAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(
            hidden_size,
            bottleneck,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.up = nn.Linear(
            bottleneck,
            hidden_size,
        )

    def delta(
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

        return correction


# History fusion

class HistoryFusionModule(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        history_width: int,
        history_length: int,
    ) -> None:
        super().__init__()

        self.hidden_size = (
            hidden_size
        )

        self.history_width = (
            history_width
        )

        self.history_length = (
            history_length
        )

        interaction_size = (
            4
            * hidden_size
        )

        self.compress = nn.Linear(
            interaction_size,
            history_width,
        )

        self.activation = nn.GELU()

        self.history_gru = nn.GRU(
            input_size=history_width,
            hidden_size=history_width,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.output = nn.Linear(
            history_width,
            hidden_size,
        )

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        nn.init.xavier_uniform_(
            self.compress.weight
        )

        nn.init.zeros_(
            self.compress.bias
        )

        # Match the existing HIST model:
        # start with zero learned correction.

        nn.init.zeros_(
            self.output.weight
        )

        nn.init.zeros_(
            self.output.bias
        )

    def forward(
        self,
        history: list[
            torch.Tensor
        ],
    ) -> torch.Tensor:
        if (
            len(history)
            != self.history_length
        ):
            raise RuntimeError(
                "History module expected "
                f"{self.history_length} "
                "interaction tensors but "
                f"received {len(history)}."
            )

        compressed = []

        for interaction in history:
            value = self.compress(
                interaction
            )

            value = self.activation(
                value
            )

            compressed.append(
                value
            )

        # Each compressed descriptor:
        #
        # [batch, acoustic_time, 128]
        #
        # Stack over encoder depth:
        #
        # [batch, acoustic_time,
        #  history_length, 128]

        sequence = torch.stack(
            compressed,
            dim=2,
        )

        batch_size = int(
            sequence.shape[0]
        )

        acoustic_steps = int(
            sequence.shape[1]
        )

        sequence = sequence.reshape(
            batch_size
            * acoustic_steps,
            self.history_length,
            self.history_width,
        )

        history_output, _ = (
            self.history_gru(
                sequence
            )
        )

        final_history = (
            history_output[
                :,
                -1,
                :,
            ]
        )

        correction = self.output(
            final_history
        )

        correction = correction.reshape(
            batch_size,
            acoustic_steps,
            self.hidden_size,
        )

        return correction


# Expert loading

def load_expert_config(
    domain: str,
) -> dict[str, Any]:
    path = (
        config
        .get_branch_model_directory(
            domain
        )
        / "adapter_config.json"
    )

    adapter_config = (
        load_json_object(
            path
        )
    )

    if (
        str(
            adapter_config.get(
                "adapter_module",
                "",
            )
        )
        != "encoder"
    ):
        raise RuntimeError(
            "HIST requires encoder-only "
            "ResAdapt experts.\n"
            f"Domain: {domain}"
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
        raise RuntimeError(
            "Expert is missing "
            "adapter_layers."
        )

    layers = [
        int(value)
        for value in raw_layers
    ]

    if layers != list(
        config.ENCODER_LAYERS
    ):
        raise RuntimeError(
            "HIST requires experts "
            "covering all 24 encoder "
            "layers.\n"
            f"Domain: {domain}\n"
            f"Found: {layers}"
        )

    return adapter_config


def load_expert_state(
    domain: str,
) -> dict[
    str,
    torch.Tensor,
]:
    path = (
        config
        .get_branch_model_directory(
            domain
        )
        / "adapter_state.pt"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Expert state does not exist:\n"
            f"{path}"
        )

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "adapter_state.pt must "
            "contain a dictionary."
        )

    return state


def extract_expert_layer_state(
    *,
    state: dict[
        str,
        torch.Tensor,
    ],
    layer_index: int,
) -> dict[
    str,
    torch.Tensor,
]:
    prefixes = (
        (
            "encoder."
            f"layer_{layer_index}."
        ),
        (
            f"layer_{layer_index}."
        ),
        (
            "encoder.layers."
            f"{layer_index}."
            "residual_adapter."
        ),
    )

    for prefix in prefixes:
        result = {
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

        if result:
            return result

    raise RuntimeError(
        "Could not find residual "
        "adapter state for encoder "
        f"layer {layer_index}."
    )


def load_expert_payloads(
    experiment: (
        config.ExperimentConfig
    ),
) -> tuple[
    dict[
        str,
        tuple[
            dict[str, Any],
            dict[str, torch.Tensor],
        ],
    ],
    list[dict[str, Any]],
]:
    payloads = {}

    architecture_experts = []

    for domain in (
        experiment.branch_domains
    ):
        model_root = (
            config
            .get_branch_model_root(
                domain
            )
        )

        model_directory = (
            config
            .get_branch_model_directory(
                domain
            )
        )

        if not model_directory.is_dir():
            raise FileNotFoundError(
                "Expert model directory "
                "does not exist:\n"
                f"{model_directory}"
            )

        adapter_config = (
            load_expert_config(
                domain
            )
        )

        adapter_state = (
            load_expert_state(
                domain
            )
        )

        payloads[
            domain
        ] = (
            adapter_config,
            adapter_state,
        )

        architecture_experts.append(
            {
                "domain": domain,
                "model_root": str(
                    model_root
                ),
                "model_directory": str(
                    model_directory
                ),
                "adapter_module": (
                    "encoder"
                ),
                "bottleneck": int(
                    adapter_config[
                        "bottleneck"
                    ]
                ),
                "dropout": float(
                    adapter_config[
                        "dropout"
                    ]
                ),
                "frozen": True,
            }
        )

    return (
        payloads,
        architecture_experts,
    )


def make_adapter(
    *,
    domain: str,
    layer_index: int,
    hidden_size: int,
    payloads: dict,
) -> ResidualAdapter:
    (
        adapter_config,
        adapter_state,
    ) = payloads[
        domain
    ]

    adapter = ResidualAdapter(
        hidden_size=hidden_size,
        bottleneck=int(
            adapter_config[
                "bottleneck"
            ]
        ),
        dropout=0.0,
    )

    adapter.load_state_dict(
        extract_expert_layer_state(
            state=adapter_state,
            layer_index=(
                layer_index
            ),
        ),
        strict=True,
    )

    return adapter


# Parallel history encoder

class HistoryFusionEncoder(nn.Module):
    def __init__(
        self,
        *,
        original_encoder: nn.Module,
        experiment: (
            config.ExperimentConfig
        ),
        payloads: dict,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.config = (
            original_encoder.config
        )

        self.dropout = (
            original_encoder.dropout
        )

        self.layerdrop = (
            original_encoder.layerdrop
        )

        self.conv1 = (
            original_encoder.conv1
        )

        self.conv2 = (
            original_encoder.conv2
        )

        self.embed_positions = (
            original_encoder
            .embed_positions
        )

        self.layers = (
            original_encoder.layers
        )

        self.layer_norm = (
            original_encoder.layer_norm
        )

        self.gradient_checkpointing = (
            False
        )

        self.branch_names = (
            experiment.branch_domains
        )

        if len(
            self.branch_names
        ) != 2:
            raise RuntimeError(
                "HIST expects exactly "
                "two expert branches."
            )

        self.history_length = (
            experiment.history_length
        )

        self.fusion_layers = set(
            experiment
            .history_fusion_layers
        )

        if (
            config.NUM_ENCODER_LAYERS - 1
            not in self.fusion_layers
        ):
            raise RuntimeError(
                "Final encoder layer must "
                "be a HIST fusion point."
            )

        # Frozen expert adapter at every
        # encoder layer for both branches.

        adapter_layers = []

        for layer_index in (
            config.ENCODER_LAYERS
        ):
            layer_adapters = {}

            for domain in (
                self.branch_names
            ):
                layer_adapters[
                    domain
                ] = make_adapter(
                    domain=domain,
                    layer_index=(
                        layer_index
                    ),
                    hidden_size=hidden_size,
                    payloads=payloads,
                )

            adapter_layers.append(
                nn.ModuleDict(
                    layer_adapters
                )
            )

        self.expert_adapters = (
            nn.ModuleList(
                adapter_layers
            )
        )

        # One trainable history fusion
        # module per resynchronisation point.

        self.history_modules = (
            nn.ModuleDict(
                {
                    str(layer_index): (
                        HistoryFusionModule(
                            hidden_size=(
                                hidden_size
                            ),
                            history_width=(
                                config
                                .HISTORY_WIDTH
                            ),
                            history_length=(
                                experiment
                                .history_length
                            ),
                        )
                    )
                    for layer_index in (
                        experiment
                        .history_fusion_layers
                    )
                }
            )
        )

    def make_interaction(
        self,
        delta_a: torch.Tensor,
        delta_b: torch.Tensor,
    ) -> torch.Tensor:
        product = (
            delta_a
            * delta_b
        )

        difference = (
            delta_a
            - delta_b
        )

        return torch.cat(
            [
                delta_a,
                delta_b,
                product,
                difference,
            ],
            dim=-1,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: (
            torch.Tensor | None
        ) = None,
        head_mask: (
            torch.Tensor | None
        ) = None,
        output_attentions: (
            bool | None
        ) = None,
        output_hidden_states: (
            bool | None
        ) = None,
        return_dict: (
            bool | None
        ) = None,
        **kwargs: Any,
    ) -> BaseModelOutput | tuple:
        output_attentions = (
            False
            if output_attentions is None
            else output_attentions
        )

        output_hidden_states = (
            False
            if output_hidden_states
            is None
            else output_hidden_states
        )

        return_dict = (
            True
            if return_dict is None
            else return_dict
        )

        if output_attentions:
            raise RuntimeError(
                "HIST encoder does not "
                "currently expose ordinary "
                "Whisper self-attention."
            )

        # Whisper convolutional frontend.

        hidden_states = (
            torch.nn.functional.gelu(
                self.conv1(
                    input_features
                )
            )
        )

        hidden_states = (
            torch.nn.functional.gelu(
                self.conv2(
                    hidden_states
                )
            )
        )

        hidden_states = (
            hidden_states.permute(
                0,
                2,
                1,
            )
        )

        sequence_length = int(
            hidden_states.shape[1]
        )

        position_embeddings = (
            self.embed_positions.weight[
                :sequence_length
            ]
        )

        hidden_states = (
            hidden_states
            + position_embeddings
        )

        hidden_states = (
            torch.nn.functional.dropout(
                hidden_states,
                p=self.dropout,
                training=self.training,
            )
        )

        # Both branches begin from exactly
        # the same Whisper frontend state.

        branch_states = {
            domain: hidden_states
            for domain in (
                self.branch_names
            )
        }

        history_buffer: list[
            torch.Tensor
        ] = []

        encoder_states = (
            ()
            if output_hidden_states
            else None
        )

        for (
            layer_index,
            base_layer,
        ) in enumerate(
            self.layers
        ):
            base_outputs = {}

            deltas = {}

            conditioned_states = {}

            for domain in (
                self.branch_names
            ):
                branch_input = (
                    branch_states[
                        domain
                    ]
                )

                layer_head_mask = (
                    head_mask[
                        layer_index
                    ]
                    if head_mask
                    is not None
                    else None
                )

                # Avoid forwarding the raw
                # integer mask to SDPA.

                layer_output = (
                    base_layer(
                        branch_input,
                        None,
                        layer_head_mask=(
                            layer_head_mask
                        ),
                        output_attentions=False,
                    )
                )

                if isinstance(
                    layer_output,
                    tuple,
                ):
                    z = layer_output[0]

                else:
                    z = layer_output

                delta = (
                    self.expert_adapters[
                        layer_index
                    ][
                        domain
                    ].delta(
                        z
                    )
                )

                conditioned = (
                    z
                    + delta
                )

                base_outputs[
                    domain
                ] = z

                deltas[
                    domain
                ] = delta

                conditioned_states[
                    domain
                ] = conditioned

            domain_a = (
                self.branch_names[0]
            )

            domain_b = (
                self.branch_names[1]
            )

            interaction = (
                self.make_interaction(
                    deltas[
                        domain_a
                    ],
                    deltas[
                        domain_b
                    ],
                )
            )

            history_buffer.append(
                interaction
            )

            if (
                layer_index
                in self.fusion_layers
            ):
                if (
                    len(history_buffer)
                    != self.history_length
                ):
                    raise RuntimeError(
                        "Unexpected history "
                        "length at encoder "
                        f"layer "
                        f"{layer_index + 1}.\n"
                        f"Expected: "
                        f"{self.history_length}\n"
                        f"Found: "
                        f"{len(history_buffer)}"
                    )

                base_stack = (
                    torch.stack(
                        [
                            base_outputs[
                                domain
                            ]
                            for domain
                            in self.branch_names
                        ],
                        dim=-2,
                    )
                )

                shared_base = (
                    base_stack.mean(
                        dim=-2
                    )
                )

                history_module = (
                    self.history_modules[
                        str(
                            layer_index
                        )
                    ]
                )

                fused_correction = (
                    history_module(
                        history_buffer
                    )
                )

                fused_state = (
                    shared_base
                    + fused_correction
                )

                # Resynchronise both expert
                # streams at the fusion point.

                branch_states = {
                    domain: fused_state
                    for domain in (
                        self.branch_names
                    )
                }

                history_buffer = []

            else:
                # Between fusion points,
                # each expert-conditioned
                # representation continues
                # independently.

                branch_states = (
                    conditioned_states
                )

            if output_hidden_states:
                aggregate = torch.stack(
                    [
                        branch_states[
                            domain
                        ]
                        for domain
                        in self.branch_names
                    ],
                    dim=-2,
                ).mean(
                    dim=-2
                )

                encoder_states = (
                    encoder_states
                    + (
                        aggregate,
                    )
                )

        if history_buffer:
            raise RuntimeError(
                "HIST encoder ended with "
                "an unfinished history "
                "window."
            )

        final_state = (
            branch_states[
                self.branch_names[
                    0
                ]
            ]
        )

        final_state = (
            self.layer_norm(
                final_state
            )
        )

        if output_hidden_states:
            encoder_states = (
                encoder_states
                + (
                    final_state,
                )
            )

        if not return_dict:
            outputs = (
                final_state,
            )

            if output_hidden_states:
                outputs = (
                    outputs
                    + (
                        encoder_states,
                    )
                )

            return outputs

        return BaseModelOutput(
            last_hidden_state=(
                final_state
            ),
            hidden_states=(
                encoder_states
            ),
            attentions=None,
        )


# Validation

def validate_experiment(
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    if (
        experiment.history_length
        <= 0
    ):
        raise RuntimeError(
            "history_length must be "
            "greater than zero."
        )

    if (
        config.NUM_ENCODER_LAYERS
        % experiment.history_length
        != 0
    ):
        raise RuntimeError(
            "History length must divide "
            "the 24-layer Whisper encoder "
            "exactly.\n"
            f"History length: "
            f"{experiment.history_length}"
        )

    if not (
        experiment
        .history_fusion_layers
    ):
        raise RuntimeError(
            "No history fusion layers "
            "configured."
        )

    if (
        experiment
        .history_fusion_layers[-1]
        != config
        .NUM_ENCODER_LAYERS - 1
    ):
        raise RuntimeError(
            "Final encoder layer must "
            "be a fusion point."
        )


# Build model

def build_history_model(
    experiment: (
        config.ExperimentConfig
    ),
    *,
    language: str | None = None,
) -> tuple[
    WhisperForConditionalGeneration,
    WhisperProcessor,
    dict[str, Any],
]:
    validate_experiment(
        experiment
    )

    resolved_language = (
        language
        if language is not None
        else experiment.training_language
    )

    if not (
        config
        .VANILLA_MODEL_DIRECTORY
        .is_dir()
    ):
        raise FileNotFoundError(
            "Vanilla Whisper directory "
            "does not exist:\n"
            f"{config.VANILLA_MODEL_DIRECTORY}"
        )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            config
            .VANILLA_MODEL_DIRECTORY
        )
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            config
            .VANILLA_MODEL_DIRECTORY
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=resolved_language,
        task=config.TASK,
        predict_timestamps=False,
    )

    configure_whisper_generation(
        model,
        language=resolved_language,
        task=config.TASK,
        use_cache=True,
    )

    hidden_size = int(
        model.config.d_model
    )

    if hidden_size != config.HIDDEN_SIZE:
        raise RuntimeError(
            "Unexpected Whisper hidden "
            f"size: {hidden_size}"
        )

    (
        payloads,
        architecture_experts,
    ) = load_expert_payloads(
        experiment
    )

    original_encoder = (
        model.model.encoder
    )

    model.model.encoder = (
        HistoryFusionEncoder(
            original_encoder=(
                original_encoder
            ),
            experiment=experiment,
            payloads=payloads,
            hidden_size=hidden_size,
        )
    )

    # Freeze everything.

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    # Re-enable history modules only.

    for (
        name,
        parameter,
    ) in model.named_parameters():
        if ".history_modules." in name:
            parameter.requires_grad = True

    trainable_names = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    invalid = [
        name
        for name in trainable_names
        if ".history_modules." not in name
    ]

    if invalid:
        raise RuntimeError(
            "Unexpected trainable "
            "parameters:\n"
            + "\n".join(
                invalid[:50]
            )
        )

    counts = parameter_summary(
        model
    )

    found = int(
        counts[
            "trainable_parameters"
        ]
    )

    expected = (
        experiment
        .expected_trainable_parameters
    )

    if found != expected:
        raise RuntimeError(
            "Unexpected HIST trainable "
            "parameter count.\n"
            f"History length: "
            f"{experiment.history_length}\n"
            f"Expected: {expected:,}\n"
            f"Found: {found:,}"
        )

    architecture = {
        "adaptation": (
            "residual_adapter_history_fusion"
        ),
        "fusion_architecture": (
            "persistent_parallel_history"
        ),
        "placement": "encoder",
        "branch_domains": list(
            experiment.branch_domains
        ),
        "target_domain": (
            experiment.target_domain
        ),
        "experts": (
            architecture_experts
        ),
        "hidden_size": hidden_size,
        "history_length": (
            experiment.history_length
        ),
        "history_width": (
            config.HISTORY_WIDTH
        ),
        "history_fusion_layers_zero_based": (
            list(
                experiment
                .history_fusion_layers
            )
        ),
        "history_fusion_layers_human": [
            layer + 1
            for layer in (
                experiment
                .history_fusion_layers
            )
        ],
        "number_of_fusion_layers": (
            experiment
            .number_of_fusion_layers
        ),
        "interaction_definition": [
            "delta_a",
            "delta_b",
            (
                "delta_a_elementwise_"
                "product_delta_b"
            ),
            "delta_a_minus_delta_b",
        ],
        "interaction_size": (
            config.INTERACTION_SIZE
        ),
        "history_compression": (
            "4096_to_128"
        ),
        "history_sequence_axis": (
            "encoder_depth"
        ),
        "history_recurrent_module": (
            "unidirectional_gru"
        ),
        "history_gru_hidden_size": (
            config.HISTORY_WIDTH
        ),
        "history_output_projection": (
            "128_to_1024"
        ),
        "output_rule": (
            "mean_current_pre_adapter_"
            "branch_state_plus_history_"
            "correction"
        ),
        "parallel_streams": True,
        "whisper_frozen": True,
        "experts_frozen": True,
        "history_modules_trainable": True,
        "trainable_parameters": (
            expected
        ),
        "training_language": (
            resolved_language
        ),
    }

    return (
        model,
        processor,
        architecture,
    )


# State helpers

def history_modules_by_layer(
    model: nn.Module,
) -> dict[
    int,
    HistoryFusionModule,
]:
    base = model

    while hasattr(
        base,
        "module",
    ):
        base = base.module

    encoder = (
        base.model.encoder
    )

    if not isinstance(
        encoder,
        HistoryFusionEncoder,
    ):
        raise RuntimeError(
            "Expected HistoryFusionEncoder."
        )

    return {
        int(key): module
        for (
            key,
            module,
        ) in (
            encoder
            .history_modules
            .items()
        )
    }


def collect_history_state(
    model: nn.Module,
) -> dict[
    str,
    torch.Tensor,
]:
    state = {}

    modules = (
        history_modules_by_layer(
            model
        )
    )

    for (
        layer_index,
        module,
    ) in modules.items():
        for (
            key,
            value,
        ) in (
            module
            .state_dict()
            .items()
        ):
            state[
                (
                    f"layer_"
                    f"{layer_index}."
                    f"{key}"
                )
            ] = (
                value
                .detach()
                .cpu()
                .clone()
            )

    if not state:
        raise RuntimeError(
            "No HIST state collected."
        )

    return state


def load_history_state(
    *,
    model: nn.Module,
    state_path: Path,
) -> None:
    if not state_path.is_file():
        raise FileNotFoundError(
            "HIST state does not exist:\n"
            f"{state_path}"
        )

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )

    modules = (
        history_modules_by_layer(
            model
        )
    )

    for (
        layer_index,
        module,
    ) in modules.items():
        prefix = (
            f"layer_{layer_index}."
        )

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

        if not layer_state:
            raise RuntimeError(
                "Missing HIST state "
                f"for layer {layer_index}."
            )

        module.load_state_dict(
            layer_state,
            strict=True,
        )


# Save

def save_final_model(
    *,
    model: nn.Module,
    processor: WhisperProcessor,
    experiment: (
        config.ExperimentConfig
    ),
    architecture: dict[
        str,
        Any,
    ],
) -> None:
    experiment.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        collect_history_state(
            model
        ),
        (
            experiment
            .model_directory
            / "history_state.pt"
        ),
    )

    save_json(
        (
            experiment
            .model_directory
            / "history_config.json"
        ),
        {
            "format_version": 2,
            "model_type": (
                config.MODEL_TYPE
            ),
            "model_name": (
                experiment.model_name
            ),
            "base_model": (
                config.BASE_MODEL
            ),
            "base_model_root": str(
                config
                .VANILLA_MODEL_ROOT
            ),
            **architecture,
        },
    )

    processor.save_pretrained(
        experiment.model_directory
    )


# Data

def load_training_data(
    experiment: (
        config.ExperimentConfig
    ),
) -> tuple[
    Path,
    list[dict[str, str]],
]:
    train_manifest = (
        config.get_train_manifest(
            experiment
        )
    )

    if not train_manifest.is_file():
        raise FileNotFoundError(
            "Training manifest does not "
            "exist:\n"
            f"{train_manifest}"
        )

    rows = load_manifest(
        train_manifest,
        required_columns=(
            "audio_path",
            "transcript",
        ),
    )

    expected = (
        experiment
        .expected_training_examples
    )

    if (
        expected is not None
        and len(rows) != expected
    ):
        raise RuntimeError(
            "Unexpected training size.\n"
            f"Expected: {expected}\n"
            f"Found: {len(rows)}"
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
        len(test_rows)
        != config
        .EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected fixed-test size."
        )

    validate_no_duplicate_audio(
        rows,
        dataset_name=(
            experiment.model_name
        ),
    )

    validate_no_duplicate_audio(
        test_rows,
        dataset_name=(
            "Llanito fixed test"
        ),
    )

    validate_no_audio_overlap(
        rows,
        test_rows,
        first_name=(
            experiment.model_name
        ),
        second_name=(
            "llanito_fixed_test"
        ),
    )

    validate_manifest_files_exist(
        rows
    )

    validate_manifest_files_exist(
        test_rows
    )

    return (
        train_manifest,
        rows,
    )


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

    (
        train_manifest,
        train_rows,
    ) = load_training_data(
        experiment
    )

    (
        model,
        processor,
        architecture,
    ) = build_history_model(
        experiment,
        language=(
            experiment
            .training_language
        ),
    )

    counts = parameter_summary(
        model
    )

    human_layers = [
        layer + 1
        for layer in (
            experiment
            .history_fusion_layers
        )
    ]

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"HIST{experiment.history_length} "
            "FUSION: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

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
            f"History fusion layers: "
            f"{human_layers}"
        )

        print(
            f"Training manifest: "
            f"{train_manifest}"
        )

        print(
            f"Training examples: "
            f"{len(train_rows)}"
        )

        print(
            f"Trainable parameters: "
            f"{counts['trainable_parameters']:,}"
        )

        print(
            f"Warmup steps: "
            f"{config.WARMUP_STEPS}"
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
            "No decoder start token."
        )

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=(
            decoder_start_token_id
        ),
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
                False
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

    # Guard against accidental warmup
    # changes in the shared builder.

    if (
        int(
            training_args
            .warmup_steps
        )
        != int(
            config.WARMUP_STEPS
        )
    ):
        raise RuntimeError(
            "Warmup-step mismatch.\n"
            f"Config: "
            f"{config.WARMUP_STEPS}\n"
            f"TrainingArguments: "
            f"{training_args.warmup_steps}"
        )

    if (
        float(
            training_args
            .warmup_ratio
        )
        != float(
            config.WARMUP_RATIO
        )
    ):
        raise RuntimeError(
            "Warmup-ratio mismatch.\n"
            f"Config: "
            f"{config.WARMUP_RATIO}\n"
            f"TrainingArguments: "
            f"{training_args.warmup_ratio}"
        )

    classification = (
        build_classification(
            family=config.FAMILY,
            transfer_type=(
                config.TRANSFER_TYPE
            ),
            domain_category=(
                experiment
                .domain_category
            ),
        )
    )

    hyperparameters = dict(
        config
        .HYPERPARAMETER_SELECTION
    )

    hyperparameters[
        "actual_experiment"
    ] = {
        "branch_domains": list(
            experiment.branch_domains
        ),
        "target_domain": (
            experiment.target_domain
        ),
        "history_length": (
            experiment.history_length
        ),
        "history_width": (
            config.HISTORY_WIDTH
        ),
        "history_fusion_layers": list(
            experiment
            .history_fusion_layers
        ),
        "trainable_parameters": (
            experiment
            .expected_trainable_parameters
        ),
    }

    metadata = (
        save_training_metadata(
            path=(
                experiment.metadata_path
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
                config.BASE_MODEL
            ),
            language=(
                experiment
                .training_language
            ),
            task=config.TASK,
            hyperparameter_selection=(
                hyperparameters
            ),
            architecture=(
                architecture
            ),
            description=(
                experiment.description
            ),
        )
        if (
            resume_from_checkpoint
            is None
        )
        else load_json_object(
            experiment.metadata_path
        )
    )

    distributed_barrier()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=(
            train_dataset
        ),
        eval_dataset=None,
        data_collator=collator,
        processing_class=processor,
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
                experiment
                .model_directory
            ),
        )

    distributed_barrier()


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