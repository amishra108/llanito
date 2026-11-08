from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    WhisperConfig,
    WhisperProcessor,
)

from transformers.masking_utils import (
    create_causal_mask,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)

from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
)


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.spt_m import config

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


# Decoder labels

def shift_tokens_right(
    labels: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    shifted = labels.new_zeros(
        labels.shape
    )

    shifted[:, 1:] = (
        labels[:, :-1].clone()
    )

    shifted[:, 0] = (
        decoder_start_token_id
    )

    shifted.masked_fill_(
        shifted == -100,
        pad_token_id,
    )

    return shifted


# Prompt bottleneck

class PromptBottleneck(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck_size: int,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(
            hidden_size,
            bottleneck_size,
        )

        self.up = nn.Linear(
            bottleneck_size,
            hidden_size,
        )

        # Start the residual reparameterizer
        # as an exact identity operation.

        nn.init.zeros_(
            self.up.weight
        )

        nn.init.zeros_(
            self.up.bias
        )

    def forward(
        self,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        residual = prompt

        transformed = self.down(
            prompt
        )

        transformed = F.gelu(
            transformed
        )

        transformed = self.up(
            transformed
        )

        return (
            residual
            + transformed
        )


# Regular decoder-input SPT

class DecoderInputSoftPromptWhisper(
    WhisperForConditionalGeneration
):
    def __init__(
        self,
        model_config: WhisperConfig,
    ) -> None:
        super().__init__(
            model_config
        )

        prompt_length = getattr(
            model_config,
            "soft_prompt_length",
            None,
        )

        if prompt_length is None:
            raise ValueError(
                "Missing soft_prompt_length."
            )

        self.prompt_length = int(
            prompt_length
        )

        self.soft_prompt = nn.Parameter(
            torch.empty(
                self.prompt_length,
                int(
                    model_config.d_model
                ),
            )
        )

        nn.init.normal_(
            self.soft_prompt,
            mean=0.0,
            std=config.PROMPT_INIT_STD,
        )

    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop(
            "labels",
            None,
        )

        kwargs[
            "use_cache"
        ] = False

        return super().generate(
            *args,
            **kwargs,
        )

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        kwargs.pop(
            "cache_position",
            None,
        )

        if (
            decoder_input_ids is None
            and decoder_inputs_embeds
            is None
        ):
            if labels is None:
                raise ValueError(
                    "Decoder inputs or labels "
                    "are required."
                )

            decoder_input_ids = (
                shift_tokens_right(
                    labels,
                    int(
                        self.config.pad_token_id
                    ),
                    int(
                        self.config
                        .decoder_start_token_id
                    ),
                )
            )

        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = (
                self.model
                .decoder
                .embed_tokens(
                    decoder_input_ids
                )
            )

        batch_size = int(
            decoder_inputs_embeds.shape[0]
        )

        prompt = (
            self.soft_prompt
            .to(
                dtype=(
                    decoder_inputs_embeds
                    .dtype
                ),
                device=(
                    decoder_inputs_embeds
                    .device
                ),
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        prompted_decoder_embeds = (
            torch.cat(
                [
                    prompt,
                    decoder_inputs_embeds,
                ],
                dim=1,
            )
        )

        model_outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=encoder_outputs,
            past_key_values=None,
            decoder_inputs_embeds=(
                prompted_decoder_embeds
            ),
            decoder_position_ids=None,
            use_cache=False,
            **kwargs,
        )

        hidden_states = (
            model_outputs
            .last_hidden_state
        )

        all_logits = self.proj_out(
            hidden_states
        )

        token_logits = (
            all_logits[
                :,
                self.prompt_length:,
                :,
            ]
        )

        loss = None

        if labels is not None:
            if (
                token_logits.shape[1]
                != labels.shape[1]
            ):
                raise RuntimeError(
                    "Token/label length "
                    "mismatch."
                )

            loss = CrossEntropyLoss(
                ignore_index=-100
            )(
                token_logits
                .float()
                .reshape(
                    -1,
                    token_logits.shape[-1],
                ),
                labels.reshape(-1),
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=(
                token_logits
                if labels is not None
                else all_logits
            ),
            past_key_values=None,
        )


# Deep decoder SPT

class DeepDecoderSoftPromptWhisper(
    WhisperForConditionalGeneration
):
    def __init__(
        self,
        model_config: WhisperConfig,
    ) -> None:
        super().__init__(
            model_config
        )

        prompt_length = int(
            getattr(
                model_config,
                "soft_prompt_length",
            )
        )

        architecture = str(
            getattr(
                model_config,
                "soft_prompt_architecture",
            )
        )

        self.prompt_length = (
            prompt_length
        )

        self.prompt_architecture = (
            architecture
        )

        hidden_size = int(
            model_config.d_model
        )

        decoder_layers = int(
            model_config.decoder_layers
        )

        self.deep_prompts = nn.Parameter(
            torch.empty(
                decoder_layers,
                prompt_length,
                hidden_size,
            )
        )

        nn.init.normal_(
            self.deep_prompts,
            mean=0.0,
            std=config.PROMPT_INIT_STD,
        )

        if (
            architecture
            == "decoder_deep_bneck"
        ):
            bottleneck_size = int(
                getattr(
                    model_config,
                    "soft_prompt_bottleneck",
                )
            )

            self.prompt_bottlenecks = (
                nn.ModuleList(
                    [
                        PromptBottleneck(
                            hidden_size=(
                                hidden_size
                            ),
                            bottleneck_size=(
                                bottleneck_size
                            ),
                        )
                        for _ in range(
                            decoder_layers
                        )
                    ]
                )
            )

        else:
            self.prompt_bottlenecks = None

    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop(
            "labels",
            None,
        )

        # Deep prompt reconstruction is
        # easiest and safest without KV
        # caching because every generation
        # step rebuilds each layer's prompt.

        kwargs["use_cache"] = False

        return super().generate(
            *args,
            **kwargs,
        )

    def get_layer_prompt(
        self,
        *,
        layer_index: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        prompt = (
            self.deep_prompts[
                layer_index
            ]
        )

        if (
            self.prompt_bottlenecks
            is not None
        ):
            prompt = (
                self.prompt_bottlenecks[
                    layer_index
                ](
                    prompt
                )
            )

        return (
            prompt
            .to(
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

    def run_deep_decoder(
        self,
        *,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        decoder = self.model.decoder

        inputs_embeds = (
            decoder.embed_tokens(
                decoder_input_ids
            )
            * decoder.embed_scale
        )

        positions = (
            decoder.embed_positions(
                decoder_input_ids,
                past_key_values_length=0,
                position_ids=None,
            )
        )

        hidden_states = (
            inputs_embeds
            + positions.to(
                inputs_embeds.device
            )
        )

        hidden_states = F.dropout(
            hidden_states,
            p=decoder.dropout,
            training=self.training,
        )

        batch_size = int(
            hidden_states.shape[0]
        )

        for (
            layer_index,
            decoder_layer,
        ) in enumerate(
            decoder.layers
        ):
            prompt = (
                self.get_layer_prompt(
                    layer_index=(
                        layer_index
                    ),
                    batch_size=(
                        batch_size
                    ),
                    dtype=(
                        hidden_states.dtype
                    ),
                    device=(
                        hidden_states.device
                    ),
                )
            )

            # Each layer gets:
            #
            # [its own prompt] + [tokens]
            #
            # The prior layer's prompt is
            # already gone.

            layer_input = torch.cat(
                [
                    prompt,
                    hidden_states,
                ],
                dim=1,
            )

            # Build the causal mask for
            # exactly this temporary
            # prompt+token sequence.
            #
            # Prompt tokens are before all
            # ordinary decoder tokens, so
            # each token may attend to the
            # prompt while remaining causal
            # with respect to future tokens.

            causal_mask = (
                create_causal_mask(
                    config=self.config,
                    inputs_embeds=(
                        layer_input
                    ),
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=None,
                )
            )

            layer_output = (
                decoder_layer(
                    layer_input,
                    causal_mask,
                    encoder_hidden_states,
                    encoder_attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                )
            )

            if isinstance(
                layer_output,
                tuple,
            ):
                layer_output = (
                    layer_output[0]
                )

            # Remove this layer's prompt
            # positions before entering the
            # next decoder layer.

            hidden_states = (
                layer_output[
                    :,
                    self.prompt_length:,
                    :,
                ]
            )

        hidden_states = (
            decoder.layer_norm(
                hidden_states
            )
        )

        return hidden_states

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        kwargs.pop(
            "cache_position",
            None,
        )

        if decoder_inputs_embeds is not None:
            raise ValueError(
                "Deep SPT expects "
                "decoder_input_ids rather "
                "than decoder_inputs_embeds."
            )

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError(
                    "Deep SPT requires "
                    "decoder inputs or labels."
                )

            decoder_input_ids = (
                shift_tokens_right(
                    labels,
                    int(
                        self.config.pad_token_id
                    ),
                    int(
                        self.config
                        .decoder_start_token_id
                    ),
                )
            )

        # Encoder stays ordinary frozen
        # Whisper.

        if encoder_outputs is None:
            encoder_outputs = (
                self.model.encoder(
                    input_features,
                    attention_mask=(
                        attention_mask
                    ),
                    return_dict=True,
                )
            )

        if isinstance(
            encoder_outputs,
            BaseModelOutput,
        ):
            encoder_hidden_states = (
                encoder_outputs
                .last_hidden_state
            )

        elif hasattr(
            encoder_outputs,
            "last_hidden_state",
        ):
            encoder_hidden_states = (
                encoder_outputs
                .last_hidden_state
            )

        else:
            encoder_hidden_states = (
                encoder_outputs[0]
            )

        hidden_states = (
            self.run_deep_decoder(
                decoder_input_ids=(
                    decoder_input_ids
                ),
                encoder_hidden_states=(
                    encoder_hidden_states
                ),
            )
        )

        logits = self.proj_out(
            hidden_states
        )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Deep SPT produced "
                "non-finite logits."
            )

        loss = None

        if labels is not None:
            if (
                logits.shape[1]
                != labels.shape[1]
            ):
                raise RuntimeError(
                    "Deep SPT token/label "
                    "length mismatch.\n"
                    f"Logits: {logits.shape}\n"
                    f"Labels: {labels.shape}"
                )

            loss = CrossEntropyLoss(
                ignore_index=-100
            )(
                logits
                .float()
                .reshape(
                    -1,
                    logits.shape[-1],
                ),
                labels.reshape(-1),
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Deep SPT produced "
                    "non-finite loss."
                )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )


# Model construction

def build_spt_model(
    *,
    experiment: (
        config.ExperimentConfig
    ),
) -> WhisperForConditionalGeneration:
    model_config = (
        WhisperConfig
        .from_pretrained(
            config.BASE_MODEL_DIRECTORY
        )
    )

    setattr(
        model_config,
        "soft_prompt_length",
        int(
            experiment.prompt_length
        ),
    )

    setattr(
        model_config,
        "soft_prompt_architecture",
        experiment.architecture,
    )

    if (
        experiment.bottleneck_size
        is not None
    ):
        setattr(
            model_config,
            "soft_prompt_bottleneck",
            int(
                experiment
                .bottleneck_size
            ),
        )

    if (
        experiment.architecture
        == "decoder_input"
    ):
        model = (
            DecoderInputSoftPromptWhisper
            .from_pretrained(
                config
                .BASE_MODEL_DIRECTORY,
                config=model_config,
                attn_implementation="eager",
            )
        )

    else:
        model = (
            DeepDecoderSoftPromptWhisper
            .from_pretrained(
                config
                .BASE_MODEL_DIRECTORY,
                config=model_config,
                attn_implementation="eager",
            )
        )

    # Freeze Whisper and all newly
    # constructed prompt components first.

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    decoder_embeddings = (
        model
        .model
        .decoder
        .embed_tokens
        .weight[
            :experiment.prompt_length
        ]
        .detach()
    )

    # Prompt initialization mirrors the
    # working existing SPT implementation.

    if (
        experiment.architecture
        == "decoder_input"
    ):
        with torch.no_grad():
            model.soft_prompt.copy_(
                decoder_embeddings.to(
                    dtype=(
                        model.soft_prompt.dtype
                    ),
                    device=(
                        model.soft_prompt.device
                    ),
                )
            )

        model.soft_prompt.requires_grad = (
            True
        )

    else:
        with torch.no_grad():
            for layer_index in range(
                config.DECODER_LAYERS
            ):
                model.deep_prompts[
                    layer_index
                ].copy_(
                    decoder_embeddings.to(
                        dtype=(
                            model
                            .deep_prompts
                            .dtype
                        ),
                        device=(
                            model
                            .deep_prompts
                            .device
                        ),
                    )
                )

        model.deep_prompts.requires_grad = (
            True
        )

        if (
            model.prompt_bottlenecks
            is not None
        ):
            for parameter in (
                model
                .prompt_bottlenecks
                .parameters()
            ):
                parameter.requires_grad = (
                    True
                )

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    model.config.use_cache = False

    if getattr(
        model,
        "generation_config",
        None,
    ) is not None:
        model.generation_config.use_cache = (
            False
        )

    return model


# Trainable-count verification

def expected_trainable_parameters(
    experiment: (
        config.ExperimentConfig
    ),
) -> int:
    if (
        experiment.architecture
        == "decoder_input"
    ):
        return (
            experiment.prompt_length
            * config.HIDDEN_SIZE
        )

    prompt_parameters = (
        config.DECODER_LAYERS
        * experiment.prompt_length
        * config.HIDDEN_SIZE
    )

    if (
        experiment.architecture
        == "decoder_deep"
    ):
        return prompt_parameters

    if (
        experiment.architecture
        == "decoder_deep_bneck"
    ):
        b = int(
            experiment.bottleneck_size
        )

        per_layer = (
            config.HIDDEN_SIZE * b
            + b
            + b * config.HIDDEN_SIZE
            + config.HIDDEN_SIZE
        )

        return (
            prompt_parameters
            + (
                config.DECODER_LAYERS
                * per_layer
            )
        )

    raise ValueError(
        "Unknown SPT architecture."
    )


def verify_trainable_parameters(
    *,
    model: nn.Module,
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    expected = (
        expected_trainable_parameters(
            experiment
        )
    )

    actual = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if actual != expected:
        names = [
            name
            for (
                name,
                parameter,
            ) in model.named_parameters()
            if parameter.requires_grad
        ]

        raise RuntimeError(
            "Unexpected SPT trainable "
            "parameter count.\n"
            f"Expected: {expected:,}\n"
            f"Found: {actual:,}\n"
            f"Trainable names: "
            f"{names[:100]}"
        )


# Save/load prompt payload

def collect_prompt_state(
    model: nn.Module,
) -> dict[
    str,
    torch.Tensor,
]:
    base = model

    while hasattr(
        base,
        "module",
    ):
        base = base.module

    state = {}

    for (
        name,
        parameter,
    ) in base.named_parameters():
        if (
            name == "soft_prompt"
            or name == "deep_prompts"
            or name.startswith(
                "prompt_bottlenecks."
            )
        ):
            state[
                name
            ] = (
                parameter
                .detach()
                .cpu()
                .clone()
            )

    if not state:
        raise RuntimeError(
            "No prompt parameters found."
        )

    return state


def save_spt_model(
    *,
    model: nn.Module,
    processor: WhisperProcessor,
    training_args: Any,
    trainer: Any,
    experiment: (
        config.ExperimentConfig
    ),
) -> None:
    experiment.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        collect_prompt_state(
            model
        ),
        (
            experiment
            .model_directory
            / "prompt_state.pt"
        ),
    )

    save_json(
        (
            experiment
            .model_directory
            / "prompt_config.json"
        ),
        {
            "format_version": 2,
            "architecture": (
                experiment.architecture
            ),
            "prompt_length": (
                experiment.prompt_length
            ),
            "bottleneck_size": (
                experiment
                .bottleneck_size
            ),
            "hidden_size": (
                config.HIDDEN_SIZE
            ),
            "decoder_layers": (
                config.DECODER_LAYERS
            ),
            "prompt_init": (
                config.PROMPT_INIT
            ),
            "base_model": (
                config.BASE_MODEL
            ),
            "language": (
                config.LANGUAGE
            ),
            "task": (
                config.TASK
            ),
        },
    )

    processor.save_pretrained(
        experiment.model_directory
    )

    torch.save(
        training_args,
        (
            experiment
            .model_directory
            / "training_args.bin"
        ),
    )

    trainer.state.save_to_json(
        str(
            experiment
            .model_directory
            / "trainer_state.json"
        )
    )


def load_saved_spt_model(
    *,
    model_directory: Path,
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    WhisperForConditionalGeneration,
    WhisperProcessor,
]:
    prompt_config_path = (
        model_directory
        / "prompt_config.json"
    )

    prompt_state_path = (
        model_directory
        / "prompt_state.pt"
    )

    if not prompt_config_path.is_file():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{prompt_config_path}"
        )

    if not prompt_state_path.is_file():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{prompt_state_path}"
        )

    with prompt_config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        saved = json.load(
            handle
        )

    experiment = (
        config.ExperimentConfig(
            model_name=(
                model_directory
                .parent
                .name
            ),
            prompt_length=int(
                saved[
                    "prompt_length"
                ]
            ),
            architecture=str(
                saved[
                    "architecture"
                ]
            ),
            bottleneck_size=(
                None
                if saved.get(
                    "bottleneck_size"
                ) is None
                else int(
                    saved[
                        "bottleneck_size"
                    ]
                )
            ),
            description=(
                "Reconstructed saved SPT."
            ),
        )
    )

    model = build_spt_model(
        experiment=experiment
    )

    state = torch.load(
        prompt_state_path,
        map_location="cpu",
        weights_only=True,
    )

    own_state = model.state_dict()

    unexpected = [
        name
        for name in state
        if name not in own_state
    ]

    if unexpected:
        raise RuntimeError(
            "Saved SPT contains "
            "unexpected parameters:\n"
            f"{unexpected}"
        )

    with torch.no_grad():
        for (
            name,
            tensor,
        ) in state.items():
            own_state[
                name
            ].copy_(
                tensor.to(
                    dtype=(
                        own_state[
                            name
                        ].dtype
                    ),
                    device=(
                        own_state[
                            name
                        ].device
                    ),
                )
            )

    if torch_dtype is not None:
        model = model.to(
            dtype=torch_dtype
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


# Metadata

def build_architecture_metadata(
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[str, Any]:
    return {
        "adaptation": "spt",
        "architecture": (
            experiment.architecture
        ),
        "placement": "decoder",
        "prompt_length": (
            experiment.prompt_length
        ),
        "number_of_prompt_sets": (
            1
            if (
                experiment.architecture
                == "decoder_input"
            )
            else config.DECODER_LAYERS
        ),
        "hidden_size": (
            config.HIDDEN_SIZE
        ),
        "bottleneck_size": (
            experiment.bottleneck_size
        ),
        "prompt_init": (
            config.PROMPT_INIT
        ),
        "base_model_root": str(
            config.BASE_MODEL_ROOT
        ),
        "whisper_frozen": True,
        "expected_trainable_parameters": (
            expected_trainable_parameters(
                experiment
            )
        ),
    }


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
        != config
        .EXPECTED_TRAINING_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected Miami training "
            "size.\n"
            f"Expected: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}\n"
            f"Found: {len(train_rows)}"
        )

    if (
        len(test_rows)
        != config
        .EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected Llanito fixed "
            "test size."
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name=(
            "Miami SPT training"
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
            "Miami SPT training"
        ),
        second_name=(
            "Llanito fixed test"
        ),
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

def load_existing_metadata(
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[str, Any]:
    with (
        experiment.metadata_path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(
            handle
        )

    if (
        metadata.get(
            "model_name"
        )
        != experiment.model_name
    ):
        raise RuntimeError(
            "Metadata belongs to a "
            "different model."
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

    processor = (
        WhisperProcessor
        .from_pretrained(
            config.BASE_MODEL_DIRECTORY
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=config.LANGUAGE,
        task=config.TASK,
        predict_timestamps=False,
    )

    model = build_spt_model(
        experiment=experiment
    )

    verify_trainable_parameters(
        model=model,
        experiment=experiment,
    )

    counts = parameter_summary(
        model
    )

    if is_main_process():
        print()
        print("=" * 80)
        print(
            f"MIAMI SPT: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

        print(
            f"Architecture: "
            f"{experiment.architecture}"
        )

        print(
            f"Prompt length: "
            f"{experiment.prompt_length}"
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
            f"Trainable %: "
            f"{counts['trainable_percent']:.8f}%"
        )

    train_dataset = (
        WhisperManifestDataset(
            rows=train_rows,
            processor=processor,
            audio_column="audio_path",
            transcript_column="transcript",
        )
    )

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=int(
            model.config
            .decoder_start_token_id
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
            optimizer=config.OPTIMIZER,
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
        save_spt_model(
            model=trainer.model,
            processor=processor,
            training_args=(
                training_args
            ),
            trainer=trainer,
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