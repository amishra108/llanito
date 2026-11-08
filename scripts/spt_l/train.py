from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    WhisperConfig,
    WhisperProcessor,
)
from transformers.modeling_outputs import (
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


from scripts.spt_l import config

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

    shifted[
        :,
        1:,
    ] = (
        labels[
            :,
            :-1,
        ]
        .clone()
    )

    shifted[
        :,
        0,
    ] = (
        decoder_start_token_id
    )

    shifted.masked_fill_(
        shifted == -100,
        pad_token_id,
    )

    return shifted


# Decoder-input SPT

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
                "Whisper config is missing "
                "soft_prompt_length."
            )

        self.prompt_length = int(
            prompt_length
        )

        if self.prompt_length <= 0:
            raise ValueError(
                "soft_prompt_length must "
                "be positive."
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
            std=(
                config.PROMPT_INIT_STD
            ),
        )

    def _expanded_prompt(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            self.soft_prompt
            .to(
                dtype=dtype,
                device=device,
            )
            .unsqueeze(
                0
            )
            .expand(
                batch_size,
                -1,
                -1,
            )
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

        # During training the Trainer gives
        # labels but may not explicitly give
        # decoder_input_ids.

        if (
            decoder_input_ids is None
            and decoder_inputs_embeds
            is None
        ):
            if labels is None:
                raise ValueError(
                    "Decoder-input SPT requires "
                    "decoder inputs or labels."
                )

            if (
                self.config.pad_token_id
                is None
            ):
                raise ValueError(
                    "Whisper pad_token_id "
                    "is missing."
                )

            if (
                self.config
                .decoder_start_token_id
                is None
            ):
                raise ValueError(
                    "Whisper "
                    "decoder_start_token_id "
                    "is missing."
                )

            decoder_input_ids = (
                shift_tokens_right(
                    labels,
                    int(
                        self.config
                        .pad_token_id
                    ),
                    int(
                        self.config
                        .decoder_start_token_id
                    ),
                )
            )

        # Convert normal decoder token IDs
        # into embeddings.

        if decoder_inputs_embeds is None:
            if decoder_input_ids is None:
                raise ValueError(
                    "decoder_input_ids "
                    "are missing."
                )

            decoder_inputs_embeds = (
                self.model
                .decoder
                .embed_tokens(
                    decoder_input_ids
                )
            )

        if (
            decoder_inputs_embeds.ndim
            != 3
        ):
            raise RuntimeError(
                "Unexpected decoder embedding "
                "shape: "
                f"{decoder_inputs_embeds.shape}"
            )

        batch_size = int(
            decoder_inputs_embeds
            .shape[
                0
            ]
        )

        # Prepend m learned continuous
        # decoder prompt embeddings.

        prompt = (
            self._expanded_prompt(
                batch_size,
                dtype=(
                    decoder_inputs_embeds
                    .dtype
                ),
                device=(
                    decoder_inputs_embeds
                    .device
                ),
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

        # We deliberately disable KV caching.
        #
        # Each decoding step therefore sees:
        #
        # [soft prompts] + [decoder tokens]
        #
        # which keeps the implementation
        # simple and consistent between
        # training and generation.

        model_outputs = self.model(
            input_features=(
                input_features
            ),
            attention_mask=(
                attention_mask
            ),
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=(
                encoder_outputs
            ),
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

        if not torch.isfinite(
            hidden_states
        ).all():
            raise RuntimeError(
                "SPT decoder produced "
                "non-finite hidden states."
            )

        all_logits = (
            self.proj_out(
                hidden_states
            )
        )

        if not torch.isfinite(
            all_logits
        ).all():
            raise RuntimeError(
                "SPT decoder produced "
                "non-finite logits."
            )

        # Prompt positions have no reference
        # labels. Remove them when computing
        # the normal token prediction loss.

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
                    "SPT token/label "
                    "length mismatch.\n"
                    f"Token logits: "
                    f"{token_logits.shape}\n"
                    f"Labels: "
                    f"{labels.shape}"
                )

            valid_label_count = int(
                labels.ne(
                    -100
                )
                .sum()
                .item()
            )

            if valid_label_count == 0:
                raise RuntimeError(
                    "All SPT labels are -100."
                )

            loss = CrossEntropyLoss(
                ignore_index=-100
            )(
                token_logits
                .float()
                .reshape(
                    -1,
                    token_logits
                    .shape[-1],
                ),
                labels.reshape(
                    -1
                ),
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "SPT produced "
                    "non-finite loss."
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


# Build model

def build_spt_model(
    *,
    prompt_length: int,
) -> DecoderInputSoftPromptWhisper:
    if prompt_length <= 0:
        raise ValueError(
            "Prompt length must "
            "be positive."
        )

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
            prompt_length
        ),
    )

    model = (
        DecoderInputSoftPromptWhisper
        .from_pretrained(
            config.BASE_MODEL_DIRECTORY,
            config=model_config,
            attn_implementation="eager",
        )
    )

    # Freeze absolutely everything.

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    # Initialise the prompt from existing
    # Whisper decoder token embeddings.
    #
    # This follows the implementation you
    # were already using rather than random
    # prompt initialization.

    with torch.no_grad():
        source_embeddings = (
            model
            .model
            .decoder
            .embed_tokens
            .weight[
                :prompt_length
            ]
            .detach()
            .to(
                dtype=(
                    model.soft_prompt.dtype
                ),
                device=(
                    model.soft_prompt.device
                ),
            )
        )

        expected_shape = (
            prompt_length,
            int(
                model.config.d_model
            ),
        )

        if tuple(
            source_embeddings.shape
        ) != expected_shape:
            raise RuntimeError(
                "Decoder prompt initialisation "
                "shape is wrong.\n"
                f"Expected: "
                f"{expected_shape}\n"
                f"Found: "
                f"{tuple(source_embeddings.shape)}"
            )

        model.soft_prompt.copy_(
            source_embeddings
        )

    # Only the soft prompt may train.

    model.soft_prompt.requires_grad = (
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

    trainable_names = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    if trainable_names != [
        "soft_prompt"
    ]:
        raise RuntimeError(
            "Only soft_prompt may be "
            "trainable.\n"
            f"Found: "
            f"{trainable_names[:50]}"
        )

    expected_trainable = (
        int(
            prompt_length
        )
        * int(
            model.config.d_model
        )
    )

    actual_trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if (
        actual_trainable
        != expected_trainable
    ):
        raise RuntimeError(
            "Unexpected SPT trainable "
            "parameter count.\n"
            f"Expected: "
            f"{expected_trainable}\n"
            f"Found: "
            f"{actual_trainable}"
        )

    if not torch.isfinite(
        model.soft_prompt
    ).all():
        raise RuntimeError(
            "SPT prompt contains "
            "non-finite values after "
            "initialisation."
        )

    return model


# Saved model verification

def verify_saved_spt_model(
    *,
    model_directory: Path,
    expected_prompt_length: (
        int | None
    ) = None,
) -> None:
    prompt_config_path = (
        model_directory
        / "prompt_config.json"
    )

    prompt_state_path = (
        model_directory
        / "prompt_state.pt"
    )

    if not prompt_config_path.exists():
        raise FileNotFoundError(
            "Saved SPT model is missing "
            "prompt_config.json:\n"
            f"{prompt_config_path}"
        )

    if not prompt_state_path.exists():
        raise FileNotFoundError(
            "Saved SPT model is missing "
            "prompt_state.pt:\n"
            f"{prompt_state_path}"
        )

    with prompt_config_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        prompt_config = json.load(
            input_file
        )

    if not isinstance(
        prompt_config,
        dict,
    ):
        raise ValueError(
            "prompt_config.json must "
            "contain a JSON object."
        )

    architecture = str(
        prompt_config.get(
            "architecture",
            "",
        )
    ).strip()

    if (
        architecture
        != config.PROMPT_ARCHITECTURE
    ):
        raise ValueError(
            "Unsupported SPT architecture.\n"
            f"Expected: "
            f"{config.PROMPT_ARCHITECTURE}\n"
            f"Found: "
            f"{architecture!r}"
        )

    if (
        "prompt_length"
        not in prompt_config
    ):
        raise ValueError(
            "prompt_config.json is "
            "missing prompt_length."
        )

    prompt_length = int(
        prompt_config[
            "prompt_length"
        ]
    )

    if prompt_length <= 0:
        raise ValueError(
            "Saved prompt length must "
            "be positive."
        )

    if (
        expected_prompt_length
        is not None
        and prompt_length
        != expected_prompt_length
    ):
        raise RuntimeError(
            "Saved prompt length does not "
            "match the experiment.\n"
            f"Expected: "
            f"{expected_prompt_length}\n"
            f"Found: "
            f"{prompt_length}"
        )

    hidden_size = int(
        prompt_config.get(
            "hidden_size",
            1024,
        )
    )

    state = torch.load(
        prompt_state_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "prompt_state.pt must "
            "contain a dictionary."
        )

    soft_prompt = state.get(
        "soft_prompt"
    )

    if soft_prompt is None:
        raise RuntimeError(
            "prompt_state.pt contains "
            "no soft_prompt."
        )

    if not isinstance(
        soft_prompt,
        torch.Tensor,
    ):
        raise TypeError(
            "Saved soft_prompt is not "
            "a torch.Tensor."
        )

    expected_shape = (
        prompt_length,
        hidden_size,
    )

    if tuple(
        soft_prompt.shape
    ) != expected_shape:
        raise RuntimeError(
            "Saved soft-prompt shape "
            "is wrong.\n"
            f"Expected: "
            f"{expected_shape}\n"
            f"Found: "
            f"{tuple(soft_prompt.shape)}"
        )

    if not torch.isfinite(
        soft_prompt
    ).all():
        raise RuntimeError(
            "Saved soft prompt contains "
            "non-finite values."
        )


# Saved SPT reconstruction

def load_saved_spt_model(
    *,
    model_directory: Path,
    torch_dtype: (
        torch.dtype | None
    ) = None,
) -> tuple[
    DecoderInputSoftPromptWhisper,
    WhisperProcessor,
]:
    verify_saved_spt_model(
        model_directory=(
            model_directory
        )
    )

    prompt_config_path = (
        model_directory
        / "prompt_config.json"
    )

    prompt_state_path = (
        model_directory
        / "prompt_state.pt"
    )

    with prompt_config_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        prompt_config = json.load(
            input_file
        )

    prompt_length = int(
        prompt_config[
            "prompt_length"
        ]
    )

    hidden_size = int(
        prompt_config[
            "hidden_size"
        ]
    )

    model = build_spt_model(
        prompt_length=(
            prompt_length
        )
    )

    if (
        int(
            model.config.d_model
        )
        != hidden_size
    ):
        raise RuntimeError(
            "Saved SPT hidden size does "
            "not match reconstructed base.\n"
            f"Saved: "
            f"{hidden_size}\n"
            f"Base: "
            f"{model.config.d_model}"
        )

    state = torch.load(
        prompt_state_path,
        map_location="cpu",
        weights_only=True,
    )

    soft_prompt = state[
        "soft_prompt"
    ]

    expected_shape = (
        prompt_length,
        int(
            model.config.d_model
        ),
    )

    if tuple(
        soft_prompt.shape
    ) != expected_shape:
        raise RuntimeError(
            "Saved soft-prompt shape "
            "does not match reconstructed "
            "Whisper model.\n"
            f"Expected: "
            f"{expected_shape}\n"
            f"Found: "
            f"{tuple(soft_prompt.shape)}"
        )

    with torch.no_grad():
        model.soft_prompt.copy_(
            soft_prompt.to(
                dtype=(
                    model.soft_prompt.dtype
                ),
                device=(
                    model.soft_prompt.device
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


# Save SPT model

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

    base = model

    while hasattr(
        base,
        "module",
    ):
        base = base.module

    if not hasattr(
        base,
        "soft_prompt",
    ):
        raise RuntimeError(
            "SPT model has no "
            "soft_prompt parameter."
        )

    soft_prompt = (
        base.soft_prompt
        .detach()
        .cpu()
        .clone()
    )

    expected_shape = (
        experiment.prompt_length,
        int(
            base.config.d_model
        ),
    )

    if tuple(
        soft_prompt.shape
    ) != expected_shape:
        raise RuntimeError(
            "Soft prompt has unexpected "
            "shape before saving.\n"
            f"Expected: "
            f"{expected_shape}\n"
            f"Found: "
            f"{tuple(soft_prompt.shape)}"
        )

    if not torch.isfinite(
        soft_prompt
    ).all():
        raise RuntimeError(
            "Cannot save SPT model because "
            "soft prompt contains non-finite "
            "values."
        )

    # Learned SPT payload

    torch.save(
        {
            "soft_prompt": (
                soft_prompt
            )
        },
        (
            experiment
            .model_directory
            / "prompt_state.pt"
        ),
    )

    # Architecture reconstruction config

    save_json(
        (
            experiment
            .model_directory
            / "prompt_config.json"
        ),
        {
            "format_version": 1,
            "architecture": (
                config
                .PROMPT_ARCHITECTURE
            ),
            "prompt_length": (
                experiment.prompt_length
            ),
            "hidden_size": int(
                base.config.d_model
            ),
            "prompt_init": (
                config.PROMPT_INIT
            ),
            "base_model": (
                config.BASE_MODEL
            ),
            "base_model_root": str(
                config.BASE_MODEL_ROOT
            ),
            "language": (
                config.LANGUAGE
            ),
            "task": (
                config.TASK
            ),
        },
    )

    # Processor/tokenizer

    processor.save_pretrained(
        experiment.model_directory
    )

    # Preserve final training arguments.

    torch.save(
        training_args,
        (
            experiment
            .model_directory
            / "training_args.bin"
        ),
    )

    # Preserve Trainer state.

    trainer.state.save_to_json(
        str(
            experiment
            .model_directory
            / "trainer_state.json"
        )
    )

    # Final integrity check.

    verify_saved_spt_model(
        model_directory=(
            experiment
            .model_directory
        ),
        expected_prompt_length=(
            experiment.prompt_length
        ),
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
            "Cannot resume because model "
            "metadata does not exist:\n"
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


# Architecture metadata

def build_architecture_metadata(
    experiment: (
        config.ExperimentConfig
    ),
) -> dict[str, Any]:
    return {
        "adaptation": "spt",
        "architecture": (
            config.PROMPT_ARCHITECTURE
        ),
        "prompt_length": (
            experiment.prompt_length
        ),
        "hidden_size": 1024,
        "prompt_init": (
            config.PROMPT_INIT
        ),
        "base_model_name_or_path": (
            config.BASE_MODEL
        ),
        "base_model_root": str(
            config.BASE_MODEL_ROOT
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
        != config.EXPECTED_TRAINING_EXAMPLES
    ):
        raise RuntimeError(
            "Unexpected Llanito final "
            "training-set size.\n"
            f"Expected: "
            f"{config.EXPECTED_TRAINING_EXAMPLES}\n"
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
            "Llanito final training"
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
            "final_training"
        ),
        second_name=(
            "fixed_test"
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
            f"FINAL LLANITO SPT: "
            f"{experiment.model_name}"
        )
        print("=" * 80)

        print()
        print(
            "Architecture: "
            "decoder-input soft prompt"
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
            "Validation: none"
        )

        print(
            f"Base model: "
            f"{config.BASE_MODEL_DIRECTORY}"
        )

        print(
            f"Learning rate: "
            f"{config.LEARNING_RATE}"
        )

        print(
            f"Weight decay: "
            f"{config.WEIGHT_DECAY}"
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
            f"FP16: "
            f"{config.USE_FP16}"
        )

        print(
            f"BF16: "
            f"{config.USE_BF16}"
        )

        print(
            f"Seed: "
            f"{config.SEED}"
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
        prompt_length=(
            experiment.prompt_length
        )
    )

    counts = parameter_summary(
        model
    )

    expected_trainable = (
        experiment.prompt_length
        * int(
            model.config.d_model
        )
    )

    if (
        counts[
            "trainable_parameters"
        ]
        != expected_trainable
    ):
        raise RuntimeError(
            "SPT trainable parameter "
            "count is wrong.\n"
            f"Expected: "
            f"{expected_trainable}\n"
            f"Found: "
            f"{counts['trainable_parameters']}"
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
                task=(
                    config.TASK
                ),
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

    # Save permanent final prompt payload.

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

    # Only mark metadata complete after the
    # final prompt payload has been verified.

    if is_main_process():
        update_training_metadata_after_training(
            experiment.metadata_path,
            existing_metadata=(
                metadata
            ),
            trainer=trainer,
            train_result=(
                train_result
            ),
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
            "TRAINING COMPLETE"
        )
        print("=" * 80)

        print()
        print(
            f"Model directory: "
            f"{experiment.model_directory}"
        )

        print(
            f"Prompt state: "
            f"{experiment.model_directory / 'prompt_state.pt'}"
        )

        print(
            f"Prompt config: "
            f"{experiment.model_directory / 'prompt_config.json'}"
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