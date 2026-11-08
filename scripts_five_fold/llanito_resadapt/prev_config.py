from __future__ import annotations

from pathlib import Path

# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Experiment

# Keep this experiment named exactly this way.
BASE_EXPERIMENT_NAME = "llanito_resadapt"

# Valid values:
#   "encoder"
#   "decoder"
#   "encoder_decoder"
ADAPTER_MODULE = "encoder"

# Valid values:
#   "all"
#   "first6"
#   "last6"
#   "first12"
#   "last12"
#   "custom"
ADAPTER_POSITION = "custom"

# Only used when ADAPTER_POSITION == "custom".
CUSTOM_ADAPTER_LAYERS = [20]

WHISPER_MEDIUM_ENCODER_LAYERS = 24
WHISPER_MEDIUM_DECODER_LAYERS = 24


def validate_adapter_module() -> None:
    valid = {
        "encoder",
        "decoder",
        "encoder_decoder",
    }

    if ADAPTER_MODULE not in valid:
        raise ValueError(
            f"Unknown ADAPTER_MODULE: {ADAPTER_MODULE!r}. "
            f"Valid values: {sorted(valid)}"
        )


def resolve_layer_count() -> int:
    validate_adapter_module()

    if ADAPTER_MODULE == "encoder":
        return WHISPER_MEDIUM_ENCODER_LAYERS

    if ADAPTER_MODULE == "decoder":
        return WHISPER_MEDIUM_DECODER_LAYERS

    if (
        WHISPER_MEDIUM_ENCODER_LAYERS
        != WHISPER_MEDIUM_DECODER_LAYERS
    ):
        raise ValueError(
            "encoder_decoder mode requires matching encoder "
            "and decoder layer counts in this implementation."
        )

    return WHISPER_MEDIUM_ENCODER_LAYERS


ADAPTER_LAYER_COUNT = resolve_layer_count()


def resolve_adapter_layers() -> list[int]:
    if ADAPTER_POSITION == "all":
        layers = list(
            range(
                ADAPTER_LAYER_COUNT
            )
        )

    elif ADAPTER_POSITION == "first6":
        layers = list(
            range(
                0,
                min(
                    6,
                    ADAPTER_LAYER_COUNT,
                ),
            )
        )

    elif ADAPTER_POSITION == "last6":
        layers = list(
            range(
                max(
                    0,
                    ADAPTER_LAYER_COUNT - 6,
                ),
                ADAPTER_LAYER_COUNT,
            )
        )

    elif ADAPTER_POSITION == "first12":
        layers = list(
            range(
                0,
                min(
                    12,
                    ADAPTER_LAYER_COUNT,
                ),
            )
        )

    elif ADAPTER_POSITION == "last12":
        layers = list(
            range(
                max(
                    0,
                    ADAPTER_LAYER_COUNT - 12,
                ),
                ADAPTER_LAYER_COUNT,
            )
        )

    elif ADAPTER_POSITION == "custom":
        layers = list(
            CUSTOM_ADAPTER_LAYERS
        )

    else:
        raise ValueError(
            f"Unknown ADAPTER_POSITION: {ADAPTER_POSITION!r}"
        )

    if not layers:
        raise ValueError(
            "At least one adapter layer is required."
        )

    if len(set(layers)) != len(layers):
        raise ValueError(
            "CUSTOM_ADAPTER_LAYERS contains duplicate indices."
        )

    invalid = [
        layer
        for layer in layers
        if (
            layer < 0
            or layer >= ADAPTER_LAYER_COUNT
        )
    ]

    if invalid:
        raise ValueError(
            f"Invalid adapter layer indices: {invalid}. "
            f"Valid range is 0..{ADAPTER_LAYER_COUNT - 1}."
        )

    return sorted(
        layers
    )


ADAPTER_LAYERS = resolve_adapter_layers()


def adapter_position_slug() -> str:
    if ADAPTER_POSITION != "custom":
        return ADAPTER_POSITION

    return (
        "custom_"
        + "-".join(
            str(layer)
            for layer in ADAPTER_LAYERS
        )
    )


ADAPTER_POSITION_SLUG = adapter_position_slug()

EXPERIMENT_NAME = (
    f"{BASE_EXPERIMENT_NAME}_"
    f"{ADAPTER_MODULE}_"
    f"{ADAPTER_POSITION_SLUG}"
)

# This resolves to:
# llanito_resadapt_encoder_all

# Paths

# Reuse the exact same authoritative five folds and fixed test as fullLoRA.
PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

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

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

# Whisper

BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "es"
TASK = "transcribe"

# Folds

FOLD_NUMBERS = [1, 2, 3, 4, 5]

# None = run all folds.
# [3] = only fold 3.
RUN_FOLDS: list[int] | None = None

# Fold replacement

CLEAR_SELECTED_FOLD_BEFORE_TRAINING = True
REBUILD_AGGREGATES_FROM_SAVED_FOLDS = True

# Adapter architecture

ADAPTER_BOTTLENECK = 32
ADAPTER_DROPOUT = 0.05

# Backward compatibility

# Old encoder-only residual adapter checkpoints created before ADAPTER_MODULE
# existed are accepted and interpreted as encoder adapters.
ALLOW_LEGACY_ENCODER_CHECKPOINTS = True

# Old state dicts with keys such as:
#   layer_0.down.weight
# are accepted in addition to new keys such as:
#   encoder.layer_0.down.weight
ALLOW_LEGACY_STATE_KEYS = True

# Training

NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 40

TRAIN_BATCH_SIZE_PER_GPU = 1
VALIDATION_BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 8

USE_FP16 = True
USE_BF16 = False
USE_GRADIENT_CHECKPOINTING = False

LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 2
DATALOADER_WORKERS = 2
RANDOM_SEED = 2026

RESUME_FROM_LATEST_CHECKPOINT = False

MAX_TRAIN_RECORDINGS: int | None = None
MAX_VALIDATION_RECORDINGS: int | None = None

# Evaluation

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1

EVALUATE_VANILLA_BASELINE = True
MAX_TEST_RECORDINGS: int | None = None
