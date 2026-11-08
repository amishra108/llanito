from __future__ import annotations

import os
from pathlib import Path

# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Experiment family

BASE_EXPERIMENT_NAME = "joint_resadapt"

# These are the three experiments to run.
RUN_ALL_EXPERIMENTS = [
    "encoder",
    "decoder",
    "encoder_decoder",
]

DEFAULT_EXPERIMENT = "encoder"

SELECTED_EXPERIMENT = os.environ.get(
    "RESADAPT_EXPERIMENT",
    DEFAULT_EXPERIMENT,
)

if (
    SELECTED_EXPERIMENT
    not in RUN_ALL_EXPERIMENTS
):
    raise ValueError(
        "Unknown RESADAPT_EXPERIMENT: "
        f"{SELECTED_EXPERIMENT!r}.\n"
        "Available experiments:\n"
        + "\n".join(
            RUN_ALL_EXPERIMENTS
        )
    )

# Adapter module

ADAPTER_MODULE = (
    SELECTED_EXPERIMENT
)

# All three experiments put adapters
# after every selected encoder/decoder layer.

ADAPTER_POSITION = "all"

# Only used if ADAPTER_POSITION == "custom".

CUSTOM_ADAPTER_LAYERS = [
    0,
]

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
            "Unknown ADAPTER_MODULE: "
            f"{ADAPTER_MODULE!r}. "
            f"Valid values: "
            f"{sorted(valid)}"
        )


def resolve_layer_count() -> int:
    validate_adapter_module()

    if ADAPTER_MODULE == "encoder":
        return (
            WHISPER_MEDIUM_ENCODER_LAYERS
        )

    if ADAPTER_MODULE == "decoder":
        return (
            WHISPER_MEDIUM_DECODER_LAYERS
        )

    if (
        WHISPER_MEDIUM_ENCODER_LAYERS
        != WHISPER_MEDIUM_DECODER_LAYERS
    ):
        raise ValueError(
            "encoder_decoder mode requires "
            "matching encoder and decoder "
            "layer counts."
        )

    return (
        WHISPER_MEDIUM_ENCODER_LAYERS
    )


ADAPTER_LAYER_COUNT = (
    resolve_layer_count()
)


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
            "Unknown ADAPTER_POSITION: "
            f"{ADAPTER_POSITION!r}"
        )

    if not layers:
        raise ValueError(
            "At least one adapter "
            "layer is required."
        )

    if (
        len(set(layers))
        != len(layers)
    ):
        raise ValueError(
            "CUSTOM_ADAPTER_LAYERS "
            "contains duplicate indices."
        )

    invalid = [
        layer
        for layer in layers
        if (
            layer < 0
            or layer
            >= ADAPTER_LAYER_COUNT
        )
    ]

    if invalid:
        raise ValueError(
            "Invalid adapter layer "
            f"indices: {invalid}. "
            "Valid range is "
            f"0..{ADAPTER_LAYER_COUNT - 1}."
        )

    return sorted(
        layers
    )


ADAPTER_LAYERS = (
    resolve_adapter_layers()
)


def adapter_position_slug() -> str:
    if ADAPTER_POSITION != "custom":
        return ADAPTER_POSITION

    return (
        "custom_"
        + "-".join(
            str(layer)
            for layer
            in ADAPTER_LAYERS
        )
    )


ADAPTER_POSITION_SLUG = (
    adapter_position_slug()
)

EXPERIMENT_NAME = (
    f"{BASE_EXPERIMENT_NAME}_"
    f"{ADAPTER_MODULE}_"
    f"{ADAPTER_POSITION_SLUG}"
)

# Experiment names resolve to:
#
# joint_resadapt_encoder_all
# joint_resadapt_decoder_all
# joint_resadapt_encoder_decoder_all

# Data sources

# Authoritative Llanito folds.
# These provide:
#
#   fold_X/train.csv
#   fold_X/validation.csv
#   fixed_test.csv
#
# Validation and evaluation ALWAYS come
# from this ordinary Llanito dataset.

LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

# Miami larger training data.

MIAMI_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miami_seqstack_larger"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_PROCESSED_ROOT
    / "train_manifest.csv"
)

# Joint training

# The joint training set is constructed
# from each fold's Llanito train.csv
# plus Miami training data.
#
# Miami contributes TRAINING ONLY.
#
# Validation remains the untouched
# Llanito fold validation.csv.

JOINT_MIAMI_TO_LLANITO_RATIO: (
    float
    | None
) = 1.0

# 1.0 means:
#
# number of sampled Miami training rows
# =
# number of Llanito training rows
#
# This reproduces the balanced joint
# setup rather than letting the much
# larger Miami corpus dominate.
#
# Set to None only if you deliberately
# want the entire Miami training pool.

# Training manifests

# Keep the ordinary Llanito root here.
#
# train.py should load:
#
#   Llanito:
#     PROCESSED_ROOT/fold_X/train.csv
#
#   Miami:
#     MIAMI_TRAIN_MANIFEST
#
# and combine those ONLY for training.
#
# It should load validation directly from:
#
#   PROCESSED_ROOT/fold_X/validation.csv

PROCESSED_ROOT = (
    LLANITO_PROCESSED_ROOT
)

# Explicit fixed-test path for evaluate.py.

LLANITO_FIXED_TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
)

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

# Outputs

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

# Whisper

BASE_MODEL = (
    "openai/whisper-medium"
)

LANGUAGE = "es"
TASK = "transcribe"

# Folds

FOLD_NUMBERS = [
    1,
    2,
    3,
    4,
    5,
]

# None = all folds.

RUN_FOLDS: (
    list[int]
    | None
) = None

# Fold replacement

CLEAR_SELECTED_FOLD_BEFORE_TRAINING = (
    True
)

REBUILD_AGGREGATES_FROM_SAVED_FOLDS = (
    True
)

# Adapter architecture

ADAPTER_BOTTLENECK = 32
ADAPTER_DROPOUT = 0.05

# Backward compatibility

ALLOW_LEGACY_ENCODER_CHECKPOINTS = (
    True
)

ALLOW_LEGACY_STATE_KEYS = (
    True
)

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

USE_GRADIENT_CHECKPOINTING = (
    False
)

LOGGING_STEPS = 5

SAVE_TOTAL_LIMIT = 2

DATALOADER_WORKERS = 2

RANDOM_SEED = 2026

RESUME_FROM_LATEST_CHECKPOINT = (
    False
)

MAX_TRAIN_RECORDINGS: (
    int
    | None
) = None

MAX_VALIDATION_RECORDINGS: (
    int
    | None
) = None

# Evaluation

# Evaluation is Llanito-only.

GENERATION_MAX_LENGTH = 225

GENERATION_NUM_BEAMS = 1

EVALUATE_VANILLA_BASELINE = (
    True
)

MAX_TEST_RECORDINGS: (
    int
    | None
) = None