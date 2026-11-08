
from __future__ import annotations

from pathlib import Path

# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Experiments

# The whole Cold AdapterFusion pipeline can use:
#
# 1) coldadapterfusion_vanilla
#    Every expert and the fusion backbone starts from vanilla Whisper-medium.
#
# 2) coldadapterfusion_miami_base
#    Every expert and the fusion backbone starts from the already
#    Miami-full-fine-tuned Whisper model.
#
# TEMPORARY SETTING:
# Retrain only vanilla-base fold 2.

RUN_EXPERIMENTS = [
    "vanilla",
]

VANILLA_BASE_MODEL = "openai/whisper-medium"

MIAMI_BASE_MODEL = str(
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "miami_seqstack_larger"
    / "spanish_full_finetune"
)

EXPERIMENTS = {
    "vanilla": {
        "experiment_name": "coldadapterfusion_vanilla",
        "base_model": VANILLA_BASE_MODEL,
        "baseline_model_name": "vanilla_whisper_medium",
    },
    "miami_base": {
        "experiment_name": "coldadapterfusion_miami_base",
        "base_model": MIAMI_BASE_MODEL,
        "baseline_model_name": "miami_spanish_full_finetune",
    },
}


def resolve_experiment(
    experiment_key: str,
) -> dict[str, str]:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment key: {experiment_key!r}. "
            f"Valid values: {sorted(EXPERIMENTS)}"
        )

    return dict(
        EXPERIMENTS[
            experiment_key
        ]
    )


# Default experiment

_DEFAULT_EXPERIMENT = resolve_experiment(
    RUN_EXPERIMENTS[0]
)

EXPERIMENT_NAME = (
    _DEFAULT_EXPERIMENT[
        "experiment_name"
    ]
)

BASE_MODEL = (
    _DEFAULT_EXPERIMENT[
        "base_model"
    ]
)

BASELINE_MODEL_NAME = (
    _DEFAULT_EXPERIMENT[
        "baseline_model_name"
    ]
)

# Data

LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

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

MIAMI_VALIDATION_MANIFEST = (
    MIAMI_PROCESSED_ROOT
    / "validation_manifest.csv"
)

FIXED_TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
)

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)


def model_root_for_experiment(
    experiment_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "outputs_five_fold" / "models"
        / experiment_name
    )


def output_root_for_experiment(
    experiment_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "outputs_five_fold"
        / experiment_name
    )


def summary_root_for_experiment(
    experiment_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "outputs_five_fold"
        / "summary"
        / experiment_name
    )


MODEL_ROOT = model_root_for_experiment(
    EXPERIMENT_NAME
)

OUTPUT_ROOT = output_root_for_experiment(
    EXPERIMENT_NAME
)

SUMMARY_ROOT = summary_root_for_experiment(
    EXPERIMENT_NAME
)

# Whisper

LANGUAGE = "es"
TASK = "transcribe"

WHISPER_MEDIUM_ENCODER_LAYERS = 24

# Adapter placement

ADAPTER_POSITION = "all"

CUSTOM_ADAPTER_LAYERS = [20]


def resolve_adapter_layers() -> list[int]:
    if ADAPTER_POSITION == "all":
        layers = list(
            range(
                WHISPER_MEDIUM_ENCODER_LAYERS
            )
        )

    elif ADAPTER_POSITION == "first6":
        layers = list(
            range(6)
        )

    elif ADAPTER_POSITION == "last6":
        layers = list(
            range(
                WHISPER_MEDIUM_ENCODER_LAYERS - 6,
                WHISPER_MEDIUM_ENCODER_LAYERS,
            )
        )

    elif ADAPTER_POSITION == "first12":
        layers = list(
            range(12)
        )

    elif ADAPTER_POSITION == "last12":
        layers = list(
            range(
                WHISPER_MEDIUM_ENCODER_LAYERS - 12,
                WHISPER_MEDIUM_ENCODER_LAYERS,
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
            "Duplicate adapter layer indices."
        )

    invalid = [
        layer
        for layer in layers
        if (
            layer < 0
            or layer >= WHISPER_MEDIUM_ENCODER_LAYERS
        )
    ]

    if invalid:
        raise ValueError(
            f"Invalid adapter layer indices: {invalid}"
        )

    return sorted(
        layers
    )


ADAPTER_LAYERS = resolve_adapter_layers()

# Expert adapter architecture

ADAPTER_BOTTLENECK = 32
ADAPTER_DROPOUT = 0.05

# Fusion architecture

FUSION_ATTENTION_DIM = 64
FUSION_DROPOUT = 0.05

# Folds

FOLD_NUMBERS = [
    1,
    2,
    3,
    4,
    5,
]

# Retrain ONLY fold 2.

RUN_FOLDS: list[int] | None = [
    2,
]

# Stage control

# Keep the already-trained global Miami adapter.
TRAIN_MIAMI_ADAPTER_IF_MISSING = True
FORCE_RETRAIN_MIAMI_ADAPTER = False

# Retrain the fold-2 Llanito expert.
TRAIN_LLANITO_ADAPTER_IF_MISSING = True
FORCE_RETRAIN_LLANITO_ADAPTER = True

# Retrain fold-2 fusion using the existing Miami expert
# and newly retrained fold-2 Llanito expert.
TRAIN_FUSION_IF_MISSING = True
FORCE_RETRAIN_FUSION = True

# Training

MIAMI_EPOCHS = 3
LLANITO_ADAPTER_EPOCHS = 5
FUSION_EPOCHS = 5

MIAMI_LEARNING_RATE = 5e-5
LLANITO_ADAPTER_LEARNING_RATE = 5e-5
FUSION_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01
WARMUP_STEPS = 40

TRAIN_BATCH_SIZE_PER_GPU = 1
VALIDATION_BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 8

USE_FP16 = False
USE_BF16 = False
USE_GRADIENT_CHECKPOINTING = False

LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 2
DATALOADER_WORKERS = 2
RANDOM_SEED = 2026

# Optional debugging caps

MAX_MIAMI_TRAIN_RECORDINGS: int | None = None
MAX_MIAMI_VALIDATION_RECORDINGS: int | None = None

MAX_LLANITO_TRAIN_RECORDINGS: int | None = None
MAX_LLANITO_VALIDATION_RECORDINGS: int | None = None

# Generation / evaluation

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1

EVALUATE_VANILLA_BASELINE = True

MAX_TEST_RECORDINGS: int | None = None

# Aggregation

REBUILD_AGGREGATES_FROM_SAVED_FOLDS = True