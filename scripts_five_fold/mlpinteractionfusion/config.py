from __future__ import annotations

from pathlib import Path

# Project
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Experiments
# Run only the Miami-seqstack-larger backbone by default.
RUN_EXPERIMENTS = [
    "miami_base",
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
        "experiment_name": "mlpinteractionfusion_vanilla",
        "base_model": VANILLA_BASE_MODEL,
        "baseline_model_name": "vanilla_whisper_medium",
    },
    "miami_base": {
        "experiment_name": "mlpinteractionfusion_miami_base",
        "base_model": MIAMI_BASE_MODEL,
        "baseline_model_name": "miami_spanish_larger_full_finetune",
    },
}


def resolve_experiment(experiment_key: str) -> dict[str, str]:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment key: {experiment_key!r}. "
            f"Valid values: {sorted(EXPERIMENTS)}"
        )
    return dict(EXPERIMENTS[experiment_key])


# Default experiment
_DEFAULT_EXPERIMENT = resolve_experiment(RUN_EXPERIMENTS[0])
EXPERIMENT_NAME = _DEFAULT_EXPERIMENT["experiment_name"]
BASE_MODEL = _DEFAULT_EXPERIMENT["base_model"]
BASELINE_MODEL_NAME = _DEFAULT_EXPERIMENT["baseline_model_name"]

# Data
LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "llanito_cv"
)

MIAMI_PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "miami_seqstack_larger"
)

MIAMI_TRAIN_MANIFEST = MIAMI_PROCESSED_ROOT / "train_manifest.csv"
MIAMI_VALIDATION_MANIFEST = MIAMI_PROCESSED_ROOT / "validation_manifest.csv"
FIXED_TEST_MANIFEST = LLANITO_PROCESSED_ROOT / "fixed_test.csv"

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)


def model_root_for_experiment(experiment_name: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / "models" / experiment_name


def output_root_for_experiment(experiment_name: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / experiment_name


def summary_root_for_experiment(experiment_name: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / "summary" / experiment_name


MODEL_ROOT = model_root_for_experiment(EXPERIMENT_NAME)
OUTPUT_ROOT = output_root_for_experiment(EXPERIMENT_NAME)
SUMMARY_ROOT = summary_root_for_experiment(EXPERIMENT_NAME)

# Whisper
LANGUAGE = "es"
TASK = "transcribe"
WHISPER_MEDIUM_ENCODER_LAYERS = 24

# Expert/fusion placement
# Python indices 2,5,...,23 correspond to after Whisper encoder blocks
# 3,6,...,24. The expert adapters and fusion MLP exist only there.
FUSION_EVERY_N_LAYERS = 3
FUSION_LAYERS = list(
    range(
        FUSION_EVERY_N_LAYERS - 1,
        WHISPER_MEDIUM_ENCODER_LAYERS,
        FUSION_EVERY_N_LAYERS,
    )
)
ADAPTER_LAYERS = list(FUSION_LAYERS)
ADAPTER_POSITION = "every3"

# Expert adapters
# At each selected layer each frozen expert is 1024 -> 32 -> 1024.
ADAPTER_BOTTLENECK = 32
ADAPTER_DROPOUT = 0.05

# Fusion MLP
# Input per encoder position:
# [h, delta_M, delta_L, delta_M * delta_L, |delta_M - delta_L|]
# = 5 * 1024 = 5120 for Whisper-medium.
# Learned fusion path: 5120 -> 32 -> 8 -> 32 -> 1024.
FUSION_HIDDEN_1 = 32
FUSION_BOTTLENECK = 8
FUSION_HIDDEN_2 = 32
FUSION_DROPOUT = 0.05

# The final MLP projection is zero-initialised, so the fusion correction starts
# exactly at zero and the model initially behaves exactly like its backbone.

# Folds
FOLD_NUMBERS = [1, 2, 3, 4, 5]
RUN_FOLDS: list[int] | None = None

# Stage control
TRAIN_MIAMI_ADAPTER_IF_MISSING = True
FORCE_RETRAIN_MIAMI_ADAPTER = False

TRAIN_LLANITO_ADAPTER_IF_MISSING = True
FORCE_RETRAIN_LLANITO_ADAPTER = False

TRAIN_FUSION_IF_MISSING = True
FORCE_RETRAIN_FUSION = False

# Training
MIAMI_EPOCHS = 3
LLANITO_ADAPTER_EPOCHS = 5
FUSION_EPOCHS = 5

MIAMI_LEARNING_RATE = 5e-5
LLANITO_ADAPTER_LEARNING_RATE = 5e-5
FUSION_LEARNING_RATE = 5e-5

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
EVALUATE_BASELINE = True
MAX_TEST_RECORDINGS: int | None = None

# Aggregation
REBUILD_AGGREGATES_FROM_SAVED_FOLDS = True
