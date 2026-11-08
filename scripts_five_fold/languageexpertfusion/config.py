from __future__ import annotations

from pathlib import Path

# Project
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Base model
BASE_MODEL = "openai/whisper-medium"
BASELINE_MODEL_NAME = "vanilla_whisper_medium"
LANGUAGE = "es"
TASK = "transcribe"

# Data
SPANISH_DOM_ROOT = PROJECT_ROOT / "data" / "processed" / "miamispanishdom"
ENGLISH_DOM_ROOT = PROJECT_ROOT / "data" / "processed" / "englishdom"

SPANISH_TRAIN_MANIFEST = SPANISH_DOM_ROOT / "train.csv"
SPANISH_VALIDATION_MANIFEST = SPANISH_DOM_ROOT / "validation.csv"
SPANISH_TEST_MANIFEST = SPANISH_DOM_ROOT / "test.csv"

ENGLISH_TRAIN_MANIFEST = ENGLISH_DOM_ROOT / "train.csv"
ENGLISH_VALIDATION_MANIFEST = ENGLISH_DOM_ROOT / "validation.csv"
ENGLISH_TEST_MANIFEST = ENGLISH_DOM_ROOT / "test.csv"

LLANITO_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "llanito_cv"
FIXED_TEST_MANIFEST = LLANITO_PROCESSED_ROOT / "fixed_test.csv"

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

# Dataset thresholds
# Spanish-dominant: English fraction <= 0.30.
# English-dominant: Spanish fraction <= 0.30, equivalent to English fraction >= 0.70.
SPANISH_DOM_MAX_ENGLISH_FRACTION = 0.30
ENGLISH_DOM_MAX_SPANISH_FRACTION = 0.10
MIN_CLASSIFIED_WORDS = 5

# Speaker-disjoint Miami splits
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
RANDOM_SEED = 2026

# Architectures
RUN_ARCHITECTURES = [
    "cold",
    "mlp",
]

EXPERIMENT_NAMES = {
    "cold": "coldadapterfusion_langexperts",
    "mlp": "mlpinteractionfusion_langexperts",
}

# Whisper-medium
WHISPER_MEDIUM_ENCODER_LAYERS = 24
HIDDEN_SIZE = 1024

# Cold AdapterFusion
# Residual experts and fusion router after all 24 encoder blocks.
COLD_ADAPTER_LAYERS = list(range(WHISPER_MEDIUM_ENCODER_LAYERS))
COLD_ADAPTER_BOTTLENECK = 32
COLD_ADAPTER_DROPOUT = 0.05
COLD_FUSION_ATTENTION_DIM = 64
COLD_FUSION_DROPOUT = 0.05

# MLP interaction fusion
# Preserve the uploaded architecture: experts/fusion after blocks 3,6,...,24.
MLP_FUSION_EVERY_N_LAYERS = 3
MLP_ADAPTER_LAYERS = list(
    range(
        MLP_FUSION_EVERY_N_LAYERS - 1,
        WHISPER_MEDIUM_ENCODER_LAYERS,
        MLP_FUSION_EVERY_N_LAYERS,
    )
)
MLP_ADAPTER_BOTTLENECK = 32
MLP_ADAPTER_DROPOUT = 0.05

# Per position:
# [h, delta_spa, delta_eng, delta_spa * delta_eng, |delta_spa - delta_eng|]
# = 5 * 1024 = 5120.
MLP_FUSION_HIDDEN_1 = 32
MLP_FUSION_BOTTLENECK = 8
MLP_FUSION_HIDDEN_2 = 32
MLP_FUSION_DROPOUT = 0.05

# Cross-validation
FOLD_NUMBERS = [1, 2, 3, 4, 5]
RUN_FOLDS: list[int] | None = None

# Training
EXPERT_EPOCHS = 3
FUSION_EPOCHS = 5

EXPERT_LEARNING_RATE = 5e-5
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

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1

MAX_EXPERT_TRAIN_RECORDINGS: int | None = None
MAX_EXPERT_VALIDATION_RECORDINGS: int | None = None
MAX_LLANITO_TRAIN_RECORDINGS: int | None = None
MAX_LLANITO_VALIDATION_RECORDINGS: int | None = None
MAX_TEST_RECORDINGS: int | None = None

FORCE_RETRAIN_EXPERTS = False
FORCE_RETRAIN_FUSION = False
EVALUATE_BASELINE = True

# Miami larger data
MIAMI_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miami_seqstack_larger"
)

SPANISH_DOM_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miamispanishdom"
)

ENGLISH_DOM_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miamienglishdom"
)

SPANISH_DOM_MAX_ENGLISH_FRACTION = 0.20
ENGLISH_DOM_MAX_SPANISH_FRACTION = 0.50
MIN_CLASSIFIED_WORDS = 5


def experiment_name(architecture: str) -> str:
    if architecture not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    return EXPERIMENT_NAMES[architecture]


def model_root(architecture: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / "models" / experiment_name(architecture)


def output_root(architecture: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / experiment_name(architecture)


def summary_root(architecture: str) -> Path:
    return PROJECT_ROOT / "outputs_five_fold" / "summary" / experiment_name(architecture)


def adapter_layers(architecture: str) -> list[int]:
    if architecture == "cold":
        return list(COLD_ADAPTER_LAYERS)
    if architecture == "mlp":
        return list(MLP_ADAPTER_LAYERS)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def adapter_bottleneck(architecture: str) -> int:
    if architecture == "cold":
        return COLD_ADAPTER_BOTTLENECK
    if architecture == "mlp":
        return MLP_ADAPTER_BOTTLENECK
    raise ValueError(f"Unknown architecture: {architecture!r}")


def adapter_dropout(architecture: str) -> float:
    if architecture == "cold":
        return COLD_ADAPTER_DROPOUT
    if architecture == "mlp":
        return MLP_ADAPTER_DROPOUT
    raise ValueError(f"Unknown architecture: {architecture!r}")
