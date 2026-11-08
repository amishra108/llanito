from __future__ import annotations

from pathlib import Path


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Identity

MODEL_NAME = "fft_l"

MODEL_TYPE = "whisper_full"

BASE_MODEL = "openai/whisper-medium"


# Classification

FAMILY = "full_finetune"

TRANSFER_TYPE = "plain"

DOMAIN_CATEGORY = "l"


# Base model

BASE_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
)


# Final model

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / MODEL_NAME
)

MODEL_DIRECTORY = (
    MODEL_ROOT
    / "model"
)

METADATA_PATH = (
    MODEL_ROOT
    / "metadata.json"
)

CHECKPOINT_DIRECTORY = (
    MODEL_ROOT
    / "checkpoints"
)


# Data

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

TRAIN_MANIFEST = (
    PROCESSED_ROOT
    / "train_validation_union_manifest.csv"
)

TEST_MANIFEST = (
    PROCESSED_ROOT
    / "fixed_test.csv"
)

EXPECTED_TRAINING_EXAMPLES = 904

EXPECTED_TEST_RECORDINGS = 57


# Evaluation output

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / MODEL_NAME
)

DATASET_NAME = "llanito_fixed_test"


# Whisper

LANGUAGE = "es"

TASK = "transcribe"

GENERATION_MAX_LENGTH = 225

GENERATION_NUM_BEAMS = 1


# Llanito normalisation

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

NORMALISATION_MAPPING_COLUMN = "task"

NORMALISATION_SECTION_MARKERS = {
    "passage": (
        "SECTION 1 - "
        "PASSAGE RECORDINGS ONLY"
    ),
    "sentence": (
        "SECTION 2 - "
        "SENTENCE-TASK RECORDINGS ONLY"
    ),
}


# Optuna-selected recipe

LEARNING_RATE = (
    5.693456550958914e-06
)

WARMUP_RATIO = 0.05

WARMUP_STEPS = 0

EPOCHS = 4


# Fixed training settings

TRAIN_BATCH_SIZE_PER_DEVICE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.01

LR_SCHEDULER_TYPE = "linear"

MAX_GRAD_NORM = 1.0

OPTIMIZER = "adamw_torch"

USE_FP16 = True

USE_BF16 = False

USE_GRADIENT_CHECKPOINTING = True

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# Hyperparameter-selection provenance

HYPERPARAMETER_SELECTION = {
    "method": "optuna_5_fold_cv",
    "study_name": "llanito_fft_cv",
    "best_trial": 2,
    "objective": (
        "mean best normalised validation "
        "WER across 5 CV folds"
    ),
    "best_cv_normalised_wer": (
        27.793280364080033
    ),
    "fold_scores": [
        16.73469387755102,
        24.796747967479675,
        40.17857142857143,
        31.640625,
        25.615763546798032,
    ],
    "fixed_test_used": False,
}