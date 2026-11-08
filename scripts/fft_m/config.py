from __future__ import annotations

from pathlib import Path


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Identity

MODEL_NAME = "fft_m"

MODEL_TYPE = "whisper_full"

BASE_MODEL = "openai/whisper-medium"


# Classification

FAMILY = "full_finetune"

TRANSFER_TYPE = "plain"

DOMAIN_CATEGORY = "m"


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


# Miami training data

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

TRAIN_MANIFEST = (
    PROCESSED_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_TRAINING_EXAMPLES = 4364


# Llanito fixed test

LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
)

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


# Selected Miami FFT recipe

LEARNING_RATE = 5e-06

WARMUP_STEPS = 0

WARMUP_RATIO = 0.05

EPOCHS = 5.0


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
    "method": "fixed_2x2_grid",
    "study_name": "miami_fft_2x2_batch16",
    "objective": (
        "lowest Miami validation WER"
    ),
    "best_recipe_id": 1,
    "best_validation_wer": (
        28.060941828254848
    ),
    "selection_training_recordings": 3878,
    "selection_validation_recordings": 486,
    "search_space": {
        "learning_rates": [
            5e-06,
            1e-05,
        ],
        "warmup_ratios": [
            0.05,
            0.1,
        ],
    },
    "selected_recipe": {
        "learning_rate": (
            LEARNING_RATE
        ),
        "warmup_steps": (
            WARMUP_STEPS
        ),
        "warmup_ratio": (
            WARMUP_RATIO
        ),
        "epochs": EPOCHS,
        "batch_size_per_device": (
            TRAIN_BATCH_SIZE_PER_DEVICE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "optimizer": (
            OPTIMIZER
        ),
        "scheduler": (
            LR_SCHEDULER_TYPE
        ),
        "max_grad_norm": (
            MAX_GRAD_NORM
        ),
        "fp16": USE_FP16,
        "bf16": USE_BF16,
        "gradient_checkpointing": (
            USE_GRADIENT_CHECKPOINTING
        ),
    },
    "final_training_uses_validation": True,
    "final_training_uses_cross_validation": False,
    "fixed_test_used": False,
}