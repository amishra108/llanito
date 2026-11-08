from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Types

LoraPlacement = Literal[
    "encdec",
    "enc",
    "encattn",
    "dec",
]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    placement: LoraPlacement
    description: str

    @property
    def model_root(self) -> Path:
        return (
            PROJECT_ROOT
            / "models"
            / self.model_name
        )

    @property
    def model_directory(self) -> Path:
        return (
            self.model_root
            / "model"
        )

    @property
    def checkpoint_directory(self) -> Path:
        return (
            self.model_root
            / "checkpoints"
        )

    @property
    def metadata_path(self) -> Path:
        return (
            self.model_root
            / "metadata.json"
        )

    @property
    def output_root(self) -> Path:
        return (
            PROJECT_ROOT
            / "outputs"
            / self.model_name
        )


# Identity

MODEL_TYPE = "peft_lora"

BASE_MODEL = "openai/whisper-medium"


# Classification

FAMILY = "lora"

TRANSFER_TYPE = "plain"

DOMAIN_CATEGORY = "m"


# Base model

BASE_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
)


# Miami training data

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

ORIGINAL_TRAIN_MANIFEST = (
    PROCESSED_ROOT
    / "train_manifest.csv"
)

ORIGINAL_VALIDATION_MANIFEST = (
    PROCESSED_ROOT
    / "validation_manifest.csv"
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


# Evaluation

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


# Selected Miami LoRA recipe

LEARNING_RATE = 1e-4

EPOCHS = 5.0

WARMUP_STEPS = 40

WARMUP_RATIO = 0.0


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


# LoRA architecture

LORA_RANK = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

LORA_BIAS = "none"

USE_DORA = False

USE_RSLORA = False


# Hyperparameter-selection provenance

HYPERPARAMETER_SELECTION = {
    "method": (
        "miami_full_lora_validation_search"
    ),
    "study_name": (
        "miami_full_lora_validation_v3"
    ),
    "objective": (
        "lowest Miami validation WER"
    ),
    "historical_trial_000_retrained": False,
    "historical_trial_000_recipe": {
        "learning_rate": 5e-05,
        "warmup_steps": 40,
        "epochs": 5,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "seed": 2026,
        "retrained": False,
    },
    "best_new_trial": 4,
    "optuna_internal_best_trial": 3,
    "best_new_validation_wer": (
        28.698060941828256
    ),
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
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
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
        "lora_rank": (
            LORA_RANK
        ),
        "lora_alpha": (
            LORA_ALPHA
        ),
        "lora_dropout": (
            LORA_DROPOUT
        ),
        "seed": SEED,
        "data_seed": SEED,
    },
    "final_training_uses_validation": True,
    "final_training_uses_cross_validation": False,
    "fixed_test_used": False,
}


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "lora_enc_m": (
        ExperimentConfig(
            model_name=(
                "lora_enc_m"
            ),
            placement="enc",
            description=(
                "Final plain Miami encoder "
                "LoRA trained on the union "
                "of the original Miami train "
                "and validation recordings."
            ),
        )
    ),

    "lora_encattn_m": (
        ExperimentConfig(
            model_name=(
                "lora_encattn_m"
            ),
            placement="encattn",
            description=(
                "Final plain Miami encoder "
                "attention-only LoRA trained "
                "on the union of the original "
                "Miami train and validation "
                "recordings."
            ),
        )
    ),

    "lora_dec_m": (
        ExperimentConfig(
            model_name=(
                "lora_dec_m"
            ),
            placement="dec",
            description=(
                "Final plain Miami decoder "
                "LoRA trained on the union "
                "of the original Miami train "
                "and validation recordings."
            ),
        )
    ),

    "lora_encdec_m": (
        ExperimentConfig(
            model_name=(
                "lora_encdec_m"
            ),
            placement="encdec",
            description=(
                "Final plain Miami encoder"
                "+decoder LoRA trained on "
                "the union of the original "
                "Miami train and validation "
                "recordings."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)