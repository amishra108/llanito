from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Experiment

@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    prompt_length: int
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

MODEL_TYPE = "spt"

BASE_MODEL = "openai/whisper-medium"


# Classification

FAMILY = "spt"

TRANSFER_TYPE = "plain"

DOMAIN_CATEGORY = "l"


# Base model

BASE_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
)

BASE_MODEL_DIRECTORY = (
    BASE_MODEL_ROOT
    / "model"
)


# Llanito data

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


# Selected SPT recipe

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.01

EPOCHS = 5.0

WARMUP_STEPS = 40

WARMUP_RATIO = 0.0


# Fixed training settings

TRAIN_BATCH_SIZE_PER_DEVICE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LR_SCHEDULER_TYPE = "linear"

MAX_GRAD_NORM = 1.0

OPTIMIZER = "adamw_torch"

USE_FP16 = False

USE_BF16 = False

USE_GRADIENT_CHECKPOINTING = False

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# SPT architecture

PROMPT_ARCHITECTURE = "decoder_input"

PROMPT_INIT = "decoder_token_embeddings"

PROMPT_INIT_STD = 0.02


# Hyperparameter provenance

HYPERPARAMETER_SELECTION = {
    "method": (
        "previous_llanito_spt_search"
    ),
    "source_model": (
        "spt_dec_l_p20_cv"
    ),
    "objective": (
        "Select a stable decoder-input "
        "soft-prompt training recipe before "
        "final training on all non-test "
        "Llanito data."
    ),
    "selected_recipe": {
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "epochs": (
            EPOCHS
        ),
        "warmup_steps": (
            WARMUP_STEPS
        ),
        "warmup_ratio": (
            WARMUP_RATIO
        ),
        "batch_size_per_device": (
            TRAIN_BATCH_SIZE_PER_DEVICE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "optimizer": (
            OPTIMIZER
        ),
        "lr_scheduler_type": (
            LR_SCHEDULER_TYPE
        ),
        "max_grad_norm": (
            MAX_GRAD_NORM
        ),
        "fp16": (
            USE_FP16
        ),
        "bf16": (
            USE_BF16
        ),
        "gradient_checkpointing": (
            USE_GRADIENT_CHECKPOINTING
        ),
        "seed": (
            SEED
        ),
        "data_seed": (
            SEED
        ),
    },
    "validation_during_final_training": False,
    "final_training_uses_cross_validation": False,
    "fixed_test_used": False,
}


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "spt_dec_l_p5": (
        ExperimentConfig(
            model_name=(
                "spt_dec_l_p5"
            ),
            prompt_length=5,
            description=(
                "Final Llanito decoder-input "
                "soft-prompt tuning model with "
                "5 learned prompt embeddings. "
                "Whisper-medium is frozen and "
                "only the decoder soft prompts "
                "are trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),

    "spt_dec_l_p20": (
        ExperimentConfig(
            model_name=(
                "spt_dec_l_p20"
            ),
            prompt_length=20,
            description=(
                "Final Llanito decoder-input "
                "soft-prompt tuning model with "
                "20 learned prompt embeddings. "
                "Whisper-medium is frozen and "
                "only the decoder soft prompts "
                "are trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),

    "spt_dec_l_p40": (
        ExperimentConfig(
            model_name=(
                "spt_dec_l_p40"
            ),
            prompt_length=40,
            description=(
                "Final Llanito decoder-input "
                "soft-prompt tuning model with "
                "40 learned prompt embeddings. "
                "Whisper-medium is frozen and "
                "only the decoder soft prompts "
                "are trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)