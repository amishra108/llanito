from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Types

PromptArchitecture = Literal[
    "decoder_input",
    "decoder_deep",
    "decoder_deep_bneck",
]


# Experiment

@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    prompt_length: int
    architecture: PromptArchitecture
    bottleneck_size: int | None
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

DOMAIN_CATEGORY = "m"


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


# Miami data

MIAMI_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

TRAIN_MANIFEST = (
    MIAMI_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_TRAINING_EXAMPLES = 4364


# Llanito fixed test

LLANITO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

TEST_MANIFEST = (
    LLANITO_ROOT
    / "fixed_test.csv"
)

EXPECTED_TEST_RECORDINGS = 57

DATASET_NAME = "llanito_fixed_test"


# Whisper

LANGUAGE = "es"

TASK = "transcribe"

GENERATION_MAX_LENGTH = 225

GENERATION_NUM_BEAMS = 1

HIDDEN_SIZE = 1024

DECODER_LAYERS = 24


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
#
# Deliberately kept the same across
# regular/deep/deep-bneck so this is
# initially an architecture comparison.

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

# Keep the same precision policy as
# the working Llanito SPT pipeline.

USE_FP16 = False

USE_BF16 = False

USE_GRADIENT_CHECKPOINTING = False

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# Prompt architecture

PROMPT_INIT = (
    "decoder_token_embeddings"
)

PROMPT_INIT_STD = 0.02

DEEP_BOTTLENECK_SIZE = 32


# Expected parameters

REGULAR_PROMPT_PARAMETERS = (
    20
    * HIDDEN_SIZE
)

DEEP_PROMPT_PARAMETERS = (
    DECODER_LAYERS
    * 20
    * HIDDEN_SIZE
)

BOTTLENECK_PARAMETERS_PER_LAYER = (
    # down.weight
    HIDDEN_SIZE
    * DEEP_BOTTLENECK_SIZE

    # down.bias
    + DEEP_BOTTLENECK_SIZE

    # up.weight
    + DEEP_BOTTLENECK_SIZE
    * HIDDEN_SIZE

    # up.bias
    + HIDDEN_SIZE
)

DEEP_BNECK_PARAMETERS = (
    DEEP_PROMPT_PARAMETERS
    + (
        DECODER_LAYERS
        * BOTTLENECK_PARAMETERS_PER_LAYER
    )
)


# Hyperparameter provenance

HYPERPARAMETER_SELECTION = {
    "method": (
        "reuse_selected_spt_recipe"
    ),
    "source_recipe": (
        "Llanito decoder-input SPT"
    ),
    "rationale": (
        "Use the same SPT optimisation "
        "recipe for regular, deep and "
        "deep-bottleneck Miami models so "
        "the first comparison isolates "
        "prompt architecture."
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
    "validation_during_final_training": (
        False
    ),
    "fixed_test_used": False,
}


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "spt_dec_m_p20": (
        ExperimentConfig(
            model_name=(
                "spt_dec_m_p20"
            ),
            prompt_length=20,
            architecture=(
                "decoder_input"
            ),
            bottleneck_size=None,
            description=(
                "Miami decoder-input SPT "
                "with one learned 20-token "
                "soft prompt prepended at "
                "the decoder input. Whisper "
                "is frozen."
            ),
        )
    ),

    "spt_dec_m_p20_deep": (
        ExperimentConfig(
            model_name=(
                "spt_dec_m_p20_deep"
            ),
            prompt_length=20,
            architecture=(
                "decoder_deep"
            ),
            bottleneck_size=None,
            description=(
                "Miami deep decoder SPT. "
                "Each of the 24 frozen "
                "Whisper decoder layers "
                "receives its own independent "
                "20-token learned prompt. "
                "Prompt positions are removed "
                "after each layer before the "
                "next layer receives its own "
                "prompt."
            ),
        )
    ),

    "spt_dec_m_p20_deep_bneck": (
        ExperimentConfig(
            model_name=(
                "spt_dec_m_p20_deep_bneck"
            ),
            prompt_length=20,
            architecture=(
                "decoder_deep_bneck"
            ),
            bottleneck_size=(
                DEEP_BOTTLENECK_SIZE
            ),
            description=(
                "Miami deep decoder SPT with "
                "one independent 20-token "
                "prompt per decoder layer. "
                "Before injection, each "
                "layer prompt passes through "
                "a trainable residual "
                "1024->32->1024 bottleneck. "
                "Whisper itself remains "
                "frozen."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)