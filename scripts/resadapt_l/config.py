from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Types

AdapterPlacement = Literal[
    "enc",
    "enc20_23",
    "dec",
    "encdec",
]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    placement: AdapterPlacement
    adapter_module: str
    adapter_layers: tuple[int, ...]
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

MODEL_TYPE = "residual_adapter"

BASE_MODEL = "openai/whisper-medium"


# Classification

FAMILY = "resadapt"

TRANSFER_TYPE = "plain"

DOMAIN_CATEGORY = "l"


# Base model

BASE_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
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


# Final selected training recipe

LEARNING_RATE = 1e-4

EPOCHS = 5.0

WARMUP_STEPS = 40

WARMUP_RATIO = 0.0

TRAIN_BATCH_SIZE_PER_DEVICE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.01

LR_SCHEDULER_TYPE = "linear"

MAX_GRAD_NORM = 1.0

OPTIMIZER = "adamw_torch"

USE_FP16 = True

USE_BF16 = False

USE_GRADIENT_CHECKPOINTING = False

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# Residual-adapter architecture

ADAPTER_BOTTLENECK = 32

ADAPTER_DROPOUT = 0.1

ADAPTER_POSITION = "all"

ALL_LAYERS = tuple(
    range(24)
)

UPPER_LAYERS = (
    20,
    21,
    22,
    23,
)


# Hyperparameter-selection provenance

HYPERPARAMETER_SELECTION = {
    "method": "previous_5_fold_cv",
    "source_model": "resadapt_enc_l_cv",
    "objective": (
        "Use the previously established "
        "Llanito residual-adapter recipe "
        "for final training on all non-test "
        "Llanito data."
    ),
    "selected_recipe": {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "batch_size_per_device": (
            TRAIN_BATCH_SIZE_PER_DEVICE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": (
            LR_SCHEDULER_TYPE
        ),
        "max_grad_norm": (
            MAX_GRAD_NORM
        ),
        "optimizer": OPTIMIZER,
        "fp16": USE_FP16,
        "bf16": USE_BF16,
        "gradient_checkpointing": (
            USE_GRADIENT_CHECKPOINTING
        ),
        "adapter_bottleneck": (
            ADAPTER_BOTTLENECK
        ),
        "adapter_dropout": (
            ADAPTER_DROPOUT
        ),
    },
    "final_training_uses_cross_validation": False,
    "fixed_test_used": False,
}


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "resadapt_enc_l": (
        ExperimentConfig(
            model_name="resadapt_enc_l",
            placement="enc",
            adapter_module="encoder",
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Llanito encoder "
                "residual-adapter model with "
                "adapters after encoder layers "
                "0-23, trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),

    "resadapt_enc20_23_l": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc20_23_l"
            ),
            placement="enc20_23",
            adapter_module="encoder",
            adapter_layers=UPPER_LAYERS,
            description=(
                "Final plain Llanito encoder "
                "residual-adapter model with "
                "adapters after encoder layers "
                "20-23 only, trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),

    "resadapt_dec_l": (
        ExperimentConfig(
            model_name="resadapt_dec_l",
            placement="dec",
            adapter_module="decoder",
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Llanito decoder "
                "residual-adapter model with "
                "adapters after decoder layers "
                "0-23, trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),

    "resadapt_encdec_l": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_l"
            ),
            placement="encdec",
            adapter_module=(
                "encoder_decoder"
            ),
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Llanito encoder"
                "+decoder residual-adapter "
                "model with adapters after "
                "all encoder and decoder "
                "layers, trained on all 904 "
                "non-test Llanito recordings."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)

   
