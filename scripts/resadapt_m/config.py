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


# Final selected training recipe

LEARNING_RATE = 0.0001

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

ADAPTER_BOTTLENECK = 64

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
    "method": (
        "previous_miami_validation_search"
    ),
    "study_name": (
        "miami_resadapt_bottleneck_validation_v1"
    ),
    "best_recipe_id": 1,
    "historical_best_validation_wer": (
        34.18282548476454
    ),
    "fixed_test_used": False,
    "final_training_uses_validation": True,
    "final_training_uses_cross_validation": False,
    "note": (
        "Final model is trained on the union "
        "of the original Miami train and "
        "validation manifests. Final evaluation "
        "uses the independent Llanito fixed "
        "test set."
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
}


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "resadapt_enc_m": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_m"
            ),
            placement="enc",
            adapter_module="encoder",
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Miami encoder "
                "residual-adapter model with "
                "adapters after encoder layers "
                "0-23, trained on the union of "
                "the original Miami train and "
                "validation sets."
            ),
        )
    ),

    "resadapt_enc20_23_m": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc20_23_m"
            ),
            placement="enc20_23",
            adapter_module="encoder",
            adapter_layers=UPPER_LAYERS,
            description=(
                "Final plain Miami encoder "
                "residual-adapter model with "
                "adapters after encoder layers "
                "20-23 only, trained on the "
                "union of the original Miami "
                "train and validation sets."
            ),
        )
    ),

    "resadapt_dec_m": (
        ExperimentConfig(
            model_name=(
                "resadapt_dec_m"
            ),
            placement="dec",
            adapter_module="decoder",
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Miami decoder "
                "residual-adapter model with "
                "adapters after decoder layers "
                "0-23, trained on the union of "
                "the original Miami train and "
                "validation sets."
            ),
        )
    ),

    "resadapt_encdec_m": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_m"
            ),
            placement="encdec",
            adapter_module=(
                "encoder_decoder"
            ),
            adapter_layers=ALL_LAYERS,
            description=(
                "Final plain Miami encoder"
                "+decoder residual-adapter "
                "model with adapters after "
                "all encoder and decoder "
                "layers, trained on the union "
                "of the original Miami train "
                "and validation sets."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)