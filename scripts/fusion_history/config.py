from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Types

Domain = Literal[
    "l",
    "m",
    "e",
]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str

    branch_domains: tuple[
        Domain,
        Domain,
    ]

    target_domain: Domain

    history_length: int

    expected_training_examples: (
        int | None
    )

    domain_category: str

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
    def checkpoint_directory(
        self,
    ) -> Path:
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

    @property
    def training_language(self) -> str:
        if self.target_domain == "e":
            return "en"

        return "es"

    @property
    def history_fusion_layers(
        self,
    ) -> tuple[int, ...]:
        if self.history_length <= 0:
            raise ValueError(
                "history_length must be "
                "greater than zero."
            )

        if (
            NUM_ENCODER_LAYERS
            % self.history_length
            != 0
        ):
            raise ValueError(
                "history_length must "
                "divide the 24-layer "
                "encoder exactly."
            )

        return tuple(
            range(
                self.history_length - 1,
                NUM_ENCODER_LAYERS,
                self.history_length,
            )
        )

    @property
    def number_of_fusion_layers(
        self,
    ) -> int:
        return len(
            self.history_fusion_layers
        )

    @property
    def expected_trainable_parameters(
        self,
    ) -> int:
        return (
            self.number_of_fusion_layers
            * PARAMETERS_PER_HISTORY_MODULE
        )


# Identity

MODEL_TYPE = (
    "resadapt_history_fusion"
)

FAMILY = (
    "resadapt_fusion_history"
)

TRANSFER_TYPE = "cold"

BASE_MODEL = "openai/whisper-medium"


# Whisper

HIDDEN_SIZE = 1024

NUM_ENCODER_LAYERS = 24

ENCODER_LAYERS = tuple(
    range(NUM_ENCODER_LAYERS)
)


# History architecture

HISTORY_WIDTH = 128

INTERACTION_FEATURES = 4

INTERACTION_SIZE = (
    INTERACTION_FEATURES
    * HIDDEN_SIZE
)


# Trainable parameter count
#
# At every HIST fusion point:
#
# interaction:
#
# [
#     delta_a,
#     delta_b,
#     delta_a * delta_b,
#     delta_a - delta_b,
# ]
#
# gives:
#
# 4 * 1024 = 4096
#
#
# Compressor:
#
# Linear:
# 4096 -> 128
#
#
# GRU:
#
# input_size = 128
# hidden_size = 128
#
#
# Output:
#
# Linear:
# 128 -> 1024


COMPRESSOR_PARAMETERS = (
    INTERACTION_SIZE
    * HISTORY_WIDTH
    + HISTORY_WIDTH
)


# PyTorch GRU parameterisation:
#
# weight_ih:
# 3 * hidden * input
#
# weight_hh:
# 3 * hidden * hidden
#
# bias_ih:
# 3 * hidden
#
# bias_hh:
# 3 * hidden

GRU_PARAMETERS = (
    3
    * HISTORY_WIDTH
    * HISTORY_WIDTH

    + 3
    * HISTORY_WIDTH
    * HISTORY_WIDTH

    + 2
    * 3
    * HISTORY_WIDTH
)


OUTPUT_PARAMETERS = (
    HISTORY_WIDTH
    * HIDDEN_SIZE
    + HIDDEN_SIZE
)


PARAMETERS_PER_HISTORY_MODULE = (
    COMPRESSOR_PARAMETERS
    + GRU_PARAMETERS
    + OUTPUT_PARAMETERS
)


# Expected arithmetic:
#
# compressor:
#
# 4096 * 128 + 128
# = 524,416
#
# GRU:
#
# 3*128*128
# + 3*128*128
# + 2*3*128
# = 99,072
#
# output:
#
# 128*1024 + 1024
# = 132,096
#
# total:
#
# 524,416
# + 99,072
# + 132,096
# = 755,584

if (
    PARAMETERS_PER_HISTORY_MODULE
    != 755_584
):
    raise RuntimeError(
        "Unexpected HIST module "
        "parameter count.\n"
        f"Found: "
        f"{PARAMETERS_PER_HISTORY_MODULE:,}"
    )


# Vanilla Whisper

VANILLA_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
)

VANILLA_MODEL_DIRECTORY = (
    VANILLA_MODEL_ROOT
    / "model"
)


# Frozen ResAdapt experts

BRANCH_MODEL_ROOTS = {
    "l": (
        PROJECT_ROOT
        / "models"
        / "resadapt_enc_l"
    ),
    "m": (
        PROJECT_ROOT
        / "models"
        / "resadapt_enc_m"
    ),
    "e": (
        PROJECT_ROOT
        / "models"
        / "resadapt_enc_e"
    ),
}


def get_branch_model_root(
    domain: str,
) -> Path:
    if domain not in BRANCH_MODEL_ROOTS:
        raise ValueError(
            "Unknown expert domain: "
            f"{domain!r}"
        )

    return BRANCH_MODEL_ROOTS[
        domain
    ]


def get_branch_model_directory(
    domain: str,
) -> Path:
    return (
        get_branch_model_root(
            domain
        )
        / "model"
    )


# Miami training data

MIAMI_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_MIAMI_TRAINING_EXAMPLES = 4364


# English training data

ENGLISH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miamienglishdom"
)

ENGLISH_TRAIN_MANIFEST = (
    ENGLISH_ROOT
    / (
        "train_validation_union_"
        "manifest_duration_matched.csv"
    )
)


# Llanito training data

LLANITO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

LLANITO_TRAIN_MANIFEST = (
    LLANITO_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_LLANITO_TRAINING_EXAMPLES = 904


# Llanito fixed test

TEST_MANIFEST = (
    LLANITO_ROOT
    / "fixed_test.csv"
)

EXPECTED_TEST_RECORDINGS = 57

DATASET_NAME = "llanito_fixed_test"


def get_train_manifest(
    experiment: ExperimentConfig,
) -> Path:
    if experiment.target_domain == "m":
        return MIAMI_TRAIN_MANIFEST

    if experiment.target_domain == "e":
        return ENGLISH_TRAIN_MANIFEST

    if experiment.target_domain == "l":
        return LLANITO_TRAIN_MANIFEST

    raise RuntimeError(
        "Unsupported HIST target "
        f"domain: "
        f"{experiment.target_domain!r}"
    )


# Whisper generation

EVALUATION_LANGUAGE = "es"

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


# Optimisation
#
# Keep the fusion-family training
# protocol fixed across DELpara,
# DESC and HIST.
#
# Warmup is deliberately zero.

LEARNING_RATE = 5e-05

EPOCHS = 5.0

WARMUP_STEPS = 0

WARMUP_RATIO = 0.0

TRAIN_BATCH_SIZE_PER_DEVICE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.0

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


# Hyperparameter provenance

HYPERPARAMETER_SELECTION = {
    "method": (
        "reuse_fusion_training_protocol"
    ),
    "source": (
        "scripts/fusion"
    ),
    "reason": (
        "Hold the surrounding "
        "optimisation protocol fixed "
        "while varying the HIST "
        "history length."
    ),
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "warmup_steps": WARMUP_STEPS,
    "warmup_ratio": WARMUP_RATIO,
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
    "frozen_expert_dropout": 0.0,
    "seed": SEED,
    "data_seed": SEED,
    "fixed_test_used": False,
}


# Experiments
#
# Source experts:
#
# l = Llanito
# e = English
#
# Target:
#
# m = Miami
#
#
# HIST:
# history length 1
# fusion at layers 1..24
#
# HIST2:
# history length 2
# fusion at 2,4,...,24
#
# HIST3:
# history length 3
# fusion at 3,6,...,24
#
# HIST4:
# history length 4
# fusion at 4,8,...,24


EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    "resadapt_enc_leHISTtm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_leHISTtm_cold"
            ),
            branch_domains=(
                "l",
                "e",
            ),
            target_domain="m",
            history_length=1,
            expected_training_examples=(
                EXPECTED_MIAMI_TRAINING_EXAMPLES
            ),
            domain_category=(
                "leHISTtm"
            ),
            description=(
                "Persistent frozen Llanito "
                "and English encoder "
                "residual-adapter experts "
                "trained toward Miami using "
                "HIST fusion at every "
                "encoder layer. Each fusion "
                "module receives the current "
                "layer interaction descriptor "
                "and processes its one-step "
                "history with a 128-dimensional "
                "GRU before projecting a "
                "1024-dimensional correction."
            ),
        )
    ),

    "resadapt_enc_leHIST2tm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_leHIST2tm_cold"
            ),
            branch_domains=(
                "l",
                "e",
            ),
            target_domain="m",
            history_length=2,
            expected_training_examples=(
                EXPECTED_MIAMI_TRAINING_EXAMPLES
            ),
            domain_category=(
                "leHIST2tm"
            ),
            description=(
                "Persistent frozen Llanito "
                "and English encoder "
                "residual-adapter experts "
                "trained toward Miami. "
                "Every two encoder layers, "
                "the two-layer history of "
                "expert delta, elementwise "
                "product and difference "
                "interactions is compressed "
                "and processed by a "
                "128-dimensional GRU before "
                "producing a fused "
                "1024-dimensional correction."
            ),
        )
    ),

    "resadapt_enc_leHIST3tm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_leHIST3tm_cold"
            ),
            branch_domains=(
                "l",
                "e",
            ),
            target_domain="m",
            history_length=3,
            expected_training_examples=(
                EXPECTED_MIAMI_TRAINING_EXAMPLES
            ),
            domain_category=(
                "leHIST3tm"
            ),
            description=(
                "Persistent frozen Llanito "
                "and English encoder "
                "residual-adapter experts "
                "trained toward Miami. "
                "Every three encoder layers, "
                "the three-layer history of "
                "expert delta, elementwise "
                "product and difference "
                "interactions is compressed "
                "and processed by a "
                "128-dimensional GRU before "
                "producing a fused "
                "1024-dimensional correction."
            ),
        )
    ),

    "resadapt_enc_leHIST4tm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_leHIST4tm_cold"
            ),
            branch_domains=(
                "l",
                "e",
            ),
            target_domain="m",
            history_length=4,
            expected_training_examples=(
                EXPECTED_MIAMI_TRAINING_EXAMPLES
            ),
            domain_category=(
                "leHIST4tm"
            ),
            description=(
                "Persistent frozen Llanito "
                "and English encoder "
                "residual-adapter experts "
                "trained toward Miami. "
                "Every four encoder layers, "
                "the four-layer history of "
                "expert delta, elementwise "
                "product and difference "
                "interactions is compressed "
                "and processed by a "
                "128-dimensional GRU before "
                "producing a fused "
                "1024-dimensional correction."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)