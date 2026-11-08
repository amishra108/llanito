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

    fusion_layers: tuple[
        int,
        ...
    ]

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
    def metadata_path(
        self,
    ) -> Path:
        return (
            self.model_root
            / "metadata.json"
        )

    @property
    def output_root(
        self,
    ) -> Path:
        return (
            PROJECT_ROOT
            / "outputs"
            / self.model_name
        )

    @property
    def training_language(
        self,
    ) -> str:
        if self.target_domain == "e":
            return "en"

        return "es"

    @property
    def number_of_fusion_layers(
        self,
    ) -> int:
        return len(
            self.fusion_layers
        )

    @property
    def expected_trainable_parameters(
        self,
    ) -> int:
        return (
            self.number_of_fusion_layers
            * PARAMETERS_PER_DESC_MODULE
        )


# Identity

MODEL_TYPE = (
    "resadapt_descriptive_fusion"
)

FAMILY = (
    "resadapt_fusion_desc"
)

TRANSFER_TYPE = "cold"

BASE_MODEL = "openai/whisper-medium"


# Whisper

HIDDEN_SIZE = 1024

NUM_ENCODER_LAYERS = 24

ENCODER_LAYERS = tuple(
    range(
        NUM_ENCODER_LAYERS
    )
)


# Descriptive fusion
#
# For two expert deltas:
#
#   delta_a
#   delta_b
#   delta_a * delta_b
#   delta_a - delta_b
#
# concatenate:
#
#   4 * 1024 = 4096
#
# then:
#
#   4096 -> 128 -> 1024

DESC_WIDTH = 128

INTERACTION_FEATURES = 4

INTERACTION_SIZE = (
    INTERACTION_FEATURES
    * HIDDEN_SIZE
)


# Fusion schedules
#
# Internal indices are zero-based.
#
# DESC:
# human layers 1..24

ALL_FUSION_LAYERS = tuple(
    range(
        NUM_ENCODER_LAYERS
    )
)


# DESC2:
# human layers 2,4,...,24

EVERY_2ND_FUSION_LAYERS = tuple(
    range(
        1,
        NUM_ENCODER_LAYERS,
        2,
    )
)


# DESC3:
# human layers 3,6,...,24

EVERY_3RD_FUSION_LAYERS = tuple(
    range(
        2,
        NUM_ENCODER_LAYERS,
        3,
    )
)


# DESC4:
# human layers 4,8,...,24

EVERY_4TH_FUSION_LAYERS = tuple(
    range(
        3,
        NUM_ENCODER_LAYERS,
        4,
    )
)


# DESC6:
# human layers 6,12,18,24

EVERY_6TH_FUSION_LAYERS = tuple(
    range(
        5,
        NUM_ENCODER_LAYERS,
        6,
    )
)


# DESC8:
# human layers 8,16,24

EVERY_8TH_FUSION_LAYERS = tuple(
    range(
        7,
        NUM_ENCODER_LAYERS,
        8,
    )
)


# Parameter count
#
# Per fusion point:
#
# Linear 4096 -> 128
#   weight = 4096 * 128
#   bias   = 128
#
# Linear 128 -> 1024
#   weight = 128 * 1024
#   bias   = 1024

COMPRESSOR_PARAMETERS = (
    INTERACTION_SIZE
    * DESC_WIDTH
    + DESC_WIDTH
)

OUTPUT_PARAMETERS = (
    DESC_WIDTH
    * HIDDEN_SIZE
    + HIDDEN_SIZE
)

PARAMETERS_PER_DESC_MODULE = (
    COMPRESSOR_PARAMETERS
    + OUTPUT_PARAMETERS
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


# Frozen experts

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


# Llanito

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

LLANITO_TEST_MANIFEST = (
    LLANITO_ROOT
    / "fixed_test.csv"
)


# Miami

MIAMI_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_ROOT
    / "train_validation_union_manifest.csv"
)


# English

ENGLISH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miamienglishdom"
)

ENGLISH_TRAIN_MANIFEST = (
    ENGLISH_ROOT
    / "train_validation_union_manifest_duration_matched.csv"
)


# Training-data resolution

def get_train_manifest(
    experiment: ExperimentConfig,
) -> Path:
    if experiment.target_domain == "l":
        return (
            LLANITO_TRAIN_MANIFEST
        )

    if experiment.target_domain == "m":
        return (
            MIAMI_TRAIN_MANIFEST
        )

    if experiment.target_domain == "e":
        return (
            ENGLISH_TRAIN_MANIFEST
        )

    raise RuntimeError(
        "Unknown target domain."
    )


# Evaluation

TEST_MANIFEST = (
    LLANITO_TEST_MANIFEST
)

EXPECTED_TEST_RECORDINGS = 57

DATASET_NAME = "llanito_fixed_test"

EVALUATION_LANGUAGE = "es"

TASK = "transcribe"

GENERATION_MAX_LENGTH = 225

GENERATION_NUM_BEAMS = 1


# Normalisation

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
# Match DEL / DELpara / HIST.

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
        "matched_fusion_control"
    ),
    "source": (
        "scripts/fusion and "
        "scripts/fusion_history"
    ),
    "reason": (
        "Keep the DEL/DELpara/HIST "
        "optimisation protocol fixed while "
        "testing current-layer descriptive "
        "fusion without QKV attention or "
        "cross-layer history."
    ),
    "learning_rate": (
        LEARNING_RATE
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
    "fixed_test_used": False,
}


# Helper

def make_experiment(
    *,
    model_name: str,
    branches: tuple[
        Domain,
        Domain,
    ],
    target: Domain,
    fusion_layers: tuple[
        int,
        ...
    ],
    expected_examples: int | None,
    domain_category: str,
    description: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=model_name,
        branch_domains=branches,
        target_domain=target,
        fusion_layers=fusion_layers,
        expected_training_examples=(
            expected_examples
        ),
        domain_category=(
            domain_category
        ),
        description=description,
    )


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {}


# Schedules

SCHEDULES = {
    "": (
        ALL_FUSION_LAYERS
    ),
    "2": (
        EVERY_2ND_FUSION_LAYERS
    ),
    "3": (
        EVERY_3RD_FUSION_LAYERS
    ),
    "4": (
        EVERY_4TH_FUSION_LAYERS
    ),
    "6": (
        EVERY_6TH_FUSION_LAYERS
    ),
    "8": (
        EVERY_8TH_FUSION_LAYERS
    ),
}


# Miami + English -> Llanito

for suffix, layers in (
    SCHEDULES.items()
):
    model_name = (
        "resadapt_enc_meDESC"
        f"{suffix}"
        "tl_cold"
    )

    EXPERIMENTS[
        model_name
    ] = make_experiment(
        model_name=model_name,
        branches=(
            "m",
            "e",
        ),
        target="l",
        fusion_layers=layers,
        expected_examples=904,
        domain_category=(
            "meDESC"
            f"{suffix}"
            "tl"
        ),
        description=(
            "Persistent Miami and English "
            "frozen residual-adapter streams "
            "trained toward Llanito. At each "
            "configured encoder fusion point, "
            "the two current adapter deltas, "
            "their elementwise product and "
            "their difference are concatenated "
            "into a 4096-dimensional "
            "descriptor and reduced through "
            "4096->128->1024 descriptive "
            "fusion."
        ),
    )


# Llanito + English -> Miami

for suffix, layers in (
    SCHEDULES.items()
):
    model_name = (
        "resadapt_enc_leDESC"
        f"{suffix}"
        "tm_cold"
    )

    EXPERIMENTS[
        model_name
    ] = make_experiment(
        model_name=model_name,
        branches=(
            "l",
            "e",
        ),
        target="m",
        fusion_layers=layers,
        expected_examples=4364,
        domain_category=(
            "leDESC"
            f"{suffix}"
            "tm"
        ),
        description=(
            "Persistent Llanito and English "
            "frozen residual-adapter streams "
            "trained toward Miami. Current "
            "expert adapter deltas are "
            "represented by delta_a, delta_b, "
            "their product and their "
            "difference, then reduced through "
            "4096->128->1024."
        ),
    )


# Miami + Llanito -> English

for suffix, layers in (
    SCHEDULES.items()
):
    model_name = (
        "resadapt_enc_mlDESC"
        f"{suffix}"
        "te_cold"
    )

    EXPERIMENTS[
        model_name
    ] = make_experiment(
        model_name=model_name,
        branches=(
            "m",
            "l",
        ),
        target="e",
        fusion_layers=layers,
        expected_examples=None,
        domain_category=(
            "mlDESC"
            f"{suffix}"
            "te"
        ),
        description=(
            "Persistent Miami and Llanito "
            "frozen residual-adapter streams "
            "trained toward English. Current "
            "expert adapter deltas are "
            "represented by delta_a, delta_b, "
            "their product and their "
            "difference, then reduced through "
            "4096->128->1024."
        ),
    )


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)