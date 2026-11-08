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

TrainingCondition = Literal[
    "plain",
    "laug900",
]

FusionMode = Literal[
    "delta_key",
]

FusionArchitecture = Literal[
    "local",
    "parallel",
]


# Encoder

HIDDEN_SIZE = 1024
NUM_ENCODER_LAYERS = 24

ENCODER_LAYERS = tuple(
    range(NUM_ENCODER_LAYERS)
)


# Fusion-layer schedules
#
# Python indices are zero-based.
#
# ALL:
# human layers 1,2,...,24
#
# EVERY_2ND:
# human layers 2,4,...,24
#
# EVERY_3RD:
# human layers 3,6,...,24
#
# etc.

ALL_FUSION_LAYERS = tuple(
    range(
        NUM_ENCODER_LAYERS
    )
)

EVERY_2ND_FUSION_LAYERS = tuple(
    range(
        1,
        NUM_ENCODER_LAYERS,
        2,
    )
)

EVERY_3RD_FUSION_LAYERS = tuple(
    range(
        2,
        NUM_ENCODER_LAYERS,
        3,
    )
)

EVERY_4TH_FUSION_LAYERS = tuple(
    range(
        3,
        NUM_ENCODER_LAYERS,
        4,
    )
)

EVERY_6TH_FUSION_LAYERS = tuple(
    range(
        5,
        NUM_ENCODER_LAYERS,
        6,
    )
)

EVERY_8TH_FUSION_LAYERS = tuple(
    range(
        7,
        NUM_ENCODER_LAYERS,
        8,
    )
)


# Legacy schedules
#
# Retained for compatibility with
# previously trained experiments.

EVERY_12TH_FUSION_LAYERS = tuple(
    range(
        11,
        NUM_ENCODER_LAYERS,
        12,
    )
)

EVERY_24TH_FUSION_LAYERS = (
    NUM_ENCODER_LAYERS - 1,
)


# Fusion architecture

FUSION_USE_BIAS = False

ATTENTION_TYPE = (
    "scaled_dot_product"
)

ATTENTION_KEY_DIMENSION = (
    HIDDEN_SIZE
)

ATTENTION_SCALE = (
    HIDDEN_SIZE ** -0.5
)


# Q + K + V
#
# Each:
# 1024 x 1024

EXPECTED_PARAMETERS_PER_FUSION_LAYER = (
    3
    * HIDDEN_SIZE
    * HIDDEN_SIZE
)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str

    fusion_mode: FusionMode

    fusion_architecture: (
        FusionArchitecture
    )

    branch_domains: tuple[
        Domain,
        ...
    ]

    target_domain: Domain

    training_condition: (
        TrainingCondition
    )

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
            EXPECTED_PARAMETERS_PER_FUSION_LAYER
            * self.number_of_fusion_layers
        )


# Identity

MODEL_TYPE = "resadapt_fusion"

FAMILY = "resadapt_fusion"

TRANSFER_TYPE = "cold"

BASE_MODEL = "openai/whisper-medium"


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


# Frozen residual-adapter experts

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


# Llanito data

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
    / "test_manifest.csv"
)


# Miami data

MIAMI_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_ROOT
    / "train_validation_union_manifest.csv"
)


# English data

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


# Augmented Llanito data

LAUG900_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "joint"
    / "laug900"
)

LAUG900_TRAIN_MANIFEST = (
    LAUG900_ROOT
    / "train_manifest.csv"
)


# Training-data resolution

def get_train_manifest(
    experiment: ExperimentConfig,
) -> Path:
    if (
        experiment.training_condition
        == "laug900"
    ):
        if (
            experiment.target_domain
            != "l"
        ):
            raise RuntimeError(
                "laug900 is a Llanito "
                "target condition."
            )

        return (
            LAUG900_TRAIN_MANIFEST
        )

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

DATASET_NAME = (
    "llanito_fixed_test"
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

NORMALISATION_MAPPING_COLUMN = (
    "task"
)

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
# Shared across DEL experiments.

LEARNING_RATE = 5e-05

EPOCHS = 5.0

WARMUP_STEPS = 0

WARMUP_RATIO = 0.0

TRAIN_BATCH_SIZE_PER_DEVICE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.0

VALUE_L2_LAMBDA = 1e-4

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
        "adapterfusion_literature_informed"
    ),
    "reference": (
        "Pfeiffer et al. 2021 "
        "AdapterFusion"
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
    "adamw_weight_decay": (
        WEIGHT_DECAY
    ),
    "value_l2_lambda": (
        VALUE_L2_LAMBDA
    ),
    "value_l2_lambda_source": (
        "project_choice"
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
    "attention_type": (
        ATTENTION_TYPE
    ),
    "attention_key_dimension": (
        ATTENTION_KEY_DIMENSION
    ),
    "attention_scale": (
        ATTENTION_SCALE
    ),
    "seed": (
        SEED
    ),
    "data_seed": (
        SEED
    ),
    "fixed_test_used": False,
}


# Experiment helper

def make_experiment(
    *,
    model_name: str,
    architecture: (
        FusionArchitecture
    ),
    branches: tuple[
        Domain,
        ...
    ],
    target: Domain,
    condition: TrainingCondition,
    fusion_layers: tuple[
        int,
        ...
    ],
    expected_examples: (
        int | None
    ),
    domain_category: str,
    description: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=model_name,
        fusion_mode="delta_key",
        fusion_architecture=(
            architecture
        ),
        branch_domains=(
            branches
        ),
        target_domain=(
            target
        ),
        training_condition=(
            condition
        ),
        fusion_layers=(
            fusion_layers
        ),
        expected_training_examples=(
            expected_examples
        ),
        domain_category=(
            domain_category
        ),
        description=(
            description
        ),
    )


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {}


# Sweep helper
#
# IMPORTANT:
#
# Domain labels are concatenated with
# "".join(branches), NOT "+".join().
#
# ("l", "e") -> "le"
# ("m", "l") -> "ml"

def add_sweep(
    *,
    architecture: (
        FusionArchitecture
    ),
    branches: tuple[
        Domain,
        ...
    ],
    target: Domain,
    prefix: str,
    target_suffix: str,
    expected_examples: (
        int | None
    ),
) -> None:
    schedules = (
        (
            "",
            ALL_FUSION_LAYERS,
            "every encoder layer",
        ),
        (
            "2",
            EVERY_2ND_FUSION_LAYERS,
            "encoder layers "
            "2,4,...,24",
        ),
        (
            "3",
            EVERY_3RD_FUSION_LAYERS,
            "encoder layers "
            "3,6,...,24",
        ),
        (
            "4",
            EVERY_4TH_FUSION_LAYERS,
            "encoder layers "
            "4,8,...,24",
        ),
        (
            "6",
            EVERY_6TH_FUSION_LAYERS,
            "encoder layers "
            "6,12,18,24",
        ),
        (
            "8",
            EVERY_8TH_FUSION_LAYERS,
            "encoder layers "
            "8,16,24",
        ),
    )

    branch_text = "".join(
        branches
    )

    for (
        number,
        fusion_layers,
        schedule_text,
    ) in schedules:
        fusion_label = (
            f"{prefix}{number}"
        )

        model_name = (
            "resadapt_enc_"
            f"{branch_text}"
            f"{fusion_label}"
            f"{target_suffix}"
            "_cold"
        )

        domain_category = (
            f"{branch_text}"
            f"{fusion_label}"
            f"{target_suffix}"
        )

        if architecture == "parallel":
            architecture_text = (
                "Persistent-parallel"
            )
        else:
            architecture_text = (
                "Local"
            )

        description = (
            f"{architecture_text} "
            f"{branch_text} residual-"
            f"adapter expert fusion using "
            f"scaled dot-product DEL "
            f"attention at "
            f"{schedule_text}. "
            f"The Whisper backbone and "
            f"source residual-adapter "
            f"experts are frozen; only "
            f"the fusion parameters are "
            f"trained."
        )

        EXPERIMENTS[
            model_name
        ] = make_experiment(
            model_name=(
                model_name
            ),
            architecture=(
                architecture
            ),
            branches=(
                branches
            ),
            target=(
                target
            ),
            condition="plain",
            fusion_layers=(
                fusion_layers
            ),
            expected_examples=(
                expected_examples
            ),
            domain_category=(
                domain_category
            ),
            description=(
                description
            ),
        )


# Local DEL
#
# Llanito + English -> Miami
#
# DEL
# DEL2
# DEL3
# DEL4
# DEL6
# DEL8

add_sweep(
    architecture="local",
    branches=(
        "l",
        "e",
    ),
    target="m",
    prefix="DEL",
    target_suffix="tm",
    expected_examples=4364,
)


# Local DEL
#
# Miami + Llanito -> English

add_sweep(
    architecture="local",
    branches=(
        "m",
        "l",
    ),
    target="e",
    prefix="DEL",
    target_suffix="te",
    expected_examples=None,
)


# Persistent-parallel DEL
#
# Llanito + English -> Miami
#
# DELpara
# DELpara2
# DELpara3
# DELpara4
# DELpara6
# DELpara8

add_sweep(
    architecture="parallel",
    branches=(
        "l",
        "e",
    ),
    target="m",
    prefix="DELpara",
    target_suffix="tm",
    expected_examples=4364,
)


# Persistent-parallel DEL
#
# Miami + Llanito -> English

add_sweep(
    architecture="parallel",
    branches=(
        "m",
        "l",
    ),
    target="e",
    prefix="DELpara",
    target_suffix="te",
    expected_examples=None,
)


# Existing Llanito-target DEL control

EXPERIMENTS[
    "resadapt_enc_meDELtl_cold"
] = make_experiment(
    model_name=(
        "resadapt_enc_meDELtl_cold"
    ),
    architecture="local",
    branches=(
        "m",
        "e",
    ),
    target="l",
    condition="plain",
    fusion_layers=(
        ALL_FUSION_LAYERS
    ),
    expected_examples=904,
    domain_category=(
        "meDELtl"
    ),
    description=(
        "Local Miami and English "
        "residual-adapter DEL fusion "
        "using scaled dot-product "
        "attention at every encoder "
        "layer, trained toward Llanito."
    ),
)


# Existing augmented Llanito control

EXPERIMENTS[
    "resadapt_enc_mleDELtlaug900_cold"
] = make_experiment(
    model_name=(
        "resadapt_enc_mleDELtlaug900_cold"
    ),
    architecture="local",
    branches=(
        "m",
        "l",
        "e",
    ),
    target="l",
    condition="laug900",
    fusion_layers=(
        ALL_FUSION_LAYERS
    ),
    expected_examples=900,
    domain_category=(
        "mleDELtlaug900"
    ),
    description=(
        "Local Miami, Llanito and "
        "English residual-adapter DEL "
        "fusion using scaled dot-product "
        "attention at every encoder "
        "layer, trained on laug900."
    ),
)


# Scaled DELpara trial
#
# Miami + Llanito -> English.
# Persistent parallel streams.
# Fusion after every encoder layer.
# Intentionally keeps zero warmup and
# 1/sqrt(1024) attention scaling.

EXPERIMENTS[
    "resadapt_enc_mlDELparasqrttrialte_cold"
] = make_experiment(
    model_name=(
        "resadapt_enc_mlDELparasqrttrialte_cold"
    ),
    architecture="parallel",
    branches=(
        "m",
        "l",
    ),
    target="e",
    condition="plain",
    fusion_layers=(
        ALL_FUSION_LAYERS
    ),
    expected_examples=None,
    domain_category=(
        "mlDELparasqrttrialte"
    ),
    description=(
        "Trial persistent-parallel Miami "
        "and Llanito residual-adapter "
        "expert fusion toward English. "
        "Fusion is applied after every "
        "encoder layer using scaled "
        "dot-product DEL attention with "
        "1/sqrt(1024) scaling and zero "
        "warmup steps."
    ),
)


# Public experiment names

EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)