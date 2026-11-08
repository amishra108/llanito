from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Existing target-family recipes

from scripts.fft_l import config as fft_l_config
from scripts.fft_m import config as fft_m_config
from scripts.lora_l import config as lora_l_config
from scripts.lora_m import config as lora_m_config
from scripts.resadapt_l import config as resadapt_l_config
from scripts.resadapt_m import config as resadapt_m_config


# Embedded English FFT target recipe

class fft_e_config:
    LEARNING_RATE = 2e-5
    EPOCHS = 5.0
    WARMUP_STEPS = 0
    WARMUP_RATIO = 0.1
    TRAIN_BATCH_SIZE_PER_DEVICE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    WEIGHT_DECAY = 0.01
    OPTIMIZER = "adamw_torch"
    LR_SCHEDULER_TYPE = "linear"
    MAX_GRAD_NORM = 1.0
    USE_FP16 = True
    USE_BF16 = False
    USE_GRADIENT_CHECKPOINTING = True
    __name__ = "embedded_fft_e_recipe"


# Types

Mechanism = Literal[
    "fft",
    "lora",
    "resadapt",
]

Direction = Literal[
    "ltm",
    "mtl",
    "etl",
    "etm",
    "mte",
]

TransferType = Literal[
    "warm",
    "cold",
]

Placement = Literal[
    "enc",
    "encattn",
    "encdec",
    "dec",
]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    mechanism: Mechanism
    direction: Direction
    transfer_type: TransferType
    source_model_name: str
    placement: Placement | None
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

    @property
    def source_model_root(self) -> Path:
        return (
            PROJECT_ROOT
            / "models"
            / self.source_model_name
        )

    @property
    def source_model_directory(self) -> Path:
        return (
            self.source_model_root
            / "model"
        )

    @property
    def target_domain(self) -> str:
        if self.direction in (
            "ltm",
            "etm",
        ):
            return "m"

        if self.direction in (
            "mtl",
            "etl",
        ):
            return "l"

        if self.direction == "mte":
            return "e"

        raise ValueError(
            f"Unknown direction: "
            f"{self.direction!r}"
        )

    @property
    def training_language(self) -> str:
        if self.target_domain == "e":
            return "en"

        return "es"


# Shared identity

BASE_MODEL = "openai/whisper-medium"


# Miami training data

MIAMI_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_PROCESSED_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_MIAMI_TRAINING_EXAMPLES = 4364


# Llanito training data

LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

LLANITO_TRAIN_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_LLANITO_TRAINING_EXAMPLES = 904


# English training data

ENGLISH_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "miamienglishdom"
)

ENGLISH_TRAIN_MANIFEST = (
    ENGLISH_PROCESSED_ROOT
    / "train_validation_union_manifest_duration_matched.csv"
)

EXPECTED_ENGLISH_TRAINING_EXAMPLES = 405


# Common final evaluation

TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
)

EXPECTED_TEST_RECORDINGS = 57

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


# Shared final-run settings

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# Target data helpers

def get_train_manifest(
    experiment: ExperimentConfig,
) -> Path:
    if experiment.target_domain == "m":
        return MIAMI_TRAIN_MANIFEST

    if experiment.target_domain == "l":
        return LLANITO_TRAIN_MANIFEST

    if experiment.target_domain == "e":
        return ENGLISH_TRAIN_MANIFEST

    raise ValueError(
        "Unknown target domain: "
        f"{experiment.target_domain!r}"
    )


def get_expected_training_examples(
    experiment: ExperimentConfig,
) -> int:
    if experiment.target_domain == "m":
        return (
            EXPECTED_MIAMI_TRAINING_EXAMPLES
        )

    if experiment.target_domain == "l":
        return (
            EXPECTED_LLANITO_TRAINING_EXAMPLES
        )

    if experiment.target_domain == "e":
        return (
            EXPECTED_ENGLISH_TRAINING_EXAMPLES
        )

    raise ValueError(
        "Unknown target domain: "
        f"{experiment.target_domain!r}"
    )


# Target recipe helpers

def get_target_family_config(
    experiment: ExperimentConfig,
):
    target_domain = (
        experiment.target_domain
    )

    if experiment.mechanism == "fft":
        if target_domain == "m":
            return fft_m_config

        if target_domain == "l":
            return fft_l_config

        if target_domain == "e":
            return fft_e_config

        raise ValueError(
            "Unknown FFT target domain: "
            f"{target_domain!r}"
        )

    if experiment.mechanism == "lora":
        if target_domain == "m":
            return lora_m_config

        return lora_l_config

    if experiment.mechanism == "resadapt":
        if target_domain == "m":
            return resadapt_m_config

        return resadapt_l_config

    raise ValueError(
        f"Unknown mechanism: "
        f"{experiment.mechanism!r}"
    )


# Metadata family

def get_family(
    experiment: ExperimentConfig,
) -> str:
    if experiment.mechanism == "fft":
        return "full_finetune"

    if experiment.mechanism == "lora":
        return "lora"

    if experiment.mechanism == "resadapt":
        return "resadapt"

    raise ValueError(
        f"Unknown mechanism: "
        f"{experiment.mechanism!r}"
    )


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    # Llanito FFT -> Miami

    "fft_ltm_warm": (
        ExperimentConfig(
            model_name="fft_ltm_warm",
            mechanism="fft",
            direction="ltm",
            transfer_type="warm",
            source_model_name="fft_l",
            placement=None,
            description=(
                "Warm Llanito-to-Miami full "
                "fine-tuning transfer. Starts "
                "from the final fft_l model "
                "and continues full fine-tuning "
                "on the Miami train+validation "
                "union."
            ),
        )
    ),

    "fft_ltm_loraenc_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_loraenc_cold"
            ),
            mechanism="lora",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="enc",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and a new Miami encoder "
                "LoRA is trained."
            ),
        )
    ),

    "fft_ltm_loraencdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_loraencdec_cold"
            ),
            mechanism="lora",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="encdec",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and a new Miami "
                "encoder+decoder LoRA is trained."
            ),
        )
    ),

    "fft_ltm_loraencattn_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_loraencattn_cold"
            ),
            mechanism="lora",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="encattn",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and a new Miami "
                "encoder-attention LoRA is "
                "trained."
            ),
        )
    ),

    "fft_ltm_resadaptenc_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_resadaptenc_cold"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="enc",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and new Miami encoder "
                "residual adapters are trained."
            ),
        )
    ),

    "fft_ltm_resadaptencdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_resadaptencdec_cold"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="encdec",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and new Miami "
                "encoder+decoder residual "
                "adapters are trained."
            ),
        )
    ),

    "fft_ltm_resadaptdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_ltm_resadaptdec_cold"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="cold",
            source_model_name="fft_l",
            placement="dec",
            description=(
                "Cold Llanito-to-Miami transfer. "
                "The final fft_l backbone is "
                "frozen and new Miami decoder "
                "residual adapters are trained."
            ),
        )
    ),

    # Miami FFT -> Llanito

    "fft_mtl_warm": (
        ExperimentConfig(
            model_name="fft_mtl_warm",
            mechanism="fft",
            direction="mtl",
            transfer_type="warm",
            source_model_name="fft_m",
            placement=None,
            description=(
                "Warm Miami-to-Llanito full "
                "fine-tuning transfer. Starts "
                "from the final fft_m model "
                "and continues full fine-tuning "
                "on all non-test Llanito data."
            ),
        )
    ),

    "fft_mtl_loraenc_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_loraenc_cold"
            ),
            mechanism="lora",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="enc",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and a new Llanito "
                "encoder LoRA is trained."
            ),
        )
    ),

    "fft_mtl_loraencdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_loraencdec_cold"
            ),
            mechanism="lora",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="encdec",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and a new Llanito "
                "encoder+decoder LoRA is trained."
            ),
        )
    ),

    "fft_mtl_loraencattn_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_loraencattn_cold"
            ),
            mechanism="lora",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="encattn",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and a new Llanito "
                "encoder-attention LoRA is "
                "trained."
            ),
        )
    ),

    "fft_mtl_resadaptenc_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_resadaptenc_cold"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="enc",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and new Llanito encoder "
                "residual adapters are trained."
            ),
        )
    ),

    "fft_mtl_resadaptencdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_resadaptencdec_cold"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="encdec",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and new Llanito "
                "encoder+decoder residual "
                "adapters are trained."
            ),
        )
    ),

    "fft_mtl_resadaptdec_cold": (
        ExperimentConfig(
            model_name=(
                "fft_mtl_resadaptdec_cold"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="cold",
            source_model_name="fft_m",
            placement="dec",
            description=(
                "Cold Miami-to-Llanito transfer. "
                "The final fft_m backbone is "
                "frozen and new Llanito decoder "
                "residual adapters are trained."
            ),
        )
    ),
    # English FFT -> Llanito

    "fft_etl_warm": (
        ExperimentConfig(
            model_name="fft_etl_warm",
            mechanism="fft",
            direction="etl",
            transfer_type="warm",
            source_model_name="fft_e",
            placement=None,
            description=(
                "Warm English-to-Llanito full "
                "fine-tuning transfer. Starts "
                "from the final fft_e model "
                "and continues full fine-tuning "
                "on all non-test Llanito data."
            ),
        )
    ),

    # English FFT -> Miami

    "fft_etm_warm": (
        ExperimentConfig(
            model_name="fft_etm_warm",
            mechanism="fft",
            direction="etm",
            transfer_type="warm",
            source_model_name="fft_e",
            placement=None,
            description=(
                "Warm English-to-Miami full "
                "fine-tuning transfer. Starts "
                "from the final fft_e model "
                "and continues full fine-tuning "
                "on the Miami train+validation "
                "union."
            ),
        )
    ),

    # Miami FFT -> English

    "fft_mte_warm": (
        ExperimentConfig(
            model_name="fft_mte_warm",
            mechanism="fft",
            direction="mte",
            transfer_type="warm",
            source_model_name="fft_m",
            placement=None,
            description=(
                "Warm Miami-to-English full "
                "fine-tuning transfer. Starts "
                "from the final fft_m model "
                "and continues full fine-tuning "
                "on the duration-matched English "
                "train+validation union."
            ),
        )
    ),
}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)