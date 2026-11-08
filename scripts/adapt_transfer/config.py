from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


# Existing target-family recipes

from scripts.lora_l import config as lora_l_config
from scripts.lora_m import config as lora_m_config
#from scripts.lora_e import config as lora_e_config
from scripts.resadapt_l import config as resadapt_l_config
from scripts.resadapt_m import config as resadapt_m_config


# Embedded English ResAdapt target recipe

class resadapt_e_config:
    LEARNING_RATE = 1e-4
    EPOCHS = 5.0
    WARMUP_STEPS = 0
    WARMUP_RATIO = 0.0
    TRAIN_BATCH_SIZE_PER_DEVICE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    WEIGHT_DECAY = 0.01
    OPTIMIZER = "adamw_torch"
    LR_SCHEDULER_TYPE = "linear"
    MAX_GRAD_NORM = 1.0
    USE_FP16 = True
    USE_BF16 = False
    USE_GRADIENT_CHECKPOINTING = False
    ADAPTER_BOTTLENECK = 128
    ADAPTER_DROPOUT = 0.1
    ADAPTER_POSITION = "all"
    __name__ = "embedded_resadapt_e_recipe"


# Types

Mechanism = Literal[
    "lora",
    "resadapt",
]

Direction = Literal[
    "ltm",
    "mtl",
    "etl",
    "lte",
    "etm",
    "mte",
]

TransferType = Literal[
    "warm",
    "cold",
]

Placement = Literal[
    "enc",
    "encdec",
]


# Experiment

@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    mechanism: Mechanism
    direction: Direction
    transfer_type: TransferType
    source_model_name: str
    placement: Placement
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

        if self.direction in (
            "lte",
            "mte",
        ):
            return "e"

        raise ValueError(
            f"Unknown direction: "
            f"{self.direction!r}"
        )


# Identity

BASE_MODEL = "openai/whisper-medium"

VANILLA_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "vanilla_whisper_medium"
)

VANILLA_MODEL_DIRECTORY = (
    VANILLA_MODEL_ROOT
    / "model"
)


# Miami data

MIAMI_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data" / "processed" / "miami_seqstack_larger" / "manifests"
)

MIAMI_TRAIN_MANIFEST = (
    MIAMI_PROCESSED_ROOT
    / "train_validation_union_manifest.csv"
)

EXPECTED_MIAMI_TRAINING_EXAMPLES = 4364


# Llanito data

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


# English-domain data

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


# Final evaluation

TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
)

EXPECTED_TEST_RECORDINGS = 57

DATASET_NAME = "llanito_fixed_test"


# Whisper

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


# Shared final-run settings

SEED = 2026

DATALOADER_WORKERS = 4

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# Target data

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


def get_training_language(
    experiment: ExperimentConfig,
) -> str:
    if experiment.target_domain == "e":
        return "en"

    if experiment.target_domain in (
        "l",
        "m",
    ):
        return "es"

    raise ValueError(
        "Unknown target domain: "
        f"{experiment.target_domain!r}"
    )


# Target recipe

def get_target_family_config(
    experiment: ExperimentConfig,
):
    if experiment.mechanism == "lora":
        if experiment.target_domain == "m":
            return lora_m_config

        if experiment.target_domain == "l":
            return lora_l_config

        #if experiment.target_domain == "e":
        #    return lora_e_config

        raise ValueError(
            "Unknown LoRA target domain: "
            f"{experiment.target_domain!r}"
        )

    if experiment.mechanism == "resadapt":
        if experiment.target_domain == "m":
            return resadapt_m_config

        if experiment.target_domain == "l":
            return resadapt_l_config

        if experiment.target_domain == "e":
            return resadapt_e_config

        raise ValueError(
            "Unknown ResAdapt target domain: "
            f"{experiment.target_domain!r}"
        )

    raise ValueError(
        "Unknown mechanism: "
        f"{experiment.mechanism!r}"
    )


# Metadata

def get_family(
    experiment: ExperimentConfig,
) -> str:
    if experiment.mechanism == "lora":
        return "lora"

    if experiment.mechanism == "resadapt":
        return "resadapt"

    raise ValueError(
        "Unknown mechanism: "
        f"{experiment.mechanism!r}"
    )


# Experiments

EXPERIMENTS: dict[
    str,
    ExperimentConfig,
] = {
    # LoRA encoder Llanito -> Miami

    "lora_enc_ltm_warm": (
        ExperimentConfig(
            model_name=(
                "lora_enc_ltm_warm"
            ),
            mechanism="lora",
            direction="ltm",
            transfer_type="warm",
            source_model_name=(
                "lora_enc_l"
            ),
            placement="enc",
            description=(
                "Warm Llanito-to-Miami "
                "encoder LoRA transfer. "
                "Continues the same Llanito "
                "LoRA matrices on Miami using "
                "the Miami LoRA optimisation "
                "recipe."
            ),
        )
    ),

    "lora_enc_ltm_cold": (
        ExperimentConfig(
            model_name=(
                "lora_enc_ltm_cold"
            ),
            mechanism="lora",
            direction="ltm",
            transfer_type="cold",
            source_model_name=(
                "lora_enc_l"
            ),
            placement="enc",
            description=(
                "Cold Llanito-to-Miami "
                "encoder LoRA transfer. "
                "The Llanito LoRA is merged "
                "into the Whisper backbone, "
                "the resulting source "
                "representation is frozen, "
                "and a fresh Miami encoder "
                "LoRA is trained."
            ),
        )
    ),

    # LoRA encoder Miami -> Llanito

    "lora_enc_mtl_warm": (
        ExperimentConfig(
            model_name=(
                "lora_enc_mtl_warm"
            ),
            mechanism="lora",
            direction="mtl",
            transfer_type="warm",
            source_model_name=(
                "lora_enc_m"
            ),
            placement="enc",
            description=(
                "Warm Miami-to-Llanito "
                "encoder LoRA transfer. "
                "Continues the same Miami "
                "LoRA matrices on Llanito "
                "using the Llanito LoRA "
                "optimisation recipe."
            ),
        )
    ),

    "lora_enc_mtl_cold": (
        ExperimentConfig(
            model_name=(
                "lora_enc_mtl_cold"
            ),
            mechanism="lora",
            direction="mtl",
            transfer_type="cold",
            source_model_name=(
                "lora_enc_m"
            ),
            placement="enc",
            description=(
                "Cold Miami-to-Llanito "
                "encoder LoRA transfer. "
                "The Miami LoRA is merged "
                "into the Whisper backbone, "
                "the resulting source "
                "representation is frozen, "
                "and a fresh Llanito encoder "
                "LoRA is trained."
            ),
        )
    ),

    # LoRA encoder+decoder Llanito -> Miami

    "lora_encdec_ltm_warm": ExperimentConfig(
        model_name="lora_encdec_ltm_warm",
        mechanism="lora",
        direction="ltm",
        transfer_type="warm",
        source_model_name="lora_encdec_l",
        placement="encdec",
        description=(
            "Warm Llanito-to-Miami encoder+decoder LoRA transfer. "
            "Continues the same Llanito encoder+decoder LoRA matrices "
            "on Miami. Source LoRA architecture and learned matrices "
            "are retained; Miami supplies the optimisation hyperparameters."
        ),
    ),

    "lora_encdec_ltm_cold": ExperimentConfig(
        model_name="lora_encdec_ltm_cold",
        mechanism="lora",
        direction="ltm",
        transfer_type="cold",
        source_model_name="lora_encdec_l",
        placement="encdec",
        description=(
            "Cold Llanito-to-Miami encoder+decoder LoRA transfer. "
            "Merges the learned Llanito LoRA into the Whisper backbone, "
            "freezes the merged source representation, and trains a fresh "
            "Miami encoder+decoder LoRA using the Miami target architecture "
            "and optimisation recipe."
        ),
    ),

    # LoRA encoder+decoder Miami -> Llanito

    "lora_encdec_mtl_warm": ExperimentConfig(
        model_name="lora_encdec_mtl_warm",
        mechanism="lora",
        direction="mtl",
        transfer_type="warm",
        source_model_name="lora_encdec_m",
        placement="encdec",
        description=(
            "Warm Miami-to-Llanito encoder+decoder LoRA transfer. "
            "Continues the same Miami encoder+decoder LoRA matrices "
            "on Llanito. Source LoRA architecture and learned matrices "
            "are retained; Llanito supplies the optimisation hyperparameters."
        ),
    ),

    "lora_encdec_mtl_cold": ExperimentConfig(
        model_name="lora_encdec_mtl_cold",
        mechanism="lora",
        direction="mtl",
        transfer_type="cold",
        source_model_name="lora_encdec_m",
        placement="encdec",
        description=(
            "Cold Miami-to-Llanito encoder+decoder LoRA transfer. "
            "Merges the learned Miami LoRA into the Whisper backbone, "
            "freezes the merged source representation, and trains a fresh "
            "Llanito encoder+decoder LoRA using the Llanito target architecture "
            "and optimisation recipe."
        ),
    ),

    # ResAdapt encoder Llanito -> Miami

    "resadapt_enc_ltm_warm": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_ltm_warm"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="warm",
            source_model_name=(
                "resadapt_enc_l"
            ),
            placement="enc",
            description=(
                "Warm Llanito-to-Miami "
                "encoder residual-adapter "
                "transfer. Continues the same "
                "Llanito residual adapters "
                "on Miami."
            ),
        )
    ),

    "resadapt_enc_ltm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_ltm_cold"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="cold",
            source_model_name=(
                "resadapt_enc_l"
            ),
            placement="enc",
            description=(
                "Cold Llanito-to-Miami "
                "encoder residual-adapter "
                "transfer. The Llanito "
                "adapters remain active and "
                "frozen and fresh Miami "
                "encoder adapters are added."
            ),
        )
    ),

    # ResAdapt encoder+decoder Llanito -> Miami

    "resadapt_encdec_ltm_warm": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_ltm_warm"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="warm",
            source_model_name=(
                "resadapt_encdec_l"
            ),
            placement="encdec",
            description=(
                "Warm Llanito-to-Miami "
                "encoder+decoder residual-"
                "adapter transfer. Continues "
                "the same Llanito adapters "
                "on Miami."
            ),
        )
    ),

    "resadapt_encdec_ltm_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_ltm_cold"
            ),
            mechanism="resadapt",
            direction="ltm",
            transfer_type="cold",
            source_model_name=(
                "resadapt_encdec_l"
            ),
            placement="encdec",
            description=(
                "Cold Llanito-to-Miami "
                "encoder+decoder residual-"
                "adapter transfer. Source "
                "adapters remain active and "
                "frozen and fresh Miami "
                "encoder+decoder adapters "
                "are added."
            ),
        )
    ),

    # ResAdapt encoder Miami -> Llanito

    "resadapt_enc_mtl_warm": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_mtl_warm"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="warm",
            source_model_name=(
                "resadapt_enc_m"
            ),
            placement="enc",
            description=(
                "Warm Miami-to-Llanito "
                "encoder residual-adapter "
                "transfer. Continues the same "
                "Miami adapters on Llanito."
            ),
        )
    ),

    "resadapt_enc_mtl_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_enc_mtl_cold"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="cold",
            source_model_name=(
                "resadapt_enc_m"
            ),
            placement="enc",
            description=(
                "Cold Miami-to-Llanito "
                "encoder residual-adapter "
                "transfer. Source adapters "
                "remain active and frozen "
                "and fresh Llanito encoder "
                "adapters are added."
            ),
        )
    ),

    # ResAdapt encoder+decoder Miami -> Llanito

    "resadapt_encdec_mtl_warm": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_mtl_warm"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="warm",
            source_model_name=(
                "resadapt_encdec_m"
            ),
            placement="encdec",
            description=(
                "Warm Miami-to-Llanito "
                "encoder+decoder residual-"
                "adapter transfer. Continues "
                "the same Miami adapters "
                "on Llanito."
            ),
        )
    ),

    "resadapt_encdec_mtl_cold": (
        ExperimentConfig(
            model_name=(
                "resadapt_encdec_mtl_cold"
            ),
            mechanism="resadapt",
            direction="mtl",
            transfer_type="cold",
            source_model_name=(
                "resadapt_encdec_m"
            ),
            placement="encdec",
            description=(
                "Cold Miami-to-Llanito "
                "encoder+decoder residual-"
                "adapter transfer. Source "
                "adapters remain active and "
                "frozen and fresh Llanito "
                "encoder+decoder adapters "
                "are added."
            ),
        )
    ),
    # LoRA encoder+decoder English -> Llanito

    "lora_encdec_etl_warm": ExperimentConfig(
        model_name="lora_encdec_etl_warm",
        mechanism="lora",
        direction="etl",
        transfer_type="warm",
        source_model_name="lora_encdec_e",
        placement="encdec",
        description=(
            "Warm English-to-Llanito encoder+decoder LoRA transfer. "
            "Continues the same English-domain LoRA matrices on Llanito "
            "using the Llanito LoRA optimisation recipe."
        ),
    ),

    "lora_encdec_etl_cold": ExperimentConfig(
        model_name="lora_encdec_etl_cold",
        mechanism="lora",
        direction="etl",
        transfer_type="cold",
        source_model_name="lora_encdec_e",
        placement="encdec",
        description=(
            "Cold English-to-Llanito encoder+decoder LoRA transfer. "
            "Merges the English LoRA into Whisper and trains a fresh "
            "Llanito encoder+decoder LoRA on top."
        ),
    ),

    # LoRA encoder+decoder Llanito -> English

    "lora_encdec_lte_warm": ExperimentConfig(
        model_name="lora_encdec_lte_warm",
        mechanism="lora",
        direction="lte",
        transfer_type="warm",
        source_model_name="lora_encdec_l",
        placement="encdec",
        description=(
            "Warm Llanito-to-English encoder+decoder LoRA transfer. "
            "Continues the same Llanito LoRA matrices on English-domain data."
        ),
    ),

    "lora_encdec_lte_cold": ExperimentConfig(
        model_name="lora_encdec_lte_cold",
        mechanism="lora",
        direction="lte",
        transfer_type="cold",
        source_model_name="lora_encdec_l",
        placement="encdec",
        description=(
            "Cold Llanito-to-English encoder+decoder LoRA transfer. "
            "Merges the Llanito LoRA into Whisper and trains a fresh "
            "English encoder+decoder LoRA on top."
        ),
    ),

    # LoRA encoder+decoder English -> Miami

    "lora_encdec_etm_warm": ExperimentConfig(
        model_name="lora_encdec_etm_warm",
        mechanism="lora",
        direction="etm",
        transfer_type="warm",
        source_model_name="lora_encdec_e",
        placement="encdec",
        description=(
            "Warm English-to-Miami encoder+decoder LoRA transfer. "
            "Continues the same English-domain LoRA matrices on Miami."
        ),
    ),

    "lora_encdec_etm_cold": ExperimentConfig(
        model_name="lora_encdec_etm_cold",
        mechanism="lora",
        direction="etm",
        transfer_type="cold",
        source_model_name="lora_encdec_e",
        placement="encdec",
        description=(
            "Cold English-to-Miami encoder+decoder LoRA transfer. "
            "Merges the English LoRA into Whisper and trains a fresh "
            "Miami encoder+decoder LoRA on top."
        ),
    ),

    # LoRA encoder+decoder Miami -> English

    "lora_encdec_mte_warm": ExperimentConfig(
        model_name="lora_encdec_mte_warm",
        mechanism="lora",
        direction="mte",
        transfer_type="warm",
        source_model_name="lora_encdec_m",
        placement="encdec",
        description=(
            "Warm Miami-to-English encoder+decoder LoRA transfer. "
            "Continues the same Miami LoRA matrices on English-domain data."
        ),
    ),

    "lora_encdec_mte_cold": ExperimentConfig(
        model_name="lora_encdec_mte_cold",
        mechanism="lora",
        direction="mte",
        transfer_type="cold",
        source_model_name="lora_encdec_m",
        placement="encdec",
        description=(
            "Cold Miami-to-English encoder+decoder LoRA transfer. "
            "Merges the Miami LoRA into Whisper and trains a fresh "
            "English encoder+decoder LoRA on top."
        ),
    ),

    # ResAdapt encoder Llanito -> English

    "resadapt_enc_lte_warm": ExperimentConfig(
        model_name="resadapt_enc_lte_warm",
        mechanism="resadapt",
        direction="lte",
        transfer_type="warm",
        source_model_name="resadapt_enc_l",
        placement="enc",
        description=(
            "Warm Llanito-to-English encoder residual-adapter transfer. "
            "Continues the same Llanito residual adapters on English-domain "
            "data. The source adapter architecture is retained while the "
            "English ResAdapt optimisation recipe is used."
        ),
    ),

    "resadapt_enc_lte_cold": ExperimentConfig(
        model_name="resadapt_enc_lte_cold",
        mechanism="resadapt",
        direction="lte",
        transfer_type="cold",
        source_model_name="resadapt_enc_l",
        placement="enc",
        description=(
            "Cold Llanito-to-English encoder residual-adapter transfer. "
            "The Llanito adapters remain active and frozen and fresh English "
            "encoder adapters are added using the English ResAdapt architecture "
            "and optimisation recipe."
        ),
    ),

    # ResAdapt encoder English -> Llanito

    "resadapt_enc_etl_warm": ExperimentConfig(
        model_name="resadapt_enc_etl_warm",
        mechanism="resadapt",
        direction="etl",
        transfer_type="warm",
        source_model_name="resadapt_enc_e",
        placement="enc",
        description=(
            "Warm English-to-Llanito encoder residual-adapter transfer. "
            "Continues the same English residual adapters on Llanito. "
            "The source adapter architecture is retained while the Llanito "
            "ResAdapt optimisation recipe is used."
        ),
    ),

    "resadapt_enc_etl_cold": ExperimentConfig(
        model_name="resadapt_enc_etl_cold",
        mechanism="resadapt",
        direction="etl",
        transfer_type="cold",
        source_model_name="resadapt_enc_e",
        placement="enc",
        description=(
            "Cold English-to-Llanito encoder residual-adapter transfer. "
            "The English adapters remain active and frozen and fresh Llanito "
            "encoder adapters are added using the Llanito ResAdapt architecture "
            "and optimisation recipe."
        ),
    ),

}


EXPERIMENT_NAMES = tuple(
    EXPERIMENTS.keys()
)